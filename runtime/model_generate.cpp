// The persistent decode slice (M3-2): every layer is opened once, the prompt prefills
// through those objects and generation then runs token by token with the KV and DeltaNet
// state carried. It owns the expert cache, the miss legs and telemetry, which the prefill
// and logits paths never touch.

#include "model.h"
#include "vk_transfer.h"

#include "model_internal.h"
#include "model_io.h"
#include "qrt_config.h"
#include "qrt_telemetry.h"
#include "qrt_util.h"

#include "cpu_moe.h"
#include "expert_cache.h"
#include "ffn.h"
#include "gguf_index.h"
#include "layer.h"
#include "miss_split.h"
#include "vk_dev.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

// End-to-end vertical slice (M3-2): open every requested layer once with one shared FFN
// slot, prefill the prompt through the same objects, then decode one token at a time with
// the KV/DeltaNet state carried. The head stays on the host for this slice.
int qrt_model_generate(const qrt_config * cfg, const qrt_model_options * opt,
                       const char * gguf_path, const char * store_prefix, const char * spv_dir,
                       const uint32_t * ids, uint32_t n_tokens, uint32_t n_new,
                       const char * ids_out, char * err, size_t err_size) {
    if (cfg == NULL || opt == NULL || gguf_path == NULL || store_prefix == NULL ||
        spv_dir == NULL || ids == NULL || n_tokens == 0 || n_new == 0) {
        return fail(err, err_size, "generate needs config, paths, prompt ids and n_new");
    }
    if (opt->n_layers == 0 || opt->n_layers > 40) {
        return fail(err, err_size, "generate needs 1..40 layers, got %u", opt->n_layers);
    }
    uint32_t chunk = (uint32_t) cfg->prefill_chunk;
    if (chunk > 64u) {
        // The 40-layer slice is sized for a small resident chunk. A 2048-token chunk across
        // 40 layers does not fit in 8 GiB; the shared scratch arena is the next step.
        chunk = 32u;
    }
    if (chunk < 32u || (chunk % 32u) != 0u) {
        return fail(err, err_size, "generate chunk %u is not a 32-aligned 32..64", chunk);
    }
    const uint64_t n_ctx64 = (uint64_t) n_tokens + n_new + 1u;
    if (n_ctx64 > QRT_KV_CTX_TOKENS) {
        return fail(err, err_size, "prompt %u + new %u exceeds the %u context",
                    n_tokens, n_new, (uint32_t) QRT_KV_CTX_TOKENS);
    }
    const uint32_t n_ctx = (uint32_t) n_ctx64;

    char e[512] = {0};
    qrt_model_run_owner owner;
    qrt_vk_dev * & dev = owner.dev;
    qrt_gguf * & g = owner.g;
    qrt_vk_staging * & stg = owner.stg;
    VkBuffer slot = VK_NULL_HANDLE;
    VkDeviceMemory slot_mem = VK_NULL_HANDLE;
    qrt_weights * wstore = NULL;
    std::vector<qrt_layer *> layers;
    std::vector<uint32_t> generated;
    qrt_cache * cache = NULL;
    qrt_cpu_moe * cpu = NULL;
    std::vector<VkBuffer> cache_bufs;
    std::vector<VkDeviceMemory> cache_mems;
    std::vector<uint32_t> cache_slots;
    qrt_split_rates rates{};
    qrt_split_carry carry{};
    int cache_on = 0;
    qrt_telemetry * tel = NULL;
    double leg_gpu = 0.0, leg_h2d = 0.0, leg_cpu = 0.0;
    uint64_t leg_hits = 0, leg_miss = 0, leg_h2d_n = 0, leg_cpu_n = 0, leg_h2d_bytes = 0;
    int rc = -1;
    do {
        if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        if (qrt_gguf_open(gguf_path, &g, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(g);
        if (hp == NULL || !(hp->present & QRT_HP_BLOCK_COUNT) || hp->block_count <= 0) {
            fail(err, err_size, "missing or invalid block_count");
            break;
        }
        const int32_t nextn = hp->present & QRT_HP_NEXTN ? hp->nextn : 0;
        if (nextn < 0 || nextn >= hp->block_count ||
            opt->n_layers > (uint32_t) (hp->block_count - nextn)) {
            fail(err, err_size, "requested %u layers, block_count=%d nextn=%d",
                 opt->n_layers, hp->block_count, nextn);
            break;
        }
        if (hp->n_embd <= 0 || hp->n_embd % 32 != 0) {
            fail(err, err_size, "n_embd must be a positive multiple of 32");
            break;
        }
        const uint32_t E = (uint32_t) hp->n_embd;
        const qrt_gguf_tensor * emb = qrt_gguf_find(g, "token_embd.weight");
        const qrt_gguf_tensor * outw = qrt_gguf_find(g, "output.weight");
        if (emb == NULL || emb->type != 8 || emb->n_dims != 2 || emb->ne[0] != E ||
            emb->ne[1] == 0 || emb->ne[1] > INT32_MAX) {
            fail(err, err_size, "token_embd must be q8_0 with n_embd x vocabulary rows");
            break;
        }
        if (outw == NULL || outw->type != 8 || outw->n_dims != 2 || outw->ne[0] != E ||
            outw->ne[1] != emb->ne[1]) {
            fail(err, err_size, "output must match the q8_0 embedding shape");
            break;
        }
        const uint32_t V = (uint32_t) emb->ne[1];
        std::vector<float> x((size_t) n_tokens * E);
        if (qrt_model_embed_rows(g, emb, E, V, ids, n_tokens, x.data(), e, sizeof(e)) != 0) {
            fail(err, err_size, "embedding: %s", e);
            break;
        }
        std::vector<float> onorm;
        if (qrt_gguf_read_f32(g, "output_norm.weight", &onorm, e, sizeof(e)) != 0 ||
            onorm.size() != E) {
            fail(err, err_size, "output_norm: %s", e[0] ? e : "wrong width");
            break;
        }

        if (qrt_weights_open(store_prefix, &wstore, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        const uint32_t n_expert = qrt_weights_experts(wstore);
        const uint64_t max_expert = qrt_weights_max_expert_bytes(wstore);
        qrt_weights_close(wstore);
        wstore = NULL;
        if (n_expert == 0 || max_expert == 0) {
            fail(err, err_size, "the store has no experts");
            break;
        }
        uint64_t slot_bytes = (uint64_t) n_expert * max_expert;
        slot_bytes = (slot_bytes + 65535u) & ~(uint64_t) 65535u;
        if (qrt_vk_alloc(dev, slot_bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         &slot, &slot_mem, e, sizeof(e)) != 0) {
            fail(err, err_size, "shared FFN slot: %s", e);
            break;
        }
        const uint64_t out_row = (uint64_t) chunk * E * 4;
        if (qrt_vk_staging_create(dev, out_row, &stg, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        printf("generate_run tokens=%u new=%u layers=%u chunk=%u n_ctx=%u slot_mib=%.1f\n",
               n_tokens, n_new, opt->n_layers, chunk, n_ctx,
               (double) slot_bytes / (1024.0 * 1024.0));

        qrt_layer_cfg lcfg{};
        lcfg.ffn = 1;
        lcfg.single_submit = 1;
        lcfg.dn_recur = 1;
        lcfg.kv_k = cfg->kv_k;
        lcfg.kv_v = cfg->kv_v;
        lcfg.kv_page_tokens = (uint32_t) cfg->kv_page_tokens;
        lcfg.n_ctx = n_ctx;
        lcfg.ffn_slot = slot;
        lcfg.ffn_slot_bytes = slot_bytes;
        layers.assign(opt->n_layers, nullptr);
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            if (qrt_layer_open(dev, gguf_path, store_prefix, l, chunk, spv_dir, &lcfg,
                               &layers[l], e, sizeof(e)) != 0) {
                fail(err, err_size, "open layer %u: %s", l, e);
                break;
            }
            if (qrt_layer_zero_state(layers[l], e, sizeof(e)) != 0) {
                fail(err, err_size, "zero layer %u: %s", l, e);
                break;
            }
        }
        if (layers[opt->n_layers - 1] == nullptr) {
            break;
        }

        std::vector<float> xnext((size_t) chunk * E);
        const uint32_t n_chunks = 1u + (n_tokens - 1u) / chunk;
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            const double t0 = now_ms();
            for (uint32_t c = 0; c < n_chunks; ++c) {
                const uint32_t pos = c * chunk;
                const uint32_t len = n_tokens - pos < chunk ? n_tokens - pos : chunk;
                if (qrt_layer_set_input_rows(layers[l], x.data() + (size_t) pos * E, len, e,
                                             sizeof(e)) != 0 ||
                    qrt_layer_run_chunk(layers[l], pos, len, e, sizeof(e)) != 0 ||
                    qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, xnext.data(),
                                    (uint64_t) len * E * 4, e, sizeof(e)) != 0) {
                    fail(err, err_size, "prefill layer %u chunk %u: %s", l, c, e);
                    break;
                }
                memcpy(x.data() + (size_t) pos * E, xnext.data(), (size_t) len * E * 4);
            }
            if (e[0] != '\0') {
                break;
            }
            printf("generate_prefill layer=%u chunks=%u ms=%.1f\n", l, n_chunks,
                   now_ms() - t0);
        }
        if (e[0] != '\0') {
            break;
        }

        // Decode expert cache (M3-2a). The prefill above packed every used expert into
        // the shared slot; decode switches to per-layer slot arrays sized by the cache
        // plan, and the split sends the misses to H2D or the CPU leg. A failed cache
        // allocation falls back to the prefill slot (slow but correct), not to an abort.
        uint64_t cache_mib = 0;
        if (cfg->expert_cache_mib > 0) {
            uint64_t layer_slot[QRT_CACHE_MAX_LAYERS] = {0};
            if (qrt_weights_open(store_prefix, &wstore, e, sizeof(e)) != 0) {
                fail(err, err_size, "%s", e);
                break;
            }
            int size_ok = 1;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                if (qrt_weights_layer_expert_bytes(wstore, l, &layer_slot[l]) != 0) {
                    size_ok = 0;
                    break;
                }
            }
            qrt_weights_close(wstore);
            wstore = NULL;
            qrt_cache_plan plan{};
            if (!size_ok ||
                qrt_cache_plan_build(opt->n_layers, n_expert, layer_slot, NULL,
                                     (uint64_t) cfg->expert_cache_mib << 20,
                                     cfg->cache_policy, QRT_CACHE_DEFAULT_PROTECTED_PCT,
                                     &plan, e, sizeof(e)) != 0) {
                fail(err, err_size, "expert cache plan: %s", e[0] ? e : "no layer sizes");
                break;
            }
            // The prefill slot has done its job; the cache takes its VRAM.
            qrt_vk_free(dev, slot, slot_mem);
            slot = VK_NULL_HANDLE;
            slot_mem = VK_NULL_HANDLE;
            int alloc_ok = 1;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                const uint64_t bytes = (uint64_t) plan.slots_per_layer[l] * plan.slot_bytes[l];
                VkBuffer b = VK_NULL_HANDLE;
                VkDeviceMemory m = VK_NULL_HANDLE;
                if (bytes != 0 &&
                    qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE |
                                 QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                                 &b, &m, e, sizeof(e)) != 0) {
                    alloc_ok = 0;
                    break;
                }
                cache_bufs.push_back(b);
                cache_mems.push_back(m);
                cache_slots.push_back(plan.slots_per_layer[l]);
            }
            if (!alloc_ok) {
                for (size_t i = 0; i < cache_bufs.size(); ++i) {
                    if (cache_bufs[i] != VK_NULL_HANDLE) {
                        qrt_vk_free(dev, cache_bufs[i], cache_mems[i]);
                    }
                }
                cache_bufs.clear();
                cache_mems.clear();
                cache_slots.clear();
                if (qrt_vk_alloc(dev, slot_bytes, QRT_VK_USAGE_STORAGE |
                                 QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                                 &slot, &slot_mem, e, sizeof(e)) != 0) {
                    fail(err, err_size, "cache allocation failed and the fallback slot "
                         "did not fit: %s", e);
                    break;
                }
                printf("generate_cache disabled: %s\n", e[0] ? e : "allocation");
                e[0] = '\0';
            } else if (qrt_cache_open(&plan, NULL, &cache, e, sizeof(e)) != 0) {
                fail(err, err_size, "%s", e);
                break;
            } else {
                cpu = qrt_cpu_moe_create(1);
                if (cpu == NULL) {
                    fail(err, err_size, "out of memory for the CPU expert core");
                    break;
                }
                qrt_miss_split_default_rates(&rates);
                for (uint32_t l = 0; l < opt->n_layers; ++l) {
                    if (qrt_layer_set_cache(layers[l], cache, cache_bufs[l],
                                            plan.slot_bytes[l], cpu, cfg->miss_split, &rates,
                                            &carry, e, sizeof(e)) != 0) {
                        fail(err, err_size, "cache on layer %u: %s", l, e);
                        break;
                    }
                }
                if (e[0] != '\0') {
                    break;
                }
                cache_on = 1;
                cache_mib = plan.bytes >> 20;
                printf("generate_cache slots=%llu mib=%llu degraded=%u policy=%s "
                       "split=%s threads=8\n",
                       (unsigned long long) plan.slots, (unsigned long long) cache_mib,
                       plan.degraded_layers, qrt_cache_policy_name(cfg->cache_policy),
                       cfg->miss_split == QRT_MISS_SPLIT_AUTO ? "auto" : "fixed");
            }
        }

        if (cfg->telemetry == 1) {
            char terr[256] = {0};
            tel = qrt_telemetry_open(cfg->telemetry_path, terr, sizeof(terr));
            if (tel == NULL) {
                printf("generate_telemetry disabled: %s\n", terr);
            }
        }

        std::vector<float> logits((size_t) V);
        auto pick = [&](const std::vector<float> & lg, uint32_t * id, float * best,
                        float * margin) {
            uint32_t bi = 0, si = 0;
            float bv = lg[0], sv = -INFINITY;
            for (uint32_t v = 1; v < V; ++v) {
                if (lg[v] > bv) {
                    sv = bv; si = bi; bv = lg[v]; bi = v;
                } else if (lg[v] > sv) {
                    sv = lg[v]; si = v;
                }
            }
            (void) si;
            *id = bi; *best = bv; *margin = bv - sv;
        };
        auto head_one = [&](const float * hidden, std::vector<float> & lg) -> int {
            return qrt_model_head_rows(g, outw, onorm.data(), hidden, NULL, E, V, 1, 1, 0, &lg, NULL,
                             e, sizeof(e));
        };
        uint32_t tok = 0;
        float best = 0.0f, margin = 0.0f;
        if (head_one(x.data() + (size_t) (n_tokens - 1u) * E, logits) != 0) {
            fail(err, err_size, "prefill head: %s", e);
            break;
        }
        pick(logits, &tok, &best, &margin);
        generated.push_back(tok);
        printf("generate_token step=0 pos=%u id=%u logit=%.4f margin=%.4f\n",
               n_tokens, tok, best, margin);

        std::vector<float> x1((size_t) E);
        uint32_t pos = n_tokens;
        if (opt->batch_k > 1) {
            // Diagnostic decode width (M3-2a): one pass carries k rows so the miss legs
            // are measured at k=5. The rows are prompt ids fed teacher-forced; the head
            // output is not used as the next input, so this is not an acceptance path.
            const uint32_t BK = opt->batch_k;
            std::vector<float> xb((size_t) BK * E);
            std::vector<uint32_t> bid(BK);
            uint32_t produced = 1;
            uint32_t pass = 0;
            while (produced < n_new) {
                const uint32_t k = std::min(BK, n_new - produced);
                for (uint32_t i = 0; i < k; ++i) {
                    bid[i] = ids[(size_t) (pass * BK + i) % n_tokens];
                }
                if (qrt_model_embed_rows(g, emb, E, V, bid.data(), k, xb.data(), e, sizeof(e)) != 0) {
                    fail(err, err_size, "batch embedding: %s", e);
                    break;
                }
                double step_gpu = 0.0, step_h2d = 0.0, step_cpu = 0.0;
                uint32_t step_hits = 0, step_miss = 0, step_h2d_n = 0, step_cpu_n = 0;
                const double t0 = now_ms();
                for (uint32_t l = 0; l < opt->n_layers; ++l) {
                    if (qrt_layer_set_input_rows(layers[l], xb.data(), k, e, sizeof(e)) != 0 ||
                        qrt_layer_run_chunk(layers[l], pos, k, e, sizeof(e)) != 0 ||
                        qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, xb.data(),
                                        (uint64_t) k * E * 4, e, sizeof(e)) != 0) {
                        fail(err, err_size, "batch layer %u pass %u: %s", l, pass, e);
                        break;
                    }
                    if (cache_on) {
                        const qrt_ffn_legs lg = qrt_layer_last_legs(layers[l]);
                        const double gpu = qrt_layer_gpu_ms(layers[l]);
                        step_gpu += gpu;
                        step_h2d += lg.h2d_ms;
                        step_cpu += lg.cpu_ms;
                        step_hits += lg.hits;
                        step_miss += lg.misses;
                        step_h2d_n += lg.h2d;
                        step_cpu_n += lg.cpu;
                        leg_gpu += gpu;
                        leg_h2d += lg.h2d_ms;
                        leg_cpu += lg.cpu_ms;
                        leg_hits += lg.hits;
                        leg_miss += lg.misses;
                        leg_h2d_n += lg.h2d;
                        leg_cpu_n += lg.cpu;
                        leg_h2d_bytes += lg.h2d_bytes;
                        if (qrt_layer_take_cpu_out(layers[l], xb.data(), k) != 0) {
                            fail(err, err_size, "layer %u CPU output on pass %u", l, pass);
                            break;
                        }
                    }
                }
                if (e[0] != '\0') {
                    break;
                }
                if (head_one(xb.data() + (size_t) (k - 1u) * E, logits) != 0) {
                    fail(err, err_size, "batch head pass %u: %s", pass, e);
                    break;
                }
                pick(logits, &tok, &best, &margin);
                generated.push_back(tok);
                printf("generate_batch pass=%u pos=%u k=%u id=%u logit=%.4f margin=%.4f "
                       "gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u h2d=%u cpu=%u "
                       "pass_ms=%.1f\n",
                       pass, pos, k, tok, best, margin, step_gpu, step_h2d, step_cpu,
                       step_hits, step_miss, step_h2d_n, step_cpu_n, now_ms() - t0);
                pos += k;
                produced += k;
                ++pass;
            }
        } else
        for (uint32_t s = 1; s < n_new; ++s) {
            if (qrt_model_embed_rows(g, emb, E, V, &generated.back(), 1, x1.data(), e, sizeof(e)) != 0) {
                fail(err, err_size, "decode embedding: %s", e);
                break;
            }
            const double t0 = now_ms();
            double step_gpu = 0.0, step_h2d = 0.0, step_cpu = 0.0;
            uint32_t step_hits = 0, step_miss = 0, step_h2d_n = 0, step_cpu_n = 0;
            uint64_t step_h2d_bytes = 0;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                const double tl = now_ms();
                if (qrt_layer_set_input_rows(layers[l], x1.data(), 1, e, sizeof(e)) != 0 ||
                    qrt_layer_run_chunk(layers[l], pos, 1, e, sizeof(e)) != 0 ||
                    qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, x1.data(),
                                    (uint64_t) E * 4, e, sizeof(e)) != 0) {
                    fail(err, err_size, "decode layer %u step %u: %s", l, s, e);
                    break;
                }
                if (cache_on) {
                    const qrt_ffn_legs lg = qrt_layer_last_legs(layers[l]);
                    const double gpu = qrt_layer_gpu_ms(layers[l]);
                    (void) tl;
                    step_gpu += gpu;
                    step_h2d += lg.h2d_ms;
                    step_cpu += lg.cpu_ms;
                    step_hits += lg.hits;
                    step_miss += lg.misses;
                    step_h2d_n += lg.h2d;
                    step_cpu_n += lg.cpu;
                    step_h2d_bytes += lg.h2d_bytes;
                    leg_gpu += gpu;
                    leg_h2d += lg.h2d_ms;
                    leg_cpu += lg.cpu_ms;
                    leg_hits += lg.hits;
                    leg_miss += lg.misses;
                    leg_h2d_n += lg.h2d;
                    leg_cpu_n += lg.cpu;
                    leg_h2d_bytes += lg.h2d_bytes;
                    if (qrt_layer_take_cpu_out(layers[l], x1.data(), 1) != 0) {
                        fail(err, err_size, "layer %u CPU output on step %u", l, s);
                        break;
                    }
                }
            }
            if (e[0] != '\0') {
                break;
            }
            if (cache_on) {
                printf("generate_legs step=%u gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f "
                       "hits=%u misses=%u h2d=%u cpu=%u\n", s, step_gpu, step_h2d, step_cpu,
                       step_hits, step_miss, step_h2d_n, step_cpu_n);
            }
            if (head_one(x1.data(), logits) != 0) {
                fail(err, err_size, "decode head step %u: %s", s, e);
                break;
            }
            pick(logits, &tok, &best, &margin);
            generated.push_back(tok);
            printf("generate_token step=%u pos=%u id=%u logit=%.4f margin=%.4f layer_ms=%.1f\n",
                   s, pos, tok, best, margin, now_ms() - t0);
            if (tel != NULL) {
                qrt_pass_stats ps{};
                ps.k = 1;
                ps.pos = pos;
                ps.accepted = 1;
                ps.hit_experts = step_hits;
                ps.miss_experts = step_miss;
                ps.miss_cpu = step_cpu_n;
                ps.miss_h2d = step_h2d_n;
                ps.h2d_bytes = step_h2d_bytes;
                ps.draft_ms = 0.0;
                ps.gpu_ms = step_gpu;
                ps.cpu_leg_ms = step_cpu;
                ps.pass_ms = now_ms() - t0;
                qrt_telemetry_pass(tel, s, &ps);
            }
            ++pos;
        }
        if (e[0] != '\0') {
            break;
        }
        uint32_t max_submits = 0, max_dispatches = 0;
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            max_submits = std::max(max_submits, qrt_layer_submits(layers[l]));
            max_dispatches = std::max(max_dispatches, qrt_layer_dispatches(layers[l]));
        }
        printf("generate_done ids=%zu alloc_mib=%.1f peak_mib=%.1f failures=%u "
               "submits_per_layer=%u dispatches_max=%u topk_readbacks_per_layer=1\n",
               generated.size(), (double) dev->alloc_bytes / (1024.0 * 1024.0),
               (double) dev->alloc_peak_bytes / (1024.0 * 1024.0), dev->alloc_failures,
               max_submits, max_dispatches);
        if (cache_on) {
            qrt_cache_stats cs{};
            qrt_cache_get_stats(cache, &cs);
            const double hi = std::max(leg_gpu, std::max(leg_h2d, leg_cpu));
            const double lo = std::min(leg_gpu, std::min(leg_h2d, leg_cpu));
            printf("generate_legs_total gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f spread_pct=%.1f "
                   "hits=%llu misses=%llu h2d_experts=%llu cpu_experts=%llu h2d_mib=%.1f "
                   "cache_lookups=%llu cache_hits=%llu\n",
                   leg_gpu, leg_h2d, leg_cpu, hi > 0.0 ? (hi - lo) * 100.0 / hi : 0.0,
                   (unsigned long long) leg_hits, (unsigned long long) leg_miss,
                   (unsigned long long) leg_h2d_n, (unsigned long long) leg_cpu_n,
                   (double) leg_h2d_bytes / (1024.0 * 1024.0),
                   (unsigned long long) cs.lookups, (unsigned long long) cs.hits);
        }
        if (ids_out != NULL &&
            qrt_model_write_ids(ids_out, generated.data(), generated.size(), e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        if (tel != NULL) {
            if (qrt_telemetry_failed(tel)) {
                printf("generate_telemetry had write failures after %llu rows\n",
                       (unsigned long long) tel->rows);
            }
            qrt_telemetry_close(tel);
            tel = NULL;
        }
        rc = 0;
    } while (0);

    for (size_t i = layers.size(); i > 0; --i) {
        if (layers[i - 1] != NULL) {
            qrt_layer_close(dev, layers[i - 1]);
        }
    }
    if (tel != NULL) {
        qrt_telemetry_close(tel);
    }
    if (cache != NULL) {
        qrt_cache_close(cache);
    }
    for (size_t i = 0; i < cache_bufs.size(); ++i) {
        if (cache_bufs[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, cache_bufs[i], cache_mems[i]);
        }
    }
    if (cpu != NULL) {
        qrt_cpu_moe_free(cpu);
    }
    if (slot_mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, slot, slot_mem);
    }
    if (wstore != NULL) {
        qrt_weights_close(wstore);
    }
    return rc;
}
