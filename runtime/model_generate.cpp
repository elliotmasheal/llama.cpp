// The persistent decode slice (M3-2) and the speculation cycle (M3-3): every layer is opened
// once, the prompt prefills through those objects and generation then runs with the KV and
// DeltaNet state carried. It owns the expert cache, the miss legs and telemetry, which the
// prefill and logits paths never touch.
//
// One session serves both decode shapes: rows = 1 is the serial greedy reference, rows = k is
// a verified speculative pass. The compare of the two is what gates accept/rollback.

#include "model.h"
#include "vk_transfer.h"

#include "model_internal.h"
#include "model_io.h"
#include "qrt_config.h"
#include "qrt_telemetry.h"
#include "qrt_util.h"
#include "spec.h"

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

namespace {

// Everything a decode session holds open. The owner frees the device, the GGUF reader and
// the staging buffer when the frame ends; the layers, the cache and its buffers are closed
// explicitly because they must go before the device does.
struct qrt_gen {
    qrt_model_run_owner owner;
    qrt_vk_dev * dev = nullptr;
    qrt_gguf * g = nullptr;
    qrt_vk_staging * stg = nullptr;
    VkBuffer slot = VK_NULL_HANDLE;
    VkDeviceMemory slot_mem = VK_NULL_HANDLE;
    uint64_t slot_bytes = 0;
    std::vector<qrt_layer *> layers;
    const qrt_gguf_tensor * emb = nullptr;
    const qrt_gguf_tensor * outw = nullptr;
    std::vector<float> onorm;
    std::vector<float> x;        // the rows of the pass in flight
    uint32_t E = 0;
    uint32_t V = 0;
    uint32_t n_layers = 0;
    uint32_t chunk = 0;
    uint32_t n_ctx = 0;
    uint32_t rows_max = 0;
    qrt_cache * cache = nullptr;
    qrt_cpu_moe * cpu = nullptr;
    std::vector<VkBuffer> cache_bufs;
    std::vector<VkDeviceMemory> cache_mems;
    std::vector<uint32_t> cache_slots;
    qrt_split_rates rates{};
    qrt_split_carry carry{};
    int cache_on = 0;
    qrt_telemetry * tel = nullptr;
    // leg totals across the session, for the report at the end of the run
    double leg_gpu = 0.0, leg_h2d = 0.0, leg_cpu = 0.0;
    uint64_t leg_hits = 0, leg_miss = 0, leg_h2d_n = 0, leg_cpu_n = 0, leg_h2d_bytes = 0;
};

void gen_close(qrt_gen * s) {
    for (size_t i = s->layers.size(); i > 0; --i) {
        if (s->layers[i - 1] != NULL) {
            qrt_layer_close(s->dev, s->layers[i - 1]);
        }
    }
    s->layers.clear();
    if (s->tel != NULL) {
        qrt_telemetry_close(s->tel);
        s->tel = NULL;
    }
    if (s->cache != NULL) {
        qrt_cache_close(s->cache);
        s->cache = NULL;
    }
    for (size_t i = 0; i < s->cache_bufs.size(); ++i) {
        if (s->cache_bufs[i] != VK_NULL_HANDLE) {
            qrt_vk_free(s->dev, s->cache_bufs[i], s->cache_mems[i]);
        }
    }
    s->cache_bufs.clear();
    s->cache_mems.clear();
    if (s->cpu != NULL) {
        qrt_cpu_moe_free(s->cpu);
        s->cpu = NULL;
    }
    if (s->slot_mem != VK_NULL_HANDLE) {
        qrt_vk_free(s->dev, s->slot, s->slot_mem);
        s->slot = VK_NULL_HANDLE;
        s->slot_mem = VK_NULL_HANDLE;
    }
    // the device stays open only as long as the session does
    if (s->stg != NULL) {
        qrt_vk_staging_destroy(s->dev, s->stg);
        s->stg = NULL;
        s->owner.stg = NULL;
    }
    if (s->g != NULL) {
        qrt_gguf_close(s->g);
        s->g = NULL;
        s->owner.g = NULL;
    }
    if (s->dev != NULL) {
        qrt_vk_close(s->dev);
        s->dev = NULL;
        s->owner.dev = NULL;
    }
}

// Open the device, the reader and every layer of the session. snap_rows is the widest
// speculative pass the caller may run: it sizes the DeltaNet state ring, and 0 keeps the
// serial shape (two ring slots, no per-row snapshots). n_reserve is how many positions past
// the prompt the caller may still address.
int gen_open(qrt_gen * s, const qrt_config * cfg, const qrt_model_options * opt,
             const char * gguf_path, const char * store_prefix, const char * spv_dir,
             uint32_t n_tokens, uint32_t n_reserve, uint32_t snap_rows, char * err,
             size_t err_size) {
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &s->dev, err, err_size) != 0) {
        return -1;
    }
    s->owner.dev = s->dev;
    if (qrt_gguf_open(gguf_path, &s->g, err, err_size) != 0) {
        return -1;
    }
    s->owner.g = s->g;
    const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(s->g);
    if (hp == NULL || !(hp->present & QRT_HP_BLOCK_COUNT) || hp->block_count <= 0) {
        return fail(err, err_size, "missing or invalid block_count");
    }
    const int32_t nextn = hp->present & QRT_HP_NEXTN ? hp->nextn : 0;
    if (nextn < 0 || nextn >= hp->block_count ||
        opt->n_layers > (uint32_t) (hp->block_count - nextn)) {
        return fail(err, err_size, "requested %u layers, block_count=%d nextn=%d",
                    opt->n_layers, hp->block_count, nextn);
    }
    if (hp->n_embd <= 0 || hp->n_embd % 32 != 0) {
        return fail(err, err_size, "n_embd must be a positive multiple of 32");
    }
    s->E = (uint32_t) hp->n_embd;
    s->n_layers = opt->n_layers;
    s->emb = qrt_gguf_find(s->g, "token_embd.weight");
    s->outw = qrt_gguf_find(s->g, "output.weight");
    if (s->emb == NULL || s->emb->type != 8 || s->emb->n_dims != 2 || s->emb->ne[0] != s->E ||
        s->emb->ne[1] == 0 || s->emb->ne[1] > INT32_MAX) {
        return fail(err, err_size, "token_embd must be q8_0 with n_embd x vocabulary rows");
    }
    if (s->outw == NULL || s->outw->type != 8 || s->outw->n_dims != 2 ||
        s->outw->ne[0] != s->E || s->outw->ne[1] != s->emb->ne[1]) {
        return fail(err, err_size, "output must match the q8_0 embedding shape");
    }
    s->V = (uint32_t) s->emb->ne[1];
    if (qrt_gguf_read_f32(s->g, "output_norm.weight", &s->onorm, err, err_size) != 0 ||
        s->onorm.size() != s->E) {
        return fail(err, err_size, "output_norm: %s", err[0] ? err : "wrong width");
    }
    const uint64_t n_ctx64 = (uint64_t) n_tokens + n_reserve + 1u;
    if (n_ctx64 > QRT_KV_CTX_TOKENS) {
        return fail(err, err_size, "prompt %u + %u reserved exceeds the %u context",
                    n_tokens, n_reserve, (uint32_t) QRT_KV_CTX_TOKENS);
    }
    s->n_ctx = (uint32_t) n_ctx64;
    s->chunk = (uint32_t) cfg->prefill_chunk;
    if (s->chunk > 64u) {
        // The 40-layer slice is sized for a small resident chunk. A 2048-token chunk across
        // 40 layers does not fit in 8 GiB; the shared scratch arena is the next step.
        s->chunk = 32u;
    }
    if (s->chunk < 32u || (s->chunk % 32u) != 0u) {
        return fail(err, err_size, "generate chunk %u is not a 32-aligned 32..64", s->chunk);
    }
    s->rows_max = std::max(s->chunk, snap_rows);

    qrt_weights * wstore = NULL;
    if (qrt_weights_open(store_prefix, &wstore, err, err_size) != 0) {
        return -1;
    }
    const uint32_t n_expert = qrt_weights_experts(wstore);
    const uint64_t max_expert = qrt_weights_max_expert_bytes(wstore);
    qrt_weights_close(wstore);
    if (n_expert == 0 || max_expert == 0) {
        return fail(err, err_size, "the store has no experts");
    }
    s->slot_bytes = (uint64_t) n_expert * max_expert;
    s->slot_bytes = (s->slot_bytes + 65535u) & ~(uint64_t) 65535u;
    if (qrt_vk_alloc(s->dev, s->slot_bytes, QRT_VK_USAGE_STORAGE |
                     QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                     &s->slot, &s->slot_mem, err, err_size) != 0) {
        return fail(err, err_size, "shared FFN slot: %s", err);
    }
    const uint64_t out_row = (uint64_t) s->chunk * s->E * 4;
    if (qrt_vk_staging_create(s->dev, out_row, &s->stg, err, err_size) != 0) {
        return -1;
    }
    s->owner.stg = s->stg;

    qrt_layer_cfg lcfg{};
    lcfg.ffn = 1;
    lcfg.single_submit = 1;
    lcfg.dn_recur = 1;
    lcfg.dn_snap_rows = snap_rows;
    lcfg.kv_k = cfg->kv_k;
    lcfg.kv_v = cfg->kv_v;
    lcfg.kv_page_tokens = (uint32_t) cfg->kv_page_tokens;
    lcfg.n_ctx = s->n_ctx;
    lcfg.ffn_slot = s->slot;
    lcfg.ffn_slot_bytes = s->slot_bytes;
    s->layers.assign(s->n_layers, nullptr);
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        if (qrt_layer_open(s->dev, gguf_path, store_prefix, l, s->chunk, spv_dir, &lcfg,
                           &s->layers[l], err, err_size) != 0) {
            return fail(err, err_size, "open layer %u: %s", l, err);
        }
        if (qrt_layer_zero_state(s->layers[l], err, err_size) != 0) {
            return fail(err, err_size, "zero layer %u: %s", l, err);
        }
    }
    s->x.assign((size_t) s->rows_max * s->E, 0.0f);
    return 0;
}

// Layer-first prefill through the open layers; x comes back holding the n_tokens final
// hidden rows, which is what the head and the first decode step read.
int gen_prefill(qrt_gen * s, const uint32_t * ids, uint32_t n_tokens, std::vector<float> & x,
                char * err, size_t err_size) {
    x.assign((size_t) n_tokens * s->E, 0.0f);
    if (qrt_model_embed_rows(s->g, s->emb, s->E, s->V, ids, n_tokens, x.data(), err,
                             err_size) != 0) {
        return fail(err, err_size, "embedding: %s", err);
    }
    std::vector<float> xnext((size_t) s->chunk * s->E);
    const uint32_t n_chunks = 1u + (n_tokens - 1u) / s->chunk;
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        const double t0 = now_ms();
        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t pos = c * s->chunk;
            const uint32_t len = n_tokens - pos < s->chunk ? n_tokens - pos : s->chunk;
            if (qrt_layer_set_input_rows(s->layers[l], x.data() + (size_t) pos * s->E, len, err,
                                         err_size) != 0 ||
                qrt_layer_run_chunk(s->layers[l], pos, len, err, err_size) != 0 ||
                qrt_vk_download(s->dev, s->stg, qrt_layer_out(s->layers[l]), 0, xnext.data(),
                                (uint64_t) len * s->E * 4, err, err_size) != 0) {
                return fail(err, err_size, "prefill layer %u chunk %u: %s", l, c, err);
            }
            memcpy(x.data() + (size_t) pos * s->E, xnext.data(), (size_t) len * s->E * 4);
        }
        printf("generate_prefill layer=%u chunks=%u ms=%.1f\n", l, n_chunks, now_ms() - t0);
    }
    return 0;
}

// The decode expert cache (M3-2a). Prefill packed every used expert into the shared slot;
// decode switches to per-layer slot arrays sized by the cache plan, and the split sends the
// misses to H2D or the CPU leg. A failed cache *allocation* falls back to the prefill slot
// (slow but correct); a plan that cannot be built is an error the caller reports.
int gen_decode_cache(qrt_gen * s, const qrt_config * cfg, const char * store_prefix,
                     char * err, size_t err_size) {
    if (cfg->expert_cache_mib <= 0) {
        return 0;
    }
    uint64_t layer_slot[QRT_CACHE_MAX_LAYERS] = {0};
    qrt_weights * wstore = NULL;
    if (qrt_weights_open(store_prefix, &wstore, err, err_size) != 0) {
        return -1;
    }
    int size_ok = 1;
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        if (qrt_weights_layer_expert_bytes(wstore, l, &layer_slot[l]) != 0) {
            size_ok = 0;
            break;
        }
    }
    const uint32_t store_experts = qrt_weights_experts(wstore);
    qrt_weights_close(wstore);
    qrt_cache_plan plan{};
    if (!size_ok ||
        qrt_cache_plan_build(s->n_layers, store_experts, layer_slot, NULL,
                             (uint64_t) cfg->expert_cache_mib << 20, cfg->cache_policy,
                             QRT_CACHE_DEFAULT_PROTECTED_PCT, &plan, err, err_size) != 0) {
        return fail(err, err_size, "expert cache plan: %s", err[0] ? err : "no layer sizes");
    }
    // The prefill slot has done its job; the cache takes its VRAM.
    qrt_vk_free(s->dev, s->slot, s->slot_mem);
    s->slot = VK_NULL_HANDLE;
    s->slot_mem = VK_NULL_HANDLE;
    int alloc_ok = 1;
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        const uint64_t bytes = (uint64_t) plan.slots_per_layer[l] * plan.slot_bytes[l];
        VkBuffer b = VK_NULL_HANDLE;
        VkDeviceMemory m = VK_NULL_HANDLE;
        if (bytes != 0 &&
            qrt_vk_alloc(s->dev, bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         &b, &m, err, err_size) != 0) {
            alloc_ok = 0;
            break;
        }
        s->cache_bufs.push_back(b);
        s->cache_mems.push_back(m);
        s->cache_slots.push_back(plan.slots_per_layer[l]);
    }
    if (!alloc_ok) {
        for (size_t i = 0; i < s->cache_bufs.size(); ++i) {
            if (s->cache_bufs[i] != VK_NULL_HANDLE) {
                qrt_vk_free(s->dev, s->cache_bufs[i], s->cache_mems[i]);
            }
        }
        s->cache_bufs.clear();
        s->cache_mems.clear();
        s->cache_slots.clear();
        if (qrt_vk_alloc(s->dev, s->slot_bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         &s->slot, &s->slot_mem, err, err_size) != 0) {
            return fail(err, err_size, "the cache failed and the fallback slot did not fit: %s",
                        err);
        }
        printf("generate_cache disabled: %s\n", err[0] ? err : "allocation");
        err[0] = '\0';
        return 0;
    }
    if (qrt_cache_open(&plan, NULL, &s->cache, err, err_size) != 0) {
        return -1;
    }
    s->cpu = qrt_cpu_moe_create(1);
    if (s->cpu == NULL) {
        return fail(err, err_size, "out of memory for the CPU expert core");
    }
    qrt_miss_split_default_rates(&s->rates);
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        if (qrt_layer_set_cache(s->layers[l], s->cache, s->cache_bufs[l], plan.slot_bytes[l],
                                s->cpu, cfg->miss_split, &s->rates, &s->carry, err,
                                err_size) != 0) {
            return fail(err, err_size, "cache on layer %u: %s", l, err);
        }
    }
    s->cache_on = 1;
    printf("generate_cache slots=%llu mib=%llu degraded=%u policy=%s split=%s threads=8\n",
           (unsigned long long) plan.slots, (unsigned long long) (plan.bytes >> 20),
           plan.degraded_layers, qrt_cache_policy_name(cfg->cache_policy),
           cfg->miss_split == QRT_MISS_SPLIT_AUTO ? "auto" : "fixed");
    return 0;
}

// Greedy argmax of one logits row, with the runner-up margin the fidelity gate reports.
void gen_pick(const std::vector<float> & lg, uint32_t row, uint32_t V, uint32_t * id,
              float * best, float * margin) {
    const float * p = lg.data() + (size_t) row * V;
    uint32_t bi = 0, si = 0;
    float bv = p[0], sv = -INFINITY;
    for (uint32_t v = 1; v < V; ++v) {
        if (p[v] > bv) {
            sv = bv; si = bi; bv = p[v]; bi = v;
        } else if (p[v] > sv) {
            sv = p[v]; si = v;
        }
    }
    (void) si;
    *id = bi; *best = bv; *margin = bv - sv;
}

// The host output head over the pass's hidden rows: n_rows rows of full-vocabulary logits.
int gen_head(qrt_gen * s, const float * hidden, uint32_t rows, std::vector<float> & logits,
             char * err, size_t err_size) {
    return qrt_model_head_rows(s->g, s->outw, s->onorm.data(), hidden, NULL, s->E, s->V, rows,
                               rows, 0, &logits, NULL, err, err_size);
}

// rel_L1 of one logits row against the serial arm's row for the same sequence index. A
// batched pass and a single-row pass do not share a tile shape, so bit equality is not the
// bar here (RESULTS 86.1: m=1 is its own kernel); 1e-4 is. Lost state is orders of magnitude
// above it: the chunk check's no-carry counterexample lands at 2.2e-01.
double gen_row_rel_l1(const float * a, const float * b, uint32_t n, double * max_abs) {
    double num = 0.0, den = 0.0, m = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = fabs((double) a[i] - (double) b[i]);
        num += d;
        den += fabs((double) b[i]);
        if (d > m) {
            m = d;
        }
    }
    if (max_abs != NULL) {
        *max_abs = m;
    }
    return den > 0.0 ? num / den : (num > 0.0 ? 1.0 : 0.0);
}

// One verified pass of `rows` tokens at pos0 .. pos0+rows-1: embed, run every layer, leave
// the resulting hidden rows in s->x and fold this pass's leg counters into the totals.
// spec 1 takes the snapshot path, so qrt_spec_rollback() can put the sequence back anywhere
// inside these rows.
int gen_pass(qrt_gen * s, uint32_t pos0, const uint32_t * toks, uint32_t rows, int spec,
             double * gpu_ms, double * h2d_ms, double * cpu_ms,
             uint32_t * hits, uint32_t * miss, uint32_t * h2d_n, uint32_t * cpu_n,
             uint64_t * h2d_bytes, char * err, size_t err_size) {
    *gpu_ms = *h2d_ms = *cpu_ms = 0.0;
    *hits = *miss = *h2d_n = *cpu_n = 0;
    *h2d_bytes = 0;
    std::vector<float> xin((size_t) rows * s->E);
    if (qrt_model_embed_rows(s->g, s->emb, s->E, s->V, toks, rows, xin.data(), err,
                             err_size) != 0) {
        return fail(err, err_size, "decode embedding: %s", err);
    }
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        if (qrt_layer_set_input_rows(s->layers[l], xin.data(), rows, err, err_size) != 0 ||
            (spec ? qrt_layer_run_spec(s->layers[l], pos0, rows, err, err_size)
                  : qrt_layer_run_chunk(s->layers[l], pos0, rows, err, err_size))) {
            return fail(err, err_size, "decode layer %u pos %u: %s", l, pos0, err);
        }
        if (qrt_vk_download(s->dev, s->stg, qrt_layer_out(s->layers[l]), 0, xin.data(),
                            (uint64_t) rows * s->E * 4, err, err_size) != 0) {
            return fail(err, err_size, "download layer %u pos %u: %s", l, pos0, err);
        }
        if (s->cache_on) {
            const qrt_ffn_legs lg = qrt_layer_last_legs(s->layers[l]);
            const double gpu = qrt_layer_gpu_ms(s->layers[l]);
            *gpu_ms += gpu;
            *h2d_ms += lg.h2d_ms;
            *cpu_ms += lg.cpu_ms;
            *hits += lg.hits;
            *miss += lg.misses;
            *h2d_n += lg.h2d;
            *cpu_n += lg.cpu;
            *h2d_bytes += lg.h2d_bytes;
            s->leg_gpu += gpu;
            s->leg_h2d += lg.h2d_ms;
            s->leg_cpu += lg.cpu_ms;
            s->leg_hits += lg.hits;
            s->leg_miss += lg.misses;
            s->leg_h2d_n += lg.h2d;
            s->leg_cpu_n += lg.cpu;
            s->leg_h2d_bytes += lg.h2d_bytes;
            if (qrt_layer_take_cpu_out(s->layers[l], xin.data(), rows) != 0) {
                return fail(err, err_size, "layer %u CPU output at pos %u", l, pos0);
            }
        }
    }
    memcpy(s->x.data(), xin.data(), (size_t) rows * s->E * 4);
    return 0;
}

void gen_legs(const qrt_gen * s, const char * tag, uint32_t step, double gpu, double h2d,
              double cpu, uint32_t hits, uint32_t miss, uint32_t h2d_n, uint32_t cpu_n) {
    if (s->cache_on) {
        printf("generate_legs %s=%u gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u "
               "h2d=%u cpu=%u\n", tag, step, gpu, h2d, cpu, hits, miss, h2d_n, cpu_n);
    }
}

void gen_report(const qrt_gen * s, size_t n_ids) {
    uint32_t max_submits = 0, max_dispatches = 0;
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        if (s->layers[l] != NULL) {
            max_submits = std::max(max_submits, qrt_layer_submits(s->layers[l]));
            max_dispatches = std::max(max_dispatches, qrt_layer_dispatches(s->layers[l]));
        }
    }
    printf("generate_done ids=%zu alloc_mib=%.1f peak_mib=%.1f failures=%u "
           "submits_per_layer=%u dispatches_max=%u topk_readbacks_per_layer=1\n",
           n_ids, (double) s->dev->alloc_bytes / (1024.0 * 1024.0),
           (double) s->dev->alloc_peak_bytes / (1024.0 * 1024.0), s->dev->alloc_failures,
           max_submits, max_dispatches);
    if (s->cache_on) {
        qrt_cache_stats cs{};
        qrt_cache_get_stats(s->cache, &cs);
        const double hi = std::max(s->leg_gpu, std::max(s->leg_h2d, s->leg_cpu));
        const double lo = std::min(s->leg_gpu, std::min(s->leg_h2d, s->leg_cpu));
        printf("generate_legs_total gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f spread_pct=%.1f "
               "hits=%llu misses=%llu h2d_experts=%llu cpu_experts=%llu h2d_mib=%.1f "
               "cache_lookups=%llu cache_hits=%llu\n",
               s->leg_gpu, s->leg_h2d, s->leg_cpu,
               hi > 0.0 ? (hi - lo) * 100.0 / hi : 0.0,
               (unsigned long long) s->leg_hits, (unsigned long long) s->leg_miss,
               (unsigned long long) s->leg_h2d_n, (unsigned long long) s->leg_cpu_n,
               (double) s->leg_h2d_bytes / (1024.0 * 1024.0),
               (unsigned long long) cs.lookups, (unsigned long long) cs.hits);
    }
}

} // namespace

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
    qrt_gen s;
    std::vector<float> x;
    std::vector<uint32_t> generated;
    int rc = -1;
    do {
        const uint32_t BK = opt->batch_k > 1 ? opt->batch_k : 1u;
        if (gen_open(&s, cfg, opt, gguf_path, store_prefix, spv_dir, n_tokens, n_new, BK, err,
                     err_size) != 0) {
            break;
        }
        printf("generate_run tokens=%u new=%u layers=%u chunk=%u n_ctx=%u slot_mib=%.1f\n",
               n_tokens, n_new, s.n_layers, s.chunk, s.n_ctx,
               (double) s.slot_bytes / (1024.0 * 1024.0));
        if (gen_prefill(&s, ids, n_tokens, x, err, err_size) != 0) {
            break;
        }
        if (gen_decode_cache(&s, cfg, store_prefix, err, err_size) != 0) {
            break;
        }
        if (cfg->telemetry == 1) {
            char terr[256] = {0};
            s.tel = qrt_telemetry_open(cfg->telemetry_path, terr, sizeof(terr));
            if (s.tel == NULL) {
                printf("generate_telemetry disabled: %s\n", terr);
            }
        }
        std::vector<float> logits;
        uint32_t tok = 0;
        float best = 0.0f, margin = 0.0f;
        if (gen_head(&s, x.data() + (size_t) (n_tokens - 1u) * s.E, 1, logits, err,
                     err_size) != 0) {
            fail(err, err_size, "prefill head: %s", err);
            break;
        }
        gen_pick(logits, 0, s.V, &tok, &best, &margin);
        generated.push_back(tok);
        printf("generate_token step=0 pos=%u id=%u logit=%.4f margin=%.4f\n",
               n_tokens, tok, best, margin);

        uint32_t pos = n_tokens;
        if (BK > 1) {
            // Diagnostic decode width (M3-2a): one pass carries k rows so the miss legs are
            // measured at k=5. The rows are prompt ids fed teacher-forced; the head output is
            // not used as the next input, so this is not an acceptance path.
            std::vector<uint32_t> bid(BK);
            uint32_t produced = 1, pass = 0;
            while (produced < n_new) {
                const uint32_t k = std::min(BK, n_new - produced);
                for (uint32_t i = 0; i < k; ++i) {
                    bid[i] = ids[(size_t) (pass * BK + i) % n_tokens];
                }
                double gpu = 0.0, h2d = 0.0, cpu = 0.0;
                uint32_t hits = 0, miss = 0, h2d_n = 0, cpu_n = 0;
                uint64_t h2d_bytes = 0;
                const double t0 = now_ms();
                if (gen_pass(&s, pos, bid.data(), k, 0, &gpu, &h2d, &cpu, &hits, &miss,
                             &h2d_n, &cpu_n, &h2d_bytes, err, err_size) != 0) {
                    break;
                }
                if (gen_head(&s, s.x.data() + (size_t) (k - 1u) * s.E, 1, logits, err,
                             err_size) != 0) {
                    fail(err, err_size, "batch head pass %u: %s", pass, err);
                    break;
                }
                gen_pick(logits, 0, s.V, &tok, &best, &margin);
                generated.push_back(tok);
                printf("generate_batch pass=%u pos=%u k=%u id=%u logit=%.4f margin=%.4f "
                       "gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u h2d=%u cpu=%u "
                       "pass_ms=%.1f\n",
                       pass, pos, k, tok, best, margin, gpu, h2d, cpu, hits, miss, h2d_n,
                       cpu_n, now_ms() - t0);
                pos += k;
                produced += k;
                ++pass;
            }
            if (err[0] != '\0') {
                break;
            }
        } else {
            for (uint32_t step = 1; step < n_new; ++step) {
                double gpu = 0.0, h2d = 0.0, cpu = 0.0;
                uint32_t hits = 0, miss = 0, h2d_n = 0, cpu_n = 0;
                uint64_t h2d_bytes = 0;
                const double t0 = now_ms();
                if (gen_pass(&s, pos, &generated.back(), 1, 0, &gpu, &h2d, &cpu, &hits,
                             &miss, &h2d_n, &cpu_n, &h2d_bytes, err, err_size) != 0) {
                    break;
                }
                gen_legs(&s, "step", step, gpu, h2d, cpu, hits, miss, h2d_n, cpu_n);
                if (gen_head(&s, s.x.data(), 1, logits, err, err_size) != 0) {
                    fail(err, err_size, "decode head step %u: %s", step, err);
                    break;
                }
                gen_pick(logits, 0, s.V, &tok, &best, &margin);
                generated.push_back(tok);
                printf("generate_token step=%u pos=%u id=%u logit=%.4f margin=%.4f "
                       "layer_ms=%.1f\n", step, pos, tok, best, margin, now_ms() - t0);
                if (s.tel != NULL) {
                    qrt_pass_stats ps{};
                    ps.k = 1;
                    ps.pos = pos;
                    ps.accepted = 1;
                    ps.hit_experts = hits;
                    ps.miss_experts = miss;
                    ps.miss_cpu = cpu_n;
                    ps.miss_h2d = h2d_n;
                    ps.h2d_bytes = h2d_bytes;
                    ps.draft_ms = 0.0;
                    ps.gpu_ms = gpu;
                    ps.cpu_leg_ms = cpu;
                    ps.pass_ms = now_ms() - t0;
                    qrt_telemetry_pass(s.tel, step, &ps);
                }
                ++pos;
            }
            if (err[0] != '\0') {
                break;
            }
        }
        gen_report(&s, generated.size());
        if (ids_out != NULL &&
            qrt_model_write_ids(ids_out, generated.data(), generated.size(), err,
                                err_size) != 0) {
            break;
        }
        if (s.tel != NULL) {
            if (qrt_telemetry_failed(s.tel)) {
                printf("generate_telemetry had write failures after %llu rows\n",
                       (unsigned long long) s.tel->rows);
            }
        }
        rc = 0;
    } while (0);
    gen_close(&s);
    return rc;
}

// The M3-3 acceptance run (design 7, accept and rollback): decode the same prompt twice from
// the same state, once serially and once as speculative cycles whose drafts are deliberately
// broken at position i, and require that both arms agree token for token. Every verified row
// is also compared against the serial arm's logits for the same sequence index, so a rollback
// that lost state shows up even when the argmax happens to survive. The no-rollback arm is
// the counterexample that proves the comparison can see it.
int qrt_model_spec_check(const qrt_config * cfg, const qrt_model_options * opt,
                         const char * gguf_path, const char * store_prefix, const char * spv_dir,
                         const uint32_t * ids, uint32_t n_tokens, uint32_t n_new,
                         int rollback_off, char * err, size_t err_size) {
    if (cfg == NULL || opt == NULL || ids == NULL || n_tokens == 0 || n_new < 2) {
        return fail(err, err_size, "spec-check needs a prompt and at least 2 new tokens");
    }
    const uint32_t k = (uint32_t) cfg->spec_k;
    if (k < 2 || k > 8) {
        return fail(err, err_size, "spec-check needs QRT_SPEC_K in 2..8, got %d", cfg->spec_k);
    }
    if (n_new > 64) {
        return fail(err, err_size, "spec-check keeps both arms' logits: use at most 64 tokens");
    }
    int rc = -1;
    std::vector<uint32_t> serial_ids;
    std::vector<float> serial_logits;      // one full row per emitted token, for the compare
    std::vector<uint32_t> spec_ids;
    qrt_gen s;
    do {
        // arm A: the serial greedy reference, one row per pass
        std::vector<float> x;
        if (gen_open(&s, cfg, opt, gguf_path, store_prefix, spv_dir, n_tokens, n_new, 0, err,
                     err_size) != 0) {
            break;
        }
        printf("spec_run tokens=%u new=%u layers=%u k=%u\n", n_tokens, n_new, s.n_layers, k);
        if (gen_prefill(&s, ids, n_tokens, x, err, err_size) != 0) {
            break;
        }
        // both arms take the same decode path, so the cache and the miss split stay open in
        // the compare; fix --miss-split to keep the two arms' sums in the same order
        if (gen_decode_cache(&s, cfg, store_prefix, err, err_size) != 0) {
            break;
        }
        std::vector<float> logits;
        uint32_t tok = 0;
        float best = 0.0f, margin = 0.0f;
        if (gen_head(&s, x.data() + (size_t) (n_tokens - 1u) * s.E, 1, logits, err,
                     err_size) != 0) {
            break;
        }
        gen_pick(logits, 0, s.V, &tok, &best, &margin);
        serial_ids.push_back(tok);
        serial_logits.insert(serial_logits.end(), logits.begin(), logits.end());
        uint32_t pos = n_tokens;
        for (uint32_t step = 1; step < n_new; ++step) {
            double gpu = 0.0, h2d = 0.0, cpu = 0.0;
            uint32_t hits = 0, miss = 0, h2d_n = 0, cpu_n = 0;
            uint64_t h2d_bytes = 0;
            if (gen_pass(&s, pos, &serial_ids.back(), 1, 0, &gpu, &h2d, &cpu, &hits, &miss,
                         &h2d_n, &cpu_n, &h2d_bytes, err, err_size) != 0 ||
                gen_head(&s, s.x.data(), 1, logits, err, err_size) != 0) {
                break;
            }
            gen_pick(logits, 0, s.V, &tok, &best, &margin);
            serial_ids.push_back(tok);
            serial_logits.insert(serial_logits.end(), logits.begin(), logits.end());
            printf("spec_serial step=%u pos=%u id=%u logit=%.4f\n", step, pos, tok, best);
            ++pos;
        }
        if (err[0] != '\0') {
            break;
        }
        const uint32_t E = s.E, V = s.V;
        const uint32_t n_layers = s.n_layers;
        gen_close(&s);

        // arm B: the same prompt, the same first token, then cycles of up to k rows
        if (gen_open(&s, cfg, opt, gguf_path, store_prefix, spv_dir, n_tokens,
                     rollback_off ? n_new * k : n_new, k, err, err_size) != 0) {
            break;
        }
        if (gen_prefill(&s, ids, n_tokens, x, err, err_size) != 0) {
            break;
        }
        if (gen_decode_cache(&s, cfg, store_prefix, err, err_size) != 0) {
            break;
        }
        if (cfg->telemetry == 1) {
            char terr[256] = {0};
            s.tel = qrt_telemetry_open(cfg->telemetry_path, terr, sizeof(terr));
            if (s.tel == NULL) {
                printf("spec_telemetry disabled: %s\n", terr);
            }
        }
        if (gen_head(&s, x.data() + (size_t) (n_tokens - 1u) * E, 1, logits, err,
                     err_size) != 0) {
            break;
        }
        gen_pick(logits, 0, V, &tok, &best, &margin);
        spec_ids.push_back(tok);
        pos = n_tokens;
        uint32_t cycles = 0, accepted_sum = 0, rows_checked = 0, rows_bad = 0, rows_loose = 0;
        double worst_row = 0.0;
        // Under the auto split the two arms do not put the same experts on the same legs (the
        // multiplicity per miss differs), so there only the ids gate and the row gap is a
        // reading: the CPU leg already differs from the GPU kernel by ~1e-2 in a logit
        // (RESULTS.md#m3-2a-miss-split 87.2).
        // no cache means no legs at all, so the split does not make the arms differ
        const int same_leg = !s.cache_on || cfg->miss_split != QRT_MISS_SPLIT_AUTO;
        while (spec_ids.size() < n_new) {
            const uint32_t have = (uint32_t) spec_ids.size();
            const uint32_t left = (uint32_t) serial_ids.size() - have;
            // The drafts are the serial arm's continuation, so the accept length a correct
            // cycle should reach is known. One position per cycle is broken, which walks the
            // accept counts 0..k-1 that design 7 asks for.
            const uint32_t nd = left - 1u < k - 1u ? left - 1u : k - 1u;
            const uint32_t rows = nd + 1u;
            std::vector<uint32_t> draft(nd), pass_rows(rows), verified(rows);
            for (uint32_t i = 0; i < nd; ++i) {
                draft[i] = serial_ids[have + i];
            }
            const uint32_t break_at = cycles % k;
            if (nd != 0 && break_at < nd) {
                draft[break_at] ^= 0x10000u;
                if (draft[break_at] >= V) {
                    draft[break_at] = V - 1u;
                }
            }
            pass_rows[0] = spec_ids.back();
            for (uint32_t i = 0; i < nd; ++i) {
                pass_rows[i + 1u] = draft[i];
            }
            double gpu = 0.0, h2d = 0.0, cpu = 0.0;
            uint32_t hits = 0, miss = 0, h2d_n = 0, cpu_n = 0;
            uint64_t h2d_bytes = 0;
            const double t0 = now_ms();
            std::vector<float> all;
            if (gen_pass(&s, pos, pass_rows.data(), rows, nd != 0 ? 1 : 0, &gpu, &h2d,
                         &cpu, &hits, &miss, &h2d_n, &cpu_n, &h2d_bytes, err, err_size) != 0 ||
                gen_head(&s, s.x.data(), rows, all, err, err_size) != 0) {
                break;
            }
            uint32_t keep = 1;
            std::vector<uint32_t> out(rows);
            for (uint32_t i = 0; i < rows; ++i) {
                gen_pick(all, i, V, &verified[i], &best, &margin);
            }
            if (nd == 0) {
                out[0] = verified[0];   // no draft to accept: the row's own argmax is the token
            } else if (qrt_spec_emit(draft.data(), verified.data(), nd, out.data(), &keep) != 0) {
                fail(err, err_size, "spec-check: emit rejected its arguments");
                break;
            }
            // Rows 0..keep-1 carry the true tokens, so their hidden state and logits must be
            // the serial arm's for the same predicted index. The row after the break was fed
            // a wrong draft, so it is expected to differ and is not compared.
            if (!rollback_off) {
                for (uint32_t i = 0; i < keep; ++i) {
                    const uint32_t r = pos + i + 1u - n_tokens;
                    if (r >= serial_ids.size()) {
                        continue;
                    }
                    double max_abs = 0.0;
                    const double rel = gen_row_rel_l1(all.data() + (size_t) i * V,
                                                      serial_logits.data() + (size_t) r * V,
                                                      V, &max_abs);
                    ++rows_checked;
                    if (rel > worst_row) {
                        worst_row = rel;
                    }
                    // the token is the gate in every configuration; the row gap is only a
                    // gate when both arms put the same experts on the same legs
                    if (verified[i] != serial_ids[r] || (same_leg && !(rel <= 1e-4))) {
                        ++rows_bad;
                    } else if (!(rel <= 1e-4)) {
                        ++rows_loose;
                    }
                    if (verified[i] != serial_ids[r] || !(rel <= 1e-4)) {
                        printf("spec_rows cycle=%u row=%u rel_l1=%.3e max_abs=%.3e id=%u "
                               "want=%u legs=%s\n", cycles, i, rel, max_abs, verified[i],
                               serial_ids[r], same_leg ? "same" : "mixed");
                    }
                }
            }
            for (uint32_t i = 0; i < keep; ++i) {
                spec_ids.push_back(out[i]);
            }
            // the counterexample arm keeps every row, so the state and the length both move
            // past what was accepted and the next cycle starts from a broken sequence
            const uint32_t kept_rows = rollback_off ? rows : keep;
            if (nd != 0 && qrt_spec_rollback(s.layers.data(), n_layers, kept_rows, err,
                                             err_size) != 0) {
                break;
            }
            pos += kept_rows;
            ++cycles;
            accepted_sum += keep;
            printf("spec_cycle cycle=%u pos=%u rows=%u drafts=%u accepted=%u kept=%u "
                   "break_at=%u gpu_ms=%.2f pass_ms=%.1f\n",
                   cycles - 1u, pos - kept_rows, rows, nd, keep, kept_rows, break_at, gpu,
                   now_ms() - t0);
            if (s.tel != NULL) {
                qrt_pass_stats ps{};
                ps.k = rows;
                ps.pos = pos - kept_rows;
                ps.accepted = keep;
                ps.hit_experts = hits;
                ps.miss_experts = miss;
                ps.miss_cpu = cpu_n;
                ps.miss_h2d = h2d_n;
                ps.h2d_bytes = h2d_bytes;
                ps.draft_ms = 0.0;
                ps.gpu_ms = gpu;
                ps.cpu_leg_ms = cpu;
                ps.pass_ms = now_ms() - t0;
                qrt_telemetry_pass(s.tel, cycles, &ps);
            }
        }
        if (err[0] != '\0') {
            break;
        }
        gen_report(&s, spec_ids.size());
        uint32_t first_diff = 0;
        int ids_match = spec_ids.size() == serial_ids.size();
        for (uint32_t i = 0; ids_match && i < serial_ids.size(); ++i) {
            if (spec_ids[i] != serial_ids[i]) {
                ids_match = 0;
                first_diff = i;
            }
        }
        // with the rollback switched off the two arms must part ways, otherwise the check
        // would not be able to see a lost state at all
        const int ok = rollback_off ? !ids_match
                                    : (ids_match && rows_bad == 0 && rows_checked > 0);
        printf("spec_check tokens=%u k=%u cycles=%u E=%.3f serial=%zu spec=%zu ids_equal=%d "
               "first_diff=%u rows_checked=%u rows_bad=%u rows_loose=%u worst_rel_l1=%.3e "
               "legs=%s break_cases=%u rollback=%s verdict=%s\n",
               n_new, k, cycles, cycles ? (double) accepted_sum / cycles : 0.0,
               serial_ids.size(), spec_ids.size(), ids_match, first_diff, rows_checked,
               rows_bad, rows_loose, worst_row, same_leg ? "same" : "mixed", k,
               rollback_off ? "off" : "on", ok ? "PASS" : "FAIL");
        if (!rollback_off && !ids_match) {
            printf("spec_check FAIL at %u: serial %u, spec %u\n", first_diff,
                   serial_ids[first_diff], first_diff < spec_ids.size() ? spec_ids[first_diff]
                                                                       : 0u);
        }
        rc = ok ? 0 : 1;
    } while (0);
    gen_close(&s);
    return rc;
}
