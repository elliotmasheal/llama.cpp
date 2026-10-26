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

// The MTP draft block (design 4.1). blk.40 is a full-attention MoE layer that shares
// token_embd and output.weight with the main model, so the only extra weights are the
// prefix: enorm/hnorm on (embedding, hidden), their concat with the embedding first, and
// nextn.eh_proj back to n_embd - the order of the fork's graph_mtp. The block's output is
// normalised by nextn.shared_head_norm, which is both what the draft head reads and what
// the next draft step feeds back as h.
struct qrt_mtp {
    int      on = 0;
    qrt_layer * layer = nullptr;
    const qrt_gguf_tensor * eh_proj = nullptr;
    std::vector<float> enorm, hnorm, shnorm;   // n_embd each
    std::vector<double> h;                     // the normed hidden the next step consumes
    std::vector<double> concat;                // [2*n_embd] prefix input
    std::vector<float> x;                      // eh_proj output row, the block's input
    std::vector<float> out;                    // the block's raw output row
    std::vector<float> cand;                   // draft head logits
    std::vector<float> e_row;                  // the drafted token's embedding
    qrt_spec_vocab * vocab = nullptr;          // NULL = the whole vocabulary
    uint32_t steps = 0;
    uint64_t host_bytes = 0;                   // rows gathered and read on the host
    double step_ms = 0.0, gpu_ms = 0.0;
};
// Everything a decode session holds open. The owner frees the device, the GGUF reader and
// the staging buffer when the frame ends; the layers, the cache and its buffers are closed
// explicitly because they must go before the device does.
struct qrt_gen {
    qrt_mtp mtp;
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
    uint32_t mtp_block = 0;    // block_count - nextn, the MTP block index
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
    if (s->mtp.layer != NULL) {
        qrt_layer_close(s->dev, s->mtp.layer);
        s->mtp.layer = NULL;
    }
    qrt_spec_vocab_free(s->mtp.vocab);
    s->mtp.vocab = NULL;
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
struct qrt_mtp_cfg {
    int on;                // 1 = open blk.40 and draft from it
    int32_t draft_vocab;   // candidates per step, 0 = the whole vocabulary
    const char * vocab_path;
};

// The MTP block is opened with the session, after the main layers: it shares the FFN slot,
// the staging buffer, the embedding rows and the output head.
int mtp_open(qrt_gen * s, const char * gguf_path, const char * store_prefix,
             const char * spv_dir, const qrt_layer_cfg & lcfg, const qrt_mtp_cfg & mc,
             char * err, size_t err_size);

int gen_open(qrt_gen * s, const qrt_config * cfg, const qrt_model_options * opt,
             const char * gguf_path, const char * store_prefix, const char * spv_dir,
             uint32_t n_tokens, uint32_t n_reserve, uint32_t snap_rows,
             const qrt_mtp_cfg & mc, char * err, size_t err_size) {
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
    s->mtp_block = (uint32_t) (hp->block_count - nextn);
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
    if (mc.on != 0 && mtp_open(s, gguf_path, store_prefix, spv_dir, lcfg, mc, err,
                               err_size) != 0) {
        return -1;
    }
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
// misses to H2D or the CPU leg. A failed cache allocation leaves the session on the prefill
// slot (slow but correct); a plan that cannot be built is an error the caller reports.
int gen_decode_cache(qrt_gen * s, const qrt_config * cfg, const char * store_prefix,
                     char * err, size_t err_size) {
    if (cfg->expert_cache_mib <= 0) {
        return 0;
    }
    // The MTP block is a full MoE layer as well, so it takes part of the cache - and must,
    // because the shared prefill slot it was opened with is exactly what the cache replaces
    // (M3-3b).
    std::vector<qrt_layer *> cl = s->layers;
    std::vector<uint32_t> cb;
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        cb.push_back(l);
    }
    if (s->mtp.on != 0) {
        cl.push_back(s->mtp.layer);
        cb.push_back(s->mtp_block);
    }
    const uint32_t n_plan = (uint32_t) cl.size();
    if (n_plan > QRT_CACHE_MAX_LAYERS) {
        return fail(err, err_size, "%u layers do not fit the cache plan", n_plan);
    }
    uint64_t layer_slot[QRT_CACHE_MAX_LAYERS] = {0};
    qrt_weights * wstore = NULL;
    if (qrt_weights_open(store_prefix, &wstore, err, err_size) != 0) {
        return -1;
    }
    int size_ok = 1;
    for (uint32_t i = 0; i < n_plan; ++i) {
        if (qrt_weights_layer_expert_bytes(wstore, cb[i], &layer_slot[i]) != 0) {
            size_ok = 0;
            break;
        }
    }
    const uint32_t store_experts = qrt_weights_experts(wstore);
    qrt_weights_close(wstore);
    qrt_cache_plan plan{};
    if (!size_ok ||
        qrt_cache_plan_build(n_plan, store_experts, layer_slot, NULL,
                            (uint64_t) cfg->expert_cache_mib << 20, cfg->cache_policy,
                            QRT_CACHE_DEFAULT_PROTECTED_PCT, &plan, err, err_size) != 0) {
        return fail(err, err_size, "expert cache plan: %s", err[0] ? err : "no layer sizes");
    }
    // Allocate the slot arrays before giving the prefill slot up: a layer keeps the handle it
    // was opened with, so there is no fallback that could leave it pointing at freed VRAM.
    int alloc_ok = 1;
    for (uint32_t i = 0; alloc_ok && i < n_plan; ++i) {
        const uint64_t bytes = (uint64_t) plan.slots_per_layer[i] * plan.slot_bytes[i];
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
        s->cache_slots.push_back(plan.slots_per_layer[i]);
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
        printf("generate_cache disabled: %s\n", err[0] ? err : "allocation");
        err[0] = '\0';
        return 0;
    }
    // The prefill slot has done its job; the cache takes its VRAM.
    qrt_vk_free(s->dev, s->slot, s->slot_mem);
    s->slot = VK_NULL_HANDLE;
    s->slot_mem = VK_NULL_HANDLE;
    if (qrt_cache_open(&plan, NULL, &s->cache, err, err_size) != 0) {
        return -1;
    }
    s->cpu = qrt_cpu_moe_create(1);
    if (s->cpu == NULL) {
        return fail(err, err_size, "out of memory for the CPU expert core");
    }
    qrt_miss_split_default_rates(&s->rates);
    for (uint32_t i = 0; i < n_plan; ++i) {
        if (qrt_layer_set_cache(cl[i], s->cache, i, s->cache_bufs[i], plan.slot_bytes[i],
                                s->cpu, cfg->miss_split, &s->rates, &s->carry, err,
                                err_size) != 0) {
            return fail(err, err_size, "cache on block %u: %s", cb[i], err);
        }
    }
    s->cache_on = 1;
    printf("generate_cache layers=%u slots=%llu mib=%llu degraded=%u policy=%s split=%s "
           "threads=8\n",
           n_plan, (unsigned long long) plan.slots, (unsigned long long) (plan.bytes >> 20),
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

// Open the MTP block and load its four prefix tensors. The block itself is an ordinary
// full-attention MoE layer of the same store, so it shares the session's FFN slot and, when
// the decode cache is on, its own slot array (gen_decode_cache covers it).
int mtp_open(qrt_gen * s, const char * gguf_path, const char * store_prefix,
             const char * spv_dir, const qrt_layer_cfg & lcfg, const qrt_mtp_cfg & mc,
             char * err, size_t err_size) {
    qrt_mtp & m = s->mtp;
    const uint32_t B = s->mtp_block;
    const uint32_t E = s->E;
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.nextn.eh_proj.weight", B);
    m.eh_proj = qrt_gguf_find(s->g, name);
    if (m.eh_proj == NULL || m.eh_proj->type != 8 || m.eh_proj->n_dims != 2 ||
        m.eh_proj->ne[0] != (uint64_t) 2 * E || m.eh_proj->ne[1] != E) {
        return fail(err, err_size, "%s is not a q8_0 [%u x 2*n_embd] tensor", name, E);
    }
    const char * which[3] = { "enorm", "hnorm", "shared_head_norm" };
    std::vector<float> * dst[3] = { &m.enorm, &m.hnorm, &m.shnorm };
    for (int i = 0; i < 3; ++i) {
        snprintf(name, sizeof(name), "blk.%u.nextn.%s.weight", B, which[i]);
        err[0] = '\0';
        if (qrt_gguf_read_f32(s->g, name, dst[i], err, err_size) != 0 ||
            dst[i]->size() != E) {
            return fail(err, err_size, "%s: %s", name, err[0] ? err : "wrong width");
        }
    }
    if (mc.draft_vocab > 0) {
        if (mc.vocab_path == NULL) {
            return fail(err, err_size, "a restricted draft vocabulary needs --draft-vocab-path");
        }
        if (qrt_spec_vocab_open(mc.vocab_path, (uint32_t) mc.draft_vocab, &m.vocab, err,
                               err_size) != 0) {
            return -1;
        }
    }
    qrt_layer_cfg mcfg = lcfg;
    mcfg.dn_snap_rows = 0;   // full attention, and the draft is one row per step
    if (qrt_layer_open(s->dev, gguf_path, store_prefix, B, s->chunk, spv_dir, &mcfg,
                       &m.layer, err, err_size) != 0) {
        return fail(err, err_size, "open the MTP block %u: %s", B, err);
    }
    if (qrt_layer_zero_state(m.layer, err, err_size) != 0) {
        return -1;
    }
    m.e_row.assign(E, 0.0f);
    m.h.assign(E, 0.0);
    m.concat.assign((size_t) 2 * E, 0.0);
    m.x.assign(E, 0.0f);
    m.out.assign(E, 0.0f);
    m.cand.assign(m.vocab != NULL ? qrt_spec_vocab_count(m.vocab) : s->V, 0.0f);
    m.on = 1;
    printf("mtp_open block=%u candidates=%u rows=%s\n", B,
           m.vocab != NULL ? qrt_spec_vocab_count(m.vocab) : s->V,
           m.vocab != NULL ? "gathered" : "whole vocabulary");
    return 0;
}

// The hidden a draft step starts from: the last kept row of the pass that just ran, normed
// by output_norm. That is exactly the vector the fork hands the MTP graph (t_h_nextn).
void mtp_seed(qrt_gen * s, const float * hidden_row) {
    qrt_model_norm_row(hidden_row, s->onorm.data(), s->E, s->mtp.h.data());
}

// One draft step: enorm(embed) beside the running h, concat with the embedding first,
// nextn.eh_proj, the block at `pos`, then shared_head_norm into both the draft head and the
// next step's h.
int mtp_step(qrt_gen * s, uint32_t tok, uint32_t pos, uint32_t * id, float * logit,
             float * margin, char * err, size_t err_size) {
    qrt_mtp & m = s->mtp;
    if (m.on == 0) {
        return fail(err, err_size, "the session has no MTP block open");
    }
    const uint32_t E = s->E;
    const double t0 = now_ms();
    if (qrt_model_embed_rows(s->g, s->emb, E, s->V, &tok, 1, m.e_row.data(), err,
                            err_size) != 0) {
        return fail(err, err_size, "draft embedding: %s", err);
    }
    m.host_bytes += (uint64_t) (E / 32u) * 34u;
    // both halves are normed here: enorm on the embedding, hnorm on the h the previous
    // block or the verified pass produced (the fork's graph_mtp order)
    qrt_model_norm_row(m.e_row.data(), m.enorm.data(), E, m.concat.data());
    qrt_model_norm_row_d(m.h.data(), m.hnorm.data(), E, m.concat.data() + E);
    if (qrt_model_gemv_q8_0(s->g, m.eh_proj, 2u * E, NULL, E, m.concat.data(), m.x.data(),
                           err, err_size) != 0) {
        return fail(err, err_size, "nextn.eh_proj: %s", err);
    }
    m.host_bytes += (uint64_t) E * (2u * E / 32u) * 34u;
    if (qrt_layer_set_input_rows(m.layer, m.x.data(), 1, err, err_size) != 0 ||
        qrt_layer_run_chunk(m.layer, pos, 1, err, err_size) != 0 ||
        qrt_vk_download(s->dev, s->stg, qrt_layer_out(m.layer), 0, m.out.data(),
                        (uint64_t) E * 4, err, err_size) != 0) {
        return fail(err, err_size, "MTP block at pos %u: %s", pos, err);
    }
    qrt_model_norm_row(m.out.data(), m.shnorm.data(), E, m.h.data());
    const uint32_t * idx = m.vocab != NULL ? qrt_spec_vocab_ids(m.vocab) : NULL;
    const uint32_t n = m.vocab != NULL ? qrt_spec_vocab_count(m.vocab) : s->V;
    if (qrt_model_gemv_q8_0(s->g, s->outw, E, idx, n, m.h.data(), m.cand.data(), err,
                           err_size) != 0) {
        return fail(err, err_size, "draft head: %s", err);
    }
    m.host_bytes += (uint64_t) n * (E / 32u) * 34u;
    *id = qrt_spec_draft(m.cand.data(), idx, n, logit, margin);
    ++m.steps;
    m.step_ms += now_ms() - t0;
    m.gpu_ms += qrt_layer_gpu_ms(m.layer);
    return 0;
}

// n drafts for the cycle that feeds `first` at `pos`. The block writes its own KV rows at
// the positions the drafts would take; a rejected draft leaves a stale row there that the
// next cycle overwrites before anything reads it, as in the main layers.
int mtp_draft(qrt_gen * s, uint32_t first, uint32_t pos, uint32_t n,
             std::vector<uint32_t> * ids, char * err, size_t err_size) {
    ids->assign(n, 0u);
    uint32_t tok = first;
    for (uint32_t i = 0; i < n; ++i) {
        float logit = 0.0f, margin = 0.0f;
        if (mtp_step(s, tok, pos + i, &tok, &logit, &margin, err, err_size) != 0) {
            return -1;
        }
        (*ids)[i] = tok;
    }
    return 0;
}

// What one verified pass produced, in the shape both consumers report.
struct qrt_cycle {
    std::vector<uint32_t> draft;     // the rows-1 tokens the cycle tested
    std::vector<uint32_t> verified;  // one argmax per row; the last is the bonus
    std::vector<uint32_t> emit;      // the accepted drafts plus the bonus
    std::vector<float> logits;       // rows * V, kept only when the caller asks
    uint32_t rows = 0;
    uint32_t accepted = 0;
    uint32_t kept = 0;       // tokens the cycle emits, and the rows that stay
    uint32_t advance = 0;    // the position step: kept, or rows in the counterexample arm
    double gpu_ms = 0.0, h2d_ms = 0.0, cpu_ms = 0.0, pass_ms = 0.0;
    uint32_t hits = 0, miss = 0, h2d_n = 0, cpu_n = 0;
    uint64_t h2d_bytes = 0;
};

// Verify `pass_rows` in one pass, accept the longest draft prefix, take the verified token
// at the first mismatch, and roll every layer back to that length. kept is both the number
// of rows that stay and the number of tokens the cycle emits.
int gen_cycle(qrt_gen * s, uint32_t pos, const std::vector<uint32_t> & pass_rows,
              int keep_logits, int keep_all_rows, qrt_cycle * out, char * err,
              size_t err_size) {
    const uint32_t rows = (uint32_t) pass_rows.size();
    *out = qrt_cycle{};
    out->rows = rows;
    out->draft.assign(pass_rows.begin() + 1, pass_rows.end());
    const double t0 = now_ms();
    if (gen_pass(s, pos, pass_rows.data(), rows, rows > 1u ? 1 : 0, &out->gpu_ms,
                &out->h2d_ms, &out->cpu_ms, &out->hits, &out->miss, &out->h2d_n,
                &out->cpu_n, &out->h2d_bytes, err, err_size) != 0) {
        return -1;
    }
    std::vector<float> all;
    if (gen_head(s, s->x.data(), rows, all, err, err_size) != 0) {
        return fail(err, err_size, "cycle head at pos %u: %s", pos, err);
    }
    out->verified.assign(rows, 0u);
    float best = 0.0f, margin = 0.0f;
    for (uint32_t i = 0; i < rows; ++i) {
        gen_pick(all, i, s->V, &out->verified[i], &best, &margin);
    }
    std::vector<uint32_t> emit(rows);
    uint32_t kept = 1;
    if (rows > 1u) {
        if (qrt_spec_emit(out->draft.data(), out->verified.data(), rows - 1u, emit.data(),
                         &kept) != 0) {
            return fail(err, err_size, "the cycle could not emit its accepted prefix");
        }
    } else {
        emit[0] = out->verified[0];
    }
    emit.resize(kept);
    out->emit.swap(emit);
    out->accepted = kept - 1u;
    out->kept = kept;
    out->pass_ms = now_ms() - t0;
    if (keep_logits) {
        out->logits.swap(all);
    }
    // the draft continues from the last row that stayed
    if (s->mtp.on != 0) {
        mtp_seed(s, s->x.data() + (size_t) (kept - 1u) * s->E);
    }
    out->advance = keep_all_rows != 0 ? rows : kept;
    if (keep_all_rows != 0 || rows == 1u) {
        // the counterexample arm leaves the state past the accepted prefix, and a single-row
        // pass has nothing to undo: it is a plain serial step whose cursor already moved
        return 0;
    }
    return qrt_spec_rollback(s->layers.data(), s->n_layers, kept, err, err_size);
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
        const qrt_mtp_cfg mc = { opt->mtp != 0 ? 1 : 0, cfg->draft_vocab,
                                 opt->draft_vocab_path };
        // a speculative cycle needs one ring slot per row it may carry
        const uint32_t snap = opt->mtp != 0
                                 ? std::max(BK, (uint32_t) cfg->spec_k) : BK;
        if (gen_open(&s, cfg, opt, gguf_path, store_prefix, spv_dir, n_tokens, n_new, snap, mc,
                     err, err_size) != 0) {
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
        } else if (opt->mtp != 0) {
            // The speculative loop (design 5.4): the MTP block drafts k-1 tokens, one pass
            // verifies them beside the last accepted token, and the cycle emits the accepted
            // prefix plus the verified bonus.
            const uint32_t k = (uint32_t) cfg->spec_k;
            if (k < 2u || k > 8u) {
                fail(err, err_size, "--spec-mtp needs QRT_SPEC_K in 2..8, got %d", cfg->spec_k);
                break;
            }
            mtp_seed(&s, x.data() + (size_t) (n_tokens - 1u) * s.E);
            std::vector<uint32_t> draft;
            uint32_t cycles = 0, kept_total = 0, rows_total = 0, hist[9] = {0};
            while (generated.size() < n_new) {
                const uint32_t left = n_new - (uint32_t) generated.size();
                const uint32_t rows = std::min(k, left + 1u);
                std::vector<uint32_t> pass_rows;
                if (rows > 1u && mtp_draft(&s, generated.back(), pos, rows - 1u, &draft, err,
                                           err_size) != 0) {
                    break;
                }
                pass_rows.push_back(generated.back());
                for (uint32_t i = 0; i + 1u < rows; ++i) {
                    pass_rows.push_back(draft[i]);
                }
                qrt_cycle c;
                if (gen_cycle(&s, pos, pass_rows, 0, 0, &c, err, err_size) != 0) {
                    break;
                }
                for (uint32_t i = 0; i < c.emit.size() && generated.size() < n_new; ++i) {
                    generated.push_back(c.emit[i]);
                }
                pos += c.kept;
                ++cycles;
                kept_total += c.kept;
                rows_total += c.rows;
                if (c.accepted < 9u) {
                    ++hist[c.accepted];
                }
                printf("generate_cycle cycle=%u pos=%u rows=%u accepted=%u kept=%u gpu_ms=%.2f "
                       "h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u pass_ms=%.1f\n",
                       cycles - 1u, pos - c.kept, c.rows, c.accepted, c.kept, c.gpu_ms,
                       c.h2d_ms, c.cpu_ms, c.hits, c.miss, c.pass_ms);
                if (s.tel != NULL) {
                    qrt_pass_stats ps{};
                    ps.k = c.rows;
                    ps.pos = pos - c.kept;
                    ps.accepted = c.kept;
                    ps.hit_experts = c.hits;
                    ps.miss_experts = c.miss;
                    ps.miss_cpu = c.cpu_n;
                    ps.miss_h2d = c.h2d_n;
                    ps.h2d_bytes = c.h2d_bytes;
                    ps.draft_ms = s.mtp.steps ? s.mtp.step_ms / s.mtp.steps : 0.0;
                    ps.gpu_ms = c.gpu_ms;
                    ps.cpu_leg_ms = c.cpu_ms;
                    ps.pass_ms = c.pass_ms;
                    qrt_telemetry_pass(s.tel, cycles, &ps);
                }
            }
            if (err[0] != '\0') {
                break;
            }
            printf("generate_spec cycles=%u rows=%u E=%.3f accepted=%u/%u/%u/%u/%u "
                   "draft_steps=%u draft_ms=%.2f draft_mib=%.1f draft_gpu_ms=%.2f\n",
                   cycles, rows_total, cycles ? (double) kept_total / cycles : 0.0,
                   hist[0], hist[1], hist[2], hist[3], hist[4], s.mtp.steps,
                   s.mtp.steps ? s.mtp.step_ms / s.mtp.steps : 0.0,
                   s.mtp.steps ? (double) s.mtp.host_bytes / s.mtp.steps / (1024.0 * 1024.0)
                               : 0.0,
                   s.mtp.steps ? s.mtp.gpu_ms / s.mtp.steps : 0.0);
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
        const qrt_mtp_cfg mc_off = { 0, 0, nullptr };
        const qrt_mtp_cfg mc = { opt->mtp != 0 ? 1 : 0, cfg->draft_vocab,
                                opt->draft_vocab_path };
        if (gen_open(&s, cfg, opt, gguf_path, store_prefix, spv_dir, n_tokens, n_new, 0, mc_off,
                    err, err_size) != 0) {
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
                     rollback_off ? n_new * k : n_new, k, mc, err, err_size) != 0) {
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
        // the first cycle drafts from the prefill's last hidden, like --generate does
        if (s.mtp.on != 0) {
            mtp_seed(&s, x.data() + (size_t) (n_tokens - 1u) * E);
        }
        uint32_t cycles = 0, accepted_sum = 0, rows_checked = 0, rows_bad = 0, rows_loose = 0;
        double worst_row = 0.0;
        // The row gap can only gate exactness when both arms run the same arithmetic on every
        // expert. With the cache off that holds by construction. With the cache on it does not:
        // a pass that routes more unique experts than the layer has slots keeps its leftovers
        // on the CPU leg even at --miss-split 100 (512 MiB over 41 layers is 6 slots per layer
        // against ~20 experts per pass), and the CPU leg itself differs from the GPU kernel by
        // ~1e-2 in a logit (RESULTS.md#m3-2a-miss-split 87.2). Cached arms therefore gate the
        // per-row top-1 and report the float gap as a reading.
        const int exact_legs = !s.cache_on;
        uint32_t hist[9] = {0};
        while (spec_ids.size() < n_new) {
            const uint32_t have = (uint32_t) spec_ids.size();
            const uint32_t left = (uint32_t) serial_ids.size() - have;
            const uint32_t nd = left - 1u < k - 1u ? left - 1u : k - 1u;
            const uint32_t rows = nd + 1u;
            std::vector<uint32_t> draft(nd), pass_rows(rows);
            uint32_t break_at = 0;
            if (s.mtp.on != 0) {
                // Real drafts: the MTP block proposes the continuation and nothing is
                // injected, so the accept lengths are whatever the draft actually achieves.
                pass_rows[0] = spec_ids.back();
                if (nd != 0 && mtp_draft(&s, pass_rows[0], pos, nd, &draft, err, err_size) != 0) {
                    break;
                }
            } else {
                // The drafts are the serial arm's continuation, so the accept length a correct
                // cycle should reach is known. One position per cycle is broken, which walks
                // the accept counts 0..k-1 that design 7 asks for.
                for (uint32_t i = 0; i < nd; ++i) {
                    draft[i] = serial_ids[have + i];
                }
                break_at = cycles % k;
                if (nd != 0 && break_at < nd) {
                    draft[break_at] ^= 0x10000u;
                    if (draft[break_at] >= V) {
                        draft[break_at] = V - 1u;
                    }
                }
                pass_rows[0] = spec_ids.back();
            }
            for (uint32_t i = 0; i < nd; ++i) {
                pass_rows[i + 1u] = draft[i];
            }
            qrt_cycle c;
            if (gen_cycle(&s, pos, pass_rows, rollback_off ? 0 : 1, rollback_off, &c, err,
                          err_size) != 0) {
                break;
            }
            // Rows 0..kept-1 carry the true tokens, so their logits must be the serial arm's
            // for the same predicted index. A row past the break was fed a wrong draft token,
            // so it is expected to differ and is not compared.
            for (uint32_t i = 0; !rollback_off && i < c.kept; ++i) {
                const uint32_t r = pos + i + 1u - n_tokens;
                if (r >= serial_ids.size()) {
                    continue;
                }
                double max_abs = 0.0;
                const double rel = gen_row_rel_l1(c.logits.data() + (size_t) i * V,
                                                 serial_logits.data() + (size_t) r * V, V,
                                                 &max_abs);
                ++rows_checked;
                if (rel > worst_row) {
                    worst_row = rel;
                }
                // the per-row token is the gate in every configuration; the float gap only
                // gates where both arms share the arithmetic path
                if (c.verified[i] != serial_ids[r] || (exact_legs && !(rel <= 1e-4))) {
                    ++rows_bad;
                } else if (!(rel <= 1e-4)) {
                    ++rows_loose;
                }
                if (c.verified[i] != serial_ids[r] || !(rel <= 1e-4)) {
                    printf("spec_rows cycle=%u row=%u rel_l1=%.3e max_abs=%.3e id=%u want=%u "
                           "legs=%s\n", cycles, i, rel, max_abs, c.verified[i], serial_ids[r],
                           exact_legs ? "exact" : "cached");
                }
            }
            for (size_t i = 0; i < c.emit.size(); ++i) {
                spec_ids.push_back(c.emit[i]);
            }
            pos += c.advance;
            ++cycles;
            accepted_sum += c.kept;
            if (c.accepted < 9u) {
                ++hist[c.accepted];
            }
            printf("spec_cycle cycle=%u pos=%u rows=%u drafts=%u accepted=%u kept=%u "
                   "src=%s break_at=%u gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u "
                   "pass_ms=%.1f\n",
                   cycles - 1u, pos - c.advance, c.rows, nd, c.accepted, c.kept,
                   s.mtp.on != 0 ? "mtp" : "forced", break_at, c.gpu_ms, c.h2d_ms, c.cpu_ms,
                   c.hits, c.miss, c.pass_ms);
            if (s.tel != NULL) {
                qrt_pass_stats ps{};
                ps.k = c.rows;
                ps.pos = pos - c.advance;
                ps.accepted = c.kept;
                ps.hit_experts = c.hits;
                ps.miss_experts = c.miss;
                ps.miss_cpu = c.cpu_n;
                ps.miss_h2d = c.h2d_n;
                ps.h2d_bytes = c.h2d_bytes;
                ps.draft_ms = s.mtp.steps ? s.mtp.step_ms / s.mtp.steps : 0.0;
                ps.gpu_ms = c.gpu_ms;
                ps.cpu_leg_ms = c.cpu_ms;
                ps.pass_ms = c.pass_ms;
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
        // The draft's own account: what a step costs, and how often each prefix survived.
        // p_j = P(accept at least j drafts), the per-position rate design 6.3 reports.
        {
            std::string hstr, pstr;
            for (uint32_t j = 0; j < 8u; ++j) {
                char b[24];
                snprintf(b, sizeof(b), j ? "/%u" : "%u", hist[j]);
                hstr += b;
            }
            for (uint32_t j = 1; j <= k && j < 9u; ++j) {
                uint32_t reached = 0;
                for (uint32_t q = j; q < 9u; ++q) {
                    reached += hist[q];
                }
                char b[24];
                snprintf(b, sizeof(b), j == 1u ? "%.3f" : "/%.3f",
                         cycles ? (double) reached / (double) cycles : 0.0);
                pstr += b;
            }
            printf("spec_draft source=%s steps=%u ms_per_step=%.2f mib_per_step=%.1f "
                   "gpu_ms_per_step=%.2f accepted=%s p=%s\n",
                   s.mtp.on != 0 ? "mtp" : "forced", s.mtp.steps,
                   s.mtp.steps ? s.mtp.step_ms / s.mtp.steps : 0.0,
                   s.mtp.steps ? (double) s.mtp.host_bytes / s.mtp.steps / (1024.0 * 1024.0)
                               : 0.0,
                   s.mtp.steps ? s.mtp.gpu_ms / s.mtp.steps : 0.0, hstr.c_str(), pstr.c_str());
        }
        printf("spec_check tokens=%u k=%u cycles=%u E=%.3f serial=%zu spec=%zu ids_equal=%d "
               "first_diff=%u rows_checked=%u rows_bad=%u rows_loose=%u worst_rel_l1=%.3e "
               "legs=%s break_cases=%u rollback=%s verdict=%s\n",
               n_new, k, cycles, cycles ? (double) accepted_sum / cycles : 0.0,
               serial_ids.size(), spec_ids.size(), ids_match, first_diff, rows_checked,
               rows_bad, rows_loose, worst_row, exact_legs ? "exact" : "cached", k,
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
