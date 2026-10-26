// M2-5 step 1: layer-first prefill with uploads in flight. Runs the same layer set twice,
// serial (1 slot) and pipelined (slots), so the overlap is a single variable. Timings
// only: this command never validates an output.

#include "qrt_checks.h"

#include "qrt_config.h"
#include "qrt_util.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "kernels/moe_expert.h"
#include "prefill.h"
#include "weights.h"

#include <stdio.h>
#include <string.h>
#include <vector>

int qrt_check_prefill(const qrt_config * cfg, const char * prefix, const char * spv,
                      uint32_t slots, uint32_t layers, int use_moe, int layer_batch,
                      int prefill_pipeline, int prefill_group_kib,
                      int prefill_readers, int prefill_chunks) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_weights * w = NULL;
    if (qrt_weights_open(prefix, &w, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t all_layers = qrt_weights_layers(w);
    const uint32_t experts = qrt_weights_experts(w);
    if (layers == 0 || layers > all_layers) {
        layers = all_layers;
    }
    const uint32_t unique = qrt_prefill_unique_experts(cfg->prefill_chunk, (uint32_t) 8, experts);
    std::vector<uint32_t> ids((size_t) all_layers * experts);
    std::vector<uint32_t> counts(all_layers, 0);
    for (uint32_t l = 0; l < all_layers; ++l) {
        counts[l] = l < layers ? unique : 0;
        for (uint32_t e = 0; e < unique; ++e) {
            // Spread the picked ids over the whole table instead of a prefix.
            ids[(size_t) l * experts + e] = (uint32_t) ((uint64_t) e * experts / unique);
        }
    }
    char reader[8] = {0};
    qrt_weights_reader_name(w, reader, sizeof(reader));
    printf("prefill_check chunk=%u layers=%u experts=%u unique_per_layer=%u top_k=8 "
           "max_expert_bytes=%llu reader=%s\n", cfg->prefill_chunk, layers, experts, unique,
           (unsigned long long) qrt_weights_max_expert_bytes(w), reader);

    const uint32_t arms[] = { 1, slots };
    double arm_ms[2] = { 0.0, 0.0 };
    int rc = 0;
    // Warm-up: the first pass over a mapping pays the page faults of the store, and
    // whichever arm runs first would otherwise look slow for that reason alone.
    {
        qrt_prefill_cfg wcfg;
        memset(&wcfg, 0, sizeof(wcfg));
        wcfg.chunk = cfg->prefill_chunk;
        wcfg.slots = slots;
        qrt_prefill * warm = NULL;
        if (qrt_prefill_open(dev, w, &wcfg, &warm, err, sizeof(err)) == 0) {
            qrt_prefill_stats wst;
            qrt_prefill_backend upload_only{};
            upload_only.compute = [](void *, uint32_t, uint32_t, VkBuffer, uint64_t,
                                      uint64_t, char *, size_t) { return 0; };
            if (qrt_prefill_run(warm, ids.data(), experts, counts.data(), &upload_only,
                                &wst, err, sizeof(err)) == 0) {
                printf("prefill_warmup slots=%u blocks=%u bytes=%llu ms=%.1f per_block_ms=%.3f\n",
                       wcfg.slots, wst.blocks_done, (unsigned long long) wst.bytes,
                       wst.total_ms, wst.total_ms / (wst.blocks_done > 0 ? wst.blocks_done : 1));
            }
            qrt_prefill_close(dev, warm);
        }
    }
    // The compute leg is either the read probe (scheduling only) or the real int8 GEMM,
    // so the same scheduler can be compared before and after the math lands.
    qrt_moe * moe = NULL;
    if (use_moe) {
        if (!dev->has_dot_product) {
            fprintf(stderr, "qrt: this device has no shaderIntegerDotProduct; "
                            "run with --prefill-compute probe\n");
            qrt_weights_close(w);
            qrt_vk_close(dev);
            return 2;
        }
        qrt_moe_cfg mcfg;
        memset(&mcfg, 0, sizeof(mcfg));
        // Layer major over `chunks` chunks: the layer is uploaded once and every chunk's
        // tokens go through it, so an expert sees chunks x the tokens of one chunk.
        const uint32_t per_expert = (cfg->prefill_chunk * 8u * (uint32_t) prefill_chunks) / experts;
        mcfg.tokens = per_expert < 32 ? 32 : (per_expert + 31u) / 32u * 32u;
        mcfg.hidden = 2048;
        mcfg.ffn    = 512;
        if (qrt_moe_open(dev, w, spv, &mcfg, &moe, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_weights_close(w);
            qrt_vk_close(dev);
            return 2;
        }
        printf("prefill_compute moe tokens_per_expert=%u hidden=%u ffn=%u dot_product=%d\n",
               mcfg.tokens, mcfg.hidden, mcfg.ffn, dev->has_dot_product);
    } else {
        printf("prefill_compute probe (read kernel, no math)\n");
    }

    for (int a = 0; a < 2 && rc == 0; ++a) {
        qrt_prefill_cfg pcfg;
        memset(&pcfg, 0, sizeof(pcfg));
        pcfg.chunk = cfg->prefill_chunk;
        pcfg.slots = arms[a];
        pcfg.slot_bytes = 0;
        pcfg.pipeline = (uint32_t) prefill_pipeline;
        pcfg.group_bytes = (uint32_t) prefill_group_kib * 1024u;
        pcfg.readers = (uint32_t) prefill_readers;
        qrt_prefill * p = NULL;
        if (qrt_prefill_open(dev, w, &pcfg, &p, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
            break;
        }
        qrt_prefill_probe * probe = NULL;
        if (!use_moe && qrt_prefill_probe_open(dev, spv, p, &probe, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_prefill_close(dev, p);
            rc = 2;
            break;
        }
        const qrt_prefill_backend backend = use_moe
            ? qrt_prefill_backend{qrt_moe_compute, qrt_moe_compute_batch, qrt_moe_wait, moe}
            : qrt_prefill_backend{qrt_prefill_probe_compute, nullptr, nullptr, probe};
        const uint32_t skipped_before = use_moe && moe != NULL ? qrt_moe_down_skipped(moe) : 0;
        qrt_prefill_stats st;
        const int rc_run = layer_batch
                           ? qrt_prefill_run_layers(p, ids.data(), experts, counts.data(), &backend,
                                                    &st, err, sizeof(err))
                           : qrt_prefill_run(p, ids.data(), experts, counts.data(), &backend,
                                             &st, err, sizeof(err));
        if (rc_run != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
        } else {
            const double blocks = st.blocks_done > 0 ? (double) st.blocks_done : 1.0;
            arm_ms[a] = st.total_ms;
            const double flops = use_moe && moe != NULL
                                 ? (double) qrt_moe_flops(moe) * st.blocks_done : 0.0;
            printf("prefill mode=%s pipe=%d slots=%u chunk=%u layers=%u of %u blocks=%u done=%u failed=%u "
                   "bytes=%llu slot_mib=%.2f total_ms=%.1f per_block_ms=%.3f "
                   "read_ms=%.1f upload_ms=%.1f wait_ms=%.1f compute_ms=%.1f "
                   "readers=%d groups=%u gflop=%.1f tflops_wall=%.2f tflops_rec=%.2f down_skipped=%u "
                   "reductions=%u\n",
                   layer_batch ? "layer" : "perexpert", prefill_pipeline, pcfg.slots,
                   pcfg.chunk, (layers),
                   st.layers, st.blocks, st.blocks_done,
                   st.blocks_failed, (unsigned long long) st.bytes,
                   (double) (layer_batch ? 0 : qrt_prefill_slot_bytes(p)) / (1024.0 * 1024.0),
                   st.total_ms,
                   st.total_ms / blocks, st.read_ms, st.upload_ms, st.wait_ms, st.compute_ms,
                   prefill_readers, st.groups, flops / 1e9,
                   st.total_ms > 0.0 ? flops / st.total_ms / 1e9 : 0.0,
                   st.compute_ms > 0.0 ? flops / st.compute_ms / 1e9 : 0.0,
                   use_moe && moe != NULL ? qrt_moe_down_skipped(moe) - skipped_before : 0,
                   st.slot_reductions);
        }
        qrt_prefill_probe_close(dev, probe);
        qrt_prefill_close(dev, p);
    }
    if (rc == 0 && arm_ms[0] > 0.0) {
        printf("prefill_speedup serial_ms=%.1f pipelined_ms=%.1f ratio=%.3f "
               "hidden_ms=%.1f\n", arm_ms[0], arm_ms[1], arm_ms[0] / arm_ms[1],
               arm_ms[0] - arm_ms[1]);
        const double tokens = (double) cfg->prefill_chunk * (double) prefill_chunks;
        printf("prefill_prompt chunks=%d tokens=%.0f layers=%u pipelined_ms=%.1f "
               "tokens_per_s=%.0f\n", prefill_chunks, tokens, layers, arm_ms[1],
               arm_ms[1] > 0.0 ? tokens / (arm_ms[1] / 1000.0) : 0.0);
    }
    qrt_moe_close(dev, moe);
    qrt_weights_close(w);
    printf("prefill_io_failures=%u alloc_mib=%llu\n", dev->io_failures,
           (unsigned long long) (dev->alloc_bytes >> 20));
    qrt_vk_close(dev);
    return rc;
}
