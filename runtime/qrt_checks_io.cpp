#include "qrt_checks.h"
#include "vk_transfer.h"

#include "qrt_check_support.h"
#include "qrt_config.h"
#include "qrt_util.h"
#include "expert_cache.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

// M2-3 dense residency: streams every non-expert tensor into one device buffer and
// compares a sample back against the file. Reports bytes, rate and the check result.
int qrt_check_dense_residency(const qrt_config * cfg, const char * gguf_path) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    const uint64_t staging_bytes = 8ull << 20;
    const double t0 = now_ms();
    qrt_dense * d = NULL;
    if (qrt_dense_open(dev, gguf_path, staging_bytes, &d, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const double t1 = now_ms();
    const uint64_t bytes = qrt_dense_bytes(d);
    printf("dense_upload tensors=%u bytes=%llu mib=%.1f staging_kib=%llu ms=%.1f gbs=%.2f\n",
           qrt_dense_tensor_count(d), (unsigned long long) bytes, (double) bytes / (1024.0 * 1024.0),
           (unsigned long long) (staging_bytes >> 10), t1 - t0,
           (t1 - t0) > 0.0 ? (double) bytes / (t1 - t0) / 1e6 : 0.0);

    qrt_vk_staging * stg = NULL;
    if (qrt_vk_staging_create(dev, staging_bytes, &stg, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_dense_close(dev, d);
        qrt_vk_close(dev);
        return 2;
    }
    const double t2 = now_ms();
    const uint64_t checked = qrt_dense_verify(dev, d, stg, 64ull << 20, err, sizeof(err));
    const double t3 = now_ms();
    const bool ok = checked > 0 && err[0] == '\0';
    printf("dense_verify compared_bytes=%llu ms=%.1f match=%d %s%s\n",
           (unsigned long long) checked, t3 - t2, ok ? 1 : 0, ok ? "" : "err=",
           ok ? "" : err);
    qrt_vk_staging_destroy(dev, stg);
    printf("dense_check io_failures=%u alloc_mib=%llu\n", dev->io_failures,
           (unsigned long long) (dev->alloc_bytes >> 20));
    qrt_dense_close(dev, d);
    qrt_vk_close(dev);
    return ok ? 0 : 1;
}

// M2-3 VRAM side: proves the staging round trip works before dense residency and
// prefill come near it. With a repack prefix it also round trips real expert
// blocks, which is the exact case that faulted in DECISIONS 29.
int qrt_check_vk_io(const qrt_config * cfg, const char * prefix) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }

    const char * sizes_name[] = { "4k", "64k", "1.9m", "bulk64m" };
    const uint64_t sizes[] = { 4096ull, 65536ull, 1916796ull, 64ull * 1024ull * 1024ull };
    int bad = 0;
    for (int s = 0; s < 4; ++s) {
        const uint64_t bytes = sizes[s];
        qrt_vk_staging * stg = NULL;
        if (qrt_vk_staging_create(dev, bytes, &stg, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_vk_close(dev);
            return 2;
        }
        VkBuffer buf = 0;
        VkDeviceMemory mem = 0;
        if (qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST |
                                   QRT_VK_USAGE_TRANSFER_SRC, &buf, &mem, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_vk_staging_destroy(dev, stg);
            qrt_vk_close(dev);
            return 2;
        }
        std::vector<unsigned char> src((size_t) bytes);
        std::vector<unsigned char> back((size_t) bytes, 0);
        qrt_check_pattern_fill(src.data(), src.size(), (uint32_t) s * 7919u);
        unsigned char * map = (unsigned char *) qrt_vk_staging_map(stg);

        // One warm-up pair, then the samples. The first submit on a fresh queue
        // carries driver setup, which is not part of the steady-state cost.
        int up = 0, down = 0;
        double up_ms[7];
        double down_ms[7];
        double fill_ms[7];
        double unmap_ms[7];
        const int reps = 7;
        int same = 0;
        for (int it = -1; it < reps; ++it) {
            const double f0 = now_ms();
            memcpy(map, src.data(), src.size());
            const double f1 = now_ms();
            up = qrt_vk_upload_mapped(dev, stg, buf, 0, bytes, err, sizeof(err));
            const double t1 = now_ms();
            down = up == 0 ? qrt_vk_download_mapped(dev, stg, buf, 0, bytes,
                                                    err, sizeof(err)) : -1;
            const double t2 = now_ms();
            memcpy(back.data(), map, back.size());
            const double t3 = now_ms();
            if (up != 0 || down != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            if (it < 0) {
                same = qrt_check_pattern_compare(src.data(), back.data(), src.size()) == 0;
                continue;
            }
            fill_ms[it] = f1 - f0;
            up_ms[it]   = t1 - f1;
            down_ms[it] = t2 - t1;
            unmap_ms[it] = t3 - t2;
        }
        double up_min = up_ms[0], down_min = down_ms[0];
        for (int i = 1; i < reps; ++i) {
            if (up_ms[i] < up_min) up_min = up_ms[i];
            if (down_ms[i] < down_min) down_min = down_ms[i];
        }
        const double fill_med = qrt_check_median(fill_ms, reps);
        const double up_med = qrt_check_median(up_ms, reps);
        const double down_med = qrt_check_median(down_ms, reps);
        const double unmap_med = qrt_check_median(unmap_ms, reps);
        printf("roundtrip size=%s bytes=%llu fill_ms=%.3f copy_up_ms=%.3f (min %.3f) "
               "copy_down_ms=%.3f copy_out_ms=%.3f h2d_gbs=%.2f h2d_min_gbs=%.2f d2h_gbs=%.2f "
               "e2e_h2d_gbs=%.2f match=%d reps=%d %s\n",
               sizes_name[s], (unsigned long long) bytes, fill_med, up_med, up_min, down_med,
               unmap_med,
               up_med > 0.0 ? (double) bytes / up_med / 1e6 : 0.0,
               up_min > 0.0 ? (double) bytes / up_min / 1e6 : 0.0,
               down_med > 0.0 ? (double) bytes / down_med / 1e6 : 0.0,
               (fill_med + up_med) > 0.0 ? (double) bytes / (fill_med + up_med) / 1e6 : 0.0,
               same, reps, qrt_vk_staging_label(stg));
        if (up != 0 || down != 0 || !same) {
            ++bad;
        }
        qrt_vk_free(dev, buf, mem);
        qrt_vk_staging_destroy(dev, stg);
    }

    // Pipelined upload: K staging slots in flight is the M2-5 shape (double
    // buffering now, wider later). The serial rate above is submit+fence bound,
    // so this is the number the engine can actually reach.
    {
        const int K = 8;
        const uint64_t bytes = 1916796ull;
        VkBuffer pbuf = 0;
        VkDeviceMemory pmem = 0;
        if (qrt_vk_alloc(dev, bytes * K, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         &pbuf, &pmem, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_vk_close(dev);
            return 2;
        }
        qrt_vk_staging * slot[8];
        memset(slot, 0, sizeof(slot));
        int ok = 1;
        for (int k = 0; k < K && ok; ++k) {
            if (qrt_vk_staging_create(dev, bytes, &slot[k], err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                ok = 0;
                break;
            }
            unsigned char * m = (unsigned char *) qrt_vk_staging_map(slot[k]);
            qrt_check_pattern_fill(m, (size_t) bytes, (uint32_t) k);
        }
        double best = 0.0;
        const int rounds = 5;
        double samples[5];
        for (int r = 0; ok && r < rounds; ++r) {
            const double t0 = now_ms();
            for (int k = 0; k < K && ok; ++k) {
                if (qrt_vk_upload_submit(dev, slot[k], pbuf, bytes * (uint64_t) k, bytes,
                                         err, sizeof(err)) != 0) {
                    fprintf(stderr, "qrt: %s\n", err);
                    ok = 0;
                }
            }
            for (int k = 0; k < K && ok; ++k) {
                if (qrt_vk_staging_wait(dev, slot[k], err, sizeof(err)) != 0) {
                    fprintf(stderr, "qrt: %s\n", err);
                    ok = 0;
                }
            }
            const double t1 = now_ms();
            samples[r] = t1 - t0;
            const double gbs = (double) bytes * K / (t1 - t0) / 1e6;
            if (gbs > best) {
                best = gbs;
            }
        }
        if (ok) {
            const double med = qrt_check_median(samples, rounds);
            printf("pipelined_upload slots=%d bytes_each=%llu total_bytes=%llu median_ms=%.3f "
                   "gbs=%.2f best_gbs=%.2f %s\n", K, (unsigned long long) bytes,
                   (unsigned long long) (bytes * K), med,
                   med > 0.0 ? (double) bytes * K / med / 1e6 : 0.0, best,
                   qrt_vk_staging_label(slot[0]));
        } else {
            ++bad;
        }
        for (int k = 0; k < K; ++k) {
            qrt_vk_staging_destroy(dev, slot[k]);
        }
        qrt_vk_free(dev, pbuf, pmem);
    }

    if (prefix != NULL) {
        qrt_weights * w = NULL;
        if (qrt_weights_open(prefix, &w, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_vk_close(dev);
            return 2;
        }
        const uint32_t layers = qrt_weights_layers(w);
        const uint32_t experts = qrt_weights_experts(w);
        qrt_vk_staging * stg = NULL;
        // The slot must hold the largest expert, not the mean (RESULTS 35).
        const uint64_t slot = qrt_weights_max_expert_bytes(w);
        if (qrt_vk_staging_create(dev, slot, &stg, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_weights_close(w);
            qrt_vk_close(dev);
            return 2;
        }
        VkBuffer buf = 0;
        VkDeviceMemory mem = 0;
        if (qrt_vk_alloc(dev, slot, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST |
                                  QRT_VK_USAGE_TRANSFER_SRC, &buf, &mem, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_vk_staging_destroy(dev, stg);
            qrt_weights_close(w);
            qrt_vk_close(dev);
            return 2;
        }
        std::vector<unsigned char> src((size_t) slot);
        std::vector<unsigned char> back((size_t) slot, 0);
        unsigned char * map = (unsigned char *) qrt_vk_staging_map(stg);
        std::vector<double> up_ms, down_ms;
        uint64_t total = 0;
        int checked = 0;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            // Three experts per layer: the first, the middle and the last.
            const uint32_t picks[] = { 0, experts / 2, experts - 1 };
            for (int p = 0; p < 3; ++p) {
                qrt_expert_loc loc;
                if (qrt_weights_locate(w, layer, picks[p], &loc, err, sizeof(err)) != 0) {
                    fprintf(stderr, "qrt: %s\n", err);
                    ++bad;
                    continue;
                }
                if (loc.bytes > slot) {
                    printf("skip layer=%u expert=%u bytes=%llu above the %llu slot\n", layer,
                           picks[p], (unsigned long long) loc.bytes, (unsigned long long) slot);
                    continue;
                }
                // The engine reads the expert straight into the staging mapping:
                // one file read plus one DMA, no intermediate host buffer.
                if (qrt_weights_read_expert(w, layer, picks[p], map, (size_t) slot,
                                            err, sizeof(err)) != 0) {
                    fprintf(stderr, "qrt: %s\n", err);
                    ++bad;
                    continue;
                }
                memcpy(src.data(), map, (size_t) loc.bytes);
                const double a0 = now_ms();
                const int up = qrt_vk_upload_mapped(dev, stg, buf, 0, loc.bytes, err, sizeof(err));
                const double a1 = now_ms();
                const int down = up == 0 ? qrt_vk_download_mapped(dev, stg, buf, 0, loc.bytes,
                                                                  err, sizeof(err)) : -1;
                const double a2 = now_ms();
                if (up != 0 || down != 0) {
                    fprintf(stderr, "qrt: layer %u expert %u: %s\n", layer, picks[p], err);
                    ++bad;
                    continue;
                }
                memcpy(back.data(), map, (size_t) loc.bytes);
                if (qrt_check_pattern_compare(src.data(), back.data(), (size_t) loc.bytes) != 0) {
                    printf("  layer %u expert %u does not round trip\n", layer, picks[p]);
                    ++bad;
                    continue;
                }
                ++checked;
                total += loc.bytes;
                up_ms.push_back(a1 - a0);
                down_ms.push_back(a2 - a1);
            }
        }
        const double up_med = qrt_check_median(up_ms.empty() ? NULL : &up_ms[0], (int) up_ms.size());
        const double down_med = qrt_check_median(down_ms.empty() ? NULL : &down_ms[0], (int) down_ms.size());
        double sum_up = 0.0, sum_down = 0.0;
        for (size_t i = 0; i < up_ms.size(); ++i) {
            sum_up += up_ms[i];
            sum_down += down_ms[i];
        }
        printf("expert_roundtrip experts=%d of %u x %u bytes=%llu slot=%llu size_min=%llu size_max=%llu "
               "upload_ms=%.3f download_ms=%.3f "
               "sum_upload_ms=%.3f sum_download_ms=%.3f h2d_gbs=%.2f d2h_gbs=%.2f %s\n",
               checked, layers, experts, (unsigned long long) total, (unsigned long long) slot,
               (unsigned long long) qrt_weights_min_expert_bytes(w),
               (unsigned long long) qrt_weights_max_expert_bytes(w), up_med, down_med,
               sum_up, sum_down,
               sum_up > 0.0 ? (double) total / sum_up / 1e6 : 0.0,
               sum_down > 0.0 ? (double) total / sum_down / 1e6 : 0.0,
               qrt_vk_staging_label(stg));
        qrt_vk_free(dev, buf, mem);
        qrt_vk_staging_destroy(dev, stg);
        qrt_weights_close(w);
    }

    printf("io_check failures=%d io_failures=%u\n", bad, dev->io_failures);
    qrt_vk_close(dev);
    return bad == 0 ? 0 : 1;
}

int qrt_check_cache(const qrt_config * cfg, const char * prefix, const char * trace_path,
                           uint32_t decode_tokens, uint32_t protected_percent) {
    char err[QRT_ERR_SIZE] = {};
    qrt_weights * w = NULL;
    if (qrt_weights_open(prefix, &w, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    const uint32_t layers = qrt_weights_layers(w);
    const uint32_t experts = qrt_weights_experts(w);
    if (layers == 0 || layers > QRT_CACHE_MAX_LAYERS || experts == 0 ||
        experts > QRT_CACHE_MAX_EXPERTS) {
        fprintf(stderr, "qrt: store has %u layers x %u experts, the cache supports %u x %u\n",
                layers, experts, QRT_CACHE_MAX_LAYERS, QRT_CACHE_MAX_EXPERTS);
        qrt_weights_close(w);
        return 2;
    }
    std::vector<uint64_t> slot_bytes(layers, 0), weights(layers, 1);
    uint32_t mixed_layers = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        uint64_t first = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            qrt_expert_loc loc;
            if (qrt_weights_locate(w, l, e, &loc, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_weights_close(w);
                return 2;
            }
            if (loc.bytes == 0) {
                fprintf(stderr, "qrt: layer %u expert %u has zero bytes\n", l, e);
                qrt_weights_close(w);
                return 2;
            }
            if (e == 0) {
                first = loc.bytes;
            } else if (loc.bytes != first) {
                ++mixed_layers;
            }
            if (loc.bytes > slot_bytes[l]) {
                slot_bytes[l] = loc.bytes;
            }
        }
    }
    qrt_weights_close(w);

    qrt_cache_trace * trace = NULL;
    uint32_t observed = 0;
    const char * weight_source = "uniform";
    if (trace_path != NULL) {
        if (qrt_cache_trace_open(trace_path, experts, &trace, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        std::fill(weights.begin(), weights.end(), 0);
        if (qrt_cache_trace_weights(trace, decode_tokens, layers, weights.data(), &observed, err,
                                    sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_cache_trace_close(trace);
            return 2;
        }
        weight_source = "trace";
    }

    const uint64_t budget = (uint64_t) cfg->expert_cache_mib << 20;
    qrt_cache_plan plan{};
    if (qrt_cache_plan_build(layers, experts, slot_bytes.data(), weights.data(), budget,
                             cfg->cache_policy, protected_percent, &plan, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_cache_trace_close(trace);
        return 2;
    }
    printf("cache_plan layers=%u experts=%u policy=%s budget_mib=%d bytes=%llu bytes_mib=%.1f "
           "slots=%llu degraded=%u protected_pct=%u mixed_layers=%u weights=%s\n",
           layers, experts, qrt_cache_policy_name(cfg->cache_policy), cfg->expert_cache_mib,
           (unsigned long long) plan.bytes, (double) plan.bytes / (1024.0 * 1024.0),
           (unsigned long long) plan.slots, plan.degraded_layers, protected_percent,
           mixed_layers, weight_source);
    for (uint32_t l = 0; l < layers; ++l) {
        printf("cache_layer layer=%u slot_bytes=%llu slots=%u offset=%llu\n", l,
               (unsigned long long) plan.slot_bytes[l], plan.slots_per_layer[l],
               (unsigned long long) plan.layer_offset[l]);
    }
    if (trace != NULL) {
        qrt_cache_stats st{};
        uint32_t simulated_layers = 0;
        uint32_t decode_rows = 0;
        if (qrt_cache_simulate(&plan, trace, decode_tokens, &st, &simulated_layers, &decode_rows,
                               err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_cache_trace_close(trace);
            return 2;
        }
        const double hit_pct = st.lookups != 0 ? 100.0 * (double) st.hits / (double) st.lookups : 0.0;
        printf("cache_trace path=%s tokens=%u layers=%u top_k=%u decode=%u observed_layers=%u "
               "decode_rows=%u\n", trace_path, qrt_cache_trace_tokens(trace),
               qrt_cache_trace_layers(trace), qrt_cache_trace_top_k(trace), decode_tokens,
               simulated_layers, decode_rows);
        printf("cache_stats policy=%s lookups=%llu hits=%llu misses=%llu hit_pct=%.3f "
               "evictions=%llu promotions=%llu demotions=%llu miss_mib=%.1f "
               "h2d_uploads=%llu h2d_mib=%.1f upload_failures=%llu wait_failures=%llu "
               "no_slot=%llu pending=%u mode=policy_only\n",
               qrt_cache_policy_name(cfg->cache_policy), (unsigned long long) st.lookups,
               (unsigned long long) st.hits, (unsigned long long) st.misses, hit_pct,
               (unsigned long long) st.evictions, (unsigned long long) st.promotions,
               (unsigned long long) st.demotions,
               (double) st.miss_bytes / (1024.0 * 1024.0),
               (unsigned long long) st.h2d_uploads,
               (double) st.h2d_bytes / (1024.0 * 1024.0),
               (unsigned long long) st.upload_failures, (unsigned long long) st.wait_failures,
               (unsigned long long) st.no_slot_misses, st.pending);
        for (uint32_t l = 0; l < layers; ++l) {
            printf("cache_layer_stats layer=%u hits=%llu misses=%llu evictions=%llu\n", l,
                   (unsigned long long) st.layer_hits[l],
                   (unsigned long long) st.layer_misses[l],
                   (unsigned long long) st.layer_evictions[l]);
        }
        qrt_cache_trace_close(trace);
    }
    return 0;
}
