// The decode expert store for one layer (M3-2a): resolve the routed union against the
// cache, split the misses between the H2D and the CPU leg, and pack the winners into the
// layer's slot. This is the part of the FFN that touches RAM, PCIe and the cache policy,
// so it is kept apart from the command recording in ffn.cpp.

#include "ffn_internal.h"

#include "qrt_util.h"

#include "cpu_moe.h"
#include "expert_cache.h"
#include "kernels/gemm.h"
#include "miss_split.h"
#include "vk_transfer.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <thread>
#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

float half_bits_to_float(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((uint32_t) (113 - e) << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

int expert_type_of(uint64_t bytes, uint32_t rows, uint32_t cols) {
    const uint64_t vals = (uint64_t) rows * cols;
    if (vals % 256u != 0) {
        return -1;
    }
    const uint64_t q4 = vals / 256u * 144u;
    const uint64_t q5 = vals / 256u * 176u;
    const uint64_t q6 = vals / 256u * 210u;
    if (bytes == q4) {
        return QRT_CPU_Q4K;
    }
    if (bytes == q5) {
        return QRT_CPU_Q5K;
    }
    if (bytes == q6) {
        return QRT_CPU_Q6K;
    }
    return -1;
}

} // namespace

// One layer of the decode cache (M3-2a): unique experts -> residency probe -> split ->
// slot assignment -> CPU leg and H2D leg, which is measured while the other runs.
int qrt_ffn_plan_cache_layer(qrt_ffn * f, char * err, size_t err_size) {
    const uint32_t NE = f->cfg.n_expert;
    const uint32_t T = f->run_tokens;
    const uint32_t E = f->cfg.n_embd;
    const uint32_t K = f->cfg.top_k;
    f->legs = {};
    f->uniq.clear();
    f->use_gpu.assign(NE, 1);
    uint64_t layer_bytes = 0;
    if (qrt_weights_layer_expert_bytes(f->store, f->layer, &layer_bytes) != 0) {
        return fail(err, err_size, "layer %u has no expert size", f->layer);
    }
    std::vector<qrt_split_miss> miss;
    std::vector<uint32_t> miss_uniq;
    for (uint32_t e = 0; e < NE; ++e) {
        if (f->cnt[e] == 0) {
            continue;
        }
        const uint32_t u = (uint32_t) f->uniq.size();
        f->uniq.push_back(e);
        if (!qrt_cache_resident(f->cache, f->layer, e)) {
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0) {
                return -1;
            }
            qrt_split_miss m{};
            m.expert = e;
            m.bytes = loc.bytes;
            m.tokens = f->cnt[e];
            miss.push_back(m);
            miss_uniq.push_back(u);
        }
    }
    f->h2d_mask.assign(f->uniq.size(), 1);
    std::vector<uint8_t> mm(miss.size(), 1);
    qrt_split_stats ss{};
    if (qrt_miss_split_plan(miss.data(), (uint32_t) miss.size(), f->miss_split, &f->rates,
                            f->carry, mm.data(), &ss, err, err_size) != 0) {
        return -1;
    }
    for (size_t i = 0; i < miss.size(); ++i) {
        f->h2d_mask[miss_uniq[i]] = mm[i];
    }
    f->lookups.assign(f->uniq.size(), qrt_cache_lookup{});
    if (qrt_cache_acquire_mask(f->cache, f->layer, f->uniq.data(),
                               (uint32_t) f->uniq.size(), f->h2d_mask.data(),
                               f->lookups.data(), err, err_size) != 0) {
        f->legs.misses = (uint32_t) miss.size();
        return -1;
    }
    f->cpu_pos.clear();
    for (uint32_t i = 0; i < f->uniq.size(); ++i) {
        const uint32_t e = f->uniq[i];
        const qrt_cache_lookup & r = f->lookups[i];
        if (r.hit) {
            f->dev_off[e] = (uint32_t) ((uint64_t) r.slot * f->cache_slot_bytes);
            ++f->legs.hits;
        } else if (!r.failed && r.slot >= 0) {
            f->dev_off[e] = (uint32_t) ((uint64_t) r.slot * f->cache_slot_bytes);
            ++f->legs.h2d;
        } else {
            f->use_gpu[e] = 0;
            f->cnt[e] = 0;
            f->cpu_pos.push_back(i);
            ++f->legs.cpu;
        }
    }
    f->legs.misses = f->legs.h2d + f->legs.cpu;
    if (f->uniq.empty()) {
        return 0;
    }

    // The CPU leg reads the same Q8_1 planes the device produced for the layer row.
    // ds packs the f16 delta in the low half and sum(q)+32768 in the high half.
    const uint32_t nb = E / 32u;
    f->cpu_out.clear();
    f->cpu_res.clear();
    if (!f->cpu_pos.empty()) {
        f->cpu_out.assign((size_t) T * E, 0.0f);
        f->cpu_res.assign((size_t) f->cpu_pos.size() * E, 0.0f);
    }
    std::vector<std::thread> workers;
    std::vector<double> cpu_worker_ms;
    std::atomic<int> cpu_failed{0};
    if (!f->cpu_pos.empty()) {
        f->h_x_qs.resize((size_t) T * E);
        f->h_x_ds.resize((size_t) T * nb);
        f->h_x_d.resize((size_t) T * nb);
        f->h_x_s.resize((size_t) T * nb);
        if (qrt_vk_download(f->dev, f->stg, f->in.x_qs, 0, f->h_x_qs.data(),
                            (uint64_t) T * E, err, err_size) != 0 ||
            qrt_vk_download(f->dev, f->stg, f->in.x_ds, 0, f->h_x_ds.data(),
                            (uint64_t) T * nb * 4u, err, err_size) != 0) {
            return -1;
        }
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t word = f->h_x_ds[(size_t) t * nb + b];
                const float d = half_bits_to_float(word & 0xffffu);
                const int32_t sum = (int32_t) (word >> 16) - 32768;
                f->h_x_d[(size_t) t * nb + b] = d;
                f->h_x_s[(size_t) t * nb + b] = d * (float) sum;
            }
        }
        std::atomic<uint32_t> next{0};
        uint32_t nthreads = (uint32_t) f->cpu_pos.size();
        if (nthreads > 8) {
            nthreads = 8;
        }
        cpu_worker_ms.assign(nthreads, 0.0);
        for (uint32_t w = 0; w < nthreads; ++w) {
            workers.emplace_back([&, w]() {
                const double t_thread = now_ms();
                char tls[256] = {0};
                std::vector<unsigned char> buf((size_t) layer_bytes);
                std::vector<float> row((size_t) E);
                for (;;) {
                    const uint32_t j = next.fetch_add(1);
                    if (j >= f->cpu_pos.size()) {
                        break;
                    }
                    const uint32_t e = f->uniq[f->cpu_pos[j]];
                    qrt_expert_loc loc;
                    if (qrt_weights_locate(f->store, f->layer, e, &loc, tls, sizeof(tls)) != 0 ||
                        qrt_weights_read_expert(f->store, f->layer, e, buf.data(),
                                                buf.size(), tls, sizeof(tls)) != 0) {
                        cpu_failed = 1;
                        break;
                    }
                    const int tg = expert_type_of(loc.gate, f->cfg.ffn, E);
                    const int tu = expert_type_of(loc.up, f->cfg.ffn, E);
                    const int td = expert_type_of(loc.down, E, f->cfg.ffn);
                    if (tg < 0 || tu < 0 || td < 0) {
                        cpu_failed = 1;
                        break;
                    }
                    float * out = f->cpu_res.data() + (size_t) j * E;
                    for (uint32_t t = 0; t < T; ++t) {
                        for (uint32_t k = 0; k < K; ++k) {
                            if (f->h_ids[(size_t) t * K + k] != e) {
                                continue;
                            }
                            const float wgt = f->h_wts[(size_t) t * K + k];
                            qrt_cpu_moe_expert_q8(
                                f->cpu, tg, tu, td, buf.data(), buf.data() + loc.gate,
                                buf.data() + loc.gate + loc.up,
                                f->h_x_qs.data() + (size_t) t * E,
                                f->h_x_d.data() + (size_t) t * nb,
                                f->h_x_s.data() + (size_t) t * nb, (int) E,
                                (int) f->cfg.ffn, row.data());
                            for (uint32_t i = 0; i < E; ++i) {
                                out[i] += wgt * row[i];
                            }
                        }
                    }
                }
                cpu_worker_ms[w] = now_ms() - t_thread;
            });
        }
    }

    // H2D leg on this thread, beside the CPU workers. The misses are packed into the
    // staging buffer and copied in one submit per staging load: a copy per expert costs
    // a submit+fence each and measured 2.3 GB/s instead of the design's 9.9.
    if (f->legs.h2d != 0) {
        const double t_h = now_ms();
        unsigned char * stage = (unsigned char *) qrt_vk_staging_map(f->stg);
        if (stage == NULL) {
            for (size_t w = 0; w < workers.size(); ++w) {
                workers[w].join();
            }
            return fail(err, err_size, "staging is not mapped for the expert cache");
        }
        std::vector<qrt_vk_copy_region> regions;
        uint64_t fill = 0;
        auto join_workers = [&]() {
            for (size_t w = 0; w < workers.size(); ++w) {
                workers[w].join();
            }
        };
        int h2d_rc = 0;
        auto flush = [&]() -> int {
            if (regions.empty()) {
                return 0;
            }
            if (qrt_vk_upload_regions(f->dev, f->stg, regions.data(),
                                      (uint32_t) regions.size(), err, err_size) != 0) {
                return -1;
            }
            uint64_t bytes = 0;
            for (size_t r = 0; r < regions.size(); ++r) {
                bytes += regions[r].bytes;
            }
            qrt_cache_note_upload(f->cache, f->layer, bytes);
            regions.clear();
            fill = 0;
            return 0;
        };
        for (uint32_t i = 0; i < f->uniq.size() && h2d_rc == 0; ++i) {
            const qrt_cache_lookup & r = f->lookups[i];
            if (r.hit || r.failed || r.slot < 0) {
                continue;
            }
            const uint32_t e = f->uniq[i];
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0 ||
                loc.bytes > QRT_FFN_STAGING_BYTES) {
                h2d_rc = -1;
                break;
            }
            if (fill + loc.bytes > QRT_FFN_STAGING_BYTES && flush() != 0) {
                h2d_rc = -1;
                break;
            }
            if (qrt_weights_read_expert(f->store, f->layer, e, stage + fill,
                                        QRT_FFN_STAGING_BYTES - fill, err, err_size) != 0) {
                h2d_rc = -1;
                break;
            }
            qrt_vk_copy_region reg;
            reg.src = stage + fill;
            reg.dst = f->cache_buf;
            reg.dst_offset = (uint64_t) r.slot * f->cache_slot_bytes;
            reg.bytes = loc.bytes;
            regions.push_back(reg);
            fill += loc.bytes;
            f->legs.h2d_bytes += loc.bytes;
        }
        if (h2d_rc == 0 && flush() != 0) {
            h2d_rc = -1;
        }
        if (h2d_rc != 0) {
            join_workers();
            return -1;
        }
        f->legs.h2d_ms = now_ms() - t_h;
    }
    if (!workers.empty()) {
        for (size_t w = 0; w < workers.size(); ++w) {
            workers[w].join();
        }
        if (cpu_failed) {
            return fail(err, err_size, "layer %u CPU expert leg failed", f->layer);
        }
        // Busy time of the whole CPU leg: the mean over workers is its wall time while
        // the H2D leg ran beside it.
        double busy = 0.0;
        for (size_t w = 0; w < cpu_worker_ms.size(); ++w) {
            busy += cpu_worker_ms[w];
        }
        f->legs.cpu_ms = cpu_worker_ms.empty() ? 0.0 : busy / (double) cpu_worker_ms.size();
    }

    // Merge the per-expert CPU rows into the per-token routed sum.
    for (uint32_t j = 0; j < f->cpu_pos.size(); ++j) {
        const uint32_t e = f->uniq[f->cpu_pos[j]];
        const float * res = f->cpu_res.data() + (size_t) j * E;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < K; ++k) {
                if (f->h_ids[(size_t) t * K + k] != e) {
                    continue;
                }
                float * dst = f->cpu_out.data() + (size_t) t * E;
                for (uint32_t i = 0; i < E; ++i) {
                    dst[i] += res[i];  // the worker already applied the router weight
                }
            }
        }
    }

    if (f->carry != NULL) {
        f->carry->h2d_ms += f->legs.h2d_ms;
        f->carry->cpu_ms += f->legs.cpu_ms;
    }
    if (f->legs.h2d != 0) {
        qrt_miss_split_update(&f->rates, f->legs.h2d, f->legs.h2d_bytes, f->legs.h2d_ms,
                              0, 0, 0.0);
    }
    return 0;
}

#endif // QRT_HAVE_VULKAN
