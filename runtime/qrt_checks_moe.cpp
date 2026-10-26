#include "qrt_checks.h"

#include "qrt_config.h"
#include "qrt_util.h"
#include "gguf_index.h"
#include "kernels/norm.h"
#include "moe.h"
#include "vk_dev.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static double moe_rel_l1(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
        den += fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : num;
}

int qrt_check_moe_batch(const qrt_config * cfg, const char * gguf_path,
                               const char * store_prefix, uint32_t layer, const char * spv_dir) {
    char err[QRT_ERR_SIZE] = {};
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    const qrt_gguf_hparams hp = *qrt_gguf_hparams_of(g);
    qrt_gguf_close(g);
    const uint32_t E = (uint32_t) hp.n_embd;
    const uint32_t NE = (uint32_t) hp.expert_count;
    const uint32_t K = (uint32_t) hp.expert_used_count;
    if (E == 0 || E % 32 != 0 || NE == 0 || K == 0 || K > 8) {
        fprintf(stderr, "qrt: GGUF has n_embd=%d experts=%d top_k=%d\n", hp.n_embd,
                hp.expert_count, hp.expert_used_count);
        return 2;
    }
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    const uint32_t max_tokens = 8;
    std::vector<float> x((size_t) max_tokens * E), ones(E, 1.0f);
    std::vector<float> xnorm((size_t) max_tokens * E);
    std::vector<signed char> qs((size_t) max_tokens * E);
    std::vector<uint32_t> ds((size_t) max_tokens * (E / 32u));
    uint32_t seed = 0x5EED1234u;
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) ((seed >> 8) % 2001u) - 1000) * 0.001f;
    }
    qrt_norm_cpu(max_tokens, E, x.data(), ones.data(), 1e-6f, xnorm.data(), qs.data(),
                 ds.data());
    qrt_batch_moe * serial = NULL;
    if (qrt_batch_moe_open(dev, gguf_path, store_prefix, spv_dir, layer, 1, &serial, err,
                     sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t ks[] = { 1, 2, 4, 5, 8 };
    int failures = 0;
    for (uint32_t k : ks) {
        qrt_batch_moe * batch = NULL;
        if (qrt_batch_moe_open(dev, gguf_path, store_prefix, spv_dir, layer, k, &batch, err,
                         sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            failures = 1;
            break;
        }
        std::vector<float> bm((size_t) k * E), sm((size_t) k * E), rm((size_t) k * E);
        std::vector<uint32_t> bi((size_t) k * K), si((size_t) k * K), ri((size_t) k * K);
        std::vector<float> bw((size_t) k * K), sw((size_t) k * K), rw((size_t) k * K);
        const double mb_t0 = now_ms();
        int rc = qrt_batch_moe_run(batch, xnorm.data(), qs.data(), ds.data(), bm.data(), bi.data(),
                             bw.data(), err, sizeof(err));
        const double mb_ms = now_ms() - mb_t0;
        for (uint32_t t = 0; t < k && rc == 0; ++t) {
            rc = qrt_batch_moe_run(serial, xnorm.data() + (size_t) t * E, qs.data() + (size_t) t * E,
                             ds.data() + (size_t) t * (E / 32u), sm.data() + (size_t) t * E,
                             si.data() + (size_t) t * K, sw.data() + (size_t) t * K, err,
                             sizeof(err));
        }
        if (rc == 0) {
            rc = qrt_batch_moe_ref_cpu(batch, xnorm.data(), qs.data(), ds.data(), rm.data(),
                                 ri.data(), rw.data(), err, sizeof(err));
        }
        if (rc != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_batch_moe_close(dev, batch);
            failures = 1;
            break;
        }
        uint32_t serial_ids = 0, ref_ids = 0;
        std::vector<uint8_t> seen(NE, 0);
        for (uint32_t i = 0; i < k * K; ++i) {
            serial_ids += bi[i] == si[i] ? 1u : 0u;
            ref_ids += bi[i] == ri[i] ? 1u : 0u;
            seen[si[i]] = 1;
        }
        uint32_t union_experts = 0;
        for (uint8_t v : seen) {
            union_experts += v != 0 ? 1u : 0u;
        }
        const uint32_t batch_experts = qrt_batch_moe_experts_used(batch);
        const double weights_rel = moe_rel_l1(bw.data(), rw.data(), (size_t) k * K);
        const double serial_rel = moe_rel_l1(bm.data(), sm.data(), (size_t) k * E);
        const double ref_rel = moe_rel_l1(bm.data(), rm.data(), (size_t) k * E);
        const bool ok = serial_ids == k * K && batch_experts == union_experts &&
                        weights_rel <= 1e-5 && serial_rel <= 1e-6 && ref_ids == k * K &&
                        ref_rel <= 1e-4;
        printf("moe_batch k=%u layer=%u top_k=%u ids=%u/%u union=%u/%u weights_rel=%.3e "
               "serial_rel=%.3e ref_ids=%u/%u ref_rel=%.3e pairs=%u rows=%u ms=%.3f verdict=%s\n",
               k, layer, K, serial_ids, k * K, batch_experts, union_experts, weights_rel,
               serial_rel, ref_ids, k * K, ref_rel, qrt_batch_moe_pairs(batch),
               qrt_batch_moe_rows(batch), mb_ms, ok ? "PASS" : "FAIL");
        if (!ok) {
            ++failures;
        }
        qrt_batch_moe_close(dev, batch);
    }
    qrt_batch_moe_close(dev, serial);
    qrt_vk_close(dev);
    return failures == 0 ? 0 : 1;
}
