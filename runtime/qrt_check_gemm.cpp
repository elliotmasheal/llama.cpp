// M2-6b: the Q8_0 dense projection kernel. Reads one q8_0 matrix out of the GGUF, checks
// it against a double-precision dequant reference at a small token count, then times the
// real prefill shape so the dense leg stops being an assumption (RESULTS 46).

#include "qrt_checks.h"

#include "qrt_config.h"
#include "qrt_util.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "kernels/gemm.h"
#include "qrt_check_support.h"
#include "qrt_half.h"
#include "vk_transfer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

int qrt_check_dense_gemm(const qrt_config * cfg, const char * gguf_path,
                         const char * tensor_name, uint32_t tokens, uint32_t dense_rows,
                         const char * spv_dir) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, tensor_name, &t, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    if (t->type != 8) {
        fprintf(stderr, "qrt: %s is %s, the dense kernel reads q8_0\n", tensor_name,
                qrt_gguf_type_name(t->type));
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t k = (uint32_t) t->ne[0];
    const uint32_t n = (uint32_t) t->ne[1];
    const uint32_t check_tokens = 64;
    if (tokens < check_tokens) {
        tokens = check_tokens;
    }
    tokens = (tokens + 31u) / 32u * 32u;
    if (k % 256 != 0 || n % 64 != 0) {
        fprintf(stderr, "qrt: shape k=%u n=%u does not fit the tiles\n", k, n);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }

    // The tensor pointer dies with the index, so snapshot what the run needs first.
    const uint64_t raw_bytes = t->nbytes;
    std::vector<unsigned char> wraw((size_t) raw_bytes);
    if (qrt_gguf_read_tensor(g, t, wraw.data(), wraw.size(), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_gguf_close(g);

    // Repack into the aligned quants/deltas layout the kernel expects (RESULTS 47).
    const uint64_t wbytes_n = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, n, k);
    std::vector<unsigned char> wbytes((size_t) wbytes_n);
    qrt_gemm_q8k_prepare(wraw.data(), n, k, wbytes.data());

    int rc = 2;
    qrt_gemm * gemm = NULL;
    VkBuffer wbuf = VK_NULL_HANDLE;
    VkDeviceMemory wmem = VK_NULL_HANDLE;
    VkBuffer aqs = VK_NULL_HANDLE;
    VkDeviceMemory aqs_mem = VK_NULL_HANDLE;
    VkBuffer ads = VK_NULL_HANDLE;
    VkDeviceMemory ads_mem = VK_NULL_HANDLE;
    VkBuffer out = VK_NULL_HANDLE;
    VkDeviceMemory out_mem = VK_NULL_HANDLE;
    VkBuffer out2 = VK_NULL_HANDLE;
    VkDeviceMemory out2_mem = VK_NULL_HANDLE;
    qrt_vk_staging * stg = NULL;
    unsigned char * host_qs = NULL;
    uint32_t * host_ds = NULL;
    do {
        if (qrt_gemm_open(dev, spv_dir, &gemm, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const uint64_t qs_bytes = (uint64_t) tokens * k;
        const uint64_t ds_bytes = (uint64_t) tokens * (k / 32) * 4;
        const uint64_t out_bytes = (uint64_t) tokens * n * 4;
        if (qrt_vk_alloc(dev, wbytes_n, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         &wbuf, &wmem, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, qs_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         &aqs, &aqs_mem, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, ds_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         &ads, &ads_mem, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, out_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC,
                         &out, &out_mem, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        host_qs = (unsigned char *) malloc((size_t) qs_bytes);
        host_ds = (uint32_t *) malloc((size_t) ds_bytes);
        if (host_qs == NULL || host_ds == NULL) {
            fprintf(stderr, "qrt: out of memory for the activation planes\n");
            break;
        }
        // Deterministic Q8_1 activations, same generator as the MoE harness.
        uint32_t seed = 99991u;
        for (uint64_t b = 0; b < (uint64_t) tokens * (k / 32); ++b) {
            unsigned char * q = host_qs + b * 32;
            int32_t sum = 0;
            for (uint32_t j = 0; j < 32; ++j) {
                seed = seed * 1664525u + 1013904223u;
                const int32_t v = (int32_t) ((seed >> 8) % 255u) - 127;
                q[j] = (unsigned char) (int8_t) v;
                sum += v;
            }
            seed = seed * 1664525u + 1013904223u;
            const uint32_t d = 0x2800u + ((seed >> 8) & 0x0FFFu);
            host_ds[b] = d | ((uint32_t) (uint16_t) (sum + 32768) << 16);
        }
        if (qrt_vk_staging_create(dev, qs_bytes > wbytes_n ? qs_bytes : wbytes_n, &stg,
                                  err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, wbuf, 0, wbytes.data(), wbytes_n, err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, aqs, 0, host_qs, qs_bytes, err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, ads, 0, host_ds, ds_bytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        qrt_gemm_args args;
        memset(&args, 0, sizeof(args));
        args.w = wbuf;
        args.wbase = 0;            // q8_0 is byte addressed
        args.a_qs = aqs;
        args.a_ds = ads;
        args.out = out;
        args.n = n;
        args.k = k;
        args.m = check_tokens;
        args.type = QRT_QUANT_Q8_0;

        // Correctness: first 32 rows x check_tokens, against the canonical q8_0 dequant.
        // Two dispatches go into one command buffer with their own output buffers: the old
        // shared-set design made the second write apply to both (DECISIONS 65.2).
        if (qrt_vk_alloc(dev, (uint64_t) check_tokens * n * 4,
                         QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC, &out2, &out2_mem,
                         err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        qrt_gemm_args args2 = args;
        args2.out = out2;
        if (qrt_gemm_begin(gemm, 0, err, sizeof(err)) != 0 ||
            qrt_gemm_record(gemm, &args, err, sizeof(err)) != 0 ||
            qrt_gemm_record(gemm, &args2, err, sizeof(err)) != 0 ||
            qrt_gemm_submit(gemm, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        std::vector<float> got((size_t) check_tokens * n);
        std::vector<float> got2((size_t) check_tokens * n);
        if (qrt_vk_download(dev, stg, out, 0, got.data(), (uint64_t) check_tokens * n * 4,
                            err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, out2, 0, got2.data(), (uint64_t) check_tokens * n * 4,
                            err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const uint32_t rows = dense_rows;   // rows 32..63 were never checked before (RESULTS 72)
        // One reference, two outputs: a mismatch between A and B means the two dispatches
        // did not each see their own descriptor set.
        auto rel_of = [&](const std::vector<float> & g, double * num_out) -> double {
            double num = 0.0, den = 0.0;
            for (uint32_t row = 0; row < rows; ++row) {
                const unsigned char * wrow = wraw.data() + (size_t) row * (k / 32) * 34;
                for (uint32_t tok = 0; tok < check_tokens; ++tok) {
                    double acc = 0.0;
                    for (uint32_t b = 0; b < k / 32; ++b) {
                        const unsigned char * blk = wrow + (size_t) b * 34;
                        const uint32_t dh = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
                        const float d = f16_to_f32(dh);
                        const uint32_t ab = tok * (k / 32) + b;
                        const float aq = f16_to_f32(host_ds[ab] & 0xFFFFu);
                        double sub = 0.0;
                        for (uint32_t e = 0; e < 32; ++e) {
                            const int8_t qv = (int8_t) blk[2 + e];
                            const int8_t qa = (int8_t) host_qs[(size_t) ab * 32 + e];
                            sub += (double) qv * (double) qa;
                        }
                        acc += (double) d * (double) aq * sub;
                    }
                    const double a = g[(size_t) row * check_tokens + tok];
                    num += fabs(a - acc);
                    den += fabs(acc);
                }
            }
            if (num_out != NULL) {
                *num_out = num;
            }
            return den > 0.0 ? num / den : num;
        };
        double num_a = 0.0, num_b = 0.0;
        const double rel = rel_of(got, &num_a);
        const double rel2 = rel_of(got2, &num_b);
        printf("dense_gemm_pair out_a_rel_l1=%.3e out_b_rel_l1=%.3e same_l1=%.3e verdict=%s\n",
               rel, rel2, num_a == num_b ? 0.0 : fabs(rel - rel2),
               (rel <= 1e-5 && rel2 <= 1e-5) ? "PASS" : "FAIL");
        if (rel > 1e-5 || rel2 > 1e-5) {
            break;
        }
        printf("dense_gemm_check tensor=%s type=q8_0 k=%u n=%u rows=%u tokens=%u rel_l1=%.3e "
               "verdict=%s\n", tensor_name, k, n, rows, check_tokens, rel,
               rel <= 1e-5 ? "PASS" : "FAIL");

        // Timing at the prefill shape.
        args.m = tokens;
        double t[5];
        for (int i = 0; i < 5; ++i) {
            const double t0 = now_ms();
            if (qrt_gemm_dispatch(gemm, &args, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            t[i] = now_ms() - t0;
        }
        const double med = qrt_check_median(t, 5);
        const double flops = 2.0 * (double) tokens * n * k;
        printf("dense_gemm m=%u n=%u k=%u median_ms=%.3f tflops=%.2f out_mib=%.0f\n",
               tokens, n, k, med, med > 0.0 ? flops / med / 1e9 : 0.0,
               (double) tokens * n * 4.0 / (1024.0 * 1024.0));
        rc = rel <= 1e-5 ? 0 : 1;
    } while (0);
    free(host_qs);
    free(host_ds);
    qrt_vk_staging_destroy(dev, stg);
    if (wmem != VK_NULL_HANDLE) qrt_vk_free(dev, wbuf, wmem);
    if (aqs_mem != VK_NULL_HANDLE) qrt_vk_free(dev, aqs, aqs_mem);
    if (ads_mem != VK_NULL_HANDLE) qrt_vk_free(dev, ads, ads_mem);
    if (out_mem != VK_NULL_HANDLE) qrt_vk_free(dev, out, out_mem);
    if (out2_mem != VK_NULL_HANDLE) qrt_vk_free(dev, out2, out2_mem);
    qrt_gemm_close(dev, gemm);
    qrt_vk_close(dev);
    return rc;
}

