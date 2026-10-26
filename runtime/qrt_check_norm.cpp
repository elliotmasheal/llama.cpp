// M2-6 first piece of the layer chain: rmsnorm + Q8_1 pack feeding the dense q8_0 GEMM.
// Three gates in one run, because the planes are only proven if the GEMM that reads them
// is right too: the normalised row against a double reference, the Q8_1 planes against
// the same quantiser, and the projection computed from the device planes.

#include "qrt_checks.h"

#include "qrt_config.h"
#include "qrt_util.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "kernels/gemm.h"
#include "kernels/norm.h"
#include "qrt_check_support.h"
#include "qrt_half.h"
#include "vk_transfer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int qrt_check_norm(const qrt_config * cfg, const char * gguf_path, const char * norm_name,
                          const char * proj_name, uint32_t tokens, const char * spv_dir) {
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
    const qrt_gguf_tensor * tw = NULL;
    const qrt_gguf_tensor * tp = NULL;
    if (qrt_gguf_require(g, norm_name, &tw, err, sizeof(err)) != 0 ||
        qrt_gguf_require(g, proj_name, &tp, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    if (tw->type != 0) {
        fprintf(stderr, "qrt: %s is %s, the norm weight must be f32\n", norm_name,
                qrt_gguf_type_name(tw->type));
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    if (tp->type != 8) {
        fprintf(stderr, "qrt: %s is %s, the projection must be q8_0\n", proj_name,
                qrt_gguf_type_name(tp->type));
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t k = (uint32_t) tw->ne[0];
    const uint32_t n = (uint32_t) tp->ne[1];
    if ((uint32_t) tp->ne[0] != k) {
        fprintf(stderr, "qrt: %s reduces over %llu, the norm is %u wide\n", proj_name,
                (unsigned long long) tp->ne[0], k);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    if (k % 32u != 0 || n % 64u != 0) {
        fprintf(stderr, "qrt: shape k=%u n=%u does not fit the norm and GEMM tiles\n", k, n);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    if (tokens == 0) {
        tokens = 1;
    }

    std::vector<float> w((size_t) k);
    if (qrt_gguf_read_tensor(g, tw, w.data(), (size_t) k * sizeof(float), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    std::vector<unsigned char> wraw((size_t) tp->nbytes);
    if (qrt_gguf_read_tensor(g, tp, wraw.data(), wraw.size(), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_gguf_close(g);

    const uint64_t wbytes_n = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, n, k);
    std::vector<unsigned char> wbytes((size_t) wbytes_n);
    qrt_gemm_q8k_prepare(wraw.data(), n, k, wbytes.data());

    // Deterministic input in [-1, 1), the same generator style as the other checks.
    std::vector<float> x((size_t) tokens * k);
    uint32_t seed = 12345u;
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) ((seed >> 8) % 2001u) - 1000) * 0.001f;
    }
    const uint32_t blocks = k / 32u;
    std::vector<signed char> ref_qs((size_t) tokens * k);
    std::vector<uint32_t> ref_ds((size_t) tokens * blocks);
    std::vector<float> ref_out((size_t) tokens * k);
    // The metadata key is attention.layer_norm_rms_epsilon, which this model leaves at the
    // 1e-6 the other modules already assume.
    const float eps = 1e-6f;
    qrt_norm_cpu(tokens, k, x.data(), w.data(), eps, ref_out.data(), ref_qs.data(), ref_ds.data());

    int rc = 2;
    qrt_gemm * gemm = NULL;
    qrt_norm * nrm = NULL;
    VkBuffer xb = VK_NULL_HANDLE, wb = VK_NULL_HANDLE, ob = VK_NULL_HANDLE;
    VkBuffer qsb = VK_NULL_HANDLE, dsb = VK_NULL_HANDLE, wbuf = VK_NULL_HANDLE;
    VkBuffer gout = VK_NULL_HANDLE;
    VkDeviceMemory xm = VK_NULL_HANDLE, wm = VK_NULL_HANDLE, om = VK_NULL_HANDLE;
    VkDeviceMemory qsm = VK_NULL_HANDLE, dsm = VK_NULL_HANDLE, wbm = VK_NULL_HANDLE;
    VkDeviceMemory gm = VK_NULL_HANDLE;
    qrt_vk_staging * stg = NULL;
    do {
        const uint32_t alloc_tokens = tokens < 2u ? 2u : tokens;
        const uint64_t x_bytes  = (uint64_t) tokens * k * 4;
        const uint64_t qs_bytes = (uint64_t) tokens * k;
        const uint64_t ds_bytes = (uint64_t) tokens * blocks * 4;
        const uint64_t out_bytes = x_bytes;
        const uint64_t x_alloc  = (uint64_t) alloc_tokens * k * 4;
        const uint64_t qs_alloc = (uint64_t) alloc_tokens * k;
        const uint64_t ds_alloc = (uint64_t) alloc_tokens * blocks * 4;
        const uint64_t gout_bytes = (uint64_t) n * tokens * 4;
        const uint64_t stg_bytes = x_alloc > gout_bytes ? (x_alloc > wbytes_n ? x_alloc : wbytes_n)
                                                        : (gout_bytes > wbytes_n ? gout_bytes : wbytes_n);
        const VkBufferUsageFlags usage = QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                                         QRT_VK_USAGE_TRANSFER_DST;
        if (qrt_gemm_open(dev, spv_dir, &gemm, err, sizeof(err)) != 0 ||
            qrt_norm_open(dev, spv_dir, &nrm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, x_alloc, usage, &xb, &xm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, (uint64_t) k * 4, usage, &wb, &wm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, x_alloc, usage, &ob, &om, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, qs_alloc, usage, &qsb, &qsm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, ds_alloc, usage, &dsb, &dsm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, wbytes_n, usage, &wbuf, &wbm, err, sizeof(err)) != 0 ||
            qrt_vk_alloc(dev, gout_bytes, usage, &gout, &gm, err, sizeof(err)) != 0 ||
            qrt_vk_staging_create(dev, stg_bytes, &stg, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        if (qrt_vk_upload(dev, stg, xb, 0, x.data(), x_bytes, err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, wb, 0, w.data(), (uint64_t) k * 4, err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, wbuf, 0, wbytes.data(), wbytes_n, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        qrt_norm_args na;
        memset(&na, 0, sizeof(na));
        na.x = xb; na.xbase = 0;
        na.w = wb; na.wbase = 0;
        na.out = ob; na.obase = 0;
        na.qs = qsb; na.qsbase = 0;
        na.ds = dsb; na.dsbase = 0;
        na.k = k;
        na.tokens = tokens;
        na.eps = eps;
        na.quant = 1;
        if (qrt_norm_run(nrm, &na, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        std::vector<float> got_out((size_t) tokens * k);
        std::vector<unsigned char> got_qs((size_t) tokens * k);
        std::vector<uint32_t> got_ds((size_t) tokens * blocks);
        if (qrt_vk_download(dev, stg, ob, 0, got_out.data(), out_bytes, err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, qsb, 0, got_qs.data(), qs_bytes, err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, dsb, 0, got_ds.data(), ds_bytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        double num = 0.0, den = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            const double d = (double) got_out[i] - (double) ref_out[i];
            num += fabs(d);
            den += fabs((double) ref_out[i]);
            max_abs = fmax(max_abs, fabs(d));
        }
        const double rel = den > 0.0 ? num / den : num;
        const int norm_ok = rel <= 1e-6;
        printf("norm_row k=%u tokens=%u rel_l1=%.3e max_abs=%.3e verdict=%s\n", k, tokens,
               rel, max_abs, norm_ok ? "PASS" : "FAIL");

        // The planes are compared block by block: a structural error (wrong interleave,
        // wrong block order) shows up as nearly every block differing, a rounding tie as a
        // handful of single steps.
        const uint32_t nblocks = tokens * blocks;
        uint32_t qs_bad = 0, qs_maxdiff = 0, ds_bad = 0, sum_bad = 0;
        for (uint32_t b = 0; b < nblocks; ++b) {
            int bad = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const int a = (int) (signed char) got_qs[(size_t) b * 32u + i];
                const int r = (int) ref_qs[(size_t) b * 32u + i];
                const int diff = abs(a - r);
                if (diff != 0) {
                    bad = 1;
                }
                qs_maxdiff = (uint32_t) (diff > (int) qs_maxdiff ? diff : (int) qs_maxdiff);
            }
            qs_bad += (uint32_t) bad;
            const uint32_t gd = got_ds[b];
            const uint32_t rd = ref_ds[b];
            ds_bad += (gd & 0xFFFFu) != (rd & 0xFFFFu) ? 1u : 0u;
            sum_bad += (gd >> 16) != (rd >> 16) ? 1u : 0u;
        }
        const double bad_frac = (double) qs_bad / (double) nblocks;
        const int plane_ok = qs_maxdiff <= 1u && bad_frac <= 1e-3;
        printf("norm_q8 blocks=%u qs_bad=%u qs_frac=%.2e qs_maxdiff=%u ds_bad=%u sum_bad=%u "
               "verdict=%s\n", nblocks, qs_bad, bad_frac, qs_maxdiff, ds_bad, sum_bad,
               plane_ok ? "PASS" : "FAIL");

        // The chain: the GEMM eats exactly the device planes the norm just wrote.
        qrt_gemm_args ga;
        memset(&ga, 0, sizeof(ga));
        ga.w = wbuf; ga.wbase = 0;
        ga.a_qs = qsb; ga.a_ds = dsb; ga.abase = 0;
        ga.out = gout; ga.obase = 0;
        ga.n = n; ga.k = k; ga.m = tokens;
        ga.type = QRT_QUANT_Q8_0;
        if (qrt_gemm_dispatch(gemm, &ga, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const uint32_t rows = 32, cols = tokens < 64u ? tokens : 64u;
        std::vector<float> got((size_t) n * cols);
        // The dense output is [n][m], so one compared row is a 64-float slice at row * m.
        for (uint32_t row = 0; row < rows; ++row) {
            if (qrt_vk_download(dev, stg, gout, (uint64_t) row * tokens * 4,
                                got.data() + (size_t) row * cols, (uint64_t) cols * 4,
                                err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
        }
        double gnum = 0.0, gden = 0.0;
        for (uint32_t row = 0; row < rows; ++row) {
            const unsigned char * wrow = wraw.data() + (size_t) row * blocks * 34u;
            for (uint32_t tok = 0; tok < cols; ++tok) {
                double acc = 0.0;
                for (uint32_t b = 0; b < blocks; ++b) {
                    const unsigned char * blk = wrow + (size_t) b * 34u;
                    const uint32_t dh = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
                    const float d = f16_to_f32(dh);
                    const uint32_t ab = tok * blocks + b;
                    const float aq = f16_to_f32(got_ds[ab] & 0xFFFFu);
                    double sub = 0.0;
                    for (uint32_t e = 0; e < 32u; ++e) {
                        sub += (double) (int8_t) blk[2 + e] *
                               (double) (signed char) got_qs[(size_t) ab * 32u + e];
                    }
                    acc += (double) d * (double) aq * sub;
                }
                const double a = (double) got[(size_t) row * cols + tok];
                gnum += fabs(a - acc);
                gden += fabs(acc);
            }
        }
        const double grel = gden > 0.0 ? gnum / gden : gnum;
        const int chain_ok = grel <= 1e-5;
        printf("norm_gemm n=%u k=%u m=%u rows=%u cols=%u rel_l1=%.3e verdict=%s\n", n, k, tokens,
               rows, cols, grel, chain_ok ? "PASS" : "FAIL");

        // C high-precision slice: the same device projection against a float-activation
        // reference. The weights are still the model's q8_0 values (dequantised in double);
        // only the activation side skips Q8_1. This isolates the activation quantiser error
        // on the largest dense leg without a second device path.
        double fnum = 0.0, fden = 0.0, fmax = 0.0;
        for (uint32_t row = 0; row < rows; ++row) {
            const unsigned char * wrow = wraw.data() + (size_t) row * blocks * 34u;
            for (uint32_t tok = 0; tok < cols; ++tok) {
                double acc = 0.0;
                for (uint32_t b = 0; b < blocks; ++b) {
                    const unsigned char * blk = wrow + (size_t) b * 34u;
                    const uint32_t dh = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
                    const float d = f16_to_f32(dh);
                    double sub = 0.0;
                    for (uint32_t e = 0; e < 32u; ++e) {
                        sub += (double) (int8_t) blk[2 + e] *
                               (double) got_out[(size_t) tok * k + b * 32u + e];
                    }
                    acc += sub * (double) d;
                }
                const double a = (double) got[(size_t) row * cols + tok];
                const double diff = fabs(a - acc);
                fnum += diff;
                fden += fabs(acc);
                fmax = fmax > diff ? fmax : diff;
            }
        }
        const double frel = fden > 0.0 ? fnum / fden : fnum;
        const int f32_ok = frel <= 1e-2;
        printf("norm_gemm_f32 n=%u k=%u m=%u rows=%u cols=%u rel_l1=%.3e max_abs=%.3e "
               "verdict=%s\n", n, k, tokens, rows, cols, frel, fmax,
               f32_ok ? "PASS" : "FAIL");

        // Timing: the norm leg alone, then the pair as the layer would run it.
        double tn[5], tg[5];
        int timed = 1;
        for (int i = 0; i < 5; ++i) {
            const double t0 = now_ms();
            if (qrt_norm_run(nrm, &na, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                timed = 0;
                break;
            }
            tn[i] = now_ms() - t0;
            const double t1 = now_ms();
            if (qrt_gemm_dispatch(gemm, &ga, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                timed = 0;
                break;
            }
            tg[i] = now_ms() - t1;
        }
        if (timed) {
            const double norm_ms = qrt_check_median(tn, 5);
            const double gemm_ms = qrt_check_median(tg, 5);
            const double moved = (double) tokens * k * 8.0 + (double) tokens * k * 1.125;
            printf("norm_perf tokens=%u k=%u n=%u norm_ms=%.3f us_per_token=%.3f read_gbs=%.1f "
                   "gemm_ms=%.3f chain_ms=%.3f us_per_token_chain=%.3f\n", tokens, k, n, norm_ms,
                   norm_ms * 1000.0 / (double) tokens, moved / norm_ms / 1e6, gemm_ms,
                   norm_ms + gemm_ms, (norm_ms + gemm_ms) * 1000.0 / (double) tokens);
        }

        // A: the quantiser itself, on identical f32 inputs. This bypasses the RMS
        // reduction: the normalised row above can differ by one ulp between the shader and
        // the double CPU norm, which is an arithmetic difference, not a quantiser rule.
        std::vector<float> bx((size_t) 2 * k, 0.0f);
        const float ties[16] = { 127.0f, 0.5f, -0.5f, 1.5f, -1.5f, 126.5f, -126.5f, 64.5f,
                                 -64.5f, 2.5f, -2.5f, 63.5f, -63.5f, 0.0f, -0.0f, 1.0f };
        for (uint32_t i = 0; i < 16u; ++i) {
            bx[i] = ties[i];
        }
        bx[k + 0u] = 1.0e-30f;
        bx[k + 1u] = -1.0e-30f;
        bx[k + 2u] = 1.0e-38f;
        bx[k + 3u] = -1.0e-38f;
        bx[k + 4u] = 3.0e-39f;
        bx[k + 5u] = 7.0e-39f;
        std::vector<unsigned char> bq((size_t) 2 * k);
        std::vector<uint32_t> bd((size_t) 2 * blocks);
        std::vector<signed char> bqr((size_t) 2 * k);
        std::vector<uint32_t> bdr((size_t) 2 * blocks);
        if (qrt_vk_upload(dev, stg, xb, 0, bx.data(), (uint64_t) 2 * k * 4, err,
                          sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        qrt_norm_args qa = na;
        qa.tokens = 2;
        qa.no_norm = 1;
        qa.no_out = 1;
        if (qrt_norm_run(nrm, &qa, err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, qsb, 0, bq.data(), (uint64_t) 2 * k, err,
                            sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, dsb, 0, bd.data(), (uint64_t) 2 * blocks * 4, err,
                            sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        qrt_norm_quant_cpu(2, k, bx.data(), NULL, bqr.data(), bdr.data());
        uint32_t bq_bad = 0, bd_bad = 0, bs_bad = 0, bq_max = 0;
        for (uint32_t b = 0; b < 2u * blocks; ++b) {
            int bad = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const int a = (int) (signed char) bq[(size_t) b * 32u + i];
                const int r = (int) bqr[(size_t) b * 32u + i];
                const int diff = abs(a - r);
                bad |= diff != 0;
                bq_max = (uint32_t) (diff > (int) bq_max ? diff : (int) bq_max);
            }
            bq_bad += (uint32_t) bad;
            bd_bad += (bd[b] & 0xFFFFu) != (bdr[b] & 0xFFFFu) ? 1u : 0u;
            bs_bad += (bd[b] >> 16) != (bdr[b] >> 16) ? 1u : 0u;
        }
        const int quant_ok = bq_bad == 0 && bd_bad == 0 && bs_bad == 0;
        printf("norm_quant tokens=2 blocks=%u qs_bad=%u ds_bad=%u sum_bad=%u maxdiff=%u "
               "verdict=%s\n", 2u * blocks, bq_bad, bd_bad, bs_bad, bq_max,
               quant_ok ? "PASS" : "FAIL");
        rc = (norm_ok && plane_ok && chain_ok && quant_ok && f32_ok) ? 0 : 1;
    } while (0);

    qrt_vk_staging_destroy(dev, stg);
    if (xm != VK_NULL_HANDLE) qrt_vk_free(dev, xb, xm);
    if (wm != VK_NULL_HANDLE) qrt_vk_free(dev, wb, wm);
    if (om != VK_NULL_HANDLE) qrt_vk_free(dev, ob, om);
    if (qsm != VK_NULL_HANDLE) qrt_vk_free(dev, qsb, qsm);
    if (dsm != VK_NULL_HANDLE) qrt_vk_free(dev, dsb, dsm);
    if (wbm != VK_NULL_HANDLE) qrt_vk_free(dev, wbuf, wbm);
    if (gm != VK_NULL_HANDLE) qrt_vk_free(dev, gout, gm);
    qrt_norm_close(dev, nrm);
    qrt_gemm_close(dev, gemm);
    qrt_vk_close(dev);
    return rc;
}

