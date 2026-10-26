#include "qrt_config.h"
#include "qrt_cli.h"
#include "qrt_telemetry.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "weights.h"
#include "prefill.h"
#include "attn_deltanet.h"
#include "attn_full.h"
#include "layer.h"
#include "model.h"
#include "expert_cache.h"
#include "moe.h"
#include "kernels/norm.h"
#include "kernels/moe_expert.h"

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

static int g_failures = 0;

// SPIR-V directory resolved from argv[0]; every check that needs a shader uses it.
static char g_spv_dir[1024] = ".";

static float f16_to_f32_local(uint32_t h);
static double median_ms(double * v, int n);

// Defined with the measurement helpers below; the MoE check needs it earlier.
static double now_ms(void);

static void expect(int ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_failures;
    }
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// qrt_config_parse wants char **; every argument it reads is a literal or argv.
static char * lit(const char * s) {
    return const_cast<char *>(s);
}

static int parse_argv(qrt_config * cfg, int n, const char * const * args, char * err, size_t err_size) {
    char * av[26];
    if (n > 24) {
        return -1;
    }
    av[0] = lit("qrt");
    for (int i = 0; i < n; ++i) {
        av[i + 1] = lit(args[i]);
    }
    av[n + 1] = NULL;
    return qrt_config_parse(cfg, n + 1, av, err, err_size);
}

static int count_lines(const char * path, char * first, size_t first_size) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    int lines = 0;
    char buf[512];
    while (fgets(buf, sizeof(buf), f) != NULL) {
        if (lines == 0 && first != NULL && first_size > 0) {
            snprintf(first, first_size, "%s", buf);
        }
        ++lines;
    }
    fclose(f);
    return lines;
}

// M2-5 step 2: the int8 GEMM as the compute leg. Runs one expert's gate/up through the
// Q4_K x Q8_1 kernel (the down matrix is Q5_K in this store and still skipped), then an
// optional CPU dequant cross-check of the gate output.
static int qrt_moe_check(const qrt_config * cfg, const char * prefix, const char * spv_dir,
                         uint32_t layer, uint32_t expert, uint32_t rows) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    printf("moe_check device_dot_product=%d\n", dev->has_dot_product);
    qrt_weights * w = NULL;
    if (qrt_weights_open(prefix, &w, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_moe_cfg mcfg;
    memset(&mcfg, 0, sizeof(mcfg));
    mcfg.tokens = 64;        // 2048 tokens x top-8 over 256 experts
    mcfg.hidden = 2048;
    mcfg.ffn    = 512;
    qrt_moe * moe = NULL;
    int rc = 0;
    if (qrt_moe_open(dev, w, spv_dir, &mcfg, &moe, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        rc = 2;
    } else {
        static const char * type_name[] = { "q4k", "q5k", "q6k" };
        qrt_moe_check_result res;
        const double t0 = now_ms();
        if (qrt_moe_check(dev, moe, w, layer, expert, rows, &res, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
        } else {
            const double dt = now_ms() - t0;
            const int pass = res.gate_rel_l1 <= 1e-4 && res.down_rel_l1 <= 1e-4 &&
                             res.chain_rel_l1 <= 1e-4 && res.batch_rel_l1 <= 1e-4;
            printf("moe_matrix_check layer=%u expert=%u rows=%u tokens=%u "
                   "gate=%s gate_rel_l1=%.3e down=%s down_rel_l1=%.3e "
                   "chain_rel_l1=%.3e batch_rel_l1=%.3e ms=%.1f verdict=%s\n",
                   layer, expert, rows, mcfg.tokens, type_name[res.gate_type],
                   res.gate_rel_l1, type_name[res.down_type], res.down_rel_l1,
                   res.chain_rel_l1, res.batch_rel_l1, dt, pass ? "PASS" : "FAIL");
            if (!pass) {
                rc = 1;
            }
        }
        qrt_moe_close(dev, moe);
    }
    qrt_weights_close(w);
    printf("moe_io_failures=%u alloc_mib=%llu\n", dev->io_failures,
           (unsigned long long) (dev->alloc_bytes >> 20));
    qrt_vk_close(dev);
    return rc;
}

// M2-6b: the Q8_0 dense projection kernel. Reads one q8_0 matrix out of the GGUF, checks
// it against a double-precision dequant reference at a small token count, then times the
// real prefill shape so the dense leg stops being an assumption (RESULTS 46).
static int qrt_dense_gemm_check(const qrt_config * cfg, const char * gguf_path,
                                const char * tensor_name, uint32_t tokens,
                                uint32_t dense_rows) {
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
        char dir[1024];
        snprintf(dir, sizeof(dir), "%s", g_spv_dir);
        if (qrt_gemm_open(dev, dir, &gemm, err, sizeof(err)) != 0) {
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
                        const float d = f16_to_f32_local(dh);
                        const uint32_t ab = tok * (k / 32) + b;
                        const float aq = f16_to_f32_local(host_ds[ab] & 0xFFFFu);
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
        const double med = median_ms(t, 5);
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

// M2-6 first piece of the layer chain: rmsnorm + Q8_1 pack feeding the dense q8_0 GEMM.
// Three gates in one run, because the planes are only proven if the GEMM that reads them
// is right too: the normalised row against a double reference, the Q8_1 planes against the
// same quantiser, and the projection computed from the device planes.
static int qrt_norm_check(const qrt_config * cfg, const char * gguf_path, const char * norm_name,
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
                    const float d = f16_to_f32_local(dh);
                    const uint32_t ab = tok * blocks + b;
                    const float aq = f16_to_f32_local(got_ds[ab] & 0xFFFFu);
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
                    const float d = f16_to_f32_local(dh);
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
            const double norm_ms = median_ms(tn, 5);
            const double gemm_ms = median_ms(tg, 5);
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

// M2-5 step 1: layer-first prefill with uploads in flight. Runs the same layer set
// twice, serial (1 slot) and pipelined (slots), so the overlap is a single variable.
static int qrt_prefill_check(const qrt_config * cfg, const char * prefix, const char * spv,
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

static int qrt_selftest(void) {
    char err[QRT_ERR_SIZE];

    // The selftest must not depend on the caller's environment.
    for (int i = 0; i < qrt_config_switch_count(); ++i) {
        set_env(qrt_config_env_name(i), NULL);
    }

    qrt_config d;
    qrt_config_defaults(&d);
    expect(d.spec_k == 4 && d.draft_vocab == 16384 && d.expert_cache_mib == 4198 &&
           d.miss_split == QRT_MISS_SPLIT_AUTO && d.cache_policy == QRT_CACHE_SLRU &&
           d.prefill_chunk == 2048 && d.kv_k == QRT_KV_Q8_0 && d.kv_v == QRT_KV_Q4_0 &&
           d.kv_page_tokens == 16384 && d.telemetry == 0 && d.vram_tier_mib == 0,
           "defaults match design section 7");

    {
        const char * all[] = {
            "--spec-k", "5", "--draft-vocab", "32768", "--expert-cache-mib", "3800",
            "--miss-split", "60", "--cache-policy", "lru", "--prefill-chunk", "1024",
            "--kv-k", "f16", "--kv-v", "q8_0", "--kv-page-tokens", "8192", "--telemetry", "1",
            "--telemetry-path", "bench/raw/x.csv", "--vram-tier-mib", "6144",
        };
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 24, all, err, sizeof(err)) == 0;
        expect(ok && c.spec_k == 5 && c.draft_vocab == 32768 && c.expert_cache_mib == 3800 &&
               c.miss_split == 60 && c.cache_policy == QRT_CACHE_LRU && c.prefill_chunk == 1024 &&
               c.kv_k == QRT_KV_F16 && c.kv_v == QRT_KV_Q8_0 && c.kv_page_tokens == 8192 &&
               c.telemetry == 1 &&
               strcmp(c.telemetry_path, "bench/raw/x.csv") == 0 && c.vram_tier_mib == 6144,
               "all twelve switches parse from the CLI");
    }

    {
        const char * eq[] = { "--spec-k=2", "--miss-split=auto" };
        qrt_config c;
        qrt_config_defaults(&c);
        expect(parse_argv(&c, 2, eq, err, sizeof(err)) == 0 && c.spec_k == 2 &&
               c.miss_split == QRT_MISS_SPLIT_AUTO,
               "flag=value form and the auto miss split");
    }

    {
        set_env("QRT_SPEC_K", "2");
        set_env("QRT_DRAFT_VOCAB", "0");
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 0, NULL, err, sizeof(err)) == 0;
        expect(ok && c.spec_k == 2 && c.draft_vocab == 0, "environment fills in when the CLI is silent");

        const char * over[] = { "--spec-k", "8" };
        expect(parse_argv(&c, 2, over, err, sizeof(err)) == 0 && c.spec_k == 8,
               "the CLI overrides the environment");
        set_env("QRT_SPEC_K", NULL);
        set_env("QRT_DRAFT_VOCAB", NULL);
    }

    {
        const char * bad[][2] = {
            { "--spec-k", "3" },        { "--draft-vocab", "12" },
            { "--expert-cache-mib", "9000" }, { "--miss-split", "101" },
            { "--cache-policy", "foo" }, { "--prefill-chunk", "0" },
            { "--kv-k", "q5_0" }, { "--kv-v", "q4_1" }, { "--kv-page-tokens", "1000" },
            { "--telemetry", "2" },     { "--vram-tier-mib", "40000" },
            { "--nope", "1" },
        };
        const int n_bad = (int) (sizeof(bad) / sizeof(bad[0]));
        int rejected = 0;
        for (int i = 0; i < n_bad; ++i) {
            qrt_config c;
            qrt_config_defaults(&c);
            char e[QRT_ERR_SIZE] = {0};
            const int ok = parse_argv(&c, 2, bad[i], e, sizeof(e)) != 0 && e[0] != '\0' && c.spec_k == 4;
            if (ok) {
                ++rejected;
            } else {
                printf("  not rejected: %s %s\n", bad[i][0], bad[i][1]);
            }
        }
        expect(rejected == n_bad, "every bad value is rejected with a message and leaves the config usable");
    }

    {
        const char * missing[] = { "--spec-k" };
        qrt_config c;
        qrt_config_defaults(&c);
        expect(parse_argv(&c, 1, missing, err, sizeof(err)) != 0 && c.spec_k == 4,
               "a flag without a value is a recoverable error");
    }

    {
        qrt_config c;
        qrt_config_defaults(&c);
        c.spec_k = 3;
        expect(qrt_config_validate(&c, err, sizeof(err)) != 0, "validate rejects a programmatic spec_k");
        qrt_config_defaults(&c);
        c.kv_k = 9;
        expect(qrt_config_validate(&c, err, sizeof(err)) != 0,
               "validate rejects a KV precision outside f16/q8_0/q4_0");
        qrt_config_defaults(&c);
        c.draft_vocab = 262145;
        expect(qrt_config_validate(&c, err, sizeof(err)) != 0,
               "validate and CLI share the draft vocabulary upper bound");
    }

    {
        qrt_config c;
        qrt_config_defaults(&c);
        const char * bad[] = { "--spec-k", "2", "--kv-v", "bad" };
        expect(parse_argv(&c, 4, bad, err, sizeof(err)) != 0 && c.spec_k == 4 &&
               c.kv_v == QRT_KV_Q4_0, "a rejected configuration is not partially applied");
        set_env("QRT_SPEC_K", "bad");
        const char * good[] = { "--spec-k", "4" };
        expect(parse_argv(&c, 2, good, err, sizeof(err)) == 0 && c.spec_k == 4,
               "an explicit CLI value overrides an invalid environment value");
        set_env("QRT_SPEC_K", NULL);
    }

    {
        const char * low[] = { "--prefill-chunk", "512" };
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 2, low, err, sizeof(err)) == 0 &&
                       qrt_config_validate(&c, err, sizeof(err)) == 0;
        const int notes = qrt_config_notes(&c, stdout);
        expect(ok && notes >= 1, "a legal but costly value is a note, not an error");
    }

    {
        const char * path = "qrt-selftest-telemetry.csv";
        remove(path);
        char e[QRT_ERR_SIZE] = {0};
        qrt_telemetry * t = qrt_telemetry_open(path, e, sizeof(e));
        expect(t != NULL, "the telemetry sink opens");

        qrt_pass_stats s;
        memset(&s, 0, sizeof(s));
        s.k = 4;
        s.pos = 100;
        s.accepted = 3;
        s.hit_experts = 30;
        s.miss_experts = 10;
        s.miss_cpu = 4;
        s.miss_h2d = 6;
        s.h2d_bytes = 6ull * 1916796ull;
        s.draft_ms = 0.6;
        s.gpu_ms = 18.2;
        s.cpu_leg_ms = 0.4;
        s.pass_ms = 19.1;
        qrt_telemetry_pass(t, 7, &s);
        qrt_telemetry_pass(t, 8, &s);
        expect(t != NULL && t->rows == 2 && qrt_telemetry_failed(t) == 0,
               "two pass rows are written and flushed");
        qrt_telemetry_close(t);

        char first[256] = {0};
        const int lines = count_lines(path, first, sizeof(first));
        expect(lines == 3 && strncmp(first, "pass,pos,k,accepted", 19) == 0,
               "the CSV holds the header and one row per pass");
        remove(path);

        qrt_telemetry_pass(NULL, 0, &s);
        qrt_telemetry_close(NULL);
        expect(1, "a NULL sink is a no-op");

        char e2[QRT_ERR_SIZE] = {0};
        expect(qrt_telemetry_open("", e2, sizeof(e2)) == NULL && e2[0] != '\0',
               "an unusable sink path is a recoverable error");
    }

    printf("\nselftest: %d failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

// Fills with a position-dependent pattern so a partial copy or a stale staging
// byte shows up as a mismatch instead of matching zeros.
static void pattern_fill(unsigned char * p, size_t bytes, uint32_t salt) {
    for (size_t i = 0; i < bytes; ++i) {
        p[i] = (unsigned char) ((i * 31u + salt + (i >> 8)) & 0xffu);
    }
}

static int pattern_compare(const unsigned char * a, const unsigned char * b, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        if (a[i] != b[i]) {
            printf("  first mismatch at byte %llu: %02x != %02x\n", (unsigned long long) i,
                   a[i], b[i]);
            return -1;
        }
    }
    return 0;
}

static float f16_to_f32_local(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static double now_ms(void) {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
}

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

// Median of n samples; the array is sorted in place. Copy-call latency on this
// GPU is dominated by submit+sync, so a single sample is not a reading.
static double median_ms(double * v, int n) {
    if (n <= 0) {
        return 0.0;
    }
    qsort(v, (size_t) n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// M2-3 dense residency: streams every non-expert tensor into one device buffer and
// compares a sample back against the file. Reports bytes, rate and the check result.
static int qrt_dense_check(const qrt_config * cfg, const char * gguf_path) {
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
static int qrt_vk_io_check(const qrt_config * cfg, const char * prefix) {
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
        pattern_fill(src.data(), src.size(), (uint32_t) s * 7919u);
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
                same = pattern_compare(src.data(), back.data(), src.size()) == 0;
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
        const double fill_med = median_ms(fill_ms, reps);
        const double up_med = median_ms(up_ms, reps);
        const double down_med = median_ms(down_ms, reps);
        const double unmap_med = median_ms(unmap_ms, reps);
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
            pattern_fill(m, (size_t) bytes, (uint32_t) k);
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
            const double med = median_ms(samples, rounds);
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
                if (pattern_compare(src.data(), back.data(), (size_t) loc.bytes) != 0) {
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
        const double up_med = median_ms(up_ms.empty() ? NULL : &up_ms[0], (int) up_ms.size());
        const double down_med = median_ms(down_ms.empty() ? NULL : &down_ms[0], (int) down_ms.size());
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

static double moe_rel_l1(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
        den += fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : num;
}

static int qrt_moe_batch_check(const qrt_config * cfg, const char * gguf_path,
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

static int qrt_cache_check(const qrt_config * cfg, const char * prefix, const char * trace_path,
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

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    qrt_cli_options options;
    qrt_config cfg;
    qrt_config_defaults(&cfg);
    char err[QRT_ERR_SIZE] = {};
    if (qrt_cli_parse(&options, &cfg, argc, argv, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    if (options.command == QRT_CMD_HELP) {
        qrt_cli_print_help(stdout);
        return 0;
    }
    if (options.command == QRT_CMD_SELFTEST) {
        return qrt_selftest();
    }
    qrt_config_notes(&cfg, stderr);
    if (options.command == QRT_CMD_CONFIG) {
        qrt_config_print_effective(&cfg, stdout);
        return 0;
    }
    const std::string exe = argc > 0 ? argv[0] : "qrt";
    const size_t cut = exe.find_last_of("/\\");
    const std::string spv_dir = cut == std::string::npos ? "." : exe.substr(0, cut);
    snprintf(g_spv_dir, sizeof(g_spv_dir), "%s", spv_dir.c_str());
    if (options.command == QRT_CMD_CACHE) {
        return qrt_cache_check(&cfg, options.cache_prefix, options.cache_trace,
                               (uint32_t) options.cache_decode,
                               (uint32_t) options.cache_protected);
    }
    if (options.command == QRT_CMD_MOE_BATCH) {
        return qrt_moe_batch_check(&cfg, options.moe_batch_gguf, options.layer_store,
                                   (uint32_t) options.moe_layer, g_spv_dir);
    }

    if (options.gguf_path != NULL) {
        qrt_gguf * g = NULL;
        if (qrt_gguf_open(options.gguf_path, &g, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        if (options.require_tensor != NULL) {
            const qrt_gguf_tensor * t = NULL;
            if (qrt_gguf_require(g, options.require_tensor, &t, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_gguf_close(g);
                return 2;
            }
            printf("tensor=%s offset=%llu nbytes=%llu type=%s\n", t->name,
                   (unsigned long long) t->offset, (unsigned long long) t->nbytes,
                   qrt_gguf_type_name(t->type));
        }
        qrt_gguf_print_report(g, stdout);
        qrt_gguf_close(g);
        return 0;
    }
    if (options.require_tensor != NULL) {
        fprintf(stderr, "qrt: --require-tensor needs --gguf-index <path>\n");
        return 2;
    }

    if (options.command == QRT_CMD_VK_IO) {
        return qrt_vk_io_check(&cfg, options.repack_prefix);
    }
    if (options.dense_gguf != NULL) {
        return qrt_dense_check(&cfg, options.dense_gguf);
    }
    if (options.prefill_prefix != NULL) {
        const char * shader_path = options.prefill_shader ? options.prefill_shader : g_spv_dir;
        return qrt_prefill_check(&cfg, options.prefill_prefix, shader_path,
                                 (uint32_t) options.prefill_slots, (uint32_t) options.prefill_layers,
                                 options.prefill_compute_moe, options.prefill_layer_batch, options.prefill_pipeline,
                                 options.prefill_group_kib, options.prefill_readers, options.prefill_chunks);
    }
    if (options.layer_gguf_check != NULL) {
        qrt_layer_cfg lcfg;
        memset(&lcfg, 0, sizeof(lcfg));
        lcfg.ffn = options.layer_ffn && options.layer_store != NULL ? 1 : 0;
        lcfg.single_submit = options.layer_pass;
        lcfg.dn_recur = options.dn_recur;
        lcfg.kv_k = cfg.kv_k;
        lcfg.kv_v = cfg.kv_v;
        lcfg.kv_page_tokens = (uint32_t) cfg.kv_page_tokens;
        return qrt_layer_check(options.layer_gguf_check, options.layer_store, (uint32_t) options.layer_index,
                               (uint32_t) options.layer_tokens, &lcfg, g_spv_dir);
    }
    if (options.norm_gguf_check != NULL) {
        return qrt_norm_check(&cfg, options.norm_gguf_check, options.norm_tensor, options.dense_tensor,
                              (uint32_t) options.dense_tokens, g_spv_dir);
    }
    if (options.dense_gguf_check != NULL) {
        return qrt_dense_gemm_check(&cfg, options.dense_gguf_check, options.dense_tensor, (uint32_t) options.dense_tokens,
                                    (uint32_t) options.dense_rows);
    }
    if (options.deltanet_gguf != NULL) {
        return qrt_dn_check(options.deltanet_gguf, (uint32_t) options.deltanet_layer, (uint32_t) options.deltanet_tokens,
                            g_spv_dir);
    }
    if (options.command == QRT_CMD_GENERATE) {
        qrt_model_options mo{};
        mo.n_layers = (uint32_t) options.logits_layers;
        mo.n_rows = 1;
        mo.batch_k = (uint32_t) options.generate_batch;
        uint32_t * ids = NULL;
        uint32_t n_ids = 0;
        if (qrt_model_read_ids(options.logits_tokens, &ids, &n_ids, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        const int rc = qrt_model_generate(&cfg, &mo, options.generate_gguf, options.layer_store,
                                          g_spv_dir, ids, n_ids, (uint32_t) options.generate_new,
                                          options.generate_out, err, sizeof(err));
        free(ids);
        if (rc != 0 && err[0] != '\0') {
            fprintf(stderr, "qrt: %s\n", err);
        }
        return rc < 0 ? 2 : rc;
    }
    if (options.command == QRT_CMD_CHUNK_CHECK || options.command == QRT_CMD_MODEL_CHECK ||
        options.command == QRT_CMD_LOGITS) {
        qrt_model_options model_options;
        model_options.n_layers = (uint32_t) options.logits_layers;
        model_options.n_rows = (uint32_t) options.logits_rows;
        model_options.out_path = options.logits_out;
        model_options.ref_out_path = options.model_check_ref;
        model_options.reseed = options.model_check_reseed;
        model_options.device_planes = options.model_check_planes;
        uint32_t * ids = NULL;
        uint32_t n_ids = 0;
        if (qrt_model_read_ids(options.logits_tokens, &ids, &n_ids, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        const auto run = options.command == QRT_CMD_CHUNK_CHECK ? qrt_model_chunk_check :
                         options.command == QRT_CMD_MODEL_CHECK ? qrt_model_check : qrt_model_logits;
        const char * path = options.command == QRT_CMD_CHUNK_CHECK ? options.chunk_check_gguf :
                            options.command == QRT_CMD_MODEL_CHECK ? options.model_check_gguf : options.logits_gguf;
        const int rc = run(&cfg, &model_options, path, options.layer_store, g_spv_dir, ids,
                           n_ids, err, sizeof(err));
        free(ids);
        if (rc != 0 && err[0] != '\0') {
            fprintf(stderr, "qrt: %s\n", err);
        }
        return rc < 0 ? 2 : rc;
    }
    if (options.fullattn_gguf != NULL) {
        return qrt_full_check(options.fullattn_gguf, (uint32_t) options.fullattn_layer, (uint32_t) options.fullattn_tokens,
                              g_spv_dir, cfg.kv_k, cfg.kv_v, (uint32_t) options.fullattn_resident,
                              (uint32_t) options.fullattn_page);
    }
    if (options.moe_prefix != NULL) {
        std::string dir = argc > 0 ? argv[0] : "qrt";
        const size_t cut = dir.find_last_of("/\\");
        dir = (cut == std::string::npos ? std::string(".") : dir.substr(0, cut));
        return qrt_moe_check(&cfg, options.moe_prefix, (options.prefill_shader != NULL ? options.prefill_shader : dir.c_str()),
                             (uint32_t) options.moe_layer, (uint32_t) options.moe_expert, (uint32_t) options.moe_rows);
    }
    if (options.command == QRT_CMD_VK_INFO) {
        qrt_vk_dev * dev = NULL;
        if (qrt_vk_open((uint32_t) cfg.vram_tier_mib, &dev, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        qrt_vk_print_info(dev, stdout);
        int rc = 0;
        // RESULTS section 25: the routed_experts total of the target model.
        const uint64_t expert_store_bytes = 20122173440ull;
        qrt_vk_plan plan;
        if (qrt_vk_plan_budget(dev, &cfg, expert_store_bytes, &plan, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
        } else {
            printf("plan dense_mib=%llu cache_mib=%llu kv_mib=%llu compute_mib=%llu "
                   "draft_mib=%llu total_mib=%llu kv_tokens=%u ctx_tokens=%u "
                   "coverage_pct=%.2f reductions=%u\n",
                   (unsigned long long) (plan.dense_bytes >> 20),
                   (unsigned long long) (plan.expert_cache_bytes >> 20),
                   (unsigned long long) (plan.kv_bytes >> 20),
                   (unsigned long long) (plan.compute_bytes >> 20),
                   (unsigned long long) (plan.draft_bytes >> 20),
                   (unsigned long long) (plan.total_bytes >> 20),
                   plan.kv_tokens, (unsigned) QRT_KV_CTX_TOKENS,
                   (double) plan.cache_coverage_x100 / 100.0, plan.cache_reductions);
        }
        if (options.vk_probe_mib > 0) {
            const uint64_t before = dev->budget_bytes;
            VkBuffer buf = 0;
            VkDeviceMemory mem = 0;
            if (qrt_vk_alloc(dev, (uint64_t) options.vk_probe_mib << 20, QRT_VK_USAGE_STORAGE,
                             &buf, &mem, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                rc = 2;
            } else {
                qrt_vk_query_budget(dev, err, sizeof(err));
                printf("probe alloc_mib=%d budget_before_mib=%llu budget_after_mib=%llu\n",
                       options.vk_probe_mib, (unsigned long long) (before >> 20),
                       (unsigned long long) (dev->budget_bytes >> 20));
                qrt_vk_free(dev, buf, mem);
            }
        }
        qrt_vk_close(dev);
        return rc;
    }

    fprintf(stderr, "qrt: unsupported command\n");
    return 2;
}
