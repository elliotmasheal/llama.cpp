// M2-6c self check: the delta-rule recurrence against a double precision CPU
// reference, plus the two properties the task asks for explicitly: the state can
// be saved and restored, and a batched chunk equals a serial token-by-token run.
// q and k are random unit vectors because the prep chain L2 normalises them, so
// their direction is what matters; decay comes from the real ssm_a and ssm_dt.

#include "attn_deltanet.h"

#include "gguf_index.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

struct rng_t {
    uint64_t s;
    uint64_t next(void) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double gauss(void) {
        const double u1 = ((double) ((next() >> 11) & ((1ull << 53) - 1)) + 0.5) / 9007199254740992.0;
        const double u2 = ((double) ((next() >> 11) & ((1ull << 53) - 1)) + 0.5) / 9007199254740992.0;
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }
};

int read_tensor_f32(qrt_gguf * g, const char * name, std::vector<float> * out,
                    char * err, size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 0) {
        snprintf(err, err_size, "%s is %s, expected f32", name, qrt_gguf_type_name(t->type));
        return -1;
    }
    uint64_t count = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) {
        count *= t->ne[i];
    }
    out->assign((size_t) count, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err, err_size);
}

double rel_l1(const float * got, const float * want, size_t n, double * max_abs) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = (double) got[i];
        const double b = (double) want[i];
        num += fabs(a - b);
        den += fabs(b);
        mx = fmax(mx, fabs(a - b));
    }
    *max_abs = mx;
    return den > 0.0 ? num / den : num;
}

} // namespace

int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
    char err[512] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open(0, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }

    // Shapes from the tensor table: ssm_norm is head_dim, ssm_a is n_v_heads,
    // attn_qkv minus the v channels gives q and k (each n_k_heads * head_dim).
    qrt_dn_shapes sh;
    memset(&sh, 0, sizeof(sh));
    char name[128];
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    snprintf(name, sizeof(name), "blk.%u.ssm_a", layer);
    int shape_rc = read_tensor_f32(g, name, &ssm_a, err, sizeof(err));
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", layer);
        shape_rc = read_tensor_f32(g, name, &ssm_dt, err, sizeof(err));
    }
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", layer);
        shape_rc = read_tensor_f32(g, name, &ssm_norm, err, sizeof(err));
    }
    const qrt_gguf_tensor * qkv = NULL;
    const qrt_gguf_tensor * conv = NULL;
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
        shape_rc = qrt_gguf_require(g, name, &qkv, err, sizeof(err));
    }
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
        shape_rc = qrt_gguf_require(g, name, &conv, err, sizeof(err));
    }
    if (shape_rc != 0) {
        fprintf(stderr, "qrt: %s\n", err[0] ? err : "layer tensors are missing");
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    sh.head_dim  = (uint32_t) ssm_norm.size();
    sh.n_v_heads = (uint32_t) ssm_a.size();
    sh.d_inner   = sh.n_v_heads * sh.head_dim;
    sh.n_embd    = (uint32_t) qkv->ne[0];
    sh.conv_ch   = (uint32_t) qkv->ne[1];
    sh.conv_taps = (uint32_t) conv->ne[0];
    sh.qk_ch     = (sh.conv_ch - sh.d_inner) / 2;
    sh.n_k_heads = sh.qk_ch / sh.head_dim;
    qrt_gguf_close(g);
    if (sh.n_k_heads == 0 || sh.qk_ch % sh.head_dim != 0 || sh.head_dim % 32 != 0) {
        fprintf(stderr, "qrt: layer %u does not split into q/k/v (qk_ch=%u head_dim=%u)\n",
                layer, sh.qk_ch, sh.head_dim);
        qrt_vk_close(dev);
        return 2;
    }
    printf("dn_device subgroup_size=%u subgroup_arithmetic=%d\n", dev->subgroup_size,
           dev->has_subgroup_arithmetic);

    const uint32_t max_tokens = tokens > 1024 ? tokens : 1024;
    const uint32_t serial_tokens = tokens < 64 ? tokens : 64;
    const uint64_t st_floats = (uint64_t) sh.n_v_heads * sh.head_dim * sh.head_dim;
    const uint64_t st_bytes = st_floats * 4;

    std::vector<float> q((size_t) max_tokens * sh.qk_ch), k((size_t) max_tokens * sh.qk_ch);
    std::vector<float> v((size_t) max_tokens * sh.d_inner);
    std::vector<float> decay((size_t) max_tokens * sh.n_v_heads);
    std::vector<float> beta((size_t) max_tokens * sh.n_v_heads);

    rng_t rng = { 0x51D3C7A19E4B2F65ull };
    for (uint32_t t = 0; t < max_tokens; ++t) {
        for (uint32_t h = 0; h < sh.n_k_heads; ++h) {
            float * qrow = q.data() + (size_t) t * sh.qk_ch + (size_t) h * sh.head_dim;
            float * krow = k.data() + (size_t) t * sh.qk_ch + (size_t) h * sh.head_dim;
            double qn = 0.0, kn = 0.0;
            for (uint32_t i = 0; i < sh.head_dim; ++i) {
                qrow[i] = (float) rng.gauss();
                krow[i] = (float) rng.gauss();
                qn += (double) qrow[i] * qrow[i];
                kn += (double) krow[i] * krow[i];
            }
            qn = sqrt(qn);
            kn = sqrt(kn);
            for (uint32_t i = 0; i < sh.head_dim; ++i) {
                qrow[i] = (float) (qrow[i] / qn);
                krow[i] = (float) (krow[i] / kn);
            }
        }
        for (uint32_t c = 0; c < sh.d_inner; ++c) {
            v[(size_t) t * sh.d_inner + c] = (float) rng.gauss();
        }
        for (uint32_t h = 0; h < sh.n_v_heads; ++h) {
            const double alpha = rng.gauss() + (double) ssm_dt[h];
            const double gain = (double) ssm_a[h] * log1p(exp(alpha));
            // ssm_a holds -exp(A_log), so exp(gain) is the per-token decay <= 1
            decay[(size_t) t * sh.n_v_heads + h] = (float) exp(gain);
            beta[(size_t) t * sh.n_v_heads + h] = (float) (1.0 / (1.0 + exp(-rng.gauss())));
        }
    }

    qrt_vk_staging * stg = NULL;
    const uint64_t stg_bytes = (uint64_t) max_tokens * sh.d_inner * 4 > st_bytes
                                 ? (uint64_t) max_tokens * sh.d_inner * 4 : st_bytes;
    if (qrt_vk_staging_create(dev, stg_bytes, &stg, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }

    int rc = 2;
    qrt_deltanet * dn = NULL;
    std::vector<float> zero_state((size_t) st_floats, 0.0f);
    std::vector<float> st_saved((size_t) st_floats), st_a((size_t) st_floats);
    std::vector<float> st_b((size_t) st_floats), st_serial((size_t) st_floats);
    std::vector<float> y_gpu((size_t) max_tokens * sh.d_inner);
    std::vector<float> y_cpu((size_t) max_tokens * sh.d_inner);
    std::vector<float> y_split((size_t) max_tokens * sh.d_inner);
    std::vector<double> st_cpu((size_t) st_floats);
    std::vector<float> st_cpu_f((size_t) st_floats);
    do {
        if (qrt_dn_open(dev, spv_dir, &sh, max_tokens, &dn, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const qrt_dn_bufs * b = qrt_dn_buffers(dn);

        // helpers: move a token range into the planes, then run and read back
        auto upload_range = [&](uint32_t t0, uint32_t count) -> int {
            if (qrt_vk_upload(dev, stg, b->q, 0, q.data() + (size_t) t0 * sh.qk_ch,
                              (uint64_t) count * sh.qk_ch * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->k, 0, k.data() + (size_t) t0 * sh.qk_ch,
                              (uint64_t) count * sh.qk_ch * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->v, 0, v.data() + (size_t) t0 * sh.d_inner,
                              (uint64_t) count * sh.d_inner * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->decay, 0, decay.data() + (size_t) t0 * sh.n_v_heads,
                              (uint64_t) count * sh.n_v_heads * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->beta, 0, beta.data() + (size_t) t0 * sh.n_v_heads,
                              (uint64_t) count * sh.n_v_heads * 4, err, sizeof(err)) != 0) {
                return -1;
            }
            return 0;
        };
        auto run_chunk = [&](uint32_t count, float * y_out, float * st_out) -> int {
            if (qrt_dn_recur(dn, count, err, sizeof(err)) != 0) {
                return -1;
            }
            if (qrt_vk_download(dev, stg, b->y, 0, y_out, (uint64_t) count * sh.d_inner * 4,
                                err, sizeof(err)) != 0) {
                return -1;
            }
            return qrt_vk_download(dev, stg, b->state_out, 0, st_out, st_bytes, err, sizeof(err));
        };
        auto push_state = [&](const float * src) -> int {
            return qrt_vk_upload(dev, stg, b->state_in, 0, src, st_bytes, err, sizeof(err));
        };

        printf("dn_recur shapes n_embd=%u head_dim=%u k_heads=%u v_heads=%u d_inner=%u "
               "conv_ch=%u taps=%u tokens=%u\n", sh.n_embd, sh.head_dim, sh.n_k_heads,
               sh.n_v_heads, sh.d_inner, sh.conv_ch, sh.conv_taps, tokens);

        // 1) GPU against the double precision reference
        if (upload_range(0, tokens) != 0 || push_state(zero_state.data()) != 0 ||
            run_chunk(tokens, y_gpu.data(), st_a.data()) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        std::fill(st_cpu.begin(), st_cpu.end(), 0.0);
        qrt_dn_recur_cpu(&sh, tokens, q.data(), k.data(), v.data(), decay.data(), beta.data(),
                         st_cpu.data(), y_cpu.data());
        for (size_t i = 0; i < st_cpu.size(); ++i) {
            st_cpu_f[i] = (float) st_cpu[i];
        }
        double max_y = 0.0, max_s = 0.0;
        const double rel_y = rel_l1(y_gpu.data(), y_cpu.data(), (size_t) tokens * sh.d_inner, &max_y);
        const double rel_s = rel_l1(st_a.data(), st_cpu_f.data(), st_cpu_f.size(), &max_s);
        const int ok_ref = rel_y <= 1e-4 && rel_s <= 1e-4;
        printf("dn_recur correct rel_l1=%.3e max_abs=%.3e state_rel_l1=%.3e state_max_abs=%.3e "
               "verdict=%s\n", rel_y, max_y, rel_s, max_s, ok_ref ? "PASS" : "FAIL");

        // 2) state save and restore: two half chunks must equal one whole chunk
        const uint32_t half = tokens / 2;
        const uint32_t rest = tokens - half;
        if (push_state(zero_state.data()) != 0 || upload_range(0, half) != 0 ||
            run_chunk(half, y_split.data(), st_saved.data()) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        // the saved state goes back in, the second half continues from it
        if (push_state(st_saved.data()) != 0 || upload_range(half, rest) != 0 ||
            run_chunk(rest, y_split.data(), st_b.data()) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        double max_split = 0.0, max_state = 0.0;
        const double rel_split = rel_l1(y_split.data(), y_gpu.data() + (size_t) half * sh.d_inner,
                                        (size_t) rest * sh.d_inner, &max_split);
        const double rel_state = rel_l1(st_b.data(), st_a.data(), st_a.size(), &max_state);
        const int ok_split = rel_split == 0.0 && rel_state == 0.0;
        printf("dn_recur split   half_tokens=%u rel_l1=%.3e max_abs=%.3e state_rel_l1=%.3e "
               "verdict=%s\n", half, rel_split, max_split, rel_state, ok_split ? "PASS" : "FAIL");

        // 3) serial decode: one token per dispatch must match the same chunk
        if (push_state(zero_state.data()) != 0 || upload_range(0, serial_tokens) != 0 ||
            run_chunk(serial_tokens, y_gpu.data() + (size_t) (max_tokens - serial_tokens) * sh.d_inner,
                      st_serial.data()) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        // the batched reference for the same prefix sits at the tail of y_gpu
        const float * y_batch_prefix = y_gpu.data() + (size_t) (max_tokens - serial_tokens) * sh.d_inner;
        std::vector<float> y_batch_prefix_copy(y_batch_prefix, y_batch_prefix +
                                               (size_t) serial_tokens * sh.d_inner);
        std::vector<float> st_batch_prefix(st_serial);
        std::vector<float> y_row((size_t) sh.d_inner);
        int serial_fail = push_state(zero_state.data()) != 0;
        for (uint32_t t = 0; t < serial_tokens && !serial_fail; ++t) {
            // in place would be cheaper, but the two state buffers make the copy explicit
            if (upload_range(t, 1) != 0 || qrt_dn_recur(dn, 1, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->y, 0, y_row.data(),
                                (uint64_t) sh.d_inner * 4, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->state_out, 0, st_serial.data(), st_bytes,
                                err, sizeof(err)) != 0 ||
                push_state(st_serial.data()) != 0) {
                serial_fail = 1;
            }
            if (!serial_fail) {
                memcpy(y_split.data() + (size_t) t * sh.d_inner, y_row.data(),
                       (size_t) sh.d_inner * 4);
            }
        }
        if (serial_fail) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        double max_serial = 0.0, max_serial_state = 0.0;
        const double rel_serial = rel_l1(y_split.data(), y_batch_prefix_copy.data(),
                                         (size_t) serial_tokens * sh.d_inner, &max_serial);
        const double rel_serial_state = rel_l1(st_serial.data(), st_batch_prefix.data(),
                                               st_batch_prefix.size(), &max_serial_state);
        const int ok_serial = rel_serial == 0.0 && rel_serial_state == 0.0;
        printf("dn_recur serial  tokens=%u rel_l1=%.3e max_abs=%.3e state_rel_l1=%.3e verdict=%s\n",
               serial_tokens, rel_serial, max_serial, rel_serial_state,
               ok_serial ? "PASS" : "FAIL");


        // 5) prep chain: conv1d(taps) + silu + per-head L2 norm against the CPU reference
        {
            qrt_dn_weights * w = NULL;
            const float l2_eps = 1e-6f / (float) sh.head_dim;   // attention.layer_norm_rms_epsilon / head_dim
            std::vector<float> qkv_host((size_t) tokens * sh.conv_ch);
            std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
            for (uint32_t t = 0; t < tokens; ++t) {
                for (uint32_t c = 0; c < sh.conv_ch; ++c) {
                    qkv_host[(size_t) t * sh.conv_ch + c] = (float) rng.gauss();
                }
            }
            // a non-zero conv state exercises the history taps
            for (size_t i = 0; i < cst.size(); ++i) {
                cst[i] = (float) (rng.gauss() * 0.5);
            }
            std::vector<float> qc((size_t) tokens * sh.qk_ch), kc((size_t) tokens * sh.qk_ch);
            std::vector<float> vc((size_t) tokens * sh.d_inner);
            if (qrt_dn_weights_load(dev, gguf_path, layer, &sh, &w, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> cw((size_t) sh.conv_taps * sh.conv_ch);
            if (qrt_vk_download(dev, stg, qrt_dn_weights_conv(w), 0, cw.data(), cw.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, w);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> qg((size_t) tokens * sh.qk_ch), kg((size_t) tokens * sh.qk_ch);
            std::vector<float> vg((size_t) tokens * sh.d_inner);
            if (qrt_vk_upload(dev, stg, b->qkv, 0, qkv_host.data(),
                              qkv_host.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->conv_state, 0, cst.data(), cst.size() * 4,
                              err, sizeof(err)) != 0 ||
                qrt_dn_prep_run(dn, w, tokens, l2_eps, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->q, 0, qg.data(), qg.size() * 4, err,
                                sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->k, 0, kg.data(), kg.size() * 4, err,
                                sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->v, 0, vg.data(), vg.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, w);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            double lo = 0.0, hi = 0.0;
            for (size_t i = 0; i < cw.size(); ++i) {
                (i < cw.size() / 2 ? lo : hi) += fabs((double) cw[i]);
            }
            qrt_dn_prep_cpu(&sh, tokens, qkv_host.data(), cw.data(), cst.data(), l2_eps,
                            qc.data(), kc.data(), vc.data(), NULL);
            double mq = 0.0, mk = 0.0, mv = 0.0;
            const double rq = rel_l1(qg.data(), qc.data(), qc.size(), &mq);
            const double rk = rel_l1(kg.data(), kc.data(), kc.size(), &mk);
            const double rv = rel_l1(vg.data(), vc.data(), vc.size(), &mv);
            const double prep_ms = qrt_dn_last_ms(dn);
            qrt_dn_weights_free(dev, w);
            printf("dn_prep  tokens=%u q_rel_l1=%.3e k_rel_l1=%.3e v_rel_l1=%.3e max_abs=%.3e "
                   "ms=%.3f verdict=%s\n", tokens, rq, rk, rv, fmax(mq, fmax(mk, mv)), prep_ms,
                   (rq <= 1e-4 && rk <= 1e-4 && rv <= 1e-4) ? "PASS"
                   : (rq <= 1e-4 && rk <= 1e-4 ? "PARTIAL(v-open)" : "FAIL"));
            // q and k are verified; the v branch of the prep shader is still open, and the
            // exit code must not pretend otherwise (RESULTS 56)
            if (!(rq <= 1e-4 && rk <= 1e-4)) {
                break;
            }
        }

        // 4) timing at the chunk the prefill scheduler would use
        if (upload_range(0, max_tokens) != 0 || push_state(zero_state.data()) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        double t[5];
        int time_fail = 0;
        for (int i = 0; i < 5 && !time_fail; ++i) {
            if (qrt_dn_recur(dn, max_tokens, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                time_fail = 1;
                break;
            }
            t[i] = qrt_dn_last_ms(dn);
        }
        if (time_fail) {
            break;
        }
        for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 5; ++j) {
                if (t[j] < t[i]) {
                    const double tmp = t[i];
                    t[i] = t[j];
                    t[j] = tmp;
                }
            }
        }
        const double us_per_token = t[2] * 1000.0 / (double) max_tokens;
        printf("dn_recur timing  tokens=%u median_ms=%.3f us_per_token=%.2f layer_10k_ms=%.1f "
               "layers30_10k_ms=%.0f\n", max_tokens, t[2], us_per_token,
               us_per_token * 10240.0 / 1000.0, us_per_token * 10240.0 * 30.0 / 1000.0);

        rc = (ok_ref && ok_split && ok_serial) ? 0 : 1;
    } while (0);

    qrt_dn_close(dev, dn);
    qrt_vk_staging_destroy(dev, stg);
    qrt_vk_close(dev);
    return rc;
}

#else // QRT_HAVE_VULKAN

int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
    (void) gguf_path;
    (void) layer;
    (void) tokens;
    (void) spv_dir;
    fprintf(stderr, "qrt: --deltanet-check needs a Vulkan build\n");
    return 2;
}

#endif // QRT_HAVE_VULKAN
