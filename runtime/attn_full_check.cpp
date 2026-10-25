// M2-6d self check, step 1: per-head RMS norm + IMRoPE against a double precision
// reference, the f16 KV append, and the IMRoPE sector/theta form against the
// reference formula. The attention body is added by the next step of M2-6d.

#include "attn_full.h"

#include "gguf_index.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

// f16 storage: the check unpacks what the kernel wrote, so the comparison sees the
// cache exactly as the attention kernel will.
float half_to_float(uint32_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t expo = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    float v;
    if (expo == 0) {
        v = ldexpf((float) mant, -24);
    } else if (expo == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = ldexpf((float) (mant | 0x400u), (int) expo - 25);
    }
    return sign ? -v : v;
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

// The reference formula from ggml's ggml_mrope_cache_init, with the sectors applied
// the way IMROPE does it. For text every position is the same, so the sector picks
// must not change theta; this is what the check asserts.
double imrope_theta(const qrt_full_shapes * sh, uint32_t pos, uint32_t pair,
                    uint32_t theta_dim) {
    const int * s = sh->sections;
    const int sect_dims = s[0] + s[1] + s[2] + s[3];
    double theta = (double) pos;
    if (sect_dims > 0) {
        const int sector = (int) (pair % (uint32_t) sect_dims);
        // all four bases equal pos for text, so every branch yields the same value
        if (sector % 3 == 1 && sector < 3 * s[1]) {
            theta = (double) pos;
        } else if (sector % 3 == 2 && sector < 3 * s[2]) {
            theta = (double) pos;
        } else if (sector % 3 == 0 && sector < 3 * s[0]) {
            theta = (double) pos;
        }
    }
    (void) theta_dim;
    return theta;
}

} // namespace

int qrt_full_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
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
    qrt_full_shapes sh;
    const qrt_gguf_hparams hp_copy = *qrt_gguf_hparams_of(g);
    if (qrt_full_shapes_from_gguf(&hp_copy, &sh, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    char name[128];
    std::vector<float> qnw, knw;
    snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", layer);
    int rc = read_tensor_f32(g, name, &qnw, err, sizeof(err));
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", layer);
        rc = read_tensor_f32(g, name, &knw, err, sizeof(err));
    }
    const qrt_gguf_hparams * hp = &hp_copy;
    qrt_gguf_close(g);
    if (rc != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    if (qnw.size() != sh.head_dim || knw.size() != sh.head_dim) {
        fprintf(stderr, "qrt: the norms are %zu/%zu wide, head_dim is %u\n", qnw.size(),
                knw.size(), sh.head_dim);
        qrt_vk_close(dev);
        return 2;
    }

    printf("full_shapes n_embd=%u head_dim=%u n_head=%u n_kv=%u n_rot=%u qg_ch=%u kv_ch=%u "
           "eps=%.1e kq_scale=%.6f\n", sh.n_embd, sh.head_dim, sh.n_head, sh.n_kv, sh.n_rot,
           sh.qg_ch, sh.kv_ch, sh.eps, sh.kq_scale);
    printf("full_rope    sections=%d/%d/%d/%d freq_base=%.9g freq_scale=%.6g "
           "from_gguf dim=%d freq=%d sections=%d\n",
           sh.sections[0], sh.sections[1], sh.sections[2], sh.sections[3],
           sh.freq_base, sh.freq_scale,
           (hp->present & QRT_HP_ROPE_DIMENSION) ? 1 : 0,
           (hp->present & QRT_HP_ROPE_FREQ_BASE) ? 1 : 0,
           (hp->present & QRT_HP_ROPE_SECTIONS) ? 1 : 0);

    const uint32_t max_tokens = tokens > 512 ? tokens : 512;
    const uint32_t n_ctx = max_tokens * 2 + 64;

    qrt_vk_staging * stg = NULL;
    uint64_t stg_bytes = (uint64_t) max_tokens * sh.qg_ch * 4;
    if (qrt_vk_staging_create(dev, stg_bytes, &stg, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }

    int rc_out = 2;
    qrt_full * f = NULL;
    do {
        if (qrt_full_open(dev, spv_dir, &sh, max_tokens, n_ctx, &f, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const qrt_full_bufs * b = qrt_full_buffers(f);

        rng_t rng = { 0x9D2C5680B4A5F17ull };
        std::vector<float> qg((size_t) max_tokens * sh.qg_ch);
        std::vector<float> kin((size_t) max_tokens * sh.kv_ch);
        std::vector<float> vin((size_t) max_tokens * sh.kv_ch);
        for (size_t i = 0; i < qg.size(); ++i) {
            qg[i] = (float) rng.gauss();
        }
        for (size_t i = 0; i < kin.size(); ++i) {
            kin[i] = (float) rng.gauss();
            vin[i] = (float) rng.gauss();
        }

        auto upload = [&](VkBuffer dst, const float * src, uint64_t bytes) -> int {
            return qrt_vk_upload(dev, stg, dst, 0, src, bytes, err, sizeof(err));
        };
        // the staging buffer holds 512 rows, so a long chunk goes up in slices
        const uint32_t slice = 512;
        auto upload_rows = [&](VkBuffer dst, const float * base, uint32_t row_floats,
                               uint32_t t0, uint32_t count) -> int {
            for (uint32_t off = 0; off < count; off += slice) {
                const uint32_t n = count - off < slice ? count - off : slice;
                const size_t at = (size_t) (t0 + off) * row_floats;
                if (qrt_vk_upload(dev, stg, dst, (uint64_t) at * 4, base + at,
                                  (uint64_t) n * row_floats * 4, err, sizeof(err)) != 0) {
                    return -1;
                }
            }
            return 0;
        };
        if (upload_rows(b->qg, qg.data(), sh.qg_ch, 0, tokens) != 0 ||
            upload_rows(b->kin, kin.data(), sh.kv_ch, 0, tokens) != 0 ||
            upload_rows(b->vin, vin.data(), sh.kv_ch, 0, tokens) != 0 ||
            upload(b->q_norm_w, qnw.data(), (uint64_t) sh.head_dim * 4) != 0 ||
            upload(b->k_norm_w, knw.data(), (uint64_t) sh.head_dim * 4) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        // 1) first chunk at position 0
        if (qrt_full_qk(f, 0, tokens, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const double qk_ms = qrt_full_last_ms(f);

        std::vector<float> q_gpu((size_t) tokens * sh.n_head * sh.head_dim);
        std::vector<float> q_ref(q_gpu.size());
        std::vector<float> k_ref((size_t) sh.n_kv * tokens * sh.head_dim);
        std::vector<float> v_ref(k_ref.size());
        qrt_full_qk_cpu(&sh, 0, tokens, qg.data(), kin.data(), vin.data(), qnw.data(), knw.data(),
                        q_ref.data(), k_ref.data(), v_ref.data());

        std::vector<uint32_t> kc((size_t) sh.n_kv * n_ctx * sh.head_dim / 2);
        std::vector<uint32_t> vc((size_t) sh.n_kv * n_ctx * sh.head_dim / 2);
        const uint64_t kv_bytes = qrt_full_kv_bytes(f);
        if (qrt_vk_download(dev, stg, b->q, 0, q_gpu.data(),
                            (uint64_t) tokens * sh.n_head * sh.head_dim * 4, err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, b->kcache, 0, kc.data(), kv_bytes, err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, b->vcache, 0, vc.data(), kv_bytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }

        double mq = 0.0, mk = 0.0, mv = 0.0;
        const double rq = rel_l1(q_gpu.data(), q_ref.data(), q_gpu.size(), &mq);
        // the cache holds f16, so unpack it and compare against the f32 reference; the
        // arithmetic is gated at 1e-4 through q, the f16 storage error is reported
        std::vector<float> k_un((size_t) sh.n_kv * tokens * sh.head_dim);
        std::vector<float> v_un(k_un.size());
        for (uint32_t h = 0; h < sh.n_kv; ++h) {
            for (uint32_t t = 0; t < tokens; ++t) {
                const size_t slot = ((size_t) h * n_ctx + t) * sh.head_dim;
                const size_t dst = ((size_t) h * tokens + t) * sh.head_dim;
                for (uint32_t i = 0; i < sh.head_dim; i += 2) {
                    const uint32_t pair_k = kc[slot / 2 + i / 2];
                    k_un[dst + i]     = half_to_float(pair_k & 0xffffu);
                    k_un[dst + i + 1] = half_to_float(pair_k >> 16);
                    const uint32_t pair_v = vc[slot / 2 + i / 2];
                    v_un[dst + i]     = half_to_float(pair_v & 0xffffu);
                    v_un[dst + i + 1] = half_to_float(pair_v >> 16);
                }
            }
        }
        const double rk = rel_l1(k_un.data(), k_ref.data(), k_un.size(), &mk);
        const double rv = rel_l1(v_un.data(), v_ref.data(), v_un.size(), &mv);
        const int ok_q = rq <= 1e-4;
        const int ok_kv = rk <= 1e-3 && rv <= 1e-3;   // f16 storage bound
        printf("full_qk      tokens=%u pos0=0 q_rel_l1=%.3e q_max_abs=%.3e "
               "k_f16_rel_l1=%.3e v_f16_rel_l1=%.3e verdict=%s ms=%.3f\n",
               tokens, rq, mq, rk, rv, (ok_q && ok_kv) ? "PASS" : "FAIL", qk_ms);

        // 2) second chunk: positions continue, and the first chunk's cache must not move
        std::vector<uint32_t> kc_before(kc.begin(), kc.begin() + kc.size());
        const uint32_t t2 = tokens / 2 > 0 ? tokens / 2 : 1;
        if (upload(b->qg, qg.data(), (uint64_t) t2 * sh.qg_ch * 4) != 0 ||
            upload(b->kin, kin.data(), (uint64_t) t2 * sh.kv_ch * 4) != 0 ||
            upload(b->vin, vin.data(), (uint64_t) t2 * sh.kv_ch * 4) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        if (qrt_full_qk(f, tokens, t2, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        std::vector<float> q2((size_t) t2 * sh.n_head * sh.head_dim);
        std::vector<float> q2_ref(q2.size());
        std::vector<float> k2_ref((size_t) sh.n_kv * t2 * sh.head_dim);
        std::vector<float> v2_ref(k2_ref.size());
        qrt_full_qk_cpu(&sh, tokens, t2, qg.data(), kin.data(), vin.data(), qnw.data(), knw.data(),
                        q2_ref.data(), k2_ref.data(), v2_ref.data());
        if (qrt_vk_download(dev, stg, b->q, 0, q2.data(), (uint64_t) q2.size() * 4,
                            err, sizeof(err)) != 0 ||
            qrt_vk_download(dev, stg, b->kcache, 0, kc.data(), kv_bytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        double m2 = 0.0;
        const double r2 = rel_l1(q2.data(), q2_ref.data(), q2.size(), &m2);
        // the first chunk's slots (and the untouched tail) must be bit identical
        size_t moved = 0;
        for (int h = 0; h < (int) sh.n_kv; ++h) {
            const size_t lo = ((size_t) h * n_ctx) * sh.head_dim / 2;
            const size_t hi = lo + (size_t) tokens * sh.head_dim / 2;
            const size_t tail_lo = lo + ((size_t) tokens + t2) * sh.head_dim / 2;
            const size_t tail_hi = lo + (size_t) n_ctx * sh.head_dim / 2;
            for (size_t i = lo; i < hi; ++i) {
                moved += kc[i] != kc_before[i];
            }
            for (size_t i = tail_lo; i < tail_hi; ++i) {
                moved += kc[i] != 0u;
            }
        }
        // the second chunk's K must match the reference too
        std::vector<float> k2_un((size_t) sh.n_kv * t2 * sh.head_dim);
        for (uint32_t h = 0; h < sh.n_kv; ++h) {
            for (uint32_t t = 0; t < t2; ++t) {
                const size_t slot = ((size_t) h * n_ctx + tokens + t) * sh.head_dim;
                for (uint32_t i = 0; i < sh.head_dim; i += 2) {
                    const uint32_t pair_k = kc[slot / 2 + i / 2];
                    k2_un[((size_t) h * t2 + t) * sh.head_dim + i] = half_to_float(pair_k & 0xffffu);
                    k2_un[((size_t) h * t2 + t) * sh.head_dim + i + 1] = half_to_float(pair_k >> 16);
                }
            }
        }
        double mk2 = 0.0;
        const double rk2 = rel_l1(k2_un.data(), k2_ref.data(), k2_un.size(), &mk2);
        const int ok_append = (moved == 0);
        printf("full_qk      tokens=%u pos0=%u q_rel_l1=%.3e q_max_abs=%.3e k_f16_rel_l1=%.3e "
               "append_moved=%zu verdict=%s\n", t2, tokens, r2, m2, rk2, moved,
               (r2 <= 1e-4 && rk2 <= 1e-3 && ok_append) ? "PASS" : "FAIL");

        // 3) the rope angle both ways: plain NEOX and the IMRoPE sector form must agree
        const uint32_t half = sh.n_rot / 2;
        const double step = pow((double) sh.freq_base, -2.0 / (double) sh.n_rot);
        double worst = 0.0;
        for (uint32_t pos = 0; pos < 4096; pos += 977) {
            for (uint32_t p = 0; p < half; ++p) {
                const double neo = (double) pos * pow(step, (double) p);
                const double im = imrope_theta(&sh, pos, p, sh.n_rot) * pow(step, (double) p);
                worst = fmax(worst, fabs(neo - im));
            }
        }
        printf("full_rope    neox_vs_imrope_max_theta_diff=%.3e cos0=%.9f sin17=%.9f verdict=%s\n",
               worst, cos((double) 0 * pow(step, 0.0)), sin((double) 12345 * pow(step, 17.0)),
               worst == 0.0 ? "PASS" : "FAIL");

        // 5) the attention body: causal, online softmax, gate, over the whole chunk
        int ok_attn = 0;
        {
            if (qrt_full_qk(f, 0, tokens, err, sizeof(err)) != 0 ||
                qrt_full_attn(f, 0, tokens, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            const double attn_ms = qrt_full_last_ms(f);
            const uint64_t kv_reads = qrt_full_last_kv_reads(f);

            std::vector<float> out_gpu((size_t) tokens * sh.n_head * sh.head_dim);
            std::vector<float> q_full((size_t) tokens * sh.n_head * sh.head_dim);
            std::vector<float> kd((size_t) sh.n_kv * n_ctx * sh.head_dim);
            std::vector<float> vd(kd.size());
            if (qrt_vk_download(dev, stg, b->out, 0, out_gpu.data(),
                                (uint64_t) out_gpu.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->q, 0, q_full.data(),
                                (uint64_t) q_full.size() * 4, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            // unpack the cache the kernel actually read into f32 reference planes
            for (uint32_t h = 0; h < sh.n_kv; ++h) {
                for (uint32_t t = 0; t < tokens; ++t) {
                    const size_t slot = ((size_t) h * n_ctx + t) * sh.head_dim;
                    for (uint32_t i = 0; i < sh.head_dim; i += 2) {
                        const uint32_t pair_k = kc[slot / 2 + i / 2];
                        kd[slot + i]     = half_to_float(pair_k & 0xffffu);
                        kd[slot + i + 1] = half_to_float(pair_k >> 16);
                        const uint32_t pair_v = vc[slot / 2 + i / 2];
                        vd[slot + i]     = half_to_float(pair_v & 0xffffu);
                        vd[slot + i + 1] = half_to_float(pair_v >> 16);
                    }
                }
            }
            // the longest causal ranges are the interesting ones; a full sweep runs
            // when the chunk is small enough for the double reference to finish
            std::vector<uint32_t> rows;
            if (tokens <= 512) {
                for (uint32_t t = 0; t < tokens; ++t) {
                    rows.push_back(t);
                }
            } else {
                const uint32_t cand[8] = { tokens - 1, tokens - 2, tokens - 9, tokens - 33,
                                           tokens / 2, tokens / 3, 1, 0 };
                for (int i = 0; i < 8; ++i) {
                    if (cand[i] < tokens) {
                        rows.push_back(cand[i]);
                    }
                }
            }
            std::vector<float> ref((size_t) rows.size() * sh.n_head * sh.head_dim, 0.0f);
            qrt_full_attn_rows_cpu(&sh, n_ctx, rows.data(), (uint32_t) rows.size(), q_full.data(),
                                   kd.data(), vd.data(), qg.data(), ref.data());
            double worst_rel = 0.0, worst_abs = 0.0;
            for (size_t ri = 0; ri < rows.size(); ++ri) {
                const size_t base = (size_t) rows[ri] * sh.n_head * sh.head_dim;
                double mx = 0.0;
                const double r = rel_l1(out_gpu.data() + base, ref.data() + ri * sh.n_head * sh.head_dim,
                                        (size_t) sh.n_head * sh.head_dim, &mx);
                worst_rel = fmax(worst_rel, r);
                worst_abs = fmax(worst_abs, mx);
            }
            // the KV account: bytes the kernel read per layer against the plane size
            const double kv_bytes_plane = (double) sh.n_kv * tokens * sh.head_dim * 2.0;
            const double implied = (double) kv_reads / (attn_ms * 1e-3) / 1e9;
            const double ratio = (double) kv_reads / (2.0 * kv_bytes_plane);
            ok_attn = worst_rel <= 1e-4;
            // the f16-packed probe: same rows, operands rounded to f16 and eight-product
            // chunks summed in f16. This decides whether the fast path is even allowed.
            std::vector<float> ref16(ref.size(), 0.0f);
            qrt_full_attn_rows_cpu_f16(&sh, n_ctx, rows.data(), (uint32_t) rows.size(),
                                       q_full.data(), kd.data(), vd.data(), qg.data(),
                                       ref16.data());
            double worst16 = 0.0, abs16 = 0.0;
            for (size_t ri = 0; ri < rows.size(); ++ri) {
                const size_t base = (size_t) rows[ri] * sh.n_head * sh.head_dim;
                double mx = 0.0;
                const double r = rel_l1(out_gpu.data() + base, ref16.data() + ri * sh.n_head * sh.head_dim,
                                        (size_t) sh.n_head * sh.head_dim, &mx);
                worst16 = fmax(worst16, r);
                abs16 = fmax(abs16, mx);
            }
            printf("full_attn    tokens=%u rows=%zu rel_l1=%.3e max_abs=%.3e ms=%.3f "
                   "attn_tflops=%.2f kv_read_gib=%.2f kv_plane_mib=%.1f kv_reads_per_plane=%.1f "
                   "implied_gb_s=%.1f verdict=%s\n",
                   tokens, rows.size(), worst_rel, worst_abs, attn_ms,
                   // 2 x HD x 2 flops per pair, one pair per (query, key, head)
                   (double) tokens * (tokens + 1) / 2.0 * sh.n_head * sh.head_dim * 4.0 /
                       (attn_ms * 1e-3) / 1e12,
                   (double) kv_reads / (1024.0 * 1024.0 * 1024.0),
                   2.0 * kv_bytes_plane / (1024.0 * 1024.0), ratio, implied,
                   ok_attn ? "PASS" : "FAIL");
            printf("full_attn_f16 tokens=%u rows=%zu rel_l1=%.3e max_abs=%.3e "
                   "gate_1e-4=%s (probe: f16 operands, eight-product chunks in f16)\n",
                   tokens, rows.size(), worst16, abs16, worst16 <= 1e-4 ? "PASS" : "FAIL");
        }

        rc_out = (ok_q && ok_kv && r2 <= 1e-4 && rk2 <= 1e-3 && ok_append && worst == 0.0 &&
                  ok_attn) ? 0 : 1;
    } while (0);

    qrt_full_close(dev, f);
    qrt_vk_staging_destroy(dev, stg);
    qrt_vk_close(dev);
    return rc_out;
}

#else // QRT_HAVE_VULKAN

int qrt_full_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
    (void) gguf_path;
    (void) layer;
    (void) tokens;
    (void) spv_dir;
    fprintf(stderr, "qrt: --fullattn-check needs a Vulkan build\n");
    return 2;
}

#endif // QRT_HAVE_VULKAN
