// Host-side reference of the full-attention layer: per-head RMS norm + IMRoPE, the KV row
// formats and the causal attention with the sigmoid gate. It is a separate translation unit
// so a change to the kernel launches cannot quietly change what the reference assumes.

#include "attn_full.h"

#include "qrt_config.h"
#include "qrt_half.h"
#include "qrt_util.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

// ---------------------------------------------------------------- references

static void ref_norm_rope(const qrt_full_shapes * sh, const float * src, const float * nw,
                          uint32_t pos, double * dst) {
    const uint32_t hd = sh->head_dim;
    double ss = 0.0;
    for (uint32_t i = 0; i < hd; ++i) {
        ss += (double) src[i] * (double) src[i];
    }
    const double inv = 1.0 / sqrt(ss / (double) hd + (double) sh->eps);
    for (uint32_t i = 0; i < hd; ++i) {
        dst[i] = (double) src[i] * inv * (double) nw[i];
    }
    const uint32_t half = sh->n_rot / 2;
    const double step = pow((double) sh->freq_base, -2.0 / (double) sh->n_rot);
    for (uint32_t p = 0; p < half; ++p) {
        const double theta = (double) pos * (double) sh->freq_scale * pow(step, (double) p);
        const double c = cos(theta);
        const double s = sin(theta);
        const double x0 = dst[p];
        const double x1 = dst[p + half];
        dst[p]        = x0 * c - x1 * s;
        dst[p + half] = x0 * s + x1 * c;
    }
}

void qrt_full_qk_cpu(const qrt_full_shapes * sh, uint32_t pos0, uint32_t tokens,
                     const float * qg, const float * kin, const float * vin,
                     const float * q_norm_w, const float * k_norm_w,
                     float * q_out, float * k_out, float * v_out) {
    std::vector<double> tmp(sh->head_dim);
    const uint32_t hd = sh->head_dim;
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t pos = pos0 + t;
        for (uint32_t h = 0; h < sh->n_head; ++h) {
            const float * src = qg + (size_t) t * sh->qg_ch + (size_t) h * 2 * hd;
            ref_norm_rope(sh, src, q_norm_w, pos, tmp.data());
            float * dst = q_out + ((size_t) t * sh->n_head + h) * hd;
            for (uint32_t i = 0; i < hd; ++i) {
                dst[i] = (float) tmp[i];
            }
        }
        for (uint32_t h = 0; h < sh->n_kv; ++h) {
            const float * src = kin + (size_t) t * sh->kv_ch + (size_t) h * hd;
            ref_norm_rope(sh, src, k_norm_w, pos, tmp.data());
            float * dst = k_out + ((size_t) h * tokens + t) * hd;
            for (uint32_t i = 0; i < hd; ++i) {
                dst[i] = (float) tmp[i];
            }
            memcpy(v_out + ((size_t) h * tokens + t) * hd, vin + (size_t) t * sh->kv_ch + (size_t) h * hd,
                   (size_t) hd * sizeof(float));
        }
    }
}

static float as_f16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t expo = (int32_t) ((u >> 23) & 0xffu) - 127;
    const uint32_t man = u & 0x7fffffu;
    if (expo > 15) {
        const float big = sign ? -65504.0f : 65504.0f;
        return big;
    }
    if (expo < -24) {
        return sign ? -0.0f : 0.0f;
    }
    if (expo < -14) {
        // subnormal: shift the implicit leading one into place
        const uint32_t m = (man | 0x800000u) >> (uint32_t) (-expo - 14);
        const uint32_t bits = sign | (m >> 13);
        const uint32_t rem = m & 0x1fffu;
        uint32_t h = bits + ((rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) ? 1u : 0u);
        return f16_to_f32(h);
    }
    uint32_t h = sign | ((uint32_t) (expo + 15) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        h += 1u;
    }
    return f16_to_f32(h);
}

// packHalf2x16 on this driver lowers to v_cvt_pkrtz: the mantissa is truncated, not
// rounded. The dequantised KV tile is staged with it (it never leaves the kernel), so the
// check mirrors the same conversion.
static float as_f16_trunc(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t expo = (int32_t) ((u >> 23) & 0xffu) - 127;
    const uint32_t man = u & 0x7fffffu;
    if (expo > 15) {
        return sign ? -65504.0f : 65504.0f;
    }
    if (expo < -24) {
        return sign ? -0.0f : 0.0f;
    }
    if (expo < -14) {
        const uint32_t m = (man | 0x800000u) >> (uint32_t) (-expo - 14);
        return f16_to_f32(sign | (m >> 13));
    }
    return f16_to_f32(sign | ((uint32_t) (expo + 15) << 10) | (man >> 13));
}

void qrt_full_kv_unpack(const qrt_kv_row_layout * lay, uint32_t head_dim, uint32_t n_kv,
                        uint32_t n_ctx, uint32_t pos0, uint32_t tokens, uint32_t row_stride,
                        const void * plane, int round_mode, float * out) {
    const uint8_t * base = (const uint8_t *) plane;
    for (uint32_t h = 0; h < n_kv; ++h) {
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint8_t * row = base + ((size_t) h * n_ctx + pos0 + t) * lay->row_bytes;
            float * dst = out + ((size_t) h * row_stride + t) * head_dim;
            if (lay->type == QRT_KV_F16) {
                for (uint32_t i = 0; i < head_dim; ++i) {
                    const uint32_t bits = (uint32_t) row[2 * i] | ((uint32_t) row[2 * i + 1] << 8);
                    dst[i] = f16_to_f32(bits);
                }
                continue;
            }
            const uint8_t * qs = row + lay->data_offset;
            for (uint32_t b = 0; b < lay->blocks; ++b) {
                const uint32_t sbits = (uint32_t) row[2 * b] | ((uint32_t) row[2 * b + 1] << 8);
                const float d = f16_to_f32(sbits);
                for (uint32_t i = 0; i < 32; ++i) {
                    float v;
                    if (lay->type == QRT_KV_Q8_0) {
                        v = (float) ((int8_t) qs[b * 32 + i]) * d;
                    } else {
                        const uint8_t byte = qs[b * 16 + (i >> 1)];
                        const uint32_t q = (i & 1u) ? (uint32_t) (byte >> 4) : (uint32_t) (byte & 0xFu);
                        v = ((float) q - 8.0f) * d;
                    }
                    dst[b * 32 + i] = round_mode == 1 ? as_f16(v)
                                    : (round_mode == 2 ? as_f16_trunc(v) : v);
                }
            }
        }
    }
}

// Chunked f16 dot: eight products summed in f16, each chunk added to the f32 sum. That is
// the best a packed-half kernel (V_PK_FMA_F16 in f16, or f16 dot2) can do without f32
// accumulate, and it is what decides whether the numerics allow the fast path at all.
static float dot_chunk_f16(const float * a, const float * b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i += 8u) {
        float chunk = 0.0f;
        for (uint32_t j = 0; j < 8u; ++j) {
            chunk = as_f16(chunk + as_f16(a[i + j]) * as_f16(b[i + j]));
        }
        acc += chunk;
    }
    return acc;
}

static void attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx, const uint32_t * rows,
                          uint32_t n_rows, const float * q, const float * kcache,
                          const float * vcache, const float * qg, float * out, int f16) {
    const uint32_t hd = sh->head_dim;
    std::vector<double> acc(hd);
    std::vector<float> accf(hd);
    for (uint32_t ri = 0; ri < n_rows; ++ri) {
        const uint32_t t = rows[ri];
        for (uint32_t h = 0; h < sh->n_head; ++h) {
            const uint32_t kvh = h / (sh->n_head / sh->n_kv);
            const float * qrow = q + ((size_t) t * sh->n_head + h) * hd;
            // causal range is [0, t]
            std::vector<double> scores((size_t) t + 1);
            double mx = -1e30;
            for (uint32_t u = 0; u <= t; ++u) {
                const float * krow = kcache + ((size_t) kvh * n_ctx + u) * hd;
                if (f16) {
                    scores[u] = (double) dot_chunk_f16(qrow, krow, hd) * (double) sh->kq_scale;
                } else {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < hd; ++d) {
                        dot += (double) qrow[d] * (double) krow[d];
                    }
                    scores[u] = dot * (double) sh->kq_scale;
                }
                mx = fmax(mx, scores[u]);
            }
            double sum = 0.0;
            for (uint32_t u = 0; u <= t; ++u) {
                scores[u] = exp(scores[u] - mx);
                sum += scores[u];
            }
            for (uint32_t d = 0; d < hd; ++d) {
                acc[d] = 0.0;
                accf[d] = 0.0f;
            }
            for (uint32_t u = 0; u <= t; ++u) {
                const float * vrow = vcache + ((size_t) kvh * n_ctx + u) * hd;
                const double p = scores[u] / sum;
                if (f16) {
                    const float pf = as_f16((float) p);
                    for (uint32_t d = 0; d < hd; ++d) {
                        accf[d] += pf * as_f16(vrow[d]);
                    }
                } else {
                    for (uint32_t d = 0; d < hd; ++d) {
                        acc[d] += p * (double) vrow[d];
                    }
                }
            }
            float * dst = out + ((size_t) ri * sh->n_head + h) * hd;
            for (uint32_t d = 0; d < hd; ++d) {
                const uint32_t ch = h * 2 * hd + hd + d;
                const double g = 1.0 / (1.0 + exp(-(double) qg[(size_t) t * sh->qg_ch + ch]));
                dst[d] = (float) ((f16 ? (double) accf[d] : acc[d]) * g);
            }
        }
    }
}

void qrt_full_attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx,
                            const uint32_t * rows, uint32_t n_rows,
                            const float * q, const float * kcache, const float * vcache,
                            const float * qg, float * out) {
    attn_rows_cpu(sh, n_ctx, rows, n_rows, q, kcache, vcache, qg, out, 0);
}

void qrt_full_attn_rows_cpu_f16(const qrt_full_shapes * sh, uint32_t n_ctx,
                                const uint32_t * rows, uint32_t n_rows,
                                const float * q, const float * kcache, const float * vcache,
                                const float * qg, float * out) {
    attn_rows_cpu(sh, n_ctx, rows, n_rows, q, kcache, vcache, qg, out, 1);
}

#else // QRT_HAVE_VULKAN

void qrt_full_kv_unpack(const qrt_kv_row_layout * lay, uint32_t head_dim, uint32_t n_kv,
                        uint32_t n_ctx, uint32_t pos0, uint32_t tokens, uint32_t row_stride,
                        const void * plane, int round_mode, float * out) {
    (void) lay; (void) head_dim; (void) n_kv; (void) n_ctx; (void) pos0; (void) tokens;
    (void) row_stride; (void) plane; (void) round_mode; (void) out;
}

void qrt_full_qk_cpu(const qrt_full_shapes * sh, uint32_t pos0, uint32_t tokens,
                     const float * qg, const float * kin, const float * vin,
                     const float * q_norm_w, const float * k_norm_w,
                     float * q_out, float * k_out, float * v_out) {
    (void) sh; (void) pos0; (void) tokens; (void) qg; (void) kin; (void) vin;
    (void) q_norm_w; (void) k_norm_w; (void) q_out; (void) k_out; (void) v_out;
}
void qrt_full_attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx,
                            const uint32_t * rows, uint32_t n_rows,
                            const float * q, const float * kcache, const float * vcache,
                            const float * qg, float * out) {
    (void) sh; (void) n_ctx; (void) rows; (void) n_rows; (void) q; (void) kcache;
    (void) vcache; (void) qg; (void) out;
}

void qrt_full_attn_rows_cpu_f16(const qrt_full_shapes * sh, uint32_t n_ctx,
                                const uint32_t * rows, uint32_t n_rows,
                                const float * q, const float * kcache, const float * vcache,
                                const float * qg, float * out) {
    (void) sh; (void) n_ctx; (void) rows; (void) n_rows; (void) q; (void) kcache;
    (void) vcache; (void) qg; (void) out;
}

#endif // QRT_HAVE_VULKAN
