// CPU expert core: Q4_K x Q8_1 int8 dot products, AVX2 path with a scalar fallback.

#include "cpu_moe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256
#define QK8 32
#define Q4K_BYTES 144

struct block_q4_k_local {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct qrt_cpu_moe {
    int threads;
};

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// 6-bit scale/min pair for sub-block j of a Q4_K super-block.
static void scale_min_k4(int j, const uint8_t * q, uint8_t * sc, uint8_t * mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (uint8_t) ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        *mn = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

#if defined(__AVX2__)

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static void gemv_avx2(const uint8_t * w, int rows, int cols,
                      const int8_t * q8, const float * d8, const float * ssum,
                      float * out) {
    const int nb = cols / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q4K_BYTES;
        // Two accumulator pairs: the fma chain per superblock is half as long,
        // which is what limits this kernel at one row per call.
        __m256 acc_lo[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        __m256 acc_hi[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        float acc_min = 0.0f;
        for (int b = 0; b < nb; ++b) {
            struct block_q4_k_local blk;
            memcpy(&blk, row + (size_t) b * Q4K_BYTES, Q4K_BYTES);
            if (b + 4 < nb) {
                __builtin_prefetch(row + (size_t) (b + 4) * Q4K_BYTES, 0, 3);
            }
            const float d = h2f(blk.d);
            const float dmin = h2f(blk.dmin);
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
            for (int g = 0; g < 4; ++g) {
                const int slot = g & 1;
                const __m256i q = _mm256_loadu_si256((const __m256i *) (blk.qs + 32 * g));
                const __m256i lo = _mm256_and_si256(q, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                uint8_t s0, m0, s1, m1;
                scale_min_k4(2 * g, blk.scales, &s0, &m0);
                scale_min_k4(2 * g + 1, blk.scales, &s1, &m1);
                const int k0 = b * 8 + 2 * g;
                const int k1 = k0 + 1;
                const __m256i b0 = _mm256_loadu_si256((const __m256i *) (q8 + (size_t) k0 * QK8));
                const __m256i b1 = _mm256_loadu_si256((const __m256i *) (q8 + (size_t) k1 * QK8));
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, b0), ones);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, b1), ones);
                acc_lo[slot] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0),
                                               _mm256_set1_ps(d * (float) s0 * d8[k0]), acc_lo[slot]);
                acc_hi[slot] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1),
                                               _mm256_set1_ps(d * (float) s1 * d8[k1]), acc_hi[slot]);
                acc_min -= dmin * ((float) m0 * ssum[k0] + (float) m1 * ssum[k1]);
            }
        }
        out[r] = hsum_ps(_mm256_add_ps(acc_lo[0], acc_lo[1])) +
                 hsum_ps(_mm256_add_ps(acc_hi[0], acc_hi[1])) + acc_min;
    }
}

#endif

static void gemv_scalar(const uint8_t * w, int rows, int cols,
                        const int8_t * q8, const float * d8, const float * ssum,
                        float * out) {
    const int nb = cols / QK_K;
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q4K_BYTES;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            struct block_q4_k_local blk;
            memcpy(&blk, row + (size_t) b * Q4K_BYTES, Q4K_BYTES);
            const float d = h2f(blk.d);
            const float dmin = h2f(blk.dmin);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, mn;
                scale_min_k4(j, blk.scales, &sc, &mn);
                const uint8_t * qs = blk.qs + 32 * (j / 2);
                const int shift = (j % 2) * 4;
                int32_t dot = 0;
                const int8_t * a = q8 + (size_t) (b * 8 + j) * QK8;
                for (int i = 0; i < QK8; ++i) {
                    dot += ((qs[i] >> shift) & 0x0f) * (int32_t) a[i];
                }
                acc += d * d8[b * 8 + j] * (float) sc * (float) dot - dmin * (float) mn * ssum[b * 8 + j];
            }
        }
        out[r] = acc;
    }
}

void qrt_gemv_q4k_q8_1(const void * w, int rows, int cols,
                       const int8_t * qs, const float * d, const float * ssum,
                       float * out) {
#if defined(__AVX2__)
    gemv_avx2((const uint8_t *) w, rows, cols, qs, d, ssum, out);
#else
    gemv_scalar((const uint8_t *) w, rows, cols, qs, d, ssum, out);
#endif
}

#if defined(__AVX2__)
template <int NT>
static void gemv_multi_impl(const uint8_t * wb, int rows, int cols,
                            const int8_t * qs, const float * d, const float * ssum,
                            float * out, int out_stride) {
    const int nb = cols / QK_K;
    const int qb = cols / QK8;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = wb + (size_t) r * nb * Q4K_BYTES;
        __m256 acc_lo[NT];
        __m256 acc_hi[NT];
        float acc_min[NT];
        for (int t = 0; t < NT; ++t) {
            acc_lo[t] = _mm256_setzero_ps();
            acc_hi[t] = _mm256_setzero_ps();
            acc_min[t] = 0.0f;
        }
        for (int b = 0; b < nb; ++b) {
            struct block_q4_k_local blk;
            memcpy(&blk, row + (size_t) b * Q4K_BYTES, Q4K_BYTES);
            if (b + 4 < nb) {
                __builtin_prefetch(row + (size_t) (b + 4) * Q4K_BYTES, 0, 3);
            }
            const float dv = h2f(blk.d);
            const float dmin = h2f(blk.dmin);
            for (int g = 0; g < 4; ++g) {
                const __m256i q = _mm256_loadu_si256((const __m256i *) (blk.qs + 32 * g));
                const __m256i lo = _mm256_and_si256(q, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                uint8_t s0, m0, s1, m1;
                scale_min_k4(2 * g, blk.scales, &s0, &m0);
                scale_min_k4(2 * g + 1, blk.scales, &s1, &m1);
                const int k0 = b * 8 + 2 * g;
                const int k1 = k0 + 1;
                for (int t = 0; t < NT; ++t) {
                    const int8_t * qt = qs + (size_t) t * qb * QK8;
                    const float * dt = d + (size_t) t * qb;
                    const float * st = ssum + (size_t) t * qb;
                    const __m256i b0 = _mm256_loadu_si256((const __m256i *) (qt + (size_t) k0 * QK8));
                    const __m256i b1 = _mm256_loadu_si256((const __m256i *) (qt + (size_t) k1 * QK8));
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, b0), ones);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, b1), ones);
                    acc_lo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0),
                                                _mm256_set1_ps(dv * (float) s0 * dt[k0]), acc_lo[t]);
                    acc_hi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1),
                                                _mm256_set1_ps(dv * (float) s1 * dt[k1]), acc_hi[t]);
                    acc_min[t] -= dmin * ((float) m0 * st[k0] + (float) m1 * st[k1]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            out[(size_t) t * out_stride + r] = hsum_ps(acc_lo[t]) + hsum_ps(acc_hi[t]) + acc_min[t];
        }
    }
}
#endif

void qrt_gemv_q4k_q8_1_multi(const void * w, int rows, int cols,
                             const int8_t * qs, const float * d, const float * ssum,
                             int n_tok, float * out, int out_stride) {
#if defined(__AVX2__)
    const uint8_t * wb = (const uint8_t *) w;
    switch (n_tok) {
        case 1: gemv_multi_impl<1>(wb, rows, cols, qs, d, ssum, out, out_stride); return;
        case 2: gemv_multi_impl<2>(wb, rows, cols, qs, d, ssum, out, out_stride); return;
        case 3: gemv_multi_impl<3>(wb, rows, cols, qs, d, ssum, out, out_stride); return;
        case 4: gemv_multi_impl<4>(wb, rows, cols, qs, d, ssum, out, out_stride); return;
        case 5: gemv_multi_impl<5>(wb, rows, cols, qs, d, ssum, out, out_stride); return;
        default: break;
    }
#endif
    const int qb = cols / QK8;
    for (int t = 0; t < n_tok; ++t) {
        qrt_gemv_q4k_q8_1(w, rows, cols, qs + (size_t) t * qb * QK8, d + (size_t) t * qb,
                          ssum + (size_t) t * qb, out + (size_t) t * out_stride);
    }
}

void qrt_q8_1_quantize(const float * x, int n, int8_t * qs, float * d, float * ssum) {
    const int nb = n / QK8;
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + (size_t) b * QK8;
        float amax = 0.0f;
        for (int i = 0; i < QK8; ++i) {
            const float a = fabsf(xb[i]);
            amax = a > amax ? a : amax;
        }
        const float scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        const float db = amax > 0.0f ? amax / 127.0f : 0.0f;
        int32_t sum = 0;
        int8_t * qb = qs + (size_t) b * QK8;
        for (int i = 0; i < QK8; ++i) {
            float v = xb[i] * scale;
            int32_t q = (int32_t) lrintf(v);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qb[i] = (int8_t) q;
            sum += q;
        }
        d[b] = db;
        ssum[b] = db * (float) sum;
    }
}

void qrt_silu_mul(const float * gate, const float * up, float * out, int n) {
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

qrt_cpu_moe * qrt_cpu_moe_create(int threads) {
    qrt_cpu_moe * ctx = (qrt_cpu_moe *) calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->threads = threads > 0 ? threads : 4;
    return ctx;
}

void qrt_cpu_moe_free(qrt_cpu_moe * ctx) {
    free(ctx);
}

int qrt_cpu_moe_threads(const qrt_cpu_moe * ctx) {
    return ctx ? ctx->threads : 0;
}

void qrt_cpu_moe_expert(qrt_cpu_moe * ctx,
                        const void * w_gate, const void * w_up, const void * w_down,
                        const float * x, int n_embd, int n_ff, float * out) {
    const int nq_x = n_embd / QK8;
    const int nq_y = n_ff / QK8;
    int8_t * qx = (int8_t *) malloc((size_t) n_embd);
    int8_t * qy = (int8_t *) malloc((size_t) n_ff);
    float * dx = (float *) malloc((size_t) nq_x * sizeof(float));
    float * sx = (float *) malloc((size_t) nq_x * sizeof(float));
    float * dy = (float *) malloc((size_t) nq_y * sizeof(float));
    float * sy = (float *) malloc((size_t) nq_y * sizeof(float));
    float * gate = (float *) malloc((size_t) n_ff * sizeof(float));
    float * up = (float *) malloc((size_t) n_ff * sizeof(float));
    const int threads = ctx ? ctx->threads : 1;
    if (!qx || !qy || !dx || !sx || !dy || !sy || !gate || !up) {
        goto done;
    }

    qrt_q8_1_quantize(x, n_embd, qx, dx, sx);

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int r = 0; r < n_ff; ++r) {
            float g, u;
            qrt_gemv_q4k_q8_1((const uint8_t *) w_gate + (size_t) r * (n_embd / QK_K) * Q4K_BYTES,
                              1, n_embd, qx, dx, sx, &g);
            qrt_gemv_q4k_q8_1((const uint8_t *) w_up + (size_t) r * (n_embd / QK_K) * Q4K_BYTES,
                              1, n_embd, qx, dx, sx, &u);
            gate[r] = g;
            up[r] = u;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int b = 0; b < nq_y; ++b) {
            float acc[QK8];
            for (int i = 0; i < QK8; ++i) {
                const int idx = b * QK8 + i;
                const float g = gate[idx];
                acc[i] = (g / (1.0f + expf(-g))) * up[idx];
            }
            float amax = 0.0f;
            for (int i = 0; i < QK8; ++i) {
                const float a = fabsf(acc[i]);
                amax = a > amax ? a : amax;
            }
            const float scale = amax > 0.0f ? 127.0f / amax : 0.0f;
            const float db = amax > 0.0f ? amax / 127.0f : 0.0f;
            int32_t sum = 0;
            int8_t * qb = qy + (size_t) b * QK8;
            for (int i = 0; i < QK8; ++i) {
                int32_t q = (int32_t) lrintf(acc[i] * scale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                qb[i] = (int8_t) q;
                sum += q;
            }
            dy[b] = db;
            sy[b] = db * (float) sum;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int r = 0; r < n_embd; ++r) {
            float v;
            qrt_gemv_q4k_q8_1((const uint8_t *) w_down + (size_t) r * (n_ff / QK_K) * Q4K_BYTES,
                              1, n_ff, qy, dy, sy, &v);
            out[r] = v;
        }
    }

done:
    free(qx); free(qy); free(dx); free(sx); free(dy); free(sy);
    free(gate); free(up);
}
