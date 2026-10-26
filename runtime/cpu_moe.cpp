// CPU expert core: Q4_K x Q8_1 int8 dot products, AVX2 path with a scalar fallback.

#include "cpu_moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK_K 256
#define QK8 32
#define Q4K_BYTES 144
#define Q5K_BYTES 176
#define Q6K_BYTES 210

struct block_q4_k_local {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q5_k_local {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct block_q6_k_local {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
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
        // Subnormal deltas are real in this store: a block whose d is one carries
        // ~1e-4, and dropping it changes the row (DECISIONS 65/67).
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((113u - e) << 23) | ((mant & 0x3ffu) << 13);
        }
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
            // Read the 144-byte superblock in place: the copy only fed the loop.
            const struct block_q4_k_local & blk =
                *(const struct block_q4_k_local *) (row + (size_t) b * Q4K_BYTES);
            if (b + 8 < nb) {
                __builtin_prefetch(row + (size_t) (b + 8) * Q4K_BYTES, 0, 3);
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

static void gemv_q5k_avx2(const uint8_t * w, int rows, int cols,
                          const int8_t * q8, const float * d8, const float * ssum,
                          float * out) {
    const int nb = cols / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i sixteen = _mm256_set1_epi8(16);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q5K_BYTES;
        __m256 acc_a[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        __m256 acc_b[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        float acc_min = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const struct block_q5_k_local & blk =
                *(const struct block_q5_k_local *) (row + (size_t) b * Q5K_BYTES);
            const float dv = h2f(blk.d);
            const float dmin = h2f(blk.dmin);
            const __m256i qh = _mm256_loadu_si256((const __m256i *) blk.qh);
            for (int g = 0; g < 4; ++g) {
                const __m256i ql = _mm256_loadu_si256((const __m256i *) (blk.qs + 32 * g));
                const __m256i lo = _mm256_and_si256(ql, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(ql, 4), m4);
                const __m256i bit_a = _mm256_set1_epi8((char) (unsigned) (1u << (2 * g)));
                const __m256i bit_b = _mm256_set1_epi8((char) (unsigned) (1u << (2 * g + 1)));
                // the fifth bit lives in qh, one mask bit per 32-value sub-block
                const __m256i qa = _mm256_add_epi8(lo, _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit_a), bit_a), sixteen));
                const __m256i qb = _mm256_add_epi8(hi, _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit_b), bit_b), sixteen));
                uint8_t s0, m0, s1, m1;
                scale_min_k4(2 * g, blk.scales, &s0, &m0);
                scale_min_k4(2 * g + 1, blk.scales, &s1, &m1);
                const int k0 = b * 8 + 2 * g;
                const int k1 = k0 + 1;
                const __m256i b0 = _mm256_loadu_si256((const __m256i *) (q8 + (size_t) k0 * QK8));
                const __m256i b1 = _mm256_loadu_si256((const __m256i *) (q8 + (size_t) k1 * QK8));
                const int slot = g & 1;
                acc_a[slot] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qa, b0), ones)),
                    _mm256_set1_ps(dv * (float) s0 * d8[k0]), acc_a[slot]);
                acc_b[slot] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(qb, b1), ones)),
                    _mm256_set1_ps(dv * (float) s1 * d8[k1]), acc_b[slot]);
                acc_min -= dmin * ((float) m0 * ssum[k0] + (float) m1 * ssum[k1]);
            }
        }
        out[r] = hsum_ps(_mm256_add_ps(acc_a[0], acc_a[1])) +
                 hsum_ps(_mm256_add_ps(acc_b[0], acc_b[1])) + acc_min;
    }
}

static void gemv_q6k_avx2(const uint8_t * w, int rows, int cols,
                          const int8_t * q8, const float * d8, const float * ssum,
                          float * out) {
    (void) ssum;  // Q6_K has no minimum term
    const int nb = cols / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i m2 = _mm256_set1_epi8(0x03);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i u8one = _mm256_set1_epi8(1);
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q6K_BYTES;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const struct block_q6_k_local & blk =
                *(const struct block_q6_k_local *) (row + (size_t) b * Q6K_BYTES);
            const float dv = h2f(blk.d);
            for (int half = 0; half < 2; ++half) {
                const uint8_t * ql0 = blk.ql + half * 64;
                const uint8_t * ql1 = ql0 + 32;
                const uint8_t * qh = blk.qh + half * 32;
                const int8_t * sc = blk.scales + half * 8;
                const __m256i l0 = _mm256_loadu_si256((const __m256i *) ql0);
                const __m256i l1 = _mm256_loadu_si256((const __m256i *) ql1);
                const __m256i h = _mm256_loadu_si256((const __m256i *) qh);
                const __m256i lo0 = _mm256_and_si256(l0, m4);
                const __m256i hi0 = _mm256_and_si256(_mm256_srli_epi16(l0, 4), m4);
                const __m256i lo1 = _mm256_and_si256(l1, m4);
                const __m256i hi1 = _mm256_and_si256(_mm256_srli_epi16(l1, 4), m4);
                const __m256i wq[4] = {
                    _mm256_or_si256(lo0, _mm256_slli_epi16(_mm256_and_si256(h, m2), 4)),
                    _mm256_or_si256(lo1, _mm256_slli_epi16(_mm256_and_si256(h, _mm256_set1_epi8(0x0c)), 2)),
                    _mm256_or_si256(hi0, _mm256_and_si256(h, m30)),
                    _mm256_or_si256(hi1, _mm256_and_si256(_mm256_srli_epi16(h, 2), m30)),
                };
                for (int g = 0; g < 4; ++g) {
                    const int k = b * 8 + half * 4 + g;
                    const __m256i a = _mm256_loadu_si256((const __m256i *) (q8 + (size_t) k * QK8));
                    // unsigned 6-bit weights times signed activations; the -32 offset
                    // becomes -32 * sum(a) on the same lanes (4 values per lane).
                    const __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(wq[g], a), ones);
                    const __m256i sum = _mm256_madd_epi16(_mm256_maddubs_epi16(u8one, a), ones);
                    const __m256i corr =
                        _mm256_sub_epi32(dot, _mm256_slli_epi32(sum, 5));
                    const __m256 scale = _mm256_mul_ps(
                        _mm256_set1_ps(dv * d8[k]),
                        _mm256_set_ps((float) sc[2 * g + 1], (float) sc[2 * g + 1],
                                      (float) sc[2 * g + 1], (float) sc[2 * g + 1],
                                      (float) sc[2 * g + 0], (float) sc[2 * g + 0],
                                      (float) sc[2 * g + 0], (float) sc[2 * g + 0]));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(corr), scale, acc);
                }
            }
        }
        out[r] = hsum_ps(acc);
    }
}

#endif

static void gemv_q5k_scalar(const uint8_t * w, int rows, int cols,
                            const int8_t * q8, const float * d8, const float * ssum,
                            float * out) {
    const int nb = cols / QK_K;
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q5K_BYTES;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            struct block_q5_k_local blk;
            memcpy(&blk, row + (size_t) b * Q5K_BYTES, Q5K_BYTES);
            const float dv = h2f(blk.d);
            const float dmin = h2f(blk.dmin);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, mn;
                scale_min_k4(j, blk.scales, &sc, &mn);
                const uint8_t * qs = blk.qs + 32 * (j / 2);
                const int shift = (j % 2) * 4;
                const uint8_t bit = (uint8_t) (1u << j);
                int32_t dot = 0;
                int32_t sum = 0;
                const int8_t * a = q8 + (size_t) (b * 8 + j) * QK8;
                for (int i = 0; i < QK8; ++i) {
                    const int32_t q = (int32_t) ((qs[i] >> shift) & 0x0f) +
                                      ((blk.qh[i] & bit) ? 16 : 0);
                    dot += q * (int32_t) a[i];
                    sum += (int32_t) a[i];
                }
                acc += dv * d8[b * 8 + j] * (float) sc * (float) dot -
                       dmin * (float) mn * d8[b * 8 + j] * (float) sum;
            }
        }
        out[r] = acc;
    }
}

static void gemv_q6k_scalar(const uint8_t * w, int rows, int cols,
                            const int8_t * q8, const float * d8, const float * ssum,
                            float * out) {
    (void) ssum;
    const int nb = cols / QK_K;
    for (int r = 0; r < rows; ++r) {
        const uint8_t * row = w + (size_t) r * nb * Q6K_BYTES;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            struct block_q6_k_local blk;
            memcpy(&blk, row + (size_t) b * Q6K_BYTES, Q6K_BYTES);
            const float dv = h2f(blk.d);
            for (int half = 0; half < 2; ++half) {
                const uint8_t * ql = blk.ql + half * 64;
                const uint8_t * qh = blk.qh + half * 32;
                const int8_t * sc = blk.scales + half * 8;
                for (int g = 0; g < 4; ++g) {
                    const int k = b * 8 + half * 4 + g;
                    const int8_t * a = q8 + (size_t) k * QK8;
                    for (int l = 0; l < 32; ++l) {
                        int q;
                        switch (g) {
                            case 0: q = (ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4); break;
                            case 1: q = (ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4); break;
                            case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
                            default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
                        }
                        const int s = g * 2 + (l / 16);
                        acc += dv * d8[k] * (float) sc[s] * (float) (q - 32) * (float) a[l];
                    }
                }
            }
        }
        out[r] = acc;
    }
}

void qrt_gemv_q5k_q8_1(const void * w, int rows, int cols,
                       const int8_t * qs, const float * d, const float * ssum,
                       float * out) {
#if defined(__AVX2__)
    gemv_q5k_avx2((const uint8_t *) w, rows, cols, qs, d, ssum, out);
#else
    gemv_q5k_scalar((const uint8_t *) w, rows, cols, qs, d, ssum, out);
#endif
}

void qrt_gemv_q6k_q8_1(const void * w, int rows, int cols,
                       const int8_t * qs, const float * d, const float * ssum,
                       float * out) {
#if defined(__AVX2__)
    gemv_q6k_avx2((const uint8_t *) w, rows, cols, qs, d, ssum, out);
#else
    gemv_q6k_scalar((const uint8_t *) w, rows, cols, qs, d, ssum, out);
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
            // Read the 144-byte superblock in place: the copy only fed the loop.
            const struct block_q4_k_local & blk =
                *(const struct block_q4_k_local *) (row + (size_t) b * Q4K_BYTES);
            if (b + 8 < nb) {
                __builtin_prefetch(row + (size_t) (b + 8) * Q4K_BYTES, 0, 3);
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

static int block_bytes(int type) {
    switch (type) {
        case QRT_CPU_Q4K: return Q4K_BYTES;
        case QRT_CPU_Q5K: return Q5K_BYTES;
        case QRT_CPU_Q6K: return Q6K_BYTES;
    }
    return 0;
}

static void gemv_typed(int type, const void * w, int rows, int cols,
                       const int8_t * qs, const float * d, const float * ssum, float * out) {
    switch (type) {
        case QRT_CPU_Q5K: qrt_gemv_q5k_q8_1(w, rows, cols, qs, d, ssum, out); break;
        case QRT_CPU_Q6K: qrt_gemv_q6k_q8_1(w, rows, cols, qs, d, ssum, out); break;
        default: qrt_gemv_q4k_q8_1(w, rows, cols, qs, d, ssum, out); break;
    }
}

static int expert_impl(qrt_cpu_moe * ctx, int t_gate, int t_up, int t_down,
                       const void * w_gate, const void * w_up, const void * w_down,
                       const int8_t * qx, const float * dx, const float * sx,
                       int n_embd, int n_ff, float * out) {
    const int nq_y = n_ff / QK8;
    const int tb_gate = block_bytes(t_gate);
    const int tb_up = block_bytes(t_up);
    const int tb_down = block_bytes(t_down);
    if (tb_gate == 0 || tb_up == 0 || tb_down == 0 || n_embd % QK_K != 0 || n_ff % QK_K != 0) {
        return -1;
    }
    const size_t row_gate = (size_t) (n_embd / QK_K) * (size_t) tb_gate;
    const size_t row_up = (size_t) (n_embd / QK_K) * (size_t) tb_up;
    const size_t row_down = (size_t) (n_ff / QK_K) * (size_t) tb_down;
    int8_t * qy = (int8_t *) malloc((size_t) n_ff);
    float * dy = (float *) malloc((size_t) nq_y * sizeof(float));
    float * sy = (float *) malloc((size_t) nq_y * sizeof(float));
    float * gate = (float *) malloc((size_t) n_ff * sizeof(float));
    float * up = (float *) malloc((size_t) n_ff * sizeof(float));
    const int threads = ctx ? ctx->threads : 1;
    if (!qy || !dy || !sy || !gate || !up) {
        free(qy); free(dy); free(sy); free(gate); free(up);
        return -1;
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int r = 0; r < n_ff; ++r) {
            float g, u;
            gemv_typed(t_gate, (const uint8_t *) w_gate + (size_t) r * row_gate,
                       1, n_embd, qx, dx, sx, &g);
            gemv_typed(t_up, (const uint8_t *) w_up + (size_t) r * row_up,
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
            gemv_typed(t_down, (const uint8_t *) w_down + (size_t) r * row_down,
                       1, n_ff, qy, dy, sy, &v);
            out[r] = v;
        }
    }

    free(qy); free(dy); free(sy); free(gate); free(up);
    return 0;
}

void qrt_cpu_moe_expert_typed(qrt_cpu_moe * ctx,
                              int t_gate, int t_up, int t_down,
                              const void * w_gate, const void * w_up, const void * w_down,
                              const float * x, int n_embd, int n_ff, float * out) {
    const int nq_x = n_embd / QK8;
    int8_t * qx = (int8_t *) malloc((size_t) n_embd);
    float * dx = (float *) malloc((size_t) nq_x * sizeof(float));
    float * sx = (float *) malloc((size_t) nq_x * sizeof(float));
    if (!qx || !dx || !sx) {
        free(qx); free(dx); free(sx);
        return;
    }
    qrt_q8_1_quantize(x, n_embd, qx, dx, sx);
    expert_impl(ctx, t_gate, t_up, t_down, w_gate, w_up, w_down, qx, dx, sx, n_embd, n_ff, out);
    free(qx); free(dx); free(sx);
}

void qrt_cpu_moe_expert_q8(qrt_cpu_moe * ctx,
                           int t_gate, int t_up, int t_down,
                           const void * w_gate, const void * w_up, const void * w_down,
                           const int8_t * x_qs, const float * x_d, const float * x_ssum,
                           int n_embd, int n_ff, float * out) {
    expert_impl(ctx, t_gate, t_up, t_down, w_gate, w_up, w_down, x_qs, x_d, x_ssum,
                n_embd, n_ff, out);
}

void qrt_cpu_moe_expert(qrt_cpu_moe * ctx,
                        const void * w_gate, const void * w_up, const void * w_down,
                        const float * x, int n_embd, int n_ff, float * out) {
    qrt_cpu_moe_expert_typed(ctx, QRT_CPU_Q4K, QRT_CPU_Q4K, QRT_CPU_Q4K,
                             w_gate, w_up, w_down, x, n_embd, n_ff, out);
}
