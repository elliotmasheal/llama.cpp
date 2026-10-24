// CPU expert bench (M3-1a): Q4_K x Q8_1 expert pass, ms per expert.
// Synthetic weights with the real Q4_K block layout; no model file needed.

#include "cpu_moe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

#define QK_K 256
#define QK8 32
#define Q4K_BYTES 144
#define N_EMBD 2048
#define N_FF 512
#define LAYER_EXPERTS 8

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint32_t rng_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t) (rng_state >> 32);
}

static void fill_q4k(uint8_t * w, size_t n_bytes) {
    const size_t n_blocks = n_bytes / Q4K_BYTES;
    for (size_t b = 0; b < n_blocks; ++b) {
        uint8_t * blk = w + b * Q4K_BYTES;
        uint16_t d = (uint16_t) (0x2400u + (rng_u32() % 8));
        uint16_t dmin = (uint16_t) (0x1c00u + (rng_u32() % 8));
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 0; i < 12; ++i) {
            blk[4 + i] = (uint8_t) (rng_u32() & 0xff);
        }
        for (int i = 0; i < 128; ++i) {
            blk[16 + i] = (uint8_t) (rng_u32() & 0xff);
        }
    }
}

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
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

static void scale_min_k4(int j, const uint8_t * q, uint8_t * sc, uint8_t * mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (uint8_t) ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        *mn = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Double-precision reference for one row: dequantize both sides, dot in double.
static double ref_row(const uint8_t * w, int cols, const int8_t * q8,
                      const float * d8) {
    const int nb = cols / QK_K;
    double acc = 0.0;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = w + (size_t) b * Q4K_BYTES;
        uint16_t dh, dmh;
        memcpy(&dh, blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const double d = h2f(dh);
        const double dmin = h2f(dmh);
        const uint8_t * scales = blk + 4;
        const uint8_t * qs = blk + 16;
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            scale_min_k4(j, scales, &sc, &mn);
            const uint8_t * sub = qs + 32 * (j / 2);
            const int shift = (j % 2) * 4;
            const int8_t * a = q8 + (size_t) (b * 8 + j) * QK8;
            const double da = d8[b * 8 + j];
            double dot = 0.0;
            for (int i = 0; i < QK8; ++i) {
                const double wv = d * (double) sc * (double) ((sub[i] >> shift) & 0x0f) - dmin * (double) mn;
                dot += wv * (da * (double) a[i]);
            }
            acc += dot;
        }
    }
    return acc;
}

static double now_s(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
#elif defined(_OPENMP)
    return omp_get_wtime();
#else
    return 0.0;
#endif
}

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char ** argv) {
    int n_experts = 64;
    int reps = 3;
    int threads = 4;
    int check = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else {
            fprintf(stderr, "usage: %s [--experts N] [--reps R] [--threads T] [--check]\n", argv[0]);
            return 2;
        }
    }
    if (n_experts < LAYER_EXPERTS) n_experts = LAYER_EXPERTS;
    if (n_experts % LAYER_EXPERTS) n_experts += LAYER_EXPERTS - n_experts % LAYER_EXPERTS;

    const size_t row_bytes_e = (size_t) (N_EMBD / QK_K) * Q4K_BYTES;
    const size_t row_bytes_f = (size_t) (N_FF / QK_K) * Q4K_BYTES;
    const size_t mat_gate = row_bytes_e * N_FF;
    const size_t mat_down = row_bytes_f * N_EMBD;
    const size_t expert_bytes = 2 * mat_gate + mat_down;

    uint8_t ** w_gate = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    uint8_t ** w_up = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    uint8_t ** w_down = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    float * x = (float *) malloc((size_t) N_EMBD * sizeof(float));
    int8_t * qx = (int8_t *) malloc((size_t) N_EMBD);
    float * dx = (float *) malloc((size_t) (N_EMBD / QK8) * sizeof(float));
    float * sx = (float *) malloc((size_t) (N_EMBD / QK8) * sizeof(float));
    float ** gate = (float **) calloc((size_t) LAYER_EXPERTS, sizeof(float *));
    float ** up = (float **) calloc((size_t) LAYER_EXPERTS, sizeof(float *));
    int8_t ** qy = (int8_t **) calloc((size_t) LAYER_EXPERTS, sizeof(int8_t *));
    float ** dy = (float **) calloc((size_t) LAYER_EXPERTS, sizeof(float *));
    float ** sy = (float **) calloc((size_t) LAYER_EXPERTS, sizeof(float *));
    float ** out = (float **) calloc((size_t) LAYER_EXPERTS, sizeof(float *));
    if (!w_gate || !w_up || !w_down || !x || !qx || !dx || !sx) return 3;

    for (int e = 0; e < n_experts; ++e) {
        w_gate[e] = (uint8_t *) malloc(mat_gate);
        w_up[e] = (uint8_t *) malloc(mat_gate);
        w_down[e] = (uint8_t *) malloc(mat_down);
        if (!w_gate[e] || !w_up[e] || !w_down[e]) return 3;
        fill_q4k(w_gate[e], mat_gate);
        fill_q4k(w_up[e], mat_gate);
        fill_q4k(w_down[e], mat_down);
    }
    for (int i = 0; i < LAYER_EXPERTS; ++i) {
        gate[i] = (float *) malloc((size_t) N_FF * sizeof(float));
        up[i] = (float *) malloc((size_t) N_FF * sizeof(float));
        qy[i] = (int8_t *) malloc((size_t) N_FF);
        dy[i] = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        sy[i] = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        out[i] = (float *) malloc((size_t) N_EMBD * sizeof(float));
        if (!gate[i] || !up[i] || !qy[i] || !dy[i] || !sy[i] || !out[i]) return 3;
    }

    for (int i = 0; i < N_EMBD; ++i) {
        x[i] = ((float) (rng_u32() % 2000) - 1000.0f) / 1000.0f;
    }
    qrt_q8_1_quantize(x, N_EMBD, qx, dx, sx);

    double timings[64];
    if (reps > 64) reps = 64;
    int n_groups = n_experts / LAYER_EXPERTS;
    for (int rep = 0; rep < reps + 1; ++rep) {
        const double t0 = now_s();
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
        {
            for (int g = 0; g < n_groups; ++g) {
                const int e0 = g * LAYER_EXPERTS;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < LAYER_EXPERTS; ++e) {
                    qrt_gemv_q4k_q8_1(w_gate[e0 + e], N_FF, N_EMBD, qx, dx, sx, gate[e]);
                    qrt_gemv_q4k_q8_1(w_up[e0 + e], N_FF, N_EMBD, qx, dx, sx, up[e]);
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < LAYER_EXPERTS; ++e) {
                    float act[N_FF];
                    for (int i = 0; i < N_FF; ++i) {
                        const float gv = gate[e][i];
                        act[i] = (gv / (1.0f + expf(-gv))) * up[e][i];
                    }
                    qrt_q8_1_quantize(act, N_FF, qy[e], dy[e], sy[e]);
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < LAYER_EXPERTS; ++e) {
                    qrt_gemv_q4k_q8_1(w_down[e0 + e], N_EMBD, N_FF, qy[e], dy[e], sy[e], out[e]);
                }
            }
        }
        const double t1 = now_s();
        if (rep > 0) timings[rep - 1] = t1 - t0;
    }

    double sorted[64];
    memcpy(sorted, timings, (size_t) reps * sizeof(double));
    qsort(sorted, (size_t) reps, sizeof(double), cmp_double);
    const double median = sorted[reps / 2];
    const double per_expert = median / (double) n_experts;
    const double gflops = 2.0 * 3.0 * (double) N_EMBD * (double) N_FF * (double) n_experts / median / 1e9;
    const double gbs = (double) expert_bytes * (double) n_experts / median / 1e9;

    printf("threads=%d experts=%d reps=%d bytes_per_expert=%zu per_expert_ms=%.4f "
           "gflops=%.1f gbs=%.1f\n",
           threads, n_experts, reps, expert_bytes, per_expert * 1e3, gflops, gbs);
    for (int r = 0; r < reps; ++r) {
        printf("rep %d: total_ms=%.4f per_expert_ms=%.4f\n", r, sorted[r] * 1e3,
               sorted[r] * 1e3 / (double) n_experts);
    }

    if (check) {
        const uint8_t * wg = w_gate[0];
        const uint8_t * wd = w_down[0];
        float * kg = (float *) malloc((size_t) N_FF * sizeof(float));
        float * ku = (float *) malloc((size_t) N_FF * sizeof(float));
        float * ka = (float *) malloc((size_t) N_FF * sizeof(float));
        int8_t * kqy = (int8_t *) malloc((size_t) N_FF);
        float * kdy = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        float * ksy = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        float * kout = (float *) malloc((size_t) N_EMBD * sizeof(float));
        float * rg = (float *) malloc((size_t) N_FF * sizeof(float));
        float * ru = (float *) malloc((size_t) N_FF * sizeof(float));
        float * ra = (float *) malloc((size_t) N_FF * sizeof(float));
        int8_t * rqy = (int8_t *) malloc((size_t) N_FF);
        float * rdy = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        float * rsy = (float *) malloc((size_t) (N_FF / QK8) * sizeof(float));
        float * rout = (float *) malloc((size_t) N_EMBD * sizeof(float));
        if (!kg || !ku || !ka || !kqy || !kdy || !ksy || !kout ||
            !rg || !ru || !ra || !rqy || !rdy || !rsy || !rout) return 3;
        qrt_gemv_q4k_q8_1(wg, N_FF, N_EMBD, qx, dx, sx, kg);
        qrt_gemv_q4k_q8_1(w_up[0], N_FF, N_EMBD, qx, dx, sx, ku);
        qrt_silu_mul(kg, ku, ka, N_FF);
        qrt_q8_1_quantize(ka, N_FF, kqy, kdy, ksy);
        qrt_gemv_q4k_q8_1(wd, N_EMBD, N_FF, kqy, kdy, ksy, kout);
        for (int r = 0; r < N_FF; ++r) {
            rg[r] = (float) ref_row(wg + (size_t) r * row_bytes_e, N_EMBD, qx, dx);
            ru[r] = (float) ref_row(w_up[0] + (size_t) r * row_bytes_e, N_EMBD, qx, dx);
            const float gv = rg[r];
            ra[r] = (gv / (1.0f + expf(-gv))) * ru[r];
        }
        qrt_q8_1_quantize(ra, N_FF, rqy, rdy, rsy);
        double num = 0.0;
        double den = 0.0;
        for (int r = 0; r < N_EMBD; ++r) {
            rout[r] = (float) ref_row(wd + (size_t) r * row_bytes_f, N_FF, rqy, rdy);
            const double diff = fabs((double) kout[r] - (double) rout[r]);
            const double base = fabs((double) rout[r]);
            num += diff;
            den += base;
        }
        const double rel = den > 0.0 ? num / den : num;
        printf("check=rel_l1 ratio=%.3e %s\n", rel, rel <= 1e-4 ? "pass" : "fail");
    }

    return 0;
}
