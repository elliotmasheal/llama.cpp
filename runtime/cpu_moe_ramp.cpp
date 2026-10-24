// CPU expert ramp bench (M3-1a round 2): one weight read serves t tokens.

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
#define MAX_TOK 8

static uint64_t rng_state = 0x243f6a8885a308d3ull;

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
        for (int i = 0; i < 12; ++i) blk[4 + i] = (uint8_t) (rng_u32() & 0xff);
        for (int i = 0; i < 128; ++i) blk[16 + i] = (uint8_t) (rng_u32() & 0xff);
    }
}

static double now_s(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
#else
    return omp_get_wtime();
#endif
}

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char ** argv) {
    int n_experts = 64;
    int reps = 5;
    int threads = 8;
    int n_tok = 5;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tok = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--experts N] [--reps R] [--threads T] [--tokens t]\n", argv[0]); return 2; }
    }
    if (n_tok < 1) n_tok = 1;
    if (n_tok > MAX_TOK) n_tok = MAX_TOK;
    if (n_experts % LAYER_EXPERTS) n_experts += LAYER_EXPERTS - n_experts % LAYER_EXPERTS;

    const size_t row_e = (size_t) (N_EMBD / QK_K) * Q4K_BYTES;
    const size_t row_f = (size_t) (N_FF / QK_K) * Q4K_BYTES;
    const size_t mat_gate = row_e * N_FF;
    const size_t mat_down = row_f * N_EMBD;
    const size_t expert_bytes = 2 * mat_gate + mat_down;
    const int qb_x = N_EMBD / QK8;
    const int qb_y = N_FF / QK8;
    const int G = LAYER_EXPERTS;
    const int gop = G * N_FF;
    const int oop = G * N_EMBD;

    uint8_t ** wg = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    uint8_t ** wu = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    uint8_t ** wd = (uint8_t **) calloc((size_t) n_experts, sizeof(uint8_t *));
    for (int e = 0; e < n_experts; ++e) {
        wg[e] = (uint8_t *) malloc(mat_gate);
        wu[e] = (uint8_t *) malloc(mat_gate);
        wd[e] = (uint8_t *) malloc(mat_down);
        fill_q4k(wg[e], mat_gate);
        fill_q4k(wu[e], mat_gate);
        fill_q4k(wd[e], mat_down);
    }
    float * x = (float *) malloc((size_t) n_tok * N_EMBD * sizeof(float));
    int8_t * qx = (int8_t *) malloc((size_t) n_tok * N_EMBD);
    float * dx = (float *) malloc((size_t) n_tok * qb_x * sizeof(float));
    float * sx = (float *) malloc((size_t) n_tok * qb_x * sizeof(float));
    float * gate = (float *) malloc((size_t) n_tok * gop * sizeof(float));
    float * up = (float *) malloc((size_t) n_tok * gop * sizeof(float));
    float * out = (float *) malloc((size_t) n_tok * oop * sizeof(float));
    int8_t * qy = (int8_t *) malloc((size_t) G * n_tok * N_FF);
    float * dy = (float *) malloc((size_t) G * n_tok * qb_y * sizeof(float));
    float * sy = (float *) malloc((size_t) G * n_tok * qb_y * sizeof(float));
    if (!wg || !wu || !wd || !x || !qx || !dx || !sx || !gate || !up || !out || !qy || !dy || !sy) return 3;

    for (int t = 0; t < n_tok; ++t) {
        for (int i = 0; i < N_EMBD; ++i) {
            x[(size_t) t * N_EMBD + i] = ((float) (rng_u32() % 2000) - 1000.0f) / 1000.0f;
        }
        qrt_q8_1_quantize(x + (size_t) t * N_EMBD, N_EMBD, qx + (size_t) t * N_EMBD,
                          dx + (size_t) t * qb_x, sx + (size_t) t * qb_x);
    }

    const int groups = n_experts / G;
    double timings[32];
    if (reps > 32) reps = 32;
    for (int rep = 0; rep < reps + 1; ++rep) {
        const double t0 = now_s();
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
        {
            for (int g = 0; g < groups; ++g) {
                const int e0 = g * G;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < G; ++e) {
                    qrt_gemv_q4k_q8_1_multi(wg[e0 + e], N_FF, N_EMBD, qx, dx, sx, n_tok, gate + e * N_FF, gop);
                    qrt_gemv_q4k_q8_1_multi(wu[e0 + e], N_FF, N_EMBD, qx, dx, sx, n_tok, up + e * N_FF, gop);
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < G; ++e) {
                    for (int t = 0; t < n_tok; ++t) {
                        float act[N_FF];
                        for (int i = 0; i < N_FF; ++i) {
                            const float gv = gate[(size_t) t * gop + e * N_FF + i];
                            const float uv = up[(size_t) t * gop + e * N_FF + i];
                            act[i] = (gv / (1.0f + expf(-gv))) * uv;
                        }
                        int8_t * qb = qy + ((size_t) e * n_tok + t) * N_FF;
                        qrt_q8_1_quantize(act, N_FF, qb,
                                          dy + ((size_t) e * n_tok + t) * qb_y,
                                          sy + ((size_t) e * n_tok + t) * qb_y);
                    }
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int e = 0; e < G; ++e) {
                    qrt_gemv_q4k_q8_1_multi(wd[e0 + e], N_EMBD, N_FF,
                                            qy + (size_t) e * n_tok * N_FF,
                                            dy + (size_t) e * n_tok * qb_y,
                                            sy + (size_t) e * n_tok * qb_y,
                                            n_tok, out + e * N_EMBD, oop);
                }
            }
        }
        const double t1 = now_s();
        if (rep > 0) timings[rep - 1] = t1 - t0;
    }

    double sorted[32];
    memcpy(sorted, timings, (size_t) reps * sizeof(double));
    qsort(sorted, (size_t) reps, sizeof(double), cmp_double);
    const double first = timings[0];
    const double median = sorted[reps / 2];
    const double flops = 2.0 * 3.0 * (double) N_EMBD * (double) N_FF * (double) n_tok * (double) n_experts;
    printf("threads=%d experts=%d tokens=%d reps=%d first_ms_per_expert=%.4f "
           "first_ms_per_token=%.4f median_ms_per_expert=%.4f median_ms_per_token=%.4f "
           "gflops_first=%.1f gbs_first=%.1f\n",
           threads, n_experts, n_tok, reps,
           first * 1e3 / n_experts, first * 1e3 / n_experts / n_tok,
           median * 1e3 / n_experts, median * 1e3 / n_experts / n_tok,
           flops / first / 1e9,
           (double) expert_bytes * n_experts / first / 1e9);

    // Token 0 of a fresh expert 0 pass: the multi kernel must match the single one.
    float * g0 = (float *) malloc((size_t) N_FF * sizeof(float));
    float * u0 = (float *) malloc((size_t) N_FF * sizeof(float));
    float * a0 = (float *) malloc((size_t) N_FF * sizeof(float));
    int8_t * q0 = (int8_t *) malloc((size_t) N_FF);
    float * d0 = (float *) malloc((size_t) qb_y * sizeof(float));
    float * s0 = (float *) malloc((size_t) qb_y * sizeof(float));
    float * m0out = (float *) malloc((size_t) N_EMBD * sizeof(float));
    float * ref = (float *) malloc((size_t) N_EMBD * sizeof(float));
    qrt_gemv_q4k_q8_1(wg[0], N_FF, N_EMBD, qx, dx, sx, g0);
    qrt_gemv_q4k_q8_1(wu[0], N_FF, N_EMBD, qx, dx, sx, u0);
    qrt_silu_mul(g0, u0, a0, N_FF);
    qrt_q8_1_quantize(a0, N_FF, q0, d0, s0);
    qrt_gemv_q4k_q8_1_multi(wd[0], N_EMBD, N_FF, q0, d0, s0, 1, m0out, N_EMBD);
    qrt_gemv_q4k_q8_1(wd[0], N_EMBD, N_FF, q0, d0, s0, ref);
    int bad = 0;
    for (int r = 0; r < N_EMBD; ++r) {
        if (memcmp(&ref[r], &m0out[r], sizeof(float)) != 0) bad++;
    }
    printf("check_t0 mismatches=%d/%d %s\n", bad, N_EMBD, bad == 0 ? "pass" : "fail");
    free(ref);
    return 0;
}
