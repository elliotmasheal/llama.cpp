#pragma once

#include <stdint.h>

// CPU expert core for the qwen36 runtime: Q4_K weights, Q8_1 activations.
// One call runs gate, up, silu*mul and down for a single token.

typedef struct qrt_cpu_moe qrt_cpu_moe;

qrt_cpu_moe * qrt_cpu_moe_create(int threads);
void qrt_cpu_moe_free(qrt_cpu_moe * ctx);

// Quantize n floats (n % 32 == 0) into Q8_1 blocks: qs, per-block scale d and
// per-block sum ssum. Layout matches block_q8_1 minus the fp16 headers.
void qrt_q8_1_quantize(const float * x, int n, int8_t * qs, float * d, float * ssum);

// out[rows] = W[rows][cols] * x, W in Q4_K super-blocks (256 values, 144 B).
void qrt_gemv_q4k_q8_1(const void * w, int rows, int cols,
                       const int8_t * qs, const float * d, const float * ssum,
                       float * out);

void qrt_silu_mul(const float * gate, const float * up, float * out, int n);

// t tokens through one weight matrix: W[rows][cols] is read once, activations
// for token t are at qs + t*cols, d + t*nb, ssum + t*nb; out[t*out_stride + r].
void qrt_gemv_q4k_q8_1_multi(const void * w, int rows, int cols,
                             const int8_t * qs, const float * d, const float * ssum,
                             int n_tok, float * out, int out_stride);

// Full expert pass: n_embd input floats, n_ff hidden; w_* are Q4_K row-major.
void qrt_cpu_moe_expert(qrt_cpu_moe * ctx,
                        const void * w_gate, const void * w_up, const void * w_down,
                        const float * x, int n_embd, int n_ff, float * out);

int qrt_cpu_moe_threads(const qrt_cpu_moe * ctx);
