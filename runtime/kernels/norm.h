#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// RMS norm of a token row plus the Q8_1 activation packing the dense GEMM eats (M2-6,
// first piece of the layer chain). One workgroup per token, 128 threads. The planes match
// kernels/gemm.h: qs is 32 int8 per 32-element block in uvec4 pairs, ds is one word per
// block with the f16 delta in the low half and sum(q) + 32768 in the high half.
//
// The buffers belong to the caller: the layer chain hands the same planes to the GEMM, so
// this module allocates nothing but its own device state.

struct qrt_norm;

struct qrt_norm_args {
    VkBuffer x;         // f32 [tokens][k]
    uint32_t xbase;     // first float of token 0 in x
    VkBuffer w;         // f32 [k], the rms norm weight
    uint32_t wbase;     // first float of the weight
    VkBuffer out;       // f32 [tokens][k], the normalised row
    uint32_t obase;     // first float of token 0 in out
    VkBuffer qs;        // Q8_1 quants, two uvec4 per block; unused when quant == 0
    uint32_t qsbase;    // first uvec4 of token 0
    VkBuffer ds;        // Q8_1 deltas, one word per block; unused when quant == 0
    uint32_t dsbase;    // first word of token 0
    uint32_t k;         // row length, multiple of 32
    uint32_t tokens;
    float eps;
    int quant;          // 0 = norm only
};

int qrt_norm_open(qrt_vk_dev * dev, const char * spv_dir, qrt_norm ** out,
                  char * err, size_t err_size);
void qrt_norm_close(qrt_vk_dev * dev, qrt_norm * n);

// Two command rings so a caller can record the next chunk while the current one runs.
// A ring's descriptor set is rewritten on every record, so only record into a ring whose
// previous submission has been waited for.
uint32_t qrt_norm_rings(void);
int qrt_norm_record(qrt_norm * n, uint32_t ring, const qrt_norm_args * a,
                    char * err, size_t err_size);
int qrt_norm_submit(qrt_norm * n, uint32_t ring, char * err, size_t err_size);
int qrt_norm_wait(qrt_norm * n, uint32_t ring, char * err, size_t err_size);
// record + submit + wait on ring 0: the one-shot path the checks use.
int qrt_norm_run(qrt_norm * n, const qrt_norm_args * a, char * err, size_t err_size);
double qrt_norm_last_ms(const qrt_norm * n);

// Double precision reference of the same two steps, for the gate. qs_out holds
// tokens * (k/32) blocks of 32 int8; ds_out holds tokens * (k/32) words with the same
// packing as the shader. Either output may be NULL.
void qrt_norm_cpu(uint32_t tokens, uint32_t k, const float * x, const float * w, float eps,
                  float * out, signed char * qs_out, uint32_t * ds_out);
