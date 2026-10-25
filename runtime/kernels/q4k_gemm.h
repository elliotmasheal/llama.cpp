#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// Q4_K x Q8_1 int8 GEMM and the SwiGLU+Q8_1 step of one MoE expert (design 5.1
// `kernels`, M2-5). The algebra and tile map are the M2-4 probe's; the additions are
// the base offsets that let one expert block sit in VRAM and serve all three matrices.
//
// Activations are two planes: qs (32 int8 per 32-element subblock, subblock-major) and
// ds (one word per subblock: f16 delta in the low half, sum(q) + 32768 in the high half).

struct qrt_q4k;

// spv_dir holds q4k_gemm.spv and swiglu_q8.spv, as the build puts them next to qrt.
int qrt_q4k_open(qrt_vk_dev * dev, const char * spv_dir, qrt_q4k ** out,
                 char * err, size_t err_size);
void qrt_q4k_close(qrt_vk_dev * dev, qrt_q4k * g);

struct qrt_q4k_gemm_args {
    VkBuffer w;             // Q4_K rows, any offset inside
    uint32_t wbase_words;   // first weight word of row 0
    VkBuffer a_qs;          // activation planes for this reduction length
    VkBuffer a_ds;
    uint32_t abase;         // first activation subblock of token 0
    VkBuffer out;           // floats
    uint32_t obase;         // first output float of row 0
    uint32_t n;             // rows
    uint32_t k;             // reduction length, multiple of 256
    uint32_t m;             // tokens, multiple of 32
};

int qrt_q4k_gemm(qrt_q4k * g, const qrt_q4k_gemm_args * a, char * err, size_t err_size);

// out_qs / out_ds are written for m x (hid/32) subblocks.
int qrt_q4k_swiglu(qrt_q4k * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs, VkBuffer out_ds,
                   uint32_t m, uint32_t hid, char * err, size_t err_size);

// Record several dispatches into one command buffer and submit once: per expert the
// three GEMMs plus swiglu would otherwise pay four submit+fence rounds at ~0.065 ms
// each, which is the same order as the math at 64 tokens (RESULTS 39).
int qrt_q4k_begin(qrt_q4k * g, char * err, size_t err_size);
int qrt_q4k_record_gemm(qrt_q4k * g, const qrt_q4k_gemm_args * a, char * err, size_t err_size);
int qrt_q4k_record_swiglu(qrt_q4k * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs,
                          VkBuffer out_ds, uint32_t m, uint32_t hid, char * err, size_t err_size);
int qrt_q4k_submit(qrt_q4k * g, char * err, size_t err_size);
uint32_t qrt_q4k_dispatches(const qrt_q4k * g);
