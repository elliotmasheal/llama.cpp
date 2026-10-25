#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// K-quant x Q8_1 int8 GEMM and the SwiGLU+Q8_1 step of one MoE expert (design 5.1
// `kernels`, M2-5). The store mixes types, so there are three kernels: Q4_K, Q5_K and
// Q6_K. The algebra and tile map are the M2-4 probe's; the additions are the base
// offsets that let one expert block sit in VRAM and serve all three matrices.
//
// Activations are two planes: qs (32 int8 per 32-element subblock, subblock-major) and
// ds (one word per subblock: f16 delta in the low half, sum(q) + 32768 in the high half).

struct qrt_gemm;

enum qrt_quant {
    QRT_QUANT_Q4K = 0,
    QRT_QUANT_Q5K = 1,
    QRT_QUANT_Q6K = 2,
};

// spv_dir holds gemm_q4k.spv, gemm_q5k.spv, gemm_q6k.spv and swiglu_q8.spv, as the
// build puts them next to qrt.
int qrt_gemm_open(qrt_vk_dev * dev, const char * spv_dir, qrt_gemm ** out,
                  char * err, size_t err_size);
void qrt_gemm_close(qrt_vk_dev * dev, qrt_gemm * g);

// Superblock bytes of a type, and the bytes a matrix of n rows by k columns needs.
size_t qrt_gemm_block_bytes(qrt_quant type);
uint64_t qrt_gemm_matrix_bytes(qrt_quant type, uint32_t n, uint32_t k);
// Picks the type from the byte size of a matrix slice, or -1 when it matches none.
int qrt_gemm_type_of(uint32_t n, uint32_t k, uint64_t bytes);

struct qrt_gemm_args {
    VkBuffer w;             // quantized rows, any offset inside
    uint32_t wbase;         // first weight word (Q4_K/Q5_K) or byte (Q6_K) of row 0
    VkBuffer a_qs;          // activation planes for this reduction length
    VkBuffer a_ds;
    uint32_t abase;         // first activation subblock of token 0
    VkBuffer out;           // floats
    uint32_t obase;         // first output float of row 0
    uint32_t n;             // rows
    uint32_t k;             // reduction length, multiple of 256
    uint32_t m;             // tokens, multiple of 32
    qrt_quant type;
};

int qrt_gemm_dispatch(qrt_gemm * g, const qrt_gemm_args * a, char * err, size_t err_size);

// out_qs / out_ds are written for m x (hid/32) subblocks.
int qrt_gemm_swiglu(qrt_gemm * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs, VkBuffer out_ds,
                    uint32_t m, uint32_t hid, char * err, size_t err_size);

// Record several dispatches into one command buffer and submit once: per expert the
// three GEMMs plus swiglu would otherwise pay four submit+fence rounds at ~0.065 ms
// each, which is the same order as the math at 64 tokens (RESULTS 39).
int qrt_gemm_begin(qrt_gemm * g, char * err, size_t err_size);
int qrt_gemm_record(qrt_gemm * g, const qrt_gemm_args * a, char * err, size_t err_size);
// The plane bases let one buffer hold every expert's swiglu output of a layer: qs is
// counted in uvec4, ds in words.
int qrt_gemm_record_swiglu(qrt_gemm * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs,
                           VkBuffer out_ds, uint32_t m, uint32_t hid, uint32_t qsbase,
                           uint32_t dsbase, char * err, size_t err_size);
// Compute-to-compute barrier: the next dispatch sees the writes of the previous ones.
// One layer of prefill records hundreds of these instead of going back to the CPU.
int qrt_gemm_record_barrier(qrt_gemm * g, char * err, size_t err_size);
int qrt_gemm_submit(qrt_gemm * g, char * err, size_t err_size);
uint32_t qrt_gemm_dispatches(const qrt_gemm * g);
