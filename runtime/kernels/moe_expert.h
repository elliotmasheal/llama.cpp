#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kernels/q4k_gemm.h"
#include "vk_dev.h"

// One MoE expert's forward pass on the GPU, composed from the int8 GEMM and the
// swiglu+quantize kernels (design 4.5 step 2). It plugs into qrt_prefill_run as the
// compute callback: the scheduler uploads an expert block into a slot and this runs
// gate, up, swiglu, down on it, all in one submit.

struct qrt_weights;
struct qrt_moe;

struct qrt_moe_cfg {
    uint32_t tokens;      // tokens routed to one expert, multiple of 32
    uint32_t hidden;      // 2048, from the model
    uint32_t ffn;         // 512, the expert row count of gate/up
};

int qrt_moe_open(qrt_vk_dev * dev, qrt_weights * w, const char * spv_dir,
                 const qrt_moe_cfg * cfg, qrt_moe ** out, char * err, size_t err_size);
void qrt_moe_close(qrt_vk_dev * dev, qrt_moe * m);

uint32_t qrt_moe_tokens(const qrt_moe * m);
// gate + up multiply-adds for one expert, as 2 x n x k x tokens each.
uint64_t qrt_moe_flops(const qrt_moe * m);
// Experts whose down matrix was not Q4_K (Q5_K or Q6_K in this store) and so was
// skipped: the GEMM is Q4_K only until the other block formats land.
uint32_t qrt_moe_down_skipped(const qrt_moe * m);
uint32_t qrt_moe_experts_run(const qrt_moe * m);

// qrt_prefill_compute_fn; buf is the scheduler's slot buffer, slot_offset the block.
int qrt_moe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                    uint64_t slot_offset, uint64_t bytes, char * err, size_t err_size);

// Compares the gate output of one expert against a double-precision dequant reference
// (the canonical Q4_K order of ggml dequantize_row_q4_K). Returns the relative L1 error
// over the sampled rows, or -1 with a message.
int qrt_moe_check(qrt_vk_dev * dev, qrt_moe * m, qrt_weights * w, uint32_t layer,
                  uint32_t expert, uint32_t rows, double * rel_l1, char * err, size_t err_size);
