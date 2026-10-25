#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kernels/gemm.h"
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

// qrt_prefill_batch_fn: the whole layer in one command buffer and one submit. Every
// expert's swiglu output goes to its own slice of the hidden planes, and the expert
// outputs rotate over eight slices because accumulation into the token rows is not
// implemented yet (DECISIONS 37).
int qrt_moe_compute_batch(void * user, uint32_t layer, uint32_t ring, VkBuffer buf,
                          const uint32_t * experts, const uint64_t * offsets,
                          const uint64_t * bytes, uint32_t count, char * err, size_t err_size);
// qrt_prefill_wait_fn: waits for the submit of the given ring to finish.
int qrt_moe_wait(void * user, uint32_t ring, char * err, size_t err_size);

// Per-matrix result of the cross-check below.
struct qrt_moe_check_result {
    double   gate_rel_l1;
    double   down_rel_l1;
    // gate -> silu*up -> Q8_1 -> down against a CPU chain: qrt_moe_compute (one slot) and
    // qrt_moe_compute_batch (a whole layer) must both reproduce it.
    double   chain_rel_l1;
    double   batch_rel_l1;
    uint32_t gate_type;   // qrt_quant of the gate matrix of this expert
    uint32_t down_type;
};

// Compares the gate, down and whole-expert outputs against double-precision dequant
// references (the canonical ggml dequant orders for all three K-quants) over the first
// `rows` output rows; the two chain legs use every row. The reference is token-major,
// the layout the kernels write. Returns -1 with a message on a structural failure.
int qrt_moe_check(qrt_vk_dev * dev, qrt_moe * m, qrt_weights * w, uint32_t layer,
                  uint32_t expert, uint32_t rows, qrt_moe_check_result * out, char * err,
                  size_t err_size);

// One row of a K-quant matrix times Q8_1 planes, double precision, in the same dequant
// order the kernels and --moe-check use. The FFN check reuses it so the two agree.
void qrt_moe_ref_row(const void * w_row, qrt_quant type, const unsigned char * a_qs,
                     const uint32_t * a_ds, uint32_t k, uint32_t tokens, double * out);
