#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// FFN half of one layer (M2-6, design 5.1 `kernels`): the router over f32 weights, the
// sigmoid-gated shared expert over the Q8_0 dense GEMM, and the routed experts read from
// the repacked store. The layer records the router into its first pass, reads the top-k
// back (the ids decide the expert GEMM grids, so they have to reach the host), uploads the
// layer's expert blocks, and records the rest into its second pass.
//
// The routed path is a three-step dataflow: one gather that copies each routed token's
// Q8_1 planes into its expert's padded row block, the per-expert gate/up/swiglu/down over
// that block, and a weighted accumulate into the token rows. Padding rows are zero planes,
// so they cost arithmetic but cannot change a result.

struct qrt_ffn;
struct qrt_gemm;
struct qrt_weights;

struct qrt_ffn_cfg {
    uint32_t tokens;    // chunk length, a multiple of 32
    uint32_t n_embd;    // 2048
    uint32_t n_expert;  // 256
    uint32_t top_k;     // 8
    uint32_t ffn;       // 512, expert and shared intermediate
    float    scale;     // expert_weights_scale, 1.0 when the model sets none
};

int qrt_ffn_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                 const char * spv_dir, uint32_t layer, const qrt_ffn_cfg * cfg,
                 qrt_ffn ** out, char * err, size_t err_size);
void qrt_ffn_close(qrt_vk_dev * dev, qrt_ffn * f);

// The buffers the layer hands over: the residual stream (read in place), the post-attention
// norm row and its Q8_1 planes, and where the FFN leaves the layer output.
struct qrt_ffn_in {
    VkBuffer x;
    VkBuffer xnorm;
    VkBuffer x_qs;
    VkBuffer x_ds;
    VkBuffer out;
};

// Pass mode: begin waits the ring and resets the descriptor pools, then the two record
// calls append to cmd.
int qrt_ffn_begin(qrt_ffn * f, uint32_t ring, VkCommandBuffer cmd, char * err,
                  size_t err_size);
// Phase 1: router logits, softmax + top-k, and the shared expert's sigmoid gate.
int qrt_ffn_route(qrt_ffn * f, const qrt_ffn_in * in, char * err, size_t err_size);
// Between the passes: reads ids and weights back, groups the tokens per expert, uploads
// the row table and this layer's expert blocks. Counts as host work, not compute.
int qrt_ffn_group(qrt_ffn * f, char * err, size_t err_size);
// Phase 2: gather, shared expert, routed experts, final residual add.
int qrt_ffn_experts(qrt_ffn * f, char * err, size_t err_size);

uint32_t qrt_ffn_pairs(const qrt_ffn * f);      // routed (token, expert) pairs
uint32_t qrt_ffn_rows(const qrt_ffn * f);       // rows after padding to 32
uint32_t qrt_ffn_experts_used(const qrt_ffn * f);
uint32_t qrt_ffn_dispatches(const qrt_ffn * f);
uint64_t qrt_ffn_store_bytes(const qrt_ffn * f);
uint64_t qrt_ffn_planes_bytes(const qrt_ffn * f);

struct qrt_ffn_bufs {
    VkBuffer logits;    // [tokens][n_expert] f32
    VkBuffer ids;       // [tokens][top_k] u32
    VkBuffer weights;   // [tokens][top_k] f32
    VkBuffer sgate;     // [tokens] f32, the shared expert's pre-sigmoid gate
    VkBuffer shexp;     // [tokens][n_embd] f32, gated shared expert output
    VkBuffer moe;       // [tokens][n_embd] f32, the routed sum
    VkBuffer out;       // [tokens][n_embd] f32, x + moe + shexp * sigmoid(gate)
};
const qrt_ffn_bufs * qrt_ffn_buffers(const qrt_ffn * f);

// Double precision reference of the FFN half: router logits/ids/weights for ref_tokens,
// the shared expert for ref_tokens, and the routed experts for the pairs that start in
// those tokens. out is x + moe + shexp*sigmoid(gate) for ref_tokens * n_embd floats; the
// other outputs may be NULL. Kept in the module so the check and the kernel read the same
// host copies of the weights. xpost_qs/xpost_ds are the DEVICE's Q8_1 planes of xpost
// when the caller has them: using them keeps the quantiser's 0.5 boundary decisions out of
// the gate, which is what the output projection stage already does.
int qrt_ffn_ref_cpu(qrt_ffn * f, const float * xpost, const signed char * xpost_qs,
                    const uint32_t * xpost_ds, uint32_t ref_tokens, float * logits,
                    uint32_t * ids, float * weights, float * sgate, float * shexp,
                    float * moe, char * err, size_t err_size);
