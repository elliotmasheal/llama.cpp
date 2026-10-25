#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// One transformer block (M2-6, the layer chain): input RMS norm, the q8_0 projections, the
// attention path (DeltaNet's four legs or full attention), the output projection, the
// residual add, the post-attention norm, the FFN half (gated shared expert + routed MoE),
// and the final residual add.
//
// The pieces are the existing legs wired in the order the model runs them: every stage's
// buffers are the next stage's inputs, nothing is copied through the host except the
// router's top-k, which decides the expert GEMM grids and so has to reach the host
// (design 5.2 route_ids). With the FFN on, a layer is two merged submits: the attention
// half plus the router, then the experts. Without it, one.
//
// The chain is the thing under test here, so `--layer-check` compares it against one CPU
// double chain and not against each kernel separately (DECISIONS 63.4).

struct qrt_layer;

struct qrt_layer_cfg {
    int ffn;             // 1 = include the shared expert and the routed MoE
    int single_submit;   // 1 = record the whole layer into one pass buffer
    int dn_recur;        // 1 = the 32-column DeltaNet recurrence (RESULTS 63)
    int32_t kv_k;        // enum qrt_kv_precision for the full-attention cache
    int32_t kv_v;
    uint32_t kv_page_tokens;  // page granularity; the window is the chunk in the layer check
};

struct qrt_layer_info {
    uint32_t n_embd;   // 2048
    uint32_t tokens;   // chunk length this layer was opened for
    uint32_t inter;    // output projection's k: d_inner (4096) or n_head * head_dim (4096)
    uint32_t n_expert; // 256, 0 when the FFN half is off
    uint32_t top_k;    // 8
    uint32_t ffn;      // 512, expert and shared intermediate
    int      linear;   // 1 = DeltaNet layer, 0 = full attention
};

// Shapes come from the tensor table, never from new metadata keys. layer selects the block;
// store_prefix is the repacked expert store and is needed for the FFN half.
int qrt_layer_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                   uint32_t layer, uint32_t tokens, const char * spv_dir,
                   const qrt_layer_cfg * cfg, qrt_layer ** out, char * err, size_t err_size);
void qrt_layer_close(qrt_vk_dev * dev, qrt_layer * l);
const qrt_layer_info * qrt_layer_shape(const qrt_layer * l);

// The residual stream the layer starts from; also the buffer the residual add writes.
int qrt_layer_set_input(qrt_layer * l, const float * x, char * err, size_t err_size);
// DeltaNet: the recurrence and conv states back to zero. No-op for a full-attention layer.
int qrt_layer_zero_state(qrt_layer * l, char * err, size_t err_size);
// One chunk from position 0: attention half, then the FFN half when it is on.
int qrt_layer_run(qrt_layer * l, char * err, size_t err_size);
double qrt_layer_last_ms(const qrt_layer * l);
// Per-stage timings of the last run: the attention pass, the host group, the FFN pass.
double qrt_layer_pass1_ms(const qrt_layer * l);
double qrt_layer_group_ms(const qrt_layer * l);
double qrt_layer_pass2_ms(const qrt_layer * l);
uint32_t qrt_layer_submits(const qrt_layer * l);
uint32_t qrt_layer_dispatches(const qrt_layer * l);
uint32_t qrt_layer_experts(const qrt_layer * l);
uint32_t qrt_layer_rows(const qrt_layer * l);
uint64_t qrt_layer_store_bytes(const qrt_layer * l);
uint64_t qrt_layer_plane_bytes(const qrt_layer * l);

// What the next half consumes, and what the check reads.
VkBuffer qrt_layer_x(const qrt_layer * l);          // residual stream after the add
VkBuffer qrt_layer_xnorm(const qrt_layer * l);      // input norm output
VkBuffer qrt_layer_attn_out(const qrt_layer * l);   // output projection result
VkBuffer qrt_layer_post_norm(const qrt_layer * l);  // post-attention norm output
VkBuffer qrt_layer_attn_src(const qrt_layer * l);   // the attention path's own output
// The layer's output row: the FFN's residual add when the FFN is on, x otherwise.
VkBuffer qrt_layer_out(const qrt_layer * l);
// FFN buffers, NULL-handed when the FFN half is off.
struct qrt_ffn_bufs;
const struct qrt_ffn_bufs * qrt_layer_ffn_bufs(const qrt_layer * l);
struct qrt_ffn * qrt_layer_ffn(qrt_layer * l);

// Per-stage comparison against one CPU double chain over the same weights and the same
// Q8_1 planes; prints one line per stage and returns 0 when every gate holds.
int qrt_layer_check(const char * gguf_path, const char * store_prefix, uint32_t layer,
                    uint32_t tokens, const qrt_layer_cfg * cfg, const char * spv_dir);
