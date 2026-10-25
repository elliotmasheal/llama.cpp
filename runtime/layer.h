#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// One transformer block's attention half (M2-6, the layer chain): input RMS norm, the q8_0
// projections, the attention path (DeltaNet's four legs or full attention), the output
// projection, the residual add and the post-attention norm. The FFN half stays with the
// expert leg (M2-5) and the shared expert, so this module stops at the post norm.
//
// The pieces are the existing legs wired in the order the model runs them: every stage's
// buffers are the next stage's inputs, nothing is copied through the host. The chain is the
// thing under test here, so `--layer-check` compares it against one CPU double chain and not
// against each kernel separately (DECISIONS 63.4).

struct qrt_layer;

struct qrt_layer_info {
    uint32_t n_embd;   // 2048
    uint32_t tokens;   // chunk length this layer was opened for
    uint32_t inter;    // output projection's k: d_inner (4096) or n_head * head_dim (4096)
    int      linear;   // 1 = DeltaNet layer, 0 = full attention
};

// Shapes come from the tensor table, never from new metadata keys. layer selects the block.
int qrt_layer_open(qrt_vk_dev * dev, const char * gguf_path, uint32_t layer, uint32_t tokens,
                   const char * spv_dir, qrt_layer ** out, char * err, size_t err_size);
void qrt_layer_close(qrt_vk_dev * dev, qrt_layer * l);
const qrt_layer_info * qrt_layer_shape(const qrt_layer * l);

// The residual stream the layer starts from; also the buffer the residual add writes.
int qrt_layer_set_input(qrt_layer * l, const float * x, char * err, size_t err_size);
// DeltaNet: the recurrence and conv states back to zero. No-op for a full-attention layer.
int qrt_layer_zero_state(qrt_layer * l, char * err, size_t err_size);
// One chunk from position 0: input norm -> projections -> attention -> output projection ->
// residual add -> post norm. The recurrence state and the conv state start at zero.
int qrt_layer_run(qrt_layer * l, char * err, size_t err_size);
double qrt_layer_last_ms(const qrt_layer * l);

// What the next half consumes, and what the check reads.
VkBuffer qrt_layer_x(const qrt_layer * l);          // residual stream after the add
VkBuffer qrt_layer_xnorm(const qrt_layer * l);      // input norm output
VkBuffer qrt_layer_attn_out(const qrt_layer * l);   // output projection result
VkBuffer qrt_layer_post_norm(const qrt_layer * l);  // post-attention norm output
VkBuffer qrt_layer_attn_src(const qrt_layer * l);   // the attention path's own output

// Per-stage comparison against one CPU double chain over the same weights and the same
// Q8_1 planes; prints one line per stage and returns 0 when every gate holds.
int qrt_layer_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir);
