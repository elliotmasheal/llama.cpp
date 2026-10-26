#pragma once

#include <stddef.h>
#include <stdint.h>

#include "ffn.h"
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
    // Speculative width (M3-3): the number of rows a verified pass may carry. The DeltaNet
    // state ring then holds dn_snap_rows + 1 states, enough to keep one per row plus the
    // state the pass started from. 0 = no snapshots, the ring stays two slots.
    uint32_t dn_snap_rows;
    int32_t kv_k;        // enum qrt_kv_precision for the full-attention cache
    int32_t kv_v;
    uint32_t kv_page_tokens;  // page granularity; the window is the chunk in the layer check
    uint32_t n_ctx;      // positions the KV cache must hold, 0 = one chunk (the layer check)
    // A shared FFN slot (VK_NULL_HANDLE = the FFN owns a full per-layer slot). A session
    // opens 40 layers with one shared slot; each layer packs its used experts before use.
    VkBuffer ffn_slot;
    uint64_t ffn_slot_bytes;
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
// Upload only `rows` rows (the opened chunk is the buffer capacity). Decode uses 1.
int qrt_layer_set_input_rows(qrt_layer * l, const float * x, uint32_t rows, char * err,
                             size_t err_size);
// DeltaNet: the recurrence and conv states back to zero, i.e. a new sequence. No-op for a
// full-attention layer: its rows are addressed by position and rewritten.
int qrt_layer_zero_state(qrt_layer * l, char * err, size_t err_size);
// The whole opened chunk from position 0: attention half, then the FFN half when it is on.
int qrt_layer_run(qrt_layer * l, char * err, size_t err_size);
// The next `tokens` valid rows (at most the tile-aligned opened chunk) of the input as
// positions pos0 .. pos0+tokens of the sequence this layer is in the middle of: the DeltaNet
// recurrence and conv states come from the previous call and are left ready for the next
// one, the full-attention KV rows are appended at pos0 and the chunk attends to everything
// before it. The row-parallel stages still run the whole opened chunk; rows past `tokens`
// are not part of the sequence and their output is undefined.
int qrt_layer_run_chunk(qrt_layer * l, uint32_t pos0, uint32_t tokens, char * err,
                        size_t err_size);
// Speculative pass (M3-3): the same rows as run_chunk, but every DeltaNet layer also keeps
// the state after each row, so qrt_layer_rollback() can put the sequence back to any prefix
// of them. The layer must be opened with cfg.dn_snap_rows >= tokens.
int qrt_layer_run_spec(qrt_layer * l, uint32_t pos0, uint32_t tokens, char * err,
                       size_t err_size);
// Roll the sequence back to the first `kept` rows of the last speculative pass. Full
// attention needs nothing: the caller moves its own length pointer and the dropped KV rows
// are overwritten by the next pass.
int qrt_layer_rollback(qrt_layer * l, uint32_t kept, char * err, size_t err_size);
double qrt_layer_last_ms(const qrt_layer * l);
// Per-stage timings of the last run: the attention pass, the host group, the FFN pass.
double qrt_layer_pass1_ms(const qrt_layer * l);
double qrt_layer_gpu_ms(const qrt_layer * l);  // submit->fence, both passes
double qrt_layer_group_ms(const qrt_layer * l);
double qrt_layer_pass2_ms(const qrt_layer * l);
uint32_t qrt_layer_submits(const qrt_layer * l);
uint32_t qrt_layer_dispatches(const qrt_layer * l);
uint32_t qrt_layer_experts(const qrt_layer * l);
uint32_t qrt_layer_rows(const qrt_layer * l);
uint64_t qrt_layer_store_bytes(const qrt_layer * l);
uint64_t qrt_layer_uploaded_bytes(const qrt_layer * l);
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

// Decode expert cache (M3-2a): forward the layer's FFN to the shared cache policy with
// this layer's own slot buffer. The caller keeps the cache, the CPU core and the
// pass-level leg carry.
struct qrt_cache;
struct qrt_cpu_moe;
int qrt_layer_set_cache(qrt_layer * l, qrt_cache * cache, VkBuffer buf, uint64_t slot_bytes,
                        qrt_cpu_moe * cpu, int miss_split, const qrt_split_rates * rates,
                        qrt_split_carry * carry, char * err, size_t err_size);
qrt_ffn_legs qrt_layer_last_legs(const qrt_layer * l);
// Overwrites out with the CPU leg's routed sum [tokens][n_embd]; 0 when there was none.
int qrt_layer_take_cpu_out(qrt_layer * l, float * out, uint32_t tokens);

// Per-stage comparison against one CPU double chain over the same weights and the same
// Q8_1 planes; prints one line per stage and returns 0 when every gate holds.
int qrt_layer_check(const char * gguf_path, const char * store_prefix, uint32_t layer,
                    uint32_t tokens, const qrt_layer_cfg * cfg, const char * spv_dir);

// Optional per-stage outputs of the reference chain, for localising a divergence inside
// one layer. The caller allocates each buffer it wants (the sizes are the same ones the
// device buffers have) and passes NULL for the rest.
struct qrt_layer_ref_trace {
    float * xnorm;     // [tokens][n_embd], the input norm
    float * attn_src;  // [tokens][inter],     the attention path's own output
    float * attn_out;  // [tokens][n_embd],    the output projection
    float * x_res;     // [tokens][n_embd],    after the first residual add
    float * post_norm; // [tokens][n_embd],    the post-attention norm
    float * moe;       // [tokens][n_embd],    the routed sum
    float * shexp;     // [tokens][n_embd],    the shared expert before its gate
};

// One layer of the whole-model reference chain: x_in ([tokens][n_embd]) to x_out, computed
// entirely on the host. Nothing is read back from the device, so unlike the per-stage check
// this can carry a residual stream across layers (M2-7a). The Q8_1 planes, the KV rows and
// the expert weights all come from the host copies the layer already holds.
//
// Only the f16 KV arm is modelled: the reference stages K and V through the same truncating
// half conversion the append kernel does. A quantized KV row would need the reference to
// reproduce the block-max search the append kernel runs, which is not what this chain is for
// (the row formats are gated by --full-check). It fails instead of silently comparing
// against a different precision.
// device_planes = 1 is localisation only: the reference takes the layer's own Q8_1 planes of
// the attention output and of the post-attention norm instead of packing its own, which
// separates the arithmetic difference from the quantiser's boundary flips. The input norm's
// planes are not reusable (step 7 overwrites that buffer), so they stay host-computed.
int qrt_layer_ref_cpu(qrt_layer * l, const float * x_in, float * x_out,
                      struct qrt_layer_ref_trace * trace, int device_planes,
                      char * err, size_t err_size);
