#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// Full-attention layer of qwen35moe (M2-6d): per-head RMS norm of Q and K, IMRoPE,
// causal attention over an f16 KV cache, and the sigmoid gate that the joint QG
// projection carries in the second half of every head.
//
// Shapes come from the metadata (attention.head_count / head_count_kv,
// rope.dimension_count, rope.frequency_base, attention.key_length). For this model:
// n_head 16, n_kv 2, head_dim 256, n_rot 64, sections {11,11,10,0}, freq_base 1e7.
//
// Layouts the kernels expect (the dense GEMM writes token-major for these three
// projections; DECISIONS 60.2 notes that the output layout is the host's choice):
//   qg   [tokens][n_head * 2 * head_dim]   Q in the first head_dim of each head, gate next
//   kin  [tokens][n_kv * head_dim]
//   vin  [tokens][n_kv * head_dim]
//   q    [tokens][n_head * head_dim]       normed and roped Q
//   kv   [n_kv][n_ctx][head_dim] f16       normed/roped K and raw V, appended per chunk
//   out  [tokens][n_head * head_dim]       attention output already multiplied by sigmoid(gate)

struct qrt_full;

enum { FULL_BUFFER_COUNT = 9 };

struct qrt_full_shapes {
    uint32_t n_embd;
    uint32_t head_dim;    // 256, attention.key_length
    uint32_t n_head;      // 16
    uint32_t n_kv;        // 2
    uint32_t n_rot;       // 64, rope.dimension_count
    int32_t  sections[4]; // rope.dimension_sections
    float    freq_base;   // rope.frequency_base
    float    freq_scale;  // 1.0 unless the metadata scales the rope
    float    eps;         // attention.layer_norm_rms_epsilon
    float    kq_scale;    // 1/sqrt(head_dim) unless the model sets a scale
    uint32_t qg_ch;       // n_head * 2 * head_dim
    uint32_t kv_ch;       // n_kv * head_dim
};

struct qrt_full_bufs {
    VkBuffer qg;
    VkBuffer kin;
    VkBuffer vin;
    VkBuffer q;
    VkBuffer kcache;
    VkBuffer vcache;
    VkBuffer out;
    VkBuffer q_norm_w;   // [head_dim]
    VkBuffer k_norm_w;   // [head_dim]
};

// Shapes from the GGUF metadata (attention.head_count / head_count_kv / key_length,
// rope.dimension_count / freq_base / dimension_sections). Fails on a head_dim the
// kernels cannot tile rather than computing something wrong.
struct qrt_gguf_hparams;
int qrt_full_shapes_from_gguf(const qrt_gguf_hparams * hp, qrt_full_shapes * out,
                              char * err, size_t err_size);

// max_tokens sizes the per-chunk planes; n_ctx sizes the KV cache pages.
int qrt_full_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_full_shapes * sh,
                  uint32_t max_tokens, uint32_t n_ctx, qrt_full ** out,
                  char * err, size_t err_size);
void qrt_full_close(qrt_vk_dev * dev, qrt_full * f);

const qrt_full_shapes * qrt_full_shapes_of(const qrt_full * f);
const qrt_full_bufs *   qrt_full_buffers(const qrt_full * f);
// Bytes of one KV plane (K or V); the pair is twice this. f16 storage.
uint64_t                qrt_full_kv_bytes(const qrt_full * f);
// Step 1: per-head RMS norm of Q and K, IMRoPE on both, V copied as is. Writes q,
// kcache[pos0 .. pos0+tokens) and vcache[pos0 .. pos0+tokens).
int qrt_full_qk(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size);

// Step 2: causal attention of the chunk against kcache/vcache[0 .. pos0+tokens),
// one head per workgroup, tiles of queries and keys, online softmax; the output is
// multiplied by sigmoid(gate) read from qg.
int qrt_full_attn(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size);
double qrt_full_last_ms(const qrt_full * f);
// KV reads and tokens the attention kernel handled in the last call, for the
// "read the KV once" account: bytes = tiles * tile_tokens * head_dim * 2 * 2.
uint64_t qrt_full_last_kv_reads(const qrt_full * f);

// Double precision references over f32 planes.
// q_out is [tokens][n_head*head_dim], k_out/v_out are [n_kv][tokens][head_dim].
void qrt_full_qk_cpu(const qrt_full_shapes * sh, uint32_t pos0, uint32_t tokens,
                     const float * qg, const float * kin, const float * vin,
                     const float * q_norm_w, const float * k_norm_w,
                     float * q_out, float * k_out, float * v_out);
// Attention rows `rows[i]` (token indices from 0) over their causal range, with the
// sigmoid gate. kcache/vcache are the dequantized planes [n_kv][n_ctx][head_dim], so
// the arithmetic is compared without the f16 storage error folded in again. `out`
// gets one row per entry of `rows`, in that order.
void qrt_full_attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx,
                            const uint32_t * rows, uint32_t n_rows,
                            const float * q, const float * kcache, const float * vcache,
                            const float * qg, float * out);

int qrt_full_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir);
