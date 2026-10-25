#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// DeltaNet (gated linear attention) of qwen35moe, M2-6c. The linear-attention
// layers carry a 128x128 matrix state per v-head; every token decays it, adds a
// rank-one correction and reads one output vector out of it. That recurrence is
// what this module owns first, because it is the one part of prefill with no
// equivalent elsewhere in the runtime. The prep chain (conv1d + silu + L2 norm +
// alpha/beta/dt projections) and the gated RMS norm with ssm_out come next.
//
// Shapes are derived from the tensor table, not from new metadata keys: attn_qkv
// gives qk + v channels, ssm_norm gives head_dim, ssm_a gives n_v_heads.

struct qrt_deltanet;

struct qrt_dn_shapes {
    uint32_t n_embd;      // 2048
    uint32_t head_dim;    // 128, ssm.state_size
    uint32_t n_k_heads;   // 16, ssm.group_count
    uint32_t n_v_heads;   // 32, ssm.time_step_rank
    uint32_t d_inner;     // 4096, ssm.inner_size (v and gate channels)
    uint32_t qk_ch;       // 2048, n_k_heads * head_dim (q and k channels each)
    uint32_t conv_ch;     // 8192, q + k + v channels the conv runs over
    uint32_t conv_taps;   // 4, ssm.conv_kernel
};

// Device buffers the caller fills before qrt_dn_recur(). All are f32.
struct qrt_dn_bufs {
    VkBuffer q;         // [tokens][qk_ch], already L2 normalised and scaled
    VkBuffer k;         // [tokens][qk_ch]
    VkBuffer v;         // [tokens][d_inner]
    VkBuffer decay;     // [tokens][n_v_heads], exp() already applied
    VkBuffer beta;      // [tokens][n_v_heads], sigmoid already applied
    VkBuffer state_in;  // [n_v_heads][head_dim][head_dim]
    VkBuffer y;         // [tokens][d_inner], the raw attention output
    VkBuffer state_out; // [n_v_heads][head_dim][head_dim], may be state_in
};

// max_tokens sizes the buffers; a later chunk may be shorter.
int qrt_dn_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_dn_shapes * sh,
                uint32_t max_tokens, qrt_deltanet ** out, char * err, size_t err_size);
void qrt_dn_close(qrt_vk_dev * dev, qrt_deltanet * dn);

const qrt_dn_shapes * qrt_dn_shapes_of(const qrt_deltanet * dn);
const qrt_dn_bufs *   qrt_dn_buffers(const qrt_deltanet * dn);
uint64_t              qrt_dn_state_bytes(const qrt_deltanet * dn);

// One dispatch for the whole chunk: the shader walks the tokens in order, so a
// batch and a serial decode run the same operations in the same order.
int qrt_dn_recur(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size);
double qrt_dn_last_ms(const qrt_deltanet * dn);

// Double precision serial reference of the same recurrence, for the gate.
// state is updated in place; y must hold tokens * d_inner floats.
void qrt_dn_recur_cpu(const qrt_dn_shapes * sh, uint32_t tokens,
                      const float * q, const float * k, const float * v,
                      const float * decay, const float * beta,
                      double * state, float * y);

// Self check: real ssm_a/ssm_dt from the GGUF, synthetic q/k/v/beta, then the
// GPU recurrence against the CPU reference, the state save/restore path and the
// serial-vs-batch equality. Prints its own report lines.
int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens,
                 const char * spv_dir);
