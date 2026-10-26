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
    VkBuffer state;     // the state ring, [slots][n_v_heads][head_dim][head_dim]
    VkBuffer y;         // [tokens][d_inner], the raw attention output
    // step 2a: the prep chain around the recurrence
    VkBuffer qkv;       // [tokens][conv_ch]  projection output (input of the conv)
    VkBuffer conv_state; // [conv_taps-1][conv_ch], may be NULL until written
    VkBuffer xnorm;     // [tokens][n_embd]  normalised input, input of the gate GEMV
    VkBuffer z;         // [tokens][d_inner] gate projection output (silu input)
    VkBuffer gnorm;     // [tokens][d_inner] gated norm output (ssm_out input)
    VkBuffer ssm_norm_w; // [head_dim] the gated norm weight
    VkBuffer norm;      // [tokens][n_embd] f32 rms-norm output (the gate input)
};

// The state ring (M3-3): a chunk reads slot `cur` and writes slot (cur + 1) % slots, so the
// carry between chunks is a cursor move instead of a 2 MiB copy per layer. qrt_dn_recur_snap()
// additionally leaves one state per token in the ring, which is what speculative rollback
// reads back after accepting j of k rows. state_slots must be at least the largest snap
// token count plus one, and at least two.
int qrt_dn_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_dn_shapes * sh,
                uint32_t max_tokens, uint32_t state_slots, qrt_deltanet ** out,
                char * err, size_t err_size);
void qrt_dn_close(qrt_vk_dev * dev, qrt_deltanet * dn);

const qrt_dn_shapes * qrt_dn_shapes_of(const qrt_deltanet * dn);
const qrt_dn_bufs *   qrt_dn_buffers(const qrt_deltanet * dn);
uint64_t              qrt_dn_state_bytes(const qrt_deltanet * dn);
uint32_t              qrt_dn_state_slots(const qrt_deltanet * dn);
// The slot the next recurrence reads, and the byte offset of that slot inside the ring.
uint32_t              qrt_dn_state_cur(const qrt_deltanet * dn);
uint64_t              qrt_dn_state_offset(const qrt_deltanet * dn);
// Point the sequence at another slot: rollback after accepting j rows uses the snapshot of
// row j. Out-of-range slots fail the next recurrence rather than reading past the ring.
void                  qrt_dn_state_use(qrt_deltanet * dn, uint32_t slot);

// One dispatch for the whole chunk: the shader walks the tokens in order, so a
// batch and a serial decode run the same operations in the same order.
int qrt_dn_recur(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size);
// The same dispatch with a per-token snapshot. The cursor stays where it was: the caller
// decides which snapshot becomes live once the accepted length is known.
int qrt_dn_recur_snap(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size);
// Kernel variant: 0 = two state columns per workgroup with two subgroup reductions
// per token (the original), 1 = 32 columns per workgroup with a thread-per-row-slice
// split (shared memory sum). Both keep the token loop inside the shader, so
// batch == serial holds either way.
int qrt_dn_set_recur(qrt_deltanet * dn, int cols);
double qrt_dn_last_ms(const qrt_deltanet * dn);

// Pass mode (design 4.6): the four legs append to the caller's command buffer and each
// inserts a compute barrier after itself; the caller submits once.
int qrt_dn_attach(qrt_deltanet * dn, VkCommandBuffer cmd, char * err, size_t err_size);
void qrt_dn_detach(qrt_deltanet * dn);

// Double precision serial reference of the same recurrence, for the gate.
// state is updated in place; y must hold tokens * d_inner floats.
void qrt_dn_recur_cpu(const qrt_dn_shapes * sh, uint32_t tokens,
                      const float * q, const float * k, const float * v,
                      const float * decay, const float * beta,
                      double * state, float * y);

// Step 2a: the three prep kernels with already projected inputs, so each can be gated on
// its own before the GEMMs are wired in (step 2b).
//   prep : conv1d(conv_taps) + silu + split + per-head L2 norm of q and k
//   gate : alpha/beta GEMV + decay = exp(a*softplus(alpha+dt)), beta = sigmoid
//   gnorm: gated RMS norm of the recurrence output, the input of ssm_out
struct qrt_dn_weights;
int qrt_dn_weights_load(qrt_vk_dev * dev, const char * gguf_path, uint32_t layer,
                        const qrt_dn_shapes * sh, qrt_dn_weights ** out, char * err, size_t err_size);
void qrt_dn_weights_free(qrt_vk_dev * dev, qrt_dn_weights * w);
VkBuffer qrt_dn_weights_conv(const qrt_dn_weights * w);
// Reads every weight buffer back and compares it with the host copy; weights_load calls
// it, so a broken upload fails the load instead of surfacing as wrong logits later. The
// conv matrix made this necessary: it was uploaded from a scratch vector the gate weights
// had already overwritten (DECISIONS 65, RESULTS 70).
int qrt_dn_weights_verify(qrt_vk_dev * dev, const qrt_dn_weights * w,
                          char * err, size_t err_size);
// host copies for the double precision references: 0 conv, 1 alpha, 2 beta, 3 dt, 4 a
const float * qrt_dn_weights_host(const qrt_dn_weights * w, uint32_t which);

int qrt_dn_prep_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    float l2_eps, char * err, size_t err_size);
// gate: alpha/beta GEMV over the normalised input, then
//   decay = exp(ssm_a * softplus(alpha + ssm_dt)), beta = sigmoid(beta_raw)
int qrt_dn_gate_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    char * err, size_t err_size);
// gnorm: out = rms_norm(y, eps) * ssm_norm * silu(z)
int qrt_dn_gnorm_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                     char * err, size_t err_size);
// The four legs in sequence for one linear-attention layer: prep -> gate -> recur ->
// gnorm. The caller fills qkv, xnorm and z (the projection outputs) beforehand, which
// is exactly what the layer chain will hand over; this exists so the pieces are also
// checked in the order the model runs them. snap=1 keeps a state snapshot per token.
int qrt_dn_layer_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                     float l2_eps, int snap, char * err, size_t err_size);
double qrt_dn_layer_ms(const qrt_deltanet * dn);

void qrt_dn_gnorm_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * y,
                      const float * z, const float * norm_w, float eps, float * out);
void qrt_dn_gate_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * xnorm,
                     const float * alpha_w, const float * beta_w, const float * dt,
                     const float * a, float * decay, float * beta);
void qrt_dn_prep_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * qkv,
                     const float * conv_w, const float * conv_state_in, float l2_eps,
                     float * q, float * k, float * v, float * conv_state_out);
int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens,
                 const char * spv_dir);
