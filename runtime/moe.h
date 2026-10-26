#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// Routed-expert batch (M3-1): one layer's top-8 routing and union-expert MoE for
// a decode batch of 1..8 tokens. The shared expert and residual stay in ffn's
// full path; this module sets routed_only so the same router/topk/gather/expert
// kernels accept a small token count instead of a 32-row chunk.

struct qrt_batch_moe;

int qrt_batch_moe_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                 const char * spv_dir, uint32_t layer, uint32_t tokens, qrt_batch_moe ** out,
                 char * err, size_t err_size);
void qrt_batch_moe_close(qrt_vk_dev * dev, qrt_batch_moe * m);

// xnorm is [tokens][n_embd], qs is [tokens][n_embd] int8 Q8_1, ds is
// [tokens][n_embd/32] u32. moe, ids and weights may be NULL when not wanted.
int qrt_batch_moe_run(qrt_batch_moe * m, const float * xnorm, const signed char * qs, const uint32_t * ds,
                float * moe, uint32_t * ids, float * weights, char * err, size_t err_size);

// CPU double reference of the same routed sum, ids and weights.
int qrt_batch_moe_ref_cpu(qrt_batch_moe * m, const float * xnorm, const signed char * qs,
                    const uint32_t * ds, float * moe, uint32_t * ids, float * weights,
                    char * err, size_t err_size);

uint32_t qrt_batch_moe_tokens(const qrt_batch_moe * m);
uint32_t qrt_batch_moe_experts(const qrt_batch_moe * m);
uint32_t qrt_batch_moe_experts_used(const qrt_batch_moe * m);
uint32_t qrt_batch_moe_pairs(const qrt_batch_moe * m);
uint32_t qrt_batch_moe_rows(const qrt_batch_moe * m);
