#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// The device output head (design 4.6: the greedy sample happens on the GPU and only the
// token ids come back). One command buffer carries the final RMS norm with its Q8_1 packing,
// the full-vocabulary Q8_0 x Q8_1 GEMM and the two argmax stages, so a verified pass pays
// one submit for the head instead of reading 0.5 GiB of output.weight through the host.
//
// The same head serves the restricted draft vocabulary: qrt_head_set_candidates() masks the
// entries the argmax may pick, and a run either uses that mask or the whole vocabulary.
//
// Buffers this module owns: the prepared output.weight, the norm weights, the activation
// planes, the logits tile, the argmax partials, the result words and the two masks. The
// hidden rows stay the caller's buffer, read at a row offset.

#define QRT_HEAD_ROWS 32u          // the GEMM's token tile; a head run never takes more
#define QRT_HEAD_NORM_SLOTS 2u     // 0 = output_norm, 1 = the draft block's head norm

struct qrt_head;

// Bytes of the prepared output.weight for this shape. The caller reads output.weight from
// the GGUF, repacks it with qrt_gemm_q8k_prepare and hands the result to set_weights.
uint64_t qrt_head_weight_bytes(uint32_t n_embd, uint32_t vocab);
// The token tile a run of `rows` rows uses: the m1 kernel reads the weight once per token,
// the 32-row tile reads it once per pass, so the crossover sits at two rows.
uint32_t qrt_head_tile(uint32_t rows);

int  qrt_head_open(qrt_vk_dev * dev, const char * spv_dir, uint32_t n_embd, uint32_t vocab,
                   qrt_head ** out, char * err, size_t err_size);
void qrt_head_close(qrt_vk_dev * dev, qrt_head * h);

int qrt_head_set_weights(qrt_head * h, const void * prepared, uint64_t bytes, char * err,
                         size_t err_size);
int qrt_head_set_norm(qrt_head * h, uint32_t slot, const float * w, char * err, size_t err_size);

// NULL ids with n = 0 clears the mask to the whole vocabulary. Otherwise only these
// vocabulary indices can win, which is how a draft step reads its candidate table without
// gathering weight rows.
int qrt_head_set_candidates(qrt_head * h, const uint32_t * ids, uint32_t n, char * err,
                            size_t err_size);

// rows rows of the hidden plane, starting at row0, into token ids with their logit and the
// runner-up behind it. best/second may be NULL; candidates picks the masked argmax arm.
int qrt_head_run(qrt_head * h, VkBuffer x, uint32_t row0, uint32_t rows, uint32_t norm_slot,
                 int candidates, uint32_t * ids, float * best, float * second, char * err,
                 size_t err_size);

// The logits tile and the Q8_1 planes of the last run, for the numeric gates: the reference
// shares the packing decision the device made and re-does the dot product in double.
int qrt_head_logits_row(qrt_head * h, uint32_t row, float * out, char * err, size_t err_size);
int qrt_head_planes(qrt_head * h, uint32_t rows, unsigned char * qs, uint32_t * ds,
                    char * err, size_t err_size);
uint32_t qrt_head_vocab(const qrt_head * h);
uint32_t qrt_head_embd(const qrt_head * h);
double   qrt_head_last_ms(const qrt_head * h);
