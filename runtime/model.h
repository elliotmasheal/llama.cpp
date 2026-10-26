#pragma once

#include <stddef.h>
#include <stdint.h>

struct qrt_config;

struct qrt_model_options {
    uint32_t n_layers = 40;
    uint32_t n_rows = 8;
    const char * out_path = nullptr;
    const char * ref_out_path = nullptr;
    int reseed = 0;
    int device_planes = 0;
    uint32_t batch_k = 1;   // decode rows per pass; >1 is a diagnostic, not acceptance
};

// Whole-model prefill path (M2-7a): embedding row -> the 40 main layers in order ->
// final norm -> output head, and the fp32 logits dump the quality gate reads
// (bench/tools/quality/README.md section 3, magic "QRTLOG1\0", version 1).
//
// The layers are the same qrt_layer chain the layer check drives; one chunk of the prompt is
// run by opening a layer, feeding it the previous layer's output and closing it again, so only
// one layer's weights are resident at a time. A prompt longer than the chunk runs layer first
// (design 4.5): the layer stays open while every chunk of the prompt goes through it, and the
// per-sequence state (the DeltaNet recurrence and conv taps, the full-attention KV rows and
// their positions) is carried from one chunk to the next by the layer itself.
//
// token_embd and output are q8_0 rows and are done on the host: the design keeps
// token_embd out of the dense bucket (weights.h) and the head is 248320 x 2048 per row.
int qrt_model_logits(const qrt_config * cfg, const qrt_model_options * opt,
                     const char * gguf_path, const char * store_prefix, const char * spv_dir,
                     const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size);

// The end-to-end vertical slice: open all requested layers once, prefill the prompt in
// prefill_chunk-sized chunks, then greedy-decode n_new tokens with KV/DeltaNet state
// carried. The head stays on the host for this slice; no per-token prompt rerun.
int qrt_model_generate(const qrt_config * cfg, const qrt_model_options * opt,
                       const char * gguf_path, const char * store_prefix, const char * spv_dir,
                       const uint32_t * ids, uint32_t n_tokens, uint32_t n_new,
                       const char * ids_out, char * err, size_t err_size);

// The acceptance run (M2-7a): the device chain above, and beside it one independent CPU
// double chain that never reads a device intermediate (layer.h qrt_layer_ref_cpu). Prints
// the per-layer hidden rel_L1, a re-run of the first layer of each type against the chain
// hidden, and the rel_L1 of the two arms' logits; returns 0 when the logits hold 1e-4.
// reseed 1 starts every reference layer from the device hidden, which isolates one layer
// instead of accumulating the chain. Only the f16 KV arm can be referenced: a quantized KV
// row would need the block-max search of the append kernel, which --full-check gates.
// The reference chain needs cfg.prefill_chunk to cover the whole prompt.
int qrt_model_check(const qrt_config * cfg, const qrt_model_options * opt,
                    const char * gguf_path, const char * store_prefix, const char * spv_dir,
                    const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size);

// The cross-chunk state check (M2-7a): three device-only arms over the same prompt -
// one chunk, the same prompt in chunks with the layer state carried, and the same chunks
// with the DeltaNet state zeroed again before each one (what the chain did before it carried
// anything; the attention positions stay correct in that arm, so it isolates the recurrence
// and conv carry). Every row-parallel stage is independent of the split and the recurrence is
// token serial, so the carried arm is expected to be bit-identical to the single chunk; the
// zeroed arm is the counterexample that shows the comparison can see lost state.
int qrt_model_chunk_check(const qrt_config * cfg, const qrt_model_options * opt,
                          const char * gguf_path, const char * store_prefix, const char * spv_dir,
                          const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size);

// The M3-3 acceptance run (design 7): decode the prompt twice from the same state, once
// serially and once as speculative cycles of up to spec_k rows whose drafts are broken one
// position at a time. The two arms must emit the same tokens, and every verified row that was
// fed a true token must match the serial arm's logits for the same index. rollback_off keeps
// every row instead, which is the counterexample that shows the comparison can see the loss.
int qrt_model_spec_check(const qrt_config * cfg, const qrt_model_options * opt,
                         const char * gguf_path, const char * store_prefix, const char * spv_dir,
                         const uint32_t * ids, uint32_t n_tokens, uint32_t n_new,
                         int rollback_off, char * err, size_t err_size);

// Reads a whitespace-separated token id file. Returns 0 on success.
int qrt_model_read_ids(const char * path, uint32_t ** ids, uint32_t * count,
                       char * err, size_t err_size);
