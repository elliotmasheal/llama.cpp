#pragma once

#include <stddef.h>
#include <stdint.h>

// Whole-model prefill path (M2-7a): embedding row -> the 40 main layers in order ->
// final norm -> output head, and the fp32 logits dump the quality gate reads
// (bench/tools/quality/README.md section 3, magic "QRTLOG1\0", version 1).
//
// The layers are the same qrt_layer chain the layer check drives; one chunk is run by
// opening a layer, feeding it the previous layer's output and closing it again, so only
// one layer's weights are resident at a time. Cross-chunk KV/DeltaNet state is not part
// of this first slice (a single chunk from position 0 needs none).
//
// token_embd and output are q8_0 rows and are done on the host: the design keeps
// token_embd out of the dense bucket (weights.h) and the head is 248320 x 2048 per row.
int qrt_model_logits(const char * gguf_path, const char * store_prefix, const char * spv_dir,
                     const uint32_t * ids, uint32_t n_tokens, uint32_t n_layers,
                     uint32_t n_rows, const char * out_path, char * err, size_t err_size);

// Reads a whitespace-separated token id file. Returns 0 on success.
int qrt_model_read_ids(const char * path, uint32_t ** ids, uint32_t * count,
                       char * err, size_t err_size);
