#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// Host side of the repacked expert store (design 5.1 `weights`, M2-3).
// Reads <prefix>.qwen36rt.idx into a page table and hands out expert byte
// ranges from <prefix>.qwen36rt.bin. No VRAM is touched here.

struct qrt_weights;

struct qrt_expert_loc {
    uint64_t offset;   // relative to the payload start
    uint64_t bytes;
    uint64_t gate;
    uint64_t up;
    uint64_t down;
};

// prefix is the path without .qwen36rt.{bin,idx}.
int qrt_weights_open(const char * prefix, qrt_weights ** out, char * err, size_t err_size);
void qrt_weights_close(qrt_weights * w);

uint32_t qrt_weights_layers(const qrt_weights * w);
uint32_t qrt_weights_experts(const qrt_weights * w);
uint64_t qrt_weights_payload_bytes(const qrt_weights * w);
uint32_t qrt_weights_records(const qrt_weights * w);
const char * qrt_weights_arch(const qrt_weights * w);
// The largest and smallest expert block. They differ per layer: layer 39 is
// 2301952 B while 38 of the 41 layers are 1900544 B (RESULTS 35), so a cache slot
// sized from the mean is too small and a global slot must use the maximum.
uint64_t qrt_weights_max_expert_bytes(const qrt_weights * w);
uint64_t qrt_weights_min_expert_bytes(const qrt_weights * w);

// Expert block bytes of one layer. Every expert of a layer has the same type mix in
// this model, and the cache sizes its slots per layer from this value (M3-2a).
// Returns -1 for an out-of-range layer or a missing record.
int qrt_weights_layer_expert_bytes(const qrt_weights * w, uint32_t layer, uint64_t * out);

// Returns -1 with a message when the pair is unknown or out of range.
int qrt_weights_locate(const qrt_weights * w, uint32_t layer, uint32_t expert,
                       qrt_expert_loc * out, char * err, size_t err_size);

// Reads one expert block (gate||up||down) into dst; cap must hold loc.bytes.
int qrt_weights_read_expert(const qrt_weights * w, uint32_t layer, uint32_t expert,
                            void * dst, size_t cap, char * err, size_t err_size);

// 1 when the payload is a read-only mapping (a memcpy per expert), 0 when the open
// had to fall back to seek plus fread. Reported by the prefill check.
int qrt_weights_mapped(const qrt_weights * w);
void qrt_weights_reader_name(const qrt_weights * w, char * out, size_t out_size);

// Dense residency (design 5.5, M2-3): every non-expert tensor of the GGUF lives in
// one device buffer, streamed up through a staging buffer. token_embd is not part
// of the design's 2079 MiB dense bucket and is left on the host for now.
struct qrt_dense;

// staging_bytes caps the transfer chunk; 8 MiB amortizes the 0.065 ms per copy.
int qrt_dense_open(qrt_vk_dev * dev, const char * gguf_path, uint64_t staging_bytes,
                   qrt_dense ** out, char * err, size_t err_size);
void qrt_dense_close(qrt_vk_dev * dev, qrt_dense * d);

uint64_t qrt_dense_bytes(const qrt_dense * d);
uint32_t qrt_dense_tensor_count(const qrt_dense * d);
const char * qrt_dense_tensor_name(const qrt_dense * d, uint32_t index);
int qrt_dense_locate(const qrt_dense * d, const char * name, uint64_t * offset, uint64_t * bytes);
// Returns the device buffer the dense tensors were streamed into.
VkBuffer qrt_dense_buffer(const qrt_dense * d);

// Compares up to max_bytes of the resident copy against the file, spread over the
// whole buffer including the last tensor. Returns the bytes actually compared.
uint64_t qrt_dense_verify(qrt_vk_dev * dev, qrt_dense * d, qrt_vk_staging * stg,
                          uint64_t max_bytes, char * err, size_t err_size);
