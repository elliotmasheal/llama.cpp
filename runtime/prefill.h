#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vk_dev.h"

// Prefill scheduler (design 4.5, M2-5): layer first, and while the compute of one
// expert block runs, the upload of the next blocks is already in flight. The math
// itself is a callback, so the same scheduler will drive the int8 GEMM once M2-5
// step 2 wires it in.

struct qrt_weights;
struct qrt_prefill;

// Called once per uploaded block, after its fence signalled. The slot memory must
// not be touched again until this returns. Returning -1 counts as a failed block.
typedef int (*qrt_prefill_compute_fn)(void * user, uint32_t layer, uint32_t expert,
                                      VkBuffer buf, uint64_t slot_offset, uint64_t bytes,
                                      char * err, size_t err_size);

struct qrt_prefill_cfg {
    uint32_t chunk;        // tokens per chunk (QRT_PREFILL_CHUNK), reported only
    uint32_t slots;        // staging buffers in flight; 1 is the serial baseline
    uint32_t slot_bytes;   // 0 = largest expert block in the store
    uint32_t slot_reductions; // out: slots given up because allocation failed
};

struct qrt_prefill_stats {
    uint32_t layers;
    uint32_t blocks;          // blocks the router asked for
    uint32_t blocks_done;
    uint32_t blocks_failed;
    uint32_t slot_reductions; // staging slots given up because allocation failed
    uint64_t bytes;
    double read_ms;           // host read of the repacked store into the mapping
    double upload_ms;         // CPU time inside the upload submits
    double wait_ms;           // CPU time waiting for a slot fence
    double compute_ms;        // CPU time inside the compute callback
    double total_ms;
};

int qrt_prefill_open(qrt_vk_dev * dev, qrt_weights * w, const qrt_prefill_cfg * cfg,
                     qrt_prefill ** out, char * err, size_t err_size);
void qrt_prefill_close(qrt_vk_dev * dev, qrt_prefill * p);

// ids holds `counts[layer]` expert ids per layer, row stride `stride`. The caller
// decides the set (a real router, or every expert for a chunk that hits them all).
int qrt_prefill_run(qrt_prefill * p, const uint32_t * ids, uint32_t stride,
                    const uint32_t * counts, qrt_prefill_compute_fn compute, void * user,
                    qrt_prefill_stats * out, char * err, size_t err_size);

// Layer batch (design 4.6): every expert of a layer is uploaded into one layer buffer,
// then the whole layer runs as a single submit. The callback records all of the layer's
// dispatches into one command buffer and submits once, so the GPU keeps 512 workgroups
// busy instead of 16 (RESULTS 39).
typedef int (*qrt_prefill_batch_fn)(void * user, uint32_t layer, VkBuffer buf,
                                    const uint32_t * experts, const uint64_t * offsets,
                                    const uint64_t * bytes, uint32_t count,
                                    char * err, size_t err_size);

int qrt_prefill_run_layers(qrt_prefill * p, const uint32_t * ids, uint32_t stride,
                           const uint32_t * counts, qrt_prefill_batch_fn batch, void * user,
                           qrt_prefill_stats * out, char * err, size_t err_size);

// Expected unique experts per layer for `chunk` tokens of top-k routing over
// `experts`, uniform: the prefill byte volume of RESULTS 11 and design 4.5.
uint32_t qrt_prefill_unique_experts(uint32_t chunk, uint32_t top_k, uint32_t experts);

VkBuffer qrt_prefill_buffer(const qrt_prefill * p);
uint64_t qrt_prefill_slot_bytes(const qrt_prefill * p);

// Stand-in compute leg for --prefill-check: a bandwidth read of the block that was
// just uploaded, one dispatch per block, on the compute queue. It has the geometry
// of the real GEMM dispatch but does no math, so it measures the scheduler only;
// M2-5 step 2 swaps the pipeline for the int8 kernel.
struct qrt_prefill_probe;
int qrt_prefill_probe_open(qrt_vk_dev * dev, const char * spv_dir, qrt_prefill * p,
                           qrt_prefill_probe ** out, char * err, size_t err_size);
void qrt_prefill_probe_close(qrt_vk_dev * dev, qrt_prefill_probe * probe);
int qrt_prefill_probe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                              uint64_t slot_offset, uint64_t bytes,
                              char * err, size_t err_size);
