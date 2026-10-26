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

// Compute must release the slot before returning, including on failure.
typedef int (*qrt_prefill_compute_fn)(void * user, uint32_t layer, uint32_t expert,
                                      VkBuffer buf, uint64_t slot_offset, uint64_t bytes,
                                      char * err, size_t err_size);

// Consume the arrays before returning; only device buffers may be borrowed until wait.
typedef int (*qrt_prefill_batch_fn)(void * user, uint32_t layer, uint32_t ring, VkBuffer buf,
                                    const uint32_t * experts, const uint64_t * offsets,
                                    const uint64_t * bytes, uint32_t count,
                                    char * err, size_t err_size);

// Retire all ring work, even after a partial batch failure; an empty ring is a no-op.
typedef int (*qrt_prefill_wait_fn)(void * user, uint32_t ring, char * err, size_t err_size);

struct qrt_prefill_backend {
    qrt_prefill_compute_fn compute;
    qrt_prefill_batch_fn batch;
    qrt_prefill_wait_fn wait;
    void * user;
};

struct qrt_prefill_cfg {
    uint32_t chunk;        // tokens per chunk (QRT_PREFILL_CHUNK), reported only
    uint32_t slots;        // staging buffers in flight; 1 is the serial baseline
    uint32_t slot_bytes;   // 0 = largest expert block in the store
    uint32_t group_bytes;  // coalesce consecutive experts into one copy of about this size
    uint32_t readers;      // host threads that copy the store into the staging mapping
    uint32_t pipeline;     // 1 = two layer buffers, upload L+1 during compute of L
    uint32_t slot_reductions; // out: slots given up because allocation failed
};

struct qrt_prefill_stats {
    uint32_t pipeline;        // 1 when the two layer buffers were used
    uint32_t layers;
    uint32_t blocks;          // blocks the router asked for
    uint32_t groups;          // copies actually submitted (coalesced blocks)
    uint32_t blocks_done;     // compute completed, including the batch wait
    uint32_t blocks_failed;   // observed failures; unstarted/cancelled blocks are not counted
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
                    const uint32_t * counts, const qrt_prefill_backend * backend,
                    qrt_prefill_stats * out, char * err, size_t err_size);

// With pipeline = 1 the layer buffers alternate, so layer L+1 is read and uploaded while
// layer L is still computing. pipeline = 0 is the serial shape, kept for comparison.
int qrt_prefill_run_layers(qrt_prefill * p, const uint32_t * ids, uint32_t stride,
                           const uint32_t * counts, const qrt_prefill_backend * backend,
                           qrt_prefill_stats * out, char * err, size_t err_size);

// How many layer buffers the scheduler keeps (1, or 2 with pipeline on).
uint32_t qrt_prefill_layer_buffers(const qrt_prefill * p);

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
