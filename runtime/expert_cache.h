#pragma once

#include <stddef.h>
#include <stdint.h>

#include "qrt_config.h"

// Decode expert cache (design 5.1 `expert_cache`, M3-0). The module owns slot
// geometry, the per-layer SLRU/LRU/static policy and the H2D scheduling
// contract. It never allocates VRAM or reads the store itself: the caller passes
// a per-layer slot byte table and upload/wait callbacks, so the same policy is
// testable without a device and the pass builder can supply the real transfers.

enum {
    QRT_CACHE_MAX_LAYERS = 64,
    QRT_CACHE_MAX_EXPERTS = 256,
    QRT_CACHE_DEFAULT_PROTECTED_PCT = 20,
};

struct qrt_cache_plan {
    uint32_t layers;
    uint32_t experts;
    int32_t  policy;              // enum qrt_cache_policy
    uint32_t protected_percent;   // SLRU protected segment, 0..100
    uint64_t budget_bytes;
    uint64_t bytes;               // sum of slots_per_layer[l] * slot_bytes[l]
    uint64_t slots;               // total slots
    uint32_t degraded_layers;     // layers that received no slot
    uint64_t slot_bytes[QRT_CACHE_MAX_LAYERS];
    uint32_t slots_per_layer[QRT_CACHE_MAX_LAYERS];
    uint64_t layer_offset[QRT_CACHE_MAX_LAYERS];
};

// weights[l] is the layer's miss traffic (for example the cold-cache miss count
// of a routing trace). NULL means uniform. One slot is reserved per layer when
// the budget allows; the rest is distributed in proportion to weights / slot
// bytes. A budget below one slot per layer degrades to as many layers as fit,
// never to an error.
int qrt_cache_plan_build(uint32_t layers, uint32_t experts, const uint64_t * slot_bytes,
                         const uint64_t * weights, uint64_t budget_bytes, int32_t policy,
                         uint32_t protected_percent, qrt_cache_plan * out,
                         char * err, size_t err_size);

// Upload one expert into the slot at dst_offset; the callback owns the source
// and the transfer. wait must retire a submission before the slot is read.
struct qrt_cache_io {
    int (*upload)(void * user, uint32_t layer, uint32_t expert, uint64_t dst_offset,
                  uint64_t bytes, char * err, size_t err_size);
    int (*wait)(void * user, uint32_t layer, uint32_t expert, uint64_t dst_offset,
                uint64_t bytes, char * err, size_t err_size);
    void * user;
};

struct qrt_cache_lookup {
    uint32_t expert;
    int32_t  slot;      // -1: no slot, upload failed or miss handed to the CPU leg
    int      hit;       // resident before this acquire
    int      ready;     // slot content is valid now; 0 means wait first
    int      pending;   // upload in flight
    int      failed;    // no slot or upload failure; caller must use the CPU leg
};

struct qrt_cache_stats {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t promotions;
    uint64_t demotions;
    uint64_t miss_bytes;       // bytes of misses that received a slot
    uint64_t h2d_uploads;      // upload callbacks submitted
    uint64_t h2d_bytes;        // bytes submitted through upload
    uint64_t upload_failures;
    uint64_t wait_failures;
    uint64_t no_slot_misses;
    uint64_t cpu_leg_misses;   // misses the split handed to the CPU leg (M3-2a)
    uint64_t layer_hits[QRT_CACHE_MAX_LAYERS];
    uint64_t layer_misses[QRT_CACHE_MAX_LAYERS];
    uint64_t layer_evictions[QRT_CACHE_MAX_LAYERS];
    uint32_t pending;
};

struct qrt_cache;

int qrt_cache_open(const qrt_cache_plan * plan, const qrt_cache_io * io, qrt_cache ** out,
                   char * err, size_t err_size);
void qrt_cache_close(qrt_cache * c);

// One layer's route set. The caller passes unique expert ids; duplicates are
// folded and counted once. The result reports every requested expert. A miss
// gets a victim slot unless the layer has no slot left in this call; that expert
// is returned with slot=-1 and failed=1 for the CPU leg. Returns -1 only for bad
// arguments or when this layer still has an upload in flight.
int qrt_cache_acquire(qrt_cache * c, uint32_t layer, const uint32_t * experts, uint32_t count,
                      qrt_cache_lookup * lookups, char * err, size_t err_size);

// Split-aware acquire (M3-2a). h2d_miss_mask has one byte per expert and is read only
// for misses: 1 = insert and take a slot (and call io.upload when set), 0 = hand the
// expert to the CPU leg with slot=-1, failed=1, without evicting or uploading
// anything. Hits ignore the mask. NULL means every miss takes a slot.
int qrt_cache_acquire_mask(qrt_cache * c, uint32_t layer, const uint32_t * experts,
                           uint32_t count, const uint8_t * h2d_miss_mask,
                           qrt_cache_lookup * lookups, char * err, size_t err_size);

// Side-effect free residency probe: 1 when the expert occupies a slot.
int qrt_cache_resident(const qrt_cache * c, uint32_t layer, uint32_t expert);

// Counts an upload the caller performed itself (the decode path uploads straight
// from its store staging, so the cache is opened without io callbacks).
void qrt_cache_note_upload(qrt_cache * c, uint32_t layer, uint64_t bytes);

// Byte base of a layer's slots; a caller that keeps one buffer per layer subtracts
// this from qrt_cache_slot_offset.
uint64_t qrt_cache_layer_base(const qrt_cache * c, uint32_t layer);
int qrt_cache_wait_expert(qrt_cache * c, uint32_t layer, uint32_t expert,
                          char * err, size_t err_size);
int qrt_cache_wait_all(qrt_cache * c, char * err, size_t err_size);
void qrt_cache_get_stats(const qrt_cache * c, qrt_cache_stats * out);
void qrt_cache_reset_stats(qrt_cache * c);
uint64_t qrt_cache_bytes(const qrt_cache * c);
uint32_t qrt_cache_layers(const qrt_cache * c);
uint32_t qrt_cache_slots(const qrt_cache * c, uint32_t layer);
uint64_t qrt_cache_slot_offset(const qrt_cache * c, uint32_t layer, uint32_t slot);

// Routing trace (M1-1 format): int32 header [n_tok, n_layer, top_k], then for
// each layer int32 n_ints followed by int32 ids. Layer records may be partial;
// the decode tail is the last rows of each record.
struct qrt_cache_trace;

int qrt_cache_trace_open(const char * path, uint32_t experts, qrt_cache_trace ** out,
                         char * err, size_t err_size);
void qrt_cache_trace_close(qrt_cache_trace * t);
uint32_t qrt_cache_trace_tokens(const qrt_cache_trace * t);
uint32_t qrt_cache_trace_layers(const qrt_cache_trace * t);
uint32_t qrt_cache_trace_top_k(const qrt_cache_trace * t);
uint32_t qrt_cache_trace_rows(const qrt_cache_trace * t, uint32_t layer);
const uint32_t * qrt_cache_trace_route(const qrt_cache_trace * t, uint32_t layer, uint32_t row);

// Per-layer miss traffic of the last decode_tokens rows. Layers with fewer rows
// than decode_tokens are skipped and reported through observed_layers.
int qrt_cache_trace_weights(const qrt_cache_trace * t, uint32_t decode_tokens, uint32_t layers,
                            uint64_t * weights, uint32_t * observed_layers,
                            char * err, size_t err_size);

// Replays the trace through a policy-only cache (no IO callbacks): warm on the
// rows before the decode tail, reset counters, then replay the tail.
int qrt_cache_simulate(const qrt_cache_plan * plan, const qrt_cache_trace * t,
                       uint32_t decode_tokens, qrt_cache_stats * out,
                       uint32_t * observed_layers, uint32_t * decode_rows,
                       char * err, size_t err_size);
