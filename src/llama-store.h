#pragma once

#include "llama.h"

#include <stdint.h>

// llama-store (bench/THREE-TIER.md F12/F13): the always-on abstraction of this machine's VRAM/RAM/SSD.
// It owns exactly two object families - expert weights (layer + expert sub-objects) and KV pages -
// and does four things: place, reside, copy, account. Everything else (dense weights, compute
// buffers, staging) keeps its own allocator but must report usage here, unreported use is a
// violation. Entries are tier sums in 2a; the per-layer placement and the transfer executor follow
// with 2c/3a, nothing in this file is a transport contract yet.
// No master switch: reproduce the old behavior through the three budget knobs, not by bypassing.

enum llama_store_tier {
    LLAMA_STORE_TIER_VRAM = 0,
    LLAMA_STORE_TIER_RAM  = 1,
    LLAMA_STORE_TIER_SSD  = 2,
};

enum llama_store_decl_kind {
    LLAMA_STORE_DECL_DENSE   = 0, // dense weights, not owned by the store
    LLAMA_STORE_DECL_COMPUTE = 1, // scheduler compute buffers, not owned
    LLAMA_STORE_DECL_STAGING = 2, // pinned upload staging, owned from 2c onwards
};

struct llama_store_params {
    int64_t vram_budget = 0; // bytes of the run-wide VRAM budget, 0 = auto (device free memory)
    int64_t ram_budget  = 0; // bytes of the run-wide RAM budget, 0 = auto (free RAM - system reserve)
    int64_t ssd_budget  = 0; // bytes of the cold-expert SSD capacity, 0 = off
    int32_t threads     = 0; // io pool size of 2c, 0 = auto: min(2, max(1, hw/4))
    int32_t exec_mode   = 0; // --store-exec transfer routing of the store (0/1/2, disabled by default)
};

// forward declaration; the shell exposes the thin entry points only (F13: five entries + adopt)
struct llama_store;

// the store active in this process (2c: src-side call sites like the MoE cache read it); skeleton
// phase single run per process, freed again in free
LLAMA_API struct llama_store * llama_store_active(void);

LLAMA_API struct llama_store * llama_store_init(const struct llama_store_params * params);
LLAMA_API void llama_store_free(struct llama_store * s);

// declared usage of the allocators the store does not own (F12 7). Passing tiers lets the shell
// account each allocator on the tier where it actually lives. Expansion declares add up.
LLAMA_API void llama_store_declare(struct llama_store * s,
                                   enum llama_store_decl_kind kind,
                                   enum llama_store_tier tier,
                                   int64_t bytes);

// adopt a host expert: two-level identity (layer, expert), O(1) array indexing after commit (F12 12)
LLAMA_API void llama_store_adopt_expert(struct llama_store * s, int32_t il, int32_t ie, int64_t bytes);

// freeze the declare/adopt surface and print the store_split lines of all three budgets (F12 7).
// Adoptions and declares after commit are legal (expansion), they just do not re-print the split.
LLAMA_API void llama_store_commit(struct llama_store * s);

// 2c transfer executor: bind one backend's transfer entry points into store-io (the proc address
// resolution lives in this unit, the MoE cache only calls the store from then on). The routing mode
// comes from --store-exec via the params; when the bind fails, the caller keeps its old path with a
// counted degradation and no abort.
LLAMA_API bool llama_store_bind_backend(struct llama_store * s, ggml_backend_t backend);
LLAMA_API int32_t llama_store_exec_mode(const struct llama_store * s);

// the three progressions of the MoE prefetch path (F12 5 semantics, different sequences for modes 1/2)
LLAMA_API bool llama_store_pf_tensor(struct llama_store * s, ggml_backend_t backend, struct ggml_tensor * tensor,
                                     const void * data, size_t offset, size_t size);
LLAMA_API void llama_store_pf_submit(struct llama_store * s, ggml_backend_t backend, uint64_t * seq);
// seq = 0: wait for the most recent batch (draining semantics); seq != 0: wait only for that sequence number
LLAMA_API void llama_store_pf_wait(struct llama_store * s, ggml_backend_t backend, uint64_t seq);
