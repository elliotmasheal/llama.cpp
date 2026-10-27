#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

// store-io (bench/THREE-TIER.md F13): the executor of transfers and the only place that touches
// backend or file handles. This unit owns the backend prefetch proc addresses once a backend is
// bound; the MoE whole-layer prefetch routes its per-layer submissions through here (2c), while
// the policy (rings, one-layer lead) stays with the MoE cache. The pool remains uncreated until a
// caller needs asynchronous issue (2g); pool threads never block on the work they issue.
typedef bool     (*llama_store_pf_tensor_fn)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
typedef void     (*llama_store_pf_submit_fn)(ggml_backend_t);
typedef void     (*llama_store_pf_wait_fn)(ggml_backend_t);
typedef uint64_t (*llama_store_pf_submit_seq_fn)(ggml_backend_t);
typedef void     (*llama_store_pf_wait_seq_fn)(ggml_backend_t, uint64_t);

struct llama_store_pf_api {
    llama_store_pf_tensor_fn     tensor     = nullptr;
    llama_store_pf_submit_fn     submit     = nullptr;
    llama_store_pf_wait_fn       wait       = nullptr;
    llama_store_pf_submit_seq_fn submit_seq = nullptr;
    llama_store_pf_wait_seq_fn   wait_seq   = nullptr;

    ggml_backend_t bound = nullptr;

    bool ok() const {
        return tensor != nullptr && submit != nullptr && wait != nullptr;
    }
    bool seq_ok() const {
        return submit_seq != nullptr && wait_seq != nullptr;
    }
};

// --store-exec: 0 = callers keep their own submission path, 1 = route through store-io with the same
// primitives, 2 = also wait only on the batch's own sequence number (later batches stay in flight)
enum llama_store_exec_mode {
    LLAMA_STORE_EXEC_OFF    = 0,
    LLAMA_STORE_EXEC_ROUTE  = 1,
    LLAMA_STORE_EXEC_SEQ    = 2,
};

struct llama_store_io {
    // thread pool of the M2 swap: --store-threads, 0 = auto min(2, max(1, hw/4));
    // pool threads only issue transfers and never block on them
    int32_t threads = 0;

    llama_store_pf_api pf;
    int32_t            exec_mode = 0; // LLAMA_STORE_EXEC_*

    // F12 10 accounting; the split log prints these at commit, M2's consumer reads them live
    int64_t store_sync_waits          = 0;
    int64_t store_sync_wait_ms        = 0;
    int64_t store_prefetch_issued     = 0;
    int64_t store_prefetch_lead_steps = 0;
    int64_t store_prefetch_wasted     = 0;
    int64_t store_hits_by_tier[3]     = {0, 0, 0};
    int64_t store_evict_deferred      = 0;
    int64_t store_inflight_pins       = 0;
    int64_t store_pool_cancel         = 0;
    int64_t store_ssd_sync_rejected   = 0;
    int64_t store_exec_submissions    = 0;
    int64_t store_exec_fallbacks      = 0;
    uint64_t exec_drain_seq           = 0; // drain-mode surrogate sequence numbers
    bool exec_fallback_logged         = false;

    // counter sum for the wiring sanity print; zero transfers are real data until the swaps land
    int64_t total() const {
        return store_sync_waits + store_prefetch_issued + store_pool_cancel +
               store_ssd_sync_rejected + store_evict_deferred + store_exec_submissions +
               store_exec_fallbacks;
    }
};

// bind one backend's transfer entry points into this io unit (2c). Resolution is exactly what the
// MoE cache did before (moe-cache.cpp init_prefetch_api): proc addresses through the device's
// backend reg, so both paths share one set of Vulkan entries and there is no second registration.
bool llama_store_io_bind_backend(struct llama_store_pf_api & pf, ggml_backend_t backend);
