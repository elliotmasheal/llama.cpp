#pragma once

#include <cstdint>

// Draft Slice 1 interface (bench/THREE-TIER.md F12/F13).
// This header is not included by the runtime. The tests pin the budget and
// degradation invariants before any call site moves.
// The solver is a draft test oracle: common_moe_plan_get stays the production
// arithmetic owner until the follow-up reconciles the two.

enum common_residency_tier : int {
    COMMON_RESIDENCY_TIER_VRAM = 0,
    COMMON_RESIDENCY_TIER_RAM  = 1,
    COMMON_RESIDENCY_TIER_SSD  = 2,
};

struct common_residency_budget {
    int64_t vram = 0;
    int64_t ram  = 0;
    int64_t ssd  = 0;

    int64_t at(common_residency_tier tier) const;
    bool    valid() const;
};

struct common_residency_term {
    int64_t vram = 0;
    int64_t ram  = 0;
    int64_t ssd  = 0;

    int64_t at(common_residency_tier tier) const;
};

enum common_residency_status : int {
    COMMON_RESIDENCY_OK = 0,
    COMMON_RESIDENCY_BANK_REDUCED,
    COMMON_RESIDENCY_KV_DOES_NOT_FIT,
    COMMON_RESIDENCY_COMPUTE_DOES_NOT_FIT,
    COMMON_RESIDENCY_NEGATIVE,
    COMMON_RESIDENCY_OVER_VRAM,
    COMMON_RESIDENCY_OVER_RAM,
    COMMON_RESIDENCY_OVER_SSD,
};

struct common_residency_plan {
    common_residency_budget budget;

    common_residency_term dense;
    common_residency_term kv;
    common_residency_term bank;
    common_residency_term compute;
    common_residency_term score;
    common_residency_term gather;
    common_residency_term margin;

    // set by common_residency_solve(); check() only validates the sums
    bool degraded = false;
    common_residency_status status = COMMON_RESIDENCY_OK;

    common_residency_term allocated() const;
    int64_t remaining(common_residency_tier tier) const;
    common_residency_status check() const;
};

// Allocate in the documented order KV -> bank -> compute.
// dense / score / gather / margin are fixed claims; bank is reduced before
// compute is rejected, and no tier may borrow from another.
common_residency_plan common_residency_solve(const common_residency_plan & request);

const char * common_residency_status_name(common_residency_status status);

// Degradation decisions for the four abort sites and the store paths.
enum common_degradation_action : uint32_t {
    COMMON_DEGRADATION_KEEP             = 0,
    COMMON_DEGRADATION_DROP_ROLLING     = 1u << 0,
    COMMON_DEGRADATION_DISABLE_CACHE    = 1u << 1,
    COMMON_DEGRADATION_DISABLE_HYBRID   = 1u << 2,
    COMMON_DEGRADATION_DISABLE_PREFETCH = 1u << 3,
    COMMON_DEGRADATION_REJECT_SSD_SYNC  = 1u << 4,
    COMMON_DEGRADATION_FALLBACK_MMAP    = 1u << 5,
};

enum common_degradation_reason : int {
    COMMON_DEGRADATION_NONE = 0,
    COMMON_DEGRADATION_NO_SLRU_VICTIM,
    COMMON_DEGRADATION_CACHE_TOO_SMALL_ROLLING,
    COMMON_DEGRADATION_CACHE_TOO_SMALL,
    COMMON_DEGRADATION_HYBRID_WORKER_FAILED,
    COMMON_DEGRADATION_BUDGET_EXHAUSTED,
    COMMON_DEGRADATION_SSD_SYNC_REJECTED,
    COMMON_DEGRADATION_PREFETCH_DISABLED,
};

struct common_degradation_decision {
    uint32_t actions = COMMON_DEGRADATION_KEEP;
    common_degradation_reason reason = COMMON_DEGRADATION_NONE;
    bool fail_current_decode = false;

    bool has(common_degradation_action action) const;
    bool degraded() const;
};

struct common_degradation_policy {
    common_degradation_decision on_no_slru_victim() const;
    common_degradation_decision on_cache_too_small(bool rolling) const;
    common_degradation_decision on_hybrid_worker_failed() const;
    common_degradation_decision on_budget_exhausted() const;
    common_degradation_decision on_ssd_sync_rejected() const;
    common_degradation_decision on_prefetch_disabled() const;
};

struct common_degradation_stats {
    int64_t no_slru_victim = 0;
    int64_t cache_too_small_rolling = 0;
    int64_t cache_too_small = 0;
    int64_t hybrid_worker_failed = 0;
    int64_t budget_exhausted = 0;
    int64_t ssd_sync_rejected = 0;
    int64_t prefetch_disabled = 0;

    void record(const common_degradation_decision & decision);
};
