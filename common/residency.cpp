#include "residency.h"

#include <algorithm>

int64_t common_residency_budget::at(common_residency_tier tier) const {
    switch (tier) {
        case COMMON_RESIDENCY_TIER_VRAM: return vram;
        case COMMON_RESIDENCY_TIER_RAM:  return ram;
        case COMMON_RESIDENCY_TIER_SSD:  return ssd;
    }
    return 0;
}

bool common_residency_budget::valid() const {
    return vram >= 0 && ram >= 0 && ssd >= 0;
}

int64_t common_residency_term::at(common_residency_tier tier) const {
    switch (tier) {
        case COMMON_RESIDENCY_TIER_VRAM: return vram;
        case COMMON_RESIDENCY_TIER_RAM:  return ram;
        case COMMON_RESIDENCY_TIER_SSD:  return ssd;
    }
    return 0;
}

static int64_t & term_at(common_residency_term & term, common_residency_tier tier) {
    switch (tier) {
        case COMMON_RESIDENCY_TIER_VRAM: return term.vram;
        case COMMON_RESIDENCY_TIER_RAM:  return term.ram;
        case COMMON_RESIDENCY_TIER_SSD:  return term.ssd;
    }
    return term.vram;
}

static bool term_valid(const common_residency_term & term) {
    return term.vram >= 0 && term.ram >= 0 && term.ssd >= 0;
}

static common_residency_status over_status(common_residency_tier tier) {
    switch (tier) {
        case COMMON_RESIDENCY_TIER_VRAM: return COMMON_RESIDENCY_OVER_VRAM;
        case COMMON_RESIDENCY_TIER_RAM:  return COMMON_RESIDENCY_OVER_RAM;
        case COMMON_RESIDENCY_TIER_SSD:  return COMMON_RESIDENCY_OVER_SSD;
    }
    return COMMON_RESIDENCY_OVER_VRAM;
}

common_residency_term common_residency_plan::allocated() const {
    common_residency_term out;
    for (common_residency_tier tier : { COMMON_RESIDENCY_TIER_VRAM, COMMON_RESIDENCY_TIER_RAM, COMMON_RESIDENCY_TIER_SSD }) {
        int64_t v = 0;
        for (const common_residency_term * term : { &dense, &kv, &bank, &compute, &score, &gather, &margin }) {
            v += term->at(tier);
        }
        term_at(out, tier) = v;
    }
    return out;
}

int64_t common_residency_plan::remaining(common_residency_tier tier) const {
    return budget.at(tier) - allocated().at(tier);
}

common_residency_status common_residency_plan::check() const {
    if (!budget.valid()) {
        return COMMON_RESIDENCY_NEGATIVE;
    }
    for (const common_residency_term * term : { &dense, &kv, &bank, &compute, &score, &gather, &margin }) {
        if (!term_valid(*term)) {
            return COMMON_RESIDENCY_NEGATIVE;
        }
    }
    const common_residency_term sum = allocated();
    for (common_residency_tier tier : { COMMON_RESIDENCY_TIER_VRAM, COMMON_RESIDENCY_TIER_RAM, COMMON_RESIDENCY_TIER_SSD }) {
        if (sum.at(tier) > budget.at(tier)) {
            return over_status(tier);
        }
    }
    return COMMON_RESIDENCY_OK;
}

common_residency_plan common_residency_solve(const common_residency_plan & request) {
    common_residency_plan out = request;
    out.degraded = false;
    out.status = COMMON_RESIDENCY_OK;

    if (!out.budget.valid()) {
        out.status = COMMON_RESIDENCY_NEGATIVE;
        return out;
    }
    for (const common_residency_term * term : { &out.dense, &out.kv, &out.bank, &out.compute, &out.score, &out.gather, &out.margin }) {
        if (!term_valid(*term)) {
            out.status = COMMON_RESIDENCY_NEGATIVE;
            return out;
        }
    }

    for (common_residency_tier tier : { COMMON_RESIDENCY_TIER_VRAM, COMMON_RESIDENCY_TIER_RAM, COMMON_RESIDENCY_TIER_SSD }) {
        const int64_t budget = out.budget.at(tier);
        int64_t fixed = 0;
        for (const common_residency_term * term : { &out.dense, &out.score, &out.gather, &out.margin }) {
            fixed += term->at(tier);
        }
        if (fixed > budget) {
            out.status = over_status(tier);
            return out;
        }

        const int64_t kv = out.kv.at(tier);
        if (fixed + kv > budget) {
            out.status = COMMON_RESIDENCY_KV_DOES_NOT_FIT;
            return out;
        }

        const int64_t available = budget - fixed - kv;
        const int64_t bank_requested = out.bank.at(tier);
        const int64_t compute = out.compute.at(tier);

        int64_t bank = std::min(bank_requested, available);
        if (bank != bank_requested) {
            out.degraded = true;
        }
        if (bank + compute > available) {
            bank = std::max<int64_t>(0, available - compute);
            out.degraded = true;
        }
        if (bank + compute > available) {
            term_at(out.bank, tier) = 0;
            out.degraded = true;
            out.status = COMMON_RESIDENCY_COMPUTE_DOES_NOT_FIT;
            return out;
        }

        term_at(out.bank, tier) = bank;
        term_at(out.compute, tier) = compute;
    }

    if (out.degraded && out.status == COMMON_RESIDENCY_OK) {
        out.status = COMMON_RESIDENCY_BANK_REDUCED;
    }
    return out;
}

const char * common_residency_status_name(common_residency_status status) {
    switch (status) {
        case COMMON_RESIDENCY_OK:                     return "ok";
        case COMMON_RESIDENCY_BANK_REDUCED:           return "bank-reduced";
        case COMMON_RESIDENCY_KV_DOES_NOT_FIT:        return "kv-does-not-fit";
        case COMMON_RESIDENCY_COMPUTE_DOES_NOT_FIT:   return "compute-does-not-fit";
        case COMMON_RESIDENCY_NEGATIVE:               return "negative";
        case COMMON_RESIDENCY_OVER_VRAM:              return "over-vram";
        case COMMON_RESIDENCY_OVER_RAM:               return "over-ram";
        case COMMON_RESIDENCY_OVER_SSD:               return "over-ssd";
    }
    return "unknown";
}

bool common_degradation_decision::has(common_degradation_action action) const {
    return (actions & static_cast<uint32_t>(action)) != 0;
}

bool common_degradation_decision::degraded() const {
    return actions != COMMON_DEGRADATION_KEEP;
}

common_degradation_decision common_degradation_policy::on_no_slru_victim() const {
    return { COMMON_DEGRADATION_DROP_ROLLING, COMMON_DEGRADATION_NO_SLRU_VICTIM, false };
}

common_degradation_decision common_degradation_policy::on_cache_too_small(bool rolling) const {
    if (rolling) {
        return { COMMON_DEGRADATION_DROP_ROLLING, COMMON_DEGRADATION_CACHE_TOO_SMALL_ROLLING, false };
    }
    return {
        static_cast<uint32_t>(COMMON_DEGRADATION_DISABLE_CACHE | COMMON_DEGRADATION_DISABLE_HYBRID),
        COMMON_DEGRADATION_CACHE_TOO_SMALL,
        false
    };
}

common_degradation_decision common_degradation_policy::on_hybrid_worker_failed() const {
    return { COMMON_DEGRADATION_DISABLE_HYBRID, COMMON_DEGRADATION_HYBRID_WORKER_FAILED, true };
}

common_degradation_decision common_degradation_policy::on_budget_exhausted() const {
    return {
        static_cast<uint32_t>(
            COMMON_DEGRADATION_DROP_ROLLING |
            COMMON_DEGRADATION_DISABLE_PREFETCH |
            COMMON_DEGRADATION_FALLBACK_MMAP),
        COMMON_DEGRADATION_BUDGET_EXHAUSTED,
        false
    };
}

common_degradation_decision common_degradation_policy::on_ssd_sync_rejected() const {
    return { COMMON_DEGRADATION_REJECT_SSD_SYNC, COMMON_DEGRADATION_SSD_SYNC_REJECTED, false };
}

common_degradation_decision common_degradation_policy::on_prefetch_disabled() const {
    return { COMMON_DEGRADATION_DISABLE_PREFETCH, COMMON_DEGRADATION_PREFETCH_DISABLED, false };
}

void common_degradation_stats::record(const common_degradation_decision & decision) {
    switch (decision.reason) {
        case COMMON_DEGRADATION_NONE:                       break;
        case COMMON_DEGRADATION_NO_SLRU_VICTIM:             no_slru_victim++; break;
        case COMMON_DEGRADATION_CACHE_TOO_SMALL_ROLLING:    cache_too_small_rolling++; break;
        case COMMON_DEGRADATION_CACHE_TOO_SMALL:            cache_too_small++; break;
        case COMMON_DEGRADATION_HYBRID_WORKER_FAILED:       hybrid_worker_failed++; break;
        case COMMON_DEGRADATION_BUDGET_EXHAUSTED:           budget_exhausted++; break;
        case COMMON_DEGRADATION_SSD_SYNC_REJECTED:          ssd_sync_rejected++; break;
        case COMMON_DEGRADATION_PREFETCH_DISABLED:          prefetch_disabled++; break;
    }
}
