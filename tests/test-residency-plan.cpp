// Draft Slice 1 invariants for common/residency.h. No GPU, no model, no runtime wiring.
// The four current abort sites in src/llama-moe-cache.cpp are:
//   :129  no SLRU victim
//   :1843 hybrid worker failed
//   :2184 rolling plan too small
//   :2216 non-rolling plan too small
#include "../common/residency.h"

#include <cstdio>

static int failures = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        std::printf("FAIL %s:%d: %s\n", __FILE__, line, what);
        failures++;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

static common_residency_plan make_plan(common_residency_budget budget) {
    common_residency_plan plan;
    plan.budget = budget;
    return plan;
}

int main() {
    using tier = common_residency_tier;

    // KV -> bank -> compute: bank is reduced before compute is rejected.
    {
        common_residency_plan req = make_plan({100, 200, 10});
        req.dense.vram   = 10;
        req.score.vram   = 5;
        req.gather.vram  = 5;
        req.margin.vram  = 10;
        req.kv.vram      = 30;
        req.bank.vram    = 60;
        req.compute.vram = 20;

        const common_residency_plan plan = common_residency_solve(req);
        CHECK(plan.status == COMMON_RESIDENCY_BANK_REDUCED);
        CHECK(plan.degraded);
        CHECK(plan.bank.vram == 20);
        CHECK(plan.compute.vram == 20);
        CHECK(plan.check() == COMMON_RESIDENCY_OK);
        CHECK(plan.remaining(tier::COMMON_RESIDENCY_TIER_VRAM) == 0);
        CHECK(plan.allocated().vram == 100);
    }

    // Budgets are non-fungible: free RAM cannot pay for a VRAM overrun.
    {
        common_residency_plan over = make_plan({50, 1000, 0});
        over.dense.vram = 60;
        CHECK(over.check() == COMMON_RESIDENCY_OVER_VRAM);

        const common_residency_plan plan = common_residency_solve(over);
        CHECK(plan.status == COMMON_RESIDENCY_OVER_VRAM);
    }

    // KV is allocated first; if KV alone does not fit, the plan is not valid.
    {
        common_residency_plan req = make_plan({100, 0, 0});
        req.dense.vram = 10;
        req.margin.vram = 10;
        req.kv.vram = 90;
        const common_residency_plan plan = common_residency_solve(req);
        CHECK(plan.status == COMMON_RESIDENCY_KV_DOES_NOT_FIT);
    }

    // Compute is required; bank is sacrificed first, then compute is rejected.
    {
        common_residency_plan req = make_plan({100, 0, 0});
        req.dense.vram = 10;
        req.margin.vram = 10;
        req.kv.vram = 50;
        req.bank.vram = 30;
        req.compute.vram = 40;
        const common_residency_plan plan = common_residency_solve(req);
        CHECK(plan.status == COMMON_RESIDENCY_COMPUTE_DOES_NOT_FIT);
        CHECK(plan.bank.vram == 0);
    }

    // No cross-tier borrowing for RAM either.
    {
        common_residency_plan req = make_plan({100, 50, 0});
        req.kv.ram = 60;
        const common_residency_plan plan = common_residency_solve(req);
        CHECK(plan.status == COMMON_RESIDENCY_KV_DOES_NOT_FIT);
    }

    // Negative budgets and terms are structural errors, not "over budget".
    {
        common_residency_plan negative = make_plan({-1, 0, 0});
        CHECK(negative.check() == COMMON_RESIDENCY_NEGATIVE);
        CHECK(common_residency_solve(negative).status == COMMON_RESIDENCY_NEGATIVE);

        common_residency_plan negative_term = make_plan({10, 0, 0});
        negative_term.kv.vram = -1;
        CHECK(negative_term.check() == COMMON_RESIDENCY_NEGATIVE);
    }

    // Exact fit is valid and not degraded.
    {
        common_residency_plan req = make_plan({100, 200, 10});
        req.dense.vram = 10;
        req.margin.vram = 10;
        req.kv.vram = 30;
        req.bank.vram = 20;
        req.compute.vram = 30;
        const common_residency_plan plan = common_residency_solve(req);
        CHECK(plan.status == COMMON_RESIDENCY_OK);
        CHECK(!plan.degraded);
        CHECK(plan.bank.vram == 20);
        CHECK(plan.check() == COMMON_RESIDENCY_OK);
        CHECK(plan.remaining(tier::COMMON_RESIDENCY_TIER_VRAM) == 0);
    }

    // Four abort sites -> decisions. None of these may abort.
    {
        const common_degradation_policy policy;

        const common_degradation_decision no_victim = policy.on_no_slru_victim();
        CHECK(no_victim.has(COMMON_DEGRADATION_DROP_ROLLING));
        CHECK(no_victim.reason == COMMON_DEGRADATION_NO_SLRU_VICTIM);
        CHECK(!no_victim.fail_current_decode);

        const common_degradation_decision too_small_rolling = policy.on_cache_too_small(true);
        CHECK(too_small_rolling.has(COMMON_DEGRADATION_DROP_ROLLING));
        CHECK(too_small_rolling.reason == COMMON_DEGRADATION_CACHE_TOO_SMALL_ROLLING);
        CHECK(!too_small_rolling.fail_current_decode);

        const common_degradation_decision too_small = policy.on_cache_too_small(false);
        CHECK(too_small.has(COMMON_DEGRADATION_DISABLE_CACHE));
        CHECK(too_small.has(COMMON_DEGRADATION_DISABLE_HYBRID));
        CHECK(too_small.reason == COMMON_DEGRADATION_CACHE_TOO_SMALL);
        CHECK(!too_small.fail_current_decode);

        const common_degradation_decision hybrid_failed = policy.on_hybrid_worker_failed();
        CHECK(hybrid_failed.has(COMMON_DEGRADATION_DISABLE_HYBRID));
        CHECK(hybrid_failed.reason == COMMON_DEGRADATION_HYBRID_WORKER_FAILED);
        CHECK(hybrid_failed.fail_current_decode);

        const common_degradation_decision budget = policy.on_budget_exhausted();
        CHECK(budget.has(COMMON_DEGRADATION_DROP_ROLLING));
        CHECK(budget.has(COMMON_DEGRADATION_DISABLE_PREFETCH));
        CHECK(budget.has(COMMON_DEGRADATION_FALLBACK_MMAP));
        CHECK(budget.reason == COMMON_DEGRADATION_BUDGET_EXHAUSTED);
        CHECK(!budget.fail_current_decode);

        const common_degradation_decision ssd = policy.on_ssd_sync_rejected();
        CHECK(ssd.has(COMMON_DEGRADATION_REJECT_SSD_SYNC));
        CHECK(ssd.reason == COMMON_DEGRADATION_SSD_SYNC_REJECTED);

        const common_degradation_decision prefetch = policy.on_prefetch_disabled();
        CHECK(prefetch.has(COMMON_DEGRADATION_DISABLE_PREFETCH));
        CHECK(prefetch.reason == COMMON_DEGRADATION_PREFETCH_DISABLED);

        common_degradation_stats stats;
        const common_degradation_decision * decisions[] = {
            &no_victim, &too_small_rolling, &too_small, &hybrid_failed,
            &budget, &ssd, &prefetch,
        };
        for (const common_degradation_decision * decision : decisions) {
            stats.record(*decision);
        }
        CHECK(stats.no_slru_victim == 1);
        CHECK(stats.cache_too_small_rolling == 1);
        CHECK(stats.cache_too_small == 1);
        CHECK(stats.hybrid_worker_failed == 1);
        CHECK(stats.budget_exhausted == 1);
        CHECK(stats.ssd_sync_rejected == 1);
        CHECK(stats.prefetch_disabled == 1);
    }

    if (failures != 0) {
        std::printf("test-residency-plan: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-residency-plan: OK\n");
    return 0;
}
