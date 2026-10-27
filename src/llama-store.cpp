#include "llama-store.h"

#include "llama-impl.h"
#include "llama-store-io.h"
#include "llama-store-policy.h"
#include "llama-store-slab.h"

#include <cassert>
#include <cinttypes>
#include <memory>
#include <thread>
#include <vector>

// the shell does not include backends: the transfer executor arrives with the 2c swap and lives
// inside store-io; slab and policy are the other two units (F13), wired but not driven yet

// process-active store (2c): src-side call sites (MoE cache now, KV pages later) read this pointer.
// Skeleton phase one run per process; a second init replaces the pointer, free drops only its own.
static llama_store * g_llama_store_active = nullptr;

struct llama_store {
    llama_store_params params;

    llama_store_slab       slab;
    std::unique_ptr<llama_store_policy> policy; // hot-swappable, slab asks its adopt_tier
    llama_store_io         io;

    // two-level registry (F12 12): layer -> expert handles, indexable in O(1) after the first pass
    std::vector<int32_t>                                expert_cls;    // per layer: slab class of its expert bytes
    std::vector<std::vector<llama_store_slab::handle> > expert_handle; // per layer * expert
    int64_t                                             n_expert_adopt = 0;

    // declared usage by [kind][tier]; the split print and the later unreported-buf check read this
    int64_t declared[3][3] = {};

    bool committed = false;
};

static const char * llama_store_tier_name(int tier) {
    return tier == LLAMA_STORE_TIER_VRAM ? "vram" : tier == LLAMA_STORE_TIER_RAM ? "ram" : "ssd";
}

struct llama_store * llama_store_init(const struct llama_store_params * params) {
    if (params == nullptr) {
        LLAMA_LOG_ERROR("%s: params is null\n", __func__);
        return nullptr;
    }
    llama_store * s = new llama_store();
    s->params = *params;
    // budgets are hard tops (F12 7); negatives cannot mean anything, floor those with one warning
    if (s->params.vram_budget < 0 || s->params.ram_budget < 0 || s->params.ssd_budget < 0) {
        LLAMA_LOG_WARN("%s: negative budget ignored\n", __func__);
        if (s->params.vram_budget < 0) s->params.vram_budget = 0;
        if (s->params.ram_budget  < 0) s->params.ram_budget  = 0;
        if (s->params.ssd_budget  < 0) s->params.ssd_budget  = 0;
    }
    s->policy = std::make_unique<llama_store_policy_default>();

    // the io pool of 2c is created then; the skeleton keeps only the thread count it will use
    if (s->params.threads == 0) {
        s->io.threads = (int32_t) std::min<int32_t>(2, std::max<int32_t>(1, (int32_t) std::thread::hardware_concurrency()/4));
    } else {
        s->io.threads = s->params.threads;
    }
    s->io.exec_mode = s->params.exec_mode > LLAMA_STORE_EXEC_SEQ ? LLAMA_STORE_EXEC_SEQ : s->params.exec_mode;
    g_llama_store_active = s;
    LLAMA_LOG_INFO("%s: init: io threads %d, exec mode %d\n", __func__, s->io.threads, s->io.exec_mode);
    return s;
}

void llama_store_free(struct llama_store * s) {
    if (s == nullptr) {
        return;
    }
    if (g_llama_store_active == s) {
        g_llama_store_active = nullptr;
    }
    int64_t declared_sum = 0;
    for (int k = 0; k < 3; k++) {
        for (int t = 0; t < 3; t++) {
            declared_sum += s->declared[k][t];
        }
    }
    LLAMA_LOG_DEBUG("%s: adopted %" PRId64 " experts, declared %.1f MiB\n",
            __func__, s->n_expert_adopt, declared_sum/1048576.0);
    delete s;
}

void llama_store_declare(struct llama_store * s,
                         enum llama_store_decl_kind kind,
                         enum llama_store_tier tier,
                         int64_t bytes) {
    if (s == nullptr || bytes == 0) {
        return;
    }
    assert(kind >= 0 && kind <= 2 && tier >= 0 && tier <= 2);
    s->declared[kind][tier] += bytes;
}

struct llama_store * llama_store_active(void) {
    return g_llama_store_active;
}

bool llama_store_bind_backend(struct llama_store * s, ggml_backend_t backend) {
    if (s == nullptr || s->io.exec_mode == LLAMA_STORE_EXEC_OFF) {
        return false;
    }
    if (!llama_store_io_bind_backend(s->io.pf, backend)) {
        LLAMA_LOG_WARN("%s: the backend does not expose the transfer entry points, keeping the caller's own path\n", __func__);
        s->io.exec_fallback_logged = true;
        return false;
    }
    LLAMA_LOG_INFO("%s: transfer executor bound (mode %d, %s)\n",
            __func__, s->io.exec_mode, s->io.pf.seq_ok() ? "seq capable" : "drain only");
    return true;
}

int32_t llama_store_exec_mode(const struct llama_store * s) {
    return s == nullptr ? 0 : s->io.exec_mode;
}

bool llama_store_pf_tensor(struct llama_store * s, ggml_backend_t backend, struct ggml_tensor * tensor,
                           const void * data, size_t offset, size_t size) {
    if (s == nullptr || s->io.exec_mode == LLAMA_STORE_EXEC_OFF) {
        return false;
    }
    if (s->io.pf.tensor == nullptr) {
        s->io.store_exec_fallbacks++;
        return false;
    }
    s->io.store_exec_submissions++;
    return s->io.pf.tensor(backend, tensor, data, offset, size);
}

void llama_store_pf_submit(struct llama_store * s, ggml_backend_t backend, uint64_t * seq) {
    if (s == nullptr || s->io.exec_mode == LLAMA_STORE_EXEC_OFF) {
        return;
    }
    s->io.store_exec_submissions++;
    if (s->io.exec_mode >= LLAMA_STORE_EXEC_SEQ && s->io.pf.seq_ok()) {
        *seq = s->io.pf.submit_seq(backend);
        return;
    }
    if (seq != nullptr) {
        *seq = ++s->io.exec_drain_seq; // mode 1 ignores the number, wait is a drain
    }
    s->io.pf.submit(backend);
}

void llama_store_pf_wait(struct llama_store * s, ggml_backend_t backend, uint64_t seq) {
    if (s == nullptr || s->io.exec_mode == LLAMA_STORE_EXEC_OFF) {
        return;
    }
    if (s->io.exec_mode >= LLAMA_STORE_EXEC_SEQ && s->io.pf.seq_ok() && seq != 0) {
        s->io.pf.wait_seq(backend, seq);
        return;
    }
    s->io.pf.wait(backend);
}

void llama_store_adopt_expert(struct llama_store * s, int32_t il, int32_t ie, int64_t bytes) {
    if (s == nullptr || bytes <= 0 || il < 0 || ie < 0) {
        return;
    }
    // grow the two-level index lazily in adopt order; after the first pass both hops are O(1)
    if ((int32_t) s->expert_handle.size() <= il) {
        s->expert_handle.resize(il + 1);
        s->expert_cls.resize(il + 1, -1);
    }
    std::vector<llama_store_slab::handle> & row = s->expert_handle[il];
    if ((int32_t) row.size() <= ie) {
        row.resize(ie + 1);
    }
    if (s->expert_cls[il] < 0) {
        s->expert_cls[il] = s->slab.add_class(bytes, 64);
    }

    // slots are cheap: grow the class in chunks until the adopt succeeds (never abort on capacity)
    llama_store_slab::handle h;
    int32_t grown = 0;
    while (!s->slab.alloc(s->expert_cls[il], s->policy->adopt_tier(), &h)) {
        s->slab.grow_class(s->expert_cls[il], 64);
        if (++grown > 1024) {
            LLAMA_LOG_WARN("%s: layer %d expert %d: class growth exceeded the sanity cap, object left behind\n",
                    __func__, il, ie);
            return;
        }
    }
    // one slot per adopted object; the replica model of 2c moves objects across tiers while the
    // handle identity and this index stay untouched
    row[ie] = h;
    s->n_expert_adopt++;
}

void llama_store_commit(struct llama_store * s) {
    if (s == nullptr || s->committed) {
        return;
    }
    s->committed = true;

    constexpr double MiB = 1024.0*1024.0;
    const int64_t budgets[3] = { s->params.vram_budget, s->params.ram_budget, s->params.ssd_budget };

    for (int tier = 0; tier < 3; tier++) {
        int64_t declared = 0;
        for (int kind = 0; kind < 3; kind++) {
            declared += s->declared[kind][tier];
        }
        const int64_t adopted = s->slab.bytes(tier);
        // adopted experts still live inside the declared dense buffers until the 2c repack moves
        // them into store-owned windows, so counting them twice would be wrong today. They join
        // the accounted sum as soon as their physical bytes leave the model buffers.
        const bool colocated = adopted > 0 && s->declared[LLAMA_STORE_DECL_DENSE][tier] > 0 &&
                               tier == LLAMA_STORE_TIER_RAM;
        const int64_t accounted = declared + (colocated ? 0 : adopted);
        const bool has_budget   = budgets[tier] > 0;
        if (!has_budget && accounted == 0) {
            continue;
        }
        LLAMA_LOG_INFO("%s: store_split: %s accounted %.1f MiB = declared dense %.1f compute %.1f staging %.1f%s adopted %.1f in %d registry slots%s\n",
                __func__, llama_store_tier_name(tier), accounted/MiB,
                s->declared[LLAMA_STORE_DECL_DENSE][tier]/MiB,
                s->declared[LLAMA_STORE_DECL_COMPUTE][tier]/MiB,
                s->declared[LLAMA_STORE_DECL_STAGING][tier]/MiB,
                colocated ? "+ co-located" : "+",
                adopted/MiB, s->slab.used_slots(tier),
                has_budget ? "" : " (no budget)");
        if (has_budget) {
            const int64_t free_room = budgets[tier] - accounted;
            LLAMA_LOG_INFO("%s: store_split: %s free within budget %.1f MiB\n",
                    __func__, llama_store_tier_name(tier), free_room/MiB);
        }
    }
    // the context class (KV + RS) and the MoE bank register at F12 M3; until then they appear as
    // free room above, that is the known accounting gap of 0.0.1, not a leak
}
