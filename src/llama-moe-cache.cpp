#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
// the slots [slot_base, slot_base + n_slots) of the group are managed here, the slots of the fully resident layers are below it
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t slot_base = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = n_slots > 0 ? 0 : -1;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // drop the residency map: the whole-layer prefetch writes into the slots of the pool, so nothing is cached
    void invalidate() {
        for (int32_t * map : slot_map) {
            if (map != nullptr) {
                std::fill(map, map + n_expert, -1);
            }
        }
        std::fill(key_of.begin(), key_of.end(), -1);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = n_slots > 0 ? 0 : -1;
        tail[0] = n_slots - 1;
        head[1] = tail[1] = -1;
        is_protected.assign(n_slots, false);
        n_protected = 0;
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e] - slot_base);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = slot_base + s;
            move(s, false);
            fills.push_back({ e, slot_base + s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;              // total hub slots, zero slot included at the end only when hybrid
        int32_t n_full  = 0;              // first n_full layers keep every expert in the bank
        bool pf_dirty = false;            // the prefetch rings overwrote the SLRU slots of this group
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        bool full = false;                  // every expert of the layer is in the bank
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * miss_mask = nullptr;  // I32 [n_expert_used_max, max_batch] in host memory, hybrid only
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;                       // index of the bank
        ggml_tensor * cached;             // view of the bank used in place of the host experts
        ggml_tensor * cached_static;      // the same view when the whole layer is resident
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    // [LLAMA_MOE_HYBRID_EXEC] the experts that a decode token misses in the cache, computed by the worker on a
    // detached graph. one job per cached layer, built the first time the graph of the layer is created
    struct hybrid_job {
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;

        ggml_tensor * inp     = nullptr; // F32 [n_embd, 1, 1]
        ggml_tensor * ids     = nullptr; // I32 [n_expert_used, 1, 1]
        ggml_tensor * mask    = nullptr; // I32 [n_expert_used, 1], 1 = compute on the CPU
        ggml_tensor * weights = nullptr; // F32 [1, n_expert_used, 1]
        ggml_tensor * out     = nullptr; // F32 [n_embd, 1], the sum of the misses

        ggml_cgraph * graph = nullptr;
        ggml_cplan    cplan = {};
        std::vector<uint8_t> work; // work_data of this graph, only one job runs at a time but the graphs are built between evals

        ggml_tensor * cpu_out = nullptr; // the device F32 [n_embd, 1] read by the combine split
        ggml_tensor * out_host = nullptr; // pinned staging of the worker result

        const ggml_tensor * src_inp     = nullptr; // the tensors staged by the post-split callback
        const ggml_tensor * src_weights = nullptr;

        int32_t seq = -1; // the submission sequence number, -1 if the worker has nothing to wait for
    };

    static constexpr int64_t max_batch = 32;

    const bool hybrid_mode; // the misses are computed on the CPU from the host experts

    int32_t n_expert = 0;
    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    stats stats_hybrid; // hits and misses of the hybrid small-batch path

    bool resident = false;               // the static residency is built
    std::vector<uint32_t> usage;         // [n_layer*n_expert] selections counted in the large-batch path
    size_t n_cached_layers = 0;          // layers with a residency map, one log line per token
    size_t win_hits = 0;
    size_t win_ids  = 0;
    size_t win_calls = 0;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // [LLAMA_MOE_TRACE] appends the selected experts of every MoE-cache lookup for offline analysis
    FILE *   trace_file    = nullptr;
    uint64_t trace_records = 0;

    // [LLAMA_MOE_STATS] persistent, decayed per-expert selection weights; read by the repack tool
    std::string        stats_path;
    std::vector<float> stats_w; // [n_layer*n_expert]
    int32_t            stats_last_il = -1;
    float              stats_decay   = 0.999f;
    uint64_t           stats_tokens  = 0;
    uint64_t           stats_since_save = 0;
    bool               stats_dirty   = false;

    // [LLAMA_MOE_DEFER_UPLOAD] last expert tensor of a layer uploaded as a deferred batch
    const ggml_tensor * deferred_bank = nullptr;
    ggml_backend_t deferred_backend = nullptr;
    size_t n_deferred_bytes  = 0;
    size_t n_deferred_waits  = 0;
    size_t n_deferred_missed = 0;

    // [LLAMA_MOE_PREFETCH] whole-layer expert prefetch for large ubatches, overlapped with the compute of
    // the previous layer. The layers stream through two ring buffers taken from the LRU pool.
    struct prefetch_api {
        bool (*tensor)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t) = nullptr;
        void (*submit)(ggml_backend_t) = nullptr;
        void (*wait)(ggml_backend_t) = nullptr;

        bool ok() const { return tensor != nullptr && submit != nullptr && wait != nullptr; }
    };

    bool pf_enabled = false;
    bool pf_requested = false;             // LLAMA_MOE_PREFETCH, also reserves the rings from the LRU pool
    int32_t pf_max_full = 4;               // static whole layers per group kept for decode when prefetch is on
    int64_t pf_min_tokens = 512;           // below this the whole-layer bytes cost more than the gather saves
    bool pf_batch = false;                 // the graph being built uses the prefetch path
    bool pf_pass_dirty = false;            // a prefetch pass overwrote the LRU pools, rebuild them before a small batch
    std::vector<int32_t> pf_order;         // prefetched layers in execution order
    std::vector<int32_t> pf_pos;           // [n_layer] position in pf_order, -1 if not prefetched
    prefetch_api pf;
    size_t pf_bytes   = 0;
    size_t pf_layers  = 0;
    size_t pf_waits   = 0;
    size_t pf_fallbacks = 0;
    size_t n_split_degrades = 0;          // lookups whose ids are produced in the same split (bench/THREE-TIER.md F1)
    int64_t pf_wait_us  = 0;              // [LLAMA_MOE_PREFETCH_TIME] host time spent waiting for the prefetch
    int64_t pf_copy_us  = 0;              // [LLAMA_MOE_PREFETCH_TIME] host time spent in the prefetch callback

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    ggml_tensor * ids_host = nullptr; // pinned destination of the selected-expert readback

    // the selected experts of each layer of the graph being built, and the last ones staged for readback
    mutable std::unordered_map<const ggml_tensor *, int32_t> ids_tensors;
    mutable const ggml_tensor * ids_staged = nullptr;
    mutable int32_t ids_staged_il = -1;
    mutable size_t  ids_staged_bytes = 0;

    // [LLAMA_MOE_HYBRID_EXEC] the worker and its mailbox: the slot map callback submits a layer, the combine
    // callback of the same layer waits, so at most one job is in flight
    bool                 hybrid_exec  = false;
    // the misses of a decode token are 1-3 experts per layer, so one thread beats a pool: the barriers of the
    // ~12 tiny ops per layer cost more than they save
    int32_t              hybrid_threads = 1;
    int64_t              hybrid_n_embd = 0;
    std::vector<hybrid_job> mutable hybrid_jobs;            // [n_layer], the graphs are built during graph build
    std::unordered_map<const ggml_tensor *, int32_t> cpu_out_of; // cpu_out -> layer
    mutable std::unordered_map<const ggml_tensor *, int32_t> hybrid_srcs; // staged inp/weights -> layer
    ggml_tensor * inp_host = nullptr; // pinned staging of the layer input
    ggml_tensor * w_host   = nullptr; // pinned staging of the router weights

    // the staged tensor of the last split, nullptr while it has not been read by a submit
    mutable const ggml_tensor * inp_staged = nullptr;
    mutable const ggml_tensor * w_staged   = nullptr;

    std::thread             hybrid_thread;
    std::mutex              hybrid_mtx;
    std::condition_variable hybrid_cv;
    std::condition_variable hybrid_cv_done;
    int32_t             hybrid_slot   = -1;
    int32_t             hybrid_submitted = 0;
    int32_t             hybrid_completed = 0;
    bool                hybrid_stop   = false;
    bool                hybrid_failed = false;
    ggml_threadpool_t   hybrid_tp     = nullptr;
    size_t              hybrid_waits  = 0;
    size_t              hybrid_skips  = 0; // layers whose experts were all in the bank
    int64_t             hybrid_wait_us = 0; // host time blocked in the combine callback
    int64_t             hybrid_jobs_us = 0; // worker time
    size_t              hybrid_inp_staged  = 0;
    size_t              hybrid_inp_sync    = 0; // the readback was not staged, the submit had to drain
    size_t              hybrid_w_staged    = 0;
    size_t              hybrid_w_sync      = 0;

    // [LLAMA_MOE_HYBRID_EXEC] rolling residency: every lookup plans the next token with the LRU and starts the
    // promotion through the prefetch batch, the map published to the device lags one token behind the plan
    bool  rolling = false;
    std::vector<ggml_tensor *>        promote_map;     // [n_layer] pinned map that the device map is copied from
    std::vector<uint8_t>              promote_pending; // [n_layer] the promotion of the layer is in flight
    std::vector<moe_cache_slru::fill> promote_fills;
    size_t                            promote_bytes = 0;
    size_t                            promote_waits = 0;
    size_t                            promote_fallbacks = 0;
    int64_t                           promote_wait_us = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, bool hybrid) :
            hybrid_mode(hybrid), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        n_expert = model.hparams.n_expert;
        // whole-layer prefetch is on by default, LLAMA_MOE_PREFETCH=0 disables it
        const char * env_pf = getenv("LLAMA_MOE_PREFETCH");
        pf_requested = env_pf == nullptr || atoi(env_pf) != 0;
        if (pf_requested) {
            const char * env_max_full = getenv("LLAMA_MOE_PREFETCH_MAX_FULL");
            if (env_max_full != nullptr) {
                pf_max_full = std::max(0, atoi(env_max_full));
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (hybrid_mode) {
            usage.assign(model.layers.size()*n_expert, 0);
        }

        // the executor is experimental and off by default, LLAMA_MOE_HYBRID_EXEC=1 turns it on
        hybrid_exec = hybrid_mode && getenv("LLAMA_MOE_HYBRID_EXEC") != nullptr &&
            atoi(getenv("LLAMA_MOE_HYBRID_EXEC")) != 0;
        if (hybrid_exec) {
            const char * env_threads = getenv("LLAMA_MOE_HYBRID_EXEC_T");
            if (env_threads != nullptr) {
                hybrid_threads = std::max(1, atoi(env_threads));
            }
            hybrid_n_embd = model.hparams.n_embd;
            hybrid_jobs.resize(model.layers.size());
            // the graphs are planned when they are built, so the pool must exist before that
            struct ggml_threadpool_params ttp = ggml_threadpool_params_default(hybrid_threads);
            // the jobs arrive a few hundred microseconds apart, a sleeping pool costs more than the work
            const char * env_poll = getenv("LLAMA_MOE_HYBRID_EXEC_POLL");
            if (env_poll != nullptr) {
                ttp.poll = (uint32_t) std::max(0, atoi(env_poll));
            }
            hybrid_tp = ggml_threadpool_new(&ttp);
            if (hybrid_tp == nullptr) {
                LLAMA_LOG_WARN("%s: cannot create the worker threadpool, the hybrid executor is disabled\n", __func__);
                hybrid_exec = false;
            }
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // a bank is one tensor, so it must fit in one allocation of the device
        auto largest_bank = [](const group & g, int32_t n_slots) {
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res = std::max(res, (size_t) t->nb[2]*(n_slots + 1));
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;

        auto group_budget = [&](const group & g) {
            const device & d = devices[g.id];
            return (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
        };

        // a bank that does not fit in one buffer cannot be allocated, split the group by layers.
        // All layers of a group share the layout, so each chunk keeps its share of the budget and slots.
        // The split limit comes from the backend, not from a device specific constant.
        {
            std::vector<group> chunks;
            chunks.reserve(groups.size());
            for (group & g : groups) {
                const size_t limit    = ggml_backend_buft_get_max_alloc_size(devices[g.id].buft);
                const int32_t n_layers = (int32_t) g.layers.size();
                size_t per_slot = 0;
                for (const ggml_tensor * t : g.ref) {
                    per_slot = std::max(per_slot, (size_t) t->nb[2]);
                }
                int32_t n_slots = 0;
                while (n_slots < n_layers*g.ref[0]->ne[2] && alloc_size(g, n_slots + 1) <= group_budget(g)) {
                    n_slots++;
                }
                const size_t bank = per_slot*(n_slots + 1);

                int32_t n_chunks = 1;
                if (bank > limit && n_layers > 1) {
                    if (limit <= per_slot) {
                        n_chunks = n_layers;
                    } else {
                        n_chunks = (int32_t) ((bank - per_slot + limit - per_slot - 1)/(limit - per_slot));
                        n_chunks = std::max(2, std::min(n_chunks, n_layers));
                    }
                    LLAMA_LOG_INFO("%s: %d layers share a bank of %.1f MiB, split into %d groups to fit the device buffer\n",
                            __func__, n_layers, bank/1024.0/1024.0, n_chunks);
                }
                for (int32_t ic = 0; ic < n_chunks; ++ic) {
                    group & dst = chunks.emplace_back();
                    dst.id   = g.id;
                    dst.ref  = g.ref;
                    const int32_t first = n_layers*ic/n_chunks;
                    const int32_t last  = n_layers*(ic + 1)/n_chunks;
                    dst.layers.assign(g.layers.begin() + first, g.layers.begin() + last);
                    dst.host_bytes = g.host_bytes*(last - first)/n_layers;
                }
            }
            groups = std::move(chunks);
        }

        // pass 1: the slot count of each group, it does not depend on the residency split
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = group_budget(g);
            const size_t  limit     = ggml_backend_buft_get_max_alloc_size(d.buft);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots &&
                   alloc_size(g, g.n_slots + 1) <= budget &&
                   largest_bank(g, g.n_slots + 1) <= limit) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget or device buffer limit is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
            }
        }

        // the prefetch rings take two whole layers of slots from each group, otherwise the plain cache is kept
        // the hybrid executor resolves the API without rings: it uses the prefetch batch for the promotion
        if (pf_requested || hybrid_exec) {
            init_prefetch_api(model);
            if (hybrid_exec && pf.ok()) {
                rolling = true;
                promote_map.resize(model.layers.size(), nullptr);
                promote_pending.assign(model.layers.size(), 0);
                LLAMA_LOG_INFO("%s: rolling residency enabled, the promoted experts go through the prefetch batch\n", __func__);
            }
            for (const group & g : groups) {
                if (hybrid_mode) {
                    // the rings are disabled in hybrid mode, the slots stay in the LRU pool
                    continue;
                }
                if (g.n_slots != 0 && g.n_slots < 2*g.ref[0]->ne[2]) {
                    LLAMA_LOG_WARN("%s: prefetch needs %d slots per group, group %td has %d, disabled\n",
                            __func__, 2*g.ref[0]->ne[2], &g - groups.data(), g.n_slots);
                    pf_requested = false;
                }
            }
        }

        // pass 2: the residency split, the static whole layers and the LRU pool of the rest
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_expert = g.ref[0]->ne[2];
            // whole layers never copy host experts during prefill, at most 5 per group so the LRU pool still helps decode
            // the prefetch rings take two whole layers of slots, the rest of the static whole layers are
            // kept for decode, LLAMA_MOE_PREFETCH_MAX_FULL sets how many per group
            constexpr int32_t max_full = 5;
            int32_t n_full = std::min(max_full, g.n_slots / n_expert);
            if (pf_requested) {
                n_full = std::min(n_full, std::max(0, (g.n_slots - 2*n_expert) / n_expert));
                n_full = std::min(n_full, pf_max_full);
            }
            n_full = std::min(n_full, (int32_t) g.layers.size() - 1);
            // do not starve the LRU pool: a bank that is mostly whole layers is worse than the plain static placement
            while (n_full > 0 && g.n_slots - n_full*n_expert < g.n_slots/5) {
                n_full--;
            }
            if (hybrid_mode) {
                // the residency map and the zero slot cover every layer: a whole layer would be read through a
                // static view that build_residency does not fill
                n_full = 0;
            }
            g.n_full = n_full;
            // the rest of the slots are an LRU on the remaining layers
            g.slru.slot_base = g.n_full*n_expert;
            g.slru.init(model.layers.size(), n_expert, g.n_slots - g.slru.slot_base);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size() + g.n_full);
            n_tensors_host  += g.layers.size()*(hybrid_mode ? 2 : 1);
            if (hybrid_exec) {
                n_tensors_host += g.layers.size() + 2; // the pinned staging of the worker output, input and weights
                n_tensors[g.id] += g.layers.size();    // the device tensor that the combine add reads
            }
            if (rolling) {
                n_tensors_host += g.layers.size(); // the published promotion map of every layer
            }
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host + 1);
        ctx_views = init_ctx(2);

        // the slot maps are written by the host and copied before every split, a pinned host buffer
        // makes that copy a direct transfer instead of a staging copy followed by a device drain
        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        if (devices.size() == 1) {
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            if (host_buft != nullptr) {
                buft_host = host_buft;
            }
        }
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        // the ids are read back into pinned host memory: a pageable destination would take the
        // staging path with its extra device drain
        ids_host = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, (int64_t) n_expert_used * max_batch);
        ggml_format_name(ids_host, "moe_cache.ids_host");
        buf_host_size += GGML_PAD(ggml_nbytes(ids_host), alignment_host);

        if (hybrid_exec) {
            // the worker reads the layer input and the router weights of a decode token from host memory
            inp_host = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, hybrid_n_embd);
            ggml_format_name(inp_host, "moe_cache.inp_host");
            buf_host_size += GGML_PAD(ggml_nbytes(inp_host), alignment_host);

            w_host = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, n_expert_used);
            ggml_format_name(w_host, "moe_cache.w_host");
            buf_host_size += GGML_PAD(ggml_nbytes(w_host), alignment_host);
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (size_t j = 0; j < g.layers.size(); ++j) {
                const int32_t il = g.layers[j];
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.full    = (int32_t) j < g.n_full;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                // in hybrid mode the misses read the zero slot at the end of the bank, so the views include it
                const int64_t n_view = g.n_slots + (hybrid_mode ? 1 : 0);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], n_view, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    // a whole layer keeps its experts at slot j*n_expert, so the bank view is a drop in replacement of the host tensor
                    ggml_tensor * cached_static = nullptr;
                    if (l.full) {
                        cached_static = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], n_expert, bank->nb[1], bank->nb[2], (size_t) j*n_expert*bank->nb[2]);
                        ggml_format_name(cached_static, "moe_cache.static.%s", l.experts[ip]->name);
                    }
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached, cached_static };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (hybrid_mode) {
                    l.miss_mask = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, n_expert_used, max_batch);
                    ggml_format_name(l.miss_mask, "moe_cache.miss_mask-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.miss_mask), alignment_host);
                    if (rolling) {
                        ggml_tensor * pm = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                        ggml_format_name(pm, "moe_cache.promote_map-%d", il);
                        promote_map[il] = pm;
                        buf_host_size += GGML_PAD(ggml_nbytes(pm), alignment_host);
                    }
                }
                if (hybrid_exec) {
                    // the combine add runs on the device: a device tensor keeps the add in the MoE split and
                    // removes the second host round trip of the layer
                    ggml_tensor * cpu_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hybrid_n_embd, 1);
                    ggml_format_name(cpu_out, "moe_cache.cpu_out-%d", il);
                    cpu_out_of[cpu_out]      = il;
                    hybrid_jobs[il].cpu_out  = cpu_out;

                    ggml_tensor * out_host = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, hybrid_n_embd);
                    ggml_format_name(out_host, "moe_cache.out_host-%d", il);
                    hybrid_jobs[il].out_host = out_host;
                    buf_host_size += GGML_PAD(ggml_nbytes(out_host), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // log the bank layout: whole layers plus the LRU pool of the remaining layers
        {
            size_t n_moe = 0;
            size_t n_full = 0;
            size_t n_partial = 0;
            for (size_t il = 0; il < model.layers.size(); ++il) {
                if (llama_moe_cache_layer_experts(model.layers[il]).empty()) {
                    continue;
                }
                n_moe++;
                if (layers[il].full) {
                    n_full++;
                } else if (layers[il].ig >= 0) {
                    n_partial++;
                }
            }
            LLAMA_LOG_INFO("%s: %zu of %zu MoE layers have a bank: %zu whole layers, %zu layers share the LRU pool\n",
                __func__, n_full + n_partial, n_moe, n_full, n_partial);
            for (const group & g : groups) {
                if (g.n_slots == 0) {
                    continue;
                }
                LLAMA_LOG_INFO("%s: group %td: %zu layers, %d whole layers, %d LRU slots for %zu layers\n",
                    __func__, &g - groups.data(), g.layers.size(), g.n_full, g.slru.n_slots, g.layers.size() - g.n_full);
            }
        }
        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
            prepopulate();
        }

        init_prefetch(model);

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }

        for (const layer & l : layers) {
            n_cached_layers += l.ig >= 0;
        }

        const char * trace_path = getenv("LLAMA_MOE_TRACE");
        if (trace_path != nullptr && trace_path[0] != '\0') {
            trace_file = fopen(trace_path, "ab");
            if (trace_file == nullptr) {
                LLAMA_LOG_WARN("%s: cannot open the MoE trace file %s, tracing disabled\n", __func__, trace_path);
            } else {
                LLAMA_LOG_INFO("%s: MoE trace enabled, appending records to %s\n", __func__, trace_path);
            }
        }

        const char * stats_env = getenv("LLAMA_MOE_STATS");
        if (stats_env != nullptr && stats_env[0] != '\0') {
            stats_path = stats_env;
            const char * decay_env = getenv("LLAMA_MOE_STATS_DECAY");
            if (decay_env != nullptr) {
                const float d = (float) atof(decay_env);
                if (d > 0.0f && d <= 1.0f) {
                    stats_decay = d;
                }
            }
            stats_w.assign(model.layers.size()*n_expert, 0.0f);
            load_stats();
            LLAMA_LOG_INFO("%s: MoE stats enabled, decay %.4f, file %s\n", __func__, stats_decay, stats_path.c_str());
        }

        if (hybrid_exec) {
            hybrid_thread = std::thread([this] { hybrid_worker(); });
            LLAMA_LOG_INFO("%s: hybrid executor enabled, the misses of a decode token are computed on %d CPU threads\n",
                __func__, hybrid_threads);
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        if (hybrid_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lock(hybrid_mtx);
                hybrid_stop = true;
            }
            hybrid_cv.notify_one();
            hybrid_thread.join();
        }
        if (hybrid_tp != nullptr) {
            ggml_threadpool_free(hybrid_tp);
            hybrid_tp = nullptr;
        }
        hybrid_log();
        // only a context that routed tokens owns the stats file; load-time probe contexts must not clobber it
        if (stats_dirty) {
            save_stats();
            LLAMA_LOG_INFO("%s: MoE stats: %llu tokens recorded, file %s\n", __func__,
                    (unsigned long long) stats_tokens, stats_path.c_str());
        }
        if (trace_file != nullptr) {
            fclose(trace_file);
            trace_file = nullptr;
            LLAMA_LOG_INFO("%s: MoE trace records = %llu\n", __func__, (unsigned long long) trace_records);
        }
        log_stats();
    }

    void load_stats() {
        FILE * f = fopen(stats_path.c_str(), "rb");
        if (f == nullptr) {
            return;
        }
        uint32_t hdr[4] = { 0, 0, 0, 0 };
        const bool hdr_ok = fread(hdr, sizeof(uint32_t), 4, f) == 4 &&
            hdr[0] == 0x53454f4d && hdr[1] == 1 &&
            hdr[2] == (uint32_t) layers.size() && hdr[3] == (uint32_t) n_expert;
        std::vector<float> tmp(stats_w.size());
        const bool ok = hdr_ok && fread(tmp.data(), sizeof(float), tmp.size(), f) == tmp.size();
        fclose(f);
        if (!ok) {
            LLAMA_LOG_WARN("%s: MoE stats file %s does not match this model, starting empty\n", __func__, stats_path.c_str());
            return;
        }
        stats_w.swap(tmp);
        LLAMA_LOG_INFO("%s: MoE stats loaded from %s\n", __func__, stats_path.c_str());
    }

    void save_stats() {
        if (stats_w.empty()) {
            return;
        }
        const std::string tmp = stats_path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: cannot open %s for writing\n", __func__, tmp.c_str());
            return;
        }
        const uint32_t hdr[4] = { 0x53454f4d, 1, (uint32_t) layers.size(), (uint32_t) n_expert };
        const bool ok = fwrite(hdr, sizeof(uint32_t), 4, f) == 4 &&
            fwrite(stats_w.data(), sizeof(float), stats_w.size(), f) == stats_w.size();
        fclose(f);
        // std::rename does not replace an existing file on Windows
        remove(stats_path.c_str());
        if (!ok || std::rename(tmp.c_str(), stats_path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: cannot write the MoE stats file %s\n", __func__, stats_path.c_str());
            remove(tmp.c_str());
            return;
        }
        stats_since_save = 0;
    }

    // decay once per token (a layer-id wrap marks a token boundary), then count the selected experts
    void observe_ids(int32_t il, const std::vector<int32_t> & ids) {
        if (stats_w.empty()) {
            return;
        }
        if (stats_last_il >= 0 && il <= stats_last_il) {
            for (float & v : stats_w) {
                v *= stats_decay;
            }
            stats_tokens++;
            if (++stats_since_save >= 512) {
                save_stats();
            }
        }
        stats_last_il = il;
        stats_dirty = true;
        float * row = stats_w.data() + (size_t) il*n_expert;
        for (int32_t e : ids) {
            if (e >= 0 && e < n_expert) {
                row[e] += 1.0f;
            }
        }
    }

    // record = int32 layer, int32 count, count x int32 expert id; one record per traced layer lookup
    void trace_ids(int32_t il, const std::vector<int32_t> & ids) {
        if (trace_file == nullptr) {
            return;
        }
        const int32_t n = (int32_t) ids.size();
        const bool ok = fwrite(&il, sizeof(il), 1, trace_file) == 1 &&
            fwrite(&n, sizeof(n), 1, trace_file) == 1 &&
            (n == 0 || fwrite(ids.data(), sizeof(int32_t), (size_t) n, trace_file) == (size_t) n);
        if (!ok) {
            LLAMA_LOG_WARN("%s: failed to write the MoE trace, tracing disabled\n", __func__);
            fclose(trace_file);
            trace_file = nullptr;
            return;
        }
        trace_records++;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];

        // a whole layer reads the bank through a static view, it needs no slot map
        if (l.full) {
            return nullptr;
        }

        // the prefetch path maps every expert to a ring slot, the map is written by the copy callback
        if (pf_batch) {
            return l.slot_map;
        }

        // in hybrid mode the misses are computed on the CPU, so the residency map is always used for small batches
        if (hybrid_mode) {
            return n_tokens > 0 && n_tokens <= max_batch ? l.slot_map : nullptr;
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.slru.n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    // the bank view of w when the whole layer is resident, the plain ids can index it directly
    ggml_tensor * get_experts_static(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it == bindings.end() ? nullptr : it->second.cached_static;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    bool hybrid(int32_t il) const {
        return hybrid_mode && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0;
    }

    ggml_tensor * get_miss_mask(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (!hybrid(il) || n_tokens <= 0 || n_tokens > max_batch) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (l.miss_mask == nullptr || n_expert_used != l.miss_mask->ne[0]) {
            return nullptr;
        }
        return l.miss_mask;
    }

    void count_usage(const ggml_tensor * w, const ggml_tensor * ids, const std::vector<int32_t> & ids_data) {
        if (!hybrid_mode || resident) {
            return;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        uint32_t * cnt = usage.data() + (size_t) it->second.il*n_expert;
        const int64_t nb0 = ids->nb[0]/sizeof(int32_t);
        const int64_t nb1 = ids->nb[1]/sizeof(int32_t);
        for (int64_t j = 0; j < ids->ne[1]; ++j) {
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                const int64_t k = j*nb1 + i*nb0;
                if (k >= (int64_t) ids_data.size()) {
                    continue;
                }
                const int32_t e = ids_data[k];
                if (e >= 0 && e < n_expert) {
                    cnt[e]++;
                }
            }
        }
    }

    // put a deterministic slice of every layer's experts into the bank, so that large batches can copy them from the
    // keep every expert of the first n_full layers of each group in the bank: those layers never copy host experts
    // to the device, which removes the per-layer weight copy that serializes the prefill
    void prepopulate() {
        if (hybrid_mode) {
            return;
        }
        size_t n_pre = 0;
        for (group & g : groups) {
            if (g.n_slots == 0 || g.n_full == 0) {
                continue;
            }
            const int32_t n_expert = g.ref[0]->ne[2];
            ggml_backend_t backend = devices[g.id].backend;
            for (int32_t j = 0; j < g.n_full; ++j) {
                const int32_t il = g.layers[j];
                int32_t * map = g.slru.slot_map[il];
                GGML_ASSERT(map != nullptr);
                for (int32_t e = 0; e < n_expert; ++e) {
                    map[e] = j*n_expert + e;
                }
                for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                    const ggml_tensor * w = layers[il].experts[ip];
                    ggml_backend_tensor_set_async(backend, g.banks[ip],
                            (const uint8_t *) w->data, (size_t) j*n_expert*w->nb[2], (size_t) n_expert*w->nb[2]);
                }
                // submit the layer, the staging buffers of the next layer would otherwise pile up
                ggml_backend_synchronize(backend);
                n_pre += n_expert;
            }
        }

        // [GGML_MOE_CACHE_PREFILL=0 disables] fill the LRU pool with a deterministic slice of the experts of the
        // layers that share it, so that a prefill copy reads those experts from the bank instead of the host
        const char * env_prefill = getenv("GGML_MOE_CACHE_PREFILL");
        const bool prefill_lru = env_prefill == nullptr || atoi(env_prefill) != 0;
        int32_t n_lru = 0;
        for (group & g : groups) {
            if (!prefill_lru || g.n_slots <= g.n_full*g.ref[0]->ne[2] || g.layers.size() <= (size_t) g.n_full) {
                continue;
            }
            const int32_t n_expert   = g.ref[0]->ne[2];
            const int32_t n_partial  = (int32_t) g.layers.size() - g.n_full;
            const int32_t lru_slots  = g.n_slots - g.n_full*n_expert;
            const int32_t quota      = lru_slots / n_partial;
            const int32_t remainder  = lru_slots % n_partial;
            ggml_backend_t backend = devices[g.id].backend;
            int32_t slot = g.slru.slot_base;
            for (int32_t j = 0; j < n_partial; ++j) {
                const int32_t il = g.layers[g.n_full + j];
                const int32_t n_keep = quota + (j < remainder ? 1 : 0);
                int32_t * map = g.slru.slot_map[il];
                GGML_ASSERT(map != nullptr);
                GGML_ASSERT(slot + n_keep <= g.n_slots);
                for (int32_t k = 0; k < n_keep; ++k) {
                    int32_t e = j*n_keep + k;
                    if (e >= n_expert) {
                        e -= n_expert*(e/n_expert);
                    }
                    map[e] = slot + k;
                    g.slru.key_of[slot + k - g.slru.slot_base] = il*n_expert + e;
                }
                for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                    const ggml_tensor * w = layers[il].experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (int32_t k = 0; k < n_keep; ++k) {
                        int32_t e = j*n_keep + k;
                        if (e >= n_expert) {
                            e -= n_expert*(e/n_expert);
                        }
                        ggml_backend_tensor_set_async(backend, g.banks[ip],
                                (const uint8_t *) w->data + (size_t) e*expert_size, (size_t) (slot + k)*expert_size, expert_size);
                    }
                }
                ggml_backend_synchronize(backend);
                slot += n_keep;
                n_lru += n_keep;
            }
            GGML_ASSERT(slot == g.n_slots);
        }

        LLAMA_LOG_INFO("%s: %zu whole layers resident, %zu experts (%.1f%% of the host experts), "
                "%d LRU slots prefilled (%.1f%% of the host experts)\n",
            __func__, n_pre/n_expert, n_pre, 100.0*n_pre/((double) layers.size()*n_expert),
            n_lru, 100.0*n_lru/((double) layers.size()*n_expert));
    }

    // one-time upload of the most used experts of each layer into the slots that the residency map assigns to them
    void build_residency() {
        std::vector<int32_t> order(n_expert);
        int64_t n_uploaded_total = 0;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            // the previous large-batch copies may still read the banks
            ggml_backend_synchronize(devices[g.id].backend);

            const int32_t n_layers_g = (int32_t) g.layers.size();
            const int32_t quota      = g.n_slots / n_layers_g;
            const int32_t remainder  = g.n_slots % n_layers_g;
            int64_t n_uploaded = 0;
            int32_t slot = 0;
            for (int32_t j = 0; j < n_layers_g; ++j) {
                const int32_t il     = g.layers[j];
                const int32_t n_keep = quota + (j < remainder ? 1 : 0);
                int32_t * map = g.slru.slot_map[il];
                GGML_ASSERT(map != nullptr);
                GGML_ASSERT(slot + n_keep <= g.n_slots);

                std::iota(order.begin(), order.end(), 0);
                const uint32_t * cnt = usage.data() + (size_t) il*n_expert;
                std::stable_sort(order.begin(), order.end(), [cnt](int32_t a, int32_t b) {
                    return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b;
                });

                // rolling residency keeps the LRU convention (-1 = not in the bank), the published map turns
                // it back into the zero slot of the GPU map
                std::fill(map, map + n_expert, rolling ? -1 : g.n_slots);
                for (int32_t k = 0; k < n_keep; ++k) {
                    map[order[k]] = slot + k;
                    // the SLRU must know the owner of every slot it may evict
                    g.slru.key_of[slot + k - g.slru.slot_base] = il*n_expert + order[k];
                }
                for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                    const ggml_tensor * w = layers[il].experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (int32_t k = 0; k < n_keep; ++k) {
                        ggml_backend_tensor_set_async(devices[g.id].backend, g.banks[ip],
                                (const uint8_t *) w->data + (size_t) order[k]*expert_size, (size_t) (slot + k)*expert_size, expert_size);
                    }
                    stats_hybrid.bytes += (size_t) n_keep*expert_size;
                }
                n_uploaded += n_keep;
                slot += n_keep;
            }
            GGML_ASSERT(slot == g.n_slots);
            n_uploaded_total += n_uploaded;
            LLAMA_LOG_INFO("%s: group %td: %4d layers, %5d slots, quota %d (+%d), %5.1f%% of the experts resident, %6lld experts frozen\n",
                __func__, &g - groups.data(), n_layers_g, g.n_slots, quota, remainder,
                100.0*g.n_slots/(double) (n_layers_g*n_expert), (long long) n_uploaded);
        }
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_synchronize(d.backend);
            }
        }
        resident = true;
        LLAMA_LOG_INFO("%s: hybrid residency frozen: %lld experts, %.2f MiB uploaded once\n",
            __func__, (long long) n_uploaded_total, stats_hybrid.bytes/1024.0/1024.0);
    }

    // [LLAMA_MOE_PREFETCH] resolve the optional backend API and check the preconditions of the whole-layer
    // prefetch, called before the residency split so that an unusable prefetch leaves the plain cache unchanged
    void init_prefetch_api(const llama_model & model) {
        if (model.hparams.no_alloc || (hybrid_mode && !hybrid_exec)) {
            pf_requested = false;
            return;
        }
        const char * env_min = getenv("LLAMA_MOE_PREFETCH_MIN_TOKENS");
        if (env_min != nullptr) {
            pf_min_tokens = atoll(env_min);
        }
        if (devices.size() != 1) {
            LLAMA_LOG_WARN("%s: prefetch needs a single device, disabled\n", __func__);
            pf_requested = false;
            return;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        for (const group & g : groups) {
            if (host_buft == nullptr || g.ref.empty() || ggml_backend_buffer_get_type(g.ref[0]->buffer) != host_buft) {
                LLAMA_LOG_WARN("%s: prefetch needs pinned host experts, a group is not pinned, disabled\n", __func__);
                pf_requested = false;
                return;
            }
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        pf.tensor = (decltype(pf.tensor)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_tensor");
        pf.submit = (decltype(pf.submit)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_submit");
        pf.wait   = (decltype(pf.wait))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_wait");
        if (!pf.ok()) {
            LLAMA_LOG_WARN("%s: the backend does not provide the prefetch API, disabled\n", __func__);
            pf_requested = false;
        }
    }

    // [LLAMA_MOE_PREFETCH] the layers stream through the rings in execution order, the whole layers keep their static views
    void init_prefetch(const llama_model & model) {
        // the prefetch rings are for large ubatches, the hybrid executor uses the batch for the promotion
        if (!pf_requested || hybrid_mode) {
            return;
        }
        pf_pos.assign(layers.size(), -1);
        for (group & g : groups) {
            for (int32_t il : g.layers) {
                if (il >= (int32_t) model.hparams.n_layer() || layers[il].full) {
                    continue;
                }
                pf_order.push_back(il);
            }
        }
        std::sort(pf_order.begin(), pf_order.end());
        for (size_t i = 0; i < pf_order.size(); ++i) {
            pf_pos[pf_order[i]] = (int32_t) i;
        }
        if (pf_order.empty()) {
            pf_requested = false;
            return;
        }
        pf_enabled = true;
        size_t ring_bytes = 0;
        for (const group & g : groups) {
            ring_bytes += 2*n_expert*g.banks[0]->nb[2]*g.banks.size();
        }
        LLAMA_LOG_INFO("%s: prefetch: %zu layers stream through 2 rings (%.1f MiB), min ubatch tokens %lld\n",
                __func__, pf_order.size(), ring_bytes/1024.0/1024.0, (long long) pf_min_tokens);
    }

    // large ubatches take the prefetch path, the residency map of the LRU pool is stale after it
    void update_prefetch(int64_t n_tokens) {
        if (!pf_enabled) {
            return;
        }
        const bool batch = n_tokens >= pf_min_tokens;
        if (!batch) {
            for (group & g : groups) {
                if (g.pf_dirty) {
                    g.slru.invalidate();
                    g.pf_dirty = false;
                }
            }
        }
        pf_batch = batch;
        static const bool dbg = getenv("LLAMA_MOE_PREFETCH_DEBUG") != nullptr;
        if (dbg) {
            LLAMA_LOG_INFO("%s: n_tokens = %lld, batch = %d\n", __func__, (long long) n_tokens, (int) batch);
        }
    }

    // upload the experts of one layer into its ring slot range, the copies are not waited by the compute queue
    bool prefetch_layer(ggml_backend_t backend, int32_t pos) {
        const layer & l = layers[pf_order[pos]];
        group & g = groups[l.ig];
        const int32_t ring = g.n_full*n_expert + (pos % 2) * n_expert;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            const size_t size = (size_t) n_expert * w->nb[2];
            if (!pf.tensor(backend, g.banks[ip], w->data, (size_t) ring * w->nb[2], size)) {
                if (pf_fallbacks++ == 0) {
                    LLAMA_LOG_WARN("%s: prefetch of layer %d failed, is the transfer queue available and the source pinned?\n", __func__, pf_order[pos]);
                }
                return false;
            }
            pf_bytes += size;
        }
        pf_layers++;
        return true;
    }

    // [LLAMA_MOE_PREFETCH] whole-layer slot map, the compute reads the bank through the ring slots
    bool copy_prefetch(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, int32_t il) {
        const int32_t pos = pf_pos[il];
        if (pos < 0) {
            return false;
        }
        layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(backend == devices[g.id].backend);
        GGML_ASSERT(src == l.slot_map);

        static const bool time_it = getenv("LLAMA_MOE_PREFETCH_TIME") != nullptr;
        const int64_t t0 = time_it ? ggml_time_us() : 0;

        if (pos == 0) {
            // the first layer cannot be hidden, the compute of the previous layer is done at this point
            if (!prefetch_layer(backend, 0)) {
                return false;
            }
            pf.submit(backend);
            const int64_t tw = time_it ? ggml_time_us() : 0;
            pf.wait(backend);
            pf_waits++;
            pf_wait_us += time_it ? ggml_time_us() - tw : 0;
            if (time_it) {
                fprintf(stderr, "moe_prefetch_first: first layer upload %.3f ms\n", (ggml_time_us() - tw)/1000.0);
            }
        } else {
            // issued when the previous layer was set up, normally complete because it overlapped that layer's compute
            const int64_t tw = time_it ? ggml_time_us() : 0;
            pf.wait(backend);
            pf_waits++;
            pf_wait_us += time_it ? ggml_time_us() - tw : 0;
        }

        // the experts of the layer are read from the ring, the ids index it through this map
        const int32_t ring = g.n_full*n_expert + (pos % 2) * n_expert;
        int32_t * map = g.slru.slot_map[il];
        for (int32_t e = 0; e < n_expert; ++e) {
            map[e] = ring + e;
        }

        // the small copy that the compute of this layer waits on, it is submitted before the next prefetch
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        if (pos + 1 < (int32_t) pf_order.size() && !prefetch_layer(backend, pos + 1)) {
            return false;
        }
        pf.submit(backend);
        pf_copy_us += time_it ? ggml_time_us() - t0 : 0;
        if (time_it && pf_layers % 40 == 0) {
            fprintf(stderr, "moe_prefetch_timing: layers %zu, wait %.3f ms/layer, callback %.3f ms/layer\n",
                    pf_layers, pf_wait_us/1000.0/pf_waits, pf_copy_us/1000.0/pf_waits);
        }
        return true;
    }

    bool wait_deferred(ggml_backend_t backend, ggml_cgraph * graph) {
        if (deferred_bank == nullptr || deferred_backend != backend) {
            return false;
        }
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID &&
                (node->src[0] == deferred_bank || node->src[0]->view_src == deferred_bank)) {
                ggml_backend_wait_deferred(backend);
                deferred_bank = nullptr;
                n_deferred_waits++;
                return true;
            }
        }
        return false;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        // in hybrid mode the non-resident experts of the residency map point to the zero slot
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0 || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] >= 0 && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    // graph build: remember which tensor holds the selected experts of each cached layer
    void register_ids(int32_t il, const ggml_tensor * ids) const {
        ids_tensors[ids] = il;
    }

    void clear_ids() const {
        ids_tensors.clear();
        hybrid_srcs.clear();
        ids_staged = nullptr;
        ids_staged_il = -1;
        ids_staged_bytes = 0;
        inp_staged = nullptr;
        w_staged   = nullptr;
    }

    // the selected experts of a layer are computed by one split, the readback joins that split's submission
    // the host reads ids_host only after the scheduler waits for the backend before the next copy callback
    void stage_inputs(ggml_backend_t backend, ggml_cgraph * graph) {
        static const bool enabled = getenv("LLAMA_MOE_STAGE_IDS") == nullptr || atoi(getenv("LLAMA_MOE_STAGE_IDS")) != 0;
        if (!enabled || (ids_tensors.empty() && hybrid_srcs.empty()) ||
            ids_host == nullptr || ids_host->data == nullptr) {
            return;
        }
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            const auto it_h = hybrid_srcs.find(node);
            if (it_h != hybrid_srcs.end()) {
                // the layer input and the router weights of the hybrid executor, staged next to the ids
                hybrid_job & j = hybrid_jobs[it_h->second];
                const bool is_inp = node == j.src_inp || node == j.src_inp->view_src;
                ggml_tensor * dst = is_inp ? inp_host : w_host;
                if (dst == nullptr || dst->data == nullptr || backend != devices[layers[it_h->second].ig].backend ||
                    node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || ggml_nbytes(node) > ggml_nbytes(dst)) {
                    continue;
                }
                ggml_backend_tensor_get_async(backend, node, dst->data, 0, ggml_nbytes(node));
                if (is_inp) {
                    inp_staged = j.src_inp;
                } else {
                    w_staged = j.src_weights;
                }
                continue;
            }
            const auto it = ids_tensors.find(node);
            if (it == ids_tensors.end()) {
                continue;
            }
            const int32_t il = it->second;
            if (il < 0 || il >= (int32_t) layers.size()) {
                continue;
            }
            const layer & l = layers[il];
            if (l.ig < 0 || backend != devices[l.ig].backend || node->type != GGML_TYPE_I32 || !ggml_is_contiguous(node)) {
                continue;
            }
            const size_t bytes = ggml_nbytes(node);
            if (bytes > ggml_nbytes(ids_host)) {
                continue;
            }
            ggml_backend_tensor_get_async(backend, node, ids_host->data, 0, bytes);
            ids_staged = node;
            ids_staged_il = il;
            ids_staged_bytes = bytes;
        }
    }

    ggml_tensor * get_cpu_out(int32_t il) const {
        if (!hybrid_exec || il < 0 || il >= (int32_t) hybrid_jobs.size()) {
            return nullptr;
        }
        return hybrid_jobs[il].graph != nullptr ? hybrid_jobs[il].cpu_out : nullptr;
    }

    // the CPU branch of build_moe_ffn as a detached graph: the same ops, the same shapes, the host experts
    // a layer that does not match keeps the in-graph branch, the worker is only used by the layers registered here
    bool register_hybrid(int32_t il, const ggml_tensor * inp, const ggml_tensor * weights,
                         const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down) const {
        if (!hybrid_exec || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return false;
        }
        hybrid_job & j = hybrid_jobs[il];
        if (j.graph != nullptr) {
            j.src_inp     = inp;
            j.src_weights = weights;
            hybrid_srcs[inp]     = il;
            hybrid_srcs[weights] = il;
            // the scheduler splits on the producer, which is the base tensor when the layer input is a view
            if (inp->view_src != nullptr)     { hybrid_srcs[inp->view_src]     = il; }
            if (weights->view_src != nullptr) { hybrid_srcs[weights->view_src] = il; }
            return true;
        }

        auto fail = [&](const char * why) {
            LLAMA_LOG_WARN("%s: layer %d: %s, this layer keeps the in-graph CPU branch\n", __func__, il, why);
            j.ctx.reset();
            j.buf.reset();
            j.graph = nullptr;
            return false;
        };

        if (inp->type != GGML_TYPE_F32 || !ggml_is_contiguous(inp) ||
            inp->ne[0] != hybrid_n_embd || inp->ne[1] != 1 || inp->ne[2] != 1 || inp->ne[3] != 1 ||
            weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
            !ggml_backend_buffer_is_host(up->buffer) || !ggml_backend_buffer_is_host(gate->buffer) ||
            !ggml_backend_buffer_is_host(down->buffer) ||
            bindings.find(up) == bindings.end() || bindings.find(gate) == bindings.end() ||
            bindings.find(down) == bindings.end()) {
            return fail("the tensors of the CPU branch are not the ones the worker expects");
        }
        const int64_t n_used = ggml_nelements(weights);
        if (n_used < 1 || n_used > n_expert_used || ggml_nbytes(weights) != (size_t) n_used*sizeof(float) ||
            gate->ne[2] != down->ne[2] || gate->ne[1] != down->ne[0]) {
            return fail("the expert layout of the layer does not match the worker graph");
        }

        // the graph object lives in the same context, so its overhead is part of the budget
        const size_t graph_size = 32 + 4*n_used;
        ggml_init_params params = {
            /*.mem_size   =*/ (16 + 4*n_used)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        j.ctx.reset(ggml_init(params));
        if (!j.ctx) {
            return fail("failed to create the worker context");
        }
        ggml_context * ctx = j.ctx.get();

        j.inp  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hybrid_n_embd, 1, 1);
        j.ids  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_used, 1, 1);
        j.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
        j.weights = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, weights->ne);

        // the ggml builders take non-const srcs but do not write them
        ggml_tensor * up_cpu   = ggml_mul_mat_id_mask(ctx, const_cast<ggml_tensor *>(up),   j.inp, j.ids, j.mask);
        ggml_tensor * gate_cpu = ggml_mul_mat_id_mask(ctx, const_cast<ggml_tensor *>(gate), j.inp, j.ids, j.mask);
        ggml_tensor * act      = ggml_swiglu_split(ctx, gate_cpu, up_cpu);
        ggml_tensor * down_cpu = ggml_mul_mat_id_mask(ctx, const_cast<ggml_tensor *>(down), act, j.ids, j.mask);
        ggml_tensor * w_cpu    = ggml_mul(ctx, down_cpu, j.weights);

        ggml_tensor * sum = nullptr;
        for (int64_t i = 0; i < n_used; ++i) {
            ggml_tensor * view = ggml_view_2d(ctx, w_cpu, hybrid_n_embd, 1, w_cpu->nb[2], i*w_cpu->nb[1]);
            sum = sum == nullptr ? view : ggml_add(ctx, sum, view);
        }
        j.out = sum;

        j.graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(j.graph, j.out);

        j.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()));
        if (!j.buf || ggml_nbytes(j.out) != (size_t) hybrid_n_embd*sizeof(float)) {
            return fail("failed to allocate the worker graph");
        }
        j.cplan = ggml_graph_plan(j.graph, hybrid_threads, hybrid_tp);
        j.work.resize(j.cplan.work_size);
        j.cplan.work_data = j.work.empty() ? nullptr : j.work.data();

        j.src_inp     = inp;
        j.src_weights = weights;
        hybrid_srcs[inp]     = il;
        hybrid_srcs[weights] = il;
        if (inp->view_src != nullptr)     { hybrid_srcs[inp->view_src]     = il; }
        if (weights->view_src != nullptr) { hybrid_srcs[weights->view_src] = il; }

        LLAMA_LOG_INFO("%s: layer %d: worker graph for %d experts, %zu KiB of work data\n",
            __func__, il, (int) n_used, j.cplan.work_size/1024);
        return true;
    }

    // hand the job of a layer to the worker, the inputs are the readback staged by the split that computed them
    void hybrid_submit(int32_t il, ggml_backend_t backend) {
        hybrid_job & j = hybrid_jobs[il];

        GGML_ASSERT(j.graph != nullptr && (int64_t) ids.size() == ggml_nelements(j.ids));
        ggml_backend_tensor_set(j.ids, ids.data(), 0, ggml_nbytes(j.ids));
        ggml_backend_tensor_set(j.mask, layers[il].miss_mask->data, 0, ggml_nbytes(j.mask));

        if (inp_staged == j.src_inp) {
            ggml_backend_tensor_set(j.inp, inp_host->data, 0, ggml_nbytes(j.inp));
            inp_staged = nullptr;
            hybrid_inp_staged++;
        } else {
            hybrid_inp_sync++;
            ggml_backend_tensor_get_async(backend, j.src_inp, inp_host->data, 0, ggml_nbytes(j.inp));
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_set(j.inp, inp_host->data, 0, ggml_nbytes(j.inp));
        }
        if (w_staged == j.src_weights) {
            ggml_backend_tensor_set(j.weights, w_host->data, 0, ggml_nbytes(j.weights));
            w_staged = nullptr;
            hybrid_w_staged++;
        } else {
            hybrid_w_sync++;
            ggml_backend_tensor_get_async(backend, j.src_weights, w_host->data, 0, ggml_nbytes(j.weights));
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_set(j.weights, w_host->data, 0, ggml_nbytes(j.weights));
        }

        std::unique_lock<std::mutex> lock(hybrid_mtx);
        // the combine of the previous layer waits first, so the mailbox holds one job
        GGML_ASSERT(hybrid_completed == hybrid_submitted);
        j.seq = ++hybrid_submitted;
        hybrid_slot = il;
        lock.unlock();
        hybrid_cv.notify_one();
    }

    void hybrid_combine(ggml_backend_t backend, int32_t il, ggml_tensor * dst) {
        hybrid_job & j = hybrid_jobs[il];

        int32_t seq;
        {
            std::lock_guard<std::mutex> lock(hybrid_mtx);
            GGML_ASSERT(j.seq >= 0 && "the slot map callback of the layer must submit the job first");
            seq = j.seq;
            j.seq = -1;
        }
        if (seq == 0) {
            // every expert was in the bank, the combine only adds zeros
            hybrid_skips++;
            std::memset(j.out_host->data, 0, ggml_nbytes(j.out));
            ggml_backend_tensor_set_async(backend, j.cpu_out, j.out_host->data, 0, ggml_nbytes(j.out));
            return;
        }
        const int64_t t0 = ggml_time_us();
        {
            std::unique_lock<std::mutex> lock(hybrid_mtx);
            hybrid_cv_done.wait(lock, [&] { return hybrid_completed >= seq || hybrid_stop; });
            if (hybrid_failed) {
                GGML_ABORT("the hybrid executor worker failed");
            }
        }
        hybrid_wait_us += ggml_time_us() - t0;
        hybrid_waits++;

        // the worker output is pageable, copy it to the pinned staging first: an async copy of a pageable
        // source goes through the staging buffer and drains the device
        GGML_ASSERT(ggml_nbytes(j.out) == ggml_nbytes(j.out_host));
        memcpy(j.out_host->data, j.out->data, ggml_nbytes(j.out));
        ggml_backend_tensor_set_async(backend, dst, j.out_host->data, 0, ggml_nbytes(j.out));
    }

    void hybrid_worker() {
        for (;;) {
            int32_t il = -1;
            {
                std::unique_lock<std::mutex> lock(hybrid_mtx);
                hybrid_cv.wait(lock, [&] { return hybrid_stop || hybrid_completed < hybrid_submitted; });
                if (hybrid_stop) {
                    return;
                }
                il = hybrid_slot;
            }
            hybrid_job & j = hybrid_jobs[il];

            const int64_t t0 = ggml_time_us();
            const enum ggml_status ec = ggml_graph_compute(j.graph, &j.cplan);
            hybrid_jobs_us += ggml_time_us() - t0;

            std::unique_lock<std::mutex> lock(hybrid_mtx);
            if (ec != GGML_STATUS_SUCCESS) {
                hybrid_failed = true;
            }
            hybrid_completed = hybrid_submitted;
            lock.unlock();
            hybrid_cv_done.notify_all();
        }
    }

    // the promotion copies write into the banks, the context calls this before the backends are freed
    void wait_promotion() {
        if (rolling && pf.ok() && !devices.empty()) {
            pf.wait(devices[0].backend);
        }
    }

    // the combine add of the hybrid executor runs on the device, the worker result is uploaded before the
    // split that reads it: the transfer queue orders it, so no host weight copy and no device drain
    void pre_split(ggml_backend_t backend, ggml_cgraph * graph) {
        if (!hybrid_exec || cpu_out_of.empty()) {
            return;
        }
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = node->src[s];
                if (src == nullptr) {
                    continue;
                }
                if (src->view_src != nullptr) {
                    src = src->view_src;
                }
                const auto it = cpu_out_of.find(src);
                if (it == cpu_out_of.end()) {
                    continue;
                }
                hybrid_job & j = hybrid_jobs[it->second];
                if (j.seq < 0) {
                    continue;
                }
                hybrid_combine(backend, it->second, j.cpu_out);
            }
        }
    }

    void hybrid_log() const {
        if (hybrid_waits == 0) {
            return;
        }
        LLAMA_LOG_INFO("llama_moe_cache: hybrid executor: %zu jobs (%zu layers skipped, all hits), worker %.3f ms, "
                "combine wait %.3f ms (%.3f ms and %.3f ms per job), staged inp %zu/%zu, weights %zu/%zu\n",
            hybrid_waits, hybrid_skips, hybrid_jobs_us/1000.0, hybrid_wait_us/1000.0,
            hybrid_jobs_us/1000.0/hybrid_waits, hybrid_wait_us/1000.0/hybrid_waits,
            hybrid_inp_staged, hybrid_inp_staged + hybrid_inp_sync,
            hybrid_w_staged, hybrid_w_staged + hybrid_w_sync);
        if (rolling || promote_bytes > 0 || promote_fallbacks > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: rolling residency: %zu waits, wait %.3f ms (%.3f ms per layer), "
                    "promoted %.1f MiB, %zu fallbacks\n",
                promote_waits, promote_wait_us/1000.0,
                promote_waits > 0 ? promote_wait_us/1000.0/promote_waits : 0.0,
                promote_bytes/1024.0/1024.0, promote_fallbacks);
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }

        // a pending batch must have been consumed by the previous layer, otherwise fall back to a full wait
        if (deferred_bank != nullptr) {
            ggml_backend_synchronize(backend);
            deferred_bank = nullptr;
            n_deferred_missed++;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // [LLAMA_MOE_PREFETCH] the graph is reused between evals, so the ubatch size is taken from the
        // lookup of the slot map instead of the token count seen when the graph was built
        const ggml_tensor * lookup = nullptr;
        {
            const int n_nodes = ggml_graph_n_nodes(graph);
            for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                    lookup = node;
                }
            }
        }
        const int64_t n_tokens = lookup != nullptr ? ggml_nelements(lookup->src[1]) / n_expert_used : 0;
        const bool pf = pf_enabled && pf_pos[il] >= 0 && n_tokens >= pf_min_tokens;
        if (pf) {
            if (pf_pos[il] == 0) {
                // the rings overwrite the slots of the LRU pool, rebuild the residency map before a
                // small batch reuses it, the rings of this pass start from empty
                for (group & gg : groups) {
                    gg.slru.invalidate();
                    gg.pf_dirty = true;
                }
                pf_pass_dirty = true;
            }
            if (getenv("LLAMA_MOE_PREFETCH_DEBUG") != nullptr) {
                LLAMA_LOG_INFO("%s: layer %d prefetch path, tokens %lld\n", __func__, il, (long long) n_tokens);
            }
            return copy_prefetch(backend, src, dst, il);
        }
        if (pf_pass_dirty) {
            // the small batch after a prefetch pass rebuilds the residency map from scratch
            for (group & gg : groups) {
                if (gg.pf_dirty) {
                    gg.slru.invalidate();
                    gg.pf_dirty = false;
                }
            }
            pf_pass_dirty = false;
        }

        // a whole layer is resident, its slot map is static: no selected experts to read back
        if (l.full) {
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            return true;
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        bool same_split = false;
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                same_split = true;
                break;
            }
        }
        if (same_split) {
            // the ids are produced in this split, so they cannot be read here: upload the whole layer instead,
            // the ring map covers every expert and needs no ids (slow but correct, bench/THREE-TIER.md F1)
            if (n_split_degrades++ == 0) {
                LLAMA_LOG_WARN("%s: layer %d selects its experts in the lookup's own split, using the whole-layer upload\n",
                        __func__, il);
            }
            if (!copy_prefetch(backend, src, dst, il)) {
                LLAMA_LOG_ERROR("%s: whole-layer upload of layer %d failed, copying the previous slot map\n",
                        __func__, il);
                return false;
            }
            return true;
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        const size_t ids_bytes = ggml_nbytes(sel);
        if (ids_staged_il == il && ids_staged == sel && ids_staged_bytes == ids_bytes) {
            // the split that computed the ids already copied them to pinned host memory
            memcpy(ids.data(), ids_host->data, ids_bytes);
            ids_staged = nullptr;
            ids_staged_il = -1;
            ids_staged_bytes = 0;
        } else {
            int32_t * ids_dst = ids.data();
            if (ids_host != nullptr && ids_host->data != nullptr && ids_bytes <= ggml_nbytes(ids_host)) {
                ids_dst = (int32_t *) ids_host->data;
            }
            ggml_backend_tensor_get_async(backend, sel, ids_dst, 0, ids_bytes);
            ggml_backend_synchronize(backend);
            if (ids_dst != ids.data()) {
                memcpy(ids.data(), ids_dst, ids_bytes);
            }
        }

        trace_ids(il, ids);
        observe_ids(il, ids);

        if (hybrid_mode) {
            if (!resident) {
                build_residency();
            }
            // ids is the flat [n_expert_used, n_tokens] selection, the mask uses the same flat order
            const int64_t ne0   = l.miss_mask->ne[0];
            const int64_t n_ids = (int64_t) ids.size();
            GGML_ASSERT(n_ids > 0 && n_ids % ne0 == 0);
            GGML_ASSERT(n_ids / ne0 <= l.miss_mask->ne[1]);

            // rolling residency: the promotion planned for the previous token must be in the bank before the
            // device map of this token is built, then the planned map is published
            const int32_t * map = g.slru.slot_map[il];
            if (rolling) {
                int32_t * pub = (int32_t *) promote_map[il]->data;
                if (promote_pending[il]) {
                    const int64_t tw = ggml_time_us();
                    this->pf.wait(backend);
                    promote_wait_us += ggml_time_us() - tw;
                    promote_waits++;
                    promote_pending[il] = 0;
                }
                // the host expert tensors use the zero slot for the experts that are not in the bank
                const int32_t * planned = g.slru.slot_map[il];
                for (int32_t e = 0; e < n_expert; ++e) {
                    pub[e] = planned[e] >= 0 ? planned[e] : g.n_slots;
                }
                map = pub;
            }

            int32_t *       mask = (int32_t *) l.miss_mask->data;
            size_t n_hit = 0;
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t e = ids[i];
                GGML_ASSERT(e >= 0 && e < n_expert);
                const bool hit = map[e] < g.n_slots;
                mask[i] = hit ? 0 : 1;
                n_hit += hit;
            }
            stats_hybrid.hits   += n_hit;
            stats_hybrid.misses += (size_t) n_ids - n_hit;

            // one line per token: the average hit rate of the last n_cached_layers lookups
            win_hits += n_hit;
            win_ids  += (size_t) n_ids;
            if (++win_calls >= n_cached_layers && win_ids > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid: hit rate = %.2f%% (%zu of %zu)\n",
                    100.0*win_hits/win_ids, win_hits, win_ids);
                win_hits = win_ids = win_calls = 0;
            }

            // the residency map is copied as is, the misses are computed by the CPU branch from the host experts
            if (hybrid_exec && n_tokens == 1 && hybrid_jobs[il].graph != nullptr) {
                if (n_hit == (size_t) n_ids) {
                    // no miss in this layer, the combine reads zeros and the worker stays idle
                    hybrid_job & j = hybrid_jobs[il];
                    std::memset(j.out->data, 0, ggml_nbytes(j.out));
                    {
                        std::lock_guard<std::mutex> lock(hybrid_mtx);
                        j.seq = 0;
                    }
                    inp_staged = nullptr;
                    w_staged   = nullptr;
                } else {
                    // the worker starts while the copies of this split are still being queued
                    hybrid_submit(il, backend);
                }
            }
            ggml_backend_tensor_set_async(backend, dst, rolling ? promote_map[il]->data : src->data, 0, ggml_nbytes(src));

            // plan the next token and promote the planned experts now, no other token reads these slots: the
            // victims selected by the plan are not used by this token and the previous split is drained
            if (rolling) {
                size_t n_hit_plan = 0;
                if (!g.slru.plan(il, ids.data(), ids.size(), promote_fills, n_hit_plan)) {
                    GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
                }
                if (!promote_fills.empty()) {
                    bool ok = true;
                    for (const auto & f : promote_fills) {
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            const ggml_tensor * w = l.experts[ip];
                            const size_t expert_size = w->nb[2];
                            ok = ok && this->pf.tensor(backend, g.banks[ip], (const uint8_t *) w->data + (size_t) f.expert*expert_size,
                                    (size_t) f.slot*expert_size, expert_size);
                            promote_bytes += expert_size;
                        }
                    }
                    if (!ok) {
                        // the planned experts are not in the bank, drop the rolling plan and rebuild the static
                        // residency so that the next lookup computes the misses on the CPU
                        if (promote_fallbacks++ == 0) {
                            LLAMA_LOG_WARN("%s: promotion of layer %d failed, rolling residency disabled\n", __func__, il);
                        }
                        rolling  = false;
                        resident = false;
                    } else {
                        this->pf.submit(backend);
                        promote_pending[il] = 1;
                    }
                }
            }
            return true;
        }

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        // with LLAMA_MOE_DEFER_UPLOAD the last expert tensor goes to a deferred batch that overlaps the
        // gate/up compute of this layer and is waited only by the split that consumes it
        static const char * env_defer = getenv("LLAMA_MOE_DEFER_UPLOAD");
        static const bool defer_upload = env_defer != nullptr && atoi(env_defer) != 0;
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const bool deferred = defer_upload && l.experts.size() > 1 && ip + 1 == l.experts.size();
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                if (deferred) {
                    ggml_backend_set_tensor_async_deferred(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                    n_deferred_bytes += n*expert_size;
                } else {
                    ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                }
                bytes += n*expert_size;
                i += n;
            }
            if (deferred && !fills.empty()) {
                deferred_bank    = bank;
                deferred_backend = backend;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        if (n_deferred_bytes > 0 || n_deferred_waits > 0 || n_deferred_missed > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: deferred upload: bytes = %.2f MiB, waits = %zu, missed = %zu\n",
                n_deferred_bytes/1024.0/1024.0, n_deferred_waits, n_deferred_missed);
        }
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (hybrid_mode) {
            const size_t n = stats_hybrid.hits + stats_hybrid.misses;
            if (n > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded once = %.2f MiB\n",
                    stats_hybrid.hits, stats_hybrid.misses, 100.0*stats_hybrid.hits/n, stats_hybrid.bytes/1024.0/1024.0);
            }
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (n_split_degrades > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: split fallbacks (ids produced in the lookup's split): %zu\n",
                    n_split_degrades);
        }
        if (pf_layers > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: prefetch: %zu layers, %.2f MiB uploaded, %zu host waits, %zu fallbacks\n",
                    pf_layers, pf_bytes/1024.0/1024.0, pf_waits, pf_fallbacks);
            if (pf_waits > 0 && getenv("LLAMA_MOE_PREFETCH_TIME") != nullptr) {
                fprintf(stderr, "moe_prefetch_timing: layers %zu, wait %.3f ms/layer, callback %.3f ms/layer\n",
                        pf_layers, pf_wait_us/1000.0/pf_waits, pf_copy_us/1000.0/pf_waits);
            }
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, bool hybrid) :
    pimpl(new impl(model, backends, bufts, size, hybrid)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_experts_static(const ggml_tensor * w) const {
    return pimpl->get_experts_static(w);
}

bool llama_moe_cache::prefetch_batch() const {
    return pimpl->pf_batch;
}

bool llama_moe_cache::hybrid(int32_t il) const {
    return pimpl->hybrid(il);
}

bool llama_moe_cache::hybrid_exec() const {
    return pimpl->hybrid_exec;
}

ggml_tensor * llama_moe_cache::get_cpu_out(int32_t il) const {
    return pimpl->get_cpu_out(il);
}

bool llama_moe_cache::register_hybrid(int32_t il, const ggml_tensor * inp, const ggml_tensor * weights,
                                      const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down) const {
    return pimpl->register_hybrid(il, inp, weights, up, gate, down);
}

ggml_tensor * llama_moe_cache::get_miss_mask(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_miss_mask(il, n_tokens, n_expert_used);
}

void llama_moe_cache::count_usage(const ggml_tensor * w, const ggml_tensor * ids, const std::vector<int32_t> & ids_data) {
    pimpl->count_usage(w, ids, ids_data);
}

void llama_moe_cache::set_batch_tokens(int64_t n_tokens) const {
    pimpl->update_prefetch(n_tokens);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

void llama_moe_cache::register_ids(int32_t il, const ggml_tensor * ids) const {
    pimpl->register_ids(il, ids);
}

void llama_moe_cache::clear_ids() const {
    pimpl->clear_ids();
}

void llama_moe_cache::stage_inputs(ggml_backend_t backend, ggml_cgraph * graph) const {
    pimpl->stage_inputs(backend, graph);
}

void llama_moe_cache::wait_promotion() {
    pimpl->wait_promotion();
}

bool llama_moe_cache::wait_deferred(ggml_backend_t backend, ggml_cgraph * graph) {
    return pimpl->wait_deferred(backend, graph);
}

void llama_moe_cache::pre_split(ggml_backend_t backend, ggml_cgraph * graph) {
    pimpl->pre_split(backend, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

