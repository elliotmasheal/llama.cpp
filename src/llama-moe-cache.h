#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // in hybrid mode the experts that are not resident stay in host memory and are computed on the CPU
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, bool hybrid = false);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // [LLAMA_MOE_PREFETCH] the token count of the graph being built, selects the prefetch path for large ubatches
    void set_batch_tokens(int64_t n_tokens) const;

    // true while the graph being built is on the prefetch path: its copies upload whole layers and need no ids
    bool prefetch_batch() const;

    // true if the experts of layer il that are not in the cache are computed on the CPU from the host weights
    bool hybrid(int32_t il) const;

    // the host I32 mask of layer il for n_tokens tokens, same shape as the selected experts, 1 = not in the cache
    ggml_tensor * get_miss_mask(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // counts the experts selected by a large batch, the static residency is built from these counts
    void count_usage(const ggml_tensor * w, const ggml_tensor * ids, const std::vector<int32_t> & ids_data);

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the bank view of w when the whole layer is resident, nullptr otherwise
    ggml_tensor * get_experts_static(const ggml_tensor * w) const;

    // [LLAMA_MOE_HYBRID_EXEC] the misses of a decode token are computed by a CPU worker outside the scheduler
    bool hybrid_exec() const;

    // the host F32 [n_embd, 1] tensor that the worker of layer il fills, nullptr if the executor is not used
    ggml_tensor * get_cpu_out(int32_t il) const;

    // graph build: connect the CPU branch of layer il to the worker, returns false if the tensors of the layer
    // do not have the shapes the worker graph expects, so the layer keeps the in-graph CPU branch
    bool register_hybrid(int32_t il, const ggml_tensor * inp, const ggml_tensor * weights,
                         const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // graph build: the selected experts of layer il are the given tensor
    void register_ids(int32_t il, const ggml_tensor * ids) const;
    void clear_ids() const;

    // ggml_backend_sched post-split callback: stages the readback of the selected experts, and of the layer
    // input and the router weights of the hybrid layers, in the split that computes them, so the copy callback
    // finds them in host memory
    void stage_inputs(ggml_backend_t backend, ggml_cgraph * graph) const;

    // if a deferred upload batch is pending and this split consumes it, wait for it before the compute
    bool wait_deferred(ggml_backend_t backend, ggml_cgraph * graph);

    // ggml_backend_sched pre-split callback: the combine of the hybrid executor waits for the worker and
    // uploads its result before the split that adds it, no host weight copy and no device drain
    void pre_split(ggml_backend_t backend, ggml_cgraph * graph);

    // [LLAMA_MOE_HYBRID_EXEC] wait for the promotion of the last token, the context calls this before
    // the backends and the cache buffers are freed
    void wait_promotion();

    // free the host experts of the layers that are frozen in the bank, they never read their host copy again
    // the model cannot serve a second context afterwards, the caller checks that
    size_t release_frozen_host(const llama_model & model);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
