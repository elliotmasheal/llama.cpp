#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// tiers of the unified memory ledger, see bench/THREE-TIER.md 0.0.1 step 1 (SSD is reserved for the
// cold experts of the store, F12)
enum common_moe_tier {
    COMMON_MOE_TIER_VRAM = 0,
    COMMON_MOE_TIER_RAM  = 1,
    COMMON_MOE_TIER_SSD  = 2,
};

// one line of the unified memory ledger: a budget term with its placement per tier. The plan solve
// consumes entries and never re-derives them, so this ledger is the single owner of the budget
// arithmetic (bench/DECISIONS.md hard constraint). v1 entries are tier sums; the per-layer placement
// of dense weights stays in the fit tensor split and the per-page placement of KV follows with the
// store (0.0.1 2a/3a)
struct common_moe_ledger_entry {
    const char * name = "";
    int64_t      vram = 0;
    int64_t      ram  = 0;
    int64_t      ssd  = 0;
};

struct common_moe_ledger {
    common_moe_ledger_entry dense;      // dense weights, measured
    common_moe_ledger_entry kv_context; // measured context class: KV workset + recurrent state (RS
                                        // is unsplit until F7, the entry stays so the store gets both lines)
    common_moe_ledger_entry compute;    // scheduler compute buffers, measured
    common_moe_ledger_entry fa_score;   // reserved: long cross-chunk prefill score buffer
    common_moe_ledger_entry kv_gather;  // reserved: paged prefill dense K/V workspace
    common_moe_ledger_entry margin;     // planned free-memory reserve target, subtracted from the room
    common_moe_ledger_entry bank;       // solved: the expert cache
    int64_t free_vram = 0;              // room the solve started from
    int64_t free_ram  = 0;
};

// device memory budget for the MoE cache, see bench/THREE-TIER.md
struct common_moe_plan {
    size_t  cache_bytes = 0;    // recommended cache size, 0 = keep all experts on the host
    int64_t slots       = 0;    // expert slots the cache would hold
    int64_t n_slots_all = 0;    // expert slots of all layers with experts
    int64_t n_full      = 0;    // whole expert layers that fit instead of a cache
    int64_t ring_bytes  = 0;    // bytes of the two whole-layer prefetch rings
    bool    prefetch    = false;
    bool    resident    = false;
    // --kv-page auto: true when the initial contiguous K/V cache does not fit the device budget,
    // so paging keeps the working set there (bench/THREE-TIER.md F11)
    bool    kv_page     = false;
    int64_t kv_init_bytes = 0;  // bytes of the initial K/V allocation at this configuration
    common_moe_ledger ledger;   // the term-by-term budget the plan solved from (0.0.1 1a)
};

// measures the model once and fills the plan, returns false if the model is not a MoE model
// the plan is advisory: this function does not modify the parameters
bool common_moe_plan_get(
                const char * path_model,
      const llama_model_params * mparams,
     const llama_context_params * cparams,
                    const size_t * margins,      // margins of memory to leave per device in bytes
                             bool   kv_page_auto, // fill plan.kv_page from the budget
                          int32_t   fit_plan,     // 1 = tier-scaled margins + per-group rings, 0 = legacy
              common_moe_plan * out,
                ggml_log_level log_level);       // minimum log level to print during the measurement

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
