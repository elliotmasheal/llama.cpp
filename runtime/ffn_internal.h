#pragma once

#include <stdint.h>
#include <vector>

#include "cpu_moe.h"
#include "expert_cache.h"
#include "ffn.h"
#include "kernels/gemm.h"
#include "kernels/moe_expert.h"
#include "kernels/norm.h"
#include "vk_dev.h"
#include "weights.h"

#ifdef QRT_HAVE_VULKAN

#include "vk_transfer.h"

// Module-private state shared by the production TU and the CPU reference TU.
// Nothing outside ffn/ includes this header.

struct qrt_ffn {
    qrt_vk_dev *    dev;
    qrt_ffn_cfg     cfg;
    uint32_t        layer;
    qrt_gemm *      gemm;
    qrt_weights *   store;
    qrt_vk_staging * stg;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkPipeline            pipe_router;
    VkPipeline            pipe_topk;
    VkPipeline            pipe_gather;
    VkPipeline            pipe_accum;
    VkPipeline            pipe_out;
    VkDescriptorPool      pool[2];
    uint32_t              sets_used[2];
    uint32_t              cur;
    VkCommandBuffer       ext;

    VkBuffer       router_w, sgate_w;
    VkDeviceMemory router_mem, sgate_mem;
    VkBuffer       sgate_gate, sgate_up, sgate_down;
    VkDeviceMemory sg_mem[3];

    VkBuffer       logits, ids, wts, sgate;
    VkDeviceMemory logits_mem, ids_mem, wts_mem, sgate_mem_out;
    VkBuffer       gather_qs, gather_ds, h_qs, h_ds;
    VkDeviceMemory gq_mem, gd_mem, hq_mem, hd_mem;
    VkBuffer       shexp_qs, shexp_ds, g_out, u_out;
    VkDeviceMemory sq_mem, sd_mem, go_mem, uo_mem;
    VkBuffer       shexp, moe, scratch;
    VkDeviceMemory sh_mem, mo_mem, sc_mem;
    VkBuffer       slot, tab_tok, tab_w;
    VkDeviceMemory slot_mem, tt_mem, tw_mem;

    // host copies the reference reads
    std::vector<float>         h_router;
    std::vector<float>         h_sgate_w;
    std::vector<unsigned char> h_gate, h_up, h_down;
    std::vector<uint32_t>      h_ids;
    std::vector<float>         h_wts;

    // grouping the layer pass produced
    std::vector<uint32_t> cnt;      // routed pairs per expert
    std::vector<uint32_t> rows_e;   // padded rows per expert
    std::vector<uint32_t> row0;     // first row of the expert in the gather planes
    std::vector<uint32_t> dev_off;  // fixed byte offset of the expert's block in the slot
    std::vector<uint8_t>  loaded;   // expert bytes already resident in the slot
    std::vector<uint32_t> h_tab_tok;
    std::vector<float>    h_tab_w;

    uint32_t rows;
    uint32_t pairs;
    uint32_t experts_used;
    uint64_t store_bytes;
    uint64_t uploaded_bytes;        // expert bytes actually copied into the slot
    uint64_t slot_bytes;            // slot capacity (external cap or the whole layer)
    uint64_t planes_bytes;
    uint32_t max_rows;
    uint32_t run_tokens;            // route/group/expert rows for this call
    int      owns_slot;             // 0 for a caller-owned shared slot
    int      transient;             // 1 = pack only this call's experts into the slot

    // Decode expert cache (M3-2a). Off unless qrt_ffn_set_cache arms it.
    int               cache_active;
    qrt_cache *       cache;
    VkBuffer          cache_buf;
    uint64_t          cache_slot_bytes;
    qrt_cpu_moe *     cpu;
    int               miss_split;
    qrt_split_rates   rates;
    qrt_split_carry * carry;
    qrt_ffn_legs      legs;
    std::vector<uint8_t>         use_gpu;
    std::vector<uint32_t>        uniq;
    std::vector<qrt_cache_lookup> lookups;
    std::vector<uint8_t>         h2d_mask;
    std::vector<uint32_t>        cpu_pos;    // entries of uniq computed on the CPU
    std::vector<int8_t>          h_x_qs;     // device Q8_1 planes, one row per token
    std::vector<uint32_t>        h_x_ds;
    std::vector<float>           h_x_d;      // unpacked delta and delta*sum
    std::vector<float>           h_x_s;
    std::vector<float>           cpu_out;    // [run_tokens][n_embd], routed CPU sum
    // [cpu experts][run_tokens][n_embd]: one expert serves every row that routed to it, so
    // its down-projection result has to stay per row (M3-2c).
    std::vector<float>           cpu_res;

    uint32_t dispatches;
    qrt_ffn_in in;      // what the layer handed over this chunk
    qrt_ffn_bufs bufs;
};

// The staging window the expert packer fills before one H2D submit: sized once so the
// record path and the cache planner cannot disagree about when a flush is due.
const uint32_t QRT_FFN_STAGING_BYTES = 32u << 20;

// One layer of the decode cache: routed union -> residency probe -> miss split -> slot
// packing, with the CPU leg measured beside the H2D leg. Implemented in ffn_store.cpp.
int qrt_ffn_plan_cache_layer(qrt_ffn * f, char * err, size_t err_size);

#endif  // QRT_HAVE_VULKAN
