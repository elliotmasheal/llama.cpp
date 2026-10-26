// FFN half of one layer (M2-6): router, gated shared expert, routed MoE from the repacked
// store, and the final residual add. The record calls append to the layer's pass command
// buffer; the grouping step between the passes is host work and is timed apart.

#include "ffn.h"
#include "vk_transfer.h"
#include "vk_pass.h"
#include "vk_pipeline.h"

#include "cpu_moe.h"
#include "expert_cache.h"
#include "gguf_index.h"
#include "kernels/gemm.h"
#include "kernels/moe_expert.h"
#include "kernels/norm.h"
#include "qrt_util.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <algorithm>
#include <new>
#include <thread>
#include <vector>

#include "ffn_internal.h"
#ifdef QRT_HAVE_VULKAN

namespace {

int read_q8_0(qrt_gguf * g, const char * name, uint32_t n, uint32_t k,
              std::vector<unsigned char> * raw, VkBuffer * buf, VkDeviceMemory * mem,
              qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 8) {
        return fail(err, err_size, "%s is %s, the FFN reads q8_0 here", name,
                    qrt_gguf_type_name(t->type));
    }
    if ((uint32_t) t->ne[0] != k || (uint32_t) t->ne[1] != n) {
        return fail(err, err_size, "%s is %llux%llu, expected %ux%u", name,
                    (unsigned long long) t->ne[0], (unsigned long long) t->ne[1], k, n);
    }
    raw->assign((size_t) t->nbytes, 0);
    if (qrt_gguf_read_tensor(g, t, raw->data(), raw->size(), err, err_size) != 0) {
        return -1;
    }
    const uint64_t bytes = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, n, k);
    std::vector<unsigned char> prepared((size_t) bytes);
    qrt_gemm_q8k_prepare(raw->data(), n, k, prepared.data());
    if (qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, buf, mem,
                     err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_upload(dev, stg, *buf, 0, prepared.data(), bytes, err, err_size);
}

// Push blocks of the four FFN kernels; the widest is 8 words, one layout carries them all.
struct qrt_ffn_pc {
    uint32_t v[8];
};

const uint32_t FFN_BINDINGS = 5;
const uint32_t FFN_SETS_PER_RING = 1024;

} // namespace




namespace {

// One set per dispatch: a command buffer references the set object, not its contents
// (DECISIONS 65.2), and a pass records hundreds of dispatches.
int alloc_set(qrt_ffn * f, VkDescriptorSet * out, char * err, size_t err_size) {
    if (f->sets_used[f->cur] >= FFN_SETS_PER_RING) {
        return fail(err, err_size, "the FFN pass already holds %u dispatches",
                    FFN_SETS_PER_RING);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = f->pool[f->cur];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &f->dsl;
    const VkResult r = vkAllocateDescriptorSets(f->dev->device, &dsai, out);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (ffn)", r);
    }
    ++f->sets_used[f->cur];
    return 0;
}

int record(qrt_ffn * f, VkPipeline pipe, const VkBuffer * bufs, const qrt_ffn_pc * pc,
           uint32_t gx, uint32_t gy, char * err, size_t err_size) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (alloc_set(f, &set, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_write_set(f->dev, set, bufs, FFN_BINDINGS);
    vkCmdBindPipeline(f->ext, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(f->ext, VK_PIPELINE_BIND_POINT_COMPUTE, f->layout, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(f->ext, f->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*pc), pc);
    vkCmdDispatch(f->ext, gx, gy, 1);
    ++f->dispatches;
    return 0;
}

} // namespace

void qrt_ffn_close(qrt_vk_dev * dev, qrt_ffn * f) {
    if (f == NULL) {
        return;
    }
    if (dev != NULL) {
        const struct {
            VkBuffer        buf;
            VkDeviceMemory  mem;
        } bufs[] = {
            { f->router_w, f->router_mem }, { f->sgate_w, f->sgate_mem },
            { f->sgate_gate, f->sg_mem[0] }, { f->sgate_up, f->sg_mem[1] },
            { f->sgate_down, f->sg_mem[2] },
            { f->logits, f->logits_mem }, { f->ids, f->ids_mem }, { f->wts, f->wts_mem },
            { f->sgate, f->sgate_mem_out },
            { f->gather_qs, f->gq_mem }, { f->gather_ds, f->gd_mem },
            { f->h_qs, f->hq_mem }, { f->h_ds, f->hd_mem },
            { f->shexp_qs, f->sq_mem }, { f->shexp_ds, f->sd_mem },
            { f->g_out, f->go_mem }, { f->u_out, f->uo_mem },
            { f->shexp, f->sh_mem }, { f->moe, f->mo_mem }, { f->scratch, f->sc_mem },
            { f->slot, f->slot_mem }, { f->tab_tok, f->tt_mem }, { f->tab_w, f->tw_mem },
        };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
            if (bufs[i].mem != VK_NULL_HANDLE) {
                qrt_vk_free(dev, bufs[i].buf, bufs[i].mem);
            }
        }
        for (uint32_t i = 0; i < 2; ++i) {
            if (f->pool[i] != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(dev->device, f->pool[i], nullptr);
            }
        }
        if (f->pipe_router != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, f->pipe_router, nullptr);
        if (f->pipe_topk != VK_NULL_HANDLE)   vkDestroyPipeline(dev->device, f->pipe_topk, nullptr);
        if (f->pipe_gather != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, f->pipe_gather, nullptr);
        if (f->pipe_accum != VK_NULL_HANDLE)  vkDestroyPipeline(dev->device, f->pipe_accum, nullptr);
        if (f->pipe_out != VK_NULL_HANDLE)    vkDestroyPipeline(dev->device, f->pipe_out, nullptr);
        if (f->layout != VK_NULL_HANDLE)      vkDestroyPipelineLayout(dev->device, f->layout, nullptr);
        if (f->dsl != VK_NULL_HANDLE)         vkDestroyDescriptorSetLayout(dev->device, f->dsl, nullptr);
        qrt_gemm_close(dev, f->gemm);
        qrt_vk_staging_destroy(dev, f->stg);
    }
    if (f->store != NULL) {
        qrt_weights_close(f->store);
    }
    delete f;
}

uint32_t qrt_ffn_pairs(const qrt_ffn * f) { return f != NULL ? f->pairs : 0; }
uint32_t qrt_ffn_rows(const qrt_ffn * f) { return f != NULL ? f->rows : 0; }
int qrt_ffn_set_tokens(qrt_ffn * f, uint32_t tokens, char * err, size_t err_size) {
    if (f == NULL || tokens == 0 || tokens > f->cfg.tokens) {
        return fail(err, err_size, "FFN run of %u rows does not fit the %u opened",
                    tokens, f != NULL ? f->cfg.tokens : 0);
    }
    f->run_tokens = tokens;
    return 0;
}
uint32_t qrt_ffn_experts_used(const qrt_ffn * f) { return f != NULL ? f->experts_used : 0; }
uint64_t qrt_ffn_store_bytes(const qrt_ffn * f) { return f != NULL ? f->store_bytes : 0; }
uint64_t qrt_ffn_uploaded_bytes(const qrt_ffn * f) { return f != NULL ? f->uploaded_bytes : 0; }
uint64_t qrt_ffn_planes_bytes(const qrt_ffn * f) { return f != NULL ? f->planes_bytes : 0; }
uint32_t qrt_ffn_dispatches(const qrt_ffn * f) { return f != NULL ? f->dispatches : 0; }

const qrt_ffn_bufs * qrt_ffn_buffers(const qrt_ffn * f) {
    return f != NULL ? &f->bufs : NULL;
}

int qrt_ffn_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                 const char * spv_dir, uint32_t layer, const qrt_ffn_cfg * cfg,
                 qrt_ffn ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || gguf_path == NULL || store_prefix == NULL || spv_dir == NULL ||
        cfg == NULL) {
        return fail(err, err_size, "the FFN half needs a device, the GGUF, the store, SPIR-V "
                    "and shapes");
    }
    if (cfg->tokens == 0 || (!cfg->routed_only && cfg->tokens % 32u != 0) ||
        (cfg->routed_only && cfg->tokens > 256u)) {
        return fail(err, err_size, "the FFN chunk of %u tokens is invalid (32-aligned, or 1..256 routed-only)",
                    cfg->tokens);
    }
    if (cfg->n_expert == 0 || cfg->n_expert > 256) {
        return fail(err, err_size, "%u experts is outside what the top-k kernel holds",
                    cfg->n_expert);
    }
    if (cfg->top_k == 0 || cfg->top_k > 8) {
        return fail(err, err_size, "top_k=%u is outside the eight the kernel unrolls",
                    cfg->top_k);
    }
    qrt_ffn * f = new (std::nothrow) qrt_ffn();
    if (f == NULL) {
        return fail(err, err_size, "out of memory for the FFN half");
    }
    f->dev   = dev;
    f->cfg   = *cfg;
    f->layer = layer;
    f->run_tokens = cfg->tokens;
    memset(&f->bufs, 0, sizeof(f->bufs));
    const int external_slot = cfg->slot != VK_NULL_HANDLE && cfg->slot_bytes != 0;

    if (qrt_vk_make_set_layout(dev, FFN_BINDINGS, &f->dsl, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }
    if (qrt_vk_make_layout(dev, f->dsl, sizeof(qrt_ffn_pc), &f->layout, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }
    if (qrt_vk_make_pipeline(dev, f->layout, spv_dir, "ffn_router.spv", nullptr, 0,
                             &f->pipe_router, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, f->layout, spv_dir, "ffn_topk.spv", nullptr, 0,
                             &f->pipe_topk, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, f->layout, spv_dir, "ffn_gather.spv", nullptr, 0,
                             &f->pipe_gather, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, f->layout, spv_dir, "ffn_accum.spv", nullptr, 0,
                             &f->pipe_accum, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, f->layout, spv_dir, "ffn_out.spv", nullptr, 0,
                             &f->pipe_out, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }
    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = FFN_BINDINGS * FFN_SETS_PER_RING;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = FFN_SETS_PER_RING;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    for (uint32_t i = 0; i < 2; ++i) {
        if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &f->pool[i]) != VK_SUCCESS) {
            qrt_ffn_close(dev, f);
            return fail(err, err_size, "vkCreateDescriptorPool (ffn) failed");
        }
    }

    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }
    int rc = -1;
    do {
        char name[128];
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
        if (qrt_gguf_read_f32(g, name, &f->h_router, err, err_size) != 0) {
            break;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp_shexp.weight", layer);
        if (qrt_gguf_read_f32(g, name, &f->h_sgate_w, err, err_size) != 0) {
            break;
        }
        if (f->h_router.size() != (size_t) cfg->n_expert * cfg->n_embd ||
            f->h_sgate_w.size() != cfg->n_embd) {
            fail(err, err_size, "router is %zu floats and the shared gate %zu, expected "
                 "%ux%u and %u", f->h_router.size(), f->h_sgate_w.size(), cfg->n_expert,
                 cfg->n_embd, cfg->n_embd);
            break;
        }
        if (qrt_vk_staging_create(dev, QRT_FFN_STAGING_BYTES, &f->stg, err, err_size) != 0) {
            break;
        }
        if (qrt_vk_alloc(dev, f->h_router.size() * 4,
                         QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &f->router_w,
                         &f->router_mem, err, err_size) != 0 ||
            qrt_vk_upload(dev, f->stg, f->router_w, 0, f->h_router.data(),
                          f->h_router.size() * 4, err, err_size) != 0 ||
            qrt_vk_alloc(dev, f->h_sgate_w.size() * 4,
                         QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &f->sgate_w,
                         &f->sgate_mem, err, err_size) != 0 ||
            qrt_vk_upload(dev, f->stg, f->sgate_w, 0, f->h_sgate_w.data(),
                          f->h_sgate_w.size() * 4, err, err_size) != 0) {
            break;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", layer);
        if (read_q8_0(g, name, cfg->ffn, cfg->n_embd, &f->h_gate, &f->sgate_gate,
                      &f->sg_mem[0], dev, f->stg, err, err_size) != 0) {
            break;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", layer);
        if (read_q8_0(g, name, cfg->ffn, cfg->n_embd, &f->h_up, &f->sgate_up,
                      &f->sg_mem[1], dev, f->stg, err, err_size) != 0) {
            break;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", layer);
        if (read_q8_0(g, name, cfg->n_embd, cfg->ffn, &f->h_down, &f->sgate_down,
                      &f->sg_mem[2], dev, f->stg, err, err_size) != 0) {
            break;
        }
        rc = 0;
    } while (0);
    qrt_gguf_close(g);
    if (rc != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }

    if (qrt_weights_open(store_prefix, &f->store, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }
    const uint32_t store_layers = qrt_weights_layers(f->store);
    const uint32_t store_experts = qrt_weights_experts(f->store);
    if (store_layers <= layer || store_experts < cfg->n_expert) {
        qrt_ffn_close(dev, f);
        return fail(err, err_size, "the store holds %u layers x %u experts, the FFN needs "
                    "layer %u and %u experts", store_layers, store_experts, layer, cfg->n_expert);
    }
    if (qrt_gemm_open(dev, spv_dir, &f->gemm, err, err_size) != 0) {
        qrt_ffn_close(dev, f);
        return -1;
    }

    // Every expert of the layer is read once (layer-first prefill, design 4.5): the slot
    // holds the whole layer, and the planes hold its padded rows. The offsets are fixed at
    // open so a later chunk can load only the experts it has not seen yet.
    f->dev_off.assign(cfg->n_expert, 0);
    f->loaded.assign(cfg->n_expert, 0);
    f->uploaded_bytes = 0;
    uint64_t slot_bytes = 0;
    for (uint32_t e = 0; e < cfg->n_expert; ++e) {
        qrt_expert_loc loc;
        if (qrt_weights_locate(f->store, layer, e, &loc, err, err_size) != 0) {
            qrt_ffn_close(dev, f);
            return -1;
        }
        if (loc.bytes > QRT_FFN_STAGING_BYTES || loc.bytes > UINT32_MAX - slot_bytes) {
            qrt_ffn_close(dev, f);
            return fail(err, err_size, "expert %u exceeds staging capacity or the layer slot exceeds 32-bit offsets", e);
        }
        f->dev_off[e] = (uint32_t) slot_bytes;
        slot_bytes += loc.bytes;
    }
    f->store_bytes = slot_bytes;
    f->slot_bytes = external_slot ? cfg->slot_bytes : slot_bytes;
    f->owns_slot = external_slot ? 0 : 1;
    f->transient = external_slot ? 1 : 0;
    if (external_slot && cfg->slot_bytes < slot_bytes) {
        // A shared slot may be smaller than the layer's full store; the transient path
        // only packs the experts a single call routes.
        f->slot_bytes = cfg->slot_bytes;
    }
    const uint32_t max_pairs = cfg->tokens * cfg->top_k;
    f->max_rows = max_pairs + cfg->n_expert * 7u;
    const uint64_t gather_qs_bytes = (uint64_t) f->max_rows * cfg->n_embd;
    const uint64_t gather_ds_bytes = (uint64_t) f->max_rows * (cfg->n_embd / 32u) * 4;
    const uint64_t h_qs_bytes      = (uint64_t) f->max_rows * cfg->ffn;
    const uint64_t h_ds_bytes      = (uint64_t) f->max_rows * (cfg->ffn / 32u) * 4;
    f->planes_bytes = gather_qs_bytes + gather_ds_bytes + h_qs_bytes + h_ds_bytes;

    const struct {
        uint64_t bytes;
        VkBuffer * buf;
        VkDeviceMemory * mem;
        const char * what;
    } slots[] = {
        { (uint64_t) cfg->tokens * cfg->n_expert * 4, &f->logits, &f->logits_mem, "logits" },
        { (uint64_t) max_pairs * 4, &f->ids, &f->ids_mem, "top-k ids" },
        { (uint64_t) max_pairs * 4, &f->wts, &f->wts_mem, "top-k weights" },
        { (uint64_t) cfg->tokens * 4, &f->sgate, &f->sgate_mem_out, "shared gate" },
        { gather_qs_bytes, &f->gather_qs, &f->gq_mem, "gathered qs" },
        { gather_ds_bytes, &f->gather_ds, &f->gd_mem, "gathered ds" },
        { h_qs_bytes, &f->h_qs, &f->hq_mem, "expert hidden qs" },
        { h_ds_bytes, &f->h_ds, &f->hd_mem, "expert hidden ds" },
        { (uint64_t) cfg->tokens * cfg->ffn, &f->shexp_qs, &f->sq_mem, "shared qs" },
        { (uint64_t) cfg->tokens * (cfg->ffn / 32u) * 4, &f->shexp_ds, &f->sd_mem,
          "shared ds" },
        { (uint64_t) cfg->tokens * cfg->ffn * 4, &f->g_out, &f->go_mem, "gate scratch" },
        { (uint64_t) cfg->tokens * cfg->ffn * 4, &f->u_out, &f->uo_mem, "up scratch" },
        { (uint64_t) cfg->tokens * cfg->n_embd * 4, &f->shexp, &f->sh_mem, "shared out" },
        { (uint64_t) cfg->tokens * cfg->n_embd * 4, &f->moe, &f->mo_mem, "routed out" },
        { (uint64_t) cfg->tokens * cfg->n_embd * 4, &f->scratch, &f->sc_mem, "expert out" },
        { external_slot ? 0 : slot_bytes, &f->slot, &f->slot_mem, "layer expert slot" },
        { (uint64_t) f->max_rows * 4, &f->tab_tok, &f->tt_mem, "row table" },
        { (uint64_t) f->max_rows * 4, &f->tab_w, &f->tw_mem, "row weights" },
    };
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        if (slots[i].bytes == 0) {
            continue;
        }
        if (qrt_vk_alloc(dev, slots[i].bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         slots[i].buf, slots[i].mem, err, err_size) != 0) {
            qrt_ffn_close(dev, f);
            return fail(err, err_size, "allocating the %s buffer", slots[i].what);
        }
    }
    if (external_slot) {
        f->slot = cfg->slot;
        f->slot_mem = VK_NULL_HANDLE;
    }
    f->cnt.assign(cfg->n_expert, 0);
    f->rows_e.assign(cfg->n_expert, 0);
    f->row0.assign(cfg->n_expert, 0);
    f->h_tab_tok.assign((size_t) f->max_rows, 0xFFFFFFFFu);
    f->h_tab_w.assign((size_t) f->max_rows, 0.0f);
    f->h_ids.assign((size_t) max_pairs, 0);
    f->h_wts.assign((size_t) max_pairs, 0.0f);
    *out = f;
    return 0;
}

int qrt_ffn_begin(qrt_ffn * f, uint32_t ring, VkCommandBuffer cmd, char * err,
                  size_t err_size) {
    if (f == NULL || cmd == VK_NULL_HANDLE || ring > 1) {
        return fail(err, err_size, "the FFN pass needs the module, a ring and a buffer");
    }
    const VkResult r = vkResetDescriptorPool(f->dev->device, f->pool[ring], 0);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetDescriptorPool (ffn)", r);
    }
    f->sets_used[ring] = 0;
    f->cur = ring;
    f->ext = cmd;
    return 0;
}

int qrt_ffn_route(qrt_ffn * f, const qrt_ffn_in * in, char * err, size_t err_size) {
    if (f == NULL || in == NULL || f->ext == VK_NULL_HANDLE) {
        return fail(err, err_size, "the router needs the FFN module attached to a pass");
    }
    const double t0 = now_ms();
    const uint32_t T = f->run_tokens;
    const uint32_t E = f->cfg.n_embd;
    const uint32_t NE = f->cfg.n_expert;
    f->in = *in;
    f->dispatches = 0;

    // logits = router @ xpost, over the f32 weight matrix
    qrt_ffn_pc pc;
    memset(&pc, 0, sizeof(pc));
    pc.v[0] = NE;
    pc.v[1] = E;
    pc.v[2] = T;
    pc.v[3] = 0;
    pc.v[4] = 0;
    const VkBuffer rb[FFN_BINDINGS] = { in->xnorm, f->router_w, f->logits, in->xnorm,
                                        in->xnorm };
    if (record(f, f->pipe_router, rb, &pc, (NE + 7u) / 8u, (T + 31u) / 32u, err,
               err_size) != 0) {
        return -1;
    }
    if (!f->cfg.routed_only) {
        // the shared expert's own gate is the same kernel with one row
        const VkBuffer sb[FFN_BINDINGS] = { in->xnorm, f->sgate_w, f->sgate, in->xnorm,
                                            in->xnorm };
        qrt_ffn_pc spc;
        memset(&spc, 0, sizeof(spc));
        spc.v[0] = 1;
        spc.v[1] = E;
        spc.v[2] = T;
        if (record(f, f->pipe_router, sb, &spc, 1, (T + 31u) / 32u, err, err_size) != 0) {
            return -1;
        }
    }
    qrt_vk_cmd_record_barrier(f->ext);

    qrt_ffn_pc tpc;
    memset(&tpc, 0, sizeof(tpc));
    tpc.v[0] = NE;
    tpc.v[1] = f->cfg.top_k;
    tpc.v[2] = T;
    memcpy(&tpc.v[3], &f->cfg.scale, sizeof(float));
    const VkBuffer tb[FFN_BINDINGS] = { f->logits, f->ids, f->wts, f->logits, f->logits };
    if (record(f, f->pipe_topk, tb, &tpc, T, 1, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_cmd_record_barrier(f->ext);

    f->bufs.logits  = f->logits;
    f->bufs.ids     = f->ids;
    f->bufs.weights = f->wts;
    f->bufs.sgate   = f->sgate;
    f->bufs.shexp   = f->shexp;
    f->bufs.moe     = f->moe;
    f->bufs.out     = in->out;
    return 0;
}

int qrt_ffn_set_cache(qrt_ffn * f, const qrt_ffn_cache * cfg, char * err, size_t err_size) {
    if (f == NULL || cfg == NULL || cfg->cache == NULL) {
        return fail(err, err_size, "the expert cache needs a cache object");
    }
    const uint32_t slots = qrt_cache_slots(cfg->cache, f->layer);
    if (slots != 0 && cfg->buffer == VK_NULL_HANDLE) {
        return fail(err, err_size, "layer %u has %u cache slots but no buffer", f->layer, slots);
    }
    if (cfg->slot_bytes == 0) {
        return fail(err, err_size, "the expert cache needs a slot size");
    }
    f->cache = cfg->cache;
    f->cache_buf = cfg->buffer;
    f->cache_slot_bytes = cfg->slot_bytes;
    f->cpu = cfg->cpu;
    f->miss_split = cfg->miss_split;
    f->rates = cfg->rates;
    f->carry = cfg->carry;
    f->slot = cfg->buffer;
    f->slot_bytes = (uint64_t) slots * cfg->slot_bytes;
    f->cache_active = 1;
    return 0;
}

void qrt_ffn_clear_cache(qrt_ffn * f) {
    if (f == NULL) {
        return;
    }
    f->cache_active = 0;
    f->cache = NULL;
    f->cache_buf = VK_NULL_HANDLE;
    f->cpu = NULL;
    f->carry = NULL;
    f->cpu_pos.clear();
    f->cpu_out.clear();
    f->cpu_res.clear();
}

qrt_ffn_legs qrt_ffn_last_legs(const qrt_ffn * f) {
    return f != NULL ? f->legs : qrt_ffn_legs{};
}

int qrt_ffn_take_cpu_out(qrt_ffn * f, float * out, uint32_t tokens) {
    if (f == NULL || out == NULL || tokens == 0) {
        return -1;
    }
    if (f->cpu_out.empty()) {
        return 0;
    }
    const size_t n = (size_t) tokens * f->cfg.n_embd;
    if (n > f->cpu_out.size()) {
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] += f->cpu_out[i];
    }
    return 0;
}

int qrt_ffn_group(qrt_ffn * f, char * err, size_t err_size) {
    if (f == NULL) {
        return fail(err, err_size, "no FFN module to group");
    }
    const double t0 = now_ms();
    const uint32_t T = f->run_tokens;
    const uint32_t K = f->cfg.top_k;
    const uint32_t NE = f->cfg.n_expert;
    const size_t pairs = (size_t) T * K;

    // ids and weights come back from the pass; the routing has to be known on the host
    // because the expert GEMM grids are recorded here (design 5.2 route_ids).
    if (qrt_vk_download(f->dev, f->stg, f->ids, 0, f->h_ids.data(), pairs * 4, err,
                        err_size) != 0 ||
        qrt_vk_download(f->dev, f->stg, f->wts, 0, f->h_wts.data(), pairs * 4, err,
                        err_size) != 0) {
        return -1;
    }
    for (size_t i = 0; i < pairs; ++i) {
        if (f->h_ids[i] >= NE) {
            return fail(err, err_size, "the router picked expert %u of %u", f->h_ids[i], NE);
        }
    }
    std::fill(f->cnt.begin(), f->cnt.end(), 0u);
    for (size_t i = 0; i < pairs; ++i) {
        ++f->cnt[f->h_ids[i]];
    }
    if (f->cache_active) {
        // Residency, split and both miss legs. CPU-leg experts leave cnt at zero, so the
        // GPU rows and expert dispatches below cover the hit + H2D experts only.
        if (qrt_ffn_plan_cache_layer(f, err, err_size) != 0) {
            return -1;
        }
    } else {
        f->legs = qrt_ffn_legs{};
        f->cpu_out.clear();
        std::fill(f->use_gpu.begin(), f->use_gpu.end(), 1);
    }
    f->pairs = T * K;
    uint32_t rows = 0;
    f->experts_used = 0;
    for (uint32_t e = 0; e < NE; ++e) {
        f->rows_e[e] = (f->cnt[e] + 7u) / 8u * 8u;
        f->row0[e]   = rows;
        rows += f->rows_e[e];
        if (f->cnt[e] > 0) {
            ++f->experts_used;
        }
    }
    f->rows = rows;
    // token order inside an expert is the order of (token, slot), so two runs of the same
    // input build the same table
    std::vector<uint32_t> fill_off(f->cnt.size(), 0);
    std::fill(f->h_tab_tok.begin(), f->h_tab_tok.begin() + rows, 0xFFFFFFFFu);
    std::fill(f->h_tab_w.begin(), f->h_tab_w.begin() + rows, 0.0f);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = f->h_ids[(size_t) t * K + k];
            if (!f->use_gpu.empty() && !f->use_gpu[e]) {
                continue;  // this pair is being summed by the CPU leg
            }
            const float w = f->h_wts[(size_t) t * K + k];
            const uint32_t r = f->row0[e] + fill_off[e]++;
            f->h_tab_tok[r] = t;
            f->h_tab_w[r] = w;
        }
    }
    if (rows != 0 &&
        (qrt_vk_upload(f->dev, f->stg, f->tab_tok, 0, f->h_tab_tok.data(),
                       (uint64_t) rows * 4, err, err_size) != 0 ||
         qrt_vk_upload(f->dev, f->stg, f->tab_w, 0, f->h_tab_w.data(),
                       (uint64_t) rows * 4, err, err_size) != 0)) {
        return -1;
    }

    if (f->cache_active) {
        // The cache path already uploaded this call's misses and left dev_off at the
        // slot offsets the expert pass reads.
        return 0;
    }
    // The layer's slot holds all experts across the chunk loop. Only experts that have not
    // been uploaded yet are read; a contiguous run of missing experts is one H2D copy.
    unsigned char * chunk = (unsigned char *) qrt_vk_staging_map(f->stg);
    const uint64_t tmp_bytes = qrt_weights_max_expert_bytes(f->store);
    if (tmp_bytes == 0 || tmp_bytes > SIZE_MAX) {
        return fail(err, err_size, "invalid maximum expert size for the upload");
    }
    unsigned char * tmp = (unsigned char *) malloc((size_t) tmp_bytes);
    if (tmp == NULL) {
        return fail(err, err_size, "out of memory for the expert upload");
    }
    uint64_t chunk_fill = 0;
    uint64_t chunk_base = 0;
    uint64_t dev_bytes = 0;
    std::vector<uint32_t> pending;
    if (f->transient) {
        std::fill(f->loaded.begin(), f->loaded.end(), 0);
        f->uploaded_bytes = 0;
    }
    int rc = 0;
    auto flush = [&]() -> int {
        if (chunk_fill == 0) {
            return 0;
        }
        if (f->uploaded_bytes + chunk_fill > f->slot_bytes) {
            fail(err, err_size, "expert upload exceeds the %llu MiB slot",
                 (unsigned long long) (f->slot_bytes / (1024u * 1024u)));
            return -1;
        }
        if (qrt_vk_upload_mapped(f->dev, f->stg, f->slot, chunk_base, chunk_fill, err,
                                 err_size) != 0) {
            return -1;
        }
        for (size_t i = 0; i < pending.size(); ++i) {
            f->loaded[pending[i]] = 1;
        }
        f->uploaded_bytes += chunk_fill;
        chunk_fill = 0;
        pending.clear();
        return 0;
    };
    for (uint32_t e = 0; e < NE && rc == 0; ++e) {
        if (f->cnt[e] == 0 || f->loaded[e] != 0) {
            continue;
        }
        qrt_expert_loc loc;
        if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0) {
            rc = -1;
            break;
        }
        if (loc.bytes > QRT_FFN_STAGING_BYTES || loc.bytes > tmp_bytes) {
            rc = fail(err, err_size, "expert %u exceeds upload capacity", e);
            break;
        }
        if (f->transient) {
            if (dev_bytes + loc.bytes > f->slot_bytes) {
                rc = fail(err, err_size, "expert %u does not fit the %llu MiB shared slot",
                          e, (unsigned long long) (f->slot_bytes / (1024u * 1024u)));
                break;
            }
            f->dev_off[e] = (uint32_t) dev_bytes;
        }
        // Flush when the staging buffer is full or the slot offset is not contiguous.
        if (chunk_fill != 0 &&
            (chunk_fill + loc.bytes > QRT_FFN_STAGING_BYTES ||
             chunk_base + chunk_fill != (uint64_t) f->dev_off[e])) {
            if (flush() != 0) {
                rc = -1;
                break;
            }
        }
        if (chunk_fill == 0) {
            chunk_base = f->dev_off[e];
        }
        if (qrt_weights_read_expert(f->store, f->layer, e, tmp, (size_t) loc.bytes, err,
                                     err_size) != 0) {
            rc = -1;
            break;
        }
        memcpy(chunk + chunk_fill, tmp, (size_t) loc.bytes);
        chunk_fill += loc.bytes;
        pending.push_back(e);
        if (f->transient) {
            dev_bytes += loc.bytes;
        }
    }
    if (rc == 0 && flush() != 0) {
        rc = -1;
    }
    free(tmp);
    if (rc != 0) {
        return -1;
    }
    return 0;
}

int qrt_ffn_experts(qrt_ffn * f, char * err, size_t err_size) {
    if (f == NULL || f->ext == VK_NULL_HANDLE) {
        return fail(err, err_size, "the expert pass needs the FFN module attached");
    }
    const double t0 = now_ms();
    const uint32_t T  = f->run_tokens;
    // The shared expert is Q8_0; 1..8 rows use the small kernel, larger short tails round
    // to 32. The routed experts keep the exact token count and the 8-row tile.
    const uint32_t sh_m = T <= 8u ? T : ((T + 31u) & ~31u);
    const uint32_t E  = f->cfg.n_embd;
    const uint32_t FF = f->cfg.ffn;
    const uint32_t NE = f->cfg.n_expert;

    // the routed sum starts from zero: one fill, then the per-expert accumulates add
    vkCmdFillBuffer(f->ext, f->moe, 0, (uint64_t) T * E * 4, 0u);
    qrt_vk_cmd_record_barrier_transfer(f->ext);

    qrt_ffn_pc gpc;
    memset(&gpc, 0, sizeof(gpc));
    gpc.v[0] = E;
    gpc.v[1] = f->rows;
    const VkBuffer gb[FFN_BINDINGS] = { f->in.x_qs, f->in.x_ds, f->gather_qs, f->gather_ds,
                                        f->tab_tok };
    if (f->rows != 0 && record(f, f->pipe_gather, gb, &gpc, f->rows, 1, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_cmd_record_barrier(f->ext);

    if (qrt_gemm_attach(f->gemm, f->cur, f->ext, err, err_size) != 0) {
        return -1;
    }
    if (!f->cfg.routed_only) {
        // shared expert: gate, up, swiglu, down over the dense q8_0 matrices
        qrt_gemm_args ga;
        memset(&ga, 0, sizeof(ga));
        ga.w = f->sgate_gate;
        ga.a_qs = f->in.x_qs;
        ga.a_ds = f->in.x_ds;
        ga.out = f->g_out;
        ga.n = FF;
        ga.k = E;
        ga.m = sh_m;
        ga.type = QRT_QUANT_Q8_0;
        ga.token_major = 1;
        if (qrt_gemm_record(f->gemm, &ga, err, err_size) != 0) {
            return -1;
        }
        ga.w = f->sgate_up;
        ga.out = f->u_out;
        if (qrt_gemm_record(f->gemm, &ga, err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0 ||
            qrt_gemm_record_swiglu(f->gemm, f->g_out, f->u_out, f->shexp_qs, f->shexp_ds, sh_m,
                                   FF, 0, 0, err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0) {
            return -1;
        }
        f->dispatches += 3;
        ga.w = f->sgate_down;
        ga.a_qs = f->shexp_qs;
        ga.a_ds = f->shexp_ds;
        ga.out = f->shexp;
        ga.n = E;
        ga.k = FF;
        if (qrt_gemm_record(f->gemm, &ga, err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0) {
            return -1;
        }
        ++f->dispatches;
    }

    // routed experts, one padded row block each
    for (uint32_t e = 0; e < NE; ++e) {
        const uint32_t m = f->rows_e[e];
        if (m == 0) {
            continue;
        }
        qrt_expert_loc loc;
        if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0) {
            return -1;
        }
        const int t_gate = qrt_gemm_type_of(FF, E, loc.gate);
        const int t_up   = qrt_gemm_type_of(FF, E, loc.up);
        const int t_down = qrt_gemm_type_of(E, FF, loc.down);
        if (t_gate < 0 || t_up < 0 || t_down < 0) {
            return fail(err, err_size, "expert %u,%u has an unknown quant type", f->layer, e);
        }
        const uint64_t base = f->dev_off[e];
        const uint32_t off_up   = (uint32_t) (base + loc.gate);
        const uint32_t off_down = (uint32_t) (base + loc.gate + loc.up);

        qrt_gemm_args ea;
        memset(&ea, 0, sizeof(ea));
        ea.w = f->slot;
        ea.a_qs = f->gather_qs;
        ea.a_ds = f->gather_ds;
        ea.abase = f->row0[e] * (E / 32u);
        ea.out = f->g_out;
        ea.n = FF;
        ea.k = E;
        ea.m = m;
        ea.type = (qrt_quant) t_gate;
        ea.wbase = ea.type == QRT_QUANT_Q6K ? (uint32_t) base : (uint32_t) (base / 4u);
        ea.token_major = 1;
        if (qrt_gemm_record(f->gemm, &ea, err, err_size) != 0) {
            return -1;
        }
        ea.type = (qrt_quant) t_up;
        ea.wbase = ea.type == QRT_QUANT_Q6K ? off_up : off_up / 4u;
        ea.out = f->u_out;
        if (qrt_gemm_record(f->gemm, &ea, err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0 ||
            qrt_gemm_record_swiglu(f->gemm, f->g_out, f->u_out, f->h_qs, f->h_ds, m, FF,
                                   f->row0[e] * (FF / 16u),
                                   f->row0[e] * (FF / 32u), err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0) {
            return -1;
        }
        qrt_gemm_args da;
        memset(&da, 0, sizeof(da));
        da.w = f->slot;
        da.a_qs = f->h_qs;
        da.a_ds = f->h_ds;
        da.abase = f->row0[e] * (FF / 32u);
        da.out = f->scratch;
        da.n = E;
        da.k = FF;
        da.m = m;
        da.type = (qrt_quant) t_down;
        da.wbase = da.type == QRT_QUANT_Q6K ? off_down : off_down / 4u;
        da.token_major = 1;
        if (qrt_gemm_record(f->gemm, &da, err, err_size) != 0 ||
            qrt_gemm_record_barrier(f->gemm, err, err_size) != 0) {
            return -1;
        }
        qrt_ffn_pc apc;
        memset(&apc, 0, sizeof(apc));
        apc.v[0] = f->row0[e];
        apc.v[1] = m;
        apc.v[2] = E;
        apc.v[3] = 0;
        const VkBuffer ab[FFN_BINDINGS] = { f->scratch, f->moe, f->tab_tok, f->tab_w,
                                            f->tab_w };
        if (record(f, f->pipe_accum, ab, &apc, (E + 255u) / 256u, m, err, err_size) != 0) {
            return -1;
        }
        f->dispatches += 4;   // gate, up, swiglu, down
        qrt_vk_cmd_record_barrier(f->ext);
    }

    if (!f->cfg.routed_only) {
        qrt_ffn_pc opc;
        memset(&opc, 0, sizeof(opc));
        opc.v[0] = T * E;
        opc.v[1] = E;
        const VkBuffer ob[FFN_BINDINGS] = { f->in.x, f->moe, f->shexp, f->sgate, f->in.out };
        if (record(f, f->pipe_out, ob, &opc, (T * E + 255u) / 256u, 1, err, err_size) != 0) {
            return -1;
        }
    }
    return 0;
}


#endif // QRT_HAVE_VULKAN
