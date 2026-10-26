// FFN half of one layer (M2-6): router, gated shared expert, routed MoE from the repacked
// store, and the final residual add. The record calls append to the layer's pass command
// buffer; the grouping step between the passes is host work and is timed apart.

#include "ffn.h"

#include "cpu_moe.h"
#include "expert_cache.h"
#include "gguf_index.h"
#include "kernels/gemm.h"
#include "kernels/moe_expert.h"
#include "kernels/norm.h"
#include "weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <algorithm>
#include <new>
#include <thread>
#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
    return -1;
}

double now_ms(void) {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

int vk_fail(char * err, size_t err_size, const char * what, VkResult r) {
    return fail(err, err_size, "%s failed: VkResult %d", what, (int) r);
}

int read_f32(qrt_gguf * g, const char * name, std::vector<float> * out, char * err,
             size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 0) {
        return fail(err, err_size, "%s is %s, expected f32", name,
                    qrt_gguf_type_name(t->type));
    }
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) {
        n *= t->ne[i];
    }
    out->assign((size_t) n, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err,
                                err_size);
}

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
const uint32_t FFN_STAGING_BYTES = 32u << 20;

int make_pipeline(qrt_vk_dev * dev, VkPipelineLayout layout, const char * spv_dir,
                  const char * name, VkPipeline * out, char * err, size_t err_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", spv_dir, name);
    FILE * fp = fopen(path, "rb");
    if (fp == NULL) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    fseek(fp, 0, SEEK_END);
    const long bytes = ftell(fp);
    rewind(fp);
    std::vector<unsigned char> code((size_t) (bytes > 0 ? bytes : 0));
    if (bytes <= 0 || fread(code.data(), 1, code.size(), fp) != code.size()) {
        fclose(fp);
        return fail(err, err_size, "cannot read '%s'", path);
    }
    fclose(fp);
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode    = (const uint32_t *) code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev->device, &smci, nullptr, &mod) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateShaderModule (%s) failed", name);
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = layout;
    const VkResult r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci,
                                                 nullptr, out);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateComputePipelines", r);
    }
    return 0;
}

} // namespace

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
    std::vector<float>           cpu_res;    // [cpu experts][n_embd], per-expert down

    uint32_t dispatches;
    qrt_ffn_in in;      // what the layer handed over this chunk
    qrt_ffn_bufs bufs;
};


namespace {

void write_set(qrt_vk_dev * dev, VkDescriptorSet set, const VkBuffer * bufs, uint32_t n) {
    VkDescriptorBufferInfo info[FFN_BINDINGS];
    VkWriteDescriptorSet wr[FFN_BINDINGS];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < FFN_BINDINGS; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dev->device, n, wr, 0, nullptr);
}

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
    write_set(f->dev, set, bufs, FFN_BINDINGS);
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

    VkDescriptorSetLayoutBinding b[FFN_BINDINGS];
    memset(b, 0, sizeof(b));
    for (uint32_t i = 0; i < FFN_BINDINGS; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = FFN_BINDINGS;
    dlci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &f->dsl) != VK_SUCCESS) {
        qrt_ffn_close(dev, f);
        return fail(err, err_size, "vkCreateDescriptorSetLayout (ffn) failed");
    }
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(qrt_ffn_pc);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &f->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev->device, &plci, nullptr, &f->layout) != VK_SUCCESS) {
        qrt_ffn_close(dev, f);
        return fail(err, err_size, "vkCreatePipelineLayout (ffn) failed");
    }
    if (make_pipeline(dev, f->layout, spv_dir, "ffn_router.spv", &f->pipe_router, err,
                      err_size) != 0 ||
        make_pipeline(dev, f->layout, spv_dir, "ffn_topk.spv", &f->pipe_topk, err,
                      err_size) != 0 ||
        make_pipeline(dev, f->layout, spv_dir, "ffn_gather.spv", &f->pipe_gather, err,
                      err_size) != 0 ||
        make_pipeline(dev, f->layout, spv_dir, "ffn_accum.spv", &f->pipe_accum, err,
                      err_size) != 0 ||
        make_pipeline(dev, f->layout, spv_dir, "ffn_out.spv", &f->pipe_out, err,
                      err_size) != 0) {
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
        if (read_f32(g, name, &f->h_router, err, err_size) != 0) {
            break;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp_shexp.weight", layer);
        if (read_f32(g, name, &f->h_sgate_w, err, err_size) != 0) {
            break;
        }
        if (f->h_router.size() != (size_t) cfg->n_expert * cfg->n_embd ||
            f->h_sgate_w.size() != cfg->n_embd) {
            fail(err, err_size, "router is %zu floats and the shared gate %zu, expected "
                 "%ux%u and %u", f->h_router.size(), f->h_sgate_w.size(), cfg->n_expert,
                 cfg->n_embd, cfg->n_embd);
            break;
        }
        if (qrt_vk_staging_create(dev, FFN_STAGING_BYTES, &f->stg, err, err_size) != 0) {
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
        if (loc.bytes > FFN_STAGING_BYTES || loc.bytes > UINT32_MAX - slot_bytes) {
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

namespace {

float half_bits_to_float(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((uint32_t) (113 - e) << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

int expert_type_of(uint64_t bytes, uint32_t rows, uint32_t cols) {
    const uint64_t vals = (uint64_t) rows * cols;
    if (vals % 256u != 0) {
        return -1;
    }
    const uint64_t q4 = vals / 256u * 144u;
    const uint64_t q5 = vals / 256u * 176u;
    const uint64_t q6 = vals / 256u * 210u;
    if (bytes == q4) {
        return QRT_CPU_Q4K;
    }
    if (bytes == q5) {
        return QRT_CPU_Q5K;
    }
    if (bytes == q6) {
        return QRT_CPU_Q6K;
    }
    return -1;
}

// One layer of the decode cache (M3-2a): unique experts -> residency probe -> split ->
// slot assignment -> CPU leg and H2D leg, which is measured while the other runs.
int cache_plan_layer(qrt_ffn * f, char * err, size_t err_size) {
    const uint32_t NE = f->cfg.n_expert;
    const uint32_t T = f->run_tokens;
    const uint32_t E = f->cfg.n_embd;
    const uint32_t K = f->cfg.top_k;
    f->legs = {};
    f->uniq.clear();
    f->use_gpu.assign(NE, 1);
    uint64_t layer_bytes = 0;
    if (qrt_weights_layer_expert_bytes(f->store, f->layer, &layer_bytes) != 0) {
        return fail(err, err_size, "layer %u has no expert size", f->layer);
    }
    std::vector<qrt_split_miss> miss;
    std::vector<uint32_t> miss_uniq;
    for (uint32_t e = 0; e < NE; ++e) {
        if (f->cnt[e] == 0) {
            continue;
        }
        const uint32_t u = (uint32_t) f->uniq.size();
        f->uniq.push_back(e);
        if (!qrt_cache_resident(f->cache, f->layer, e)) {
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0) {
                return -1;
            }
            qrt_split_miss m{};
            m.expert = e;
            m.bytes = loc.bytes;
            m.tokens = f->cnt[e];
            miss.push_back(m);
            miss_uniq.push_back(u);
        }
    }
    f->h2d_mask.assign(f->uniq.size(), 1);
    std::vector<uint8_t> mm(miss.size(), 1);
    qrt_split_stats ss{};
    if (qrt_miss_split_plan(miss.data(), (uint32_t) miss.size(), f->miss_split, &f->rates,
                            f->carry, mm.data(), &ss, err, err_size) != 0) {
        return -1;
    }
    for (size_t i = 0; i < miss.size(); ++i) {
        f->h2d_mask[miss_uniq[i]] = mm[i];
    }
    f->lookups.assign(f->uniq.size(), qrt_cache_lookup{});
    if (qrt_cache_acquire_mask(f->cache, f->layer, f->uniq.data(),
                               (uint32_t) f->uniq.size(), f->h2d_mask.data(),
                               f->lookups.data(), err, err_size) != 0) {
        f->legs.misses = (uint32_t) miss.size();
        return -1;
    }
    f->cpu_pos.clear();
    for (uint32_t i = 0; i < f->uniq.size(); ++i) {
        const uint32_t e = f->uniq[i];
        const qrt_cache_lookup & r = f->lookups[i];
        if (r.hit) {
            f->dev_off[e] = (uint32_t) ((uint64_t) r.slot * f->cache_slot_bytes);
            ++f->legs.hits;
        } else if (!r.failed && r.slot >= 0) {
            f->dev_off[e] = (uint32_t) ((uint64_t) r.slot * f->cache_slot_bytes);
            ++f->legs.h2d;
        } else {
            f->use_gpu[e] = 0;
            f->cnt[e] = 0;
            f->cpu_pos.push_back(i);
            ++f->legs.cpu;
        }
    }
    f->legs.misses = f->legs.h2d + f->legs.cpu;
    if (f->uniq.empty()) {
        return 0;
    }

    // The CPU leg reads the same Q8_1 planes the device produced for the layer row.
    // ds packs the f16 delta in the low half and sum(q)+32768 in the high half.
    const uint32_t nb = E / 32u;
    f->cpu_out.clear();
    f->cpu_res.clear();
    if (!f->cpu_pos.empty()) {
        f->cpu_out.assign((size_t) T * E, 0.0f);
        f->cpu_res.assign((size_t) f->cpu_pos.size() * E, 0.0f);
    }
    std::vector<std::thread> workers;
    std::vector<double> cpu_worker_ms;
    std::atomic<int> cpu_failed{0};
    if (!f->cpu_pos.empty()) {
        f->h_x_qs.resize((size_t) T * E);
        f->h_x_ds.resize((size_t) T * nb);
        f->h_x_d.resize((size_t) T * nb);
        f->h_x_s.resize((size_t) T * nb);
        if (qrt_vk_download(f->dev, f->stg, f->in.x_qs, 0, f->h_x_qs.data(),
                            (uint64_t) T * E, err, err_size) != 0 ||
            qrt_vk_download(f->dev, f->stg, f->in.x_ds, 0, f->h_x_ds.data(),
                            (uint64_t) T * nb * 4u, err, err_size) != 0) {
            return -1;
        }
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t word = f->h_x_ds[(size_t) t * nb + b];
                const float d = half_bits_to_float(word & 0xffffu);
                const int32_t sum = (int32_t) (word >> 16) - 32768;
                f->h_x_d[(size_t) t * nb + b] = d;
                f->h_x_s[(size_t) t * nb + b] = d * (float) sum;
            }
        }
        std::atomic<uint32_t> next{0};
        uint32_t nthreads = (uint32_t) f->cpu_pos.size();
        if (nthreads > 8) {
            nthreads = 8;
        }
        cpu_worker_ms.assign(nthreads, 0.0);
        for (uint32_t w = 0; w < nthreads; ++w) {
            workers.emplace_back([&, w]() {
                const double t_thread = now_ms();
                char tls[256] = {0};
                std::vector<unsigned char> buf((size_t) layer_bytes);
                std::vector<float> row((size_t) E);
                for (;;) {
                    const uint32_t j = next.fetch_add(1);
                    if (j >= f->cpu_pos.size()) {
                        break;
                    }
                    const uint32_t e = f->uniq[f->cpu_pos[j]];
                    qrt_expert_loc loc;
                    if (qrt_weights_locate(f->store, f->layer, e, &loc, tls, sizeof(tls)) != 0 ||
                        qrt_weights_read_expert(f->store, f->layer, e, buf.data(),
                                                buf.size(), tls, sizeof(tls)) != 0) {
                        cpu_failed = 1;
                        break;
                    }
                    const int tg = expert_type_of(loc.gate, f->cfg.ffn, E);
                    const int tu = expert_type_of(loc.up, f->cfg.ffn, E);
                    const int td = expert_type_of(loc.down, E, f->cfg.ffn);
                    if (tg < 0 || tu < 0 || td < 0) {
                        cpu_failed = 1;
                        break;
                    }
                    float * out = f->cpu_res.data() + (size_t) j * E;
                    for (uint32_t t = 0; t < T; ++t) {
                        for (uint32_t k = 0; k < K; ++k) {
                            if (f->h_ids[(size_t) t * K + k] != e) {
                                continue;
                            }
                            const float wgt = f->h_wts[(size_t) t * K + k];
                            qrt_cpu_moe_expert_q8(
                                f->cpu, tg, tu, td, buf.data(), buf.data() + loc.gate,
                                buf.data() + loc.gate + loc.up,
                                f->h_x_qs.data() + (size_t) t * E,
                                f->h_x_d.data() + (size_t) t * nb,
                                f->h_x_s.data() + (size_t) t * nb, (int) E,
                                (int) f->cfg.ffn, row.data());
                            for (uint32_t i = 0; i < E; ++i) {
                                out[i] += wgt * row[i];
                            }
                        }
                    }
                }
                cpu_worker_ms[w] = now_ms() - t_thread;
            });
        }
    }

    // H2D leg on this thread, beside the CPU workers. The misses are packed into the
    // staging buffer and copied in one submit per staging load: a copy per expert costs
    // a submit+fence each and measured 2.3 GB/s instead of the design's 9.9.
    if (f->legs.h2d != 0) {
        const double t_h = now_ms();
        unsigned char * stage = (unsigned char *) qrt_vk_staging_map(f->stg);
        if (stage == NULL) {
            for (size_t w = 0; w < workers.size(); ++w) {
                workers[w].join();
            }
            return fail(err, err_size, "staging is not mapped for the expert cache");
        }
        std::vector<qrt_vk_copy_region> regions;
        uint64_t fill = 0;
        auto join_workers = [&]() {
            for (size_t w = 0; w < workers.size(); ++w) {
                workers[w].join();
            }
        };
        int h2d_rc = 0;
        auto flush = [&]() -> int {
            if (regions.empty()) {
                return 0;
            }
            if (qrt_vk_upload_regions(f->dev, f->stg, regions.data(),
                                      (uint32_t) regions.size(), err, err_size) != 0) {
                return -1;
            }
            uint64_t bytes = 0;
            for (size_t r = 0; r < regions.size(); ++r) {
                bytes += regions[r].bytes;
            }
            qrt_cache_note_upload(f->cache, f->layer, bytes);
            regions.clear();
            fill = 0;
            return 0;
        };
        for (uint32_t i = 0; i < f->uniq.size() && h2d_rc == 0; ++i) {
            const qrt_cache_lookup & r = f->lookups[i];
            if (r.hit || r.failed || r.slot < 0) {
                continue;
            }
            const uint32_t e = f->uniq[i];
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0 ||
                loc.bytes > FFN_STAGING_BYTES) {
                h2d_rc = -1;
                break;
            }
            if (fill + loc.bytes > FFN_STAGING_BYTES && flush() != 0) {
                h2d_rc = -1;
                break;
            }
            if (qrt_weights_read_expert(f->store, f->layer, e, stage + fill,
                                        FFN_STAGING_BYTES - fill, err, err_size) != 0) {
                h2d_rc = -1;
                break;
            }
            qrt_vk_copy_region reg;
            reg.src = stage + fill;
            reg.dst = f->cache_buf;
            reg.dst_offset = (uint64_t) r.slot * f->cache_slot_bytes;
            reg.bytes = loc.bytes;
            regions.push_back(reg);
            fill += loc.bytes;
            f->legs.h2d_bytes += loc.bytes;
        }
        if (h2d_rc == 0 && flush() != 0) {
            h2d_rc = -1;
        }
        if (h2d_rc != 0) {
            join_workers();
            return -1;
        }
        f->legs.h2d_ms = now_ms() - t_h;
    }
    if (!workers.empty()) {
        for (size_t w = 0; w < workers.size(); ++w) {
            workers[w].join();
        }
        if (cpu_failed) {
            return fail(err, err_size, "layer %u CPU expert leg failed", f->layer);
        }
        // Busy time of the whole CPU leg: the mean over workers is its wall time while
        // the H2D leg ran beside it.
        double busy = 0.0;
        for (size_t w = 0; w < cpu_worker_ms.size(); ++w) {
            busy += cpu_worker_ms[w];
        }
        f->legs.cpu_ms = cpu_worker_ms.empty() ? 0.0 : busy / (double) cpu_worker_ms.size();
    }

    // Merge the per-expert CPU rows into the per-token routed sum.
    for (uint32_t j = 0; j < f->cpu_pos.size(); ++j) {
        const uint32_t e = f->uniq[f->cpu_pos[j]];
        const float * res = f->cpu_res.data() + (size_t) j * E;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < K; ++k) {
                if (f->h_ids[(size_t) t * K + k] != e) {
                    continue;
                }
                float * dst = f->cpu_out.data() + (size_t) t * E;
                for (uint32_t i = 0; i < E; ++i) {
                    dst[i] += res[i];  // the worker already applied the router weight
                }
            }
        }
    }

    if (f->carry != NULL) {
        f->carry->h2d_ms += f->legs.h2d_ms;
        f->carry->cpu_ms += f->legs.cpu_ms;
    }
    if (f->legs.h2d != 0) {
        qrt_miss_split_update(&f->rates, f->legs.h2d, f->legs.h2d_bytes, f->legs.h2d_ms,
                              0, 0, 0.0);
    }
    return 0;
}

} // namespace

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
        if (cache_plan_layer(f, err, err_size) != 0) {
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
        if (loc.bytes > FFN_STAGING_BYTES || loc.bytes > tmp_bytes) {
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
            (chunk_fill + loc.bytes > FFN_STAGING_BYTES ||
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

// ---------------------------------------------------------------- reference

namespace {

float f16_to_f32_ref(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((uint32_t) (113 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

// Q8_0 x Q8_1 row in double over the canonical 34-byte blocks.
void q8_0_ref_row(const unsigned char * raw_row, uint32_t k, const signed char * a_qs,
                  const uint32_t * a_ds, uint32_t tokens, std::vector<double> * out) {
    out->assign(tokens, 0.0);
    const uint32_t nb = k / 32u;
    for (uint32_t t = 0; t < tokens; ++t) {
        double acc = 0.0;
        for (uint32_t b = 0; b < nb; ++b) {
            const unsigned char * blk = raw_row + (size_t) b * 34u;
            const double dw = (double) f16_to_f32_ref((uint32_t) blk[0] |
                                                      ((uint32_t) blk[1] << 8));
            const double aq = (double) f16_to_f32_ref(a_ds[(size_t) t * nb + b] & 0xFFFFu);
            double sub = 0.0;
            for (uint32_t e = 0; e < 32u; ++e) {
                sub += (double) (int8_t) blk[2 + e] *
                       (double) a_qs[((size_t) t * nb + b) * 32u + e];
            }
            acc += dw * aq * sub;
        }
        (*out)[t] = acc;
    }
}

float silu(float x) { return x / (1.0f + expf(-x)); }

} // namespace

int qrt_ffn_ref_cpu(qrt_ffn * f, const float * xpost, const signed char * xpost_qs,
                    const uint32_t * xpost_ds, uint32_t ref_tokens, float * logits,
                    uint32_t * ids, float * weights, float * sgate, float * shexp,
                    float * moe, char * err, size_t err_size) {
    if (f == NULL || xpost == NULL || ref_tokens == 0 || ref_tokens > f->cfg.tokens) {
        return fail(err, err_size, "the FFN reference needs a chunk and its post norm row");
    }
    const uint32_t E  = f->cfg.n_embd;
    const uint32_t FF = f->cfg.ffn;
    const uint32_t NE = f->cfg.n_expert;
    const uint32_t K  = f->cfg.top_k;
    const uint32_t T  = ref_tokens;

    // the same Q8_1 rule the device applies to the post norm row, on the device's own planes
    // when the caller passes them
    std::vector<signed char> qs((size_t) T * E);
    std::vector<uint32_t> ds((size_t) T * (E / 32u));
    if (xpost_qs != NULL && xpost_ds != NULL) {
        memcpy(qs.data(), xpost_qs, qs.size());
        memcpy(ds.data(), xpost_ds, ds.size() * sizeof(uint32_t));
    } else {
        qrt_norm_quant_cpu(T, E, xpost, nullptr, qs.data(), ds.data());
    }

    std::vector<float> probs((size_t) T * NE);
    std::vector<float> picked((size_t) K);
    for (uint32_t t = 0; t < T; ++t) {
        const float * row = xpost + (size_t) t * E;
        double mx = -1e300;
        for (uint32_t e = 0; e < NE; ++e) {
            const float * w = f->h_router.data() + (size_t) e * E;
            double acc = 0.0;
            for (uint32_t i = 0; i < E; ++i) {
                acc += (double) w[i] * (double) row[i];
            }
            probs[(size_t) t * NE + e] = (float) acc;
            if (acc > mx) {
                mx = acc;
            }
        }
        double sum = 0.0;
        for (uint32_t e = 0; e < NE; ++e) {
            sum += exp((double) probs[(size_t) t * NE + e] - mx);
        }
        // selection: strict greater on the softmax value, so ties keep the lower id
        std::vector<unsigned char> taken(NE, 0);
        float norm = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            float best = -3.0e38f;
            uint32_t bi = 0;
            for (uint32_t e = 0; e < NE; ++e) {
                const float p = (float) (exp((double) probs[(size_t) t * NE + e] - mx) / sum);
                if (taken[e] == 0 && p > best) {
                    best = p;
                    bi = e;
                }
            }
            taken[bi] = 1;
            picked[k] = best;
            if (ids != NULL) {
                ids[(size_t) t * K + k] = bi;
            }
            norm += best;
        }
        const float denom = norm > 6.103515625e-5f ? norm : 6.103515625e-5f;
        if (weights != NULL) {
            for (uint32_t k = 0; k < K; ++k) {
                weights[(size_t) t * K + k] = picked[k] / denom * f->cfg.scale;
            }
        }
        if (logits != NULL) {
            memcpy(logits + (size_t) t * NE, probs.data() + (size_t) t * NE,
                   (size_t) NE * sizeof(float));
        }
    }

    // shared expert: q8_0 gate/up, silu*up, the Q8_1 rule again, q8_0 down
    std::vector<float> gh((size_t) T * FF), uh((size_t) T * FF);
    std::vector<double> row;
    for (uint32_t r = 0; r < FF; ++r) {
        q8_0_ref_row(f->h_gate.data() + (size_t) r * (E / 32u) * 34u, E, qs.data(),
                     ds.data(), T, &row);
        for (uint32_t t = 0; t < T; ++t) {
            gh[(size_t) t * FF + r] = (float) row[t];
        }
        q8_0_ref_row(f->h_up.data() + (size_t) r * (E / 32u) * 34u, E, qs.data(), ds.data(),
                     T, &row);
        for (uint32_t t = 0; t < T; ++t) {
            uh[(size_t) t * FF + r] = (float) row[t];
        }
    }
    std::vector<float> h((size_t) T * FF);
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = silu(gh[i]) * uh[i];
    }
    std::vector<signed char> hqs((size_t) T * FF);
    std::vector<uint32_t> hds((size_t) T * (FF / 32u));
    qrt_norm_quant_cpu(T, FF, h.data(), nullptr, hqs.data(), hds.data());
    std::vector<float> gate_out((size_t) T);
    for (uint32_t t = 0; t < T; ++t) {
        double acc = 0.0;
        for (uint32_t i = 0; i < E; ++i) {
            acc += (double) f->h_sgate_w[i] * (double) xpost[(size_t) t * E + i];
        }
        gate_out[t] = (float) acc;
    }
    if (sgate != NULL) {
        memcpy(sgate, gate_out.data(), (size_t) T * sizeof(float));
    }
    if (shexp != NULL) {
        for (uint32_t r = 0; r < E; ++r) {
            q8_0_ref_row(f->h_down.data() + (size_t) r * (FF / 32u) * 34u, FF, hqs.data(),
                         hds.data(), T, &row);
            for (uint32_t t = 0; t < T; ++t) {
                shexp[(size_t) t * E + r] = (float) row[t];
            }
        }
    }

    // routed experts over the pairs of the first ref_tokens tokens
    if (moe != NULL) {
        memset(moe, 0, (size_t) T * E * sizeof(float));
        std::vector<std::vector<uint32_t>> tok_of(NE), slot_of(NE);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = ids[(size_t) t * K + k];
                tok_of[e].push_back(t);
                slot_of[e].push_back(k);
            }
        }
        unsigned char * blk = (unsigned char *) malloc((size_t) qrt_weights_max_expert_bytes(f->store));
        std::vector<signed char> rqs((size_t) T * E);
        std::vector<uint32_t> rds((size_t) T * (E / 32u));
        std::vector<float> rh((size_t) T * FF);
        std::vector<float> rgh((size_t) T * FF), ruh((size_t) T * FF);
        std::vector<signed char> rhqs((size_t) T * FF);
        std::vector<uint32_t> rhds((size_t) T * (FF / 32u));
        std::vector<float> sums((size_t) T * E);
        int rc = blk == NULL ? -1 : 0;
        for (uint32_t e = 0; e < NE && rc == 0; ++e) {
            const uint32_t n = (uint32_t) tok_of[e].size();
            if (n == 0) {
                continue;
            }
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0 ||
                qrt_weights_read_expert(f->store, f->layer, e, blk, (size_t) loc.bytes, err,
                                        err_size) != 0) {
                rc = -1;
                break;
            }
            const int t_gate = qrt_gemm_type_of(FF, E, loc.gate);
            const int t_up   = qrt_gemm_type_of(FF, E, loc.up);
            const int t_down = qrt_gemm_type_of(E, FF, loc.down);
            if (t_gate < 0 || t_up < 0 || t_down < 0) {
                rc = fail(err, err_size, "reference: expert %u has an unknown type", e);
                break;
            }
            for (uint32_t r = 0; r < n; ++r) {
                memcpy(rqs.data() + (size_t) r * E, qs.data() + (size_t) tok_of[e][r] * E,
                       E);
                memcpy(rds.data() + (size_t) r * (E / 32u),
                       ds.data() + (size_t) tok_of[e][r] * (E / 32u), (E / 32u) * 4);
            }
            const unsigned char * wg = blk;
            const unsigned char * wu = blk + loc.gate;
            const unsigned char * wd = blk + loc.gate + loc.up;
            const uint32_t nb_g = E / 256u;
            const uint32_t nb_d = FF / 256u;
            for (uint32_t r = 0; r < FF; ++r) {
                qrt_moe_ref_row(wg + (size_t) r * nb_g * qrt_gemm_block_bytes((qrt_quant) t_gate),
                                (qrt_quant) t_gate, (const unsigned char *) rqs.data(),
                                rds.data(), E, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    rgh[(size_t) j * FF + r] = (float) row[j];
                }
                qrt_moe_ref_row(wu + (size_t) r * nb_g * qrt_gemm_block_bytes((qrt_quant) t_up),
                                (qrt_quant) t_up, (const unsigned char *) rqs.data(),
                                rds.data(), E, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    ruh[(size_t) j * FF + r] = (float) row[j];
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t r = 0; r < FF; ++r) {
                    rh[(size_t) j * FF + r] =
                        silu(rgh[(size_t) j * FF + r]) * ruh[(size_t) j * FF + r];
                }
            }
            qrt_norm_quant_cpu(n, FF, rh.data(), nullptr, rhqs.data(), rhds.data());
            for (uint32_t r = 0; r < E; ++r) {
                qrt_moe_ref_row(wd + (size_t) r * nb_d * qrt_gemm_block_bytes((qrt_quant) t_down),
                                (qrt_quant) t_down, (const unsigned char *) rhqs.data(),
                                rhds.data(), FF, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t t = tok_of[e][j];
                    const uint32_t k = slot_of[e][j];
                    moe[(size_t) t * E + r] +=
                        weights[(size_t) t * K + k] * (float) row[j];
                }
            }
        }
        free(blk);
        if (rc != 0) {
            return -1;
        }
    }
    return 0;
}

#endif // QRT_HAVE_VULKAN

