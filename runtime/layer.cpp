// One layer's attention half (M2-6, the layer chain): the existing legs wired in the order
// the model runs them, plus the per-stage CPU double reference the check compares against.
// Every stage reads what the previous stage wrote; the host only appears in the check.

#include "layer.h"
#include "vk_transfer.h"
#include "vk_pass.h"

#include "attn_deltanet.h"
#include "attn_full.h"
#include "ffn.h"
#include "gguf_index.h"
#include "kernels/gemm.h"
#include "kernels/norm.h"
#include "qrt_config.h"
#include "qrt_util.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "layer_internal.h"
#ifdef QRT_HAVE_VULKAN

namespace {

// The DeltaNet shapes, from the tensor table only (the same derivation M2-6c uses).
int dn_shapes_from_gguf(qrt_gguf * g, uint32_t layer, qrt_dn_shapes * sh, char * err,
                        size_t err_size) {
    char name[128];
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    snprintf(name, sizeof(name), "blk.%u.ssm_a", layer);
    int rc = qrt_gguf_read_f32(g, name, &ssm_a, err, err_size);
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", layer);
        rc = qrt_gguf_read_f32(g, name, &ssm_dt, err, err_size);
    }
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", layer);
        rc = qrt_gguf_read_f32(g, name, &ssm_norm, err, err_size);
    }
    const qrt_gguf_tensor * qkv = NULL;
    const qrt_gguf_tensor * conv = NULL;
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
        rc = qrt_gguf_require(g, name, &qkv, err, err_size);
    }
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
        rc = qrt_gguf_require(g, name, &conv, err, err_size);
    }
    if (rc != 0) {
        return -1;
    }
    memset(sh, 0, sizeof(*sh));
    sh->head_dim  = (uint32_t) ssm_norm.size();
    sh->n_v_heads = (uint32_t) ssm_a.size();
    sh->d_inner   = sh->n_v_heads * sh->head_dim;
    sh->n_embd    = (uint32_t) qkv->ne[0];
    sh->conv_ch   = (uint32_t) qkv->ne[1];
    sh->conv_taps = (uint32_t) conv->ne[0];
    sh->qk_ch     = (sh->conv_ch - sh->d_inner) / 2;
    sh->n_k_heads = sh->qk_ch / sh->head_dim;
    if (sh->n_k_heads == 0 || sh->qk_ch % sh->head_dim != 0 || sh->head_dim % 32u != 0) {
        return fail(err, err_size, "layer %u does not split into q/k/v", layer);
    }
    return 0;
}

} // namespace



uint32_t qrt_layer_rings(void) { return 1; }

const qrt_layer_info * qrt_layer_shape(const qrt_layer * l) { return l != NULL ? &l->info : NULL; }
double qrt_layer_last_ms(const qrt_layer * l) { return l != NULL ? l->last_ms : 0.0; }
double qrt_layer_pass1_ms(const qrt_layer * l) { return l != NULL ? l->pass1_ms : 0.0; }
double qrt_layer_group_ms(const qrt_layer * l) { return l != NULL ? l->group_ms : 0.0; }
double qrt_layer_pass2_ms(const qrt_layer * l) { return l != NULL ? l->pass2_ms : 0.0; }
double qrt_layer_gpu_ms(const qrt_layer * l) { return l != NULL ? l->gpu_wait_ms : 0.0; }
uint32_t qrt_layer_submits(const qrt_layer * l) { return l != NULL ? l->submits : 0; }
uint32_t qrt_layer_dispatches(const qrt_layer * l) {
    if (l == NULL) {
        return 0;
    }
    return l->dispatches + (l->ffn != NULL ? qrt_ffn_dispatches(l->ffn) : 0);
}
uint32_t qrt_layer_experts(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_experts_used(l->ffn) : 0;
}
uint32_t qrt_layer_rows(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_rows(l->ffn) : 0;
}
uint64_t qrt_layer_store_bytes(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_store_bytes(l->ffn) : 0;
}
uint64_t qrt_layer_uploaded_bytes(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_uploaded_bytes(l->ffn) : 0;
}
uint64_t qrt_layer_plane_bytes(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_planes_bytes(l->ffn) : 0;
}
const qrt_ffn_bufs * qrt_layer_ffn_bufs(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_buffers(l->ffn) : NULL;
}
qrt_ffn * qrt_layer_ffn(qrt_layer * l) { return l != NULL ? l->ffn : NULL; }

int qrt_layer_set_cache(qrt_layer * l, qrt_cache * cache, VkBuffer buf, uint64_t slot_bytes,
                        qrt_cpu_moe * cpu, int miss_split, const qrt_split_rates * rates,
                        qrt_split_carry * carry, char * err, size_t err_size) {
    if (l == NULL || l->ffn == NULL) {
        snprintf(err, err_size, "the layer has no FFN to attach a cache to");
        return -1;
    }
    if (rates == NULL || carry == NULL) {
        snprintf(err, err_size, "the layer cache needs rates and a carry row");
        return -1;
    }
    qrt_ffn_cache fc{};
    fc.cache = cache;
    fc.buffer = buf;
    fc.slot_bytes = slot_bytes;
    fc.cpu = cpu;
    fc.miss_split = miss_split;
    fc.rates = *rates;
    fc.carry = carry;
    return qrt_ffn_set_cache(l->ffn, &fc, err, err_size);
}

qrt_ffn_legs qrt_layer_last_legs(const qrt_layer * l) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_last_legs(l->ffn) : qrt_ffn_legs{};
}

int qrt_layer_take_cpu_out(qrt_layer * l, float * out, uint32_t tokens) {
    return l != NULL && l->ffn != NULL ? qrt_ffn_take_cpu_out(l->ffn, out, tokens) : -1;
}
VkBuffer qrt_layer_x(const qrt_layer * l) { return l != NULL ? l->x : VK_NULL_HANDLE; }
VkBuffer qrt_layer_out(const qrt_layer * l) {
    if (l == NULL) {
        return VK_NULL_HANDLE;
    }
    return l->ffn != NULL ? l->xout : l->x;
}
VkBuffer qrt_layer_xnorm(const qrt_layer * l) { return l != NULL ? l->norm_out : VK_NULL_HANDLE; }
VkBuffer qrt_layer_attn_out(const qrt_layer * l) { return l != NULL ? l->attn_out : VK_NULL_HANDLE; }
VkBuffer qrt_layer_post_norm(const qrt_layer * l) { return l != NULL ? l->xpost : VK_NULL_HANDLE; }
VkBuffer qrt_layer_attn_src(const qrt_layer * l) { return l != NULL ? l->attn_src : VK_NULL_HANDLE; }

static int load_proj(qrt_layer * l, qrt_gguf * g, const char * name, uint32_t k, uint32_t which,
                     char * err, size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 8) {
        return fail(err, err_size, "%s is %s, the layer chain reads q8_0", name,
                    qrt_gguf_type_name(t->type));
    }
    if ((uint32_t) t->ne[0] != k) {
        return fail(err, err_size, "%s reduces over %llu, expected %u", name,
                    (unsigned long long) t->ne[0], k);
    }
    qrt_layer_proj * p = &l->proj[which];
    p->n = (uint32_t) t->ne[1];
    p->raw.assign((size_t) t->nbytes, 0);
    if (qrt_gguf_read_tensor(g, t, p->raw.data(), p->raw.size(), err, err_size) != 0) {
        return -1;
    }
    const uint64_t bytes = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, p->n, k);
    std::vector<unsigned char> prepared((size_t) bytes);
    qrt_gemm_q8k_prepare(p->raw.data(), p->n, k, prepared.data());
    if (qrt_vk_alloc(l->dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &p->buf,
                     &p->mem, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_upload(l->dev, l->stg, p->buf, 0, prepared.data(), bytes, err, err_size);
}

static int add_pipeline_open(qrt_layer * l, const char * spv_dir, char * err, size_t err_size) {
    qrt_vk_dev * dev = l->dev;
    VkDescriptorSetLayoutBinding b[3];
    memset(b, 0, sizeof(b));
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 3;
    dlci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &l->add_dsl) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateDescriptorSetLayout (layer add) failed");
    }
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = 4 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &l->add_dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev->device, &plci, nullptr, &l->add_layout) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreatePipelineLayout (layer add) failed");
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/layer_add.spv", spv_dir);
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    rewind(f);
    std::vector<unsigned char> code((size_t) (bytes > 0 ? bytes : 0));
    if (bytes <= 0 || fread(code.data(), 1, code.size(), f) != code.size()) {
        fclose(f);
        return fail(err, err_size, "cannot read '%s'", path);
    }
    fclose(f);
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode    = (const uint32_t *) code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev->device, &smci, nullptr, &mod) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateShaderModule (layer add) failed");
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = l->add_layout;
    const VkResult r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                                &l->add_pipe);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateComputePipelines (layer add)", r);
    }
    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 3 * 8;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &l->add_pool) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateDescriptorPool (layer add) failed");
    }
    VkCommandPoolCreateInfo cpi;
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev->compute_family;
    if (vkCreateCommandPool(dev->device, &cpi, nullptr, &l->add_cmd_pool) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateCommandPool (layer add) failed");
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = l->add_cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev->device, &cbai, &l->add_cmd) != VK_SUCCESS) {
        return fail(err, err_size, "vkAllocateCommandBuffers (layer add) failed");
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(dev->device, &fci, nullptr, &l->add_fence) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateFence (layer add) failed");
    }
    return 0;
}

static int add_run(qrt_layer * l, uint32_t count, VkBuffer a, VkBuffer b, VkBuffer out,
                   char * err, size_t err_size) {
    const VkBuffer bufs[3] = { a, b, out };
    VkDescriptorBufferInfo info[3];
    VkWriteDescriptorSet wr[3];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    // One set per record, from the pool the last wait left free (DECISIONS 65.2).
    const VkResult pr = vkResetDescriptorPool(l->dev->device, l->add_pool, 0);
    if (pr != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetDescriptorPool (layer add)", pr);
    }
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = l->add_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &l->add_dsl;
    if (vkAllocateDescriptorSets(l->dev->device, &dsai, &set) != VK_SUCCESS) {
        return fail(err, err_size, "vkAllocateDescriptorSets (layer add) failed");
    }
    for (uint32_t i = 0; i < 3; ++i) {
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
    vkUpdateDescriptorSets(l->dev->device, 3, wr, 0, nullptr);
    if (l->pass != NULL) {
        VkCommandBuffer cmd = qrt_vk_cmd_buffer(l->pass, 0);
        const uint32_t pc[4] = { count, 0, 0, 0 };
        ++l->dispatches;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_layout, 0, 1, &set,
                                0, nullptr);
        vkCmdPushConstants(cmd, l->add_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cmd, (count + 127u) / 128u, 1, 1);
        return 0;
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(l->add_cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (layer add)", r);
    }
    const uint32_t pc[4] = { count, 0, 0, 0 };
    vkCmdBindPipeline(l->add_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_pipe);
    vkCmdBindDescriptorSets(l->add_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_layout, 0, 1,
                            &set, 0, nullptr);
    vkCmdPushConstants(l->add_cmd, l->add_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(l->add_cmd, (count + 127u) / 128u, 1, 1);
    r = vkEndCommandBuffer(l->add_cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (layer add)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &l->add_cmd;
    r = vkQueueSubmit(l->dev->compute_queue, 1, &si, l->add_fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (layer add)", r);
    }
    r = vkWaitForFences(l->dev->device, 1, &l->add_fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (layer add)", r);
    }
    return vkResetFences(l->dev->device, 1, &l->add_fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (layer add) failed");
}

static int alloc_f32(qrt_layer * l, uint64_t floats, VkBuffer * buf, VkDeviceMemory * mem,
                     char * err, size_t err_size) {
    return qrt_vk_alloc(l->dev, floats * 4,
                        QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                        QRT_VK_USAGE_TRANSFER_DST, buf, mem, err, err_size);
}

int qrt_layer_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                   uint32_t layer, uint32_t tokens, const char * spv_dir,
                   const qrt_layer_cfg * cfg, qrt_layer ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || gguf_path == NULL || spv_dir == NULL || tokens == 0) {
        return fail(err, err_size, "the layer chain needs a device, GGUF, SPIR-V and tokens");
    }
    if (tokens % 32u != 0) {
        return fail(err, err_size, "tokens=%u is not a multiple of the 32 the GEMM tiles", tokens);
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        return -1;
    }
    qrt_layer * l = (qrt_layer *) calloc(1, sizeof(qrt_layer));
    if (l == NULL) {
        qrt_gguf_close(g);
        return fail(err, err_size, "out of memory for the layer chain");
    }
    l->dev = dev;
    l->layer = layer;
    l->info.tokens = tokens;
    l->run_tokens = tokens;
    if (cfg != NULL) {
        l->cfg = *cfg;
    } else {
        l->cfg.ffn = 0;
        l->cfg.single_submit = 1;
        l->cfg.dn_recur = 1;
    }
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
    l->info.linear = qrt_gguf_find(g, name) != NULL ? 1 : 0;
    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (qrt_gguf_read_f32(g, name, &l->h_norm, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }
    snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", layer);
    if (qrt_gguf_read_f32(g, name, &l->h_post, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }
    l->info.n_embd = (uint32_t) l->h_norm.size();
    if (l->h_post.size() != l->info.n_embd) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return fail(err, err_size, "the two norms are %zu and %zu wide", l->h_norm.size(),
                    l->h_post.size());
    }
    const uint32_t E = l->info.n_embd;
    const uint32_t T = tokens;
    l->eps = 1e-6f;

    if (qrt_gemm_open(dev, spv_dir, &l->gemm, err, err_size) != 0 ||
        qrt_norm_open(dev, spv_dir, &l->nrm, err, err_size) != 0 ||
        qrt_vk_cmd_create(dev, 2, &l->pass, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }

    // Attention module: one of the two, chosen by what the layer carries.
    qrt_dn_shapes dsh;
    qrt_full_shapes fsh;
    if (l->info.linear) {
        if (dn_shapes_from_gguf(g, layer, &dsh, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        if (qrt_dn_open(dev, spv_dir, &dsh, T, &l->dn, err, err_size) != 0 ||
            qrt_dn_weights_load(dev, gguf_path, layer, &dsh, &l->dnw, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        if (qrt_dn_set_recur(l->dn, l->cfg.dn_recur ? 1 : 0) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return fail(err, err_size, "the 32-column DeltaNet kernel is not in '%s'", spv_dir);
        }
        l->info.inter = dsh.d_inner;
        l->dsh = dsh;
        l->l2_eps = 1e-6f / (float) dsh.head_dim;
        const qrt_dn_bufs * db = qrt_dn_buffers(l->dn);
        l->norm_out = db->norm;
        l->attn_src = db->gnorm;
        l->n_proj = 3;
        l->proj_out[0] = db->qkv;
        l->proj_out[1] = db->z;
    } else {
        const qrt_gguf_hparams hp = *qrt_gguf_hparams_of(g);
        if (qrt_full_shapes_from_gguf(&hp, &fsh, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        if (fsh.n_embd != E) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return fail(err, err_size, "attention.n_embd=%u, the norms are %u", fsh.n_embd, E);
        }
        qrt_full_kv_cfg kvcfg;
        memset(&kvcfg, 0, sizeof(kvcfg));
        kvcfg.k_type = l->cfg.kv_k;
        kvcfg.v_type = l->cfg.kv_v;
        // The pass records the append and the attention into one command buffer, which the
        // paged cache cannot do (it uploads between dispatches), so the whole context stays
        // resident here: n_ctx x two heads x one row, 10.6 MiB per plane at 10.4K f16.
        const uint32_t n_ctx = l->cfg.n_ctx > T ? l->cfg.n_ctx : T;
        kvcfg.resident_tokens = n_ctx;
        kvcfg.page_tokens = l->cfg.kv_page_tokens;
        if (qrt_full_open(dev, spv_dir, &fsh, T, n_ctx, &kvcfg, &l->full, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", layer);
        if (qrt_gguf_read_f32(g, name, &l->h_qnw, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", layer);
        if (qrt_gguf_read_f32(g, name, &l->h_knw, err, err_size) != 0 ||
            l->h_qnw.size() != fsh.head_dim || l->h_knw.size() != fsh.head_dim) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return fail(err, err_size, "the q/k norms are not head_dim wide");
        }
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        l->info.inter = fsh.n_head * fsh.head_dim;
        l->fsh = fsh;
        l->attn_src = fb->out;
        l->n_proj = 4;
        l->proj_out[0] = fb->qg;
        l->proj_out[1] = fb->kin;
        l->proj_out[2] = fb->vin;
    }

    // The staging buffer has to exist before the first weight upload; size it from the
    // largest thing that goes through it: a weight matrix, one chunk row, or the state.
    {
        uint64_t stg_bytes = (uint64_t) T * (E > l->info.inter ? E : l->info.inter) * 4;
        if (l->info.linear) {
            const uint64_t st = (uint64_t) l->dsh.n_v_heads * l->dsh.head_dim *
                                l->dsh.head_dim * 4;
            if (st > stg_bytes) {
                stg_bytes = st;
            }
        }
        // The weight matrices go through this staging buffer too; their sizes are in the
        // tensor table, so no data is read for this pass.
        static const char * names_lin[3] = { "attn_qkv.weight", "attn_gate.weight",
                                             "ssm_out.weight" };
        static const char * names_full[4] = { "attn_q.weight", "attn_k.weight",
                                              "attn_v.weight", "attn_output.weight" };
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            snprintf(name, sizeof(name), "blk.%u.%s", layer,
                     l->info.linear ? names_lin[i] : names_full[i]);
            const qrt_gguf_tensor * t = NULL;
            if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
                qrt_layer_close(dev, l);
                qrt_gguf_close(g);
                return -1;
            }
            const uint32_t k = (i + 1 == l->n_proj) ? l->info.inter : E;
            const uint64_t prepared = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0,
                                                            (uint32_t) t->ne[1], k);
            if (prepared > stg_bytes) {
                stg_bytes = prepared;
            }
        }
        if (qrt_vk_staging_create(dev, stg_bytes, &l->stg, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
    }

    // The projections, in the order the run feeds them; the last one eats the attention
    // output and writes attn_out.
    {
        static const char * names_lin[3] = { "attn_qkv.weight", "attn_gate.weight",
                                             "ssm_out.weight" };
        static const char * names_full[4] = { "attn_q.weight", "attn_k.weight",
                                              "attn_v.weight", "attn_output.weight" };
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            snprintf(name, sizeof(name), "blk.%u.%s", layer,
                     l->info.linear ? names_lin[i] : names_full[i]);
            const uint32_t k = (i + 1 == l->n_proj) ? l->info.inter : E;
            if (load_proj(l, g, name, k, i, err, err_size) != 0) {
                qrt_layer_close(dev, l);
                qrt_gguf_close(g);
                return -1;
            }
        }
    }
    qrt_gguf_close(g);
    g = NULL;

    // Every activation buffer this chain touches.
    if (alloc_f32(l, (uint64_t) T * E, &l->x, &l->x_mem, err, err_size) != 0 ||
        alloc_f32(l, (uint64_t) T * E, &l->attn_out, &l->out_mem, err, err_size) != 0 ||
        alloc_f32(l, (uint64_t) T * E, &l->xpost, &l->post_mem, err, err_size) != 0 ||
        alloc_f32(l, (uint64_t) T * E, &l->xout, &l->xout_mem, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    l->proj_out[l->n_proj - 1] = l->attn_out;
    if (!l->info.linear) {
        if (alloc_f32(l, (uint64_t) T * E, &l->xnorm, &l->xnorm_mem, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
        l->norm_out = l->xnorm;
    }
    const uint64_t nblocks = (uint64_t) T * (E / 32u);
    const uint64_t nblocks2 = (uint64_t) T * (l->info.inter / 32u);
    if (qrt_vk_alloc(dev, (uint64_t) T * E, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->qs, &l->qs_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, nblocks * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->ds, &l->ds_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) T * l->info.inter,
                     QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &l->p2_qs, &l->p2qs_mem,
                     err, err_size) != 0 ||
        qrt_vk_alloc(dev, nblocks2 * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->p2_ds, &l->p2ds_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) E * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->norm_w, &l->nw_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) E * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->post_w, &l->pw_mem, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    if (add_pipeline_open(l, spv_dir, err, err_size) != 0 ||
        qrt_vk_upload(dev, l->stg, l->norm_w, 0, l->h_norm.data(), (uint64_t) E * 4, err,
                      err_size) != 0 ||
        qrt_vk_upload(dev, l->stg, l->post_w, 0, l->h_post.data(), (uint64_t) E * 4, err,
                      err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    if (l->full != NULL) {
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        if (qrt_vk_upload(dev, l->stg, fb->q_norm_w, 0, l->h_qnw.data(),
                          (uint64_t) l->h_qnw.size() * 4, err, err_size) != 0 ||
            qrt_vk_upload(dev, l->stg, fb->k_norm_w, 0, l->h_knw.data(),
                          (uint64_t) l->h_knw.size() * 4, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
    }

    // The FFN half: gated shared expert, router and the routed experts from the store.
    if (l->cfg.ffn) {
        if (store_prefix == NULL) {
            qrt_layer_close(dev, l);
            return fail(err, err_size, "the FFN half needs the repacked expert store");
        }
        qrt_gguf * hg = NULL;
        if (qrt_gguf_open(gguf_path, &hg, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
        const qrt_gguf_hparams hp = *qrt_gguf_hparams_of(hg);
        qrt_gguf_close(hg);
        if (hp.expert_count <= 0 || hp.expert_used_count <= 0 || hp.expert_ff <= 0) {
            qrt_layer_close(dev, l);
            return fail(err, err_size, "the FFN half needs expert_count / used / ff, got %d/%d/%d",
                        hp.expert_count, hp.expert_used_count, hp.expert_ff);
        }
        qrt_ffn_cfg fc;
        memset(&fc, 0, sizeof(fc));
        fc.tokens   = T;
        fc.n_embd   = E;
        fc.n_expert = (uint32_t) hp.expert_count;
        fc.top_k    = (uint32_t) hp.expert_used_count;
        fc.ffn      = (uint32_t) hp.expert_ff;
        fc.scale    = 1.0f;
        fc.slot     = l->cfg.ffn_slot;
        fc.slot_bytes = l->cfg.ffn_slot_bytes;
        if (qrt_ffn_open(dev, gguf_path, store_prefix, spv_dir, layer, &fc, &l->ffn, err,
                         err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
        l->info.n_expert = fc.n_expert;
        l->info.top_k    = fc.top_k;
        l->info.ffn      = fc.ffn;
    }
    *out = l;
    return 0;
}

void qrt_layer_close(qrt_vk_dev * dev, qrt_layer * l) {
    if (l == NULL) {
        return;
    }
    if (dev != NULL) {
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            if (l->proj[i].mem != VK_NULL_HANDLE) {
                qrt_vk_free(dev, l->proj[i].buf, l->proj[i].mem);
            }
        }
        const VkBuffer bufs[10] = { l->x, l->xnorm, l->attn_out, l->xpost, l->qs, l->ds,
                                    l->p2_qs, l->p2_ds, l->norm_w, l->xout };
        VkDeviceMemory mems[10] = { l->x_mem, l->xnorm_mem, l->out_mem, l->post_mem,
                                    l->qs_mem, l->ds_mem, l->p2qs_mem, l->p2ds_mem,
                                    l->nw_mem, l->xout_mem };
        for (uint32_t i = 0; i < 10; ++i) {
            if (mems[i] != VK_NULL_HANDLE) {
                qrt_vk_free(dev, bufs[i], mems[i]);
            }
        }
        if (l->pw_mem != VK_NULL_HANDLE)                 qrt_vk_free(dev, l->post_w, l->pw_mem);
        if (l->add_fence != VK_NULL_HANDLE)              vkDestroyFence(dev->device, l->add_fence, nullptr);
        if (l->add_cmd_pool != VK_NULL_HANDLE)           vkDestroyCommandPool(dev->device, l->add_cmd_pool, nullptr);
        if (l->add_pool != VK_NULL_HANDLE)               vkDestroyDescriptorPool(dev->device, l->add_pool, nullptr);
        if (l->add_pipe != VK_NULL_HANDLE)               vkDestroyPipeline(dev->device, l->add_pipe, nullptr);
        if (l->add_layout != VK_NULL_HANDLE)             vkDestroyPipelineLayout(dev->device, l->add_layout, nullptr);
        if (l->add_dsl != VK_NULL_HANDLE)                vkDestroyDescriptorSetLayout(dev->device, l->add_dsl, nullptr);
        qrt_vk_staging_destroy(dev, l->stg);
        qrt_norm_close(dev, l->nrm);
        qrt_gemm_close(dev, l->gemm);
        qrt_vk_cmd_destroy(dev, l->pass);
        qrt_ffn_close(dev, l->ffn);
        qrt_dn_weights_free(dev, l->dnw);
        qrt_dn_close(dev, l->dn);
        qrt_full_close(dev, l->full);
    }
    free(l);
}

int qrt_layer_set_input_rows(qrt_layer * l, const float * x, uint32_t rows, char * err,
                             size_t err_size) {
    if (l == NULL || x == NULL || rows == 0 || rows > l->info.tokens) {
        return fail(err, err_size, "the layer input of %u rows does not fit the %u opened",
                    rows, l != NULL ? l->info.tokens : 0);
    }
    const uint64_t bytes = (uint64_t) rows * l->info.n_embd * 4;
    l->h_x.assign(x, x + (size_t) (bytes / 4));
    return qrt_vk_upload(l->dev, l->stg, l->x, 0, x, bytes, err, err_size);
}

int qrt_layer_set_input(qrt_layer * l, const float * x, char * err, size_t err_size) {
    return qrt_layer_set_input_rows(l, x, l != NULL ? l->info.tokens : 0, err, err_size);
}

// Every stage of the layer reads what the stage before it wrote: in pass mode those are
// dispatches of one command buffer, so each boundary needs its own memory barrier.
static void stage_barrier(qrt_layer * l) {
    if (l->pass != NULL) {
        qrt_vk_cmd_record_barrier(qrt_vk_cmd_buffer(l->pass, 0));
    }
}

// The DeltaNet legs leave the next chunk's state in two places that are not where the next
// chunk reads it: the recurrence writes state_out (state_in stays the chunk's start, the
// kernel binds them apart), and the conv taps of the next chunk are the last conv_taps-1
// projected rows of this one, which only exist in the qkv plane. Two buffer copies move
// them; the shader writes have to reach the transfer stage first.
static void record_dn_carry(qrt_layer * l, VkCommandBuffer cmd) {
    const qrt_dn_bufs * b = qrt_dn_buffers(l->dn);
    const uint32_t taps = l->dsh.conv_taps - 1u;
    const uint64_t st_bytes = (uint64_t) l->dsh.n_v_heads * l->dsh.head_dim * l->dsh.head_dim * 4;
    const uint64_t row_bytes = (uint64_t) l->dsh.conv_ch * 4;
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy state_copy;
    memset(&state_copy, 0, sizeof(state_copy));
    state_copy.size = st_bytes;
    vkCmdCopyBuffer(cmd, b->state_out, b->state_in, 1, &state_copy);
    if (l->run_tokens >= taps) {
        VkBufferCopy conv_copy{};
        conv_copy.srcOffset = (uint64_t) (l->run_tokens - taps) * row_bytes;
        conv_copy.size = (uint64_t) taps * row_bytes;
        vkCmdCopyBuffer(cmd, b->qkv, b->conv_state, 1, &conv_copy);
    } else {
        const uint64_t added = (uint64_t) l->run_tokens * row_bytes;
        const uint64_t kept = (uint64_t) (taps - l->run_tokens) * row_bytes;
        // Unused QKV rows stage the surviving taps without an overlapping buffer copy.
        const VkBufferCopy retain{added, added, kept};
        vkCmdCopyBuffer(cmd, b->conv_state, b->qkv, 1, &retain);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        const VkBufferCopy copies[] = {{added, 0, kept}, {0, kept, added}};
        vkCmdCopyBuffer(cmd, b->qkv, b->conv_state, 2, copies);
    }
    qrt_vk_cmd_record_barrier_transfer(cmd);
}

// The module-by-module path has no pass buffer, so the carry gets its own submit on the
// layer add's command buffer, which is idle between its own runs.
static int dn_carry_submit(qrt_layer * l, char * err, size_t err_size) {
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(l->add_cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (deltanet carry)", r);
    }
    record_dn_carry(l, l->add_cmd);
    r = vkEndCommandBuffer(l->add_cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (deltanet carry)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &l->add_cmd;
    r = vkQueueSubmit(l->dev->compute_queue, 1, &si, l->add_fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (deltanet carry)", r);
    }
    r = vkWaitForFences(l->dev->device, 1, &l->add_fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (deltanet carry)", r);
    }
    return vkResetFences(l->dev->device, 1, &l->add_fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (deltanet carry) failed");
}

// Dense row stages use the Q8K tile: 1..8 rows use the one-token variant, larger short
// tails round up to 32. Attention/DN and the routed experts use the exact run_tokens.
static uint32_t stage_rows(const qrt_layer * l) {
    const uint32_t t = l->run_tokens;
    return t <= 8u ? t : ((t + 31u) & ~31u);
}

static int layer_norm(qrt_layer * l, VkBuffer src, VkBuffer w, VkBuffer out, VkBuffer qs,
                      VkBuffer ds, uint32_t k, int no_norm, char * err, size_t err_size) {
    qrt_norm_args na;
    memset(&na, 0, sizeof(na));
    na.x = src;
    na.w = w;
    na.out = out;
    na.qs = qs;
    na.ds = ds;
    na.k = k;
    na.tokens = stage_rows(l);
    na.eps = l->eps;
    na.quant = 1;
    na.no_norm = no_norm;
    na.no_out = out == VK_NULL_HANDLE ? 1 : 0;
    if (l->pass != NULL) {
        // pass mode: the record goes into the layer's command buffer
        ++l->dispatches;
        return qrt_norm_record(l->nrm, 0, &na, err, err_size);
    }
    return qrt_norm_run(l->nrm, &na, err, err_size);
}

static int layer_gemm(qrt_layer * l, uint32_t which, VkBuffer planes_qs, VkBuffer planes_ds,
                      VkBuffer out, uint32_t k, char * err, size_t err_size) {
    qrt_gemm_args ga;
    memset(&ga, 0, sizeof(ga));
    ga.w = l->proj[which].buf;
    ga.a_qs = planes_qs;
    ga.a_ds = planes_ds;
    ga.out = out;
    ga.n = l->proj[which].n;
    ga.k = k;
    ga.m = stage_rows(l);
    ga.type = QRT_QUANT_Q8_0;
    ga.token_major = 1;
    if (l->pass != NULL) {
        ++l->dispatches;
        return qrt_gemm_record(l->gemm, &ga, err, err_size);
    }
    return qrt_gemm_dispatch(l->gemm, &ga, err, err_size);
}

// The attention half in the order the model runs it, recorded into the layer's pass.
static int layer_attention_half(qrt_layer * l, char * err, size_t err_size) {
    const uint32_t T = l->run_tokens;
    const uint32_t E = l->info.n_embd;
    const uint32_t last = l->n_proj - 1;
    // 1. input norm; DeltaNet's gate reads the f32 row, so norm_out points at its buffer
    if (layer_norm(l, l->x, l->norm_w, l->norm_out, l->qs, l->ds, E, 0, err, err_size) != 0) {
        return -1;
    }
    stage_barrier(l);
    // 2. the projections that eat the normed row
    for (uint32_t i = 0; i < last; ++i) {
        if (layer_gemm(l, i, l->qs, l->ds, l->proj_out[i], E, err, err_size) != 0) {
            return -1;
        }
    }
    stage_barrier(l);
    // 3. the attention path, over the rows that belong to the sequence
    if (getenv("QRT_LAYER_STOP_PROJ") != NULL) {
        return 0;
    }
    const uint32_t pos0 = l->run_pos0;
    const uint32_t rows = l->run_tokens;
    if (l->info.linear) {
        if (qrt_dn_layer_run(l->dn, l->dnw, rows, l->l2_eps, err, err_size) != 0) {
            return -1;
        }
        if (l->pass != NULL) {
            record_dn_carry(l, qrt_vk_cmd_buffer(l->pass, 0));
        } else if (dn_carry_submit(l, err, err_size) != 0) {
            return -1;
        }
    } else {
        if (qrt_full_qk(l->full, pos0, rows, err, err_size) != 0 ||
            qrt_full_attn(l->full, pos0, rows, err, err_size) != 0) {
            return -1;
        }
    }
    stage_barrier(l);
    // 4. quantise the attention output: the output projection eats planes, not a row
    if (layer_norm(l, l->attn_src, l->norm_w, VK_NULL_HANDLE, l->p2_qs, l->p2_ds, l->info.inter,
                   1, err, err_size) != 0) {
        return -1;
    }
    stage_barrier(l);
    // 5. output projection, 6. residual add
    if (layer_gemm(l, last, l->p2_qs, l->p2_ds, l->attn_out, l->info.inter, err, err_size) != 0) {
        return -1;
    }
    stage_barrier(l);
    if (add_run(l, T * E, l->x, l->attn_out, l->x, err, err_size) != 0) {
        return -1;
    }
    stage_barrier(l);
    // 7. post-attention norm, and its Q8_1 planes, which is what the FFN half reads
    if (layer_norm(l, l->x, l->post_w, l->xpost, l->qs, l->ds, E, 0, err, err_size) != 0) {
        return -1;
    }
    stage_barrier(l);
    return 0;
}

// Two merged submits per layer: the attention half and the router in the first, the FFN
// with the layer's experts in the second. The split is structural: the router's top-k has
// to reach the host before the expert GEMM grids can be recorded (DECISIONS 68).
static int layer_run_pass(qrt_layer * l, char * err, size_t err_size) {
    const uint32_t ring = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const double t0 = now_ms();
    l->submits = 0;
    l->dispatches = 0;

    if (qrt_vk_cmd_begin(l->pass, ring, err, err_size) != 0) {
        return -1;
    }
    cmd = qrt_vk_cmd_buffer(l->pass, ring);
    if (qrt_norm_attach(l->nrm, ring, cmd, err, err_size) != 0 ||
        qrt_gemm_attach(l->gemm, ring, cmd, err, err_size) != 0) {
        return -1;
    }
    if (l->info.linear) {
        if (qrt_dn_attach(l->dn, cmd, err, err_size) != 0) {
            return -1;
        }
    } else {
        if (qrt_full_attach(l->full, cmd, err, err_size) != 0) {
            return -1;
        }
    }
    if (layer_attention_half(l, err, err_size) != 0) {
        return -1;
    }
    if (l->ffn != NULL) {
        qrt_ffn_in in;
        memset(&in, 0, sizeof(in));
        in.x = l->x;
        in.xnorm = l->xpost;
        in.x_qs = l->qs;
        in.x_ds = l->ds;
        in.out = l->xout;
        if (qrt_ffn_set_tokens(l->ffn, l->run_tokens, err, err_size) != 0 ||
            qrt_ffn_begin(l->ffn, ring, cmd, err, err_size) != 0 ||
            qrt_ffn_route(l->ffn, &in, err, err_size) != 0) {
            return -1;
        }
    }
    if (qrt_vk_cmd_submit(l->pass, ring, err, err_size) != 0) {
        return -1;
    }
    const double t_sub1 = now_ms();
    if (qrt_vk_cmd_wait(l->pass, ring, err, err_size) != 0) {
        return -1;
    }
    l->gpu_wait_ms += now_ms() - t_sub1;
    ++l->submits;
    l->pass1_ms = now_ms() - t0;

    if (l->ffn != NULL) {
        const double tg = now_ms();
        if (qrt_ffn_group(l->ffn, err, err_size) != 0) {
            return -1;
        }
        l->group_ms = now_ms() - tg;
        const double t2 = now_ms();
        if (qrt_vk_cmd_begin(l->pass, ring, err, err_size) != 0) {
            return -1;
        }
        cmd = qrt_vk_cmd_buffer(l->pass, ring);
        if (qrt_ffn_begin(l->ffn, ring, cmd, err, err_size) != 0 ||
            qrt_ffn_experts(l->ffn, err, err_size) != 0) {
            return -1;
        }
        if (qrt_vk_cmd_submit(l->pass, ring, err, err_size) != 0) {
            return -1;
        }
        const double t_sub2 = now_ms();
        if (qrt_vk_cmd_wait(l->pass, ring, err, err_size) != 0) {
            return -1;
        }
        l->gpu_wait_ms += now_ms() - t_sub2;
        ++l->submits;
        l->pass2_ms = now_ms() - t2;
    }
    qrt_norm_detach(l->nrm);
    qrt_gemm_detach(l->gemm);
    qrt_dn_detach(l->dn);
    qrt_full_detach(l->full);
    l->last_ms = now_ms() - t0;
    return 0;
}

int qrt_layer_run(qrt_layer * l, char * err, size_t err_size) {
    if (l == NULL) {
        return fail(err, err_size, "no layer");
    }
    return qrt_layer_run_chunk(l, 0, l->info.tokens, err, err_size);
}

int qrt_layer_run_chunk(qrt_layer * l, uint32_t pos0, uint32_t tokens, char * err,
                        size_t err_size) {
    if (l == NULL) {
        return fail(err, err_size, "no layer");
    }
    if (tokens == 0 || tokens > l->info.tokens) {
        return fail(err, err_size, "chunk of %u rows does not fit the %u token layer",
                    tokens, l->info.tokens);
    }
    const uint32_t n_ctx = l->cfg.n_ctx > l->info.tokens ? l->cfg.n_ctx : l->info.tokens;
    if (!l->info.linear && (tokens > n_ctx || pos0 > n_ctx - tokens)) {
        return fail(err, err_size, "position %u with %u rows is past the KV cache this layer opened",
                    pos0, tokens);
    }
    l->run_pos0 = pos0;
    l->run_tokens = tokens;
    l->pass1_ms = l->pass2_ms = 0.0;
    l->group_ms = 0.0;
    l->gpu_wait_ms = 0.0;
    if (l->cfg.single_submit) {
        return layer_run_pass(l, err, err_size);
    }
    // The module-by-module path: one submit per leg, kept as the switch's fallback and as
    // the A/B for the merged pass.
    qrt_vk_cmd * pass = l->pass;
    l->pass = NULL;
    const double t0 = now_ms();
    const int rc = layer_attention_half(l, err, err_size);
    l->pass = pass;
    l->submits = 1;
    l->pass1_ms = now_ms() - t0;
    l->last_ms = l->pass1_ms;
    if (rc != 0) {
        return -1;
    }
    if (l->ffn != NULL) {
        return fail(err, err_size, "the FFN half needs the merged pass (--layer-pass on)");
    }
    return 0;
}

// The recurrence state and the conv state of a DeltaNet layer start at zero for every chunk;
// full attention rewrites the positions it reads, so it needs no reset.
int qrt_layer_zero_state(qrt_layer * l, char * err, size_t err_size) {
    if (l == NULL) {
        return fail(err, err_size, "no layer");
    }
    if (!l->info.linear) {
        return 0;
    }
    const qrt_dn_bufs * b = qrt_dn_buffers(l->dn);
    const uint64_t st = (uint64_t) l->dsh.n_v_heads * l->dsh.head_dim * l->dsh.head_dim * 4;
    const uint64_t cv = (uint64_t) (l->dsh.conv_taps - 1) * l->dsh.conv_ch * 4;
    std::vector<unsigned char> zero((size_t) (st > cv ? st : cv), 0);
    if (qrt_vk_upload(l->dev, l->stg, b->state_in, 0, zero.data(), st, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_upload(l->dev, l->stg, b->conv_state, 0, zero.data(), cv, err, err_size);
}

#endif // QRT_HAVE_VULKAN
