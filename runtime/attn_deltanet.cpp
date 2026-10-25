#include "attn_deltanet.h"

#include "gguf_index.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

#include <chrono>

namespace {

const uint32_t DN_BINDINGS = 8;

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
    return -1;
}

int vk_fail(char * err, size_t err_size, const char * what, VkResult r) {
    return fail(err, err_size, "%s failed: VkResult %d", what, (int) r);
}

int read_spv(const char * dir, const char * name, unsigned char ** out, size_t * out_bytes,
             char * err, size_t err_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open %s", path);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return fail(err, err_size, "cannot size %s", path);
    }
    const long bytes = ftell(f);
    rewind(f);
    if (bytes <= 0) {
        fclose(f);
        return fail(err, err_size, "%s is empty", path);
    }
    unsigned char * buf = (unsigned char *) malloc((size_t) bytes);
    if (buf == NULL) {
        fclose(f);
        return fail(err, err_size, "out of memory for %ld bytes of SPIR-V", bytes);
    }
    const size_t got = fread(buf, 1, (size_t) bytes, f);
    fclose(f);
    if (got != (size_t) bytes) {
        free(buf);
        return fail(err, err_size, "short read on %s", path);
    }
    *out = buf;
    *out_bytes = got;
    return 0;
}

VkShaderModule load_module(VkDevice dev, const char * dir, const char * name, char * err,
                           size_t err_size) {
    unsigned char * code = NULL;
    size_t code_bytes = 0;
    if (read_spv(dir, name, &code, &code_bytes, err, err_size) != 0) {
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code_bytes;
    smci.pCode    = (const uint32_t *) code;
    VkShaderModule mod = VK_NULL_HANDLE;
    const VkResult r = vkCreateShaderModule(dev, &smci, nullptr, &mod);
    free(code);
    if (r != VK_SUCCESS) {
        fail(err, err_size, "vkCreateShaderModule(%s) failed: VkResult %d", name, (int) r);
        return VK_NULL_HANDLE;
    }
    return mod;
}

double now_ms(void) {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

} // namespace

struct qrt_deltanet {
    qrt_vk_dev *   dev;
    qrt_dn_shapes  sh;
    uint32_t       max_tokens;
    uint32_t       subgroup;
    qrt_dn_bufs    bufs;
    VkBuffer       mem_owner[DN_BINDINGS];
    VkDeviceMemory mem[DN_BINDINGS];

    // step 2a: conv1d + silu + L2 norm, one pipeline on its own descriptor set
    VkBuffer       prep_qkv;
    VkBuffer       prep_state;
    VkBuffer       prep_norm;   // [tokens][n_embd] f32, the gate input
    VkDeviceMemory prep_mem[3];
    VkPipeline     prep_pipe;
    VkDescriptorSet prep_set;
    VkPipeline     gate_pipe;
    VkDescriptorSet gate_set;

    VkDescriptorSetLayout dsl;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    VkPipelineLayout      layout;
    VkPipeline            pipe;
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
    double                last_ms;
};

namespace {

// Labels in binding order, so the report can name what it allocated.
const char * DN_LABEL[DN_BINDINGS] = { "q", "k", "v", "decay", "beta", "state_in", "y",
                                      "state_out" };

int alloc_buf(qrt_deltanet * dn, uint32_t idx, uint64_t bytes, char * err, size_t err_size) {
    if (qrt_vk_alloc(dn->dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->mem_owner[idx], &dn->mem[idx],
                     err, err_size) != 0) {
        return -1;
    }
    return 0;
}

} // namespace

int qrt_dn_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_dn_shapes * sh,
                uint32_t max_tokens, qrt_deltanet ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL || sh == NULL || max_tokens == 0) {
        return fail(err, err_size, "the DeltaNet recurrence needs a device, SPIR-V and tokens");
    }
    if (sh->head_dim % 32 != 0 || sh->n_v_heads == 0 || sh->qk_ch % sh->n_k_heads != 0) {
        return fail(err, err_size, "shapes head_dim=%u heads=%u do not fit the 32-lane block",
                    sh->head_dim, sh->n_v_heads);
    }
    const uint32_t subgroup = dev->subgroup_size ? dev->subgroup_size : 32;
    if (!dev->has_subgroup_arithmetic) {
        return fail(err, err_size, "this device has no subgroup arithmetic (subgroup_size=%u)",
                    subgroup);
    }
    if (subgroup > 128 || 128 % subgroup != 0 || sh->head_dim % subgroup != 0 ||
        (sh->head_dim % (128 / subgroup)) != 0) {
        return fail(err, err_size, "subgroup size %u does not tile head_dim %u with 128 lanes",
                    subgroup, sh->head_dim);
    }
    qrt_deltanet * dn = (qrt_deltanet *) calloc(1, sizeof(qrt_deltanet));
    if (dn == NULL) {
        return fail(err, err_size, "out of memory for the DeltaNet recurrence");
    }
    dn->dev = dev;
    dn->sh = *sh;
    dn->max_tokens = max_tokens;
    dn->subgroup = subgroup;

    const uint64_t qk_bytes = (uint64_t) max_tokens * sh->qk_ch * 4;
    const uint64_t v_bytes = (uint64_t) max_tokens * sh->d_inner * 4;
    const uint64_t gb_bytes = (uint64_t) max_tokens * sh->n_v_heads * 4;
    const uint64_t st_bytes = (uint64_t) sh->n_v_heads * sh->head_dim * sh->head_dim * 4;
    const uint64_t sizes[DN_BINDINGS] = { qk_bytes, qk_bytes, v_bytes, gb_bytes, gb_bytes,
                                          st_bytes, v_bytes, st_bytes };
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        if (alloc_buf(dn, i, sizes[i], err, err_size) != 0) {
            qrt_dn_close(dev, dn);
            return -1;
        }
    }
    dn->bufs.q = dn->mem_owner[0];
    dn->bufs.k = dn->mem_owner[1];
    dn->bufs.v = dn->mem_owner[2];
    dn->bufs.decay = dn->mem_owner[3];
    dn->bufs.beta = dn->mem_owner[4];
    dn->bufs.state_in = dn->mem_owner[5];
    dn->bufs.y = dn->mem_owner[6];
    dn->bufs.state_out = dn->mem_owner[7];

    VkDescriptorSetLayoutBinding binds[DN_BINDINGS];
    memset(binds, 0, sizeof(binds));
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        binds[i].binding         = i;
        binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = DN_BINDINGS;
    dlci.pBindings    = binds;
    VkResult r = vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &dn->dsl);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateDescriptorSetLayout (deltanet)", r);
    }

    // n_tokens, heads, kq_ch, k_ratio, qscale
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // one layout for every DeltaNet kernel, so the range covers the widest push block
    // (the prep kernel: 7 uint + 1 float)
    pcr.size       = 8 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &dn->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    r = vkCreatePipelineLayout(dev->device, &plci, nullptr, &dn->layout);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreatePipelineLayout (deltanet)", r);
    }

    VkShaderModule mod = load_module(dev->device, spv_dir, "deltanet_recur.spv", err, err_size);
    if (mod == VK_NULL_HANDLE) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = dn->layout;
    // S_V and the real subgroup size; the lane tiling depends on both
    VkSpecializationMapEntry sme[2];
    sme[0].constantID = 0;
    sme[0].offset     = 0;
    sme[0].size       = sizeof(uint32_t);
    sme[1].constantID = 1;
    sme[1].offset     = sizeof(uint32_t);
    sme[1].size       = sizeof(uint32_t);
    uint32_t spec_data[2] = { sh->head_dim, subgroup };
    VkSpecializationInfo sinfo;
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.mapEntryCount = 2;
    sinfo.pMapEntries   = sme;
    sinfo.dataSize      = sizeof(spec_data);
    sinfo.pData         = spec_data;
    cpci.stage.pSpecializationInfo = &sinfo;
    r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &dn->pipe);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateComputePipelines (deltanet)", r);
    }

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = DN_BINDINGS * 3;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 3;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &dn->pool);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (deltanet)", r);
    }
    VkDescriptorSet sets[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSetLayout layouts[3] = { dn->dsl, dn->dsl, dn->dsl };
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = dn->pool;
    dsai.descriptorSetCount = 3;
    dsai.pSetLayouts        = layouts;
    r = vkAllocateDescriptorSets(dev->device, &dsai, sets);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (deltanet)", r);
    }
    dn->set      = sets[0];
    dn->prep_set = sets[1];
    dn->gate_set = sets[2];
    VkDescriptorBufferInfo info[DN_BINDINGS];
    VkWriteDescriptorSet wr[DN_BINDINGS];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        info[i].buffer        = dn->mem_owner[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dev->device, DN_BINDINGS, wr, 0, nullptr);

    // step 2a buffers: the projected qkv of this chunk and the conv state
    const uint64_t qkv_bytes = (uint64_t) max_tokens * sh->conv_ch * 4;
    const uint64_t cst_bytes = (uint64_t) (sh->conv_taps - 1) * sh->conv_ch * 4;
    if (qrt_vk_alloc(dev, qkv_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_qkv, &dn->prep_mem[0], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.qkv = dn->prep_qkv;
    if (qrt_vk_alloc(dev, cst_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_state, &dn->prep_mem[1], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.conv_state = dn->prep_state;
    const uint64_t norm_bytes = (uint64_t) max_tokens * sh->n_embd * 4;
    if (qrt_vk_alloc(dev, norm_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_norm, &dn->prep_mem[2], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.norm = dn->prep_norm;

    VkShaderModule pmod = load_module(dev->device, spv_dir, "deltanet_prep.spv", err, err_size);
    if (pmod == VK_NULL_HANDLE) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    VkComputePipelineCreateInfo pcp;
    memset(&pcp, 0, sizeof(pcp));
    pcp.sType         = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pcp.stage.sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pcp.stage.stage   = VK_SHADER_STAGE_COMPUTE_BIT;
    pcp.stage.module  = pmod;
    pcp.stage.pName   = "main";
    pcp.layout        = dn->layout;
    r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &pcp, nullptr, &dn->prep_pipe);
    vkDestroyShaderModule(dev->device, pmod, nullptr);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateComputePipelines (deltanet prep)", r);
    }

    VkShaderModule gmod = load_module(dev->device, spv_dir, "deltanet_gate.spv", err, err_size);
    if (gmod == VK_NULL_HANDLE) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    pcp.stage.module = gmod;
    r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &pcp, nullptr, &dn->gate_pipe);
    vkDestroyShaderModule(dev->device, gmod, nullptr);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateComputePipelines (deltanet gate)", r);
    }

    VkCommandPoolCreateInfo cpi;
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev->compute_family;
    r = vkCreateCommandPool(dev->device, &cpi, nullptr, &dn->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateCommandPool (deltanet)", r);
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = dn->cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &dn->cmd);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers (deltanet)", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &dn->fence);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateFence (deltanet)", r);
    }
    *out = dn;
    return 0;
}

void qrt_dn_close(qrt_vk_dev * dev, qrt_deltanet * dn) {
    if (dn == NULL) {
        return;
    }
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        if (dn->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, dn->mem_owner[i], dn->mem[i]);
        }
    }
    const VkBuffer prepbuf[3] = { dn->prep_qkv, dn->prep_state, dn->prep_norm };
    for (uint32_t i = 0; i < 3; ++i) {
        if (dn->prep_mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, prepbuf[i], dn->prep_mem[i]);
        }
    }
    if (dn->prep_pipe != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->prep_pipe, nullptr);
    if (dn->gate_pipe != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->gate_pipe, nullptr);
    if (dn->fence != VK_NULL_HANDLE)    vkDestroyFence(dev->device, dn->fence, nullptr);
    if (dn->cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(dev->device, dn->cmd_pool, nullptr);
    if (dn->pool != VK_NULL_HANDLE)     vkDestroyDescriptorPool(dev->device, dn->pool, nullptr);
    if (dn->pipe != VK_NULL_HANDLE)     vkDestroyPipeline(dev->device, dn->pipe, nullptr);
    if (dn->layout != VK_NULL_HANDLE)   vkDestroyPipelineLayout(dev->device, dn->layout, nullptr);
    if (dn->dsl != VK_NULL_HANDLE)      vkDestroyDescriptorSetLayout(dev->device, dn->dsl, nullptr);
    free(dn);
}

const qrt_dn_shapes * qrt_dn_shapes_of(const qrt_deltanet * dn) { return &dn->sh; }
const qrt_dn_bufs *   qrt_dn_buffers(const qrt_deltanet * dn) { return &dn->bufs; }
double                qrt_dn_last_ms(const qrt_deltanet * dn) { return dn->last_ms; }

uint64_t qrt_dn_state_bytes(const qrt_deltanet * dn) {
    return (uint64_t) dn->sh.n_v_heads * dn->sh.head_dim * dn->sh.head_dim * 4;
}

int qrt_dn_recur(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, dn->max_tokens);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(dn->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (deltanet)", r);
    }
    struct {
        uint32_t n_tokens;
        uint32_t heads;
        uint32_t kq_ch;
        uint32_t k_ratio;
        float    qscale;
    } pcs;
    pcs.n_tokens = tokens;
    pcs.heads    = dn->sh.n_v_heads;
    pcs.kq_ch    = dn->sh.qk_ch;
    pcs.k_ratio  = dn->sh.n_v_heads / dn->sh.n_k_heads;
    pcs.qscale   = 1.0f / sqrtf((float) dn->sh.head_dim);
    vkCmdBindPipeline(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->pipe);
    vkCmdBindDescriptorSets(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->set, 0, nullptr);
    vkCmdPushConstants(dn->cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    // one workgroup per (v-head, column group); the group count follows the
    // device's subgroup size, which the shader is specialized on
    const uint32_t cols_per_wg = 128u / dn->subgroup;
    vkCmdDispatch(dn->cmd, dn->sh.n_v_heads, dn->sh.head_dim / cols_per_wg, 1);
    r = vkEndCommandBuffer(dn->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (deltanet)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &dn->cmd;
    const double t0 = now_ms();
    r = vkQueueSubmit(dn->dev->compute_queue, 1, &si, dn->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (deltanet)", r);
    }
    r = vkWaitForFences(dn->dev->device, 1, &dn->fence, VK_TRUE, UINT64_MAX);
    dn->last_ms = now_ms() - t0;
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (deltanet)", r);
    }
    return vkResetFences(dn->dev->device, 1, &dn->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (deltanet) failed");
}

struct qrt_dn_weights {
    VkBuffer       conv_w;    // f32 [taps][conv_ch]
    VkBuffer       alpha_w;   // f32 [n_embd][n_v_heads]
    VkBuffer       beta_w;
    VkBuffer       dt;        // f32 [n_v_heads]
    VkBuffer       a;         // f32 [n_v_heads]
    VkDeviceMemory mem[5];
    uint32_t       count;
    // host copies, for the double precision references only
    std::vector<float> h_conv, h_alpha, h_beta, h_dt, h_a;
};

static int dn_read_f32(qrt_gguf * g, const char * name, std::vector<float> * out, char * err,
                       size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 0) {
        snprintf(err, err_size, "%s is %s, expected f32", name, qrt_gguf_type_name(t->type));
        return -1;
    }
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) {
        n *= t->ne[i];
    }
    out->assign((size_t) n, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err, err_size);
}

int qrt_dn_weights_load(qrt_vk_dev * dev, const char * gguf_path, uint32_t layer,
                        const qrt_dn_shapes * sh, qrt_dn_weights ** out, char * err, size_t err_size) {
    *out = NULL;
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        return -1;
    }
    char name[128];
    std::vector<float> buf;
    snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
    const int rc = dn_read_f32(g, name, &buf, err, err_size);
    qrt_gguf_close(g);
    if (rc != 0) {
        return -1;
    }
    if (buf.size() != (size_t) sh->conv_taps * sh->conv_ch) {
        snprintf(err, err_size, "%s has %zu values, expected %u", name, buf.size(),
                 sh->conv_taps * sh->conv_ch);
        return -1;
    }
    qrt_dn_weights * w = (qrt_dn_weights *) calloc(1, sizeof(qrt_dn_weights));
    if (w == NULL) {
        snprintf(err, err_size, "out of memory for the conv weights");
        return -1;
    }
    const uint64_t bytes = buf.size() * 4;
    if (qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST |
                     QRT_VK_USAGE_TRANSFER_SRC, &w->conv_w, &w->mem[w->count], err,
                     err_size) != 0) {
        free(w);
        return -1;
    }
    ++w->count;
    w->h_conv = buf;
    // the gate weights: f32 matrices and the two per-head vectors
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        qrt_dn_weights_free(dev, w);
        return -1;
    }
    int grc = 0;
    struct { const char * suffix; VkBuffer * dst; std::vector<float> * host; } gate_w[] = {
        { ".ssm_alpha.weight", &w->alpha_w, &w->h_alpha },
        { ".ssm_beta.weight",  &w->beta_w,  &w->h_beta  },
        { ".ssm_dt.bias",      &w->dt,      &w->h_dt    },
        { ".ssm_a",            &w->a,       &w->h_a     },
    };
    for (uint32_t i = 0; i < 4 && grc == 0; ++i) {
        snprintf(name, sizeof(name), "blk.%u%s", layer, gate_w[i].suffix);
        grc = dn_read_f32(g, name, &buf, err, err_size);
        if (grc != 0) {
            break;
        }
        *gate_w[i].host = buf;
        if (qrt_vk_alloc(dev, buf.size() * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         gate_w[i].dst, &w->mem[w->count], err, err_size) != 0) {
            grc = -1;
            break;
        }
        ++w->count;
    }
    qrt_gguf_close(g);
    if (grc != 0) {
        qrt_dn_weights_free(dev, w);
        return -1;
    }
    {
        qrt_vk_staging * gstg = NULL;
        if (qrt_vk_staging_create(dev, w->h_alpha.size() * 4, &gstg, err, err_size) != 0) {
            qrt_dn_weights_free(dev, w);
            return -1;
        }
        const VkBuffer dsts[4] = { w->alpha_w, w->beta_w, w->dt, w->a };
        const std::vector<float> * srcs[4] = { &w->h_alpha, &w->h_beta, &w->h_dt, &w->h_a };
        for (uint32_t i = 0; i < 4 && grc == 0; ++i) {
            grc = qrt_vk_upload(dev, gstg, dsts[i], 0, srcs[i]->data(), srcs[i]->size() * 4,
                                err, err_size);
        }
        qrt_vk_staging_destroy(dev, gstg);
        if (grc != 0) {
            qrt_dn_weights_free(dev, w);
            return -1;
        }
    }
    qrt_vk_staging * stg = NULL;
    if (qrt_vk_staging_create(dev, bytes, &stg, err, err_size) != 0 ||
        qrt_vk_upload(dev, stg, w->conv_w, 0, buf.data(), bytes, err, err_size) != 0) {
        qrt_vk_staging_destroy(dev, stg);
        qrt_vk_free(dev, w->conv_w, w->mem[0]);
        free(w);
        return -1;
    }
    qrt_vk_staging_destroy(dev, stg);
    *out = w;
    return 0;
}

VkBuffer qrt_dn_weights_conv(const qrt_dn_weights * w) {
    return w != NULL ? w->conv_w : VK_NULL_HANDLE;
}

const float * qrt_dn_weights_host(const qrt_dn_weights * w, uint32_t which) {
    if (w == NULL) {
        return NULL;
    }
    switch (which) {
        case 0: return w->h_conv.empty() ? NULL : w->h_conv.data();
        case 1: return w->h_alpha.empty() ? NULL : w->h_alpha.data();
        case 2: return w->h_beta.empty() ? NULL : w->h_beta.data();
        case 3: return w->h_dt.empty() ? NULL : w->h_dt.data();
        case 4: return w->h_a.empty() ? NULL : w->h_a.data();
    }
    return NULL;
}

void qrt_dn_weights_free(qrt_vk_dev * dev, qrt_dn_weights * w) {
    if (w == NULL) {
        return;
    }
    const VkBuffer bufs[5] = { w->conv_w, w->alpha_w, w->beta_w, w->dt, w->a };
    for (uint32_t i = 0; i < w->count && i < 5; ++i) {
        if (w->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, bufs[i], w->mem[i]);
        }
    }
    free(w);
}

int qrt_dn_prep_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    float l2_eps, char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "prep chunk of %u tokens does not fit %u", tokens,
                    dn->max_tokens);
    }
    // bindings follow kernels/deltanet_prep.comp: qkv, conv_w, conv_state, q, k, v
    const VkBuffer bufs[6] = { dn->bufs.qkv, w->conv_w, dn->bufs.conv_state, dn->bufs.q,
                               dn->bufs.k, dn->bufs.v };
    VkDescriptorBufferInfo info[6];
    VkWriteDescriptorSet wr[6];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 6; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->prep_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dn->dev->device, 6, wr, 0, nullptr);

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(dn->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (deltanet prep)", r);
    }
    struct {
        uint32_t n_tokens;
        uint32_t conv_ch;
        uint32_t qk_ch;
        uint32_t head_dim;
        uint32_t taps;
        uint32_t qbase;
        uint32_t kbase;
        float    l2_eps;
    } pcs;
    pcs.n_tokens = tokens;
    pcs.conv_ch  = dn->sh.conv_ch;
    pcs.qk_ch    = dn->sh.qk_ch;
    pcs.head_dim = dn->sh.head_dim;
    pcs.taps     = dn->sh.conv_taps;
    pcs.qbase    = 0;
    pcs.kbase    = 0;
    pcs.l2_eps   = l2_eps;    // eps / head_dim, the caller passes attention eps / head_dim
    vkCmdBindPipeline(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->prep_pipe);
    vkCmdBindDescriptorSets(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->prep_set, 0, nullptr);
    vkCmdPushConstants(dn->cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    vkCmdDispatch(dn->cmd, dn->sh.conv_ch / dn->sh.head_dim, tokens, 1);
    r = vkEndCommandBuffer(dn->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (deltanet prep)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &dn->cmd;
    const double t0 = now_ms();
    r = vkQueueSubmit(dn->dev->compute_queue, 1, &si, dn->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (deltanet prep)", r);
    }
    r = vkWaitForFences(dn->dev->device, 1, &dn->fence, VK_TRUE, UINT64_MAX);
    dn->last_ms = now_ms() - t0;
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (deltanet prep)", r);
    }
    return vkResetFences(dn->dev->device, 1, &dn->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (deltanet prep) failed");
}

int qrt_dn_gate_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "gate chunk of %u tokens does not fit %u", tokens,
                    dn->max_tokens);
    }
    // bindings follow kernels/deltanet_gate.comp
    const VkBuffer bufs[7] = { dn->prep_norm, w->alpha_w, w->beta_w, w->dt, w->a,
                               dn->bufs.decay, dn->bufs.beta };
    VkDescriptorBufferInfo info[7];
    VkWriteDescriptorSet wr[7];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 7; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->gate_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dn->dev->device, 7, wr, 0, nullptr);

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(dn->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (deltanet gate)", r);
    }
    struct { uint32_t n_embd; uint32_t heads; uint32_t xbase; } pcs;
    pcs.n_embd = dn->sh.n_embd;
    pcs.heads  = dn->sh.n_v_heads;
    pcs.xbase  = 0;
    vkCmdBindPipeline(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->gate_pipe);
    vkCmdBindDescriptorSets(dn->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->gate_set, 0, nullptr);
    vkCmdPushConstants(dn->cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    vkCmdDispatch(dn->cmd, tokens, 1, 1);
    r = vkEndCommandBuffer(dn->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (deltanet gate)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &dn->cmd;
    const double t0 = now_ms();
    r = vkQueueSubmit(dn->dev->compute_queue, 1, &si, dn->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (deltanet gate)", r);
    }
    r = vkWaitForFences(dn->dev->device, 1, &dn->fence, VK_TRUE, UINT64_MAX);
    dn->last_ms = now_ms() - t0;
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (deltanet gate)", r);
    }
    return vkResetFences(dn->dev->device, 1, &dn->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (deltanet gate) failed");
}

#endif // QRT_HAVE_VULKAN



void qrt_dn_gate_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * xnorm,
                     const float * alpha_w, const float * beta_w, const float * dt,
                     const float * a, float * decay, float * beta) {
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t h = 0; h < sh->n_v_heads; ++h) {
            double alpha = 0.0;
            double braw = 0.0;
            for (uint32_t c = 0; c < sh->n_embd; ++c) {
                const double xv = (double) xnorm[(size_t) t * sh->n_embd + c];
                alpha += xv * (double) alpha_w[(size_t) c + (size_t) h * sh->n_embd];
                braw  += xv * (double) beta_w[(size_t) c + (size_t) h * sh->n_embd];
            }
            const double sp = log1p(exp(alpha + (double) dt[h]));
            decay[(size_t) t * sh->n_v_heads + h] = (float) exp((double) a[h] * sp);
            beta[(size_t) t * sh->n_v_heads + h] = (float) (1.0 / (1.0 + exp(-braw)));
        }
    }
}

void qrt_dn_prep_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * qkv,
                     const float * conv_w, const float * conv_state_in, float l2_eps,
                     float * q, float * k, float * v, float * conv_state_out) {
    const uint32_t taps = sh->conv_taps;
    const uint32_t ch = sh->conv_ch;
    const uint32_t qk = sh->qk_ch;
    const uint32_t hd = sh->head_dim;
    const uint32_t heads = qk / hd;
    const double eps_term = (double) l2_eps * (double) hd;
    std::vector<double> conv((size_t) ch);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < ch; ++c) {
            double acc = 0.0;
            for (uint32_t j = 0; j < taps; ++j) {
                const int src = (int) t - (int) taps + 1 + (int) j;
                const double val = src < 0
                    ? (double) conv_state_in[(size_t) (taps - 1 + (uint32_t) src) * ch + c]
                    : (double) qkv[(size_t) src * ch + c];
                acc += (double) conv_w[(size_t) j * ch + c] * val;
            }
            conv[c] = acc / (1.0 + exp(-acc));      // silu
        }
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = half * qk + h * hd;
                double ss = 0.0;
                for (uint32_t i = 0; i < hd; ++i) {
                    const double val = conv[(size_t) off + i];
                    ss += val * val;
                }
                const double inv = 1.0 / sqrt(ss + eps_term);
                float * dst = half == 0 ? q : k;
                for (uint32_t i = 0; i < hd; ++i) {
                    dst[(size_t) t * qk + (size_t) h * hd + i] = (float) (conv[(size_t) off + i] * inv);
                }
            }
        }
        for (uint32_t c = 0; c < sh->d_inner; ++c) {
            v[(size_t) t * sh->d_inner + c] = (float) conv[(size_t) 2 * qk + c];
        }
    }
    (void) conv_state_out;
}

void qrt_dn_recur_cpu(const qrt_dn_shapes * sh, uint32_t tokens,
                      const float * q, const float * k, const float * v,
                      const float * decay, const float * beta,
                      double * state, float * y) {
    const uint32_t sd = sh->head_dim;
    const uint32_t ratio = sh->n_v_heads / sh->n_k_heads;
    const double qscale = 1.0 / sqrt((double) sd);
    std::vector<double> kcol((size_t) sd), qcol((size_t) sd), acc((size_t) sd);
    for (uint32_t h = 0; h < sh->n_v_heads; ++h) {
        double * s = state + (size_t) h * sd * sd;
        const uint32_t kh = h / ratio;
        for (uint32_t t = 0; t < tokens; ++t) {
            const double d = (double) decay[(size_t) t * sh->n_v_heads + h];
            const double b = (double) beta[(size_t) t * sh->n_v_heads + h];
            const float * qrow = q + ((size_t) t * sh->qk_ch + (size_t) kh * sd);
            const float * krow = k + ((size_t) t * sh->qk_ch + (size_t) kh * sd);
            const float * vrow = v + ((size_t) t * sh->d_inner + (size_t) h * sd);
            for (uint32_t i = 0; i < sd; ++i) {
                kcol[i] = (double) krow[i];
                qcol[i] = (double) qrow[i];
            }
            // s *= d; sk[j] = sum_i s[i][j] k[i]
            // the state is stored column major, the same layout the kernel keeps in
            // registers, so a saved state can be handed to either side unchanged
            for (uint32_t j = 0; j < sd; ++j) {
                double sum = 0.0;
                for (uint32_t i = 0; i < sd; ++i) {
                    double & e = s[(size_t) j * sd + i];
                    e *= d;
                    sum += e * kcol[i];
                }
                acc[j] = ((double) vrow[j] - sum) * b;
            }
            float * yrow = y + ((size_t) t * sh->d_inner + (size_t) h * sd);
            for (uint32_t j = 0; j < sd; ++j) {
                const double delta = acc[j];
                double sum = 0.0;
                for (uint32_t i = 0; i < sd; ++i) {
                    double & e = s[(size_t) j * sd + i];
                    e += kcol[i] * delta;
                    sum += e * qcol[i];
                }
                yrow[j] = (float) (sum * qscale);
            }
        }
    }
}
