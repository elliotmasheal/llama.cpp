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
    pcr.size       = 4 * sizeof(uint32_t) + sizeof(float);
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
    psize.descriptorCount = DN_BINDINGS;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &dn->pool);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (deltanet)", r);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = dn->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dn->dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &dn->set);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (deltanet)", r);
    }
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

#endif // QRT_HAVE_VULKAN

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
