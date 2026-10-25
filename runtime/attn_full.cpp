// Full-attention layer of qwen35moe (M2-6d): the kernel launches and the double
// precision references. Shapes come from the GGUF metadata only
// (qrt_full_shapes_from_gguf), so the check and the layer chain cannot drift apart.

#include "attn_full.h"

#include "gguf_index.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
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

int vk_fail(char * err, size_t err_size, const char * what, VkResult r) {
    return fail(err, err_size, "%s failed: VkResult %d", what, (int) r);
}

int read_spv(const char * dir, const char * name, unsigned char ** out, size_t * out_bytes,
             char * err, size_t err_size) {
    char path[1024];
    if (dir != NULL && *dir != '\0') {
        snprintf(path, sizeof(path), "%s/%s", dir, name);
    } else {
        snprintf(path, sizeof(path), "%s", name);
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return fail(err, err_size, "cannot size '%s'", path);
    }
    const long n = ftell(f);
    if (n <= 0) {
        fclose(f);
        return fail(err, err_size, "'%s' is empty", path);
    }
    rewind(f);
    unsigned char * buf = (unsigned char *) malloc((size_t) n);
    if (buf == NULL) {
        fclose(f);
        return fail(err, err_size, "out of memory for '%s'", path);
    }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) {
        free(buf);
        fclose(f);
        return fail(err, err_size, "short read on '%s'", path);
    }
    fclose(f);
    *out = buf;
    *out_bytes = (size_t) n;
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

// One pipeline per kernel; the spec block is a flat list of uint32 in constantID order.
int make_pipeline(qrt_vk_dev * dev, VkPipelineLayout layout, const char * spv_dir,
                  const char * name, const uint32_t * spec, uint32_t n_spec,
                  VkPipeline * out, char * err, size_t err_size) {
    VkShaderModule mod = load_module(dev->device, spv_dir, name, err, err_size);
    if (mod == VK_NULL_HANDLE) {
        return -1;
    }
    VkSpecializationMapEntry sme[4];
    memset(sme, 0, sizeof(sme));
    for (uint32_t i = 0; i < n_spec; ++i) {
        sme[i].constantID = i;
        sme[i].offset     = i * sizeof(uint32_t);
        sme[i].size       = sizeof(uint32_t);
    }
    VkSpecializationInfo sinfo;
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.mapEntryCount = n_spec;
    sinfo.pMapEntries   = sme;
    sinfo.dataSize      = n_spec * sizeof(uint32_t);
    sinfo.pData         = spec;

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.stage.pSpecializationInfo = n_spec > 0 ? &sinfo : nullptr;
    cpci.layout       = layout;
    const VkResult r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr, out);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateComputePipelines (full attention)", r);
    }
    return 0;
}

int make_set_layout(qrt_vk_dev * dev, uint32_t bindings, VkDescriptorSetLayout * out,
                    char * err, size_t err_size) {
    VkDescriptorSetLayoutBinding b[8];
    memset(b, 0, sizeof(b));
    if (bindings > 8) {
        return fail(err, err_size, "%u bindings is above the eight the layouts carry", bindings);
    }
    for (uint32_t i = 0; i < bindings; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = bindings;
    dlci.pBindings    = b;
    const VkResult r = vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, out);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateDescriptorSetLayout (full attention)", r);
    }
    return 0;
}

int make_layout(qrt_vk_dev * dev, VkDescriptorSetLayout dsl, uint32_t push_bytes,
                VkPipelineLayout * out, char * err, size_t err_size) {
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = push_bytes;
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    const VkResult r = vkCreatePipelineLayout(dev->device, &plci, nullptr, out);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreatePipelineLayout (full attention)", r);
    }
    return 0;
}

int write_set(qrt_vk_dev * dev, VkDescriptorSet set, VkBuffer * bufs, uint32_t n) {
    VkDescriptorBufferInfo info[8];
    VkWriteDescriptorSet wr[8];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < n; ++i) {
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
    return 0;
}

} // namespace

struct qrt_full {
    qrt_vk_dev *    dev;
    qrt_full_shapes sh;
    uint32_t        max_tokens;
    uint32_t        n_ctx;
    qrt_full_bufs   bufs;
    VkDeviceMemory  mem[FULL_BUFFER_COUNT];
    double          last_ms;
    uint64_t        last_kv_reads;

    VkDescriptorSetLayout qk_dsl;
    VkDescriptorSetLayout attn_dsl;
    VkDescriptorPool      pool;
    VkDescriptorSet       qk_set;
    VkDescriptorSet       attn_set;
    VkPipelineLayout      qk_layout;
    VkPipelineLayout      attn_layout;
    VkPipeline            qk_pipe;
    VkPipeline            attn_pipe;
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
};

int qrt_full_shapes_from_gguf(const qrt_gguf_hparams * hp, qrt_full_shapes * out,
                              char * err, size_t err_size) {
    memset(out, 0, sizeof(*out));
    if (hp->arch[0] == '\0' || hp->head_count <= 0 || hp->head_count_kv <= 0 ||
        (hp->key_length <= 0 && hp->value_length <= 0)) {
        return fail(err, err_size, "metadata lacks attention.head_count / head_count_kv / key_length");
    }
    out->n_embd   = (uint32_t) hp->n_embd;
    out->head_dim = (uint32_t) (hp->key_length > 0 ? hp->key_length : hp->value_length);
    out->n_head   = (uint32_t) hp->head_count;
    out->n_kv     = (uint32_t) hp->head_count_kv;
    out->n_rot    = (uint32_t) (hp->rope_dimension > 0 ? hp->rope_dimension : out->head_dim);
    for (int i = 0; i < 4; ++i) {
        out->sections[i] = hp->rope_sections[i];
    }
    out->freq_base  = hp->rope_freq_base > 0.0f ? hp->rope_freq_base : 10000.0f;
    out->freq_scale = 1.0f;
    out->eps        = 1e-6f;
    out->kq_scale   = 1.0f / sqrtf((float) out->head_dim);
    out->qg_ch      = out->n_head * 2 * out->head_dim;
    out->kv_ch      = out->n_kv * out->head_dim;
    if (out->head_dim != 256) {
        return fail(err, err_size, "head_dim %u is not the 256 the kernels tile", out->head_dim);
    }
    if ((out->n_rot % 2) != 0 || out->n_rot > out->head_dim) {
        return fail(err, err_size, "n_rot %u does not fit head_dim %u", out->n_rot, out->head_dim);
    }
    return 0;
}

int qrt_full_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_full_shapes * sh,
                  uint32_t max_tokens, uint32_t n_ctx, qrt_full ** out,
                  char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL || sh == NULL || max_tokens == 0 || n_ctx == 0) {
        return fail(err, err_size, "full attention needs a device, SPIR-V, tokens and a ctx");
    }
    if (sh->head_dim != 256 || (sh->n_rot % 2) != 0 || sh->n_head == 0 || sh->n_kv == 0) {
        return fail(err, err_size, "shapes head_dim=%u n_rot=%u heads=%u/%u do not fit",
                    sh->head_dim, sh->n_rot, sh->n_head, sh->n_kv);
    }
    if (n_ctx < max_tokens) {
        return fail(err, err_size, "the KV cache holds %u tokens, the chunk is %u",
                    n_ctx, max_tokens);
    }
    qrt_full * f = (qrt_full *) calloc(1, sizeof(qrt_full));
    if (f == NULL) {
        return fail(err, err_size, "out of memory for full attention");
    }
    f->dev = dev;
    f->sh = *sh;
    f->max_tokens = max_tokens;
    f->n_ctx = n_ctx;

    const uint64_t qg_bytes = (uint64_t) max_tokens * sh->qg_ch * 4;
    const uint64_t kvi_bytes = (uint64_t) max_tokens * sh->kv_ch * 4;
    const uint64_t q_bytes  = (uint64_t) max_tokens * sh->n_head * sh->head_dim * 4;
    const uint64_t kv_bytes = (uint64_t) sh->n_kv * n_ctx * sh->head_dim * 2;
    const uint64_t w_bytes  = (uint64_t) sh->head_dim * 4;
    struct slot { VkBuffer * b; uint64_t bytes; } slots[FULL_BUFFER_COUNT] = {
        { &f->bufs.qg, qg_bytes }, { &f->bufs.kin, kvi_bytes }, { &f->bufs.vin, kvi_bytes },
        { &f->bufs.q, q_bytes }, { &f->bufs.kcache, kv_bytes }, { &f->bufs.vcache, kv_bytes },
        { &f->bufs.out, q_bytes }, { &f->bufs.q_norm_w, w_bytes }, { &f->bufs.k_norm_w, w_bytes },
    };
    for (uint32_t i = 0; i < FULL_BUFFER_COUNT; ++i) {
        if (qrt_vk_alloc(dev, slots[i].bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                         QRT_VK_USAGE_TRANSFER_DST, slots[i].b, &f->mem[i], err, err_size) != 0) {
            qrt_full_close(dev, f);
            return -1;
        }
    }

    if (make_set_layout(dev, 8, &f->qk_dsl, err, err_size) != 0 ||
        make_layout(dev, f->qk_dsl, 8 * sizeof(uint32_t), &f->qk_layout, err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    // the attention kernel needs 27.5 KiB of shared memory per workgroup
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev->phys, &props);
    if (props.limits.maxComputeSharedMemorySize < 30u * 1024u) {
        qrt_full_close(dev, f);
        return fail(err, err_size, "the attention kernel needs 26624 bytes of shared memory, "
                    "this device has %u", props.limits.maxComputeSharedMemorySize);
    }
    if (make_set_layout(dev, 5, &f->attn_dsl, err, err_size) != 0 ||
        make_layout(dev, f->attn_dsl, 5 * sizeof(uint32_t), &f->attn_layout, err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    if (make_pipeline(dev, f->attn_layout, spv_dir, "attn_full_attn.spv", NULL, 0, &f->attn_pipe,
                      err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    const uint32_t spec[4] = { sh->head_dim, sh->n_rot, sh->n_head, sh->n_kv };
    if (make_pipeline(dev, f->qk_layout, spv_dir, "attn_full_qk.spv", spec, 4, &f->qk_pipe,
                      err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 32;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 2;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    VkResult r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &f->pool);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (full attention)", r);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = f->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &f->qk_dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &f->qk_set);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (full attention qk)", r);
    }
    dsai.pSetLayouts = &f->attn_dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &f->attn_set);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (full attention)", r);
    }
    VkBuffer attn_bufs[5] = { f->bufs.q, f->bufs.kcache, f->bufs.vcache, f->bufs.out, f->bufs.qg };
    write_set(dev, f->attn_set, attn_bufs, 5);
    VkBuffer qk_bufs[8] = { f->bufs.qg, f->bufs.kin, f->bufs.vin, f->bufs.q,
                            f->bufs.kcache, f->bufs.vcache, f->bufs.q_norm_w, f->bufs.k_norm_w };
    write_set(dev, f->qk_set, qk_bufs, 8);

    VkCommandPoolCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = dev->compute_family;
    r = vkCreateCommandPool(dev->device, &cpci, nullptr, &f->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateCommandPool (full attention)", r);
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = f->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cbai, &f->cmd);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers (full attention)", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &f->fence);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateFence (full attention)", r);
    }
    *out = f;
    return 0;
}

void qrt_full_close(qrt_vk_dev * dev, qrt_full * f) {
    if (f == NULL) {
        return;
    }
    VkBuffer bufs[FULL_BUFFER_COUNT] = { f->bufs.qg, f->bufs.kin, f->bufs.vin, f->bufs.q,
                                         f->bufs.kcache, f->bufs.vcache, f->bufs.out,
                                         f->bufs.q_norm_w, f->bufs.k_norm_w };
    for (uint32_t i = 0; i < FULL_BUFFER_COUNT; ++i) {
        if (f->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, bufs[i], f->mem[i]);
        }
    }
    if (f->qk_pipe != VK_NULL_HANDLE)    vkDestroyPipeline(dev->device, f->qk_pipe, nullptr);
    if (f->attn_pipe != VK_NULL_HANDLE)  vkDestroyPipeline(dev->device, f->attn_pipe, nullptr);
    if (f->fence != VK_NULL_HANDLE)      vkDestroyFence(dev->device, f->fence, nullptr);
    if (f->cmd_pool != VK_NULL_HANDLE)   vkDestroyCommandPool(dev->device, f->cmd_pool, nullptr);
    if (f->pool != VK_NULL_HANDLE)       vkDestroyDescriptorPool(dev->device, f->pool, nullptr);
    if (f->qk_layout != VK_NULL_HANDLE)  vkDestroyPipelineLayout(dev->device, f->qk_layout, nullptr);
    if (f->attn_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev->device, f->attn_layout, nullptr);
    if (f->qk_dsl != VK_NULL_HANDLE)     vkDestroyDescriptorSetLayout(dev->device, f->qk_dsl, nullptr);
    if (f->attn_dsl != VK_NULL_HANDLE)   vkDestroyDescriptorSetLayout(dev->device, f->attn_dsl, nullptr);
    free(f);
}

const qrt_full_shapes * qrt_full_shapes_of(const qrt_full * f) { return &f->sh; }
const qrt_full_bufs *   qrt_full_buffers(const qrt_full * f) { return &f->bufs; }
double                  qrt_full_last_ms(const qrt_full * f) { return f->last_ms; }
uint64_t                qrt_full_last_kv_reads(const qrt_full * f) { return f->last_kv_reads; }

uint64_t qrt_full_kv_bytes(const qrt_full * f) {
    return (uint64_t) f->sh.n_kv * f->n_ctx * f->sh.head_dim * 2;   // one plane
}

static int submit_and_wait(qrt_full * f, char * err, size_t err_size) {
    const VkResult r = vkEndCommandBuffer(f->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (full attention)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &f->cmd;
    const double t0 = now_ms();
    VkResult rs = vkQueueSubmit(f->dev->compute_queue, 1, &si, f->fence);
    if (rs != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (full attention)", rs);
    }
    rs = vkWaitForFences(f->dev->device, 1, &f->fence, VK_TRUE, UINT64_MAX);
    f->last_ms = now_ms() - t0;
    if (rs != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (full attention)", rs);
    }
    return vkResetFences(f->dev->device, 1, &f->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (full attention) failed");
}

static int begin_cmd(qrt_full * f, char * err, size_t err_size) {
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(f->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (full attention)", r);
    }
    return 0;
}

int qrt_full_qk(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    if (tokens == 0 || tokens > f->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, f->max_tokens);
    }
    if (pos0 + tokens > f->n_ctx) {
        return fail(err, err_size, "positions %u..%u are past the %u token KV cache",
                    pos0, pos0 + tokens, f->n_ctx);
    }
    if (begin_cmd(f, err, err_size) != 0) {
        return -1;
    }
    struct {
        uint32_t pos0;
        uint32_t tokens;
        uint32_t qg_ch;
        uint32_t kv_ch;
        uint32_t kv_cap;
        float    eps;
        float    freq_base;
        float    freq_scale;
    } pc;
    pc.pos0       = pos0;
    pc.tokens     = tokens;
    pc.qg_ch      = f->sh.qg_ch;
    pc.kv_ch      = f->sh.kv_ch;
    pc.kv_cap     = f->n_ctx;
    pc.eps        = f->sh.eps;
    pc.freq_base  = f->sh.freq_base;
    pc.freq_scale = f->sh.freq_scale;
    vkCmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->qk_pipe);
    vkCmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->qk_layout, 0, 1,
                            &f->qk_set, 0, nullptr);
    vkCmdPushConstants(f->cmd, f->qk_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // one workgroup per token: a warp owns one head's norm and rope
    vkCmdDispatch(f->cmd, tokens, 1, 1);
    return submit_and_wait(f, err, err_size);
}

int qrt_full_attn(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    if (tokens == 0 || tokens > f->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, f->max_tokens);
    }
    if (pos0 + tokens > f->n_ctx) {
        return fail(err, err_size, "positions %u..%u are past the %u token KV cache",
                    pos0, pos0 + tokens, f->n_ctx);
    }
    if (begin_cmd(f, err, err_size) != 0) {
        return -1;
    }
    struct {
        uint32_t pos0;
        uint32_t tokens;
        uint32_t kv_cap;
        uint32_t qg_ch;
        float    kq_scale;
    } pc;
    pc.pos0     = pos0;
    pc.tokens   = tokens;
    pc.kv_cap   = f->n_ctx;
    pc.qg_ch    = f->sh.qg_ch;
    pc.kq_scale = f->sh.kq_scale;
    vkCmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->attn_pipe);
    vkCmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->attn_layout, 0, 1,
                            &f->attn_set, 0, nullptr);
    vkCmdPushConstants(f->cmd, f->attn_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // one workgroup per (query tile of 32, query head)
    vkCmdDispatch(f->cmd, (tokens + 63u) / 64u, f->sh.n_head, 1);

    // the account the "KV read once" claim is checked against: every query tile scans
    // the KV tiles up to its own diagonal
    uint64_t tiles = 0;
    for (uint32_t q0 = 0; q0 < tokens; q0 += 64u) {
        const uint32_t rows = tokens - q0 < 64u ? tokens - q0 : 64u;
        tiles += (pos0 + q0 + rows + 31u) / 32u;
    }
    f->last_kv_reads = tiles * f->sh.n_head * 32ull * f->sh.head_dim * 2ull * 2ull;

    return submit_and_wait(f, err, err_size);
}

// ---------------------------------------------------------------- references

static void ref_norm_rope(const qrt_full_shapes * sh, const float * src, const float * nw,
                          uint32_t pos, double * dst) {
    const uint32_t hd = sh->head_dim;
    double ss = 0.0;
    for (uint32_t i = 0; i < hd; ++i) {
        ss += (double) src[i] * (double) src[i];
    }
    const double inv = 1.0 / sqrt(ss / (double) hd + (double) sh->eps);
    for (uint32_t i = 0; i < hd; ++i) {
        dst[i] = (double) src[i] * inv * (double) nw[i];
    }
    const uint32_t half = sh->n_rot / 2;
    const double step = pow((double) sh->freq_base, -2.0 / (double) sh->n_rot);
    for (uint32_t p = 0; p < half; ++p) {
        const double theta = (double) pos * (double) sh->freq_scale * pow(step, (double) p);
        const double c = cos(theta);
        const double s = sin(theta);
        const double x0 = dst[p];
        const double x1 = dst[p + half];
        dst[p]        = x0 * c - x1 * s;
        dst[p + half] = x0 * s + x1 * c;
    }
}

void qrt_full_qk_cpu(const qrt_full_shapes * sh, uint32_t pos0, uint32_t tokens,
                     const float * qg, const float * kin, const float * vin,
                     const float * q_norm_w, const float * k_norm_w,
                     float * q_out, float * k_out, float * v_out) {
    std::vector<double> tmp(sh->head_dim);
    const uint32_t hd = sh->head_dim;
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t pos = pos0 + t;
        for (uint32_t h = 0; h < sh->n_head; ++h) {
            const float * src = qg + (size_t) t * sh->qg_ch + (size_t) h * 2 * hd;
            ref_norm_rope(sh, src, q_norm_w, pos, tmp.data());
            float * dst = q_out + ((size_t) t * sh->n_head + h) * hd;
            for (uint32_t i = 0; i < hd; ++i) {
                dst[i] = (float) tmp[i];
            }
        }
        for (uint32_t h = 0; h < sh->n_kv; ++h) {
            const float * src = kin + (size_t) t * sh->kv_ch + (size_t) h * hd;
            ref_norm_rope(sh, src, k_norm_w, pos, tmp.data());
            float * dst = k_out + ((size_t) h * tokens + t) * hd;
            for (uint32_t i = 0; i < hd; ++i) {
                dst[i] = (float) tmp[i];
            }
            memcpy(v_out + ((size_t) h * tokens + t) * hd, vin + (size_t) t * sh->kv_ch + (size_t) h * hd,
                   (size_t) hd * sizeof(float));
        }
    }
}

// f16 rounding of a f32 value, so the probe below can reproduce what packed half
// arithmetic would see without needing a second kernel.
static float f16_to_f32(uint32_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t expo = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    float v;
    if (expo == 0) {
        v = ldexpf((float) mant, -24);
    } else if (expo == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = ldexpf((float) (mant | 0x400u), (int) expo - 25);
    }
    return sign ? -v : v;
}

static float as_f16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t expo = (int32_t) ((u >> 23) & 0xffu) - 127;
    const uint32_t man = u & 0x7fffffu;
    if (expo > 15) {
        const float big = sign ? -65504.0f : 65504.0f;
        return big;
    }
    if (expo < -24) {
        return sign ? -0.0f : 0.0f;
    }
    if (expo < -14) {
        // subnormal: shift the implicit leading one into place
        const uint32_t m = (man | 0x800000u) >> (uint32_t) (-expo - 14);
        const uint32_t bits = sign | (m >> 13);
        const uint32_t rem = m & 0x1fffu;
        uint32_t h = bits + ((rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) ? 1u : 0u);
        return f16_to_f32(h);
    }
    uint32_t h = sign | ((uint32_t) (expo + 15) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        h += 1u;
    }
    return f16_to_f32(h);
}

// Chunked f16 dot: eight products summed in f16, each chunk added to the f32 sum. That is
// the best a packed-half kernel (V_PK_FMA_F16 in f16, or f16 dot2) can do without f32
// accumulate, and it is what decides whether the numerics allow the fast path at all.
static float dot_chunk_f16(const float * a, const float * b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i += 8u) {
        float chunk = 0.0f;
        for (uint32_t j = 0; j < 8u; ++j) {
            chunk = as_f16(chunk + as_f16(a[i + j]) * as_f16(b[i + j]));
        }
        acc += chunk;
    }
    return acc;
}

static void attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx, const uint32_t * rows,
                          uint32_t n_rows, const float * q, const float * kcache,
                          const float * vcache, const float * qg, float * out, int f16) {
    const uint32_t hd = sh->head_dim;
    std::vector<double> acc(hd);
    std::vector<float> accf(hd);
    for (uint32_t ri = 0; ri < n_rows; ++ri) {
        const uint32_t t = rows[ri];
        for (uint32_t h = 0; h < sh->n_head; ++h) {
            const uint32_t kvh = h / (sh->n_head / sh->n_kv);
            const float * qrow = q + ((size_t) t * sh->n_head + h) * hd;
            // causal range is [0, t]
            std::vector<double> scores((size_t) t + 1);
            double mx = -1e30;
            for (uint32_t u = 0; u <= t; ++u) {
                const float * krow = kcache + ((size_t) kvh * n_ctx + u) * hd;
                if (f16) {
                    scores[u] = (double) dot_chunk_f16(qrow, krow, hd) * (double) sh->kq_scale;
                } else {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < hd; ++d) {
                        dot += (double) qrow[d] * (double) krow[d];
                    }
                    scores[u] = dot * (double) sh->kq_scale;
                }
                mx = fmax(mx, scores[u]);
            }
            double sum = 0.0;
            for (uint32_t u = 0; u <= t; ++u) {
                scores[u] = exp(scores[u] - mx);
                sum += scores[u];
            }
            for (uint32_t d = 0; d < hd; ++d) {
                acc[d] = 0.0;
                accf[d] = 0.0f;
            }
            for (uint32_t u = 0; u <= t; ++u) {
                const float * vrow = vcache + ((size_t) kvh * n_ctx + u) * hd;
                const double p = scores[u] / sum;
                if (f16) {
                    const float pf = as_f16((float) p);
                    for (uint32_t d = 0; d < hd; ++d) {
                        accf[d] += pf * as_f16(vrow[d]);
                    }
                } else {
                    for (uint32_t d = 0; d < hd; ++d) {
                        acc[d] += p * (double) vrow[d];
                    }
                }
            }
            float * dst = out + ((size_t) ri * sh->n_head + h) * hd;
            for (uint32_t d = 0; d < hd; ++d) {
                const uint32_t ch = h * 2 * hd + hd + d;
                const double g = 1.0 / (1.0 + exp(-(double) qg[(size_t) t * sh->qg_ch + ch]));
                dst[d] = (float) ((f16 ? (double) accf[d] : acc[d]) * g);
            }
        }
    }
}

void qrt_full_attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx,
                            const uint32_t * rows, uint32_t n_rows,
                            const float * q, const float * kcache, const float * vcache,
                            const float * qg, float * out) {
    attn_rows_cpu(sh, n_ctx, rows, n_rows, q, kcache, vcache, qg, out, 0);
}

void qrt_full_attn_rows_cpu_f16(const qrt_full_shapes * sh, uint32_t n_ctx,
                                const uint32_t * rows, uint32_t n_rows,
                                const float * q, const float * kcache, const float * vcache,
                                const float * qg, float * out) {
    attn_rows_cpu(sh, n_ctx, rows, n_rows, q, kcache, vcache, qg, out, 1);
}

#else // QRT_HAVE_VULKAN

int qrt_full_shapes_from_gguf(const qrt_gguf_hparams * hp, qrt_full_shapes * out,
                              char * err, size_t err_size) {
    (void) hp; (void) out;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
int qrt_full_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_full_shapes * sh,
                  uint32_t max_tokens, uint32_t n_ctx, qrt_full ** out,
                  char * err, size_t err_size) {
    (void) dev; (void) spv_dir; (void) sh; (void) max_tokens; (void) n_ctx; (void) out;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
void qrt_full_close(qrt_vk_dev * dev, qrt_full * f) { (void) dev; (void) f; }
const qrt_full_shapes * qrt_full_shapes_of(const qrt_full * f) { (void) f; return NULL; }
const qrt_full_bufs *   qrt_full_buffers(const qrt_full * f) { (void) f; return NULL; }
uint64_t                qrt_full_kv_bytes(const qrt_full * f) { (void) f; return 0; }
double                  qrt_full_last_ms(const qrt_full * f) { (void) f; return 0.0; }
uint64_t                qrt_full_last_kv_reads(const qrt_full * f) { (void) f; return 0; }
int qrt_full_qk(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    (void) f; (void) pos0; (void) tokens;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
int qrt_full_attn(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    (void) f; (void) pos0; (void) tokens;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
void qrt_full_qk_cpu(const qrt_full_shapes * sh, uint32_t pos0, uint32_t tokens,
                     const float * qg, const float * kin, const float * vin,
                     const float * q_norm_w, const float * k_norm_w,
                     float * q_out, float * k_out, float * v_out) {
    (void) sh; (void) pos0; (void) tokens; (void) qg; (void) kin; (void) vin;
    (void) q_norm_w; (void) k_norm_w; (void) q_out; (void) k_out; (void) v_out;
}
void qrt_full_attn_rows_cpu(const qrt_full_shapes * sh, uint32_t n_ctx,
                            const uint32_t * rows, uint32_t n_rows,
                            const float * q, const float * kcache, const float * vcache,
                            const float * qg, float * out) {
    (void) sh; (void) n_ctx; (void) rows; (void) n_rows; (void) q; (void) kcache;
    (void) vcache; (void) qg; (void) out;
}

void qrt_full_attn_rows_cpu_f16(const qrt_full_shapes * sh, uint32_t n_ctx,
                                const uint32_t * rows, uint32_t n_rows,
                                const float * q, const float * kcache, const float * vcache,
                                const float * qg, float * out) {
    (void) sh; (void) n_ctx; (void) rows; (void) n_rows; (void) q; (void) kcache;
    (void) vcache; (void) qg; (void) out;
}

#endif // QRT_HAVE_VULKAN
