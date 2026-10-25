#include "kernels/gemm.h"

#include "kernels/tiles.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t qrt_gemm_block_bytes(qrt_quant type) {
    switch (type) {
        case QRT_QUANT_Q4K: return QRT_BLOCK_Q4K;
        case QRT_QUANT_Q5K: return QRT_BLOCK_Q5K;
        case QRT_QUANT_Q6K: return QRT_BLOCK_Q6K;
    }
    return 0;
}

uint64_t qrt_gemm_matrix_bytes(qrt_quant type, uint32_t n, uint32_t k) {
    return (uint64_t) n * (k / Q4K_QK) * qrt_gemm_block_bytes(type);
}

int qrt_gemm_type_of(uint32_t n, uint32_t k, uint64_t bytes) {
    for (int t = 0; t < 3; ++t) {
        if (qrt_gemm_matrix_bytes((qrt_quant) t, n, k) == bytes) {
            return t;
        }
    }
    return -1;
}

#ifdef QRT_HAVE_VULKAN

namespace {

const uint32_t GEMM_SET_INPUT  = 0;   // activations of the layer input
const uint32_t GEMM_SET_HIDDEN = 1;   // activations of the swiglu output

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

} // namespace

struct qrt_gemm {
    qrt_vk_dev *    dev;
    VkPipeline      gemm_pipe[3];   // Q4_K, Q5_K, Q6_K
    VkPipeline      swiglu_pipe;
    VkPipelineLayout gemm_layout;
    VkPipelineLayout swiglu_layout;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool pool;
    VkDescriptorSet gemm_sets[2];
    VkDescriptorSet swiglu_set;
    VkCommandPool   cmd_pool;
    VkCommandBuffer cmd;
    VkFence         fence;
    // Each descriptor set caches what it was last written with, so the prefill loop
    // rewrites them once per scheduler open instead of four times per expert.
    struct set_state {
        VkBuffer w, qs, ds, out;
    } st[2];
    VkBuffer   set_gate;
    VkBuffer   set_up;
    uint32_t   dispatches;
};

namespace {

// Four storage buffers bound at once; the order matches the shader bindings.
void write_set(VkDevice dev, VkDescriptorSet set, const VkBuffer * bufs) {
    VkDescriptorBufferInfo info[4];
    VkWriteDescriptorSet wr[4];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 4; ++i) {
        info[i].buffer = bufs[i];
        info[i].offset = 0;
        info[i].range  = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = set;
        wr[i].dstBinding      = (uint32_t) i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dev, 4, wr, 0, nullptr);
}

VkPipeline make_pipeline(VkDevice dev, VkShaderModule mod, VkPipelineLayout layout,
                         uint32_t pc_bytes, char * err, size_t err_size) {
    (void) pc_bytes;
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = layout;
    VkPipeline p = VK_NULL_HANDLE;
    const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
    if (r != VK_SUCCESS) {
        fail(err, err_size, "vkCreateComputePipelines failed: VkResult %d", (int) r);
        return VK_NULL_HANDLE;
    }
    return p;
}

} // namespace

int qrt_gemm_open(qrt_vk_dev * dev, const char * spv_dir, qrt_gemm ** out,
                 char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL) {
        return fail(err, err_size, "the int8 GEMM needs a device and a SPIR-V directory");
    }
    qrt_gemm * g = (qrt_gemm *) calloc(1, sizeof(qrt_gemm));
    if (g == NULL) {
        return fail(err, err_size, "out of memory for the GEMM pipelines");
    }
    g->dev = dev;

    VkDescriptorSetLayoutBinding binds[4];
    memset(binds, 0, sizeof(binds));
    for (int i = 0; i < 4; ++i) {
        binds[i].binding         = (uint32_t) i;
        binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 4;
    dlci.pBindings    = binds;
    VkResult r = vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &g->dsl);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreateDescriptorSetLayout", r);
    }

    VkPushConstantRange pc[2];
    memset(pc, 0, sizeof(pc));
    pc[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc[0].size       = 6 * sizeof(uint32_t);   // n, k, m, wbase, obase, abase
    pc[1]            = pc[0];
    pc[1].size       = 3 * sizeof(uint32_t);   // m, nk32, hid
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &g->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc[0];
    r = vkCreatePipelineLayout(dev->device, &plci, nullptr, &g->gemm_layout);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreatePipelineLayout (gemm)", r);
    }
    plci.pPushConstantRanges = &pc[1];
    r = vkCreatePipelineLayout(dev->device, &plci, nullptr, &g->swiglu_layout);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreatePipelineLayout (swiglu)", r);
    }

    const char * names[3] = { "gemm_q4k.spv", "gemm_q5k.spv", "gemm_q6k.spv" };
    for (int i = 0; i < 3; ++i) {
        VkShaderModule mod = load_module(dev->device, spv_dir, names[i], err, err_size);
        if (mod == VK_NULL_HANDLE) {
            qrt_gemm_close(dev, g);
            return -1;
        }
        g->gemm_pipe[i] = make_pipeline(dev->device, mod, g->gemm_layout, pc[0].size,
                                        err, err_size);
        vkDestroyShaderModule(dev->device, mod, nullptr);
        if (g->gemm_pipe[i] == VK_NULL_HANDLE) {
            qrt_gemm_close(dev, g);
            return -1;
        }
    }
    VkShaderModule sw_mod = load_module(dev->device, spv_dir, "swiglu_q8.spv", err, err_size);
    if (sw_mod == VK_NULL_HANDLE) {
        qrt_gemm_close(dev, g);
        return -1;
    }
    g->swiglu_pipe = make_pipeline(dev->device, sw_mod, g->swiglu_layout, pc[1].size, err, err_size);
    vkDestroyShaderModule(dev->device, sw_mod, nullptr);
    if (g->swiglu_pipe == VK_NULL_HANDLE) {
        qrt_gemm_close(dev, g);
        return -1;
    }

    VkDescriptorPoolSize size;
    memset(&size, 0, sizeof(size));
    size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 64;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &size;
    r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &g->pool);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (gemm)", r);
    }
    VkDescriptorSet sets[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    // pSetLayouts needs one entry per set: passing a single-element array with
    // descriptorSetCount 3 makes the driver read garbage and report pool exhaustion.
    VkDescriptorSetLayout layouts[3] = { g->dsl, g->dsl, g->dsl };
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = g->pool;
    dsai.descriptorSetCount = 3;
    dsai.pSetLayouts        = layouts;
    r = vkAllocateDescriptorSets(dev->device, &dsai, sets);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (gemm)", r);
    }
    g->gemm_sets[GEMM_SET_INPUT]  = sets[0];
    g->gemm_sets[GEMM_SET_HIDDEN] = sets[1];
    g->swiglu_set                 = sets[2];

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->compute_family;
    r = vkCreateCommandPool(dev->device, &pci, nullptr, &g->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreateCommandPool (gemm)", r);
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = g->cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &g->cmd);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers (gemm)", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &g->fence);
    if (r != VK_SUCCESS) {
        qrt_gemm_close(dev, g);
        return vk_fail(err, err_size, "vkCreateFence (gemm)", r);
    }
    *out = g;
    return 0;
}

void qrt_gemm_close(qrt_vk_dev * dev, qrt_gemm * g) {
    if (g == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (g->fence != VK_NULL_HANDLE) {
            vkDestroyFence(dev->device, g->fence, nullptr);
        }
        if (g->cmd_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, g->cmd_pool, nullptr);
        }
        if (g->pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(dev->device, g->pool, nullptr);
        }
        if (g->swiglu_pipe != VK_NULL_HANDLE) {
            vkDestroyPipeline(dev->device, g->swiglu_pipe, nullptr);
        }
        for (int i = 0; i < 3; ++i) {
            if (g->gemm_pipe[i] != VK_NULL_HANDLE) {
                vkDestroyPipeline(dev->device, g->gemm_pipe[i], nullptr);
            }
        }
        if (g->swiglu_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(dev->device, g->swiglu_layout, nullptr);
        }
        if (g->gemm_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(dev->device, g->gemm_layout, nullptr);
        }
        if (g->dsl != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev->device, g->dsl, nullptr);
        }
    }
    free(g);
}

namespace {

// Chooses the descriptor set for the activation planes and refreshes it when a buffer
// changed. Returns 0 on success.
int pick_set(qrt_gemm * g, const qrt_gemm_args * a, VkDescriptorSet * out) {
    // The activation planes are fixed per slot: the pair the hidden set already holds
    // means the swiglu output, anything else is the layer input set.
    const int idx = (a->a_qs == g->st[GEMM_SET_HIDDEN].qs &&
                     a->a_ds == g->st[GEMM_SET_HIDDEN].ds) ? GEMM_SET_HIDDEN : GEMM_SET_INPUT;
    qrt_gemm::set_state * st = &g->st[idx];
    if (a->w != st->w || a->out != st->out || a->a_qs != st->qs || a->a_ds != st->ds) {
        const VkBuffer bufs[4] = { a->w, a->a_qs, a->out, a->a_ds };
        write_set(g->dev->device, g->gemm_sets[idx], bufs);
        st->w = a->w;
        st->qs = a->a_qs;
        st->ds = a->a_ds;
        st->out = a->out;
    }
    *out = g->gemm_sets[idx];
    return 0;
}

} // namespace

int qrt_gemm_begin(qrt_gemm * g, char * err, size_t err_size) {
    if (g == NULL) {
        return fail(err, err_size, "no GEMM context");
    }
    VkResult r = vkResetFences(g->dev->device, 1, &g->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetFences (gemm)", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(g->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (gemm)", r);
    }
    return 0;
}

int qrt_gemm_record(qrt_gemm * g, const qrt_gemm_args * a, char * err, size_t err_size) {
    if (g == NULL || a == NULL || a->w == VK_NULL_HANDLE || a->out == VK_NULL_HANDLE) {
        return fail(err, err_size, "the GEMM needs a weight buffer and an output buffer");
    }
    if (a->n % Q4K_TILE_N != 0 || a->m % Q4K_TILE_M != 0 || a->k % Q4K_QK != 0) {
        return fail(err, err_size, "n=%u must be a multiple of %d, m=%u of %d, k=%u of %d",
                    a->n, Q4K_TILE_N, a->m, Q4K_TILE_M, a->k, Q4K_QK);
    }
    if (a->type != QRT_QUANT_Q4K && a->type != QRT_QUANT_Q5K && a->type != QRT_QUANT_Q6K) {
        return fail(err, err_size, "unknown quant type %d", (int) a->type);
    }
    VkDescriptorSet set = VK_NULL_HANDLE;
    pick_set(g, a, &set);
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->gemm_pipe[a->type]);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->gemm_layout, 0, 1,
                            &set, 0, nullptr);
    const uint32_t pc[6] = { a->n, a->k, a->m, a->wbase, a->obase, a->abase };
    vkCmdPushConstants(g->cmd, g->gemm_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(g->cmd, a->n / Q4K_TILE_N, a->m / Q4K_TILE_M, 1);
    ++g->dispatches;
    return 0;
}

int qrt_gemm_record_swiglu(qrt_gemm * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs,
                          VkBuffer out_ds, uint32_t m, uint32_t hid, char * err, size_t err_size) {
    if (g == NULL || gate == VK_NULL_HANDLE || up == VK_NULL_HANDLE ||
        out_qs == VK_NULL_HANDLE || out_ds == VK_NULL_HANDLE) {
        return fail(err, err_size, "swiglu needs gate, up and both output planes");
    }
    if (hid % 32 != 0) {
        return fail(err, err_size, "hidden %u is not a multiple of 32", hid);
    }
    if (gate != g->set_gate || up != g->set_up || out_qs != g->st[GEMM_SET_HIDDEN].qs ||
        out_ds != g->st[GEMM_SET_HIDDEN].ds) {
        const VkBuffer bufs[4] = { gate, up, out_qs, out_ds };
        write_set(g->dev->device, g->swiglu_set, bufs);
        g->set_gate = gate;
        g->set_up = up;
        // The down GEMM of the next expert consumes exactly these planes.
        g->st[GEMM_SET_HIDDEN].qs = out_qs;
        g->st[GEMM_SET_HIDDEN].ds = out_ds;
    }
    const uint32_t nk32 = hid / 32;
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->swiglu_pipe);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->swiglu_layout, 0, 1,
                            &g->swiglu_set, 0, nullptr);
    const uint32_t pc[3] = { m, nk32, hid };
    vkCmdPushConstants(g->cmd, g->swiglu_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(g->cmd, (m * nk32 + 63u) / 64u, 1, 1);
    ++g->dispatches;
    return 0;
}

int qrt_gemm_submit(qrt_gemm * g, char * err, size_t err_size) {
    if (g == NULL) {
        return fail(err, err_size, "no GEMM context");
    }
    VkResult r = vkEndCommandBuffer(g->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (gemm)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &g->cmd;
    r = vkQueueSubmit(g->dev->compute_queue, 1, &si, g->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (gemm)", r);
    }
    r = vkWaitForFences(g->dev->device, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (gemm)", r);
    }
    return 0;
}

uint32_t qrt_gemm_dispatches(const qrt_gemm * g) { return g != NULL ? g->dispatches : 0; }

int qrt_gemm_dispatch(qrt_gemm * g, const qrt_gemm_args * a, char * err, size_t err_size) {
    if (qrt_gemm_begin(g, err, err_size) != 0 || qrt_gemm_record(g, a, err, err_size) != 0) {
        return -1;
    }
    return qrt_gemm_submit(g, err, err_size);
}

int qrt_gemm_swiglu(qrt_gemm * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs, VkBuffer out_ds,
                   uint32_t m, uint32_t hid, char * err, size_t err_size) {
    if (qrt_gemm_begin(g, err, err_size) != 0 ||
        qrt_gemm_record_swiglu(g, gate, up, out_qs, out_ds, m, hid, err, err_size) != 0) {
        return -1;
    }
    return qrt_gemm_submit(g, err, err_size);
}

#else // QRT_HAVE_VULKAN

struct qrt_gemm { int unused; };

int qrt_gemm_open(qrt_vk_dev * dev, const char * spv_dir, qrt_gemm ** out,
                 char * err, size_t err_size) {
    (void) dev;
    (void) spv_dir;
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_gemm_close(qrt_vk_dev * dev, qrt_gemm * g) {
    (void) dev;
    (void) g;
}

int qrt_gemm_dispatch(qrt_gemm * g, const qrt_gemm_args * a, char * err, size_t err_size) {
    (void) g;
    (void) a;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_gemm_swiglu(qrt_gemm * g, VkBuffer gate, VkBuffer up, VkBuffer out_qs, VkBuffer out_ds,
                   uint32_t m, uint32_t hid, char * err, size_t err_size) {
    (void) g; (void) gate; (void) up; (void) out_qs; (void) out_ds; (void) m; (void) hid;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
