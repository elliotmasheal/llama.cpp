// The store read probe (M2-5 step 1): one shader that streams an expert block out of RAM so
// the read leg can be measured on its own. It is a diagnostic device, not part of the
// prefill schedule, so it keeps its own pipelines and its own translation unit.

#include "prefill.h"

#include "qrt_util.h"
#include "vk_dev.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

struct qrt_prefill_probe {
    qrt_vk_dev *          dev;
    VkShaderModule        mod;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkPipeline            pipe;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    VkBuffer              out_buf;
    VkDeviceMemory        out_mem;
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
};

namespace {

int read_file(const char * path, unsigned char ** out, size_t * out_bytes, char * err,
              size_t err_size) {
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

} // namespace

int qrt_prefill_probe_open(qrt_vk_dev * dev, const char * spv_dir, qrt_prefill * p,
                           qrt_prefill_probe ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || p == NULL || spv_dir == NULL) {
        return fail(err, err_size, "the probe needs a device, a scheduler and a shader dir");
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/qrt-read.spv", spv_dir);
    unsigned char * code = NULL;
    size_t code_bytes = 0;
    if (read_file(path, &code, &code_bytes, err, err_size) != 0) {
        return -1;
    }
    qrt_prefill_probe * probe = (qrt_prefill_probe *) calloc(1, sizeof(*probe));
    if (probe == NULL) {
        free(code);
        return fail(err, err_size, "out of memory for the prefill probe");
    }
    probe->dev = dev;
    VkResult r;
    VkShaderModuleCreateInfo smci;    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code_bytes;
    smci.pCode    = (const uint32_t *) code;
    r = vkCreateShaderModule(dev->device, &smci, nullptr, &probe->mod);
    free(code);
    if (r != VK_SUCCESS) {
        free(probe);
        return vk_fail(err, err_size, "vkCreateShaderModule", r);
    }

    VkDescriptorSetLayoutBinding binds[2];
    memset(binds, 0, sizeof(binds));
    binds[0].binding         = 0;
    binds[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1]                 = binds[0];
    binds[1].binding         = 1;
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 2;
    dlci.pBindings    = binds;
    r = vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &probe->dsl);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreateDescriptorSetLayout", r);
    }
    VkPushConstantRange pc;
    memset(&pc, 0, sizeof(pc));
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = 8;   // the shader takes (base, n) as two uints
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &probe->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    r = vkCreatePipelineLayout(dev->device, &plci, nullptr, &probe->pl);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreatePipelineLayout", r);
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module       = probe->mod;
    cpci.stage.pName        = "main";
    cpci.layout             = probe->pl;
    r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &probe->pipe);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreateComputePipelines", r);
    }

    const uint64_t out_bytes = 4096;
    if (qrt_vk_alloc(dev, out_bytes, QRT_VK_USAGE_STORAGE, &probe->out_buf, &probe->out_mem,
                     err, err_size) != 0) {
        qrt_prefill_probe_close(dev, probe);
        return -1;
    }
    VkDescriptorPoolSize size;
    memset(&size, 0, sizeof(size));
    size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &size;
    r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &probe->pool);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreateDescriptorPool", r);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = probe->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &probe->dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &probe->set);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets", r);
    }
    VkDescriptorBufferInfo info[2];
    memset(info, 0, sizeof(info));
    info[0].buffer = qrt_prefill_buffer(p);
    info[0].offset = 0;
    info[0].range  = VK_WHOLE_SIZE;
    info[1].buffer = probe->out_buf;
    info[1].offset = 0;
    info[1].range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wr[2];
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 2; ++i) {
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = probe->set;
        wr[i].dstBinding      = (uint32_t) i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dev->device, 2, wr, 0, nullptr);

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->compute_family;    r = vkCreateCommandPool(dev->device, &pci, nullptr, &probe->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreateCommandPool", r);
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = probe->cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &probe->cmd);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &probe->fence);
    if (r != VK_SUCCESS) {
        qrt_prefill_probe_close(dev, probe);
        return vk_fail(err, err_size, "vkCreateFence", r);
    }
    *out = probe;
    return 0;
}

void qrt_prefill_probe_close(qrt_vk_dev * dev, qrt_prefill_probe * probe) {
    if (probe == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (probe->fence != VK_NULL_HANDLE) {
            vkDestroyFence(dev->device, probe->fence, nullptr);
        }
        if (probe->cmd_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, probe->cmd_pool, nullptr);
        }
        if (probe->pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(dev->device, probe->pool, nullptr);
        }
        if (probe->out_mem != VK_NULL_HANDLE) {
            qrt_vk_free(dev, probe->out_buf, probe->out_mem);
        }
        if (probe->pipe != VK_NULL_HANDLE) {
            vkDestroyPipeline(dev->device, probe->pipe, nullptr);
        }
        if (probe->pl != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(dev->device, probe->pl, nullptr);
        }
        if (probe->dsl != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev->device, probe->dsl, nullptr);
        }
        if (probe->mod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(dev->device, probe->mod, nullptr);
        }
    }
    free(probe);
}

int qrt_prefill_probe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                              uint64_t slot_offset, uint64_t bytes,
                              char * err, size_t err_size) {
    (void) layer;
    (void) expert;
    (void) buf;   // the descriptor set already points at the slot buffer
    qrt_prefill_probe * probe = (qrt_prefill_probe *) user;
    if (probe == NULL) {
        return fail(err, err_size, "no probe pipeline");
    }
    if ((slot_offset % 16) != 0 || (bytes % 16) != 0) {
        return fail(err, err_size, "block at %llu (+%llu) is not vec4 aligned",
                    (unsigned long long) slot_offset, (unsigned long long) bytes);
    }
    VkResult r = vkResetFences(probe->dev->device, 1, &probe->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetFences (probe)", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(probe->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (probe)", r);
    }
    vkCmdBindPipeline(probe->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probe->pipe);
    vkCmdBindDescriptorSets(probe->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probe->pl, 0, 1,
                            &probe->set, 0, nullptr);
    const uint32_t base = (uint32_t) (slot_offset / 16);
    const uint32_t n    = (uint32_t) (bytes / 16);
    uint32_t pc[2] = { base, n };
    vkCmdPushConstants(probe->cmd, probe->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    // 256 threads x 4 vec4 per thread keeps one workgroup near a full wave.
    uint32_t groups = (n + 1023u) / 1024u;
    if (groups == 0) {
        groups = 1;
    }
    vkCmdDispatch(probe->cmd, groups, 1, 1);
    r = vkEndCommandBuffer(probe->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (probe)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &probe->cmd;
    r = vkQueueSubmit(probe->dev->compute_queue, 1, &si, probe->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (probe)", r);
    }
    r = vkWaitForFences(probe->dev->device, 1, &probe->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (probe)", r);
    }
    return 0;
}

#else // QRT_HAVE_VULKAN

struct qrt_prefill_probe { int unused; };

int qrt_prefill_probe_open(qrt_vk_dev * dev, const char * spv_path, qrt_prefill * p,
                           qrt_prefill_probe ** out, char * err, size_t err_size) {
    (void) dev;
    (void) spv_path;
    (void) p;
    *out = NULL;
    return fail(err, err_size, "this build has no Vulkan");
}

void qrt_prefill_probe_close(qrt_vk_dev * dev, qrt_prefill_probe * probe) {
    (void) dev;
    (void) probe;
}

int qrt_prefill_probe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                              uint64_t slot_offset, uint64_t bytes,
                              char * err, size_t err_size) {
    (void) user; (void) layer; (void) expert; (void) buf;
    (void) slot_offset; (void) bytes;
    return fail(err, err_size, "this build has no Vulkan");
}

#endif // QRT_HAVE_VULKAN
