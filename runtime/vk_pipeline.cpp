// Shader, pipeline, layout and descriptor boilerplate shared by the kernel modules. It only
// needs the device handle, so it is its own translation unit.

#include "vk_pipeline.h"

#include "qrt_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QRT_HAVE_VULKAN

int qrt_vk_read_spv(const char * dir, const char * name, unsigned char ** out,
                    size_t * out_bytes, char * err, size_t err_size) {
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
    const long bytes = ftell(f);
    if (bytes <= 0) {
        fclose(f);
        return fail(err, err_size, "'%s' is empty", path);
    }
    rewind(f);
    unsigned char * buf = (unsigned char *) malloc((size_t) bytes);
    if (buf == NULL) {
        fclose(f);
        return fail(err, err_size, "out of memory for %ld bytes of SPIR-V", bytes);
    }
    const size_t got = fread(buf, 1, (size_t) bytes, f);
    fclose(f);
    if (got != (size_t) bytes) {
        free(buf);
        return fail(err, err_size, "short read on '%s'", path);
    }
    *out = buf;
    *out_bytes = got;
    return 0;
}

VkShaderModule qrt_vk_load_shader(VkDevice dev, const char * dir, const char * name,
                                  char * err, size_t err_size) {
    unsigned char * code = NULL;
    size_t code_bytes = 0;
    if (qrt_vk_read_spv(dir, name, &code, &code_bytes, err, err_size) != 0) {
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

int qrt_vk_make_pipeline_from_module(qrt_vk_dev * dev, VkShaderModule mod,
                                     VkPipelineLayout layout, const uint32_t * spec,
                                     uint32_t n_spec, VkPipeline * out,
                                     char * err, size_t err_size) {
    if (n_spec > 8) {
        return fail(err, err_size, "%u spec constants is above the eight the helper carries",
                    n_spec);
    }
    VkSpecializationMapEntry sme[8];
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
    const VkResult r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci,
                                                nullptr, out);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateComputePipelines", r);
    }
    return 0;
}

int qrt_vk_make_pipeline(qrt_vk_dev * dev, VkPipelineLayout layout, const char * spv_dir,
                         const char * name, const uint32_t * spec, uint32_t n_spec,
                         VkPipeline * out, char * err, size_t err_size) {
    VkShaderModule mod = qrt_vk_load_shader(dev->device, spv_dir, name, err, err_size);
    if (mod == VK_NULL_HANDLE) {
        return -1;
    }
    const int rc = qrt_vk_make_pipeline_from_module(dev, mod, layout, spec, n_spec, out, err,
                                                    err_size);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    return rc;
}

int qrt_vk_make_set_layout(qrt_vk_dev * dev, uint32_t bindings, VkDescriptorSetLayout * out,
                           char * err, size_t err_size) {
    if (bindings > 8) {
        return fail(err, err_size, "%u bindings is above the eight the layouts carry", bindings);
    }
    VkDescriptorSetLayoutBinding b[8];
    memset(b, 0, sizeof(b));
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
        return vk_fail(err, err_size, "vkCreateDescriptorSetLayout", r);
    }
    return 0;
}

int qrt_vk_make_layout(qrt_vk_dev * dev, VkDescriptorSetLayout dsl, uint32_t push_bytes,
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
        return vk_fail(err, err_size, "vkCreatePipelineLayout", r);
    }
    return 0;
}

int qrt_vk_write_set(qrt_vk_dev * dev, VkDescriptorSet set, const VkBuffer * bufs,
                     uint32_t n) {
    if (n > 8) {
        return -1;
    }
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

#endif // QRT_HAVE_VULKAN
