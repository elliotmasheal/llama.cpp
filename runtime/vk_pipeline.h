#pragma once

#include "vk_dev.h"

#ifdef QRT_HAVE_VULKAN

// Shader, pipeline, layout and descriptor boilerplate the kernel modules share. Each
// returns an error string instead of aborting (design 9); no module keeps its own copy.

int qrt_vk_read_spv(const char * dir, const char * name, unsigned char ** out,
                    size_t * out_bytes, char * err, size_t err_size);
VkShaderModule qrt_vk_load_shader(VkDevice dev, const char * dir, const char * name,
                                  char * err, size_t err_size);

// Spec constants are a flat uint32 list in constantID order, at most eight.
int qrt_vk_make_pipeline_from_module(qrt_vk_dev * dev, VkShaderModule mod,
                                     VkPipelineLayout layout, const uint32_t * spec,
                                     uint32_t n_spec, VkPipeline * out,
                                     char * err, size_t err_size);
int qrt_vk_make_pipeline(qrt_vk_dev * dev, VkPipelineLayout layout, const char * spv_dir,
                         const char * name, const uint32_t * spec, uint32_t n_spec,
                         VkPipeline * out, char * err, size_t err_size);
int qrt_vk_make_set_layout(qrt_vk_dev * dev, uint32_t bindings, VkDescriptorSetLayout * out,
                           char * err, size_t err_size);
int qrt_vk_make_layout(qrt_vk_dev * dev, VkDescriptorSetLayout dsl, uint32_t push_bytes,
                       VkPipelineLayout * out, char * err, size_t err_size);
int qrt_vk_write_set(qrt_vk_dev * dev, VkDescriptorSet set, const VkBuffer * bufs,
                     uint32_t n);

#endif  // QRT_HAVE_VULKAN
