#pragma once

#include "vk_dev.h"

#ifdef QRT_HAVE_VULKAN

// Internals shared by the two Vulkan translation units. qrt_vk_alloc registers every buffer
// it creates, and the transfer path refuses a copy that would run past a registered buffer.
// Unregistered handles are not rejected: modules may pass buffers from a pool this registry
// never saw (RESULTS 75.5 defect 2).

void qrt_vk_buf_reg_add(VkBuffer buf, uint64_t bytes);
void qrt_vk_buf_reg_del(VkBuffer buf);
int qrt_vk_buf_reg_check(VkBuffer buf, uint64_t offset, uint64_t bytes, char * err,
                         size_t err_size);

const uint64_t QRT_VK_MIB = 1024ull * 1024ull;

#endif  // QRT_HAVE_VULKAN
