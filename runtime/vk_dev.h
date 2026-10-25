#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "qrt_config.h"

#ifdef QRT_HAVE_VULKAN
#include <vulkan/vulkan.h>
#else
typedef struct VkBuffer_T *      VkBuffer;
typedef struct VkDeviceMemory_T *VkDeviceMemory;
typedef uint32_t                 VkBufferUsageFlags;
#ifndef VK_NULL_HANDLE
#define VK_NULL_HANDLE 0
#endif
#endif

// VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spelled here so the CLI builds without the SDK.
#define QRT_VK_USAGE_STORAGE 0x00000020u
// TRANSFER_SRC/DST: a device buffer that is a copy target for qrt_vk_upload must
// carry QRT_VK_USAGE_TRANSFER_DST, and one that qrt_vk_download reads must carry
// QRT_VK_USAGE_TRANSFER_SRC. Missing bits are a validation error, not a no-op.
#define QRT_VK_USAGE_TRANSFER_SRC 0x00000001u
#define QRT_VK_USAGE_TRANSFER_DST 0x00000002u

// Device, queues, memory types and the VRAM budget of design 5.5. Opening the
// device is a recoverable failure: no abort, no exit, a message in err instead.

struct qrt_vk_dev {
#ifdef QRT_HAVE_VULKAN
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          compute_queue;
    VkQueue          transfer_queue;
    uint32_t         compute_family;
    uint32_t         transfer_family;
    uint32_t         timestamp_bits;
    VkPhysicalDeviceMemoryProperties mem;
#endif
    char     name[256];
    uint64_t heap_bytes;     // largest device-local heap
    uint64_t budget_bytes;   // usable now, after the desktop and the tier cap
    uint64_t usage_bytes;    // VK_EXT_memory_budget usage of that heap, 0 if absent
    uint32_t tier_mib;       // injected cap, 0 = none
    int      has_memory_budget;
    uint32_t alloc_count;
    uint64_t alloc_bytes;
    uint32_t alloc_failures;    // counted fallbacks, never a crash (design 9)
    uint32_t cache_reductions;  // expert-cache steps given up by the planner
    uint32_t io_failures;       // upload/download fallbacks, same rule
    int      has_dot_product;   // VK_KHR_shader_integer_dot_product enabled
};

// The five buckets of design 5.5 plus what the planner had to give up.
struct qrt_vk_plan {
    uint64_t dense_bytes;
    uint64_t expert_cache_bytes;
    uint64_t kv_bytes;
    uint64_t compute_bytes;
    uint64_t draft_bytes;
    uint64_t total_bytes;
    uint32_t cache_reductions;
    uint32_t cache_coverage_x100;   // expert cache / expert store, percent x 100
};

// tier_mib caps the budget the way the fork's GGML_VK_VRAM_LIMIT_MIB does; 0 = auto.
int qrt_vk_open(uint32_t tier_mib, qrt_vk_dev ** out, char * err, size_t err_size);
void qrt_vk_close(qrt_vk_dev * dev);

// Re-reads VK_EXT_memory_budget (or the heap size) so a long run can re-plan.
int qrt_vk_query_budget(qrt_vk_dev * dev, char * err, size_t err_size);

// Fits design 5.5 into the budget: the expert cache shrinks in 256 MiB steps and
// every step is counted. expert_store_bytes is the routed_experts total (RESULTS 25).
int qrt_vk_plan_budget(qrt_vk_dev * dev, const qrt_config * cfg, uint64_t expert_store_bytes,
                qrt_vk_plan * out, char * err, size_t err_size);

int qrt_vk_alloc(qrt_vk_dev * dev, uint64_t bytes, VkBufferUsageFlags usage,
                 VkBuffer * buf, VkDeviceMemory * mem, char * err, size_t err_size);
void qrt_vk_free(qrt_vk_dev * dev, VkBuffer buf, VkDeviceMemory mem);

// Host-visible staging buffer plus its own command buffer and fence. Mapped once
// at creation; a copy is one vkCmdCopyBuffer and one submit+wait. M2-5 keeps two
// of them to overlap the upload of block n+1 with the compute of block n.
struct qrt_vk_staging;
int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size);
void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg);
uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg);
// One line saying which memory type and heap the staging buffer landed in.
const char * qrt_vk_staging_label(const qrt_vk_staging * stg);

// Copies host memory into a device buffer, or back out of it, and returns only
// after the fence signals. Both report failure instead of aborting (design 9).
int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size);
int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size);

// The variants the engine uses: the caller writes straight into the mapping (a
// file read lands there, so one memcpy disappears) and only the copy runs here.
void * qrt_vk_staging_map(qrt_vk_staging * stg);
int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size);
int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size);

// Fire and forget: the copy is in flight when this returns. Keep one staging
// buffer per copy in flight and wait before touching its mapping again; this is
// how M2-5 overlaps the upload of block n+1 with the compute of block n.
int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size);
int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size);

void qrt_vk_print_info(const qrt_vk_dev * dev, FILE * f);
