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
typedef struct VkCommandBuffer_T *VkCommandBuffer;
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
    uint64_t alloc_peak_bytes;  // highest alloc_bytes seen, the VRAM peak
    uint32_t alloc_failures;    // counted fallbacks, never a crash (design 9)
    uint32_t cache_reductions;  // expert-cache steps given up by the planner
    uint32_t io_failures;       // upload/download fallbacks, same rule
    int      has_dot_product;   // VK_KHR_shader_integer_dot_product enabled
    uint32_t subgroup_size;     // VkPhysicalDeviceSubgroupProperties, 0 if unknown
    int      has_subgroup_arithmetic; // subgroup add is available to kernels
};

// The five buckets of design 5.5 plus what the planner had to give up.
struct qrt_vk_plan {
    uint64_t dense_bytes;
    uint64_t expert_cache_bytes;
    uint64_t kv_bytes;
    uint64_t compute_bytes;
    uint64_t draft_bytes;
    uint64_t total_bytes;
    uint32_t kv_tokens;             // resident KV window the plan grants (<= ctx)
    uint32_t cache_reductions;
    uint32_t cache_coverage_x100;   // expert cache / expert store, percent x 100
};

// The context the KV window may grow to (design 2.2: 256K for this model).
enum { QRT_KV_CTX_TOKENS = 262144 };

// Resident KV tokens that fit in the remaining VRAM: after dense/compute/draft and the
// expert-cache target, capped at n_ctx and rounded down to whole pages of
// page_tokens x kv_bytes_per_token. The cache keeps its priority, the KV window takes the
// headroom (M2-6a); 0 means not even one page fits.
uint32_t qrt_vk_kv_window_tokens(const qrt_vk_dev * dev, const qrt_config * cfg,
                                 uint64_t kv_bytes_per_token, uint32_t n_ctx);

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

// Several device copies from one staging buffer in one submit (M3-2a): the sources
// are packed back to back into the mapping by the caller, and the regions may target
// different buffers and offsets (the decode cache slots are not contiguous). The
// sum of bytes must fit the staging buffer. Waits for the submit before returning.
struct qrt_vk_copy_region {
    const void * src;          // inside the staging mapping
    VkBuffer     dst;
    uint64_t     dst_offset;
    uint64_t     bytes;
};
int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size);

// Fire and forget: the copy is in flight when this returns. Keep one staging
// buffer per copy in flight and wait before touching its mapping again; this is
// how M2-5 overlaps the upload of block n+1 with the compute of block n.
int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size);
int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size);

// One layer pass (design 4.6, DECISIONS 64.2/68): every module of a layer records its
// dispatches into the same command buffer, so the layer costs one submit instead of one
// per module. The modules keep their own descriptor pools; a module attaches to a ring,
// records, and the caller submits and waits once. Two rings so the next pass can be
// recorded while the current one is in flight.
struct qrt_vk_cmd;
int qrt_vk_cmd_create(qrt_vk_dev * dev, uint32_t rings, qrt_vk_cmd ** out,
                      char * err, size_t err_size);
void qrt_vk_cmd_destroy(qrt_vk_dev * dev, qrt_vk_cmd * c);
uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd * c);
// Waits a previous submission on this ring, resets its fence and begins the buffer.
int qrt_vk_cmd_begin(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd * c, uint32_t ring);
// Compute-to-compute memory barrier: the next dispatch sees the previous ones' writes.
int qrt_vk_cmd_barrier(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
// TRANSFER -> COMPUTE barrier, for a vkCmdFillBuffer the same pass recorded.
int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer cmd);
// Records a compute->compute barrier into a buffer another module is recording into.
void qrt_vk_cmd_record_barrier(VkCommandBuffer cmd);
// How many dispatches the pass recorded, for the readings.
void qrt_vk_cmd_count(qrt_vk_cmd * c, uint32_t dispatches);
uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd * c);
int qrt_vk_cmd_submit(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
int qrt_vk_cmd_wait(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);

void qrt_vk_print_info(const qrt_vk_dev * dev, FILE * f);
