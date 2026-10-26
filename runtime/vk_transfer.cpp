// Host <-> device transfer: the staging ring, its mapping, and the copies that run through
// it. It is a separate translation unit from the device and budget code because it only
// needs the buffer registry to prove a copy stays inside a buffer.

#include "vk_transfer.h"

#include "vk_internal.h"

#include "qrt_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

// Unlike pick_memory_type this never falls back to a type that cannot be mapped:
// mapping non-host-visible memory is a validation error, and on Windows it faults.
// Cached system memory wins because the CPU reads the mapping (RESULTS 35): reading
// a write-combined mapping runs at 0.21 GB/s, the cached one at PCIe speed, and the
// H2D rate is the same for both. *coherent is 0 when only a non-coherent type fits,
// so the caller flushes and invalidates explicitly.

uint32_t pick_host_type(const qrt_vk_dev * dev, uint32_t bits, int * coherent) {
    static const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags tries[] = {
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        host,
    };
    for (int t = 0; t < 4; ++t) {
        for (uint32_t i = 0; i < dev->mem.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (dev->mem.memoryTypes[i].propertyFlags & tries[t]) == tries[t]) {
                *coherent = (tries[t] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0;
                return i;
            }
        }
    }
    *coherent = 0;
    return UINT32_MAX;
}

} // namespace

struct qrt_vk_staging {
    VkBuffer        buffer;
    VkDeviceMemory  mem;
    void *          map;
    uint64_t        bytes;       // capacity asked for
    int             coherent;
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkFence         fence;
    int             pending;     // a copy was submitted and not waited yet
    char            label[96];   // memory type and heap, for the io check
};

const char * qrt_vk_staging_label(const qrt_vk_staging * stg) {
    return stg != NULL ? stg->label : "";
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    if (stg == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (stg->pending) {
            char err[128];
            qrt_vk_staging_wait(dev, stg, err, sizeof(err));
        }
        if (stg->fence != VK_NULL_HANDLE) {
            vkDestroyFence(dev->device, stg->fence, nullptr);
        }
        // The pool owns the command buffer, so it is never freed on its own.
        if (stg->pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, stg->pool, nullptr);
        }
        if (stg->map != NULL && stg->mem != VK_NULL_HANDLE) {
            vkUnmapMemory(dev->device, stg->mem);
        }
    }
    if (stg->mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, stg->buffer, stg->mem);
    }
    delete stg;
}

uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg) {
    return stg != NULL ? stg->bytes : 0;
}

static int staging_fail(qrt_vk_dev * dev, char * err, size_t err_size, const char * what, VkResult r) {
    ++dev->io_failures;
    return fail(err, err_size, "%s failed: VkResult %d", what, (int) r);
}

int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size) {
    *out = NULL;
    if (bytes == 0) {
        return fail(err, err_size, "staging buffer of 0 bytes");
    }
    qrt_vk_staging * stg = new qrt_vk_staging();
    memset(stg, 0, sizeof(*stg));
    stg->bytes = bytes;

    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = QRT_VK_USAGE_TRANSFER_SRC | QRT_VK_USAGE_TRANSFER_DST;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(dev->device, &bci, nullptr, &stg->buffer);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateBuffer (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->device, stg->buffer, &req);
    const uint32_t type = pick_host_type(dev, req.memoryTypeBits, &stg->coherent);
    if (type == UINT32_MAX) {
        fail(err, err_size, "no host-visible memory type for a %llu MiB staging buffer",
             (unsigned long long) (bytes / QRT_VK_MIB));
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = type;
    r = vkAllocateMemory(dev->device, &mai, nullptr, &stg->mem);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkAllocateMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    r = vkBindBufferMemory(dev->device, stg->buffer, stg->mem, 0);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkBindBufferMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    r = vkMapMemory(dev->device, stg->mem, 0, VK_WHOLE_SIZE, 0, &stg->map);
    if (r != VK_SUCCESS || stg->map == NULL) {
        stg->map = NULL;
        staging_fail(dev, err, err_size, "vkMapMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    {
        const VkMemoryType * mt = &dev->mem.memoryTypes[type];
        uint32_t heap = 0;
        for (uint32_t h = 0; h < dev->mem.memoryHeapCount; ++h) {
            if (mt->heapIndex == h) {
                heap = h;
            }
        }
        snprintf(stg->label, sizeof(stg->label), "type=%u heap=%u host=%d coherent=%d device_local=%d",
                 type, heap, 1, stg->coherent,
                 (mt->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0);
    }

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->transfer_family;
    r = vkCreateCommandPool(dev->device, &pci, nullptr, &stg->pool);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateCommandPool (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = stg->pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &stg->cmd);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkAllocateCommandBuffers (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &stg->fence);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateFence (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    *out = stg;
    return 0;
}

// One copy between the mapped staging bytes and a device buffer, then a fence
// wait. Callers own the host-side memcpy and the cache maintenance.
static int staging_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer buf, uint64_t offset,
                          uint64_t bytes, int to_device, char * err, size_t err_size) {
    if (stg->map == NULL) {
        ++dev->io_failures;
        return fail(err, err_size, "staging buffer is not mapped");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "copy of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    if (qrt_vk_buf_reg_check(buf, offset, bytes, err, err_size) != 0) {
        ++dev->io_failures;
        return -1;
    }
    if (stg->pending) {
        ++dev->io_failures;
        return fail(err, err_size, "a copy is already in flight on this staging buffer");
    }
    VkResult r = vkResetFences(dev->device, 1, &stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkResetFences", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(stg->cmd, &bi);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkBeginCommandBuffer", r);
    }
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.size      = bytes;
    region.srcOffset = to_device ? 0 : offset;
    region.dstOffset = to_device ? offset : 0;
    vkCmdCopyBuffer(stg->cmd, to_device ? stg->buffer : buf, to_device ? buf : stg->buffer,
                    1, &region);
    r = vkEndCommandBuffer(stg->cmd);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkEndCommandBuffer", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &stg->cmd;
    r = vkQueueSubmit(dev->transfer_queue, 1, &si, stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkQueueSubmit", r);
    }
    stg->pending = 1;
    return 0;
}

int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size) {
    if (stg == NULL || !stg->pending) {
        return 0;
    }
    stg->pending = 0;
    const VkResult r = vkWaitForFences(dev->device, 1, &stg->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkWaitForFences", r);
    }
    return 0;
}

static int staging_copy(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer buf, uint64_t offset,
                        uint64_t bytes, int to_device, char * err, size_t err_size) {
    if (staging_submit(dev, stg, buf, offset, bytes, to_device, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_staging_wait(dev, stg, err, err_size);
}

int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size) {
    if (stg == NULL || src == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer, a source and a size");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "upload of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    memcpy(stg->map, src, (size_t) bytes);
    return qrt_vk_upload_mapped(dev, stg, dst, dst_offset, bytes, err, err_size);
}

int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size) {
    if (stg == NULL || dst == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "download needs a staging buffer, a destination and a size");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "download of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    if (qrt_vk_download_mapped(dev, stg, src, src_offset, bytes, err, err_size) != 0) {
        return -1;
    }
    memcpy(dst, stg->map, (size_t) bytes);
    return 0;
}

void * qrt_vk_staging_map(qrt_vk_staging * stg) {
    return stg != NULL ? stg->map : NULL;
}

int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer and a size");
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult fr = vkFlushMappedMemoryRanges(dev->device, 1, &range);
        if (fr != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkFlushMappedMemoryRanges", fr);
        }
    }
    return staging_copy(dev, stg, dst, dst_offset, bytes, 1, err, err_size);
}

int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size) {
    if (stg == NULL || regions == NULL || count == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "region upload needs a staging buffer and regions");
    }
    if (stg->map == NULL || stg->pending) {
        ++dev->io_failures;
        return fail(err, err_size, "the staging buffer is not free for a region upload");
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (regions[i].src == NULL || regions[i].bytes == 0) {
            ++dev->io_failures;
            return fail(err, err_size, "region %u has no source or size", i);
        }
        if (regions[i].bytes > stg->bytes - total ||
            qrt_vk_buf_reg_check(regions[i].dst, regions[i].dst_offset, regions[i].bytes, err,
                          err_size) != 0) {
            ++dev->io_failures;
            return -1;
        }
        total += regions[i].bytes;
    }
    // Pack the sources only when they are not already in packed order: the cache path
    // reads experts straight to their packed offsets, so this is the common case.
    {
        const unsigned char * base = (const unsigned char *) stg->map;
        uint64_t packed = 0;
        int in_place = 1;
        for (uint32_t i = 0; i < count; ++i) {
            if (regions[i].src != base + packed) {
                in_place = 0;
                break;
            }
            packed += regions[i].bytes;
        }
        if (!in_place) {
            std::vector<unsigned char> tmp((size_t) total);
            packed = 0;
            for (uint32_t i = 0; i < count; ++i) {
                memcpy(tmp.data() + packed, regions[i].src, (size_t) regions[i].bytes);
                packed += regions[i].bytes;
            }
            memcpy(stg->map, tmp.data(), (size_t) total);
        }
    }
    VkResult r = vkResetFences(dev->device, 1, &stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkResetFences", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(stg->cmd, &bi);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkBeginCommandBuffer", r);
    }
    uint64_t src_off = 0;
    for (uint32_t i = 0; i < count; ++i) {
        VkBufferCopy region;
        memset(&region, 0, sizeof(region));
        region.srcOffset = src_off;
        region.dstOffset = regions[i].dst_offset;
        region.size      = regions[i].bytes;
        vkCmdCopyBuffer(stg->cmd, stg->buffer, regions[i].dst, 1, &region);
        src_off += regions[i].bytes;
    }
    r = vkEndCommandBuffer(stg->cmd);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkEndCommandBuffer", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &stg->cmd;
    r = vkQueueSubmit(dev->transfer_queue, 1, &si, stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkQueueSubmit", r);
    }
    stg->pending = 1;
    return qrt_vk_staging_wait(dev, stg, err, err_size);
}

int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer and a size");
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult fr = vkFlushMappedMemoryRanges(dev->device, 1, &range);
        if (fr != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkFlushMappedMemoryRanges", fr);
        }
    }
    return staging_submit(dev, stg, dst, dst_offset, bytes, 1, err, err_size);
}

int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "download needs a staging buffer and a size");
    }
    if (staging_copy(dev, stg, src, src_offset, bytes, 0, err, err_size) != 0) {
        return -1;
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult ir = vkInvalidateMappedMemoryRanges(dev->device, 1, &range);
        if (ir != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkInvalidateMappedMemoryRanges", ir);
        }
    }
    return 0;
}

#else // QRT_HAVE_VULKAN

struct qrt_vk_staging;

int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size) {
    (void) dev;
    (void) bytes;
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    (void) dev;
    (void) stg;
}

uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg) {
    (void) stg;
    return 0;
}

const char * qrt_vk_staging_label(const qrt_vk_staging * stg) {
    (void) stg;
    return "";
}

void * qrt_vk_staging_map(qrt_vk_staging * stg) {
    (void) stg;
    return NULL;
}

int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) src;
    (void) src_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) regions;
    (void) count;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) src;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) src;
    (void) src_offset;
    (void) dst;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
