// The layer pass: one command buffer per ring that every module of a layer records into, so
// a layer costs two submits instead of one per module (design 4.6). It shares no helpers
// with the device and transfer code, so it is its own translation unit.

#include "vk_pass.h"

#include "qrt_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QRT_HAVE_VULKAN

struct qrt_vk_cmd {
    qrt_vk_dev *  dev;
    uint32_t      rings;
    VkCommandPool pool;
    VkCommandBuffer * cmd;
    VkFence *     fence;
    int *         pending;
    int *         recording;    // a begin without a submit: recoverable on the next begin
    uint32_t      cur;
    uint32_t      dispatches;
};

int qrt_vk_cmd_create(qrt_vk_dev * dev, uint32_t rings, qrt_vk_cmd ** out,
                      char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || rings == 0) {
        return fail(err, err_size, "a pass needs a device and at least one ring");
    }
    qrt_vk_cmd * c = (qrt_vk_cmd *) calloc(1, sizeof(qrt_vk_cmd));
    if (c == NULL) {
        return fail(err, err_size, "out of memory for the pass");
    }
    c->dev   = dev;
    c->rings = rings;
    c->cmd       = (VkCommandBuffer *) calloc(rings, sizeof(VkCommandBuffer));
    c->fence     = (VkFence *) calloc(rings, sizeof(VkFence));
    c->pending   = (int *) calloc(rings, sizeof(int));
    c->recording = (int *) calloc(rings, sizeof(int));
    if (c->cmd == NULL || c->fence == NULL || c->pending == NULL || c->recording == NULL) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "out of memory for the pass rings");
    }
    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->compute_family;
    if (vkCreateCommandPool(dev->device, &pci, nullptr, &c->pool) != VK_SUCCESS) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "vkCreateCommandPool (pass) failed");
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = c->pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = rings;
    VkResult r = vkAllocateCommandBuffers(dev->device, &cbai, c->cmd);
    if (r != VK_SUCCESS) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "vkAllocateCommandBuffers (pass) failed: VkResult %d",
                    (int) r);
    }
    for (uint32_t i = 0; i < rings; ++i) {
        VkFenceCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev->device, &fci, nullptr, &c->fence[i]) != VK_SUCCESS) {
            qrt_vk_cmd_destroy(dev, c);
            return fail(err, err_size, "vkCreateFence (pass) failed");
        }
    }
    *out = c;
    return 0;
}

void qrt_vk_cmd_destroy(qrt_vk_dev * dev, qrt_vk_cmd * c) {
    if (c == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (c->cmd != NULL && c->fence != NULL && c->pending != NULL) {
            for (uint32_t i = 0; i < c->rings; ++i) {
                if (c->pending[i]) {
                    char err[128];
                    qrt_vk_cmd_wait(c, i, err, sizeof(err));
                }
                if (c->fence[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(dev->device, c->fence[i], nullptr);
                }
            }
        }
        if (c->pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, c->pool, nullptr);
        }
    }
    free(c->cmd);
    free(c->fence);
    free(c->pending);
    free(c->recording);
    free(c);
}

uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd * c) { return c != NULL ? c->rings : 0; }

int qrt_vk_cmd_wait(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to wait for", ring);
    }
    if (!c->pending[ring]) {
        return 0;
    }
    c->pending[ring] = 0;
    if (vkWaitForFences(c->dev->device, 1, &c->fence[ring], VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        ++c->dev->io_failures;
        return fail(err, err_size, "vkWaitForFences (pass) failed");
    }
    return 0;
}

int qrt_vk_cmd_begin(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to begin", ring);
    }
    if (qrt_vk_cmd_wait(c, ring, err, err_size) != 0) {
        return -1;
    }
    if (c->recording[ring]) {
        // A failed record left the buffer open; close it so the ring stays usable.
        vkEndCommandBuffer(c->cmd[ring]);
        c->recording[ring] = 0;
    }
    if (vkResetFences(c->dev->device, 1, &c->fence[ring]) != VK_SUCCESS) {
        return fail(err, err_size, "vkResetFences (pass) failed");
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(c->cmd[ring], &bi);
    if (r != VK_SUCCESS) {
        return fail(err, err_size, "vkBeginCommandBuffer (pass) failed: VkResult %d", (int) r);
    }
    c->cur = ring;
    c->recording[ring] = 1;
    return 0;
}

VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd * c, uint32_t ring) {
    if (c == NULL || ring >= c->rings) {
        return VK_NULL_HANDLE;
    }
    return c->cmd[ring];
}

void qrt_vk_cmd_record_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

int qrt_vk_cmd_barrier(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u for a barrier", ring);
    }
    qrt_vk_cmd_record_barrier(c->cmd[ring]);
    return 0;
}

int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u for a barrier", ring);
    }
    qrt_vk_cmd_record_barrier_transfer(c->cmd[ring]);
    return 0;
}

void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer cmd) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

void qrt_vk_cmd_count(qrt_vk_cmd * c, uint32_t dispatches) {
    if (c != NULL) {
        c->dispatches += dispatches;
    }
}

uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd * c) {
    return c != NULL ? c->dispatches : 0;
}

int qrt_vk_cmd_submit(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to submit", ring);
    }
    const VkResult r = vkEndCommandBuffer(c->cmd[ring]);
    if (r != VK_SUCCESS) {
        return fail(err, err_size, "vkEndCommandBuffer (pass) failed: VkResult %d", (int) r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &c->cmd[ring];
    if (vkQueueSubmit(c->dev->compute_queue, 1, &si, c->fence[ring]) != VK_SUCCESS) {
        ++c->dev->io_failures;
        return fail(err, err_size, "vkQueueSubmit (pass) failed");
    }
    c->pending[ring] = 1;
    c->recording[ring] = 0;
    return 0;
}

#else // QRT_HAVE_VULKAN

struct qrt_vk_cmd { int unused; };
int qrt_vk_cmd_create(qrt_vk_dev *, uint32_t, qrt_vk_cmd ** out, char * err, size_t err_size) {
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
void qrt_vk_cmd_destroy(qrt_vk_dev *, qrt_vk_cmd *) {}
uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd *) { return 0; }
int qrt_vk_cmd_begin(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd *, uint32_t) { return VK_NULL_HANDLE; }
int qrt_vk_cmd_barrier(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer) {}
void qrt_vk_cmd_record_barrier(VkCommandBuffer) {}
void qrt_vk_cmd_count(qrt_vk_cmd *, uint32_t) {}
uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd *) { return 0; }
int qrt_vk_cmd_submit(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
int qrt_vk_cmd_wait(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
