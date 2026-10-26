// RMS norm + Q8_1 activation packing (M2-6 layer chain, first piece): the kernel launch
// and the double precision reference. The buffers are the caller's, so the planes this
// writes are exactly the ones the dense GEMM reads.

#include "norm.h"
#include "vk_pipeline.h"

#include "qrt_half.h"
#include "qrt_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

#define QRT_NORM_RINGS 2
#define QRT_NORM_SETS_PER_RING 64

// Must match the push constant block of kernels/qrt_norm.comp in order and size.
struct qrt_norm_pc {
    uint32_t k;
    uint32_t xbase;
    uint32_t obase;
    uint32_t qsbase;
    uint32_t dsbase;
    float    eps;
    uint32_t do_quant;
    uint32_t do_norm;
    uint32_t do_out;
};

struct qrt_norm_ring {
    VkDescriptorPool pool;     // reset by attach/record: one set per dispatch (DECISIONS 65.2)
    uint32_t        sets_used;
    VkCommandPool   cmd_pool;
    VkCommandBuffer cmd;
    VkFence         fence;
    int             busy;
};

struct qrt_norm {
    qrt_vk_dev *      dev;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout  layout;
    VkPipeline        pipe;
    qrt_norm_ring     ring[QRT_NORM_RINGS];
    VkCommandBuffer   ext;       // set by attach: the pass records into it
    double            last_ms;
};

namespace {

// One set per dispatch: a command buffer only references the set object, so rewriting a
// set would change every dispatch already recorded with it (DECISIONS 65.2).
int alloc_set(qrt_norm * n, uint32_t ring, const VkBuffer * bufs, VkDescriptorSet * out,
              char * err, size_t err_size) {
    const uint32_t slots = QRT_NORM_SETS_PER_RING;
    if (n->ring[ring].sets_used >= slots) {
        return fail(err, err_size, "norm ring %u already holds %u dispatches", ring, slots);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = n->ring[ring].pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &n->dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkResult r = vkAllocateDescriptorSets(n->dev->device, &dsai, &set);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (norm)", r);
    }
    qrt_vk_write_set(n->dev, set, bufs, 5);
    ++n->ring[ring].sets_used;
    *out = set;
    return 0;
}

} // namespace

uint32_t qrt_norm_rings(void) { return QRT_NORM_RINGS; }

double qrt_norm_last_ms(const qrt_norm * n) { return n == NULL ? 0.0 : n->last_ms; }

int qrt_norm_open(qrt_vk_dev * dev, const char * spv_dir, qrt_norm ** out,
                  char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL) {
        return fail(err, err_size, "norm needs a device and a SPIR-V directory");
    }
    qrt_norm * n = (qrt_norm *) calloc(1, sizeof(qrt_norm));
    if (n == NULL) {
        return fail(err, err_size, "out of memory for the norm module");
    }
    n->dev = dev;

    if (qrt_vk_make_set_layout(dev, 5, &n->dsl, err, err_size) != 0) {
        qrt_norm_close(dev, n);
        return -1;
    }
    if (qrt_vk_make_layout(dev, n->dsl, sizeof(qrt_norm_pc), &n->layout, err, err_size) != 0) {
        qrt_norm_close(dev, n);
        return -1;
    }
    if (qrt_vk_make_pipeline(dev, n->layout, spv_dir, "qrt_norm.spv", nullptr, 0, &n->pipe,
                             err, err_size) != 0) {
        qrt_norm_close(dev, n);
        return -1;
    }

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 5 * QRT_NORM_SETS_PER_RING;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = QRT_NORM_SETS_PER_RING;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;

    for (uint32_t i = 0; i < QRT_NORM_RINGS; ++i) {
        qrt_norm_ring * g = &n->ring[i];
        if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &g->pool) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkCreateDescriptorPool (norm) failed");
        }
        VkCommandPoolCreateInfo cpi;
        memset(&cpi, 0, sizeof(cpi));
        cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = dev->compute_family;
        if (vkCreateCommandPool(dev->device, &cpi, nullptr, &g->cmd_pool) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkCreateCommandPool (norm) failed");
        }
        VkCommandBufferAllocateInfo cbai;
        memset(&cbai, 0, sizeof(cbai));
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = g->cmd_pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev->device, &cbai, &g->cmd) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkAllocateCommandBuffers (norm) failed");
        }
        VkFenceCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev->device, &fci, nullptr, &g->fence) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkCreateFence (norm) failed");
        }
    }
    *out = n;
    return 0;
}

void qrt_norm_close(qrt_vk_dev * dev, qrt_norm * n) {
    if (n == NULL) {
        return;
    }
    if (dev != NULL) {
        for (uint32_t i = 0; i < QRT_NORM_RINGS; ++i) {
            if (n->ring[i].fence != VK_NULL_HANDLE) {
                vkDestroyFence(dev->device, n->ring[i].fence, nullptr);
            }
            if (n->ring[i].cmd_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(dev->device, n->ring[i].cmd_pool, nullptr);
            }
            if (n->ring[i].pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(dev->device, n->ring[i].pool, nullptr);
            }
        }
        if (n->pipe != VK_NULL_HANDLE)   vkDestroyPipeline(dev->device, n->pipe, nullptr);
        if (n->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev->device, n->layout, nullptr);
        if (n->dsl != VK_NULL_HANDLE)    vkDestroyDescriptorSetLayout(dev->device, n->dsl, nullptr);
    }
    free(n);
}

int qrt_norm_attach(qrt_norm * n, uint32_t ring, VkCommandBuffer cmd, char * err,
                    size_t err_size) {
    if (n == NULL || cmd == VK_NULL_HANDLE) {
        return fail(err, err_size, "norm attach needs the module and a command buffer");
    }
    if (ring >= QRT_NORM_RINGS) {
        return fail(err, err_size, "norm ring %u is above the %u that exist", ring,
                    (uint32_t) QRT_NORM_RINGS);
    }
    if (qrt_norm_wait(n, ring, err, err_size) != 0) {
        return -1;
    }
    const VkResult r = vkResetDescriptorPool(n->dev->device, n->ring[ring].pool, 0);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetDescriptorPool (norm)", r);
    }
    n->ring[ring].sets_used = 0;
    n->ext = cmd;
    return 0;
}

void qrt_norm_detach(qrt_norm * n) {
    if (n != NULL) {
        n->ext = VK_NULL_HANDLE;
    }
}

int qrt_norm_record(qrt_norm * n, uint32_t ring, const qrt_norm_args * a,
                    char * err, size_t err_size) {
    if (n == NULL || a == NULL) {
        return fail(err, err_size, "norm record needs the module and args");
    }
    if (ring >= QRT_NORM_RINGS) {
        return fail(err, err_size, "norm ring %u is above the %u that exist", ring,
                    (uint32_t) QRT_NORM_RINGS);
    }
    if (a->tokens == 0 || a->k == 0 || (a->k % 32u) != 0) {
        return fail(err, err_size, "norm shape tokens=%u k=%u does not fit (k must be a "
                    "multiple of 32)", a->tokens, a->k);
    }
    if (a->x == VK_NULL_HANDLE || a->w == VK_NULL_HANDLE ||
        (a->out == VK_NULL_HANDLE && a->no_out == 0)) {
        return fail(err, err_size, "norm needs x, w and out buffers");
    }
    if (a->quant != 0 && (a->qs == VK_NULL_HANDLE || a->ds == VK_NULL_HANDLE)) {
        return fail(err, err_size, "norm quant needs the qs and ds planes");
    }
    qrt_norm_ring * g = &n->ring[ring];
    VkCommandBuffer cmd = n->ext != VK_NULL_HANDLE ? n->ext : g->cmd;
    VkBuffer bufs[5] = { a->x, a->w, a->no_out != 0 ? a->x : a->out,
                         a->quant != 0 ? a->qs : a->out,
                         a->quant != 0 ? a->ds : a->out };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (n->ext == VK_NULL_HANDLE) {
        // One-shot: the ring was waited, so its pool can be handed back and refilled.
        if (g->busy) {
            return fail(err, err_size, "norm ring %u is still in flight; wait before recording",
                        ring);
        }
        const VkResult pr = vkResetDescriptorPool(n->dev->device, g->pool, 0);
        if (pr != VK_SUCCESS) {
            return vk_fail(err, err_size, "vkResetDescriptorPool (norm)", pr);
        }
        g->sets_used = 0;
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        const VkResult r = vkBeginCommandBuffer(g->cmd, &bi);
        if (r != VK_SUCCESS) {
            return vk_fail(err, err_size, "vkBeginCommandBuffer (norm)", r);
        }
    }
    if (alloc_set(n, ring, bufs, &set, err, err_size) != 0) {
        return -1;
    }
    qrt_norm_pc pc;
    memset(&pc, 0, sizeof(pc));
    pc.k        = a->k;
    pc.xbase    = a->xbase;
    pc.obase    = a->obase;
    pc.qsbase   = a->qsbase;
    pc.dsbase   = a->dsbase;
    pc.eps      = a->eps;
    pc.do_quant = a->quant != 0 ? 1u : 0u;
    pc.do_norm  = a->no_norm != 0 ? 0u : 1u;
    pc.do_out   = a->no_out != 0 ? 0u : 1u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, n->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, n->layout, 0, 1, &set,
                            0, nullptr);
    vkCmdPushConstants(cmd, n->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, a->tokens, 1, 1);
    if (n->ext == VK_NULL_HANDLE) {
        const VkResult r = vkEndCommandBuffer(g->cmd);
        if (r != VK_SUCCESS) {
            return vk_fail(err, err_size, "vkEndCommandBuffer (norm)", r);
        }
    }
    return 0;
}

int qrt_norm_submit(qrt_norm * n, uint32_t ring, char * err, size_t err_size) {
    if (n == NULL || ring >= QRT_NORM_RINGS) {
        return fail(err, err_size, "norm submit ring %u does not exist", ring);
    }
    if (n->ext != VK_NULL_HANDLE) {
        return fail(err, err_size, "norm ring %u is attached to a pass; the pass submits",
                    ring);
    }
    qrt_norm_ring * g = &n->ring[ring];
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &g->cmd;
    const VkResult r = vkQueueSubmit(n->dev->compute_queue, 1, &si, g->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (norm)", r);
    }
    g->busy = 1;
    return 0;
}

int qrt_norm_wait(qrt_norm * n, uint32_t ring, char * err, size_t err_size) {
    if (n == NULL || ring >= QRT_NORM_RINGS) {
        return fail(err, err_size, "norm wait ring %u does not exist", ring);
    }
    qrt_norm_ring * g = &n->ring[ring];
    if (!g->busy) {
        return 0;
    }
    if (vkWaitForFences(n->dev->device, 1, &g->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return fail(err, err_size, "vkWaitForFences (norm) failed");
    }
    g->busy = 0;
    return vkResetFences(n->dev->device, 1, &g->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (norm) failed");
}

int qrt_norm_run(qrt_norm * n, const qrt_norm_args * a, char * err, size_t err_size) {
    if (qrt_norm_record(n, 0, a, err, err_size) != 0) {
        return -1;
    }
    const double t0 = now_ms();
    if (qrt_norm_submit(n, 0, err, err_size) != 0 ||
        qrt_norm_wait(n, 0, err, err_size) != 0) {
        return -1;
    }
    n->last_ms = now_ms() - t0;
    return 0;
}

#else   // QRT_HAVE_VULKAN

struct qrt_norm { int unused; };

uint32_t qrt_norm_rings(void) { return 0; }
double qrt_norm_last_ms(const qrt_norm *) { return 0.0; }
int qrt_norm_open(qrt_vk_dev *, const char *, qrt_norm ** out, char * err, size_t err_size) {
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
void qrt_norm_close(qrt_vk_dev *, qrt_norm *) {}
int qrt_norm_record(qrt_norm *, uint32_t, const qrt_norm_args *, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
int qrt_norm_submit(qrt_norm *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
int qrt_norm_wait(qrt_norm *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
int qrt_norm_run(qrt_norm *, const qrt_norm_args *, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
int qrt_norm_attach(qrt_norm *, uint32_t, VkCommandBuffer, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan SDK");
    return -1;
}
void qrt_norm_detach(qrt_norm *) {}

#endif  // QRT_HAVE_VULKAN

// ---- double precision reference (built either way: the gate compares against it) ----

void qrt_norm_cpu(uint32_t tokens, uint32_t k, const float * x, const float * w, float eps,
                  float * out, signed char * qs_out, uint32_t * ds_out) {
    const uint32_t blocks = k / 32u;
    for (uint32_t t = 0; t < tokens; ++t) {
        const float * row = x + (size_t) t * k;
        double ss = 0.0;
        for (uint32_t i = 0; i < k; ++i) {
            ss += (double) row[i] * (double) row[i];
        }
        const double scale = 1.0 / sqrt(ss / (double) k + (double) eps);
        std::vector<float> norm((size_t) k);
        for (uint32_t i = 0; i < k; ++i) {
            norm[i] = (float) ((double) row[i] * scale * (double) w[i]);
        }
        if (out != NULL) {
            memcpy(out + (size_t) t * k, norm.data(), (size_t) k * sizeof(float));
        }
        if (qs_out == NULL || ds_out == NULL) {
            continue;
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            float mx = 0.0f;
            for (uint32_t i = 0; i < 32u; ++i) {
                mx = fmaxf(mx, fabsf(norm[b * 32u + i]));
            }
            const float d = mx / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            int sum = 0;
            signed char * q = qs_out + ((size_t) t * blocks + b) * 32u;
            for (uint32_t i = 0; i < 32u; ++i) {
                float y = norm[b * 32u + i] * inv;
                y = y > 127.0f ? 127.0f : (y < -127.0f ? -127.0f : y);
                const int v = (int) roundf(y);   // half away from zero, the same rule
                q[i] = (signed char) v;
                sum += v;
            }
            const uint32_t dh = (uint16_t) f32_to_f16_bits(d);
            ds_out[(size_t) t * blocks + b] = dh | ((uint32_t) (uint16_t) (sum + 32768) << 16);
        }
    }
}

void qrt_norm_quant_cpu(uint32_t tokens, uint32_t k, const float * x, float * out,
                        signed char * qs_out, uint32_t * ds_out) {
    const uint32_t blocks = k / 32u;
    for (uint32_t t = 0; t < tokens; ++t) {
        const float * row = x + (size_t) t * k;
        if (out != NULL) {
            memcpy(out + (size_t) t * k, row, (size_t) k * sizeof(float));
        }
        if (qs_out == NULL || ds_out == NULL) {
            continue;
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            float mx = 0.0f;
            for (uint32_t i = 0; i < 32u; ++i) {
                mx = fmaxf(mx, fabsf(row[b * 32u + i]));
            }
            const float d = mx / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            int sum = 0;
            signed char * q = qs_out + ((size_t) t * blocks + b) * 32u;
            for (uint32_t i = 0; i < 32u; ++i) {
                float y = row[b * 32u + i] * inv;
                y = y > 127.0f ? 127.0f : (y < -127.0f ? -127.0f : y);
                const int v = (int) roundf(y);
                q[i] = (signed char) v;
                sum += v;
            }
            const uint32_t dh = (uint16_t) f32_to_f16_bits(d);
            ds_out[(size_t) t * blocks + b] = dh | ((uint32_t) (uint16_t) (sum + 32768) << 16);
        }
    }
}
