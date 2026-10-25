// RMS norm + Q8_1 activation packing (M2-6 layer chain, first piece): the kernel launch
// and the double precision reference. The buffers are the caller's, so the planes this
// writes are exactly the ones the dense GEMM reads.

#include "norm.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <vector>

#ifdef QRT_HAVE_VULKAN

#define QRT_NORM_RINGS 2

namespace {

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
    const long n = ftell(f);
    if (n <= 0) {
        fclose(f);
        return fail(err, err_size, "'%s' is empty", path);
    }
    rewind(f);
    unsigned char * buf = (unsigned char *) malloc((size_t) n);
    if (buf == NULL) {
        fclose(f);
        return fail(err, err_size, "out of memory for '%s'", path);
    }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) {
        free(buf);
        fclose(f);
        return fail(err, err_size, "short read on '%s'", path);
    }
    fclose(f);
    *out = buf;
    *out_bytes = (size_t) n;
    return 0;
}

double now_ms(void) {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

} // namespace

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
    VkDescriptorSet set;
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkFence         fence;
    int             busy;
};

struct qrt_norm {
    qrt_vk_dev *      dev;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout  layout;
    VkDescriptorPool  pool;
    VkPipeline        pipe;
    qrt_norm_ring     ring[QRT_NORM_RINGS];
    double            last_ms;
};

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

    VkDescriptorSetLayoutBinding b[5];
    memset(b, 0, sizeof(b));
    for (uint32_t i = 0; i < 5; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 5;
    dlci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &n->dsl) != VK_SUCCESS) {
        qrt_norm_close(dev, n);
        return fail(err, err_size, "vkCreateDescriptorSetLayout (norm) failed");
    }
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(qrt_norm_pc);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &n->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev->device, &plci, nullptr, &n->layout) != VK_SUCCESS) {
        qrt_norm_close(dev, n);
        return fail(err, err_size, "vkCreatePipelineLayout (norm) failed");
    }

    unsigned char * code = NULL;
    size_t code_bytes = 0;
    if (read_spv(spv_dir, "qrt_norm.spv", &code, &code_bytes, err, err_size) != 0) {
        qrt_norm_close(dev, n);
        return -1;
    }
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code_bytes;
    smci.pCode    = (const uint32_t *) code;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(dev->device, &smci, nullptr, &mod);
    free(code);
    if (r != VK_SUCCESS) {
        qrt_norm_close(dev, n);
        return vk_fail(err, err_size, "vkCreateShaderModule (norm)", r);
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout      = n->layout;
    r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &n->pipe);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        qrt_norm_close(dev, n);
        return vk_fail(err, err_size, "vkCreateComputePipelines (norm)", r);
    }

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 5 * QRT_NORM_RINGS;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = QRT_NORM_RINGS;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &n->pool) != VK_SUCCESS) {
        qrt_norm_close(dev, n);
        return fail(err, err_size, "vkCreateDescriptorPool (norm) failed");
    }

    for (uint32_t i = 0; i < QRT_NORM_RINGS; ++i) {
        qrt_norm_ring * g = &n->ring[i];
        VkDescriptorSetAllocateInfo dsai;
        memset(&dsai, 0, sizeof(dsai));
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = n->pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &n->dsl;
        if (vkAllocateDescriptorSets(dev->device, &dsai, &g->set) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkAllocateDescriptorSets (norm) failed");
        }
        VkCommandPoolCreateInfo cpi;
        memset(&cpi, 0, sizeof(cpi));
        cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = dev->compute_family;
        if (vkCreateCommandPool(dev->device, &cpi, nullptr, &g->pool) != VK_SUCCESS) {
            qrt_norm_close(dev, n);
            return fail(err, err_size, "vkCreateCommandPool (norm) failed");
        }
        VkCommandBufferAllocateInfo cbai;
        memset(&cbai, 0, sizeof(cbai));
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = g->pool;
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
            if (n->ring[i].pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(dev->device, n->ring[i].pool, nullptr);
            }
        }
        if (n->pipe != VK_NULL_HANDLE)   vkDestroyPipeline(dev->device, n->pipe, nullptr);
        if (n->pool != VK_NULL_HANDLE)   vkDestroyDescriptorPool(dev->device, n->pool, nullptr);
        if (n->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev->device, n->layout, nullptr);
        if (n->dsl != VK_NULL_HANDLE)    vkDestroyDescriptorSetLayout(dev->device, n->dsl, nullptr);
    }
    free(n);
}

static void write_set(qrt_vk_dev * dev, VkDescriptorSet set, VkBuffer * bufs, uint32_t n) {
    VkDescriptorBufferInfo info[5];
    VkWriteDescriptorSet wr[5];
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
    if (g->busy) {
        return fail(err, err_size, "norm ring %u is still in flight; wait before recording",
                    ring);
    }
    VkBuffer bufs[5] = { a->x, a->w, a->no_out != 0 ? a->x : a->out,
                         a->quant != 0 ? a->qs : a->out,
                         a->quant != 0 ? a->ds : a->out };
    write_set(n->dev, g->set, bufs, 5);

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(g->cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (norm)", r);
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
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, n->pipe);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, n->layout, 0, 1, &g->set,
                            0, nullptr);
    vkCmdPushConstants(g->cmd, n->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g->cmd, a->tokens, 1, 1);
    r = vkEndCommandBuffer(g->cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (norm)", r);
    }
    return 0;
}

int qrt_norm_submit(qrt_norm * n, uint32_t ring, char * err, size_t err_size) {
    if (n == NULL || ring >= QRT_NORM_RINGS) {
        return fail(err, err_size, "norm submit ring %u does not exist", ring);
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

#endif  // QRT_HAVE_VULKAN

// ---- double precision reference (built either way: the gate compares against it) ----

namespace {

// Round-to-nearest-even fp32 -> binary16, the packHalf2x16 rule. The deltas here are
// positive and normal, the subnormal and overflow branches only keep it total.
uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mag  = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u) {
        return (uint16_t) (sign | (mag > 0x7F800000u ? 0x7E00u : 0x7C00u));
    }
    if (mag >= 0x47800000u) {
        return (uint16_t) (sign | 0x7C00u);
    }
    if (mag < 0x38800000u) {
        if (mag < 0x33000000u) {
            return (uint16_t) sign;
        }
        const uint32_t exp   = mag >> 23;
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - exp;
        uint32_t q = mant >> shift;
        const uint32_t rem  = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u) != 0u)) {
            q++;
        }
        return (uint16_t) (sign | q);
    }
    uint32_t mant = mag & 0x7FFFFFu;
    uint32_t exp  = ((mag >> 23) - 127u) + 15u;
    const uint32_t rem = mant & 0x1FFFu;
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u) != 0u)) {
        mant++;
        if (mant == 0x400u) {
            mant = 0;
            exp++;
            if (exp >= 31u) {
                return (uint16_t) (sign | 0x7C00u);
            }
        }
    }
    return (uint16_t) (sign | (exp << 10) | mant);
}

} // namespace

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
            const uint32_t dh = f32_to_f16(d);
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
            const uint32_t dh = f32_to_f16(d);
            ds_out[(size_t) t * blocks + b] = dh | ((uint32_t) (uint16_t) (sum + 32768) << 16);
        }
    }
}
