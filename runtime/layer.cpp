// One layer's attention half (M2-6, the layer chain): the existing legs wired in the order
// the model runs them, plus the per-stage CPU double reference the check compares against.
// Every stage reads what the previous stage wrote; the host only appears in the check.

#include "layer.h"

#include "attn_deltanet.h"
#include "attn_full.h"
#include "gguf_index.h"
#include "kernels/gemm.h"
#include "kernels/norm.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <vector>

#ifdef QRT_HAVE_VULKAN

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

double now_ms(void) {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

float f16_to_f32(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: value = mant * 2^-24, so the exponent field is 103 + k with k the
            // top bit index; e counts the shifts so 113 - e is that field (RESULTS 72.4).
            int e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((uint32_t) (113 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round-to-nearest-even fp32 -> binary16 bits. packHalf2x16 rounds toward zero on this
// driver (DECISIONS 64.1), and the reference has to round the way the device kernel does.
uint32_t f32_to_f16_bits(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mag  = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u) {
        return sign | (mag > 0x7F800000u ? 0x7E00u : 0x7C00u);
    }
    if (mag >= 0x47800000u) {
        return sign | 0x7C00u;
    }
    if (mag < 0x38800000u) {
        if (mag < 0x33000000u) {
            return sign;
        }
        const uint32_t shift = 126u - (mag >> 23);
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        uint32_t q = mant >> shift;
        const uint32_t rem  = mant & ((1u << shift) - 1u);
        const uint32_t step = 1u << (shift - 1u);
        if (rem > step || (rem == step && (q & 1u) != 0u)) {
            q++;
        }
        return sign | q;
    }
    uint32_t mant = mag & 0x7FFFFFu;
    uint32_t exp  = ((mag >> 23) - 127u) + 15u;
    const uint32_t rem = mant & 0x1FFFu;
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u) != 0u)) {
        mant++;
        if (mant == 0x400u) {
            mant = 0u;
            exp++;
            if (exp >= 31u) {
                return sign | 0x7C00u;
            }
        }
    }
    return sign | (exp << 10u) | mant;
}

int read_tensor_f32(qrt_gguf * g, const char * name, std::vector<float> * out,
                    char * err, size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 0) {
        snprintf(err, err_size, "%s is %s, expected f32", name, qrt_gguf_type_name(t->type));
        return -1;
    }
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) {
        n *= t->ne[i];
    }
    out->assign((size_t) n, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err, err_size);
}

// The DeltaNet shapes, from the tensor table only (the same derivation M2-6c uses).
int dn_shapes_from_gguf(qrt_gguf * g, uint32_t layer, qrt_dn_shapes * sh, char * err,
                        size_t err_size) {
    char name[128];
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    snprintf(name, sizeof(name), "blk.%u.ssm_a", layer);
    int rc = read_tensor_f32(g, name, &ssm_a, err, err_size);
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", layer);
        rc = read_tensor_f32(g, name, &ssm_dt, err, err_size);
    }
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", layer);
        rc = read_tensor_f32(g, name, &ssm_norm, err, err_size);
    }
    const qrt_gguf_tensor * qkv = NULL;
    const qrt_gguf_tensor * conv = NULL;
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
        rc = qrt_gguf_require(g, name, &qkv, err, err_size);
    }
    if (rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
        rc = qrt_gguf_require(g, name, &conv, err, err_size);
    }
    if (rc != 0) {
        return -1;
    }
    memset(sh, 0, sizeof(*sh));
    sh->head_dim  = (uint32_t) ssm_norm.size();
    sh->n_v_heads = (uint32_t) ssm_a.size();
    sh->d_inner   = sh->n_v_heads * sh->head_dim;
    sh->n_embd    = (uint32_t) qkv->ne[0];
    sh->conv_ch   = (uint32_t) qkv->ne[1];
    sh->conv_taps = (uint32_t) conv->ne[0];
    sh->qk_ch     = (sh->conv_ch - sh->d_inner) / 2;
    sh->n_k_heads = sh->qk_ch / sh->head_dim;
    if (sh->n_k_heads == 0 || sh->qk_ch % sh->head_dim != 0 || sh->head_dim % 32u != 0) {
        return fail(err, err_size, "layer %u does not split into q/k/v", layer);
    }
    return 0;
}

} // namespace

// One prepared q8_0 projection; raw keeps the canonical blocks for the CPU reference.
struct qrt_layer_proj {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t n;
    std::vector<unsigned char> raw;
};

struct qrt_layer {
    qrt_vk_dev *    dev;
    qrt_layer_info  info;
    uint32_t        layer;
    float           eps;
    float           l2_eps;
    qrt_gemm *      gemm;
    qrt_norm *      nrm;
    qrt_deltanet *  dn;
    qrt_dn_weights * dnw;
    qrt_full *      full;
    qrt_vk_staging * stg;
    qrt_dn_shapes   dsh;
    qrt_full_shapes fsh;

    uint32_t        n_proj;
    qrt_layer_proj  proj[4];
    VkBuffer        proj_out[4];

    VkBuffer        x, norm_out, xnorm, attn_src, attn_out, xpost;
    VkDeviceMemory  x_mem, norm_mem, xnorm_mem, out_mem, post_mem;
    VkBuffer        qs, ds, p2_qs, p2_ds, norm_w, post_w;
    VkDeviceMemory  qs_mem, ds_mem, p2qs_mem, p2ds_mem, nw_mem, pw_mem;

    VkDescriptorSetLayout add_dsl;
    VkPipelineLayout add_layout;
    VkDescriptorPool add_pool;
    VkDescriptorSet  add_set;
    VkPipeline       add_pipe;
    VkCommandPool    add_cmd_pool;
    VkCommandBuffer  add_cmd;
    VkFence          add_fence;

    // host copies the check's reference needs
    std::vector<float> h_x, h_norm, h_post, h_qnw, h_knw;
    double             last_ms;
};

uint32_t qrt_layer_rings(void) { return 1; }

const qrt_layer_info * qrt_layer_shape(const qrt_layer * l) { return l != NULL ? &l->info : NULL; }
double qrt_layer_last_ms(const qrt_layer * l) { return l != NULL ? l->last_ms : 0.0; }
VkBuffer qrt_layer_x(const qrt_layer * l) { return l != NULL ? l->x : VK_NULL_HANDLE; }
VkBuffer qrt_layer_xnorm(const qrt_layer * l) { return l != NULL ? l->norm_out : VK_NULL_HANDLE; }
VkBuffer qrt_layer_attn_out(const qrt_layer * l) { return l != NULL ? l->attn_out : VK_NULL_HANDLE; }
VkBuffer qrt_layer_post_norm(const qrt_layer * l) { return l != NULL ? l->xpost : VK_NULL_HANDLE; }
VkBuffer qrt_layer_attn_src(const qrt_layer * l) { return l != NULL ? l->attn_src : VK_NULL_HANDLE; }

static int load_proj(qrt_layer * l, qrt_gguf * g, const char * name, uint32_t k, uint32_t which,
                     char * err, size_t err_size) {
    const qrt_gguf_tensor * t = NULL;
    if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
        return -1;
    }
    if (t->type != 8) {
        return fail(err, err_size, "%s is %s, the layer chain reads q8_0", name,
                    qrt_gguf_type_name(t->type));
    }
    if ((uint32_t) t->ne[0] != k) {
        return fail(err, err_size, "%s reduces over %llu, expected %u", name,
                    (unsigned long long) t->ne[0], k);
    }
    qrt_layer_proj * p = &l->proj[which];
    p->n = (uint32_t) t->ne[1];
    p->raw.assign((size_t) t->nbytes, 0);
    if (qrt_gguf_read_tensor(g, t, p->raw.data(), p->raw.size(), err, err_size) != 0) {
        return -1;
    }
    const uint64_t bytes = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, p->n, k);
    std::vector<unsigned char> prepared((size_t) bytes);
    qrt_gemm_q8k_prepare(p->raw.data(), p->n, k, prepared.data());
    if (qrt_vk_alloc(l->dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &p->buf,
                     &p->mem, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_upload(l->dev, l->stg, p->buf, 0, prepared.data(), bytes, err, err_size);
}

static int add_pipeline_open(qrt_layer * l, const char * spv_dir, char * err, size_t err_size) {
    qrt_vk_dev * dev = l->dev;
    VkDescriptorSetLayoutBinding b[3];
    memset(b, 0, sizeof(b));
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 3;
    dlci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(dev->device, &dlci, nullptr, &l->add_dsl) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateDescriptorSetLayout (layer add) failed");
    }
    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = 4 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &l->add_dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev->device, &plci, nullptr, &l->add_layout) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreatePipelineLayout (layer add) failed");
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/layer_add.spv", spv_dir);
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    rewind(f);
    std::vector<unsigned char> code((size_t) (bytes > 0 ? bytes : 0));
    if (bytes <= 0 || fread(code.data(), 1, code.size(), f) != code.size()) {
        fclose(f);
        return fail(err, err_size, "cannot read '%s'", path);
    }
    fclose(f);
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode    = (const uint32_t *) code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev->device, &smci, nullptr, &mod) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateShaderModule (layer add) failed");
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = l->add_layout;
    const VkResult r = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                                &l->add_pipe);
    vkDestroyShaderModule(dev->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkCreateComputePipelines (layer add)", r);
    }
    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 4;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 4;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &l->add_pool) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateDescriptorPool (layer add) failed");
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = l->add_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &l->add_dsl;
    if (vkAllocateDescriptorSets(dev->device, &dsai, &l->add_set) != VK_SUCCESS) {
        return fail(err, err_size, "vkAllocateDescriptorSets (layer add) failed");
    }
    VkCommandPoolCreateInfo cpi;
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev->compute_family;
    if (vkCreateCommandPool(dev->device, &cpi, nullptr, &l->add_cmd_pool) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateCommandPool (layer add) failed");
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = l->add_cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev->device, &cbai, &l->add_cmd) != VK_SUCCESS) {
        return fail(err, err_size, "vkAllocateCommandBuffers (layer add) failed");
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(dev->device, &fci, nullptr, &l->add_fence) != VK_SUCCESS) {
        return fail(err, err_size, "vkCreateFence (layer add) failed");
    }
    return 0;
}

static int add_run(qrt_layer * l, uint32_t count, VkBuffer a, VkBuffer b, VkBuffer out,
                   char * err, size_t err_size) {
    const VkBuffer bufs[3] = { a, b, out };
    VkDescriptorBufferInfo info[3];
    VkWriteDescriptorSet wr[3];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 3; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = l->add_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(l->dev->device, 3, wr, 0, nullptr);
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult r = vkBeginCommandBuffer(l->add_cmd, &bi);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkBeginCommandBuffer (layer add)", r);
    }
    const uint32_t pc[4] = { count, 0, 0, 0 };
    vkCmdBindPipeline(l->add_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_pipe);
    vkCmdBindDescriptorSets(l->add_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l->add_layout, 0, 1,
                            &l->add_set, 0, nullptr);
    vkCmdPushConstants(l->add_cmd, l->add_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(l->add_cmd, (count + 127u) / 128u, 1, 1);
    r = vkEndCommandBuffer(l->add_cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (layer add)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &l->add_cmd;
    r = vkQueueSubmit(l->dev->compute_queue, 1, &si, l->add_fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (layer add)", r);
    }
    r = vkWaitForFences(l->dev->device, 1, &l->add_fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (layer add)", r);
    }
    return vkResetFences(l->dev->device, 1, &l->add_fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (layer add) failed");
}

static int alloc_f32(qrt_layer * l, uint64_t floats, VkBuffer * buf, VkDeviceMemory * mem,
                     char * err, size_t err_size) {
    return qrt_vk_alloc(l->dev, floats * 4,
                        QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                        QRT_VK_USAGE_TRANSFER_DST, buf, mem, err, err_size);
}

int qrt_layer_open(qrt_vk_dev * dev, const char * gguf_path, uint32_t layer, uint32_t tokens,
                   const char * spv_dir, qrt_layer ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || gguf_path == NULL || spv_dir == NULL || tokens == 0) {
        return fail(err, err_size, "the layer chain needs a device, GGUF, SPIR-V and tokens");
    }
    if (tokens % 32u != 0) {
        return fail(err, err_size, "tokens=%u is not a multiple of the 32 the GEMM tiles", tokens);
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        return -1;
    }
    qrt_layer * l = (qrt_layer *) calloc(1, sizeof(qrt_layer));
    if (l == NULL) {
        qrt_gguf_close(g);
        return fail(err, err_size, "out of memory for the layer chain");
    }
    l->dev = dev;
    l->layer = layer;
    l->info.tokens = tokens;
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
    l->info.linear = qrt_gguf_find(g, name) != NULL ? 1 : 0;
    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (read_tensor_f32(g, name, &l->h_norm, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }
    snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", layer);
    if (read_tensor_f32(g, name, &l->h_post, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }
    l->info.n_embd = (uint32_t) l->h_norm.size();
    if (l->h_post.size() != l->info.n_embd) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return fail(err, err_size, "the two norms are %zu and %zu wide", l->h_norm.size(),
                    l->h_post.size());
    }
    const uint32_t E = l->info.n_embd;
    const uint32_t T = tokens;
    l->eps = 1e-6f;

    if (qrt_gemm_open(dev, spv_dir, &l->gemm, err, err_size) != 0 ||
        qrt_norm_open(dev, spv_dir, &l->nrm, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        qrt_gguf_close(g);
        return -1;
    }

    // Attention module: one of the two, chosen by what the layer carries.
    qrt_dn_shapes dsh;
    qrt_full_shapes fsh;
    if (l->info.linear) {
        if (dn_shapes_from_gguf(g, layer, &dsh, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        if (qrt_dn_open(dev, spv_dir, &dsh, T, &l->dn, err, err_size) != 0 ||
            qrt_dn_weights_load(dev, gguf_path, layer, &dsh, &l->dnw, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        l->info.inter = dsh.d_inner;
        l->dsh = dsh;
        l->l2_eps = 1e-6f / (float) dsh.head_dim;
        const qrt_dn_bufs * db = qrt_dn_buffers(l->dn);
        l->norm_out = db->norm;
        l->attn_src = db->gnorm;
        l->n_proj = 3;
        l->proj_out[0] = db->qkv;
        l->proj_out[1] = db->z;
    } else {
        const qrt_gguf_hparams hp = *qrt_gguf_hparams_of(g);
        if (qrt_full_shapes_from_gguf(&hp, &fsh, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        if (fsh.n_embd != E) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return fail(err, err_size, "attention.n_embd=%u, the norms are %u", fsh.n_embd, E);
        }
        if (qrt_full_open(dev, spv_dir, &fsh, T, T, &l->full, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", layer);
        if (read_tensor_f32(g, name, &l->h_qnw, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", layer);
        if (read_tensor_f32(g, name, &l->h_knw, err, err_size) != 0 ||
            l->h_qnw.size() != fsh.head_dim || l->h_knw.size() != fsh.head_dim) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return fail(err, err_size, "the q/k norms are not head_dim wide");
        }
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        l->info.inter = fsh.n_head * fsh.head_dim;
        l->fsh = fsh;
        l->attn_src = fb->out;
        l->n_proj = 4;
        l->proj_out[0] = fb->qg;
        l->proj_out[1] = fb->kin;
        l->proj_out[2] = fb->vin;
    }

    // The staging buffer has to exist before the first weight upload; size it from the
    // largest thing that goes through it: a weight matrix, one chunk row, or the state.
    {
        uint64_t stg_bytes = (uint64_t) T * (E > l->info.inter ? E : l->info.inter) * 4;
        if (l->info.linear) {
            const uint64_t st = (uint64_t) l->dsh.n_v_heads * l->dsh.head_dim *
                                l->dsh.head_dim * 4;
            if (st > stg_bytes) {
                stg_bytes = st;
            }
        }
        // The weight matrices go through this staging buffer too; their sizes are in the
        // tensor table, so no data is read for this pass.
        static const char * names_lin[3] = { "attn_qkv.weight", "attn_gate.weight",
                                             "ssm_out.weight" };
        static const char * names_full[4] = { "attn_q.weight", "attn_k.weight",
                                              "attn_v.weight", "attn_output.weight" };
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            snprintf(name, sizeof(name), "blk.%u.%s", layer,
                     l->info.linear ? names_lin[i] : names_full[i]);
            const qrt_gguf_tensor * t = NULL;
            if (qrt_gguf_require(g, name, &t, err, err_size) != 0) {
                qrt_layer_close(dev, l);
                qrt_gguf_close(g);
                return -1;
            }
            const uint32_t k = (i + 1 == l->n_proj) ? l->info.inter : E;
            const uint64_t prepared = qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0,
                                                            (uint32_t) t->ne[1], k);
            if (prepared > stg_bytes) {
                stg_bytes = prepared;
            }
        }
        if (qrt_vk_staging_create(dev, stg_bytes, &l->stg, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            qrt_gguf_close(g);
            return -1;
        }
    }

    // The projections, in the order the run feeds them; the last one eats the attention
    // output and writes attn_out.
    {
        static const char * names_lin[3] = { "attn_qkv.weight", "attn_gate.weight",
                                             "ssm_out.weight" };
        static const char * names_full[4] = { "attn_q.weight", "attn_k.weight",
                                              "attn_v.weight", "attn_output.weight" };
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            snprintf(name, sizeof(name), "blk.%u.%s", layer,
                     l->info.linear ? names_lin[i] : names_full[i]);
            const uint32_t k = (i + 1 == l->n_proj) ? l->info.inter : E;
            if (load_proj(l, g, name, k, i, err, err_size) != 0) {
                qrt_layer_close(dev, l);
                qrt_gguf_close(g);
                return -1;
            }
        }
    }
    qrt_gguf_close(g);
    g = NULL;

    // Every activation buffer this chain touches.
    if (alloc_f32(l, (uint64_t) T * E, &l->x, &l->x_mem, err, err_size) != 0 ||
        alloc_f32(l, (uint64_t) T * E, &l->attn_out, &l->out_mem, err, err_size) != 0 ||
        alloc_f32(l, (uint64_t) T * E, &l->xpost, &l->post_mem, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    l->proj_out[l->n_proj - 1] = l->attn_out;
    if (!l->info.linear) {
        if (alloc_f32(l, (uint64_t) T * E, &l->xnorm, &l->xnorm_mem, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
        l->norm_out = l->xnorm;
    }
    const uint64_t nblocks = (uint64_t) T * (E / 32u);
    const uint64_t nblocks2 = (uint64_t) T * (l->info.inter / 32u);
    if (qrt_vk_alloc(dev, (uint64_t) T * E, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->qs, &l->qs_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, nblocks * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->ds, &l->ds_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) T * l->info.inter,
                     QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &l->p2_qs, &l->p2qs_mem,
                     err, err_size) != 0 ||
        qrt_vk_alloc(dev, nblocks2 * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->p2_ds, &l->p2ds_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) E * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->norm_w, &l->nw_mem, err, err_size) != 0 ||
        qrt_vk_alloc(dev, (uint64_t) E * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &l->post_w, &l->pw_mem, err, err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    if (add_pipeline_open(l, spv_dir, err, err_size) != 0 ||
        qrt_vk_upload(dev, l->stg, l->norm_w, 0, l->h_norm.data(), (uint64_t) E * 4, err,
                      err_size) != 0 ||
        qrt_vk_upload(dev, l->stg, l->post_w, 0, l->h_post.data(), (uint64_t) E * 4, err,
                      err_size) != 0) {
        qrt_layer_close(dev, l);
        return -1;
    }
    if (l->full != NULL) {
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        if (qrt_vk_upload(dev, l->stg, fb->q_norm_w, 0, l->h_qnw.data(),
                          (uint64_t) l->h_qnw.size() * 4, err, err_size) != 0 ||
            qrt_vk_upload(dev, l->stg, fb->k_norm_w, 0, l->h_knw.data(),
                          (uint64_t) l->h_knw.size() * 4, err, err_size) != 0) {
            qrt_layer_close(dev, l);
            return -1;
        }
    }
    *out = l;
    return 0;
}

void qrt_layer_close(qrt_vk_dev * dev, qrt_layer * l) {
    if (l == NULL) {
        return;
    }
    if (dev != NULL) {
        for (uint32_t i = 0; i < l->n_proj; ++i) {
            if (l->proj[i].mem != VK_NULL_HANDLE) {
                qrt_vk_free(dev, l->proj[i].buf, l->proj[i].mem);
            }
        }
        const VkBuffer bufs[9] = { l->x, l->xnorm, l->attn_out, l->xpost, l->qs, l->ds,
                                   l->p2_qs, l->p2_ds, l->norm_w };
        VkDeviceMemory mems[9] = { l->x_mem, l->xnorm_mem, l->out_mem, l->post_mem, l->qs_mem,
                                   l->ds_mem, l->p2qs_mem, l->p2ds_mem, l->nw_mem };
        for (uint32_t i = 0; i < 9; ++i) {
            if (mems[i] != VK_NULL_HANDLE) {
                qrt_vk_free(dev, bufs[i], mems[i]);
            }
        }
        if (l->pw_mem != VK_NULL_HANDLE)                 qrt_vk_free(dev, l->post_w, l->pw_mem);
        if (l->add_fence != VK_NULL_HANDLE)              vkDestroyFence(dev->device, l->add_fence, nullptr);
        if (l->add_cmd_pool != VK_NULL_HANDLE)           vkDestroyCommandPool(dev->device, l->add_cmd_pool, nullptr);
        if (l->add_pool != VK_NULL_HANDLE)               vkDestroyDescriptorPool(dev->device, l->add_pool, nullptr);
        if (l->add_pipe != VK_NULL_HANDLE)               vkDestroyPipeline(dev->device, l->add_pipe, nullptr);
        if (l->add_layout != VK_NULL_HANDLE)             vkDestroyPipelineLayout(dev->device, l->add_layout, nullptr);
        if (l->add_dsl != VK_NULL_HANDLE)                vkDestroyDescriptorSetLayout(dev->device, l->add_dsl, nullptr);
        qrt_vk_staging_destroy(dev, l->stg);
        qrt_norm_close(dev, l->nrm);
        qrt_gemm_close(dev, l->gemm);
        qrt_dn_weights_free(dev, l->dnw);
        qrt_dn_close(dev, l->dn);
        qrt_full_close(dev, l->full);
    }
    free(l);
}

int qrt_layer_set_input(qrt_layer * l, const float * x, char * err, size_t err_size) {
    if (l == NULL || x == NULL) {
        return fail(err, err_size, "the layer needs an input");
    }
    const uint64_t bytes = (uint64_t) l->info.tokens * l->info.n_embd * 4;
    l->h_x.assign(x, x + (size_t) (bytes / 4));
    return qrt_vk_upload(l->dev, l->stg, l->x, 0, x, bytes, err, err_size);
}

static int layer_norm(qrt_layer * l, VkBuffer src, VkBuffer w, VkBuffer out, VkBuffer qs,
                      VkBuffer ds, uint32_t k, int no_norm, char * err, size_t err_size) {
    qrt_norm_args na;
    memset(&na, 0, sizeof(na));
    na.x = src;
    na.w = w;
    na.out = out;
    na.qs = qs;
    na.ds = ds;
    na.k = k;
    na.tokens = l->info.tokens;
    na.eps = l->eps;
    na.quant = 1;
    na.no_norm = no_norm;
    na.no_out = out == VK_NULL_HANDLE ? 1 : 0;
    return qrt_norm_run(l->nrm, &na, err, err_size);
}

static int layer_gemm(qrt_layer * l, uint32_t which, VkBuffer planes_qs, VkBuffer planes_ds,
                      VkBuffer out, uint32_t k, char * err, size_t err_size) {
    qrt_gemm_args ga;
    memset(&ga, 0, sizeof(ga));
    ga.w = l->proj[which].buf;
    ga.a_qs = planes_qs;
    ga.a_ds = planes_ds;
    ga.out = out;
    ga.n = l->proj[which].n;
    ga.k = k;
    ga.m = l->info.tokens;
    ga.type = QRT_QUANT_Q8_0;
    ga.token_major = 1;
    return qrt_gemm_dispatch(l->gemm, &ga, err, err_size);
}

int qrt_layer_run(qrt_layer * l, char * err, size_t err_size) {
    if (l == NULL) {
        return fail(err, err_size, "no layer");
    }
    const double t0 = now_ms();
    const uint32_t T = l->info.tokens;
    const uint32_t E = l->info.n_embd;
    const uint32_t last = l->n_proj - 1;

    // 1. input norm; DeltaNet's gate reads the f32 row, so norm_out points at its buffer
    if (layer_norm(l, l->x, l->norm_w, l->norm_out, l->qs, l->ds, E, 0, err, err_size) != 0) {
        return -1;
    }
    // 2. the projections that eat the normed row
    for (uint32_t i = 0; i < last; ++i) {
        if (layer_gemm(l, i, l->qs, l->ds, l->proj_out[i], E, err, err_size) != 0) {
            return -1;
        }
    }
    // 3. the attention path
    if (getenv("QRT_LAYER_STOP_PROJ") != NULL) {
        l->last_ms = now_ms() - t0;
        return 0;
    }
    if (l->info.linear) {
        if (qrt_dn_layer_run(l->dn, l->dnw, T, l->l2_eps, err, err_size) != 0) {
            return -1;
        }
    } else {
        if (qrt_full_qk(l->full, 0, T, err, err_size) != 0 ||
            qrt_full_attn(l->full, 0, T, err, err_size) != 0) {
            return -1;
        }
    }
    // 4. quantise the attention output: the output projection eats planes, not a row
    if (layer_norm(l, l->attn_src, l->norm_w, VK_NULL_HANDLE, l->p2_qs, l->p2_ds, l->info.inter,
                   1, err, err_size) != 0) {
        return -1;
    }
    // 5. output projection, 6. residual add
    if (layer_gemm(l, last, l->p2_qs, l->p2_ds, l->attn_out, l->info.inter, err, err_size) != 0 ||
        add_run(l, T * E, l->x, l->attn_out, l->x, err, err_size) != 0) {
        return -1;
    }
    // 7. post-attention norm, which is what the FFN half will read
    if (layer_norm(l, l->x, l->post_w, l->xpost, l->qs, l->ds, E, 0, err, err_size) != 0) {
        return -1;
    }
    l->last_ms = now_ms() - t0;
    return 0;
}

// The recurrence state and the conv state of a DeltaNet layer start at zero for every chunk;
// full attention rewrites the positions it reads, so it needs no reset.
int qrt_layer_zero_state(qrt_layer * l, char * err, size_t err_size) {
    if (l == NULL) {
        return fail(err, err_size, "no layer");
    }
    if (!l->info.linear) {
        return 0;
    }
    const qrt_dn_bufs * b = qrt_dn_buffers(l->dn);
    const uint64_t st = (uint64_t) l->dsh.n_v_heads * l->dsh.head_dim * l->dsh.head_dim * 4;
    const uint64_t cv = (uint64_t) (l->dsh.conv_taps - 1) * l->dsh.conv_ch * 4;
    std::vector<unsigned char> zero((size_t) (st > cv ? st : cv), 0);
    if (qrt_vk_upload(l->dev, l->stg, b->state_in, 0, zero.data(), st, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_upload(l->dev, l->stg, b->conv_state, 0, zero.data(), cv, err, err_size);
}

namespace {

double rel_l1_of(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
        den += fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : num;
}

// Double precision reference of a q8_0 x Q8_1 GEMM over the whole matrix, [n][tokens].
void dense_cpu(const std::vector<unsigned char> & raw, uint32_t n, uint32_t k,
               const signed char * qs, const uint32_t * ds, uint32_t tokens,
               std::vector<double> * out) {
    out->assign((size_t) n * tokens, 0.0);
    const uint32_t nb = k / 32u;
    for (uint32_t r = 0; r < n; ++r) {
        const unsigned char * row = raw.data() + (size_t) r * nb * 34u;
        for (uint32_t b = 0; b < nb; ++b) {
            const unsigned char * blk = row + (size_t) b * 34u;
            const uint32_t dh = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
            const double dw = (double) f16_to_f32(dh);
            double wq[32];
            for (uint32_t e = 0; e < 32u; ++e) {
                wq[e] = (double) (int8_t) blk[2 + e];
            }
            for (uint32_t t = 0; t < tokens; ++t) {
                const uint32_t ab = t * nb + b;
                const double aq = (double) f16_to_f32(ds[ab] & 0xFFFFu);
                const signed char * qa = qs + (size_t) ab * 32u;
                double sub = 0.0;
                for (uint32_t e = 0; e < 32u; ++e) {
                    sub += wq[e] * (double) qa[e];
                }
                (*out)[(size_t) r * tokens + t] += dw * aq * sub;
            }
        }
    }
}

void to_token_major(const std::vector<double> & src, uint32_t n, uint32_t tokens,
                    std::vector<float> * out) {
    out->assign((size_t) n * tokens, 0.0f);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t t = 0; t < tokens; ++t) {
            (*out)[(size_t) t * n + r] = (float) src[(size_t) r * tokens + t];
        }
    }
}

int download(qrt_layer * l, VkBuffer src, void * dst, uint64_t bytes, char * err,
             size_t err_size) {
    return qrt_vk_download(l->dev, l->stg, src, 0, dst, bytes, err, err_size);
}

} // namespace

int qrt_layer_check(const char * gguf_path, uint32_t layer, uint32_t tokens,
                    const char * spv_dir) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open(0, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_layer * l = NULL;
    if (qrt_layer_open(dev, gguf_path, layer, tokens, spv_dir, &l, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const qrt_layer_info info = *qrt_layer_shape(l);
    const uint32_t E = info.n_embd;
    const uint32_t T = info.tokens;
    printf("layer_info layer=%u linear=%d n_embd=%u tokens=%u inter=%u projections=%u\n",
           layer, info.linear, E, T, info.inter, l->n_proj);

    std::vector<float> x((size_t) T * E);
    uint32_t seed = 0x5EED1234u;
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) ((seed >> 8) % 2001u) - 1000) * 0.001f;
    }
    if (qrt_layer_set_input(l, x.data(), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }

    double tms[3];
    for (int i = 0; i < 3; ++i) {
        if (qrt_layer_zero_state(l, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        const double t0 = now_ms();
        if (qrt_layer_run(l, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        tms[i] = now_ms() - t0;
    }
    const double med = tms[0] <= tms[1] ? (tms[1] <= tms[2] ? tms[1] : (tms[0] <= tms[2] ? tms[2] : tms[0]))
                                        : (tms[0] <= tms[2] ? tms[0] : (tms[1] <= tms[2] ? tms[2] : tms[1]));
    double min_ms = tms[0];
    for (int i = 1; i < 3; ++i) {
        if (tms[i] < min_ms) {
            min_ms = tms[i];
        }
    }
    printf("layer_perf layer=%u tokens=%u ms=%.3f min_ms=%.3f us_per_token=%.3f\n", layer, T, med,
           min_ms, med * 1000.0 / (double) T);

    if (T > 1024) {
        printf("layer_chain skipped tokens=%u (the CPU reference is capped at 1024)\n", T);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 0;
    }

    // One more run from a zero state and the original input, so the device state is the one
    // the reference models: the residual add leaves its result in x, which is the point of
    // a layer chain but not what a single-layer check wants to accumulate.
    if (qrt_layer_set_input(l, x.data(), err, sizeof(err)) != 0 ||
        qrt_layer_zero_state(l, err, sizeof(err)) != 0 || qrt_layer_run(l, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }

    const uint32_t last = l->n_proj - 1;
    std::vector<float> g_norm((size_t) T * E), g_src((size_t) T * info.inter);
    std::vector<float> g_out((size_t) T * E), g_x((size_t) T * E), g_post((size_t) T * E);
    if (download(l, l->norm_out, g_norm.data(), g_norm.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->attn_src, g_src.data(), g_src.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->attn_out, g_out.data(), g_out.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->x, g_x.data(), g_x.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->xpost, g_post.data(), g_post.size() * 4, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t cols = 64;
    std::vector<std::vector<float>> g_proj(l->n_proj);
    std::vector<std::vector<float>> g_proj_full(l->n_proj);
    for (uint32_t i = 0; i < l->n_proj; ++i) {
        const uint32_t n = l->proj[i].n;
        g_proj[i].assign((size_t) cols * n, 0.0f);
        if (n * cols * 4 > 0 &&
            download(l, l->proj_out[i], g_proj[i].data(), (uint64_t) cols * n * 4, err,
                     sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
    }

    // The device projections, all tokens: the later stages' references use these as their
    // input, so this check validates the wiring (each stage's buffer being the next one's
    // source) while every kernel keeps its own check.
    for (uint32_t i = 0; i < last; ++i) {
        const uint32_t n = l->proj[i].n;
        g_proj_full[i].assign((size_t) T * n, 0.0f);
        // A 8192-wide plane does not fit the staging buffer in one go, so it goes in slices.
        const uint32_t slice_tok = 128;
        for (uint32_t t0 = 0; t0 < T; t0 += slice_tok) {
            const uint32_t cnt = T - t0 < slice_tok ? T - t0 : slice_tok;
            if (qrt_vk_download(dev, l->stg, l->proj_out[i], (uint64_t) t0 * n * 4,
                                g_proj_full[i].data() + (size_t) t0 * n, (uint64_t) cnt * n * 4,
                                err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_layer_close(dev, l);
                qrt_vk_close(dev);
                return 2;
            }
        }
    }

    // ---- the CPU chain over the same weights and the same Q8_1 rule ----
    std::vector<signed char> qs0((size_t) T * E);
    std::vector<uint32_t> ds0((size_t) T * (E / 32u));
    std::vector<float> ref_norm((size_t) T * E);
    qrt_norm_cpu(T, E, x.data(), l->h_norm.data(), l->eps, ref_norm.data(), qs0.data(),
                 ds0.data());
    std::vector<std::vector<float>> ref_proj(l->n_proj);
    for (uint32_t i = 0; i < last; ++i) {
        std::vector<double> acc;
        dense_cpu(l->proj[i].raw, l->proj[i].n, E, qs0.data(), ds0.data(), T, &acc);
        to_token_major(acc, l->proj[i].n, T, &ref_proj[i]);
    }
    // attention stage, from the projections the CPU just computed
    std::vector<float> ref_src((size_t) T * info.inter);
    if (info.linear) {
        const qrt_dn_shapes & sh = l->dsh;
        const uint64_t st_f = (uint64_t) sh.n_v_heads * sh.head_dim * sh.head_dim;
        std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
        std::vector<float> csto(cst.size());
        std::vector<float> q((size_t) T * sh.qk_ch), k((size_t) T * sh.qk_ch);
        std::vector<float> v((size_t) T * sh.d_inner);
        std::vector<float> drows((size_t) T * sh.n_v_heads), brows(drows.size());
        std::vector<double> st((size_t) st_f, 0.0);
        std::vector<float> yref((size_t) T * sh.d_inner);
        qrt_dn_prep_cpu(&sh, T, g_proj_full[0].data(), qrt_dn_weights_host(l->dnw, 0), cst.data(),
                        l->l2_eps, q.data(), k.data(), v.data(), csto.data());
        qrt_dn_gate_cpu(&sh, T, g_norm.data(), qrt_dn_weights_host(l->dnw, 1),
                        qrt_dn_weights_host(l->dnw, 2), qrt_dn_weights_host(l->dnw, 3),
                        qrt_dn_weights_host(l->dnw, 4), drows.data(), brows.data());
        qrt_dn_recur_cpu(&sh, T, q.data(), k.data(), v.data(), drows.data(), brows.data(),
                         st.data(), yref.data());
        qrt_dn_gnorm_cpu(&sh, T, yref.data(), g_proj_full[1].data(), qrt_dn_weights_host(l->dnw, 5),
                         l->eps, ref_src.data());
    } else {
        const qrt_full_shapes & sh = l->fsh;
        std::vector<float> q_out((size_t) T * sh.n_head * sh.head_dim);
        std::vector<float> k_out((size_t) sh.n_kv * T * sh.head_dim);
        std::vector<float> v_out(k_out.size());
        qrt_full_qk_cpu(&sh, 0, T, g_proj_full[0].data(), g_proj_full[1].data(),
                        g_proj_full[2].data(),
                        l->h_qnw.data(), l->h_knw.data(), q_out.data(), k_out.data(),
                        v_out.data());
        // The reference reads the DEVICE's f16 KV cache and dequantises it, so the 3.5e-04
        // of f16 storage error is shared by both sides instead of landing in the gate.
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        std::vector<uint16_t> kc((size_t) sh.n_kv * T * sh.head_dim);
        std::vector<uint16_t> vc(kc.size());
        if (download(l, fb->kcache, kc.data(), kc.size() * 2, err, sizeof(err)) != 0 ||
            download(l, fb->vcache, vc.data(), vc.size() * 2, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        for (size_t i = 0; i < kc.size(); ++i) {
            k_out[i] = f16_to_f32(kc[i]);
        }
        for (size_t i = 0; i < vc.size(); ++i) {
            v_out[i] = f16_to_f32(vc[i]);
        }
        std::vector<uint32_t> rows(T);
        for (uint32_t i = 0; i < T; ++i) {
            rows[i] = i;
        }
        qrt_full_attn_rows_cpu(&sh, T, rows.data(), T, q_out.data(), k_out.data(),
                               v_out.data(), g_proj_full[0].data(), ref_src.data());
    }

    std::vector<float> ref_src_copy((size_t) T * info.inter);
    std::vector<signed char> p2qs((size_t) T * info.inter);
    std::vector<uint32_t> p2ds((size_t) T * (info.inter / 32u));
    // The device's own Q8_1 planes of the attention output: the quantiser is what --norm-check
    // gates, so the output projection is compared on the same planes the device used.
    if (download(l, l->p2_qs, p2qs.data(), p2qs.size(), err, sizeof(err)) != 0 ||
        download(l, l->p2_ds, p2ds.data(), p2ds.size() * 4, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_norm_quant_cpu(T, info.inter, g_src.data(), ref_src_copy.data(), nullptr, nullptr);
    std::vector<double> acc_out;
    dense_cpu(l->proj[last].raw, l->proj[last].n, info.inter, p2qs.data(), p2ds.data(), T,
              &acc_out);
    std::vector<float> ref_out((size_t) T * E);
    to_token_major(acc_out, l->proj[last].n, T, &ref_out);
    std::vector<float> ref_x((size_t) T * E), ref_post((size_t) T * E);
    for (size_t i = 0; i < ref_x.size(); ++i) {
        ref_x[i] = x[i] + ref_out[i];
    }
    qrt_norm_cpu(T, E, ref_x.data(), l->h_post.data(), l->eps, ref_post.data(), nullptr, nullptr);

    int fails = 0;
    int open = 0;
    const double r_norm = rel_l1_of(g_norm.data(), ref_norm.data(), g_norm.size());
    printf("layer_norm rel_l1=%.3e verdict=%s\n", r_norm, r_norm <= 1e-6 ? "PASS" : "FAIL");
    fails += r_norm <= 1e-6 ? 0 : 1;
    for (uint32_t i = 0; i < last; ++i) {
        const uint32_t n = l->proj[i].n;
        const double rel = rel_l1_of(g_proj[i].data(), ref_proj[i].data(),
                                     (size_t) cols * n);
        // f32 accumulation in the kernel against a double reference: measured 2e-06..1e-05
        // depending on the matrix, while a layout bug is O(1e-1).
        const int ok = rel <= 5e-5;
        printf("layer_proj%u n=%u k=%u cols=%u rel_l1=%.3e verdict=%s\n", i, n, E, cols, rel,
               ok ? "PASS" : "OPEN");
        open += ok ? 0 : 1;
    }
    const double r_src = rel_l1_of(g_src.data(), ref_src.data(), g_src.size());
    printf("layer_attn linear=%d inter=%u rel_l1=%.3e verdict=%s\n", info.linear, info.inter,
           r_src, r_src <= 1e-4 ? "PASS" : "FAIL");
    fails += r_src <= 1e-4 ? 0 : 1;
    const double r_out = rel_l1_of(g_out.data(), ref_out.data(), g_out.size());
    printf("layer_outproj n=%u k=%u rel_l1=%.3e verdict=%s\n", l->proj[last].n, info.inter,
           r_out, r_out <= 1e-5 ? "PASS" : "FAIL");
    fails += r_out <= 1e-5 ? 0 : 1;
    const double r_x = rel_l1_of(g_x.data(), ref_x.data(), g_x.size());
    printf("layer_residual rel_l1=%.3e verdict=%s\n", r_x, r_x <= 1e-6 ? "PASS" : "FAIL");
    fails += r_x <= 1e-6 ? 0 : 1;
    const double r_post = rel_l1_of(g_post.data(), ref_post.data(), g_post.size());
    printf("layer_postnorm rel_l1=%.3e verdict=%s\n", r_post, r_post <= 1e-6 ? "PASS" : "FAIL");
    fails += r_post <= 1e-6 ? 0 : 1;
    printf("layer_check layer=%u linear=%d tokens=%u fails=%d open=%d verdict=%s\n", layer,
           info.linear, T, fails, open, fails == 0 ? (open == 0 ? "PASS" : "PASS_OPEN") : "FAIL");

    qrt_layer_close(dev, l);
    qrt_vk_close(dev);
    return fails == 0 ? 0 : 1;
}

#endif // QRT_HAVE_VULKAN