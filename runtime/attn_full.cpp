// Full-attention layer of qwen35moe (M2-6d): the kernel launches and the double
// precision references. Shapes come from the GGUF metadata only
// (qrt_full_shapes_from_gguf), so the check and the layer chain cannot drift apart.

#include "attn_full.h"
#include "vk_transfer.h"
#include "vk_pass.h"
#include "vk_pipeline.h"

#include "gguf_index.h"
#include "qrt_config.h"
#include "qrt_half.h"
#include "qrt_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

// The push blocks are named types so the pipeline layout's range can never drift from the
// bytes vkCmdPushConstants writes (RESULTS 75.5 defect 1).
struct qrt_qk_pc {
    uint32_t pos0;
    uint32_t tokens;
    uint32_t qg_ch;
    uint32_t kv_ch;
    uint32_t kv_cap;
    uint32_t k_stride_w;
    uint32_t v_stride_w;
    uint32_t kv_row0;   // resident row of the chunk's first token
    float    eps;
    float    freq_base;
    float    freq_scale;
};

struct qrt_attn_pc {
    uint32_t pos0;
    uint32_t tokens;
    uint32_t kv_cap;
    uint32_t qg_ch;
    uint32_t k_stride_w;
    uint32_t v_stride_w;
    uint32_t kv_row_base;
    uint32_t kv_pos_base;
    uint32_t kb_lo;
    uint32_t kb_hi;
    uint32_t carry;
    uint32_t finalize;
    float    kq_scale;
};

static_assert(sizeof(qrt_qk_pc) == 11 * sizeof(uint32_t), "qk push block changed");
static_assert(sizeof(qrt_attn_pc) == 13 * sizeof(uint32_t), "attn push block changed");

struct qrt_full {
    qrt_vk_dev *    dev;
    qrt_full_shapes sh;
    uint32_t        max_tokens;
    uint32_t        n_ctx;
    int32_t         kv_k;
    int32_t         kv_v;
    uint32_t        k_row;   // bytes of one K row, head_dim channels
    uint32_t        v_row;
    // paging: rows in VRAM, authoritative rows in host memory, page cache bookkeeping
    int             paged;
    uint32_t        resident_tokens;
    uint32_t        page_tokens;
    uint32_t        n_pages;
    uint32_t        slots;
    uint32_t        next_slot;
    int32_t *       slot_of_page;   // -1 = not resident
    uint32_t *      page_of_slot;
    uint8_t *       k_host;
    uint8_t *       v_host;
    qrt_vk_staging * pg_stg;        // one page of the wider row
    uint64_t        page_ins;
    uint64_t        page_writes;
    qrt_full_bufs   bufs;
    VkDeviceMemory  mem[FULL_BUFFER_COUNT];
    double          last_ms;
    uint64_t        last_kv_reads;

    VkDescriptorSetLayout qk_dsl;
    VkDescriptorSetLayout attn_dsl;
    VkDescriptorPool      pool;
    VkDescriptorSet       qk_set;
    VkDescriptorSet       attn_set;
    VkPipelineLayout      qk_layout;
    VkPipelineLayout      attn_layout;
    VkPipeline            qk_pipe;
    VkPipeline            attn_pipe;
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
    VkCommandBuffer       ext;    // set by attach: the layer pass records into it
};

int qrt_kv_row_layout_of(int32_t type, uint32_t head_dim, qrt_kv_row_layout * out) {
    if (out == NULL || head_dim == 0 || (head_dim % 32) != 0) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->type   = type;
    out->blocks = head_dim / 32;
    switch (type) {
        case QRT_KV_F16:
            out->row_bytes   = head_dim * 2;
            out->scale_bytes = 0;
            out->data_offset = 0;
            break;
        case QRT_KV_Q8_0:
            out->scale_bytes = out->blocks * 2;
            out->data_offset = out->scale_bytes;
            out->row_bytes   = out->scale_bytes + head_dim;
            break;
        case QRT_KV_Q4_0:
            out->scale_bytes = out->blocks * 2;
            out->data_offset = out->scale_bytes;
            out->row_bytes   = out->scale_bytes + head_dim / 2;
            break;
        default:
            return -1;
    }
    // the kernels index the row as 32-bit words
    return (out->row_bytes % 4u) == 0 ? 0 : -1;
}

int qrt_full_shapes_from_gguf(const qrt_gguf_hparams * hp, qrt_full_shapes * out,
                              char * err, size_t err_size) {
    memset(out, 0, sizeof(*out));
    if (hp->arch[0] == '\0' || hp->head_count <= 0 || hp->head_count_kv <= 0 ||
        (hp->key_length <= 0 && hp->value_length <= 0)) {
        return fail(err, err_size, "metadata lacks attention.head_count / head_count_kv / key_length");
    }
    out->n_embd   = (uint32_t) hp->n_embd;
    out->head_dim = (uint32_t) (hp->key_length > 0 ? hp->key_length : hp->value_length);
    out->n_head   = (uint32_t) hp->head_count;
    out->n_kv     = (uint32_t) hp->head_count_kv;
    out->n_rot    = (uint32_t) (hp->rope_dimension > 0 ? hp->rope_dimension : out->head_dim);
    for (int i = 0; i < 4; ++i) {
        out->sections[i] = hp->rope_sections[i];
    }
    out->freq_base  = hp->rope_freq_base > 0.0f ? hp->rope_freq_base : 10000.0f;
    out->freq_scale = 1.0f;
    out->eps        = 1e-6f;
    out->kq_scale   = 1.0f / sqrtf((float) out->head_dim);
    out->qg_ch      = out->n_head * 2 * out->head_dim;
    out->kv_ch      = out->n_kv * out->head_dim;
    if (out->head_dim != 256) {
        return fail(err, err_size, "head_dim %u is not the 256 the kernels tile", out->head_dim);
    }
    if ((out->n_rot % 2) != 0 || out->n_rot > out->head_dim) {
        return fail(err, err_size, "n_rot %u does not fit head_dim %u", out->n_rot, out->head_dim);
    }
    return 0;
}

uint64_t qrt_full_scratch_bytes(const qrt_full_shapes * sh, uint32_t max_tokens,
                                uint32_t n_ctx, const qrt_full_kv_cfg * kv) {
    if (sh == NULL || kv == NULL || max_tokens == 0 || n_ctx == 0) {
        return 0;
    }
    qrt_kv_row_layout klay, vlay;
    if (qrt_kv_row_layout_of(kv->k_type, sh->head_dim, &klay) != 0 ||
        qrt_kv_row_layout_of(kv->v_type, sh->head_dim, &vlay) != 0) {
        return 0;
    }
    uint32_t resident = kv->resident_tokens;
    if (resident == 0 || resident > n_ctx) {
        resident = n_ctx;
    }
    const uint64_t attn = (uint64_t) max_tokens * sh->n_head * sh->head_dim * 4;
    uint64_t bytes = (uint64_t) max_tokens * sh->qg_ch * 4 +       // qg
                     2u * (uint64_t) max_tokens * sh->kv_ch * 4 +  // kin, vin
                     2u * attn +                                  // q, out
                     2u * (uint64_t) max_tokens * sh->n_head * 4; // state_m, state_l
    if (resident < n_ctx) {
        bytes += (uint64_t) max_tokens * sh->n_kv * (klay.row_bytes + vlay.row_bytes);
    }
    return bytes;
}

int qrt_full_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_full_shapes * sh,
                  uint32_t max_tokens, uint32_t n_ctx, const qrt_full_kv_cfg * kv,
                  qrt_full ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL || sh == NULL || kv == NULL || max_tokens == 0 ||
        n_ctx == 0) {
        return fail(err, err_size, "full attention needs a device, SPIR-V, tokens and a ctx");
    }
    if (sh->head_dim != 256 || (sh->n_rot % 2) != 0 || sh->n_head == 0 || sh->n_kv == 0) {
        return fail(err, err_size, "shapes head_dim=%u n_rot=%u heads=%u/%u do not fit",
                    sh->head_dim, sh->n_rot, sh->n_head, sh->n_kv);
    }
    if (n_ctx < max_tokens) {
        return fail(err, err_size, "the KV cache holds %u tokens, the chunk is %u",
                    n_ctx, max_tokens);
    }
    qrt_kv_row_layout klay, vlay;
    if (qrt_kv_row_layout_of(kv->k_type, sh->head_dim, &klay) != 0 ||
        qrt_kv_row_layout_of(kv->v_type, sh->head_dim, &vlay) != 0) {
        return fail(err, err_size, "KV precision K=%d V=%d has no row layout at head_dim %u",
                    kv->k_type, kv->v_type, sh->head_dim);
    }
    uint32_t resident = kv->resident_tokens;
    if (resident == 0 || resident > n_ctx) {
        resident = n_ctx;
    }
    uint32_t page = kv->page_tokens;
    if (page == 0 || (page % 32u) != 0 || page > n_ctx) {
        page = n_ctx < 16384u ? n_ctx : 16384u;
        page = (page / 32u) * 32u;
    }
    if (page == 0) {
        page = 32;
    }
    if (page > resident) {
        page = resident;
    }
    if (resident < 32u || page < 32u) {
        return fail(err, err_size, "the KV window holds %u tokens with a %u token page",
                    resident, page);
    }
    qrt_full * f = (qrt_full *) calloc(1, sizeof(qrt_full));
    if (f == NULL) {
        return fail(err, err_size, "out of memory for full attention");
    }
    f->dev = dev;
    f->sh = *sh;
    f->max_tokens = max_tokens;
    f->n_ctx = n_ctx;
    f->kv_k = kv->k_type;
    f->kv_v = kv->v_type;
    f->k_row = klay.row_bytes;
    f->v_row = vlay.row_bytes;
    f->resident_tokens = resident;
    f->page_tokens = page;
    f->paged = resident < n_ctx;

    const uint64_t qg_bytes = (uint64_t) max_tokens * sh->qg_ch * 4;
    const uint64_t kvi_bytes = (uint64_t) max_tokens * sh->kv_ch * 4;
    const uint64_t q_bytes  = (uint64_t) max_tokens * sh->n_head * sh->head_dim * 4;
    const uint64_t k_rows  = f->paged ? resident : n_ctx;
    const uint64_t v_rows  = f->paged ? resident : n_ctx;
    const uint64_t k_bytes  = (uint64_t) sh->n_kv * k_rows * f->k_row;
    const uint64_t v_bytes  = (uint64_t) sh->n_kv * v_rows * f->v_row;
    const uint64_t kst_bytes = f->paged ? (uint64_t) sh->n_kv * max_tokens * f->k_row : 4;
    const uint64_t vst_bytes = f->paged ? (uint64_t) sh->n_kv * max_tokens * f->v_row : 4;
    const uint64_t state_bytes = (uint64_t) sh->n_head * max_tokens * 4;
    const uint64_t w_bytes  = (uint64_t) sh->head_dim * 4;
    // scratch = 1 for the per-chunk planes, which the session arena shares across layers;
    // the KV cache and the two norm weights stay persistent per layer.
    struct slot { VkBuffer * b; uint64_t bytes; int scratch; } slots[FULL_BUFFER_COUNT] = {
        { &f->bufs.qg, qg_bytes, 1 }, { &f->bufs.kin, kvi_bytes, 1 },
        { &f->bufs.vin, kvi_bytes, 1 },
        { &f->bufs.q, q_bytes, 1 }, { &f->bufs.kcache, k_bytes, 0 },
        { &f->bufs.vcache, v_bytes, 0 },
        { &f->bufs.out, q_bytes, 1 }, { &f->bufs.q_norm_w, w_bytes, 0 },
        { &f->bufs.k_norm_w, w_bytes, 0 },
        { &f->bufs.kstage, kst_bytes, 1 }, { &f->bufs.vstage, vst_bytes, 1 },
        { &f->bufs.state_m, state_bytes, 1 }, { &f->bufs.state_l, state_bytes, 1 },
    };
    for (uint32_t i = 0; i < FULL_BUFFER_COUNT; ++i) {
        const VkBufferUsageFlags usage = QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                                         QRT_VK_USAGE_TRANSFER_DST;
        const int rc = slots[i].scratch
                           ? qrt_vk_alloc_scratch(dev, slots[i].bytes, usage, slots[i].b,
                                                  &f->mem[i], err, err_size)
                           : qrt_vk_alloc(dev, slots[i].bytes, usage, slots[i].b,
                                          &f->mem[i], err, err_size);
        if (rc != 0) {
            qrt_full_close(dev, f);
            return -1;
        }
    }

    if (qrt_vk_make_set_layout(dev, 8, &f->qk_dsl, err, err_size) != 0 ||
        qrt_vk_make_layout(dev, f->qk_dsl, sizeof(qrt_qk_pc), &f->qk_layout, err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    // the attention kernel needs 27.5 KiB of shared memory per workgroup
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev->phys, &props);
    if (props.limits.maxComputeSharedMemorySize < 22u * 1024u) {
        qrt_full_close(dev, f);
        return fail(err, err_size, "the attention kernel needs 21760 bytes of shared memory, "
                    "this device has %u", props.limits.maxComputeSharedMemorySize);
    }
    if (qrt_vk_make_set_layout(dev, 7, &f->attn_dsl, err, err_size) != 0 ||
        qrt_vk_make_layout(dev, f->attn_dsl, sizeof(qrt_attn_pc), &f->attn_layout, err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    const uint32_t attn_spec[2] = { (uint32_t) f->kv_k, (uint32_t) f->kv_v };
    if (qrt_vk_make_pipeline(dev, f->attn_layout, spv_dir, "attn_full_attn.spv", attn_spec, 2,
                      &f->attn_pipe, err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }
    const uint32_t spec[6] = { sh->head_dim, sh->n_rot, sh->n_head, sh->n_kv,
                               (uint32_t) f->kv_k, (uint32_t) f->kv_v };
    if (qrt_vk_make_pipeline(dev, f->qk_layout, spv_dir, "attn_full_qk.spv", spec, 6, &f->qk_pipe,
                      err, err_size) != 0) {
        qrt_full_close(dev, f);
        return -1;
    }

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 32;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 2;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    VkResult r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &f->pool);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (full attention)", r);
    }
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = f->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &f->qk_dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &f->qk_set);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (full attention qk)", r);
    }
    dsai.pSetLayouts = &f->attn_dsl;
    r = vkAllocateDescriptorSets(dev->device, &dsai, &f->attn_set);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (full attention)", r);
    }
    VkBuffer attn_bufs[7] = { f->bufs.q, f->bufs.kcache, f->bufs.vcache, f->bufs.out,
                              f->bufs.qg, f->bufs.state_m, f->bufs.state_l };
    qrt_vk_write_set(dev, f->attn_set, attn_bufs, 7);
    // paged mode appends into the chunk scratch, the attention reads the resident window
    VkBuffer qk_bufs[8] = { f->bufs.qg, f->bufs.kin, f->bufs.vin, f->bufs.q,
                            f->paged ? f->bufs.kstage : f->bufs.kcache,
                            f->paged ? f->bufs.vstage : f->bufs.vcache,
                            f->bufs.q_norm_w, f->bufs.k_norm_w };
    qrt_vk_write_set(dev, f->qk_set, qk_bufs, 8);

    if (f->paged) {
        f->n_pages = (n_ctx + f->page_tokens - 1u) / f->page_tokens;
        f->slots   = resident / f->page_tokens;
        if (f->slots == 0) {
            f->slots = 1;
        }
        f->slot_of_page = (int32_t *) malloc((size_t) f->n_pages * sizeof(int32_t));
        f->page_of_slot = (uint32_t *) malloc((size_t) f->slots * sizeof(uint32_t));
        f->k_host = (uint8_t *) malloc((size_t) sh->n_kv * n_ctx * f->k_row);
        f->v_host = (uint8_t *) malloc((size_t) sh->n_kv * n_ctx * f->v_row);
        const uint64_t page_rows = f->page_tokens > max_tokens ? f->page_tokens : max_tokens;
        const uint64_t page_bytes = page_rows * (f->k_row > f->v_row ? f->k_row : f->v_row);
        if (f->slot_of_page == NULL || f->page_of_slot == NULL || f->k_host == NULL ||
            f->v_host == NULL ||
            qrt_vk_staging_create(dev, page_bytes, &f->pg_stg, err, err_size) != 0) {
            qrt_full_close(dev, f);
            if (err[0] == '\0') {
                return fail(err, err_size, "out of memory for the KV page cache");
            }
            return -1;
        }
        for (uint32_t p = 0; p < f->n_pages; ++p) {
            f->slot_of_page[p] = -1;
        }
        for (uint32_t s = 0; s < f->slots; ++s) {
            f->page_of_slot[s] = UINT32_MAX;
        }
    }

    VkCommandPoolCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = dev->compute_family;
    r = vkCreateCommandPool(dev->device, &cpci, nullptr, &f->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateCommandPool (full attention)", r);
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = f->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cbai, &f->cmd);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers (full attention)", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &f->fence);
    if (r != VK_SUCCESS) {
        qrt_full_close(dev, f);
        return vk_fail(err, err_size, "vkCreateFence (full attention)", r);
    }
    *out = f;
    return 0;
}

void qrt_full_close(qrt_vk_dev * dev, qrt_full * f) {
    if (f == NULL) {
        return;
    }
    VkBuffer bufs[FULL_BUFFER_COUNT] = { f->bufs.qg, f->bufs.kin, f->bufs.vin, f->bufs.q,
                                         f->bufs.kcache, f->bufs.vcache, f->bufs.out,
                                         f->bufs.q_norm_w, f->bufs.k_norm_w, f->bufs.kstage,
                                         f->bufs.vstage, f->bufs.state_m, f->bufs.state_l };
    for (uint32_t i = 0; i < FULL_BUFFER_COUNT; ++i) {
        if (f->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, bufs[i], f->mem[i]);
        }
    }
    if (f->pg_stg != NULL)               qrt_vk_staging_destroy(dev, f->pg_stg);
    free(f->slot_of_page);
    free(f->page_of_slot);
    free(f->k_host);
    free(f->v_host);
    if (f->qk_pipe != VK_NULL_HANDLE)    vkDestroyPipeline(dev->device, f->qk_pipe, nullptr);
    if (f->attn_pipe != VK_NULL_HANDLE)  vkDestroyPipeline(dev->device, f->attn_pipe, nullptr);
    if (f->fence != VK_NULL_HANDLE)      vkDestroyFence(dev->device, f->fence, nullptr);
    if (f->cmd_pool != VK_NULL_HANDLE)   vkDestroyCommandPool(dev->device, f->cmd_pool, nullptr);
    if (f->pool != VK_NULL_HANDLE)       vkDestroyDescriptorPool(dev->device, f->pool, nullptr);
    if (f->qk_layout != VK_NULL_HANDLE)  vkDestroyPipelineLayout(dev->device, f->qk_layout, nullptr);
    if (f->attn_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev->device, f->attn_layout, nullptr);
    if (f->qk_dsl != VK_NULL_HANDLE)     vkDestroyDescriptorSetLayout(dev->device, f->qk_dsl, nullptr);
    if (f->attn_dsl != VK_NULL_HANDLE)   vkDestroyDescriptorSetLayout(dev->device, f->attn_dsl, nullptr);
    free(f);
}

const qrt_full_shapes * qrt_full_shapes_of(const qrt_full * f) { return &f->sh; }
const qrt_full_bufs *   qrt_full_buffers(const qrt_full * f) { return &f->bufs; }
int32_t                 qrt_full_kv_k(const qrt_full * f) { return f->kv_k; }
int32_t                 qrt_full_kv_v(const qrt_full * f) { return f->kv_v; }
double                  qrt_full_last_ms(const qrt_full * f) { return f->last_ms; }
uint64_t                qrt_full_last_kv_reads(const qrt_full * f) { return f->last_kv_reads; }

uint64_t qrt_full_kv_bytes(const qrt_full * f) {
    return (uint64_t) f->sh.n_kv * f->n_ctx * f->k_row;
}

uint64_t qrt_full_v_bytes(const qrt_full * f) {
    return (uint64_t) f->sh.n_kv * f->n_ctx * f->v_row;
}

uint32_t qrt_full_kv_row_bytes(const qrt_full * f, int is_v) {
    return is_v ? f->v_row : f->k_row;
}

int qrt_full_paged(const qrt_full * f) { return f->paged; }
uint32_t qrt_full_resident_tokens(const qrt_full * f) { return f->resident_tokens; }
uint32_t qrt_full_page_tokens(const qrt_full * f) { return f->page_tokens; }
const void * qrt_full_kv_host(const qrt_full * f, int is_v) { return is_v ? f->v_host : f->k_host; }
uint64_t qrt_full_page_ins(const qrt_full * f) { return f->page_ins; }
uint64_t qrt_full_page_writes(const qrt_full * f) { return f->page_writes; }

static int submit_and_wait(qrt_full * f, VkCommandBuffer cmd, char * err, size_t err_size) {
    if (f->ext != VK_NULL_HANDLE) {
        // qk writes the KV cache the attention kernel reads: order the two dispatches.
        qrt_vk_cmd_record_barrier(cmd);
        return 0;
    }
    const VkResult r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkEndCommandBuffer (full attention)", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    const double t0 = now_ms();
    VkResult rs = vkQueueSubmit(f->dev->compute_queue, 1, &si, f->fence);
    if (rs != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkQueueSubmit (full attention)", rs);
    }
    rs = vkWaitForFences(f->dev->device, 1, &f->fence, VK_TRUE, UINT64_MAX);
    f->last_ms = now_ms() - t0;
    if (rs != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkWaitForFences (full attention)", rs);
    }
    return vkResetFences(f->dev->device, 1, &f->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "vkResetFences (full attention) failed");
}

static VkCommandBuffer begin_cmd(qrt_full * f, char * err, size_t err_size) {
    if (f->ext != VK_NULL_HANDLE) {
        return f->ext;
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(f->cmd, &bi);
    if (r != VK_SUCCESS) {
        vk_fail(err, err_size, "vkBeginCommandBuffer (full attention)", r);
        return VK_NULL_HANDLE;
    }
    return f->cmd;
}

int qrt_full_attach(qrt_full * f, VkCommandBuffer cmd, char * err, size_t err_size) {
    if (f == NULL || cmd == VK_NULL_HANDLE) {
        return fail(err, err_size, "full attention needs a pass command buffer");
    }
    f->ext = cmd;
    return 0;
}

void qrt_full_detach(qrt_full * f) {
    if (f != NULL) {
        f->ext = VK_NULL_HANDLE;
    }
}

// Uploads one page's rows of both planes into a window slot (paged mode).
static int page_in(qrt_full * f, uint32_t page, uint32_t * slot_out, char * err,
                   size_t err_size) {
    if (f->slot_of_page[page] >= 0) {
        *slot_out = (uint32_t) f->slot_of_page[page];
        return 0;
    }
    const uint32_t slot = f->next_slot % f->slots;
    f->next_slot = (f->next_slot + 1u) % f->slots;
    const uint32_t old = f->page_of_slot[slot];
    if (old != UINT32_MAX) {
        f->slot_of_page[old] = -1;
    }
    const uint32_t start = page * f->page_tokens;
    uint32_t nrows = f->n_ctx - start;
    if (nrows > f->page_tokens) {
        nrows = f->page_tokens;
    }
    for (uint32_t h = 0; h < f->sh.n_kv; ++h) {
        const uint64_t dev = ((uint64_t) h * f->resident_tokens + (uint64_t) slot * f->page_tokens) *
                             f->k_row;
        const void * host = f->k_host + ((size_t) h * f->n_ctx + start) * f->k_row;
        if (qrt_vk_upload(f->dev, f->pg_stg, f->bufs.kcache, dev, host,
                          (uint64_t) nrows * f->k_row, err, err_size) != 0) {
            return -1;
        }
        const uint64_t vdev = ((uint64_t) h * f->resident_tokens +
                               (uint64_t) slot * f->page_tokens) * f->v_row;
        const void * vhost = f->v_host + ((size_t) h * f->n_ctx + start) * f->v_row;
        if (qrt_vk_upload(f->dev, f->pg_stg, f->bufs.vcache, vdev, vhost,
                          (uint64_t) nrows * f->v_row, err, err_size) != 0) {
            return -1;
        }
    }
    ++f->page_ins;
    f->slot_of_page[page] = (int32_t) slot;
    f->page_of_slot[slot] = page;
    *slot_out = slot;
    return 0;
}

int qrt_full_qk(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    if (tokens == 0 || tokens > f->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, f->max_tokens);
    }
    if (pos0 + tokens > f->n_ctx) {
        return fail(err, err_size, "positions %u..%u are past the %u token KV cache",
                    pos0, pos0 + tokens, f->n_ctx);
    }
    if (f->paged && f->ext != VK_NULL_HANDLE) {
        return fail(err, err_size, "paged KV needs the synchronous append, not a pass buffer");
    }
    if (begin_cmd(f, err, err_size) == VK_NULL_HANDLE) {
        return -1;
    }
    VkCommandBuffer cmd = f->ext != VK_NULL_HANDLE ? f->ext : f->cmd;
    qrt_qk_pc pc;
    memset(&pc, 0, sizeof(pc));
    pc.pos0       = pos0;
    pc.tokens     = tokens;
    pc.qg_ch      = f->sh.qg_ch;
    pc.kv_ch      = f->sh.kv_ch;
    pc.kv_cap     = f->paged ? f->max_tokens : f->n_ctx;
    pc.k_stride_w = f->k_row / 4u;
    pc.v_stride_w = f->v_row / 4u;
    pc.kv_row0    = f->paged ? 0u : pos0;
    pc.eps        = f->sh.eps;
    pc.freq_base  = f->sh.freq_base;
    pc.freq_scale = f->sh.freq_scale;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->qk_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->qk_layout, 0, 1,
                            &f->qk_set, 0, nullptr);
    vkCmdPushConstants(cmd, f->qk_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // one workgroup per token: a warp owns one head's norm and rope
    vkCmdDispatch(cmd, tokens, 1, 1);
    if (submit_and_wait(f, cmd, err, err_size) != 0) {
        return -1;
    }
    if (!f->paged) {
        return 0;
    }
    // the chunk went into the scratch; the host mirror is the authoritative copy
    for (uint32_t h = 0; h < f->sh.n_kv; ++h) {
        const uint64_t kdev = ((uint64_t) h * f->max_tokens) * f->k_row;
        void * khost = f->k_host + ((size_t) h * f->n_ctx + pos0) * f->k_row;
        if (qrt_vk_download(f->dev, f->pg_stg, f->bufs.kstage, kdev, khost,
                            (uint64_t) tokens * f->k_row, err, err_size) != 0) {
            return -1;
        }
        const uint64_t vdev = ((uint64_t) h * f->max_tokens) * f->v_row;
        void * vhost = f->v_host + ((size_t) h * f->n_ctx + pos0) * f->v_row;
        if (qrt_vk_download(f->dev, f->pg_stg, f->bufs.vstage, vdev, vhost,
                            (uint64_t) tokens * f->v_row, err, err_size) != 0) {
            return -1;
        }
    }
    // the window no longer reflects the pages this chunk touched
    const uint32_t first = pos0 / f->page_tokens;
    const uint32_t last = (pos0 + tokens - 1u) / f->page_tokens;
    for (uint32_t p = first; p <= last; ++p) {
        if (f->slot_of_page[p] >= 0) {
            f->slot_of_page[p] = -1;
        }
    }
    f->page_writes += last - first + 1u;
    return 0;
}

int qrt_full_attn(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    if (tokens == 0 || tokens > f->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, f->max_tokens);
    }
    if (pos0 + tokens > f->n_ctx) {
        return fail(err, err_size, "positions %u..%u are past the %u token KV cache",
                    pos0, pos0 + tokens, f->n_ctx);
    }
    if (f->paged && f->ext != VK_NULL_HANDLE) {
        return fail(err, err_size, "paged KV needs the synchronous attention, not a pass buffer");
    }
    // the account the "KV read once" claim is checked against: every query tile scans
    // the KV tiles up to its own diagonal
    uint64_t tiles = 0;
    for (uint32_t q0 = 0; q0 < tokens; q0 += 64u) {
        const uint32_t rows = tokens - q0 < 64u ? tokens - q0 : 64u;
        tiles += (pos0 + q0 + rows + 31u) / 32u;
    }
    f->last_kv_reads = tiles * f->sh.n_head * 32ull * (uint64_t) (f->k_row + f->v_row);
    const uint32_t nkb_total = (pos0 + tokens + 31u) / 32u;
    const uint32_t pos_end = pos0 + tokens;
    const uint32_t pages = f->paged ? (pos_end + f->page_tokens - 1u) / f->page_tokens : 1u;
    const double t_all = now_ms();
    for (uint32_t p = 0; p < pages; ++p) {
        uint32_t slot = 0;
        qrt_attn_pc pc;
        memset(&pc, 0, sizeof(pc));
        if (f->paged) {
            if (page_in(f, p, &slot, err, err_size) != 0) {
                return -1;
            }
            const uint32_t lo = p * f->page_tokens;
            const uint32_t hi = lo + f->page_tokens < pos_end ? lo + f->page_tokens : pos_end;
            pc.kb_lo = lo / 32u;
            pc.kb_hi = (hi + 31u) / 32u;
            pc.kv_row_base = slot * f->page_tokens;
            pc.kv_pos_base = lo;
            pc.carry = p > 0 ? 1u : 0u;
            pc.finalize = (p + 1u == pages) ? 1u : 0u;
            pc.kv_cap = f->resident_tokens;
        } else {
            pc.kb_lo = 0;
            pc.kb_hi = nkb_total;
            pc.carry = 0;
            pc.finalize = 1;
            pc.kv_cap = f->n_ctx;
        }
        if (begin_cmd(f, err, err_size) == VK_NULL_HANDLE) {
            return -1;
        }
        pc.pos0       = pos0;
        pc.tokens     = tokens;
        pc.qg_ch      = f->sh.qg_ch;
        pc.k_stride_w = f->k_row / 4u;
        pc.v_stride_w = f->v_row / 4u;
        pc.kq_scale   = f->sh.kq_scale;
        VkCommandBuffer cmd = f->ext != VK_NULL_HANDLE ? f->ext : f->cmd;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->attn_pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f->attn_layout, 0, 1,
                                &f->attn_set, 0, nullptr);
        vkCmdPushConstants(cmd, f->attn_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        // one workgroup per (query tile of 64, query head)
        vkCmdDispatch(cmd, (tokens + 63u) / 64u, f->sh.n_head, 1);
        if (submit_and_wait(f, cmd, err, err_size) != 0) {
            return -1;
        }
    }
    if (f->paged) {
        f->last_ms = now_ms() - t_all;
    }
    return 0;
}


#else // QRT_HAVE_VULKAN

int qrt_full_shapes_from_gguf(const qrt_gguf_hparams * hp, qrt_full_shapes * out,
                              char * err, size_t err_size) {
    (void) hp; (void) out;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
int qrt_full_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_full_shapes * sh,
                  uint32_t max_tokens, uint32_t n_ctx, const qrt_full_kv_cfg * kv,
                  qrt_full ** out, char * err, size_t err_size) {
    (void) dev; (void) spv_dir; (void) sh; (void) max_tokens; (void) n_ctx; (void) out;
    (void) kv;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
void qrt_full_close(qrt_vk_dev * dev, qrt_full * f) { (void) dev; (void) f; }
const qrt_full_shapes * qrt_full_shapes_of(const qrt_full * f) { (void) f; return NULL; }
const qrt_full_bufs *   qrt_full_buffers(const qrt_full * f) { (void) f; return NULL; }
int32_t                 qrt_full_kv_k(const qrt_full * f) { (void) f; return 0; }
int32_t                 qrt_full_kv_v(const qrt_full * f) { (void) f; return 0; }
uint64_t                qrt_full_kv_bytes(const qrt_full * f) { (void) f; return 0; }
uint64_t                qrt_full_v_bytes(const qrt_full * f) { (void) f; return 0; }
uint32_t                qrt_full_kv_row_bytes(const qrt_full * f, int is_v) {
    (void) f; (void) is_v; return 0;
}
int                     qrt_full_paged(const qrt_full * f) { (void) f; return 0; }
uint32_t                qrt_full_resident_tokens(const qrt_full * f) { (void) f; return 0; }
uint32_t                qrt_full_page_tokens(const qrt_full * f) { (void) f; return 0; }
const void *            qrt_full_kv_host(const qrt_full * f, int is_v) {
    (void) f; (void) is_v; return NULL;
}
uint64_t                qrt_full_page_ins(const qrt_full * f) { (void) f; return 0; }
uint64_t                qrt_full_page_writes(const qrt_full * f) { (void) f; return 0; }
int qrt_kv_row_layout_of(int32_t type, uint32_t head_dim, qrt_kv_row_layout * out) {
    (void) type; (void) head_dim; (void) out; return -1;
}
double                  qrt_full_last_ms(const qrt_full * f) { (void) f; return 0.0; }
uint64_t                qrt_full_last_kv_reads(const qrt_full * f) { (void) f; return 0; }
int qrt_full_qk(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    (void) f; (void) pos0; (void) tokens;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
int qrt_full_attn(qrt_full * f, uint32_t pos0, uint32_t tokens, char * err, size_t err_size) {
    (void) f; (void) pos0; (void) tokens;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
int qrt_full_attach(qrt_full * f, VkCommandBuffer cmd, char * err, size_t err_size) {
    (void) f; (void) cmd;
    snprintf(err, err_size, "full attention needs a Vulkan build");
    return -1;
}
void qrt_full_detach(qrt_full * f) { (void) f; }
#endif // QRT_HAVE_VULKAN
