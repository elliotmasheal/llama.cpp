#include "attn_deltanet.h"
#include "vk_transfer.h"
#include "vk_pass.h"
#include "vk_pipeline.h"

#include "gguf_index.h"
#include "qrt_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

const uint32_t DN_BINDINGS = 8;

} // namespace

struct qrt_deltanet {
    qrt_vk_dev *   dev;
    qrt_dn_shapes  sh;
    uint32_t       max_tokens;
    uint32_t       subgroup;
    float          eps;        // attention.layer_norm_rms_epsilon
    qrt_dn_bufs    bufs;
    VkBuffer       mem_owner[DN_BINDINGS];
    VkDeviceMemory mem[DN_BINDINGS];
    // the state ring (M3-3): bindings 5 and 7 are the same buffer, the shader picks the
    // slot it reads (cur) and the slots it writes ((cur + 1 + t) % slots)
    uint32_t       state_slots;
    uint32_t       state_cur;
    uint64_t       state_bytes;

    // step 2a: conv1d + silu + L2 norm, one pipeline on its own descriptor set
    VkBuffer       prep_qkv;
    VkBuffer       prep_state;
    VkBuffer       prep_norm;   // [tokens][n_embd] f32, the gate input
    VkBuffer       z_buf;       // [d_inner][tokens] gate projection output
    VkBuffer       gnorm_out;   // [tokens][d_inner] f32, the ssm_out input
    VkBuffer       ssm_norm_w;  // [head_dim] gated norm weight
    VkPipeline     gnorm_pipe;
    VkDeviceMemory prep_mem[6];
    VkPipeline     prep_pipe;
    VkDescriptorSet prep_set;
    VkPipeline     gate_pipe;
    VkDescriptorSet gate_set;
    VkDescriptorSet gnorm_set;

    VkDescriptorSetLayout dsl;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    VkPipelineLayout      layout;
    VkPipeline            pipe;
    // 32 columns per workgroup; the same math, no per-token subgroup reduction
    VkPipeline            pipe_cols;
    int                   recur_cols;
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd;
    VkFence               fence;
    VkCommandBuffer       ext;      // set by attach: the layer pass records into it
    double                last_ms;
    double                layer_ms;
};

namespace {

int alloc_buf(qrt_deltanet * dn, uint32_t idx, uint64_t bytes, char * err, size_t err_size) {
    if (qrt_vk_alloc(dn->dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->mem_owner[idx], &dn->mem[idx],
                     err, err_size) != 0) {
        return -1;
    }
    return 0;
}

} // namespace

int qrt_dn_open(qrt_vk_dev * dev, const char * spv_dir, const qrt_dn_shapes * sh,
                uint32_t max_tokens, uint32_t state_slots, qrt_deltanet ** out,
                char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL || sh == NULL || max_tokens == 0) {
        return fail(err, err_size, "the DeltaNet recurrence needs a device, SPIR-V and tokens");
    }
    if (state_slots < 2) {
        return fail(err, err_size, "the state ring needs at least 2 slots, got %u", state_slots);
    }
    if (sh->head_dim % 32 != 0 || sh->n_v_heads == 0 || sh->qk_ch % sh->n_k_heads != 0) {
        return fail(err, err_size, "shapes head_dim=%u heads=%u do not fit the 32-lane block",
                    sh->head_dim, sh->n_v_heads);
    }
    const uint32_t subgroup = dev->subgroup_size ? dev->subgroup_size : 32;
    if (!dev->has_subgroup_arithmetic) {
        return fail(err, err_size, "this device has no subgroup arithmetic (subgroup_size=%u)",
                    subgroup);
    }
    if (subgroup > 128 || 128 % subgroup != 0 || sh->head_dim % subgroup != 0 ||
        (sh->head_dim % (128 / subgroup)) != 0) {
        return fail(err, err_size, "subgroup size %u does not tile head_dim %u with 128 lanes",
                    subgroup, sh->head_dim);
    }
    qrt_deltanet * dn = (qrt_deltanet *) calloc(1, sizeof(qrt_deltanet));
    if (dn == NULL) {
        return fail(err, err_size, "out of memory for the DeltaNet recurrence");
    }
    dn->dev = dev;
    dn->sh = *sh;
    dn->max_tokens = max_tokens;
    dn->subgroup = subgroup;
    dn->eps = 1e-6f;
    dn->state_slots = state_slots;
    dn->state_cur = 0;

    const uint64_t qk_bytes = (uint64_t) max_tokens * sh->qk_ch * 4;
    const uint64_t v_bytes = (uint64_t) max_tokens * sh->d_inner * 4;
    const uint64_t gb_bytes = (uint64_t) max_tokens * sh->n_v_heads * 4;
    dn->state_bytes = (uint64_t) sh->n_v_heads * sh->head_dim * sh->head_dim * 4;
    // Binding 7 (the write side) is the same allocation as binding 5, so it carries no size
    // and no free of its own.
    const uint64_t sizes[DN_BINDINGS] = { qk_bytes, qk_bytes, v_bytes, gb_bytes, gb_bytes,
                                          dn->state_bytes * state_slots, v_bytes, 0 };
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        if (sizes[i] != 0 && alloc_buf(dn, i, sizes[i], err, err_size) != 0) {
            qrt_dn_close(dev, dn);
            return -1;
        }
    }
    dn->bufs.q = dn->mem_owner[0];
    dn->bufs.k = dn->mem_owner[1];
    dn->bufs.v = dn->mem_owner[2];
    dn->bufs.decay = dn->mem_owner[3];
    dn->bufs.beta = dn->mem_owner[4];
    dn->bufs.state = dn->mem_owner[5];
    dn->bufs.y = dn->mem_owner[6];
    dn->mem_owner[7] = dn->mem_owner[5];

    if (qrt_vk_make_set_layout(dev, DN_BINDINGS, &dn->dsl, err, err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    // One layout for every DeltaNet kernel, so the range covers the widest push block
    // (the prep kernel: 7 uint + 1 float).
    if (qrt_vk_make_layout(dev, dn->dsl, 8 * sizeof(uint32_t), &dn->layout, err, err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    VkResult r = VK_SUCCESS;

    // S_V and the real subgroup size; the lane tiling depends on both
    const uint32_t spec_serial[2] = { sh->head_dim, subgroup };
    if (qrt_vk_make_pipeline(dev, dn->layout, spv_dir, "deltanet_recur.spv", spec_serial, 2,
                         &dn->pipe, err, err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    // the column variant only needs S_V: the row split is fixed at four threads
    if (sh->head_dim % 32 == 0) {
        const uint32_t spec_cols[1] = { sh->head_dim };
        if (qrt_vk_make_pipeline(dev, dn->layout, spv_dir, "deltanet_recur_cols.spv", spec_cols, 1,
                             &dn->pipe_cols, err, err_size) != 0) {
            qrt_dn_close(dev, dn);
            return -1;
        }
    }
    // measured default: the column variant is 3.8x faster and keeps batch == serial
    dn->recur_cols = dn->pipe_cols != VK_NULL_HANDLE ? 1 : 0;

    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = DN_BINDINGS * 4;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 4;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    r = vkCreateDescriptorPool(dev->device, &dpci, nullptr, &dn->pool);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateDescriptorPool (deltanet)", r);
    }
    VkDescriptorSet sets[4] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                VK_NULL_HANDLE };
    VkDescriptorSetLayout layouts[4] = { dn->dsl, dn->dsl, dn->dsl, dn->dsl };
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = dn->pool;
    dsai.descriptorSetCount = 4;
    dsai.pSetLayouts        = layouts;
    r = vkAllocateDescriptorSets(dev->device, &dsai, sets);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkAllocateDescriptorSets (deltanet)", r);
    }
    dn->set      = sets[0];
    dn->prep_set = sets[1];
    dn->gate_set = sets[2];
    dn->gnorm_set = sets[3];
    VkDescriptorBufferInfo info[DN_BINDINGS];
    VkWriteDescriptorSet wr[DN_BINDINGS];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        info[i].buffer        = dn->mem_owner[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dev->device, DN_BINDINGS, wr, 0, nullptr);

    // step 2a buffers: the projected qkv of this chunk and the conv state
    const uint64_t qkv_bytes = (uint64_t) max_tokens * sh->conv_ch * 4;
    const uint64_t cst_bytes = (uint64_t) (sh->conv_taps - 1) * sh->conv_ch * 4;
    if (qrt_vk_alloc(dev, qkv_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_qkv, &dn->prep_mem[0], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.qkv = dn->prep_qkv;
    if (qrt_vk_alloc(dev, cst_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_state, &dn->prep_mem[1], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.conv_state = dn->prep_state;
    const uint64_t norm_bytes = (uint64_t) max_tokens * sh->n_embd * 4;
    if (qrt_vk_alloc(dev, norm_bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                     QRT_VK_USAGE_TRANSFER_DST, &dn->prep_norm, &dn->prep_mem[2], err,
                     err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }
    dn->bufs.norm = dn->prep_norm;
    const uint64_t gn_bytes = (uint64_t) max_tokens * sh->d_inner * 4;
    const uint64_t small[3] = { gn_bytes, gn_bytes, (uint64_t) sh->head_dim * 4 };
    VkBuffer * slots[3] = { &dn->gnorm_out, &dn->z_buf, &dn->ssm_norm_w };
    for (uint32_t i = 0; i < 3; ++i) {
        if (qrt_vk_alloc(dev, small[i], QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_SRC |
                         QRT_VK_USAGE_TRANSFER_DST, slots[i], &dn->prep_mem[3 + i], err,
                         err_size) != 0) {
            qrt_dn_close(dev, dn);
            return -1;
        }
    }
    dn->bufs.gnorm = dn->gnorm_out;
    dn->bufs.z = dn->z_buf;
    dn->bufs.ssm_norm_w = dn->ssm_norm_w;

    if (qrt_vk_make_pipeline(dev, dn->layout, spv_dir, "deltanet_prep.spv", nullptr, 0,
                             &dn->prep_pipe, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, dn->layout, spv_dir, "deltanet_gate.spv", nullptr, 0,
                             &dn->gate_pipe, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, dn->layout, spv_dir, "deltanet_gnorm.spv", nullptr, 0,
                             &dn->gnorm_pipe, err, err_size) != 0) {
        qrt_dn_close(dev, dn);
        return -1;
    }

    VkCommandPoolCreateInfo cpi;
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev->compute_family;
    r = vkCreateCommandPool(dev->device, &cpi, nullptr, &dn->cmd_pool);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateCommandPool (deltanet)", r);
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = dn->cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &dn->cmd);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkAllocateCommandBuffers (deltanet)", r);
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &dn->fence);
    if (r != VK_SUCCESS) {
        qrt_dn_close(dev, dn);
        return vk_fail(err, err_size, "vkCreateFence (deltanet)", r);
    }
    *out = dn;
    return 0;
}

void qrt_dn_close(qrt_vk_dev * dev, qrt_deltanet * dn) {
    if (dn == NULL) {
        return;
    }
    for (uint32_t i = 0; i < DN_BINDINGS; ++i) {
        if (dn->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, dn->mem_owner[i], dn->mem[i]);
        }
    }
    const VkBuffer prepbuf[3] = { dn->prep_qkv, dn->prep_state, dn->prep_norm };
    for (uint32_t i = 0; i < 3; ++i) {
        if (dn->prep_mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, prepbuf[i], dn->prep_mem[i]);
        }
    }
    if (dn->prep_pipe != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->prep_pipe, nullptr);
    if (dn->gate_pipe != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->gate_pipe, nullptr);
    if (dn->gnorm_pipe != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->gnorm_pipe, nullptr);
    if (dn->fence != VK_NULL_HANDLE)    vkDestroyFence(dev->device, dn->fence, nullptr);
    if (dn->cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(dev->device, dn->cmd_pool, nullptr);
    if (dn->pool != VK_NULL_HANDLE)     vkDestroyDescriptorPool(dev->device, dn->pool, nullptr);
    if (dn->pipe != VK_NULL_HANDLE)     vkDestroyPipeline(dev->device, dn->pipe, nullptr);
    if (dn->pipe_cols != VK_NULL_HANDLE) vkDestroyPipeline(dev->device, dn->pipe_cols, nullptr);
    if (dn->layout != VK_NULL_HANDLE)   vkDestroyPipelineLayout(dev->device, dn->layout, nullptr);
    if (dn->dsl != VK_NULL_HANDLE)      vkDestroyDescriptorSetLayout(dev->device, dn->dsl, nullptr);
    free(dn);
}

const qrt_dn_shapes * qrt_dn_shapes_of(const qrt_deltanet * dn) { return &dn->sh; }
const qrt_dn_bufs *   qrt_dn_buffers(const qrt_deltanet * dn) { return &dn->bufs; }
double                qrt_dn_last_ms(const qrt_deltanet * dn) { return dn->last_ms; }
double                qrt_dn_layer_ms(const qrt_deltanet * dn) { return dn->layer_ms; }

uint64_t qrt_dn_state_bytes(const qrt_deltanet * dn) {
    return dn->state_bytes;
}

uint32_t qrt_dn_state_slots(const qrt_deltanet * dn) {
    return dn->state_slots;
}

uint32_t qrt_dn_state_cur(const qrt_deltanet * dn) {
    return dn->state_cur;
}

uint64_t qrt_dn_state_offset(const qrt_deltanet * dn) {
    return (uint64_t) dn->state_cur * dn->state_bytes;
}

void qrt_dn_state_use(qrt_deltanet * dn, uint32_t slot) {
    dn->state_cur = slot % dn->state_slots;
}

int qrt_dn_set_recur(qrt_deltanet * dn, int cols) {
    if (cols && dn->pipe_cols == VK_NULL_HANDLE) {
        return -1;
    }
    dn->recur_cols = cols ? 1 : 0;
    return 0;
}

int qrt_dn_attach(qrt_deltanet * dn, VkCommandBuffer cmd, char * err, size_t err_size) {
    if (dn == NULL || cmd == VK_NULL_HANDLE) {
        return fail(err, err_size, "the DeltaNet legs need a pass command buffer");
    }
    dn->ext = cmd;
    return 0;
}

void qrt_dn_detach(qrt_deltanet * dn) {
    if (dn != NULL) {
        dn->ext = VK_NULL_HANDLE;
    }
}

// Pass mode (design 4.6): a leg appends to the caller's command buffer and inserts the
// barrier the next leg needs; on its own it keeps the begin/submit/wait shape.
static VkCommandBuffer dn_leg_begin(qrt_deltanet * dn, const char * what, char * err,
                                    size_t err_size);
static int dn_leg_end(qrt_deltanet * dn, VkCommandBuffer cmd, const char * what, double t0,
                      char * err, size_t err_size);

static int dn_recur_run(qrt_deltanet * dn, uint32_t tokens, int snap, char * err,
                        size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "chunk of %u tokens does not fit %u", tokens, dn->max_tokens);
    }
    if (snap && tokens + 1u > dn->state_slots) {
        return fail(err, err_size, "%u snapshots need %u ring slots, this DeltaNet has %u",
                    tokens, tokens + 1u, dn->state_slots);
    }
    const double t0 = now_ms();
    VkCommandBuffer cmd = dn_leg_begin(dn, "vkBeginCommandBuffer (deltanet)", err, err_size);
    if (cmd == VK_NULL_HANDLE) {
        return -1;
    }
    struct {
        uint32_t n_tokens;
        uint32_t heads;
        uint32_t kq_ch;
        uint32_t k_ratio;
        float    qscale;
        uint32_t cur;
        uint32_t slots;
        uint32_t snap;
    } pcs;
    pcs.n_tokens = tokens;
    pcs.heads    = dn->sh.n_v_heads;
    pcs.kq_ch    = dn->sh.qk_ch;
    pcs.k_ratio  = dn->sh.n_v_heads / dn->sh.n_k_heads;
    pcs.qscale   = 1.0f / sqrtf((float) dn->sh.head_dim);
    pcs.cur      = dn->state_cur;
    pcs.slots    = dn->state_slots;
    pcs.snap     = snap ? 1u : 0u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->recur_cols ? dn->pipe_cols : dn->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->set, 0, nullptr);
    vkCmdPushConstants(cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    // serial: one workgroup per (v-head, column group), the group count follows the
    // device's subgroup size which that shader is specialized on; cols: 32 columns per
    // workgroup with a thread owning 32 rows of one column
    if (dn->recur_cols) {
        vkCmdDispatch(cmd, dn->sh.n_v_heads, dn->sh.head_dim / 32u, 1);
    } else {
        const uint32_t cols_per_wg = 128u / dn->subgroup;
        vkCmdDispatch(cmd, dn->sh.n_v_heads, dn->sh.head_dim / cols_per_wg, 1);
    }
    // A chunk without snapshots leaves exactly one new state, so the cursor moves by one.
    // With snapshots the caller picks the slot after the accepted row, so it stays here.
    const int rc = dn_leg_end(dn, cmd, "deltanet", t0, err, err_size);
    if (rc == 0 && !snap) {
        dn->state_cur = (dn->state_cur + 1u) % dn->state_slots;
    }
    return rc;
}

int qrt_dn_recur(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size) {
    return dn_recur_run(dn, tokens, 0, err, err_size);
}

int qrt_dn_recur_snap(qrt_deltanet * dn, uint32_t tokens, char * err, size_t err_size) {
    return dn_recur_run(dn, tokens, 1, err, err_size);
}

// Every device buffer a layer weight can own lives in one table, and the same table is
// walked on free. The bug that cost five attempts on M2-6c was exactly this: the
// allocation side grew while the free side kept a shorter table, so count 6 walked past
// the end of it (DECISIONS 58).
#define QRT_DN_MAX_WEIGHTS 8

struct qrt_dn_weights {
    VkBuffer       conv_w;    // f32 [taps][conv_ch]
    VkBuffer       alpha_w;   // f32 [n_embd][n_v_heads]
    VkBuffer       beta_w;
    VkBuffer       dt;        // f32 [n_v_heads]
    VkBuffer       a;         // f32 [n_v_heads]
    VkBuffer       ssm_norm;  // f32 [head_dim], the sixth buffer
    VkDeviceMemory mem[QRT_DN_MAX_WEIGHTS];
    uint32_t       count;
    // host copies, for the double precision references only
    std::vector<float> h_conv, h_alpha, h_beta, h_dt, h_a, h_norm;
};

static int dn_read_f32(qrt_gguf * g, const char * name, std::vector<float> * out, char * err,
                       size_t err_size) {
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

int qrt_dn_weights_load(qrt_vk_dev * dev, const char * gguf_path, uint32_t layer,
                        const qrt_dn_shapes * sh, qrt_dn_weights ** out, char * err, size_t err_size) {
    *out = NULL;
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        return -1;
    }
    char name[128];
    std::vector<float> buf;
    snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
    const int rc = dn_read_f32(g, name, &buf, err, err_size);
    qrt_gguf_close(g);
    if (rc != 0) {
        return -1;
    }
    if (buf.size() != (size_t) sh->conv_taps * sh->conv_ch) {
        snprintf(err, err_size, "%s has %zu values, expected %u", name, buf.size(),
                 sh->conv_taps * sh->conv_ch);
        return -1;
    }
    qrt_dn_weights * w = (qrt_dn_weights *) calloc(1, sizeof(qrt_dn_weights));
    if (w == NULL) {
        snprintf(err, err_size, "out of memory for the conv weights");
        return -1;
    }
    const uint64_t bytes = buf.size() * 4;
    if (w->count >= QRT_DN_MAX_WEIGHTS) {
        snprintf(err, err_size, "the weight table holds %u buffers", QRT_DN_MAX_WEIGHTS);
        free(w);
        return -1;
    }
    if (qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST |
                     QRT_VK_USAGE_TRANSFER_SRC, &w->conv_w, &w->mem[w->count], err,
                     err_size) != 0) {
        free(w);
        return -1;
    }
    ++w->count;
    w->h_conv = buf;
    // the gate weights: f32 matrices and the two per-head vectors
    if (qrt_gguf_open(gguf_path, &g, err, err_size) != 0) {
        qrt_dn_weights_free(dev, w);
        return -1;
    }
    int grc = 0;
    struct { const char * suffix; VkBuffer * dst; std::vector<float> * host; } gate_w[] = {
        { ".ssm_alpha.weight", &w->alpha_w, &w->h_alpha },
        { ".ssm_beta.weight",  &w->beta_w,  &w->h_beta  },
        { ".ssm_dt.bias",      &w->dt,      &w->h_dt    },
        { ".ssm_a",            &w->a,       &w->h_a     },
        { ".ssm_norm.weight",  &w->ssm_norm, &w->h_norm },
    };
    for (uint32_t i = 0; i < 5 && grc == 0; ++i) {
        snprintf(name, sizeof(name), "blk.%u%s", layer, gate_w[i].suffix);
        grc = dn_read_f32(g, name, &buf, err, err_size);
        if (grc != 0) {
            break;
        }
        *gate_w[i].host = buf;
        if (w->count >= QRT_DN_MAX_WEIGHTS) {
            snprintf(err, err_size, "the weight table holds %u buffers", QRT_DN_MAX_WEIGHTS);
            grc = -1;
            break;
        }
        if (qrt_vk_alloc(dev, buf.size() * 4, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         gate_w[i].dst, &w->mem[w->count], err, err_size) != 0) {
            grc = -1;
            break;
        }
        ++w->count;
    }
    qrt_gguf_close(g);
    if (grc != 0) {
        qrt_dn_weights_free(dev, w);
        return -1;
    }
    {
        qrt_vk_staging * stg = NULL;
        const VkBuffer dsts[5] = { w->alpha_w, w->beta_w, w->dt, w->a, w->ssm_norm };
        const std::vector<float> * srcs[5] = { &w->h_alpha, &w->h_beta, &w->h_dt, &w->h_a,
                                               &w->h_norm };
        uint64_t gate_bytes = 0;
        for (uint32_t i = 0; i < 5; ++i) {
            const uint64_t b = srcs[i]->size() * 4;
            if (b > gate_bytes) {
                gate_bytes = b;
            }
        }
        // One staging buffer for all six weights. The conv upload reads h_conv, not the `buf`
        // scratch: the gate loop above leaves the last gate weight in `buf`, so uploading
        // `bytes` from it read past the end of a 512-byte vector and the conv matrix landed
        // undefined (DECISIONS 65).
        if (qrt_vk_staging_create(dev, bytes > gate_bytes ? bytes : gate_bytes, &stg, err,
                                  err_size) != 0) {
            qrt_dn_weights_free(dev, w);
            return -1;
        }
        grc = qrt_vk_upload(dev, stg, w->conv_w, 0, w->h_conv.data(), bytes, err, err_size);
        for (uint32_t i = 0; i < 5 && grc == 0; ++i) {
            grc = qrt_vk_upload(dev, stg, dsts[i], 0, srcs[i]->data(), srcs[i]->size() * 4,
                                err, err_size);
        }
        // the staging buffer must outlive the copy: destroying it right after the submit
        // left a weight undefined, which only showed up when the four legs were chained
        // (RESULTS 68)
        if (grc == 0 && qrt_vk_staging_wait(dev, stg, err, err_size) != 0) {
            grc = -1;
        }
        qrt_vk_staging_destroy(dev, stg);
        if (grc != 0) {
            qrt_dn_weights_free(dev, w);
            return -1;
        }
    }
    if (qrt_dn_weights_verify(dev, w, err, err_size) != 0) {
        qrt_dn_weights_free(dev, w);
        return -1;
    }
    *out = w;
    return 0;
}

int qrt_dn_weights_verify(qrt_vk_dev * dev, const qrt_dn_weights * w,
                          char * err, size_t err_size) {
    if (dev == NULL || w == NULL) {
        snprintf(err, err_size, "no device or weights to verify");
        return -1;
    }
    const VkBuffer bufs[6] = { w->conv_w, w->alpha_w, w->beta_w, w->dt, w->a, w->ssm_norm };
    const std::vector<float> * srcs[6] = { &w->h_conv, &w->h_alpha, &w->h_beta, &w->h_dt,
                                           &w->h_a, &w->h_norm };
    static const char * names[6] = { "ssm_conv1d", "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_a",
                                     "ssm_norm" };
    uint64_t widest = 0;
    for (uint32_t i = 0; i < 6; ++i) {
        const uint64_t b = srcs[i]->size() * 4;
        if (b > widest) {
            widest = b;
        }
    }
    qrt_vk_staging * stg = NULL;
    if (qrt_vk_staging_create(dev, widest, &stg, err, err_size) != 0) {
        return -1;
    }
    std::vector<unsigned char> got((size_t) widest);
    int rc = 0;
    for (uint32_t i = 0; i < 6 && rc == 0; ++i) {
        const uint64_t b = srcs[i]->size() * 4;
        if (qrt_vk_download(dev, stg, bufs[i], 0, got.data(), b, err, err_size) != 0) {
            rc = -1;
            break;
        }
        uint64_t bad = 0;
        for (uint64_t j = 0; j < b; ++j) {
            bad += got[(size_t) j] != ((const unsigned char *) srcs[i]->data())[j] ? 1u : 0u;
        }
        if (bad != 0) {
            snprintf(err, err_size, "%s read-back differs in %llu of %llu bytes", names[i],
                     (unsigned long long) bad, (unsigned long long) b);
            rc = -1;
        }
    }
    qrt_vk_staging_destroy(dev, stg);
    return rc;
}

VkBuffer qrt_dn_weights_conv(const qrt_dn_weights * w) {
    return w != NULL ? w->conv_w : VK_NULL_HANDLE;
}

const float * qrt_dn_weights_host(const qrt_dn_weights * w, uint32_t which) {
    if (w == NULL) {
        return NULL;
    }
    switch (which) {
        case 0: return w->h_conv.empty() ? NULL : w->h_conv.data();
        case 1: return w->h_alpha.empty() ? NULL : w->h_alpha.data();
        case 2: return w->h_beta.empty() ? NULL : w->h_beta.data();
        case 3: return w->h_dt.empty() ? NULL : w->h_dt.data();
        case 4: return w->h_a.empty() ? NULL : w->h_a.data();
        case 5: return w->h_norm.empty() ? NULL : w->h_norm.data();
    }
    return NULL;
}

void qrt_dn_weights_free(qrt_vk_dev * dev, qrt_dn_weights * w) {
    if (w == NULL) {
        return;
    }
    const VkBuffer bufs[QRT_DN_MAX_WEIGHTS] = { w->conv_w, w->alpha_w, w->beta_w, w->dt, w->a,
                                                w->ssm_norm };
    for (uint32_t i = 0; i < w->count && i < QRT_DN_MAX_WEIGHTS; ++i) {
        if (w->mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, bufs[i], w->mem[i]);
        }
    }
    free(w);
}

// Pass mode (design 4.6): a leg appends to the caller's command buffer and inserts the
// barrier the next leg needs; on its own it keeps the begin/submit/wait shape.
static VkCommandBuffer dn_leg_begin(qrt_deltanet * dn, const char * what, char * err,
                                    size_t err_size) {
    if (dn->ext != VK_NULL_HANDLE) {
        return dn->ext;
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(dn->cmd, &bi);
    if (r != VK_SUCCESS) {
        vk_fail(err, err_size, what, r);
        return VK_NULL_HANDLE;
    }
    return dn->cmd;
}

static int dn_leg_end(qrt_deltanet * dn, VkCommandBuffer cmd, const char * what, double t0,
                      char * err, size_t err_size) {
    if (dn->ext != VK_NULL_HANDLE) {
        qrt_vk_cmd_record_barrier(cmd);
        return 0;
    }
    VkResult r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, what, r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    r = vkQueueSubmit(dn->dev->compute_queue, 1, &si, dn->fence);
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, what, r);
    }
    r = vkWaitForFences(dn->dev->device, 1, &dn->fence, VK_TRUE, UINT64_MAX);
    dn->last_ms = now_ms() - t0;
    if (r != VK_SUCCESS) {
        return vk_fail(err, err_size, what, r);
    }
    return vkResetFences(dn->dev->device, 1, &dn->fence) == VK_SUCCESS ? 0
             : fail(err, err_size, "%s: vkResetFences failed", what);
}

int qrt_dn_prep_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    float l2_eps, char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "prep chunk of %u tokens does not fit %u", tokens,
                    dn->max_tokens);
    }
    // bindings follow kernels/deltanet_prep.comp: qkv, conv_w, conv_state, q, k, v
    const VkBuffer bufs[6] = { dn->bufs.qkv, w->conv_w, dn->bufs.conv_state, dn->bufs.q,
                               dn->bufs.k, dn->bufs.v };
    VkDescriptorBufferInfo info[6];
    VkWriteDescriptorSet wr[6];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 6; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->prep_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dn->dev->device, 6, wr, 0, nullptr);

    const double t0 = now_ms();
    VkCommandBuffer cmd = dn_leg_begin(dn, "vkBeginCommandBuffer (deltanet prep)", err,
                                       err_size);
    if (cmd == VK_NULL_HANDLE) {
        return -1;
    }
    struct {
        uint32_t n_tokens;
        uint32_t conv_ch;
        uint32_t qk_ch;
        uint32_t head_dim;
        uint32_t taps;
        uint32_t qbase;
        uint32_t kbase;
        float    l2_eps;
    } pcs;
    pcs.n_tokens = tokens;
    pcs.conv_ch  = dn->sh.conv_ch;
    pcs.qk_ch    = dn->sh.qk_ch;
    pcs.head_dim = dn->sh.head_dim;
    pcs.taps     = dn->sh.conv_taps;
    pcs.qbase    = 0;
    pcs.kbase    = 0;
    pcs.l2_eps   = l2_eps;    // eps / head_dim, the caller passes attention eps / head_dim
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->prep_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->prep_set, 0, nullptr);
    vkCmdPushConstants(cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    vkCmdDispatch(cmd, dn->sh.conv_ch / dn->sh.head_dim, tokens, 1);
    return dn_leg_end(dn, cmd, "deltanet prep", t0, err, err_size);
}

int qrt_dn_gate_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                    char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "gate chunk of %u tokens does not fit %u", tokens,
                    dn->max_tokens);
    }
    // bindings follow kernels/deltanet_gate.comp
    const VkBuffer bufs[7] = { dn->prep_norm, w->alpha_w, w->beta_w, w->dt, w->a,
                               dn->bufs.decay, dn->bufs.beta };
    VkDescriptorBufferInfo info[7];
    VkWriteDescriptorSet wr[7];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 7; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->gate_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dn->dev->device, 7, wr, 0, nullptr);

    const double t0 = now_ms();
    VkCommandBuffer cmd = dn_leg_begin(dn, "vkBeginCommandBuffer (deltanet gate)", err,
                                       err_size);
    if (cmd == VK_NULL_HANDLE) {
        return -1;
    }
    struct { uint32_t n_embd; uint32_t heads; uint32_t xbase; } pcs;
    pcs.n_embd = dn->sh.n_embd;
    pcs.heads  = dn->sh.n_v_heads;
    pcs.xbase  = 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->gate_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->gate_set, 0, nullptr);
    vkCmdPushConstants(cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    vkCmdDispatch(cmd, tokens, 1, 1);
    return dn_leg_end(dn, cmd, "deltanet gate", t0, err, err_size);
}

int qrt_dn_layer_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                     float l2_eps, int snap, char * err, size_t err_size) {
    const double t0 = now_ms();
    if (qrt_dn_prep_run(dn, w, tokens, l2_eps, err, err_size) != 0) {
        return -1;
    }
    if (qrt_dn_gate_run(dn, w, tokens, err, err_size) != 0) {
        return -1;
    }
    if (snap ? qrt_dn_recur_snap(dn, tokens, err, err_size)
             : qrt_dn_recur(dn, tokens, err, err_size)) {
        return -1;
    }
    if (qrt_dn_gnorm_run(dn, w, tokens, err, err_size) != 0) {
        return -1;
    }
    dn->layer_ms = now_ms() - t0;
    return 0;
}

int qrt_dn_gnorm_run(qrt_deltanet * dn, const qrt_dn_weights * w, uint32_t tokens,
                     char * err, size_t err_size) {
    if (tokens == 0 || tokens > dn->max_tokens) {
        return fail(err, err_size, "gnorm chunk of %u tokens does not fit %u", tokens,
                    dn->max_tokens);
    }
    const VkBuffer bufs[4] = { dn->bufs.y, dn->bufs.z, w->ssm_norm, dn->gnorm_out };
    VkDescriptorBufferInfo info[4];
    VkWriteDescriptorSet wr[4];
    memset(info, 0, sizeof(info));
    memset(wr, 0, sizeof(wr));
    for (uint32_t i = 0; i < 4; ++i) {
        info[i].buffer        = bufs[i];
        info[i].offset        = 0;
        info[i].range         = VK_WHOLE_SIZE;
        wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet          = dn->gnorm_set;
        wr[i].dstBinding      = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo     = &info[i];
    }
    vkUpdateDescriptorSets(dn->dev->device, 4, wr, 0, nullptr);

    const double t0 = now_ms();
    VkCommandBuffer cmd = dn_leg_begin(dn, "vkBeginCommandBuffer (deltanet gnorm)", err,
                                       err_size);
    if (cmd == VK_NULL_HANDLE) {
        return -1;
    }
    struct { uint32_t head_dim; uint32_t m; float eps; uint32_t heads; } pcs;
    pcs.head_dim = dn->sh.head_dim;
    pcs.m        = tokens;
    pcs.eps      = dn->eps;
    pcs.heads    = dn->sh.n_v_heads;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->gnorm_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dn->layout, 0, 1,
                            &dn->gnorm_set, 0, nullptr);
    vkCmdPushConstants(cmd, dn->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
    // one workgroup per token; the shader walks the heads four at a time
    vkCmdDispatch(cmd, tokens, 1, 1);
    return dn_leg_end(dn, cmd, "deltanet gnorm", t0, err, err_size);
}

#endif // QRT_HAVE_VULKAN
