// The device output head: final RMS norm with its Q8_1 packing, the full-vocabulary
// Q8_0 x Q8_1 GEMM and the two-stage argmax, recorded into one command buffer so a verified
// pass costs one submit for its sample (design 4.6).
//
// The point of the module is that nothing vocab-sized crosses the PCIe bus per pass: the
// host head reads output.weight row by row (about 0.5 GiB per pass) and shares the cores with
// the CPU expert leg.

#include "head.h"
#include "kernels/gemm.h"
#include "kernels/norm.h"
#include "kernels/tiles.h"
#include "qrt_util.h"
#include "vk_pass.h"
#include "vk_pipeline.h"
#include "vk_transfer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifdef QRT_HAVE_VULKAN

#define QRT_HEAD_NORM_SLOTS 2u
#define QRT_HEAD_ARGM_MAX_SETS 8u
// Entries one argmax workgroup owns. 248320 vocabularies give 61 segments, so stage 2 runs
// in one workgroup per row without a second trip over the logits.
#define QRT_HEAD_ARGM_SPAN 4096u
#define QRT_HEAD_STAGING_BYTES (8u << 20)

struct qrt_head {
    qrt_vk_dev *   dev;
    qrt_norm *     nrm;
    qrt_gemm *     gemm;
    qrt_vk_cmd *   pass;
    qrt_vk_staging * stg;
    uint32_t       E;
    uint32_t       V;
    uint32_t       n_seg;
    double         last_ms;

    VkBuffer w;      VkDeviceMemory w_mem;      uint64_t w_bytes;
    VkBuffer norm_w; VkDeviceMemory norm_mem;
    VkBuffer qs;     VkDeviceMemory qs_mem;
    VkBuffer ds;     VkDeviceMemory ds_mem;
    VkBuffer logits; VkDeviceMemory logits_mem;
    VkBuffer part;   VkDeviceMemory part_mem;
    VkBuffer res;    VkDeviceMemory res_mem;
    VkBuffer mask_all;  VkDeviceMemory mask_all_mem;
    VkBuffer mask_cand; VkDeviceMemory mask_cand_mem;

    VkDescriptorSetLayout argm_dsl;
    VkPipelineLayout      argm_layout;
    VkPipeline            argm_max1;
    VkPipeline            argm_max2;
    VkDescriptorPool      argm_pool;
    VkCommandPool         argm_cmd_pool;   // unused sets are reset with the pool
};

namespace {

int alloc_buf(qrt_head * h, uint64_t bytes, VkBufferUsageFlags extra, VkBuffer * buf,
              VkDeviceMemory * mem, char * err, size_t err_size) {
    return qrt_vk_alloc(h->dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST | extra,
                        buf, mem, err, err_size);
}

// The weight is bigger than the staging buffer, so it goes over in staging-sized pieces.
int upload_chunks(qrt_head * h, VkBuffer dst, const unsigned char * src, uint64_t bytes,
                  char * err, size_t err_size) {
    const uint64_t step = qrt_vk_staging_bytes(h->stg);
    for (uint64_t off = 0; off < bytes; off += step) {
        const uint64_t n = bytes - off < step ? bytes - off : step;
        if (qrt_vk_upload(h->dev, h->stg, dst, off, src + off, n, err, err_size) != 0) {
            return -1;
        }
    }
    return 0;
}

int argm_set(qrt_head * h, const VkBuffer * bufs, uint32_t n, VkDescriptorSet * out,
             char * err, size_t err_size) {
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = h->argm_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &h->argm_dsl;
    if (vkAllocateDescriptorSets(h->dev->device, &dsai, out) != VK_SUCCESS) {
        return fail(err, err_size, "vkAllocateDescriptorSets (head argmax) failed");
    }
    return qrt_vk_write_set(h->dev, *out, bufs, n);
}

int argm_record(qrt_head * h, VkCommandBuffer cmd, VkPipeline pipe, const VkBuffer * bufs,
                uint32_t n_bufs, const uint32_t pc[4], uint32_t gx, uint32_t gy,
                char * err, size_t err_size) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (argm_set(h, bufs, n_bufs, &set, err, err_size) != 0) {
        return -1;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, h->argm_layout, 0, 1, &set,
                            0, nullptr);
    vkCmdPushConstants(cmd, h->argm_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc[0]) * 4,
                       pc);
    vkCmdDispatch(cmd, gx, gy, 1);
    return 0;
}

// The argmax writes its floats as bits, because a push-constant block and an SSBO of uints
// are what the shader can hand back without a second buffer type.
float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

} // namespace

uint64_t qrt_head_weight_bytes(uint32_t n_embd, uint32_t vocab) {
    return qrt_gemm_matrix_bytes(QRT_QUANT_Q8_0, vocab, n_embd);
}

int qrt_head_open(qrt_vk_dev * dev, const char * spv_dir, uint32_t n_embd, uint32_t vocab,
                  qrt_head ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || spv_dir == NULL || n_embd == 0 || vocab == 0) {
        return fail(err, err_size, "the head needs a device, shaders, n_embd and a vocabulary");
    }
    if ((n_embd % Q4K_QK) != 0u || (vocab % Q4K_TILE_N) != 0u) {
        return fail(err, err_size, "head shape n_embd=%u vocab=%u does not fit the GEMM tiles",
                    n_embd, vocab);
    }
    qrt_head * h = (qrt_head *) calloc(1, sizeof(qrt_head));
    if (h == NULL) {
        return fail(err, err_size, "out of memory for the output head");
    }
    h->dev = dev;
    h->E = n_embd;
    h->V = vocab;
    h->n_seg = (vocab + QRT_HEAD_ARGM_SPAN - 1u) / QRT_HEAD_ARGM_SPAN;

    if (qrt_gemm_open(dev, spv_dir, &h->gemm, err, err_size) != 0 ||
        qrt_norm_open(dev, spv_dir, &h->nrm, err, err_size) != 0 ||
        qrt_vk_cmd_create(dev, 1, &h->pass, err, err_size) != 0 ||
        qrt_vk_staging_create(dev, QRT_HEAD_STAGING_BYTES, &h->stg, err, err_size) != 0) {
        qrt_head_close(dev, h);
        return -1;
    }
    if (qrt_vk_make_set_layout(dev, 3, &h->argm_dsl, err, err_size) != 0 ||
        qrt_vk_make_layout(dev, h->argm_dsl, 4 * sizeof(uint32_t), &h->argm_layout, err,
                           err_size) != 0) {
        qrt_head_close(dev, h);
        return -1;
    }
    if (qrt_vk_make_pipeline(dev, h->argm_layout, spv_dir, "head_max1.spv", nullptr, 0,
                             &h->argm_max1, err, err_size) != 0 ||
        qrt_vk_make_pipeline(dev, h->argm_layout, spv_dir, "head_max2.spv", nullptr, 0,
                             &h->argm_max2, err, err_size) != 0) {
        qrt_head_close(dev, h);
        return -1;
    }
    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 3 * QRT_HEAD_ARGM_MAX_SETS;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = QRT_HEAD_ARGM_MAX_SETS;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psize;
    if (vkCreateDescriptorPool(dev->device, &dpci, nullptr, &h->argm_pool) != VK_SUCCESS) {
        qrt_head_close(dev, h);
        return fail(err, err_size, "vkCreateDescriptorPool (head argmax) failed");
    }
    const uint64_t plane_rows = QRT_HEAD_ROWS;
    const uint64_t norm_bytes = (uint64_t) QRT_HEAD_NORM_SLOTS * n_embd * 4;
    h->w_bytes = qrt_head_weight_bytes(n_embd, vocab);
    if (alloc_buf(h, h->w_bytes, 0, &h->w, &h->w_mem, err, err_size) != 0 ||
        alloc_buf(h, norm_bytes, 0, &h->norm_w, &h->norm_mem, err, err_size) != 0 ||
        alloc_buf(h, plane_rows * n_embd, QRT_VK_USAGE_TRANSFER_SRC, &h->qs, &h->qs_mem, err,
                  err_size) != 0 ||
        alloc_buf(h, plane_rows * (n_embd / 32u) * 4, QRT_VK_USAGE_TRANSFER_SRC, &h->ds,
                  &h->ds_mem, err, err_size) != 0 ||
        alloc_buf(h, plane_rows * (uint64_t) vocab * 4, QRT_VK_USAGE_TRANSFER_SRC, &h->logits,
                  &h->logits_mem, err, err_size) != 0 ||
        alloc_buf(h, plane_rows * h->n_seg * 4 * 4, 0, &h->part, &h->part_mem, err,
                  err_size) != 0 ||
        alloc_buf(h, plane_rows * 4 * 4, QRT_VK_USAGE_TRANSFER_SRC, &h->res, &h->res_mem, err,
                  err_size) != 0 ||
        alloc_buf(h, (uint64_t) vocab * 4, 0, &h->mask_all, &h->mask_all_mem, err,
                  err_size) != 0 ||
        alloc_buf(h, (uint64_t) vocab * 4, 0, &h->mask_cand, &h->mask_cand_mem, err,
                  err_size) != 0) {
        qrt_head_close(dev, h);
        return -1;
    }
    // Every entry may win until a draft step says otherwise; one fill, never again.
    std::vector<uint32_t> ones((size_t) vocab, 1u);
    if (upload_chunks(h, h->mask_all, (const unsigned char *) ones.data(),
                      (uint64_t) vocab * 4, err, err_size) != 0) {
        qrt_head_close(dev, h);
        return -1;
    }
    *out = h;
    return 0;
}

void qrt_head_close(qrt_vk_dev * dev, qrt_head * h) {
    if (h == NULL) {
        return;
    }
    if (dev != NULL) {
        const VkBuffer bufs[9] = { h->w, h->norm_w, h->qs, h->ds, h->logits, h->part, h->res,
                                   h->mask_all, h->mask_cand };
        VkDeviceMemory mems[9] = { h->w_mem, h->norm_mem, h->qs_mem, h->ds_mem, h->logits_mem,
                                   h->part_mem, h->res_mem, h->mask_all_mem, h->mask_cand_mem };
        for (uint32_t i = 0; i < 9; ++i) {
            if (mems[i] != VK_NULL_HANDLE) {
                qrt_vk_free(dev, bufs[i], mems[i]);
            }
        }
        if (h->argm_pool != VK_NULL_HANDLE)    vkDestroyDescriptorPool(dev->device, h->argm_pool, nullptr);
        if (h->argm_max1 != VK_NULL_HANDLE)    vkDestroyPipeline(dev->device, h->argm_max1, nullptr);
        if (h->argm_max2 != VK_NULL_HANDLE)    vkDestroyPipeline(dev->device, h->argm_max2, nullptr);
        if (h->argm_layout != VK_NULL_HANDLE)  vkDestroyPipelineLayout(dev->device, h->argm_layout, nullptr);
        if (h->argm_dsl != VK_NULL_HANDLE)     vkDestroyDescriptorSetLayout(dev->device, h->argm_dsl, nullptr);
        qrt_vk_staging_destroy(dev, h->stg);
        qrt_vk_cmd_destroy(dev, h->pass);
        qrt_norm_close(dev, h->nrm);
        qrt_gemm_close(dev, h->gemm);
    }
    free(h);
}

int qrt_head_set_norm(qrt_head * h, uint32_t slot, const float * w, char * err,
                     size_t err_size) {
    if (h == NULL || w == NULL || slot >= QRT_HEAD_NORM_SLOTS) {
        return fail(err, err_size, "the head has %u norm slots", (uint32_t) QRT_HEAD_NORM_SLOTS);
    }
    return qrt_vk_upload(h->dev, h->stg, h->norm_w, (uint64_t) slot * h->E * 4, w,
                         (uint64_t) h->E * 4, err, err_size);
}

int qrt_head_set_weights(qrt_head * h, const void * prepared, uint64_t bytes, char * err,
                        size_t err_size) {
    if (h == NULL || prepared == NULL) {
        return fail(err, err_size, "the head needs prepared output.weight rows");
    }
    if (bytes != h->w_bytes) {
        return fail(err, err_size, "output.weight is %llu bytes, the head holds %llu",
                    (unsigned long long) bytes, (unsigned long long) h->w_bytes);
    }
    return upload_chunks(h, h->w, (const unsigned char *) prepared, bytes, err, err_size);
}

int qrt_head_set_candidates(qrt_head * h, const uint32_t * ids, uint32_t n, char * err,
                            size_t err_size) {
    if (h == NULL || (ids == NULL && n != 0)) {
        return fail(err, err_size, "the candidate list needs ids");
    }
    std::vector<uint32_t> mask((size_t) h->V, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        if (ids[i] >= h->V) {
            return fail(err, err_size, "candidate %u is id %u, the vocabulary is %u", i, ids[i],
                        h->V);
        }
        mask[ids[i]] = 1u;
    }
    if (n == 0) {
        std::fill(mask.begin(), mask.end(), 1u);
    }
    return upload_chunks(h, h->mask_cand, (const unsigned char *) mask.data(),
                         (uint64_t) h->V * 4, err, err_size);
}

// The token tile the GEMM runs. Up to two rows the one-token kernel is cheaper: it reads the
// resident weight once per row, while the 32-row tile reads it once per pass and pays for
// thirty blank rows. Crossover at three rows.
uint32_t qrt_head_tile(uint32_t rows) {
    return rows <= 2u ? rows : QRT_HEAD_ROWS;
}

int qrt_head_run(qrt_head * h, VkBuffer x, uint32_t row0, uint32_t rows, uint32_t norm_slot,
                 int candidates, uint32_t * ids, float * best, float * second, char * err,
                 size_t err_size) {
    if (h == NULL || ids == NULL || x == VK_NULL_HANDLE) {
        return fail(err, err_size, "the head needs the module, a hidden plane and an id slot");
    }
    if (rows == 0 || rows > QRT_HEAD_ROWS || norm_slot >= QRT_HEAD_NORM_SLOTS) {
        return fail(err, err_size, "head run of %u rows at norm slot %u does not fit", rows,
                    norm_slot);
    }
    const double t0 = now_ms();
    const uint32_t m = qrt_head_tile(rows);
    if (qrt_vk_cmd_begin(h->pass, 0, err, err_size) != 0) {
        return -1;
    }
    VkCommandBuffer cmd = qrt_vk_cmd_buffer(h->pass, 0);
    const VkResult rr = vkResetDescriptorPool(h->dev->device, h->argm_pool, 0);
    if (rr != VK_SUCCESS) {
        return vk_fail(err, err_size, "vkResetDescriptorPool (head argmax)", rr);
    }
    if (qrt_norm_attach(h->nrm, 0, cmd, err, err_size) != 0 ||
        qrt_gemm_attach(h->gemm, 0, cmd, err, err_size) != 0) {
        return -1;
    }
    qrt_norm_args na;
    memset(&na, 0, sizeof(na));
    na.x = x;
    na.xbase = row0 * h->E;
    na.w = h->norm_w;
    na.wbase = norm_slot * h->E;
    na.qs = h->qs;
    na.ds = h->ds;
    na.k = h->E;
    na.tokens = m;
    na.eps = 1e-6f;
    na.quant = 1;
    na.no_out = 1;
    if (qrt_norm_record(h->nrm, 0, &na, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_cmd_record_barrier(cmd);
    qrt_gemm_args ga;
    memset(&ga, 0, sizeof(ga));
    ga.w = h->w;
    ga.a_qs = h->qs;
    ga.a_ds = h->ds;
    ga.out = h->logits;
    ga.n = h->V;
    ga.k = h->E;
    ga.m = m;
    ga.type = QRT_QUANT_Q8_0;
    ga.token_major = 1;
    if (qrt_gemm_record(h->gemm, &ga, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_cmd_record_barrier(cmd);
    const VkBuffer max1_bufs[3] = { h->logits, candidates != 0 ? h->mask_cand : h->mask_all,
                                    h->part };
    const uint32_t max1_pc[4] = { h->V, QRT_HEAD_ARGM_SPAN, 0u, 0u };
    if (argm_record(h, cmd, h->argm_max1, max1_bufs, 3, max1_pc, h->n_seg, rows, err,
                    err_size) != 0) {
        return -1;
    }
    qrt_vk_cmd_record_barrier(cmd);
    const VkBuffer max2_bufs[2] = { h->part, h->res };
    const uint32_t max2_pc[4] = { h->V, h->n_seg, 0u, 0u };
    if (argm_record(h, cmd, h->argm_max2, max2_bufs, 2, max2_pc, rows, 1, err, err_size) != 0) {
        return -1;
    }
    if (qrt_vk_cmd_submit(h->pass, 0, err, err_size) != 0) {
        return -1;
    }
    qrt_norm_detach(h->nrm);
    qrt_gemm_detach(h->gemm);
    if (qrt_vk_cmd_wait(h->pass, 0, err, err_size) != 0) {
        return -1;
    }
    std::vector<uint32_t> words((size_t) rows * 4u);
    if (qrt_vk_download(h->dev, h->stg, h->res, 0, words.data(),
                        (uint64_t) rows * 4u * 4u, err, err_size) != 0) {
        return -1;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t id = words[(size_t) r * 4u];
        if (id == 0xFFFFFFFFu) {
            return fail(err, err_size, "row %u of the head has no candidate the mask allows", r);
        }
        ids[r] = id;
        if (best != NULL) {
            best[r] = bits_to_float(words[(size_t) r * 4u + 1u]);
        }
        if (second != NULL) {
            second[r] = bits_to_float(words[(size_t) r * 4u + 2u]);
        }
    }
    h->last_ms = now_ms() - t0;
    return 0;
}

int qrt_head_logits_row(qrt_head * h, uint32_t row, float * out, char * err, size_t err_size) {
    if (h == NULL || out == NULL || row >= QRT_HEAD_ROWS) {
        return fail(err, err_size, "the head tile holds %u logits rows",
                    (uint32_t) QRT_HEAD_ROWS);
    }
    return qrt_vk_download(h->dev, h->stg, h->logits, (uint64_t) row * h->V * 4, out,
                           (uint64_t) h->V * 4, err, err_size);
}

int qrt_head_planes(qrt_head * h, uint32_t rows, unsigned char * qs, uint32_t * ds,
                    char * err, size_t err_size) {
    if (h == NULL || rows == 0 || rows > QRT_HEAD_ROWS) {
        return fail(err, err_size, "the head planes hold %u rows", (uint32_t) QRT_HEAD_ROWS);
    }
    const uint64_t qb = (uint64_t) rows * h->E;
    const uint64_t db = (uint64_t) rows * (h->E / 32u) * 4;
    if (qs != NULL && qrt_vk_download(h->dev, h->stg, h->qs, 0, qs, qb, err, err_size) != 0) {
        return -1;
    }
    if (ds != NULL && qrt_vk_download(h->dev, h->stg, h->ds, 0, ds, db, err, err_size) != 0) {
        return -1;
    }
    return 0;
}

uint32_t qrt_head_vocab(const qrt_head * h) { return h == NULL ? 0 : h->V; }
uint32_t qrt_head_embd(const qrt_head * h) { return h == NULL ? 0 : h->E; }
double   qrt_head_last_ms(const qrt_head * h) { return h == NULL ? 0.0 : h->last_ms; }

#endif // QRT_HAVE_VULKAN
