#include "moe.h"

#include "ffn.h"
#include "gguf_index.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <new>
#include <vector>

namespace {

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (err != NULL && err_size != 0) {
        vsnprintf(err, err_size, fmt, ap);
    }
    va_end(ap);
    return -1;
}

bool bytes_for(uint32_t tokens, uint64_t per_token, uint64_t * out) {
    if (tokens == 0 || per_token > UINT64_MAX / tokens) {
        return false;
    }
    *out = (uint64_t) tokens * per_token;
    return true;
}

} // namespace

struct qrt_batch_moe {
    qrt_vk_dev * dev;
    uint32_t tokens;
    uint32_t n_embd;
    uint32_t top_k;
    uint32_t n_expert;
    qrt_ffn * ffn;
    VkBuffer xnorm;
    VkDeviceMemory xnorm_mem;
    VkBuffer qs;
    VkDeviceMemory qs_mem;
    VkBuffer ds;
    VkDeviceMemory ds_mem;
    VkBuffer moe;
    VkDeviceMemory moe_mem;
    qrt_vk_staging * stg;
    qrt_vk_cmd * cmd;
};

int qrt_batch_moe_open(qrt_vk_dev * dev, const char * gguf_path, const char * store_prefix,
                 const char * spv_dir, uint32_t layer, uint32_t tokens, qrt_batch_moe ** out,
                 char * err, size_t err_size) {
    if (out == NULL) {
        return fail(err, err_size, "moe needs an output pointer");
    }
    *out = NULL;
    if (dev == NULL || gguf_path == NULL || store_prefix == NULL || spv_dir == NULL ||
        tokens == 0 || tokens > 256) {
        return fail(err, err_size, "the MoE batch needs a device, GGUF, store, SPIR-V and 1..256 tokens");
    }
    qrt_gguf * gguf = NULL;
    if (qrt_gguf_open(gguf_path, &gguf, err, err_size) != 0) {
        return -1;
    }
    const qrt_gguf_hparams hp = *qrt_gguf_hparams_of(gguf);
    qrt_gguf_close(gguf);
    if (hp.n_embd <= 0 || hp.n_embd % 32 != 0 || hp.expert_count <= 0 ||
        hp.expert_used_count <= 0 || hp.expert_ff <= 0) {
        return fail(err, err_size, "the MoE batch needs n_embd, experts and expert_ff");
    }
    qrt_ffn_cfg cfg{};
    cfg.tokens = tokens;
    cfg.n_embd = (uint32_t) hp.n_embd;
    cfg.n_expert = (uint32_t) hp.expert_count;
    cfg.top_k = (uint32_t) hp.expert_used_count;
    cfg.ffn = (uint32_t) hp.expert_ff;
    cfg.scale = 1.0f;
    cfg.routed_only = 1;
    qrt_batch_moe * m = new (std::nothrow) qrt_batch_moe{};
    if (m == NULL) {
        return fail(err, err_size, "out of memory for the MoE batch");
    }
    m->dev = dev;
    m->tokens = tokens;
    m->n_embd = cfg.n_embd;
    m->top_k = cfg.top_k;
    m->n_expert = cfg.n_expert;
    if (qrt_ffn_open(dev, gguf_path, store_prefix, spv_dir, layer, &cfg, &m->ffn, err,
                     err_size) != 0) {
        qrt_batch_moe_close(dev, m);
        return -1;
    }
    uint64_t xnorm_bytes = 0, qs_bytes = 0, ds_bytes = 0, moe_bytes = 0;
    if (!bytes_for(tokens, (uint64_t) m->n_embd * 4u, &xnorm_bytes) ||
        !bytes_for(tokens, m->n_embd, &qs_bytes) ||
        !bytes_for(tokens, (uint64_t) (m->n_embd / 32u) * 4u, &ds_bytes) ||
        !bytes_for(tokens, (uint64_t) m->n_embd * 4u, &moe_bytes)) {
        qrt_batch_moe_close(dev, m);
        return fail(err, err_size, "the MoE batch buffers overflow");
    }
    const struct {
        uint64_t bytes;
        VkBuffer * buf;
        VkDeviceMemory * mem;
        const char * what;
    } slots[] = {
        { xnorm_bytes, &m->xnorm, &m->xnorm_mem, "xnorm" },
        { qs_bytes, &m->qs, &m->qs_mem, "Q8_1 quants" },
        { ds_bytes, &m->ds, &m->ds_mem, "Q8_1 deltas" },
        { moe_bytes, &m->moe, &m->moe_mem, "routed output" },
    };
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        if (qrt_vk_alloc(dev, slots[i].bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         slots[i].buf, slots[i].mem, err, err_size) != 0) {
            qrt_batch_moe_close(dev, m);
            return fail(err, err_size, "allocating the MoE %s buffer", slots[i].what);
        }
    }
    if (qrt_vk_staging_create(dev, xnorm_bytes, &m->stg, err, err_size) != 0) {
        qrt_batch_moe_close(dev, m);
        return -1;
    }
    if (qrt_vk_cmd_create(dev, 1, &m->cmd, err, err_size) != 0) {
        qrt_batch_moe_close(dev, m);
        return -1;
    }
    *out = m;
    return 0;
}

void qrt_batch_moe_close(qrt_vk_dev * dev, qrt_batch_moe * m) {
    if (m == NULL) {
        return;
    }
    if (m->ffn != NULL) {
        qrt_ffn_close(dev, m->ffn);
    }
    if (m->cmd != NULL) {
        qrt_vk_cmd_destroy(dev, m->cmd);
    }
    if (m->stg != NULL) {
        qrt_vk_staging_destroy(dev, m->stg);
    }
    const struct {
        VkBuffer buf;
        VkDeviceMemory mem;
    } slots[] = {
        { m->xnorm, m->xnorm_mem }, { m->qs, m->qs_mem },
        { m->ds, m->ds_mem }, { m->moe, m->moe_mem },
    };
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        if (slots[i].mem != VK_NULL_HANDLE) {
            qrt_vk_free(dev, slots[i].buf, slots[i].mem);
        }
    }
    delete m;
}

int qrt_batch_moe_run(qrt_batch_moe * m, const float * xnorm, const signed char * qs, const uint32_t * ds,
                float * moe, uint32_t * ids, float * weights, char * err, size_t err_size) {
    if (m == NULL || xnorm == NULL || qs == NULL || ds == NULL) {
        return fail(err, err_size, "the MoE batch needs xnorm and its Q8_1 planes");
    }
    const uint64_t xnorm_bytes = (uint64_t) m->tokens * m->n_embd * 4u;
    const uint64_t qs_bytes = (uint64_t) m->tokens * m->n_embd;
    const uint64_t ds_bytes = (uint64_t) m->tokens * (m->n_embd / 32u) * 4u;
    const uint64_t moe_bytes = xnorm_bytes;
    if (qrt_vk_upload(m->dev, m->stg, m->xnorm, 0, xnorm, xnorm_bytes, err, err_size) != 0 ||
        qrt_vk_upload(m->dev, m->stg, m->qs, 0, qs, qs_bytes, err, err_size) != 0 ||
        qrt_vk_upload(m->dev, m->stg, m->ds, 0, ds, ds_bytes, err, err_size) != 0) {
        return -1;
    }
    qrt_ffn_in in{};
    in.xnorm = m->xnorm;
    in.x_qs = m->qs;
    in.x_ds = m->ds;
    if (qrt_vk_cmd_begin(m->cmd, 0, err, err_size) != 0 ||
        qrt_ffn_begin(m->ffn, 0, qrt_vk_cmd_buffer(m->cmd, 0), err, err_size) != 0 ||
        qrt_ffn_route(m->ffn, &in, err, err_size) != 0 ||
        qrt_vk_cmd_submit(m->cmd, 0, err, err_size) != 0 ||
        qrt_vk_cmd_wait(m->cmd, 0, err, err_size) != 0 ||
        qrt_ffn_group(m->ffn, err, err_size) != 0) {
        return -1;
    }
    if (qrt_vk_cmd_begin(m->cmd, 0, err, err_size) != 0 ||
        qrt_ffn_begin(m->ffn, 0, qrt_vk_cmd_buffer(m->cmd, 0), err, err_size) != 0 ||
        qrt_ffn_experts(m->ffn, err, err_size) != 0 ||
        qrt_vk_cmd_submit(m->cmd, 0, err, err_size) != 0 ||
        qrt_vk_cmd_wait(m->cmd, 0, err, err_size) != 0) {
        return -1;
    }
    const qrt_ffn_bufs * bufs = qrt_ffn_buffers(m->ffn);
    if ((moe != NULL &&
         qrt_vk_download(m->dev, m->stg, bufs->moe, 0, moe, moe_bytes, err, err_size) != 0) ||
        (ids != NULL &&
         qrt_vk_download(m->dev, m->stg, bufs->ids, 0, ids,
                         (uint64_t) m->tokens * m->top_k * 4u, err, err_size) != 0) ||
        (weights != NULL &&
         qrt_vk_download(m->dev, m->stg, bufs->weights, 0, weights,
                         (uint64_t) m->tokens * m->top_k * 4u, err, err_size) != 0)) {
        return -1;
    }
    return 0;
}

int qrt_batch_moe_ref_cpu(qrt_batch_moe * m, const float * xnorm, const signed char * qs,
                    const uint32_t * ds, float * moe, uint32_t * ids, float * weights,
                    char * err, size_t err_size) {
    if (m == NULL || xnorm == NULL || qs == NULL || ds == NULL || ids == NULL) {
        return fail(err, err_size, "the MoE reference needs xnorm, planes and ids");
    }
    return qrt_ffn_ref_cpu(m->ffn, xnorm, qs, ds, m->tokens, NULL, ids, weights, NULL, NULL,
                           moe, err, err_size);
}

uint32_t qrt_batch_moe_tokens(const qrt_batch_moe * m) {
    return m != NULL ? m->tokens : 0;
}

uint32_t qrt_batch_moe_experts(const qrt_batch_moe * m) {
    return m != NULL ? m->n_expert : 0;
}

uint32_t qrt_batch_moe_experts_used(const qrt_batch_moe * m) {
    return m != NULL && m->ffn != NULL ? qrt_ffn_experts_used(m->ffn) : 0;
}

uint32_t qrt_batch_moe_pairs(const qrt_batch_moe * m) {
    return m != NULL && m->ffn != NULL ? qrt_ffn_pairs(m->ffn) : 0;
}

uint32_t qrt_batch_moe_rows(const qrt_batch_moe * m) {
    return m != NULL && m->ffn != NULL ? qrt_ffn_rows(m->ffn) : 0;
}
