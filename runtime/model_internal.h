#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "gguf_index.h"
#include "vk_dev.h"
#include "vk_transfer.h"

// Internals the model run, the host weight math and the decode slice share. Only files in
// the model module include this header.

struct qrt_gguf;

struct qrt_model_run_owner {
    qrt_vk_dev * dev = nullptr;
    qrt_gguf * g = nullptr;
    qrt_vk_staging * stg = nullptr;

    qrt_model_run_owner() = default;
    qrt_model_run_owner(const qrt_model_run_owner &) = delete;
    qrt_model_run_owner & operator=(const qrt_model_run_owner &) = delete;
    ~qrt_model_run_owner() {
        if (stg != nullptr) qrt_vk_staging_destroy(dev, stg);
        if (g != nullptr) qrt_gguf_close(g);
        if (dev != nullptr) qrt_vk_close(dev);
    }
};

// The q8_0 host rows: one row per prompt token, then the whole-vocabulary head over the
// final norm rows (model.h keeps token_embd out of the dense bucket).
int qrt_model_embed_rows(qrt_gguf * g, const qrt_gguf_tensor * emb, uint32_t E, uint32_t V,
                         const uint32_t * ids, uint32_t n_tokens, float * x,
                         char * err, size_t err_size);
int qrt_model_head_rows(qrt_gguf * g, const qrt_gguf_tensor * outw, const float * onorm,
                        const float * xa, const float * xb, uint32_t E, uint32_t V,
                        uint32_t n_tokens, uint32_t n_rows, uint32_t start,
                        std::vector<float> * la, std::vector<float> * lb,
                        char * err, size_t err_size);
