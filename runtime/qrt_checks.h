#pragma once

#include <stdint.h>

// Check commands grouped by the subsystem they exercise. qrt_main only parses options
// and dispatches; each file below owns one group of checks and its GPU resources.

struct qrt_config;

int qrt_check_selftest(void);

int qrt_check_moe(const qrt_config * cfg, const char * prefix, const char * spv_dir,
                  uint32_t layer, uint32_t expert, uint32_t rows);
int qrt_check_dense_gemm(const qrt_config * cfg, const char * gguf_path,
                         const char * tensor_name, uint32_t tokens, uint32_t dense_rows,
                         const char * spv_dir);
int qrt_check_norm(const qrt_config * cfg, const char * gguf_path, const char * norm_name,
                   const char * proj_name, uint32_t tokens, const char * spv_dir);
int qrt_check_prefill(const qrt_config * cfg, const char * prefix, const char * spv,
                      uint32_t slots, uint32_t layers, int use_moe, int layer_batch,
                      int prefill_pipeline, int prefill_group_kib, int prefill_readers,
                      int prefill_chunks);
int qrt_check_dense_residency(const qrt_config * cfg, const char * gguf_path);
int qrt_check_vk_io(const qrt_config * cfg, const char * prefix);
int qrt_check_moe_batch(const qrt_config * cfg, const char * gguf_path,
                        const char * store_prefix, uint32_t layer, const char * spv_dir);
int qrt_check_cache(const qrt_config * cfg, const char * prefix, const char * trace_path,
                    uint32_t decode_tokens, uint32_t protected_percent);
