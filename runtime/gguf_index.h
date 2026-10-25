#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Read-only GGUF index: metadata, tensor table and absolute byte offsets. The
// runtime reads the container itself (design 3.3) and nothing here aborts: a
// truncated, crafted or foreign file comes back as an error string.

struct qrt_gguf_tensor {
    char *   name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;   // absolute file offset of the first byte
    uint64_t nbytes;
};

enum qrt_gguf_hp {
    QRT_HP_BLOCK_COUNT         = 1u << 0,
    QRT_HP_EXPERT_COUNT        = 1u << 1,
    QRT_HP_EXPERT_USED_COUNT   = 1u << 2,
    QRT_HP_N_EMBD              = 1u << 3,
    QRT_HP_EXPERT_FF           = 1u << 4,
    QRT_HP_NEXTN               = 1u << 5,
    QRT_HP_HEAD_COUNT_KV       = 1u << 6,
    QRT_HP_KEY_LENGTH          = 1u << 7,
    QRT_HP_VALUE_LENGTH        = 1u << 8,
    QRT_HP_CONTEXT_LENGTH      = 1u << 9,
    QRT_HP_FULL_ATTN_INTERVAL  = 1u << 10,
    QRT_HP_ROPE_DIMENSION      = 1u << 11,
    QRT_HP_ROPE_FREQ_BASE      = 1u << 12,
    QRT_HP_ROPE_SECTIONS       = 1u << 13,
    QRT_HP_HEAD_COUNT          = 1u << 14,
};

struct qrt_gguf_hparams {
    char     arch[64];
    int32_t  block_count;
    int32_t  expert_count;
    int32_t  expert_used_count;
    int32_t  n_embd;
    int32_t  expert_ff;
    int32_t  nextn;
    int32_t  head_count_kv;
    int32_t  head_count;          // attention.head_count
    int32_t  key_length;
    int32_t  value_length;
    int32_t  context_length;
    int32_t  full_attn_interval;
    int32_t  rope_dimension;      // rope.dimension_count (n_rot)
    float    rope_freq_base;      // rope.freq_base
    int32_t  rope_sections[4];    // rope.dimension_sections, the IMRoPE sectors
    uint32_t present;   // enum qrt_gguf_hp bits
};

// The account bench/tools/gguf_dims.py prints, so the two can be compared.
struct qrt_gguf_account {
    uint64_t expert_bytes;
    uint64_t dense_bytes;
    uint64_t token_embd_bytes;
    double   per_expert_bytes;
    uint32_t expert_layers;
    uint32_t main_layers;
    uint32_t full_attn_main;
    uint32_t full_attn_nextn;
};

struct qrt_gguf;

int qrt_gguf_open(const char * path, qrt_gguf ** out, char * err, size_t err_size);
void qrt_gguf_close(qrt_gguf * g);

uint32_t qrt_gguf_version(const qrt_gguf * g);
uint32_t qrt_gguf_tensor_count(const qrt_gguf * g);
uint64_t qrt_gguf_data_offset(const qrt_gguf * g);
uint64_t qrt_gguf_file_size(const qrt_gguf * g);
const qrt_gguf_tensor * qrt_gguf_tensor_at(const qrt_gguf * g, uint32_t index);
const qrt_gguf_tensor * qrt_gguf_find(const qrt_gguf * g, const char * name);
const qrt_gguf_hparams * qrt_gguf_hparams_of(const qrt_gguf * g);
const char * qrt_gguf_type_name(uint32_t type);

// A missing tensor is a caller error, not a crash: -1 with a message.
int qrt_gguf_require(const qrt_gguf * g, const char * name, const qrt_gguf_tensor ** out,
                     char * err, size_t err_size);

// The index parses metadata and closes the file; a read reopens it once and keeps
// it open until qrt_gguf_close. Both are short reads and seek failures, never aborts.
int qrt_gguf_read_tensor(qrt_gguf * g, const qrt_gguf_tensor * t, void * dst,
                         size_t cap, char * err, size_t err_size);
int qrt_gguf_read_range(qrt_gguf * g, uint64_t file_offset, void * dst, size_t bytes,
                        char * err, size_t err_size);

int qrt_gguf_account_compute(const qrt_gguf * g, qrt_gguf_account * out, char * err, size_t err_size);

// Machine-readable report: '#' lines, then one CSV row per tensor.
void qrt_gguf_print_report(const qrt_gguf * g, FILE * f);
