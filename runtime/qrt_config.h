#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Effective runtime configuration. The switches and their defaults are design
// section 7; a CLI flag overrides the matching QRT_* variable, which overrides
// the default. A value that is legal but costs something is a note, not an error.

enum qrt_cache_policy {
    QRT_CACHE_SLRU   = 0,
    QRT_CACHE_LRU    = 1,
    QRT_CACHE_STATIC = 2,
};

enum qrt_kv_precision {
    QRT_KV_Q4_0 = 0,
    QRT_KV_F16  = 1,
};

enum { QRT_MISS_SPLIT_AUTO = -1, QRT_ERR_SIZE = 256 };

struct qrt_config {
    int32_t      spec_k;          // tokens verified per pass
    int32_t      draft_vocab;     // 0 = full vocab
    int32_t      expert_cache_mib;
    int32_t      miss_split;      // QRT_MISS_SPLIT_AUTO, or the H2D share in percent
    int32_t      cache_policy;    // enum qrt_cache_policy
    int32_t      prefill_chunk;
    int32_t      kv_precision;    // enum qrt_kv_precision
    int32_t      kv_page_tokens;  // resident KV window
    int32_t      telemetry;       // 0 = off, 1 = one CSV row per pass
    const char * telemetry_path;  // not copied: argv and env live for the process
};

void qrt_config_defaults(qrt_config * cfg);

// Applies the environment, then argv. Returns 0 on success, -1 with a message in
// err on the first bad value; calling qrt_config_defaults again recovers.
int qrt_config_parse(qrt_config * cfg, int argc, char ** argv, char * err, size_t err_size);

// Rejects values that would silently change the meaning of a switch.
int qrt_config_validate(const qrt_config * cfg, char * err, size_t err_size);

// Design cross-checks that are legal but worth printing; returns how many fired.
int qrt_config_notes(const qrt_config * cfg, FILE * f);

void qrt_config_print_switches(FILE * f);
void qrt_config_print_effective(const qrt_config * cfg, FILE * f);

const char * qrt_cache_policy_name(int32_t policy);
const char * qrt_kv_precision_name(int32_t precision);

// Switch table accessors, in the order of design section 7.
int qrt_config_switch_count(void);
const char * qrt_config_env_name(int index);
const char * qrt_config_flag_name(int index);
