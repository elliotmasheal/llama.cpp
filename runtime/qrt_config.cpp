#include "qrt_config.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum {
    OPT_SPEC_K = 0,
    OPT_DRAFT_VOCAB,
    OPT_EXPERT_CACHE_MIB,
    OPT_MISS_SPLIT,
    OPT_CACHE_POLICY,
    OPT_PREFILL_CHUNK,
    OPT_KV_PRECISION,
    OPT_KV_PAGE_TOKENS,
    OPT_TELEMETRY,
    OPT_TELEMETRY_PATH,
    OPT_VRAM_TIER_MIB,
    OPT_COUNT,
};

struct qrt_opt_desc {
    const char * flag;
    const char * env;
    const char * want;
};

static const struct qrt_opt_desc OPT_DESC[OPT_COUNT] = {
    { "--spec-k",           "QRT_SPEC_K",           "1, 2, 4, 5 or 8" },
    { "--draft-vocab",      "QRT_DRAFT_VOCAB",      "0 (full vocab) or 1024..262144" },
    { "--expert-cache-mib", "QRT_EXPERT_CACHE_MIB", "0..8176 MiB" },
    { "--miss-split",       "QRT_MISS_SPLIT",       "auto or 0..100 percent to H2D" },
    { "--cache-policy",     "QRT_CACHE_POLICY",     "slru, lru or static" },
    { "--prefill-chunk",    "QRT_PREFILL_CHUNK",    "1..16384 tokens" },
    { "--kv-precision",     "QRT_KV_PRECISION",     "q4_0 or f16" },
    { "--kv-page-tokens",   "QRT_KV_PAGE_TOKENS",   "32..1048576 tokens, multiple of 32" },
    { "--telemetry",        "QRT_TELEMETRY",        "0 or 1" },
    { "--telemetry-path",   "QRT_TELEMETRY_PATH",   "a writable file path" },
    { "--vram-tier-mib",    "QRT_VRAM_TIER_MIB",    "0 = auto, or a VRAM cap in MiB" },
};

void qrt_config_defaults(qrt_config * cfg) {
    cfg->spec_k          = 4;
    cfg->draft_vocab     = 16384;
    cfg->expert_cache_mib = 4198;
    cfg->miss_split      = QRT_MISS_SPLIT_AUTO;
    cfg->cache_policy    = QRT_CACHE_SLRU;
    cfg->prefill_chunk   = 2048;
    cfg->kv_precision    = QRT_KV_Q4_0;
    cfg->kv_page_tokens  = 16384;
    cfg->telemetry       = 0;
    cfg->telemetry_path  = "qrt-telemetry.csv";
    cfg->vram_tier_mib   = 0;
}

const char * qrt_cache_policy_name(int32_t policy) {
    switch (policy) {
        case QRT_CACHE_SLRU:   return "slru";
        case QRT_CACHE_LRU:    return "lru";
        case QRT_CACHE_STATIC: return "static";
    }
    return "?";
}

const char * qrt_kv_precision_name(int32_t precision) {
    switch (precision) {
        case QRT_KV_Q4_0: return "q4_0";
        case QRT_KV_F16:  return "f16";
    }
    return "?";
}

int qrt_config_switch_count(void) {
    return OPT_COUNT;
}

const char * qrt_config_env_name(int index) {
    return index >= 0 && index < OPT_COUNT ? OPT_DESC[index].env : "";
}

const char * qrt_config_flag_name(int index) {
    return index >= 0 && index < OPT_COUNT ? OPT_DESC[index].flag : "";
}

// Strict: the whole string must be an integer in range, so "4x" is not 4.
static int parse_int(const char * s, long lo, long hi, int32_t * out) {
    if (s == NULL || *s == '\0') {
        return -1;
    }
    char * end = NULL;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi) {
        return -1;
    }
    *out = (int32_t) v;
    return 0;
}

static int parse_policy(const char * s, int32_t * out) {
    for (int32_t p = QRT_CACHE_SLRU; p <= QRT_CACHE_STATIC; ++p) {
        if (strcmp(s, qrt_cache_policy_name(p)) == 0) {
            *out = p;
            return 0;
        }
    }
    return -1;
}

static int parse_precision(const char * s, int32_t * out) {
    for (int32_t p = QRT_KV_Q4_0; p <= QRT_KV_F16; ++p) {
        if (strcmp(s, qrt_kv_precision_name(p)) == 0) {
            *out = p;
            return 0;
        }
    }
    return -1;
}

// Parses one value into a local and assigns only on success, so a rejected value
// leaves the previous configuration in place.
static int apply_opt(qrt_config * cfg, int opt, const char * value, char * err, size_t err_size) {
    int32_t v = 0;
    int ok = -1;

    switch (opt) {
        case OPT_SPEC_K:
            ok = parse_int(value, 1, 8, &v);
            if (ok == 0 && !(v == 1 || v == 2 || v == 4 || v == 5 || v == 8)) {
                ok = -1;
            }
            if (ok == 0) {
                cfg->spec_k = v;
            }
            break;
        case OPT_DRAFT_VOCAB:
            ok = parse_int(value, 0, 262144, &v);
            if (ok == 0 && v != 0 && v < 1024) {
                ok = -1;
            }
            if (ok == 0) {
                cfg->draft_vocab = v;
            }
            break;
        case OPT_EXPERT_CACHE_MIB:
            ok = parse_int(value, 0, 8176, &v);
            if (ok == 0) {
                cfg->expert_cache_mib = v;
            }
            break;
        case OPT_MISS_SPLIT:
            if (value != NULL && strcmp(value, "auto") == 0) {
                cfg->miss_split = QRT_MISS_SPLIT_AUTO;
                ok = 0;
                break;
            }
            ok = parse_int(value, 0, 100, &v);
            if (ok == 0) {
                cfg->miss_split = v;
            }
            break;
        case OPT_CACHE_POLICY:
            ok = value != NULL ? parse_policy(value, &v) : -1;
            if (ok == 0) {
                cfg->cache_policy = v;
            }
            break;
        case OPT_PREFILL_CHUNK:
            ok = parse_int(value, 1, 16384, &v);
            if (ok == 0) {
                cfg->prefill_chunk = v;
            }
            break;
        case OPT_KV_PRECISION:
            ok = value != NULL ? parse_precision(value, &v) : -1;
            if (ok == 0) {
                cfg->kv_precision = v;
            }
            break;
        case OPT_KV_PAGE_TOKENS:
            ok = parse_int(value, 32, 1048576, &v);
            if (ok == 0 && (v % 32) != 0) {
                ok = -1;
            }
            if (ok == 0) {
                cfg->kv_page_tokens = v;
            }
            break;
        case OPT_TELEMETRY:
            ok = parse_int(value, 0, 1, &v);
            if (ok == 0) {
                cfg->telemetry = v;
            }
            break;
        case OPT_TELEMETRY_PATH:
            ok = (value != NULL && *value != '\0') ? 0 : -1;
            if (ok == 0) {
                cfg->telemetry_path = value;
            }
            break;
        case OPT_VRAM_TIER_MIB:
            ok = parse_int(value, 0, 32768, &v);
            if (ok == 0) {
                cfg->vram_tier_mib = v;
            }
            break;
        default:
            ok = -1;
            break;
    }

    if (ok != 0) {
        snprintf(err, err_size, "%s: bad value '%s' (want %s)",
                 OPT_DESC[opt].flag, value != NULL ? value : "", OPT_DESC[opt].want);
        return -1;
    }
    return 0;
}

static int find_opt(const char * flag, size_t len) {
    for (int i = 0; i < OPT_COUNT; ++i) {
        if (strlen(OPT_DESC[i].flag) == len && strncmp(OPT_DESC[i].flag, flag, len) == 0) {
            return i;
        }
    }
    return -1;
}

int qrt_config_parse(qrt_config * cfg, int argc, char ** argv, char * err, size_t err_size) {
    for (int i = 0; i < OPT_COUNT; ++i) {
        const char * v = getenv(OPT_DESC[i].env);
        if (v != NULL && *v != '\0' && apply_opt(cfg, i, v, err, err_size) != 0) {
            return -1;
        }
    }

    // argv order matters: the last occurrence of a flag wins.
    for (int a = 1; a < argc; ++a) {
        const char * arg = argv[a];
        if (arg == NULL || strncmp(arg, "--", 2) != 0) {
            continue;
        }
        const char * eq = strchr(arg, '=');
        const size_t name_len = eq != NULL ? (size_t) (eq - arg) : strlen(arg);
        const int opt = find_opt(arg, name_len);
        if (opt < 0) {
            snprintf(err, err_size, "unknown option '%s' (see --help)", arg);
            return -1;
        }
        const char * value = NULL;
        if (eq != NULL) {
            value = eq + 1;
        } else if (a + 1 < argc) {
            value = argv[++a];
        } else {
            snprintf(err, err_size, "%s: missing value (want %s)", OPT_DESC[opt].flag, OPT_DESC[opt].want);
            return -1;
        }
        if (apply_opt(cfg, opt, value, err, err_size) != 0) {
            return -1;
        }
    }
    return 0;
}

int qrt_config_validate(const qrt_config * cfg, char * err, size_t err_size) {
    if (!(cfg->spec_k == 1 || cfg->spec_k == 2 || cfg->spec_k == 4 || cfg->spec_k == 5 || cfg->spec_k == 8)) {
        snprintf(err, err_size, "QRT_SPEC_K %d is not one of 1, 2, 4, 5, 8", cfg->spec_k);
        return -1;
    }
    if (cfg->draft_vocab < 0 || (cfg->draft_vocab > 0 && cfg->draft_vocab < 1024)) {
        snprintf(err, err_size, "QRT_DRAFT_VOCAB %d is neither 0 nor at least 1024", cfg->draft_vocab);
        return -1;
    }
    if (cfg->expert_cache_mib < 0 || cfg->expert_cache_mib > 8176) {
        snprintf(err, err_size, "QRT_EXPERT_CACHE_MIB %d is outside 0..8176", cfg->expert_cache_mib);
        return -1;
    }
    if (cfg->miss_split < 0 && cfg->miss_split != QRT_MISS_SPLIT_AUTO) {
        snprintf(err, err_size, "QRT_MISS_SPLIT %d is neither auto nor 0..100", cfg->miss_split);
        return -1;
    }
    if (cfg->miss_split > 100) {
        snprintf(err, err_size, "QRT_MISS_SPLIT %d is above 100 percent", cfg->miss_split);
        return -1;
    }
    if (cfg->cache_policy < QRT_CACHE_SLRU || cfg->cache_policy > QRT_CACHE_STATIC) {
        snprintf(err, err_size, "QRT_CACHE_POLICY %d is not slru, lru or static", cfg->cache_policy);
        return -1;
    }
    if (cfg->prefill_chunk < 1 || cfg->prefill_chunk > 16384) {
        snprintf(err, err_size, "QRT_PREFILL_CHUNK %d is outside 1..16384", cfg->prefill_chunk);
        return -1;
    }
    if (cfg->kv_precision != QRT_KV_Q4_0 && cfg->kv_precision != QRT_KV_F16) {
        snprintf(err, err_size, "QRT_KV_PRECISION %d is not q4_0 or f16", cfg->kv_precision);
        return -1;
    }
    if (cfg->kv_page_tokens < 32 || (cfg->kv_page_tokens % 32) != 0 || cfg->kv_page_tokens > 1048576) {
        snprintf(err, err_size, "QRT_KV_PAGE_TOKENS %d is not a multiple of 32 in 32..1048576",
                 cfg->kv_page_tokens);
        return -1;
    }
    if (cfg->telemetry != 0 && cfg->telemetry != 1) {
        snprintf(err, err_size, "QRT_TELEMETRY %d is neither 0 nor 1", cfg->telemetry);
        return -1;
    }
    if (cfg->telemetry == 1 && (cfg->telemetry_path == NULL || *cfg->telemetry_path == '\0')) {
        snprintf(err, err_size, "QRT_TELEMETRY_PATH is empty while telemetry is on");
        return -1;
    }
    if (cfg->vram_tier_mib < 0 || cfg->vram_tier_mib > 32768) {
        snprintf(err, err_size, "QRT_VRAM_TIER_MIB %d is outside 0..32768", cfg->vram_tier_mib);
        return -1;
    }
    return 0;
}

int qrt_config_notes(const qrt_config * cfg, FILE * f) {
    int n = 0;
    if (cfg->prefill_chunk < 1420) {
        fprintf(f, "note: QRT_PREFILL_CHUNK %d is below 1420, PP 1000 needs chunk >= 1420 at 6 TFLOPS (design 4.5)\n",
                cfg->prefill_chunk);
        ++n;
    }
    if (cfg->expert_cache_mib > 4198) {
        fprintf(f, "note: QRT_EXPERT_CACHE_MIB %d is above the 21.9%% budget of 4198 MiB (design 5.5)\n",
                cfg->expert_cache_mib);
        ++n;
    }
    if (cfg->expert_cache_mib == 0) {
        fprintf(f, "note: QRT_EXPERT_CACHE_MIB 0 runs every expert as a miss; comparison arm only\n");
        ++n;
    }
    if (cfg->draft_vocab == 0) {
        fprintf(f, "note: QRT_DRAFT_VOCAB 0 costs about 3.2 ms per draft step (design 4.1)\n");
        ++n;
    }
    if (cfg->spec_k == 5) {
        fprintf(f, "note: QRT_SPEC_K 5 is a speed arm only, k=4 is the acceptance width (design 3.1, U-7)\n");
        ++n;
    }
    if (cfg->spec_k == 8) {
        fprintf(f, "note: QRT_SPEC_K 8 is a speed arm only, it reads more expert bytes than k=4 (design 3.1)\n");
        ++n;
    }
    if (cfg->kv_precision == QRT_KV_F16) {
        fprintf(f, "note: QRT_KV_PRECISION f16 drops the cache budget to about 20.6%% (design 5.5)\n");
        ++n;
    }
    if (cfg->kv_page_tokens < 16384) {
        fprintf(f, "note: QRT_KV_PAGE_TOKENS %d is below the tested 16384 window (RESULTS section 10)\n",
                cfg->kv_page_tokens);
        ++n;
    }
    return n;
}

void qrt_config_print_switches(FILE * f) {
    qrt_config d;
    qrt_config_defaults(&d);
    char def[OPT_COUNT][32];

    snprintf(def[OPT_SPEC_K], 32, "%d", d.spec_k);
    snprintf(def[OPT_DRAFT_VOCAB], 32, "%d", d.draft_vocab);
    snprintf(def[OPT_EXPERT_CACHE_MIB], 32, "%d", d.expert_cache_mib);
    snprintf(def[OPT_MISS_SPLIT], 32, "%s", d.miss_split == QRT_MISS_SPLIT_AUTO ? "auto" : "percent");
    snprintf(def[OPT_CACHE_POLICY], 32, "%s", qrt_cache_policy_name(d.cache_policy));
    snprintf(def[OPT_PREFILL_CHUNK], 32, "%d", d.prefill_chunk);
    snprintf(def[OPT_KV_PRECISION], 32, "%s", qrt_kv_precision_name(d.kv_precision));
    snprintf(def[OPT_KV_PAGE_TOKENS], 32, "%d", d.kv_page_tokens);
    snprintf(def[OPT_TELEMETRY], 32, "%d", d.telemetry);
    snprintf(def[OPT_TELEMETRY_PATH], 32, "%s", d.telemetry_path);
    snprintf(def[OPT_VRAM_TIER_MIB], 32, "%d", d.vram_tier_mib);

    fprintf(f, "qwen36 runtime switches (design section 7); a CLI flag overrides its QRT_* variable.\n\n");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "environment", "flag", "default", "meaning");
    for (int i = 0; i < OPT_COUNT; ++i) {
        fprintf(f, "  %-22s %-20s %-19s %s\n", OPT_DESC[i].env, OPT_DESC[i].flag, def[i], OPT_DESC[i].want);
    }
    fprintf(f, "\n  %-22s %-20s %-19s %s\n", "-", "--print-config", "-", "print the effective values and the notes");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--selftest", "-", "check the switches and the telemetry sink");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--gguf-index <path>", "-", "print the GGUF tensor table and account");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--require-tensor <name>", "-", "exit 2 when the tensor is missing");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--vk-info", "-", "device, memory types and the VRAM plan");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--vk-alloc-probe <mib>", "-", "allocate once and re-read the budget");
    fprintf(f, "  %-22s %-20s %-19s %s\n", "-", "--help, -h", "-", "this table");
}

void qrt_config_print_effective(const qrt_config * cfg, FILE * f) {
    fprintf(f, "spec_k=%d\n", cfg->spec_k);
    fprintf(f, "draft_vocab=%d\n", cfg->draft_vocab);
    fprintf(f, "expert_cache_mib=%d\n", cfg->expert_cache_mib);
    if (cfg->miss_split == QRT_MISS_SPLIT_AUTO) {
        fprintf(f, "miss_split=auto\n");
    } else {
        fprintf(f, "miss_split=%d\n", cfg->miss_split);
    }
    fprintf(f, "cache_policy=%s\n", qrt_cache_policy_name(cfg->cache_policy));
    fprintf(f, "prefill_chunk=%d\n", cfg->prefill_chunk);
    fprintf(f, "kv_precision=%s\n", qrt_kv_precision_name(cfg->kv_precision));
    fprintf(f, "kv_page_tokens=%d\n", cfg->kv_page_tokens);
    fprintf(f, "telemetry=%d\n", cfg->telemetry);
    fprintf(f, "telemetry_path=%s\n", cfg->telemetry_path != NULL ? cfg->telemetry_path : "");
    fprintf(f, "vram_tier_mib=%d\n", cfg->vram_tier_mib);
}
