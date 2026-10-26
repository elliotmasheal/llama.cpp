#include "qrt_cli.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace {

enum option_kind { FLAG, TEXT, NUMBER, CHOICE };

struct option_desc {
    const char * flag;
    option_kind kind;
    size_t offset;
    int min;
    int max;
    const char * values[2];
    qrt_command command;
    const char * help;
};

#define CMD(flag, cmd, help) {flag, FLAG, 0, 0, 0, {nullptr, nullptr}, cmd, help}
#define PATH(flag, field, cmd, help) \
    {flag, TEXT, offsetof(qrt_cli_options, field), 0, 0, {nullptr, nullptr}, cmd, help}
#define INT(flag, field, lo, hi, cmd, help) \
    {flag, NUMBER, offsetof(qrt_cli_options, field), lo, hi, {nullptr, nullptr}, cmd, help}
#define ENUM(flag, field, zero, one, help) \
    {flag, CHOICE, offsetof(qrt_cli_options, field), 0, 1, {zero, one}, QRT_CMD_NONE, help}

const option_desc options[] = {
    CMD("--help", QRT_CMD_HELP, "show all commands and configuration switches"),
    CMD("--selftest", QRT_CMD_SELFTEST, "run CPU-only configuration and telemetry checks"),
    CMD("--print-config", QRT_CMD_CONFIG, "print effective runtime configuration"),
    PATH("--gguf-index", gguf_path, QRT_CMD_INDEX, "read the GGUF tensor index"),
    PATH("--require-tensor", require_tensor, QRT_CMD_NONE, "require a tensor in --gguf-index"),
    CMD("--vk-info", QRT_CMD_VK_INFO, "report the device and configured VRAM budget"),
    INT("--vk-alloc-probe", vk_probe_mib, 1, INT_MAX, QRT_CMD_VK_INFO, "probe an allocation in MiB"),
    CMD("--vk-io-check", QRT_CMD_VK_IO, "check host/device transfers"),
    PATH("--repack", repack_prefix, QRT_CMD_NONE, "expert-store prefix for --vk-io-check"),
    PATH("--dense-check", dense_gguf, QRT_CMD_DENSE, "check dense weight residency"),
    PATH("--prefill-check", prefill_prefix, QRT_CMD_PREFILL, "time the expert prefill leg (not a quality gate)"),
    PATH("--prefill-shader", prefill_shader, QRT_CMD_NONE, "compute shader directory or probe SPIR-V file"),
    INT("--prefill-slots", prefill_slots, 1, 64, QRT_CMD_NONE, "staging slots"),
    INT("--prefill-layers", prefill_layers, 1, INT_MAX, QRT_CMD_NONE, "layers to measure"),
    INT("--prefill-chunks", prefill_chunks, 1, 16, QRT_CMD_NONE, "chunks per layer"),
    INT("--prefill-readers", prefill_readers, 1, 16, QRT_CMD_NONE, "host reader threads"),
    INT("--prefill-group-kib", prefill_group_kib, 0, INT_MAX / 1024, QRT_CMD_NONE, "coalesced transfer size; 0 disables grouping"),
    INT("--prefill-pipeline", prefill_pipeline, 0, 1, QRT_CMD_NONE, "overlap layer upload and compute"),
    ENUM("--prefill-compute", prefill_compute_moe, "probe", "moe", "compute implementation"),
    ENUM("--prefill-mode", prefill_layer_batch, "perexpert", "layer", "submission granularity"),
    PATH("--layer-check", layer_gguf_check, QRT_CMD_LAYER, "check one transformer layer"),
    INT("--layer", layer_index, 0, INT_MAX, QRT_CMD_NONE, "layer index"),
    INT("--layer-tokens", layer_tokens, 64, INT_MAX, QRT_CMD_NONE, "tokens in the layer check"),
    PATH("--layer-store", layer_store, QRT_CMD_NONE, "repacked expert-store prefix"),
    ENUM("--layer-ffn", layer_ffn, "off", "on", "include the FFN half in --layer-check"),
    ENUM("--layer-pass", layer_pass, "off", "on", "merge layer submissions"),
    ENUM("--dn-recur", dn_recur, "serial", "cols", "DeltaNet recurrence implementation"),
    PATH("--norm-check", norm_gguf_check, QRT_CMD_NORM, "check RMS norm and activation packing"),
    PATH("--norm-tensor", norm_tensor, QRT_CMD_NONE, "normalization tensor name"),
    PATH("--dense-gemm-check", dense_gguf_check, QRT_CMD_GEMM, "check dense GEMM"),
    PATH("--dense-tensor", dense_tensor, QRT_CMD_NONE, "dense projection tensor name"),
    INT("--dense-tokens", dense_tokens, 1, INT_MAX, QRT_CMD_NONE, "dense check token count"),
    INT("--dense-rows", dense_rows, 1, INT_MAX, QRT_CMD_NONE, "dense rows checked against CPU"),
    PATH("--deltanet-check", deltanet_gguf, QRT_CMD_DELTANET, "check DeltaNet recurrence"),
    INT("--deltanet-layer", deltanet_layer, 0, INT_MAX, QRT_CMD_NONE, "DeltaNet layer index"),
    INT("--deltanet-tokens", deltanet_tokens, 16, INT_MAX, QRT_CMD_NONE, "DeltaNet token count"),
    PATH("--fullattn-check", fullattn_gguf, QRT_CMD_FULL_ATTN, "check full attention and KV precision"),
    INT("--fullattn-layer", fullattn_layer, 0, INT_MAX, QRT_CMD_NONE, "full-attention layer index"),
    INT("--fullattn-tokens", fullattn_tokens, 16, INT_MAX, QRT_CMD_NONE, "full-attention token count"),
    INT("--fullattn-resident", fullattn_resident, 0, INT_MAX, QRT_CMD_NONE, "resident KV tokens; 0 keeps the full context"),
    INT("--fullattn-page", fullattn_page, 0, INT_MAX, QRT_CMD_NONE, "KV page tokens; 0 uses runtime configuration"),
    PATH("--logits-dump", logits_gguf, QRT_CMD_LOGITS, "run prefill and write QRTLOG1 logits"),
    PATH("--model-check", model_check_gguf, QRT_CMD_MODEL_CHECK, "compare the model with a CPU reference chain"),
    PATH("--chunk-check", chunk_check_gguf, QRT_CMD_CHUNK_CHECK, "compare carried, single and reset chunk states"),
    PATH("--logits-out", logits_out, QRT_CMD_NONE, "device logits output path"),
    PATH("--logits-tokens", logits_tokens, QRT_CMD_NONE, "whitespace-separated decimal token IDs"),
    INT("--logits-layers", logits_layers, 0, 40, QRT_CMD_NONE, "main layers to run; 0 isolates embedding/head"),
    INT("--logits-rows", logits_rows, 1, INT_MAX, QRT_CMD_NONE, "tail logits rows to retain"),
    PATH("--model-check-ref", model_check_ref, QRT_CMD_NONE, "CPU reference logits output path"),
    ENUM("--model-check-reseed", model_check_reseed, "off", "on", "isolate each reference layer using its device input"),
    ENUM("--model-check-planes", model_check_planes, "cpu", "device", "device planes are diagnostic only"),
    PATH("--moe-check", moe_prefix, QRT_CMD_MOE, "check an expert from the repacked store"),
    PATH("--moe-batch-check", moe_batch_gguf, QRT_CMD_MOE_BATCH, "check k=1/2/4/5/8 routed MoE against k=1 serial and CPU"),
    PATH("--generate", generate_gguf, QRT_CMD_GENERATE, "prefill a prompt then greedy-decode tokens"),
    INT("--generate-new", generate_new, 1, 128, QRT_CMD_NONE, "greedy tokens to generate"),
    INT("--generate-batch", generate_batch, 1, 8, QRT_CMD_NONE, "decode rows per pass; >1 feeds prompt ids"),
    PATH("--generate-out", generate_out, QRT_CMD_NONE, "write generated token IDs"),
    PATH("--spec-check", spec_check_gguf, QRT_CMD_SPEC_CHECK, "compare speculative cycles with a serial decode"),
    ENUM("--spec-rollback", spec_rollback, "off", "on", "off keeps every draft row: the counterexample arm"),
    PATH("--cache-check", cache_prefix, QRT_CMD_CACHE, "plan and simulate the decode expert cache (host only)"),
    PATH("--cache-trace", cache_trace, QRT_CMD_NONE, "routing trace for per-layer miss weights"),
    INT("--cache-decode", cache_decode, 1, 1000000, QRT_CMD_NONE, "decode tail rows replayed"),
    INT("--cache-protected", cache_protected, 0, 100, QRT_CMD_NONE, "SLRU protected segment percent"),
    INT("--moe-layer", moe_layer, 0, INT_MAX, QRT_CMD_NONE, "expert layer index"),
    INT("--moe-expert", moe_expert, 0, INT_MAX, QRT_CMD_NONE, "expert index"),
    INT("--moe-rows", moe_rows, 0, INT_MAX, QRT_CMD_NONE, "expert rows checked against CPU"),
};

#undef CMD
#undef PATH
#undef INT
#undef ENUM

const option_desc * find_option(const char * name, size_t size) {
    for (const auto & option : options) {
        if (strlen(option.flag) == size && strncmp(name, option.flag, size) == 0) {
            return &option;
        }
    }
    return nullptr;
}

int apply_option(qrt_cli_options * out, const option_desc & option, const char * value,
                  char * err, size_t err_size) {
    char * field = reinterpret_cast<char *>(out) + option.offset;
    if (option.kind == TEXT) {
        *reinterpret_cast<const char **>(field) = value;
    } else if (option.kind == NUMBER) {
        errno = 0;
        char * end = nullptr;
        const long long n = strtoll(value, &end, 10);
        if (errno == ERANGE || end == value || *end != '\0' || n < option.min || n > option.max) {
            snprintf(err, err_size, "%s: expected %d..%d, got '%s'", option.flag,
                     option.min, option.max, value);
            return -1;
        }
        *reinterpret_cast<int *>(field) = static_cast<int>(n);
    } else if (option.kind == CHOICE) {
        int selected = -1;
        for (int i = 0; i < 2; ++i) {
            if (strcmp(value, option.values[i]) == 0) {
                selected = i;
            }
        }
        if (selected < 0) {
            snprintf(err, err_size, "%s: expected %s or %s, got '%s'", option.flag,
                     option.values[0], option.values[1], value);
            return -1;
        }
        *reinterpret_cast<int *>(field) = selected;
    }
    return 0;
}

} // namespace

int qrt_cli_parse(qrt_cli_options * out, qrt_config * cfg, int argc, char ** argv,
                  char * err, size_t err_size) {
    qrt_cli_options parsed;
    if (argc <= 1) {
        parsed.command = QRT_CMD_HELP;
    }
    qrt_config effective = *cfg;
    std::vector<char *> config_args;
    config_args.push_back(argc > 0 ? argv[0] : const_cast<char *>("qrt"));
    const char * selected_command = nullptr;
    for (int i = 1; i < argc; ++i) {
        const char * arg = strcmp(argv[i], "-h") == 0 ? "--help" : argv[i];
        if (strncmp(arg, "--", 2) != 0) {
            snprintf(err, err_size, "unexpected argument '%s'", arg);
            return -1;
        }
        const char * eq = strchr(arg, '=');
        const size_t name_size = eq ? static_cast<size_t>(eq - arg) : strlen(arg);
        const auto * option = find_option(arg, name_size);
        if (!option) {
            bool known = false;
            for (int k = 0; k < qrt_config_switch_count(); ++k) {
                const char * flag = qrt_config_flag_name(k);
                known |= strlen(flag) == name_size && strncmp(arg, flag, name_size) == 0;
            }
            if (!known) {
                snprintf(err, err_size, "unknown option '%s' (see --help)", arg);
                return -1;
            }
        }
        if (option && option->command != QRT_CMD_NONE) {
            if (parsed.command != QRT_CMD_NONE && parsed.command != option->command) {
                snprintf(err, err_size, "conflicting commands '%s' and '%s'", selected_command,
                         option->flag);
                return -1;
            }
            parsed.command = option->command;
            selected_command = option->flag;
        }
        if (option && option->kind == FLAG) {
            if (eq) {
                snprintf(err, err_size, "%s takes no value", option->flag);
                return -1;
            }
            continue;
        }
        const char * value = eq ? eq + 1 : nullptr;
        if (!eq) {
            if (i + 1 >= argc || strncmp(argv[i + 1], "--", 2) == 0) {
                snprintf(err, err_size, "%s: missing value", arg);
                return -1;
            }
            value = argv[++i];
        }
        if (!*value) {
            snprintf(err, err_size, "%s: empty value", option ? option->flag : arg);
            return -1;
        }
        if (option) {
            if (apply_option(&parsed, *option, value, err, err_size) != 0) {
                return -1;
            }
        } else {
            config_args.push_back(const_cast<char *>(arg));
            if (!eq) {
                config_args.push_back(const_cast<char *>(value));
            }
        }
    }
    if (parsed.command == QRT_CMD_HELP || parsed.command == QRT_CMD_SELFTEST) {
        *out = parsed;
        return 0;
    }
    if (qrt_config_parse(&effective, static_cast<int>(config_args.size()), config_args.data(),
                         err, err_size) != 0 ||
        qrt_config_validate(&effective, err, err_size) != 0) {
        return -1;
    }
    if (parsed.command == QRT_CMD_NONE) {
        if (argc <= 1) {
            parsed.command = QRT_CMD_HELP;
        } else {
            snprintf(err, err_size, "no command selected; use --help or --print-config");
            return -1;
        }
    }
    if (parsed.require_tensor && parsed.command != QRT_CMD_INDEX) {
        snprintf(err, err_size, "--require-tensor needs the --gguf-index command");
        return -1;
    }
    if (parsed.repack_prefix && parsed.command != QRT_CMD_VK_IO) {
        snprintf(err, err_size, "--repack needs the --vk-io-check command");
        return -1;
    }
    if (parsed.cache_trace && parsed.command != QRT_CMD_CACHE) {
        snprintf(err, err_size, "--cache-trace needs the --cache-check command");
        return -1;
    }
    if (parsed.command == QRT_CMD_MOE_BATCH && !parsed.layer_store) {
        snprintf(err, err_size, "--moe-batch-check needs --layer-store");
        return -1;
    }
    if (parsed.command == QRT_CMD_PREFILL && !parsed.prefill_compute_moe && parsed.prefill_layer_batch) {
        snprintf(err, err_size, "--prefill-compute probe requires --prefill-mode perexpert");
        return -1;
    }
    const bool model_command = parsed.command == QRT_CMD_LOGITS ||
                               parsed.command == QRT_CMD_MODEL_CHECK ||
                               parsed.command == QRT_CMD_CHUNK_CHECK ||
                               parsed.command == QRT_CMD_GENERATE ||
                               parsed.command == QRT_CMD_SPEC_CHECK;
    if (model_command && !parsed.logits_tokens) {
        snprintf(err, err_size, "model commands require --logits-tokens");
        return -1;
    }
    if (parsed.command == QRT_CMD_GENERATE && !parsed.layer_store) {
        snprintf(err, err_size, "--generate requires --layer-store");
        return -1;
    }
    if (parsed.command == QRT_CMD_SPEC_CHECK && !parsed.layer_store) {
        snprintf(err, err_size, "--spec-check requires --layer-store");
        return -1;
    }
    if (parsed.generate_out && parsed.command != QRT_CMD_GENERATE) {
        snprintf(err, err_size, "--generate-out needs the --generate command");
        return -1;
    }
    if (parsed.command == QRT_CMD_LOGITS && !parsed.logits_out) {
        snprintf(err, err_size, "--logits-dump requires --logits-out");
        return -1;
    }
    if (parsed.command == QRT_CMD_MODEL_CHECK && !parsed.model_check_ref) {
        snprintf(err, err_size, "--model-check requires --model-check-ref");
        return -1;
    }
    if (model_command && parsed.logits_layers > 0 && !parsed.layer_store) {
        snprintf(err, err_size, "model layers require --layer-store; use --logits-layers 0 only for embedding/head diagnostics");
        return -1;
    }
    *out = parsed;
    *cfg = effective;
    return 0;
}

void qrt_cli_print_help(FILE * out) {
    fprintf(out, "qrt: Qwen3.6 prefill diagnostics and a k=1 greedy decode slice\n\n");
    for (const auto & option : options) {
        if (option.kind == CHOICE) {
            fprintf(out, "  %-24s %s|%s  %s\n", option.flag, option.values[0], option.values[1],
                    option.help);
        } else {
            fprintf(out, "  %-24s %-7s %s\n", option.flag,
                    option.kind == FLAG ? "" : option.kind == NUMBER ? "N" : "PATH", option.help);
        }
    }
    fprintf(out, "\nRuntime configuration (CLI > environment > defaults):\n");
    qrt_config_print_switches(out);
}
