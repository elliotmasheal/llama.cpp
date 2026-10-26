#include "qrt_checks.h"

#include "qrt_cli.h"
#include "qrt_config.h"
#include "attn_deltanet.h"
#include "attn_full.h"
#include "gguf_index.h"
#include "layer.h"
#include "model.h"
#include "vk_dev.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    qrt_cli_options options;
    qrt_config cfg;
    qrt_config_defaults(&cfg);
    char err[QRT_ERR_SIZE] = {};
    if (qrt_cli_parse(&options, &cfg, argc, argv, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    if (options.command == QRT_CMD_HELP) {
        qrt_cli_print_help(stdout);
        return 0;
    }
    if (options.command == QRT_CMD_SELFTEST) {
        return qrt_check_selftest();
    }
    qrt_config_notes(&cfg, stderr);
    if (options.command == QRT_CMD_CONFIG) {
        qrt_config_print_effective(&cfg, stdout);
        return 0;
    }
    const std::string exe = argc > 0 ? argv[0] : "qrt";
    const size_t cut = exe.find_last_of("/\\");
    const std::string spv_dir = cut == std::string::npos ? "." : exe.substr(0, cut);
    if (options.command == QRT_CMD_CACHE) {
        return qrt_check_cache(&cfg, options.cache_prefix, options.cache_trace,
                               (uint32_t) options.cache_decode,
                               (uint32_t) options.cache_protected);
    }
    if (options.command == QRT_CMD_MOE_BATCH) {
        return qrt_check_moe_batch(&cfg, options.moe_batch_gguf, options.layer_store,
                                   (uint32_t) options.moe_layer, spv_dir.c_str());
    }

    if (options.gguf_path != NULL) {
        qrt_gguf * g = NULL;
        if (qrt_gguf_open(options.gguf_path, &g, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        if (options.require_tensor != NULL) {
            const qrt_gguf_tensor * t = NULL;
            if (qrt_gguf_require(g, options.require_tensor, &t, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_gguf_close(g);
                return 2;
            }
            printf("tensor=%s offset=%llu nbytes=%llu type=%s\n", t->name,
                   (unsigned long long) t->offset, (unsigned long long) t->nbytes,
                   qrt_gguf_type_name(t->type));
        }
        qrt_gguf_print_report(g, stdout);
        qrt_gguf_close(g);
        return 0;
    }
    if (options.require_tensor != NULL) {
        fprintf(stderr, "qrt: --require-tensor needs --gguf-index <path>\n");
        return 2;
    }

    if (options.command == QRT_CMD_VK_IO) {
        return qrt_check_vk_io(&cfg, options.repack_prefix);
    }
    if (options.dense_gguf != NULL) {
        return qrt_check_dense_residency(&cfg, options.dense_gguf);
    }
    if (options.prefill_prefix != NULL) {
        const char * shader_path = options.prefill_shader ? options.prefill_shader : spv_dir.c_str();
        return qrt_check_prefill(&cfg, options.prefill_prefix, shader_path,
                                 (uint32_t) options.prefill_slots, (uint32_t) options.prefill_layers,
                                 options.prefill_compute_moe, options.prefill_layer_batch, options.prefill_pipeline,
                                 options.prefill_group_kib, options.prefill_readers, options.prefill_chunks);
    }
    if (options.layer_gguf_check != NULL) {
        qrt_layer_cfg lcfg;
        memset(&lcfg, 0, sizeof(lcfg));
        lcfg.ffn = options.layer_ffn && options.layer_store != NULL ? 1 : 0;
        lcfg.single_submit = options.layer_pass;
        lcfg.dn_recur = options.dn_recur;
        lcfg.kv_k = cfg.kv_k;
        lcfg.kv_v = cfg.kv_v;
        lcfg.kv_page_tokens = (uint32_t) cfg.kv_page_tokens;
        return qrt_layer_check(options.layer_gguf_check, options.layer_store, (uint32_t) options.layer_index,
                               (uint32_t) options.layer_tokens, &lcfg, spv_dir.c_str());
    }
    if (options.norm_gguf_check != NULL) {
        return qrt_check_norm(&cfg, options.norm_gguf_check, options.norm_tensor, options.dense_tensor,
                              (uint32_t) options.dense_tokens, spv_dir.c_str());
    }
    if (options.dense_gguf_check != NULL) {
        return qrt_check_dense_gemm(&cfg, options.dense_gguf_check, options.dense_tensor,
                                    (uint32_t) options.dense_tokens, (uint32_t) options.dense_rows,
                                    spv_dir.c_str());
    }
    if (options.deltanet_gguf != NULL) {
        return qrt_dn_check(options.deltanet_gguf, (uint32_t) options.deltanet_layer, (uint32_t) options.deltanet_tokens,
                            spv_dir.c_str());
    }
    if (options.command == QRT_CMD_GENERATE) {
        qrt_model_options mo{};
        mo.n_layers = (uint32_t) options.logits_layers;
        mo.n_rows = 1;
        mo.batch_k = (uint32_t) options.generate_batch;
        uint32_t * ids = NULL;
        uint32_t n_ids = 0;
        if (qrt_model_read_ids(options.logits_tokens, &ids, &n_ids, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        const int rc = qrt_model_generate(&cfg, &mo, options.generate_gguf, options.layer_store,
                                          spv_dir.c_str(), ids, n_ids, (uint32_t) options.generate_new,
                                          options.generate_out, err, sizeof(err));
        free(ids);
        if (rc != 0 && err[0] != '\0') {
            fprintf(stderr, "qrt: %s\n", err);
        }
        return rc < 0 ? 2 : rc;
    }
    if (options.command == QRT_CMD_CHUNK_CHECK || options.command == QRT_CMD_MODEL_CHECK ||
        options.command == QRT_CMD_LOGITS) {
        qrt_model_options model_options;
        model_options.n_layers = (uint32_t) options.logits_layers;
        model_options.n_rows = (uint32_t) options.logits_rows;
        model_options.out_path = options.logits_out;
        model_options.ref_out_path = options.model_check_ref;
        model_options.reseed = options.model_check_reseed;
        model_options.device_planes = options.model_check_planes;
        uint32_t * ids = NULL;
        uint32_t n_ids = 0;
        if (qrt_model_read_ids(options.logits_tokens, &ids, &n_ids, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        const auto run = options.command == QRT_CMD_CHUNK_CHECK ? qrt_model_chunk_check :
                         options.command == QRT_CMD_MODEL_CHECK ? qrt_model_check : qrt_model_logits;
        const char * path = options.command == QRT_CMD_CHUNK_CHECK ? options.chunk_check_gguf :
                            options.command == QRT_CMD_MODEL_CHECK ? options.model_check_gguf : options.logits_gguf;
        const int rc = run(&cfg, &model_options, path, options.layer_store, spv_dir.c_str(), ids,
                           n_ids, err, sizeof(err));
        free(ids);
        if (rc != 0 && err[0] != '\0') {
            fprintf(stderr, "qrt: %s\n", err);
        }
        return rc < 0 ? 2 : rc;
    }
    if (options.fullattn_gguf != NULL) {
        return qrt_full_check(options.fullattn_gguf, (uint32_t) options.fullattn_layer, (uint32_t) options.fullattn_tokens,
                              spv_dir.c_str(), cfg.kv_k, cfg.kv_v, (uint32_t) options.fullattn_resident,
                              (uint32_t) options.fullattn_page);
    }
    if (options.moe_prefix != NULL) {
        std::string dir = argc > 0 ? argv[0] : "qrt";
        const size_t cut = dir.find_last_of("/\\");
        dir = (cut == std::string::npos ? std::string(".") : dir.substr(0, cut));
        return qrt_check_moe(&cfg, options.moe_prefix, (options.prefill_shader != NULL ? options.prefill_shader : dir.c_str()),
                             (uint32_t) options.moe_layer, (uint32_t) options.moe_expert, (uint32_t) options.moe_rows);
    }
    if (options.command == QRT_CMD_VK_INFO) {
        qrt_vk_dev * dev = NULL;
        if (qrt_vk_open((uint32_t) cfg.vram_tier_mib, &dev, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        qrt_vk_print_info(dev, stdout);
        int rc = 0;
        // RESULTS section 25: the routed_experts total of the target model.
        const uint64_t expert_store_bytes = 20122173440ull;
        qrt_vk_plan plan;
        if (qrt_vk_plan_budget(dev, &cfg, expert_store_bytes, &plan, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
        } else {
            printf("plan dense_mib=%llu cache_mib=%llu kv_mib=%llu compute_mib=%llu "
                   "draft_mib=%llu total_mib=%llu kv_tokens=%u ctx_tokens=%u "
                   "coverage_pct=%.2f reductions=%u\n",
                   (unsigned long long) (plan.dense_bytes >> 20),
                   (unsigned long long) (plan.expert_cache_bytes >> 20),
                   (unsigned long long) (plan.kv_bytes >> 20),
                   (unsigned long long) (plan.compute_bytes >> 20),
                   (unsigned long long) (plan.draft_bytes >> 20),
                   (unsigned long long) (plan.total_bytes >> 20),
                   plan.kv_tokens, (unsigned) QRT_KV_CTX_TOKENS,
                   (double) plan.cache_coverage_x100 / 100.0, plan.cache_reductions);
        }
        if (options.vk_probe_mib > 0) {
            const uint64_t before = dev->budget_bytes;
            VkBuffer buf = 0;
            VkDeviceMemory mem = 0;
            if (qrt_vk_alloc(dev, (uint64_t) options.vk_probe_mib << 20, QRT_VK_USAGE_STORAGE,
                             &buf, &mem, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                rc = 2;
            } else {
                qrt_vk_query_budget(dev, err, sizeof(err));
                printf("probe alloc_mib=%d budget_before_mib=%llu budget_after_mib=%llu\n",
                       options.vk_probe_mib, (unsigned long long) (before >> 20),
                       (unsigned long long) (dev->budget_bytes >> 20));
                qrt_vk_free(dev, buf, mem);
            }
        }
        qrt_vk_close(dev);
        return rc;
    }

    fprintf(stderr, "qrt: unsupported command\n");
    return 2;
}
