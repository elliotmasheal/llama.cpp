// Whole-model prefill and the QRTLOG1 logits dump (M2-7a). See model.h for the scope.

#include "model.h"
#include "vk_transfer.h"

#include "ffn.h"
#include "gguf_index.h"
#include "layer.h"
#include "model_internal.h"
#include "model_io.h"
#include "qrt_config.h"
#include "qrt_util.h"
#include "vk_dev.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace {

double rms(const float * a, size_t n) {
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ss += (double) a[i] * (double) a[i];
    }
    return n > 0 ? sqrt(ss / (double) n) : 0.0;
}

double l1_cmp(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
        den += fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : num;
}

int download_stage(qrt_vk_dev * dev, qrt_vk_staging * stg, uint64_t stg_bytes, VkBuffer src,
                   std::vector<float> & dst, uint32_t row, char * err, size_t err_size) {
    if (src == VK_NULL_HANDLE || dst.empty()) {
        return 0;
    }
    // The staging buffer is sized for the hidden rows of the chunk, so a 4096-wide plane, or
    // any plane at a short chunk, goes in slices that fit it.
    const uint32_t tokens = (uint32_t) (dst.size() / row);
    const uint64_t per_tok = (uint64_t) row * 4u;
    uint32_t slice = (uint32_t) (stg_bytes / per_tok);
    if (slice == 0u || slice > tokens) {
        slice = tokens;
    }
    for (uint32_t t0 = 0; t0 < tokens; t0 += slice) {
        const uint32_t cnt = tokens - t0 < slice ? tokens - t0 : slice;
        if (qrt_vk_download(dev, stg, src, (uint64_t) t0 * row * 4, dst.data() + (size_t) t0 * row,
                            (uint64_t) cnt * row * 4, err, err_size) != 0) {
            return -1;
        }
    }
    return 0;
}

// rel_L1 over the whole dumped block: the sums are taken across rows, so one row cannot
// hide behind another.
double l1_num(const float * a, const float * b, size_t n) {
    double num = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
    }
    return num;
}

double l1_den(const float * b, size_t n) {
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        den += fabs((double) b[i]);
    }
    return den;
}

double logsumexp(const float * row, uint32_t V) {
    double mx = row[0];
    for (uint32_t v = 1; v < V; ++v) {
        if (row[v] > mx) {
            mx = row[v];
        }
    }
    double sum = 0.0;
    for (uint32_t v = 0; v < V; ++v) {
        sum += exp((double) row[v] - mx);
    }
    return mx + log(sum);
}

struct run_opts {
    uint32_t n_layers;
    uint32_t n_rows;
    uint32_t chunk;      // tokens per layer pass, 0 = the whole prompt in one chunk
    int carry;           // 0 = zero the per-sequence state before every chunk (counterexample)
    uint32_t replay;     // 1 = re-run chunk 0 of the first layer of each kind against the chain
    int reseed;          // 1 = the reference starts each layer from the device hidden
    int device_planes;   // 1 = localisation: the reference reuses the device's Q8_1 planes
    const char * out_path;      // QRTLOG1 dump of the device arm, NULL to skip
    const char * ref_out_path;  // QRTLOG1 dump of the reference arm, NULL to skip
};

struct run_result {
    double logits_rel_l1;   // device arm against the reference arm, mean over the dumped rows
    double hidden_max;      // worst per-layer hidden rel_l1 of the chain
    double replay_max;      // worst re-run-vs-chain rel_l1 of the same layer
    uint32_t top1_rows;
    uint32_t top1_match;
    double nll_device;
    double nll_ref;
    // set when the caller wants the arm's own end state, for the cross-chunk comparison
    bool want_hidden;
    std::vector<float> hidden;
};


struct layer_owner {
    qrt_vk_dev * dev;
    qrt_layer * layer = nullptr;

    explicit layer_owner(qrt_vk_dev * d) : dev(d) {}
    layer_owner(const layer_owner &) = delete;
    layer_owner & operator=(const layer_owner &) = delete;
    ~layer_owner() {
        if (layer != nullptr) qrt_layer_close(dev, layer);
    }
};

enum class model_mode { logits, check, chunk_check };

int distinct_paths(const char * a, const char * b, char * err, size_t err_size) {
    if (a == nullptr || b == nullptr) return 0;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path pa = fs::absolute(a, ec);
    if (ec) return fail(err, err_size, "cannot resolve output path '%s'", a);
    pa = fs::weakly_canonical(pa, ec);
    if (ec) return fail(err, err_size, "cannot resolve output path '%s'", a);
    fs::path pb = fs::absolute(b, ec);
    if (ec) return fail(err, err_size, "cannot resolve output path '%s'", b);
    pb = fs::weakly_canonical(pb, ec);
    if (ec) return fail(err, err_size, "cannot resolve output path '%s'", b);
    std::string sa = pa.generic_string(), sb = pb.generic_string();
#ifdef _WIN32
    for (char & c : sa) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    for (char & c : sb) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
#endif
    if (sa == sb || fs::equivalent(pa, pb, ec)) {
        return fail(err, err_size, "model input and output paths must differ");
    }
    return 0;
}

int request_opts(const qrt_config * cfg, const qrt_model_options * opt, model_mode mode,
                 const char * gguf_path, const char * store_prefix, const char * spv_dir,
                 const uint32_t * ids, uint32_t n_tokens, run_opts * run,
                 char * err, size_t err_size) {
    if (cfg == nullptr || opt == nullptr) {
        return fail(err, err_size, "the model run needs configuration and options");
    }
    if (qrt_config_validate(cfg, err, err_size) != 0) return -1;
    if (gguf_path == nullptr || *gguf_path == '\0' || spv_dir == nullptr || *spv_dir == '\0' ||
        (store_prefix != nullptr && *store_prefix == '\0') || ids == nullptr || n_tokens == 0) {
        return fail(err, err_size, "the model run needs a GGUF, SPIR-V and tokens");
    }
    if (n_tokens > QRT_KV_CTX_TOKENS || cfg->prefill_chunk % 32 != 0) {
        return fail(err, err_size, "tokens exceed the context or prefill chunk is not a multiple of 32");
    }
    if (mode == model_mode::check && n_tokens % 32u != 0) {
        return fail(err, err_size, "the CPU reference requires a whole 32-row tile");
    }
    if (opt->n_rows == 0 || opt->n_rows >= n_tokens) {
        return fail(err, err_size, "model rows must be in 1..tokens-1");
    }
    if ((opt->reseed != 0 && opt->reseed != 1) ||
        (opt->device_planes != 0 && opt->device_planes != 1)) {
        return fail(err, err_size, "reseed and device_planes must be 0 or 1");
    }
    if ((opt->out_path != nullptr && *opt->out_path == '\0') ||
        (opt->ref_out_path != nullptr && *opt->ref_out_path == '\0')) {
        return fail(err, err_size, "output paths must not be empty");
    }
    if (distinct_paths(opt->out_path, opt->ref_out_path, err, err_size) != 0) return -1;
    for (const char * output : {opt->out_path, opt->ref_out_path}) {
        if (distinct_paths(output, gguf_path, err, err_size) != 0) return -1;
        if (store_prefix != nullptr) {
            const std::string prefix(store_prefix);
            if (distinct_paths(output, (prefix + ".qwen36rt.idx").c_str(), err, err_size) != 0 ||
                distinct_paths(output, (prefix + ".qwen36rt.bin").c_str(), err, err_size) != 0) return -1;
        }
    }
    if (mode == model_mode::check) {
        if (opt->ref_out_path == nullptr || cfg->kv_k != QRT_KV_F16 || cfg->kv_v != QRT_KV_F16) {
            return fail(err, err_size, "model-check needs a reference output and f16/f16 KV");
        }
        if ((uint32_t) cfg->prefill_chunk < n_tokens) {
            return fail(err, err_size, "the CPU reference needs one chunk covering the prompt");
        }
    } else if (opt->ref_out_path != nullptr || opt->reseed != 0 || opt->device_planes != 0) {
        return fail(err, err_size, "reference options require model-check");
    }
    if (mode == model_mode::chunk_check &&
        ((uint32_t) cfg->prefill_chunk >= n_tokens || opt->out_path != nullptr)) {
        return fail(err, err_size, "chunk-check needs a split prompt and no dump path");
    }
    *run = {};
    run->n_layers = opt->n_layers;
    run->n_rows = opt->n_rows;
    run->chunk = (uint32_t) cfg->prefill_chunk;
    run->carry = 1;
    run->replay = mode == model_mode::check ? 1u : 0u;
    run->reseed = opt->reseed;
    run->device_planes = opt->device_planes;
    run->out_path = opt->out_path;
    run->ref_out_path = opt->ref_out_path;
    return 0;
}

// The one layer loop both entries use: embedding -> n_layers -> final norm -> head. With
// reseed 0 the reference chain feeds itself, which is the number the acceptance wants; with
// reseed 1 every layer's reference starts from the device hidden, which isolates one layer.
int model_forward(const qrt_config & config, const char * gguf_path, const char * store_prefix,
                  const char * spv_dir, const uint32_t * ids, uint32_t n_tokens,
                  const run_opts & opt, run_result * res, char * err, size_t err_size) {
    const uint32_t padded_tokens = (n_tokens + 31u) / 32u * 32u;
    uint32_t chunk = opt.chunk != 0u ? opt.chunk : padded_tokens;
    if (chunk > padded_tokens) {
        chunk = padded_tokens;
    }
    if ((chunk % 32u) != 0u) {
        return fail(err, err_size, "chunk %u is not a multiple of the 32 the FFN tiles", chunk);
    }
    const uint32_t n_chunks = 1u + (n_tokens - 1u) / chunk;
    if (n_chunks > 1u && opt.ref_out_path != NULL) {
        return fail(err, err_size,
                    "the CPU reference chain models one chunk from position 0: --model-check "
                    "needs a chunk covering the whole prompt, use --logits-dump or --chunk-check");
    }
    char e[512] = {0};
    qrt_model_run_owner owner;
    qrt_vk_dev * & dev = owner.dev;
    qrt_gguf * & g = owner.g;
    qrt_vk_staging * & stg = owner.stg;
    std::vector<float> onorm;
    std::vector<float> x, xref;
    const bool want_head = !res->want_hidden || opt.out_path != NULL || opt.ref_out_path != NULL;
    do {
        if (qrt_vk_open((uint32_t) config.vram_tier_mib, &dev, e, sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }
        if (qrt_gguf_open(gguf_path, &g, e, sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }
        const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(g);
        if (hp == NULL || !(hp->present & QRT_HP_BLOCK_COUNT) || hp->block_count <= 0) {
            return fail(err, err_size, "missing or invalid block_count");
        }
        const int32_t nextn = hp->present & QRT_HP_NEXTN ? hp->nextn : 0;
        if (nextn < 0 || nextn >= hp->block_count ||
            opt.n_layers > (uint32_t) (hp->block_count - nextn)) {
            return fail(err, err_size, "requested %u layers, block_count=%d nextn=%d (main layers only)",
                        opt.n_layers, hp->block_count, nextn);
        }
        const uint32_t layer_count = opt.n_layers;
        if (hp->n_embd <= 0 || hp->n_embd % 32 != 0) {
            return fail(err, err_size, "n_embd must be a positive multiple of 32");
        }
        const uint32_t E = (uint32_t) hp->n_embd;
        const qrt_gguf_tensor * emb = qrt_gguf_find(g, "token_embd.weight");
        const qrt_gguf_tensor * outw = want_head ? qrt_gguf_find(g, "output.weight") : NULL;
        if (emb == NULL || emb->type != 8 || emb->n_dims != 2 || emb->ne[0] != E ||
            emb->ne[1] == 0 || emb->ne[1] > INT32_MAX) {
            return fail(err, err_size, "token_embd must be q8_0 with n_embd x vocabulary rows");
        }
        if (want_head && (outw == NULL || outw->type != 8 || outw->n_dims != 2 ||
                          outw->ne[0] != E || outw->ne[1] != emb->ne[1])) {
            return fail(err, err_size, "output must match the q8_0 embedding shape");
        }
        const uint32_t V = (uint32_t) emb->ne[1];
        if ((uint64_t) n_tokens * E > SIZE_MAX / sizeof(float) ||
            (uint64_t) opt.n_rows * V > SIZE_MAX / sizeof(float)) {
            return fail(err, err_size, "model rows exceed host address space");
        }
        x.assign((size_t) n_tokens * E, 0.0f);
        if (qrt_model_embed_rows(g, emb, E, V, ids, n_tokens, x.data(), e, sizeof(e)) != 0) {
            return fail(err, err_size, "embedding: %s", e);
        }
        if (want_head && (qrt_gguf_read_f32(g, "output_norm.weight", &onorm, e, sizeof(e)) != 0 ||
                          onorm.size() != E)) {
            return fail(err, err_size, "output_norm: %s", e[0] ? e : "wrong width");
        }
        if (opt.ref_out_path != NULL) {
            xref.assign(x.data(), x.data() + x.size());
        }
        const uint64_t stg_bytes = (uint64_t) chunk * E * 4;
        if (qrt_vk_staging_create(dev, stg_bytes, &stg, e, sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }

        qrt_layer_cfg cfg{};
        cfg.ffn = store_prefix != NULL ? 1 : 0;
        cfg.single_submit = 1;
        cfg.dn_recur = 1;
        cfg.kv_k = config.kv_k;
        cfg.kv_v = config.kv_v;
        cfg.kv_page_tokens = (uint32_t) config.kv_page_tokens;
        cfg.n_ctx = n_tokens;
        printf("model_run tokens=%u layers=%u rows=%u chunk=%u carry=%d reseed=%d kv=%s/%s "
               "ffn=%d\n", n_tokens, layer_count, opt.n_rows, chunk, opt.carry, opt.reseed,
               qrt_kv_precision_name(cfg.kv_k), qrt_kv_precision_name(cfg.kv_v), cfg.ffn);

        // Layer first (design 4.5): the layer is opened once for a chunk and every chunk of
        // the prompt goes through it, so a layer's weights and expert blocks are read once
        // per prompt and the per-sequence state is carried inside the layer.
        std::vector<float> xcur((size_t) chunk * E), xnext(xcur.size());
        int seen[2] = { 0, 0 };
        for (uint32_t l = 0; l < layer_count; ++l) {
            layer_owner owned_layer(dev);
            qrt_layer * & layer = owned_layer.layer;
            if (qrt_layer_open(dev, gguf_path, store_prefix, l, chunk, spv_dir, &cfg, &layer,
                               e, sizeof(e)) != 0) {
                return fail(err, err_size, "layer %u: %s", l, e);
            }
            double ms_layer = 0.0;
            double replay_rel = -1.0, replay_max_val = -1.0;
            uint64_t last_uploaded = 0;
            int lrc = qrt_layer_zero_state(layer, e, sizeof(e));
            if (lrc != 0) {
                return fail(err, err_size, "layer %u reset: %s", l, e);
            }
            for (uint32_t c = 0; c < n_chunks; ++c) {
                const uint32_t pos0 = c * chunk;
                const uint32_t len = n_tokens - pos0 < chunk ? n_tokens - pos0 : chunk;
                memcpy(xcur.data(), x.data() + (size_t) pos0 * E, (size_t) len * E * sizeof(float));
                if (len < chunk) {
                    // rows past the prompt are not part of the sequence: the row-parallel
                    // stages still run the opened chunk, so their input is zeroed
                    std::fill(xcur.begin() + (size_t) len * E, xcur.end(), 0.0f);
                }
                if (c > 0u && opt.carry == 0) {
                    lrc = qrt_layer_zero_state(layer, e, sizeof(e));
                }
                if (lrc == 0) {
                    lrc = qrt_layer_set_input(layer, xcur.data(), e, sizeof(e));
                }
                if (lrc == 0) {
                    lrc = qrt_layer_run_chunk(layer, pos0, len, e, sizeof(e));
                }
                if (lrc == 0) {
                    lrc = qrt_vk_download(dev, stg, qrt_layer_out(layer), 0, xnext.data(),
                                          (uint64_t) chunk * E * 4, e, sizeof(e));
                }
                if (lrc != 0) {
                    return fail(err, err_size, "layer %u chunk %u: %s", l, c, e);
                }
                const uint64_t uploaded = qrt_layer_uploaded_bytes(layer);
                if (uploaded != last_uploaded) {
                    printf("model_upload layer=%u chunk=%u uploaded_mib=%.1f\n", l, c,
                           (double) uploaded / (1024.0 * 1024.0));
                    last_uploaded = uploaded;
                }
                memcpy(x.data() + (size_t) pos0 * E, xnext.data(), (size_t) len * E * sizeof(float));
                ms_layer += qrt_layer_last_ms(layer);
                if (opt.replay != 0u && c == 0u && !seen[qrt_layer_shape(layer)->linear]) {
                    // The runner opens, feeds and closes one layer at a time, so the same
                    // layer re-run from the same hidden must reproduce it: that is the
                    // "adjacent layer hidden equals the single-layer chain" item, and it
                    // catches a leaked state.
                    const int kind = qrt_layer_shape(layer)->linear;
                    seen[kind] = 1;
                    if (qrt_layer_set_input(layer, xcur.data(), e, sizeof(e)) != 0 ||
                        qrt_layer_zero_state(layer, e, sizeof(e)) != 0 ||
                        qrt_layer_run_chunk(layer, 0u, len, e, sizeof(e)) != 0) {
                        return fail(err, err_size, "layer %u replay: %s", l, e);
                    }
                    std::vector<float> again((size_t) chunk * E, 0.0f);
                    if (qrt_vk_download(dev, stg, qrt_layer_out(layer), 0, again.data(),
                                        (uint64_t) chunk * E * 4, e, sizeof(e)) != 0) {
                        return fail(err, err_size, "layer %u replay readback: %s", l, e);
                    }
                    double num = 0.0, den = 0.0, mx = 0.0;
                    for (size_t i = 0; i < (size_t) len * E; ++i) {
                        const double d = fabs((double) again[i] - (double) x[i]);
                        num += d;
                        den += fabs((double) x[i]);
                        if (d > mx) {
                            mx = d;
                        }
                    }
                    const double rel = den > 0.0 ? num / den : num;
                    replay_rel = rel; replay_max_val = mx;
                    if (rel > res->replay_max) {
                        res->replay_max = rel;
                    }
                }
            }
            printf("model_layer layer=%u chunks=%u ms=%.1f up_mib=%.1f x0=%.4f", l, n_chunks,
                   ms_layer, (double) last_uploaded / (1024.0 * 1024.0),
                   x[(size_t) (n_tokens - 1) * E]);
            if (replay_rel >= 0.0) {
                printf(" replay_rel_l1=%.3e replay_max=%.3e", replay_rel, replay_max_val);
            }
            printf("\n");

            if (opt.ref_out_path != NULL) {
                // The reference is single-chunk, so xcur still holds this layer's input.
                const std::vector<float> & ref_in = opt.reseed ? xcur : xref;
                const uint32_t inter = qrt_layer_shape(layer)->inter;
                const qrt_ffn_bufs * fb = qrt_layer_ffn_bufs(layer);
                struct qrt_layer_ref_trace tr;
                memset(&tr, 0, sizeof(tr));
                std::vector<float> r_xnorm, r_src, r_out, r_res, r_post, r_moe, r_shexp;
                r_xnorm.assign((size_t) n_tokens * E, 0.0f);
                r_src.assign((size_t) n_tokens * inter, 0.0f);
                r_out.assign((size_t) n_tokens * E, 0.0f);
                r_res.assign((size_t) n_tokens * E, 0.0f);
                r_post.assign((size_t) n_tokens * E, 0.0f);
                tr.xnorm = r_xnorm.data();
                tr.attn_src = r_src.data();
                tr.attn_out = r_out.data();
                tr.x_res = r_res.data();
                tr.post_norm = r_post.data();
                if (fb != NULL) {
                    r_moe.assign((size_t) n_tokens * E, 0.0f);
                    r_shexp = r_moe;
                    tr.moe = r_moe.data();
                    tr.shexp = r_shexp.data();
                }
                if (qrt_layer_ref_cpu(layer, ref_in.data(), xref.data(), &tr, opt.device_planes,
                                      e, sizeof(e)) != 0) {
                    return fail(err, err_size, "layer %u reference: %s", l, e);
                }
                // The device side of the same stages, for localisation only: a reference that
                // reads a device buffer back cannot be the acceptance evidence (runtime/AGENTS 2).
                std::vector<float> d_xnorm(r_xnorm.size()), d_src(r_src.size());
                std::vector<float> d_out(r_out.size()), d_res(r_res.size());
                std::vector<float> d_post(r_post.size()), d_moe, d_shexp;
                int drc = download_stage(dev, stg, stg_bytes, qrt_layer_xnorm(layer), d_xnorm, E, e, sizeof(e)) ||
                          download_stage(dev, stg, stg_bytes, qrt_layer_attn_src(layer), d_src, inter, e, sizeof(e)) ||
                          download_stage(dev, stg, stg_bytes, qrt_layer_attn_out(layer), d_out, E, e, sizeof(e)) ||
                          download_stage(dev, stg, stg_bytes, qrt_layer_x(layer), d_res, E, e, sizeof(e)) ||
                          download_stage(dev, stg, stg_bytes, qrt_layer_post_norm(layer), d_post, E, e, sizeof(e));
                if (fb != NULL && drc == 0) {
                    d_moe = d_post;
                    d_shexp = d_post;
                    drc = download_stage(dev, stg, stg_bytes, fb->moe, d_moe, E, e, sizeof(e)) ||
                          download_stage(dev, stg, stg_bytes, fb->shexp, d_shexp, E, e, sizeof(e));
                }
                if (drc != 0) {
                    return fail(err, err_size, "layer %u stage readback: %s", l, e);
                }
                printf("model_stage layer=%u xnorm=%.3e src=%.3e outproj=%.3e res=%.3e "
                       "post=%.3e", l, l1_cmp(d_xnorm.data(), r_xnorm.data(), d_xnorm.size()),
                       l1_cmp(d_src.data(), r_src.data(), d_src.size()),
                       l1_cmp(d_out.data(), r_out.data(), d_out.size()),
                       l1_cmp(d_res.data(), r_res.data(), d_res.size()),
                       l1_cmp(d_post.data(), r_post.data(), d_post.size()));
                if (fb != NULL) {
                    printf(" moe=%.3e shexp=%.3e",
                           l1_cmp(d_moe.data(), r_moe.data(), d_moe.size()),
                           l1_cmp(d_shexp.data(), r_shexp.data(), d_shexp.size()));
                }
                printf("\n");
                const double rel = l1_cmp(x.data(), xref.data(), xref.size());
                printf("model_ref layer=%u hidden_rel_l1=%.3e rms_gpu=%.5f rms_ref=%.5f "
                       "ms_gpu=%.1f\n", l, rel, rms(x.data(), x.size()),
                       rms(xref.data(), xref.size()), ms_layer);
                if (res != NULL && rel > res->hidden_max) {
                    res->hidden_max = rel;
                }
            }
        }
        if (cfg.ffn == 0) {
            printf("model_note note=no_layer_store_expert_half_off\n");
        }
        if (res->want_hidden) {
            res->hidden = x;
        }
        if (!want_head) {
            printf("model_logits skipped=1\n");
            return 0;
        }

        const uint32_t start = n_tokens - 1u - opt.n_rows;
        std::vector<float> dev_logits, ref_logits;
        if (qrt_model_head_rows(g, outw, onorm.data(), x.data(), xref.data(), E, V, n_tokens, opt.n_rows,
                      start, &dev_logits, opt.ref_out_path != NULL ? &ref_logits : NULL,
                      e, sizeof(e)) != 0) {
            return fail(err, err_size, "head: %s", e);
        }
        {
            uint32_t top_dev = 0;
            for (uint32_t v = 1; v < V; ++v) {
                if (dev_logits[(size_t) (opt.n_rows - 1) * V + v] >
                    dev_logits[(size_t) (opt.n_rows - 1) * V + top_dev]) {
                    top_dev = v;
                }
            }
            printf("model_logits rows=%u start=%u top1=%u next=%u logit=%.4f\n", opt.n_rows,
                   start, top_dev, ids[start + opt.n_rows],
                   dev_logits[(size_t) (opt.n_rows - 1) * V + top_dev]);
        }

        if (opt.out_path != NULL &&
            qrt_model_write_dump(opt.out_path, V, opt.n_rows, start, ids, dev_logits.data(), V, e,
                       sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }
        if (opt.ref_out_path != NULL &&
            qrt_model_write_dump(opt.ref_out_path, V, opt.n_rows, start, ids, ref_logits.data(), V, e,
                       sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }
        if (opt.out_path != NULL || opt.ref_out_path != NULL) {
            printf("model_dump path=%s tokens=%u layers=%u vocab=%u rows=%u start=%u\n",
                   opt.out_path != NULL ? opt.out_path : opt.ref_out_path, n_tokens, layer_count,
                   V, opt.n_rows, start);
        }

        if (res != NULL && opt.ref_out_path != NULL) {
            double num = 0.0, den = 0.0, nll_d = 0.0, nll_r = 0.0;
            for (uint32_t r = 0; r < opt.n_rows; ++r) {
                const float * a = &dev_logits[(size_t) r * V];
                const float * b = &ref_logits[(size_t) r * V];
                num += l1_num(a, b, V);
                den += l1_den(b, V);
                uint32_t ta = 0, tb = 0;
                for (uint32_t v = 1; v < V; ++v) {
                    if (a[v] > a[ta]) {
                        ta = v;
                    }
                    if (b[v] > b[tb]) {
                        tb = v;
                    }
                }
                res->top1_rows++;
                res->top1_match += ta == tb ? 1u : 0u;
                const uint32_t target = ids[start + r + 1u];
                nll_d += logsumexp(a, V) - a[target];
                nll_r += logsumexp(b, V) - b[target];
            }
            res->logits_rel_l1 = den > 0.0 ? num / den : num;
            res->nll_device = nll_d / (double) opt.n_rows;
            res->nll_ref = nll_r / (double) opt.n_rows;
        }
    } while (0);
    return 0;
}

} // namespace

int qrt_model_logits(const qrt_config * cfg, const qrt_model_options * opt,
                     const char * gguf_path, const char * store_prefix, const char * spv_dir,
                     const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size) {
    run_opts run{};
    if (request_opts(cfg, opt, model_mode::logits, gguf_path, store_prefix, spv_dir, ids,
                     n_tokens, &run, err, err_size) != 0) return -1;
    run_result res{};
    return model_forward(*cfg, gguf_path, store_prefix, spv_dir, ids, n_tokens, run, &res,
                         err, err_size);
}

int qrt_model_check(const qrt_config * cfg, const qrt_model_options * opt,
                    const char * gguf_path, const char * store_prefix, const char * spv_dir,
                    const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size) {
    run_opts run{};
    if (request_opts(cfg, opt, model_mode::check, gguf_path, store_prefix, spv_dir, ids,
                     n_tokens, &run, err, err_size) != 0) return -1;
    run_result res{};
    const int rc = model_forward(*cfg, gguf_path, store_prefix, spv_dir, ids, n_tokens, run,
                                 &res, err, err_size);
    if (rc != 0) {
        return rc;
    }
    printf("model_check tokens=%u layers=%u reseed=%d hidden_max=%.3e replay_max=%.3e "
           "logits_rel_l1=%.3e top1=%u/%u nll_gpu=%.4f nll_ref=%.4f verdict=%s\n",
           n_tokens, opt->n_layers, opt->reseed, res.hidden_max, res.replay_max, res.logits_rel_l1,
           res.top1_match, res.top1_rows, res.nll_device, res.nll_ref,
           res.logits_rel_l1 <= 1e-4 ? "PASS" : "FAIL");
    return res.logits_rel_l1 <= 1e-4 ? 0 : 1;
}

// End-to-end rel_L1 and the worst element of one arm's hidden against another's.
namespace {

void hidden_diff(const run_result & a, const run_result & b, double * rel, double * max_abs) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < b.hidden.size(); ++i) {
        const double d = fabs((double) a.hidden[i] - (double) b.hidden[i]);
        num += d;
        den += fabs((double) b.hidden[i]);
        if (d > mx) {
            mx = d;
        }
    }
    *rel = den > 0.0 ? num / den : num;
    *max_abs = mx;
}

} // namespace

int qrt_model_chunk_check(const qrt_config * cfg, const qrt_model_options * opt,
                          const char * gguf_path, const char * store_prefix, const char * spv_dir,
                          const uint32_t * ids, uint32_t n_tokens, char * err, size_t err_size) {
    run_opts run{};
    if (request_opts(cfg, opt, model_mode::chunk_check, gguf_path, store_prefix, spv_dir, ids,
                     n_tokens, &run, err, err_size) != 0) return -1;
    const uint32_t chunk = (uint32_t) cfg->prefill_chunk;
    run_result arms[3]{};
    const uint32_t specs[3][2] = { { 0u, 1u }, { chunk, 1u }, { chunk, 0u } };
    for (uint32_t a = 0; a < 3; ++a) {
        run.chunk = specs[a][0];
        run.carry = (int) specs[a][1];
        arms[a].want_hidden = true;
        const int rc = model_forward(*cfg, gguf_path, store_prefix, spv_dir, ids, n_tokens, run,
                                     &arms[a], err, err_size);
        if (rc != 0) {
            return rc;
        }
    }
    double rel_one = 0.0, max_one = 0.0, rel_no = 0.0, max_no = 0.0;
    hidden_diff(arms[1], arms[0], &rel_one, &max_one);
    hidden_diff(arms[2], arms[0], &rel_no, &max_no);
    const int ok = rel_one == 0.0 && max_one == 0.0 && rel_no > 0.0;
    printf("chunk_check tokens=%u chunk=%u carry=%d layers=%u "
           "carry_vs_single_rel_l1=%.3e carry_vs_single_max=%.3e "
           "nocarry_vs_single_rel_l1=%.3e nocarry_vs_single_max=%.3e verdict=%s\n",
           n_tokens, chunk, 1, opt->n_layers, rel_one, max_one, rel_no, max_no,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

