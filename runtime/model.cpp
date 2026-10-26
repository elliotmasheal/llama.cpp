// Whole-model prefill and the QRTLOG1 logits dump (M2-7a). See model.h for the scope.

#include "model.h"

#include "attn_full.h"
#include "cpu_moe.h"
#include "expert_cache.h"
#include "ffn.h"
#include "gguf_index.h"
#include "layer.h"
#include "miss_split.h"
#include "qrt_config.h"
#include "qrt_telemetry.h"
#include "vk_dev.h"
#include "weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (err != NULL && err_size != 0) {
        vsnprintf(err, err_size, fmt, ap);
    }
    va_end(ap);
    return -1;
}

float f16_to_f32(uint32_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t expo = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    float v;
    if (expo == 0) {
        v = ldexpf((float) mant, -24);
    } else if (expo == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = ldexpf((float) (mant | 0x400u), (int) expo - 25);
    }
    return sign ? -v : v;
}

// One q8_0 row (ne0/32 blocks of 34 B) to f32.
void q8_0_row(const uint8_t * src, uint32_t n, float * dst) {
    for (uint32_t b = 0; b < n / 32u; ++b) {
        const uint8_t * blk = src + (size_t) b * 34u;
        const float d = f16_to_f32((uint32_t) blk[0] | ((uint32_t) blk[1] << 8));
        for (uint32_t i = 0; i < 32u; ++i) {
            dst[b * 32u + i] = (float) ((int8_t) blk[2u + i]) * d;
        }
    }
}

int read_tensor_f32(qrt_gguf * g, const char * name, std::vector<float> * out,
                    char * err, size_t err_size) {
    const qrt_gguf_tensor * t = qrt_gguf_find(g, name);
    if (t == NULL) {
        return fail(err, err_size, "%s is not in the tensor table", name);
    }
    if (t->type != 0) {
        return fail(err, err_size, "%s is not f32", name);
    }
    uint64_t count = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) {
        count *= t->ne[i];
    }
    out->assign((size_t) count, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err, err_size);
}

// token_embd is q8_0 and stays on the host (model.h): one row read per token.
int embed_rows(qrt_gguf * g, const qrt_gguf_tensor * emb, uint32_t E, uint32_t V,
               const uint32_t * ids, uint32_t n_tokens, float * x, char * err, size_t err_size) {
    const uint32_t row_bytes = (E / 32u) * 34u;
    std::vector<uint8_t> row((size_t) row_bytes);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        if (ids[t] >= V) {
            return fail(err, err_size, "token %u is %u, the vocabulary is %u", t, ids[t], V);
        }
        if (qrt_gguf_read_range(g, emb->offset + (uint64_t) ids[t] * row_bytes, row.data(),
                                row_bytes, err, err_size) != 0) {
            return -1;
        }
        q8_0_row(row.data(), E, x + (size_t) t * E);
    }
    return 0;
}

// Final rms norm of the dumped rows, then output.weight over the whole vocabulary, in one
// pass. Both arms go through the same vocab row, so output.weight is read once per run:
// at 512 rows the per-row version would re-read 1.7 GB for every row. `b` and `la` may be
// NULL when that arm is not wanted.
int head_rows(qrt_gguf * g, const qrt_gguf_tensor * outw, const float * onorm,
              const float * xa, const float * xb, uint32_t E, uint32_t V, uint32_t n_tokens,
              uint32_t n_rows, uint32_t start, std::vector<float> * la, std::vector<float> * lb,
              char * err, size_t err_size) {
    const uint32_t row_bytes = (E / 32u) * 34u;
    std::vector<double> xn_a((size_t) n_rows * E);
    std::vector<double> xn_b(lb != NULL ? (size_t) n_rows * E : 0);
    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t j = start + r;
        for (uint32_t arm = 0; arm < (lb != NULL ? 2u : 1u); ++arm) {
            const float * src = (arm == 0 ? xa : xb) + (size_t) j * E;
            std::vector<double> & xn = arm == 0 ? xn_a : xn_b;
            double ss = 0.0;
            for (uint32_t i = 0; i < E; ++i) {
                ss += (double) src[i] * (double) src[i];
            }
            const double inv = 1.0 / sqrt(ss / (double) E + 1e-6);
            for (uint32_t i = 0; i < E; ++i) {
                xn[(size_t) r * E + i] = (double) src[i] * inv * (double) onorm[i];
            }
        }
    }
    if (la != NULL) {
        la->assign((size_t) n_rows * V, 0.0f);
    }
    if (lb != NULL) {
        lb->assign((size_t) n_rows * V, 0.0f);
    }
    std::vector<uint8_t> raw((size_t) row_bytes);
    std::vector<float> row_f((size_t) E);
    for (uint32_t v = 0; v < V; ++v) {
        if (qrt_gguf_read_range(g, outw->offset + (uint64_t) v * row_bytes, raw.data(),
                                row_bytes, err, err_size) != 0) {
            return -1;
        }
        q8_0_row(raw.data(), E, row_f.data());
        for (uint32_t r = 0; r < n_rows; ++r) {
            if (la != NULL) {
                const double * a = &xn_a[(size_t) r * E];
                double acc = 0.0;
                for (uint32_t i = 0; i < E; ++i) {
                    acc += (double) row_f[i] * a[i];
                }
                (*la)[(size_t) r * V + v] = (float) acc;
            }
            if (lb != NULL) {
                const double * b = &xn_b[(size_t) r * E];
                double acc_b = 0.0;
                for (uint32_t i = 0; i < E; ++i) {
                    acc_b += (double) row_f[i] * b[i];
                }
                (*lb)[(size_t) r * V + v] = (float) acc_b;
            }
        }
    }
    return 0;
}

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

struct run_owner {
    qrt_vk_dev * dev = nullptr;
    qrt_gguf * g = nullptr;
    qrt_vk_staging * stg = nullptr;

    run_owner() = default;
    run_owner(const run_owner &) = delete;
    run_owner & operator=(const run_owner &) = delete;
    ~run_owner() {
        if (stg != nullptr) qrt_vk_staging_destroy(dev, stg);
        if (g != nullptr) qrt_gguf_close(g);
        if (dev != nullptr) qrt_vk_close(dev);
    }
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

// Write one QRTLOG1 v1 dump: 24 B header, the tokens of the covered range plus one, then
// n_rows f32 rows (bench/tools/quality/README.md section 3).
int write_dump(const char * path, uint32_t V, uint32_t n_rows, uint32_t start,
               const uint32_t * ids, const float * logits, size_t row_stride,
               char * err, size_t err_size) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return fail(err, err_size, "cannot write '%s'", path);
    }
    const char magic[8] = { 'Q', 'R', 'T', 'L', 'O', 'G', '1', '\0' };
    const uint32_t version = 1;
    const uint32_t hdr[4] = { version, V, n_rows, start };
    bool short_write = fwrite(magic, 1, 8, f) != 8 ||
                       fwrite(hdr, sizeof(uint32_t), 4, f) != 4 ||
                       fwrite(ids + start, sizeof(uint32_t), n_rows + 1u, f) != n_rows + 1u;
    for (uint32_t r = 0; r < n_rows && !short_write; ++r) {
        short_write = fwrite(logits + (size_t) r * row_stride, sizeof(float), V, f) != V;
    }
    const int close_rc = fclose(f);
    if (short_write || close_rc != 0) {
        return fail(err, err_size, "cannot write '%s': %s%s", path,
                    short_write ? "short write" : "", close_rc != 0 ? " close failed" : "");
    }
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
    run_owner owner;
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
        if (embed_rows(g, emb, E, V, ids, n_tokens, x.data(), e, sizeof(e)) != 0) {
            return fail(err, err_size, "embedding: %s", e);
        }
        if (want_head && (read_tensor_f32(g, "output_norm.weight", &onorm, e, sizeof(e)) != 0 ||
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
        if (head_rows(g, outw, onorm.data(), x.data(), xref.data(), E, V, n_tokens, opt.n_rows,
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
            write_dump(opt.out_path, V, opt.n_rows, start, ids, dev_logits.data(), V, e,
                       sizeof(e)) != 0) {
            return fail(err, err_size, "%s", e);
        }
        if (opt.ref_out_path != NULL &&
            write_dump(opt.ref_out_path, V, opt.n_rows, start, ids, ref_logits.data(), V, e,
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

int qrt_model_read_ids(const char * path, uint32_t ** ids, uint32_t * count,
                       char * err, size_t err_size) {
    if (ids != NULL) *ids = NULL;
    if (count != NULL) *count = 0;
    if (path == NULL || *path == '\0' || ids == NULL || count == NULL) {
        return fail(err, err_size, "token input needs a path and output pointers");
    }
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path, "rb"), &fclose);
    if (!f) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    std::vector<uint32_t> v;
    uint32_t value = 0;
    bool in_token = false;
    for (;;) {
        const int c = fgetc(f.get());
        if (c >= '0' && c <= '9') {
            const uint32_t digit = (uint32_t) (c - '0');
            if (value > ((uint32_t) INT32_MAX - digit) / 10u) {
                return fail(err, err_size, "'%s': token exceeds INT32_MAX", path);
            }
            value = value * 10u + digit;
            in_token = true;
            continue;
        }
        if (c != EOF && c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') {
            return fail(err, err_size, "'%s': token ids must be complete unsigned decimal integers", path);
        }
        if (c == EOF && ferror(f.get())) {
            return fail(err, err_size, "cannot read '%s'", path);
        }
        if (in_token) {
            if (v.size() >= UINT32_MAX || v.size() >= SIZE_MAX / sizeof(uint32_t)) {
                return fail(err, err_size, "'%s': too many token ids", path);
            }
            v.push_back(value);
            value = 0;
            in_token = false;
        }
        if (c == EOF) break;
    }
    if (fclose(f.release()) != 0) {
        return fail(err, err_size, "cannot close token input '%s'", path);
    }
    if (v.empty()) {
        return fail(err, err_size, "'%s' holds no token ids", path);
    }
    uint32_t * result = (uint32_t *) malloc(v.size() * sizeof(uint32_t));
    if (result == NULL) {
        return fail(err, err_size, "out of memory for %zu token ids", v.size());
    }
    memcpy(result, v.data(), v.size() * sizeof(uint32_t));
    *ids = result;
    *count = (uint32_t) v.size();
    return 0;
}

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

// End-to-end vertical slice (M3-2): open every requested layer once with one shared FFN
// slot, prefill the prompt through the same objects, then decode one token at a time with
// the KV/DeltaNet state carried. The head stays on the host for this slice.
int qrt_model_generate(const qrt_config * cfg, const qrt_model_options * opt,
                       const char * gguf_path, const char * store_prefix, const char * spv_dir,
                       const uint32_t * ids, uint32_t n_tokens, uint32_t n_new,
                       const char * ids_out, char * err, size_t err_size) {
    if (cfg == NULL || opt == NULL || gguf_path == NULL || store_prefix == NULL ||
        spv_dir == NULL || ids == NULL || n_tokens == 0 || n_new == 0) {
        return fail(err, err_size, "generate needs config, paths, prompt ids and n_new");
    }
    if (opt->n_layers == 0 || opt->n_layers > 40) {
        return fail(err, err_size, "generate needs 1..40 layers, got %u", opt->n_layers);
    }
    uint32_t chunk = (uint32_t) cfg->prefill_chunk;
    if (chunk > 64u) {
        // The 40-layer slice is sized for a small resident chunk. A 2048-token chunk across
        // 40 layers does not fit in 8 GiB; the shared scratch arena is the next step.
        chunk = 32u;
    }
    if (chunk < 32u || (chunk % 32u) != 0u) {
        return fail(err, err_size, "generate chunk %u is not a 32-aligned 32..64", chunk);
    }
    const uint64_t n_ctx64 = (uint64_t) n_tokens + n_new + 1u;
    if (n_ctx64 > QRT_KV_CTX_TOKENS) {
        return fail(err, err_size, "prompt %u + new %u exceeds the %u context",
                    n_tokens, n_new, (uint32_t) QRT_KV_CTX_TOKENS);
    }
    const uint32_t n_ctx = (uint32_t) n_ctx64;

    char e[512] = {0};
    run_owner owner;
    qrt_vk_dev * & dev = owner.dev;
    qrt_gguf * & g = owner.g;
    qrt_vk_staging * & stg = owner.stg;
    VkBuffer slot = VK_NULL_HANDLE;
    VkDeviceMemory slot_mem = VK_NULL_HANDLE;
    qrt_weights * wstore = NULL;
    std::vector<qrt_layer *> layers;
    std::vector<uint32_t> generated;
    qrt_cache * cache = NULL;
    qrt_cpu_moe * cpu = NULL;
    std::vector<VkBuffer> cache_bufs;
    std::vector<VkDeviceMemory> cache_mems;
    std::vector<uint32_t> cache_slots;
    qrt_split_rates rates{};
    qrt_split_carry carry{};
    int cache_on = 0;
    qrt_telemetry * tel = NULL;
    double leg_gpu = 0.0, leg_h2d = 0.0, leg_cpu = 0.0;
    uint64_t leg_hits = 0, leg_miss = 0, leg_h2d_n = 0, leg_cpu_n = 0, leg_h2d_bytes = 0;
    int rc = -1;
    do {
        if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        if (qrt_gguf_open(gguf_path, &g, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(g);
        if (hp == NULL || !(hp->present & QRT_HP_BLOCK_COUNT) || hp->block_count <= 0) {
            fail(err, err_size, "missing or invalid block_count");
            break;
        }
        const int32_t nextn = hp->present & QRT_HP_NEXTN ? hp->nextn : 0;
        if (nextn < 0 || nextn >= hp->block_count ||
            opt->n_layers > (uint32_t) (hp->block_count - nextn)) {
            fail(err, err_size, "requested %u layers, block_count=%d nextn=%d",
                 opt->n_layers, hp->block_count, nextn);
            break;
        }
        if (hp->n_embd <= 0 || hp->n_embd % 32 != 0) {
            fail(err, err_size, "n_embd must be a positive multiple of 32");
            break;
        }
        const uint32_t E = (uint32_t) hp->n_embd;
        const qrt_gguf_tensor * emb = qrt_gguf_find(g, "token_embd.weight");
        const qrt_gguf_tensor * outw = qrt_gguf_find(g, "output.weight");
        if (emb == NULL || emb->type != 8 || emb->n_dims != 2 || emb->ne[0] != E ||
            emb->ne[1] == 0 || emb->ne[1] > INT32_MAX) {
            fail(err, err_size, "token_embd must be q8_0 with n_embd x vocabulary rows");
            break;
        }
        if (outw == NULL || outw->type != 8 || outw->n_dims != 2 || outw->ne[0] != E ||
            outw->ne[1] != emb->ne[1]) {
            fail(err, err_size, "output must match the q8_0 embedding shape");
            break;
        }
        const uint32_t V = (uint32_t) emb->ne[1];
        std::vector<float> x((size_t) n_tokens * E);
        if (embed_rows(g, emb, E, V, ids, n_tokens, x.data(), e, sizeof(e)) != 0) {
            fail(err, err_size, "embedding: %s", e);
            break;
        }
        std::vector<float> onorm;
        if (read_tensor_f32(g, "output_norm.weight", &onorm, e, sizeof(e)) != 0 ||
            onorm.size() != E) {
            fail(err, err_size, "output_norm: %s", e[0] ? e : "wrong width");
            break;
        }

        if (qrt_weights_open(store_prefix, &wstore, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        const uint32_t n_expert = qrt_weights_experts(wstore);
        const uint64_t max_expert = qrt_weights_max_expert_bytes(wstore);
        qrt_weights_close(wstore);
        wstore = NULL;
        if (n_expert == 0 || max_expert == 0) {
            fail(err, err_size, "the store has no experts");
            break;
        }
        uint64_t slot_bytes = (uint64_t) n_expert * max_expert;
        slot_bytes = (slot_bytes + 65535u) & ~(uint64_t) 65535u;
        if (qrt_vk_alloc(dev, slot_bytes, QRT_VK_USAGE_STORAGE |
                         QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                         &slot, &slot_mem, e, sizeof(e)) != 0) {
            fail(err, err_size, "shared FFN slot: %s", e);
            break;
        }
        const uint64_t out_row = (uint64_t) chunk * E * 4;
        if (qrt_vk_staging_create(dev, out_row, &stg, e, sizeof(e)) != 0) {
            fail(err, err_size, "%s", e);
            break;
        }
        printf("generate_run tokens=%u new=%u layers=%u chunk=%u n_ctx=%u slot_mib=%.1f\n",
               n_tokens, n_new, opt->n_layers, chunk, n_ctx,
               (double) slot_bytes / (1024.0 * 1024.0));

        qrt_layer_cfg lcfg{};
        lcfg.ffn = 1;
        lcfg.single_submit = 1;
        lcfg.dn_recur = 1;
        lcfg.kv_k = cfg->kv_k;
        lcfg.kv_v = cfg->kv_v;
        lcfg.kv_page_tokens = (uint32_t) cfg->kv_page_tokens;
        lcfg.n_ctx = n_ctx;
        lcfg.ffn_slot = slot;
        lcfg.ffn_slot_bytes = slot_bytes;
        layers.assign(opt->n_layers, nullptr);
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            if (qrt_layer_open(dev, gguf_path, store_prefix, l, chunk, spv_dir, &lcfg,
                               &layers[l], e, sizeof(e)) != 0) {
                fail(err, err_size, "open layer %u: %s", l, e);
                break;
            }
            if (qrt_layer_zero_state(layers[l], e, sizeof(e)) != 0) {
                fail(err, err_size, "zero layer %u: %s", l, e);
                break;
            }
        }
        if (layers[opt->n_layers - 1] == nullptr) {
            break;
        }

        std::vector<float> xnext((size_t) chunk * E);
        const uint32_t n_chunks = 1u + (n_tokens - 1u) / chunk;
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            const double t0 = now_ms();
            for (uint32_t c = 0; c < n_chunks; ++c) {
                const uint32_t pos = c * chunk;
                const uint32_t len = n_tokens - pos < chunk ? n_tokens - pos : chunk;
                if (qrt_layer_set_input_rows(layers[l], x.data() + (size_t) pos * E, len, e,
                                             sizeof(e)) != 0 ||
                    qrt_layer_run_chunk(layers[l], pos, len, e, sizeof(e)) != 0 ||
                    qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, xnext.data(),
                                    (uint64_t) len * E * 4, e, sizeof(e)) != 0) {
                    fail(err, err_size, "prefill layer %u chunk %u: %s", l, c, e);
                    break;
                }
                memcpy(x.data() + (size_t) pos * E, xnext.data(), (size_t) len * E * 4);
            }
            if (e[0] != '\0') {
                break;
            }
            printf("generate_prefill layer=%u chunks=%u ms=%.1f\n", l, n_chunks,
                   now_ms() - t0);
        }
        if (e[0] != '\0') {
            break;
        }

        // Decode expert cache (M3-2a). The prefill above packed every used expert into
        // the shared slot; decode switches to per-layer slot arrays sized by the cache
        // plan, and the split sends the misses to H2D or the CPU leg. A failed cache
        // allocation falls back to the prefill slot (slow but correct), not to an abort.
        uint64_t cache_mib = 0;
        if (cfg->expert_cache_mib > 0) {
            uint64_t layer_slot[QRT_CACHE_MAX_LAYERS] = {0};
            if (qrt_weights_open(store_prefix, &wstore, e, sizeof(e)) != 0) {
                fail(err, err_size, "%s", e);
                break;
            }
            int size_ok = 1;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                if (qrt_weights_layer_expert_bytes(wstore, l, &layer_slot[l]) != 0) {
                    size_ok = 0;
                    break;
                }
            }
            qrt_weights_close(wstore);
            wstore = NULL;
            qrt_cache_plan plan{};
            if (!size_ok ||
                qrt_cache_plan_build(opt->n_layers, n_expert, layer_slot, NULL,
                                     (uint64_t) cfg->expert_cache_mib << 20,
                                     cfg->cache_policy, QRT_CACHE_DEFAULT_PROTECTED_PCT,
                                     &plan, e, sizeof(e)) != 0) {
                fail(err, err_size, "expert cache plan: %s", e[0] ? e : "no layer sizes");
                break;
            }
            // The prefill slot has done its job; the cache takes its VRAM.
            qrt_vk_free(dev, slot, slot_mem);
            slot = VK_NULL_HANDLE;
            slot_mem = VK_NULL_HANDLE;
            int alloc_ok = 1;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                const uint64_t bytes = (uint64_t) plan.slots_per_layer[l] * plan.slot_bytes[l];
                VkBuffer b = VK_NULL_HANDLE;
                VkDeviceMemory m = VK_NULL_HANDLE;
                if (bytes != 0 &&
                    qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE |
                                 QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                                 &b, &m, e, sizeof(e)) != 0) {
                    alloc_ok = 0;
                    break;
                }
                cache_bufs.push_back(b);
                cache_mems.push_back(m);
                cache_slots.push_back(plan.slots_per_layer[l]);
            }
            if (!alloc_ok) {
                for (size_t i = 0; i < cache_bufs.size(); ++i) {
                    if (cache_bufs[i] != VK_NULL_HANDLE) {
                        qrt_vk_free(dev, cache_bufs[i], cache_mems[i]);
                    }
                }
                cache_bufs.clear();
                cache_mems.clear();
                cache_slots.clear();
                if (qrt_vk_alloc(dev, slot_bytes, QRT_VK_USAGE_STORAGE |
                                 QRT_VK_USAGE_TRANSFER_DST | QRT_VK_USAGE_TRANSFER_SRC,
                                 &slot, &slot_mem, e, sizeof(e)) != 0) {
                    fail(err, err_size, "cache allocation failed and the fallback slot "
                         "did not fit: %s", e);
                    break;
                }
                printf("generate_cache disabled: %s\n", e[0] ? e : "allocation");
                e[0] = '\0';
            } else if (qrt_cache_open(&plan, NULL, &cache, e, sizeof(e)) != 0) {
                fail(err, err_size, "%s", e);
                break;
            } else {
                cpu = qrt_cpu_moe_create(1);
                if (cpu == NULL) {
                    fail(err, err_size, "out of memory for the CPU expert core");
                    break;
                }
                qrt_miss_split_default_rates(&rates);
                for (uint32_t l = 0; l < opt->n_layers; ++l) {
                    if (qrt_layer_set_cache(layers[l], cache, cache_bufs[l],
                                            plan.slot_bytes[l], cpu, cfg->miss_split, &rates,
                                            &carry, e, sizeof(e)) != 0) {
                        fail(err, err_size, "cache on layer %u: %s", l, e);
                        break;
                    }
                }
                if (e[0] != '\0') {
                    break;
                }
                cache_on = 1;
                cache_mib = plan.bytes >> 20;
                printf("generate_cache slots=%llu mib=%llu degraded=%u policy=%s "
                       "split=%s threads=8\n",
                       (unsigned long long) plan.slots, (unsigned long long) cache_mib,
                       plan.degraded_layers, qrt_cache_policy_name(cfg->cache_policy),
                       cfg->miss_split == QRT_MISS_SPLIT_AUTO ? "auto" : "fixed");
            }
        }

        if (cfg->telemetry == 1) {
            char terr[256] = {0};
            tel = qrt_telemetry_open(cfg->telemetry_path, terr, sizeof(terr));
            if (tel == NULL) {
                printf("generate_telemetry disabled: %s\n", terr);
            }
        }

        std::vector<float> logits((size_t) V);
        auto pick = [&](const std::vector<float> & lg, uint32_t * id, float * best,
                        float * margin) {
            uint32_t bi = 0, si = 0;
            float bv = lg[0], sv = -INFINITY;
            for (uint32_t v = 1; v < V; ++v) {
                if (lg[v] > bv) {
                    sv = bv; si = bi; bv = lg[v]; bi = v;
                } else if (lg[v] > sv) {
                    sv = lg[v]; si = v;
                }
            }
            (void) si;
            *id = bi; *best = bv; *margin = bv - sv;
        };
        auto head_one = [&](const float * hidden, std::vector<float> & lg) -> int {
            return head_rows(g, outw, onorm.data(), hidden, NULL, E, V, 1, 1, 0, &lg, NULL,
                             e, sizeof(e));
        };
        uint32_t tok = 0;
        float best = 0.0f, margin = 0.0f;
        if (head_one(x.data() + (size_t) (n_tokens - 1u) * E, logits) != 0) {
            fail(err, err_size, "prefill head: %s", e);
            break;
        }
        pick(logits, &tok, &best, &margin);
        generated.push_back(tok);
        printf("generate_token step=0 pos=%u id=%u logit=%.4f margin=%.4f\n",
               n_tokens, tok, best, margin);

        std::vector<float> x1((size_t) E);
        uint32_t pos = n_tokens;
        if (opt->batch_k > 1) {
            // Diagnostic decode width (M3-2a): one pass carries k rows so the miss legs
            // are measured at k=5. The rows are prompt ids fed teacher-forced; the head
            // output is not used as the next input, so this is not an acceptance path.
            const uint32_t BK = opt->batch_k;
            std::vector<float> xb((size_t) BK * E);
            std::vector<uint32_t> bid(BK);
            uint32_t produced = 1;
            uint32_t pass = 0;
            while (produced < n_new) {
                const uint32_t k = std::min(BK, n_new - produced);
                for (uint32_t i = 0; i < k; ++i) {
                    bid[i] = ids[(size_t) (pass * BK + i) % n_tokens];
                }
                if (embed_rows(g, emb, E, V, bid.data(), k, xb.data(), e, sizeof(e)) != 0) {
                    fail(err, err_size, "batch embedding: %s", e);
                    break;
                }
                double step_gpu = 0.0, step_h2d = 0.0, step_cpu = 0.0;
                uint32_t step_hits = 0, step_miss = 0, step_h2d_n = 0, step_cpu_n = 0;
                const double t0 = now_ms();
                for (uint32_t l = 0; l < opt->n_layers; ++l) {
                    if (qrt_layer_set_input_rows(layers[l], xb.data(), k, e, sizeof(e)) != 0 ||
                        qrt_layer_run_chunk(layers[l], pos, k, e, sizeof(e)) != 0 ||
                        qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, xb.data(),
                                        (uint64_t) k * E * 4, e, sizeof(e)) != 0) {
                        fail(err, err_size, "batch layer %u pass %u: %s", l, pass, e);
                        break;
                    }
                    if (cache_on) {
                        const qrt_ffn_legs lg = qrt_layer_last_legs(layers[l]);
                        const double gpu = qrt_layer_gpu_ms(layers[l]);
                        step_gpu += gpu;
                        step_h2d += lg.h2d_ms;
                        step_cpu += lg.cpu_ms;
                        step_hits += lg.hits;
                        step_miss += lg.misses;
                        step_h2d_n += lg.h2d;
                        step_cpu_n += lg.cpu;
                        leg_gpu += gpu;
                        leg_h2d += lg.h2d_ms;
                        leg_cpu += lg.cpu_ms;
                        leg_hits += lg.hits;
                        leg_miss += lg.misses;
                        leg_h2d_n += lg.h2d;
                        leg_cpu_n += lg.cpu;
                        leg_h2d_bytes += lg.h2d_bytes;
                        if (qrt_layer_take_cpu_out(layers[l], xb.data(), k) != 0) {
                            fail(err, err_size, "layer %u CPU output on pass %u", l, pass);
                            break;
                        }
                    }
                }
                if (e[0] != '\0') {
                    break;
                }
                if (head_one(xb.data() + (size_t) (k - 1u) * E, logits) != 0) {
                    fail(err, err_size, "batch head pass %u: %s", pass, e);
                    break;
                }
                pick(logits, &tok, &best, &margin);
                generated.push_back(tok);
                printf("generate_batch pass=%u pos=%u k=%u id=%u logit=%.4f margin=%.4f "
                       "gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f hits=%u misses=%u h2d=%u cpu=%u "
                       "pass_ms=%.1f\n",
                       pass, pos, k, tok, best, margin, step_gpu, step_h2d, step_cpu,
                       step_hits, step_miss, step_h2d_n, step_cpu_n, now_ms() - t0);
                pos += k;
                produced += k;
                ++pass;
            }
        } else
        for (uint32_t s = 1; s < n_new; ++s) {
            if (embed_rows(g, emb, E, V, &generated.back(), 1, x1.data(), e, sizeof(e)) != 0) {
                fail(err, err_size, "decode embedding: %s", e);
                break;
            }
            const double t0 = now_ms();
            double step_gpu = 0.0, step_h2d = 0.0, step_cpu = 0.0;
            uint32_t step_hits = 0, step_miss = 0, step_h2d_n = 0, step_cpu_n = 0;
            uint64_t step_h2d_bytes = 0;
            for (uint32_t l = 0; l < opt->n_layers; ++l) {
                const double tl = now_ms();
                if (qrt_layer_set_input_rows(layers[l], x1.data(), 1, e, sizeof(e)) != 0 ||
                    qrt_layer_run_chunk(layers[l], pos, 1, e, sizeof(e)) != 0 ||
                    qrt_vk_download(dev, stg, qrt_layer_out(layers[l]), 0, x1.data(),
                                    (uint64_t) E * 4, e, sizeof(e)) != 0) {
                    fail(err, err_size, "decode layer %u step %u: %s", l, s, e);
                    break;
                }
                if (cache_on) {
                    const qrt_ffn_legs lg = qrt_layer_last_legs(layers[l]);
                    const double gpu = qrt_layer_gpu_ms(layers[l]);
                    (void) tl;
                    step_gpu += gpu;
                    step_h2d += lg.h2d_ms;
                    step_cpu += lg.cpu_ms;
                    step_hits += lg.hits;
                    step_miss += lg.misses;
                    step_h2d_n += lg.h2d;
                    step_cpu_n += lg.cpu;
                    step_h2d_bytes += lg.h2d_bytes;
                    leg_gpu += gpu;
                    leg_h2d += lg.h2d_ms;
                    leg_cpu += lg.cpu_ms;
                    leg_hits += lg.hits;
                    leg_miss += lg.misses;
                    leg_h2d_n += lg.h2d;
                    leg_cpu_n += lg.cpu;
                    leg_h2d_bytes += lg.h2d_bytes;
                    if (qrt_layer_take_cpu_out(layers[l], x1.data(), 1) != 0) {
                        fail(err, err_size, "layer %u CPU output on step %u", l, s);
                        break;
                    }
                }
            }
            if (e[0] != '\0') {
                break;
            }
            if (cache_on) {
                printf("generate_legs step=%u gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f "
                       "hits=%u misses=%u h2d=%u cpu=%u\n", s, step_gpu, step_h2d, step_cpu,
                       step_hits, step_miss, step_h2d_n, step_cpu_n);
            }
            if (head_one(x1.data(), logits) != 0) {
                fail(err, err_size, "decode head step %u: %s", s, e);
                break;
            }
            pick(logits, &tok, &best, &margin);
            generated.push_back(tok);
            printf("generate_token step=%u pos=%u id=%u logit=%.4f margin=%.4f layer_ms=%.1f\n",
                   s, pos, tok, best, margin, now_ms() - t0);
            if (tel != NULL) {
                qrt_pass_stats ps{};
                ps.k = 1;
                ps.pos = pos;
                ps.accepted = 1;
                ps.hit_experts = step_hits;
                ps.miss_experts = step_miss;
                ps.miss_cpu = step_cpu_n;
                ps.miss_h2d = step_h2d_n;
                ps.h2d_bytes = step_h2d_bytes;
                ps.draft_ms = 0.0;
                ps.gpu_ms = step_gpu;
                ps.cpu_leg_ms = step_cpu;
                ps.pass_ms = now_ms() - t0;
                qrt_telemetry_pass(tel, s, &ps);
            }
            ++pos;
        }
        if (e[0] != '\0') {
            break;
        }
        uint32_t max_submits = 0, max_dispatches = 0;
        for (uint32_t l = 0; l < opt->n_layers; ++l) {
            max_submits = std::max(max_submits, qrt_layer_submits(layers[l]));
            max_dispatches = std::max(max_dispatches, qrt_layer_dispatches(layers[l]));
        }
        printf("generate_done ids=%zu alloc_mib=%.1f peak_mib=%.1f failures=%u "
               "submits_per_layer=%u dispatches_max=%u topk_readbacks_per_layer=1\n",
               generated.size(), (double) dev->alloc_bytes / (1024.0 * 1024.0),
               (double) dev->alloc_peak_bytes / (1024.0 * 1024.0), dev->alloc_failures,
               max_submits, max_dispatches);
        if (cache_on) {
            qrt_cache_stats cs{};
            qrt_cache_get_stats(cache, &cs);
            const double hi = std::max(leg_gpu, std::max(leg_h2d, leg_cpu));
            const double lo = std::min(leg_gpu, std::min(leg_h2d, leg_cpu));
            printf("generate_legs_total gpu_ms=%.2f h2d_ms=%.2f cpu_ms=%.2f spread_pct=%.1f "
                   "hits=%llu misses=%llu h2d_experts=%llu cpu_experts=%llu h2d_mib=%.1f "
                   "cache_lookups=%llu cache_hits=%llu\n",
                   leg_gpu, leg_h2d, leg_cpu, hi > 0.0 ? (hi - lo) * 100.0 / hi : 0.0,
                   (unsigned long long) leg_hits, (unsigned long long) leg_miss,
                   (unsigned long long) leg_h2d_n, (unsigned long long) leg_cpu_n,
                   (double) leg_h2d_bytes / (1024.0 * 1024.0),
                   (unsigned long long) cs.lookups, (unsigned long long) cs.hits);
        }
        if (ids_out != NULL) {
            FILE * fo = fopen(ids_out, "wb");
            if (fo == NULL) {
                fail(err, err_size, "cannot open '%s'", ids_out);
                break;
            }
            for (size_t i = 0; i < generated.size(); ++i) {
                if (fprintf(fo, "%u\n", generated[i]) < 0) {
                    fclose(fo);
                    fail(err, err_size, "cannot write '%s'", ids_out);
                    break;
                }
            }
            if (fclose(fo) != 0) {
                fail(err, err_size, "cannot close '%s'", ids_out);
                break;
            }
        }
        if (tel != NULL) {
            if (qrt_telemetry_failed(tel)) {
                printf("generate_telemetry had write failures after %llu rows\n",
                       (unsigned long long) tel->rows);
            }
            qrt_telemetry_close(tel);
            tel = NULL;
        }
        rc = 0;
    } while (0);

    for (size_t i = layers.size(); i > 0; --i) {
        if (layers[i - 1] != NULL) {
            qrt_layer_close(dev, layers[i - 1]);
        }
    }
    if (tel != NULL) {
        qrt_telemetry_close(tel);
    }
    if (cache != NULL) {
        qrt_cache_close(cache);
    }
    for (size_t i = 0; i < cache_bufs.size(); ++i) {
        if (cache_bufs[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, cache_bufs[i], cache_mems[i]);
        }
    }
    if (cpu != NULL) {
        qrt_cpu_moe_free(cpu);
    }
    if (slot_mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, slot, slot_mem);
    }
    if (wstore != NULL) {
        qrt_weights_close(wstore);
    }
    return rc;
}
