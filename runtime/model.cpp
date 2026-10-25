// Whole-model prefill and the QRTLOG1 logits dump (M2-7a). See model.h for the scope.

#include "model.h"

#include "attn_full.h"
#include "gguf_index.h"
#include "layer.h"
#include "qrt_config.h"
#include "vk_dev.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace {

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
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

} // namespace

int qrt_model_read_ids(const char * path, uint32_t ** ids, uint32_t * count,
                       char * err, size_t err_size) {
    *ids = NULL;
    *count = 0;
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    std::vector<uint32_t> v;
    long x = 0;
    while (fscanf(f, "%ld", &x) == 1) {
        if (x < 0 || x > 0x7fffffffL) {
            fclose(f);
            return fail(err, err_size, "'%s': token %ld is out of range", path, x);
        }
        v.push_back((uint32_t) x);
    }
    fclose(f);
    if (v.empty()) {
        return fail(err, err_size, "'%s' holds no token ids", path);
    }
    *ids = (uint32_t *) malloc(v.size() * sizeof(uint32_t));
    if (*ids == NULL) {
        return fail(err, err_size, "out of memory for %zu token ids", v.size());
    }
    memcpy(*ids, v.data(), v.size() * sizeof(uint32_t));
    *count = (uint32_t) v.size();
    return 0;
}

int qrt_model_logits(const char * gguf_path, const char * store_prefix, const char * spv_dir,
                     const uint32_t * ids, uint32_t n_tokens, uint32_t n_layers,
                     uint32_t n_rows, const char * out_path, char * err, size_t err_size) {
    if (gguf_path == NULL || spv_dir == NULL || ids == NULL || n_tokens == 0 || out_path == NULL) {
        return fail(err, err_size, "the model run needs a GGUF, SPIR-V, tokens and an output path");
    }
    if ((n_tokens % 32u) != 0) {
        return fail(err, err_size, "%u tokens is not a multiple of the 32 the FFN tiles", n_tokens);
    }
    if (n_rows == 0 || n_rows + 1u > n_tokens) {
        return fail(err, err_size, "%u dump rows need at least %u tokens", n_rows, n_rows + 1u);
    }
    char e[512] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open(0, &dev, e, sizeof(e)) != 0) {
        fprintf(stderr, "qrt: %s\n", e);
        return 2;
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, e, sizeof(e)) != 0) {
        fprintf(stderr, "qrt: %s\n", e);
        qrt_vk_close(dev);
        return 2;
    }
    const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(g);
    const uint32_t E = (uint32_t) hp->n_embd;
    const qrt_gguf_tensor * emb = qrt_gguf_find(g, "token_embd.weight");
    const qrt_gguf_tensor * outw = qrt_gguf_find(g, "output.weight");
    std::vector<float> onorm;
    int rc = 2;
    std::vector<float> x((size_t) n_tokens * E, 0.0f);
    qrt_vk_staging * stg = NULL;
    do {
        if (emb == NULL || outw == NULL || emb->type != 8 || outw->type != 8) {
            fprintf(stderr, "qrt: token_embd / output are not q8_0\n");
            break;
        }
        if (emb->ne[0] != E || outw->ne[0] != E || emb->ne[1] != outw->ne[1]) {
            fprintf(stderr, "qrt: embedding %llu x %llu, output %llu x %llu, n_embd %u\n",
                    (unsigned long long) emb->ne[0], (unsigned long long) emb->ne[1],
                    (unsigned long long) outw->ne[0], (unsigned long long) outw->ne[1], E);
            break;
        }
        const uint32_t V = (uint32_t) emb->ne[1];
        const uint32_t row_bytes = (E / 32u) * 34u;
        std::vector<uint8_t> row((size_t) row_bytes);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            if (ids[t] >= V) {
                fprintf(stderr, "qrt: token %u is %u, the vocabulary is %u\n", t, ids[t], V);
                break;
            }
            if (qrt_gguf_read_range(g, emb->offset + (uint64_t) ids[t] * row_bytes, row.data(),
                                    row_bytes, e, sizeof(e)) != 0) {
                fprintf(stderr, "qrt: %s\n", e);
                break;
            }
            q8_0_row(row.data(), E, x.data() + (size_t) t * E);
        }
        if (read_tensor_f32(g, "output_norm.weight", &onorm, e, sizeof(e)) != 0 ||
            onorm.size() != E) {
            fprintf(stderr, "qrt: output_norm: %s\n", e[0] ? e : "wrong width");
            break;
        }
        uint64_t stg_bytes = (uint64_t) n_tokens * E * 4;
        if (qrt_vk_staging_create(dev, stg_bytes, &stg, e, sizeof(e)) != 0) {
            fprintf(stderr, "qrt: %s\n", e);
            break;
        }
        uint32_t layer_count = n_layers;
        if (layer_count > (uint32_t) hp->block_count) {
            layer_count = (uint32_t) hp->block_count;
        }
        qrt_layer_cfg cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.ffn = store_prefix != NULL ? 1 : 0;
        cfg.single_submit = 1;
        cfg.dn_recur = 1;
        cfg.kv_k = QRT_KV_Q8_0;
        cfg.kv_v = QRT_KV_Q4_0;
        cfg.kv_page_tokens = 16384;
        for (uint32_t l = 0; l < layer_count; ++l) {
            qrt_layer * layer = NULL;
            if (qrt_layer_open(dev, gguf_path, store_prefix, l, n_tokens, spv_dir, &cfg, &layer,
                              e, sizeof(e)) != 0) {
                fprintf(stderr, "qrt: layer %u: %s\n", l, e);
                break;
            }
            int lrc = qrt_layer_set_input(layer, x.data(), e, sizeof(e));
            std::vector<float> xprev(2 * (size_t) E, 0.0f);
            memcpy(xprev.data(), x.data() + (size_t) (n_tokens - 2) * E, (size_t) E * 4);
            memcpy(xprev.data() + E, x.data() + (size_t) (n_tokens - 1) * E, (size_t) E * 4);
            if (lrc == 0) {
                lrc = qrt_layer_zero_state(layer, e, sizeof(e));
            }
            if (lrc == 0) {
                lrc = qrt_layer_run(layer, e, sizeof(e));
            }
            if (lrc == 0) {
                lrc = qrt_vk_download(dev, stg, qrt_layer_out(layer), 0, x.data(),
                                      (uint64_t) n_tokens * E * 4, e, sizeof(e));
            }
            qrt_layer_close(dev, layer);
            if (lrc != 0) {
                fprintf(stderr, "qrt: layer %u: %s\n", l, e);
                break;
            }
            printf("model_layer layer=%u x0=%.4f", l, x[(size_t) (n_tokens - 1) * E]);
            {
                // is the layer's contribution token-independent? delta for two adjacent tokens
                const float * a0 = x.data() + (size_t) (n_tokens - 2) * E;
                const float * a1 = x.data() + (size_t) (n_tokens - 1) * E;
                const float * b0 = xprev.data();
                const float * b1 = xprev.data() + E;
                double d0 = 0.0, d1 = 0.0, dn = 0.0, nn = 0.0;
                for (uint32_t i = 0; i < E; ++i) {
                    const double p = (double) a0[i] - b0[i], q = (double) a1[i] - b1[i];
                    d0 += p * p;
                    d1 += q * q;
                    dn += (p - q) * (p - q);
                    nn += p * p;
                }
                printf(" delta_rms=%.4f/%.4f delta_cos=%.4f",
                       sqrt(d0 / (double) E), sqrt(d1 / (double) E),
                       (d0 > 0.0 && d1 > 0.0) ? (d0 + d1 - dn) / (2.0 * sqrt(d0 * d1)) : 0.0);
            }
            printf("\n");
        }
        if (cfg.ffn == 0) {
            fprintf(stderr, "qrt: note: no --model-store, the expert half is off\n");
        }

        // final norm of the last row, then the head rows we dump
        const uint32_t start = n_tokens - 1u - n_rows;
        {
            const float * a = x.data() + (size_t) (n_tokens - 1) * E;
            const float * b = x.data() + (size_t) (n_tokens - 2) * E;
            double ab = 0.0, aa = 0.0, bb = 0.0, ra = 0.0;
            for (uint32_t i = 0; i < E; ++i) {
                ab += (double) a[i] * b[i];
                aa += (double) a[i] * a[i];
                bb += (double) b[i] * b[i];
            }
            ra = sqrt(aa / (double) E);
            printf("model_hidden cos_adj=%.4f rms=%.4f\n",
                   (aa > 0.0 && bb > 0.0) ? ab / sqrt(aa * bb) : 0.0, ra);
        }
        FILE * f = fopen(out_path, "wb");
        if (f == NULL) {
            fprintf(stderr, "qrt: cannot write '%s'\n", out_path);
            break;
        }
        const char magic[8] = { 'Q', 'R', 'T', 'L', 'O', 'G', '1', '\0' };
        const uint32_t version = 1;
        const uint32_t hdr[4] = { version, V, n_rows, start };
        fwrite(magic, 1, 8, f);
        fwrite(hdr, sizeof(uint32_t), 4, f);
        fwrite(ids + start, sizeof(uint32_t), n_rows + 1u, f);

        std::vector<double> xn(E, 0.0);
        std::vector<double> logits((size_t) V);
        std::vector<float> lerp((size_t) V);
        const double eps = 1e-6;
        for (uint32_t r = 0; r < n_rows; ++r) {
            const uint32_t j = start + r;
            const float * src = x.data() + (size_t) j * E;
            double ss = 0.0;
            for (uint32_t i = 0; i < E; ++i) {
                ss += (double) src[i] * (double) src[i];
            }
            const double inv = 1.0 / sqrt(ss / (double) E + eps);
            for (uint32_t i = 0; i < E; ++i) {
                xn[i] = (double) src[i] * inv * (double) onorm[i];
            }
            for (uint32_t v = 0; v < V; ++v) {
                if (qrt_gguf_read_range(g, outw->offset + (uint64_t) v * row_bytes, row.data(),
                                        row_bytes, e, sizeof(e)) != 0) {
                    fprintf(stderr, "qrt: %s\n", e);
                    break;
                }
                q8_0_row(row.data(), E, lerp.data());
                double acc = 0.0;
                for (uint32_t i = 0; i < E; ++i) {
                    acc += (double) lerp[i] * xn[i];
                }
                logits[v] = acc;
            }
            for (uint32_t v = 0; v < V; ++v) {
                const float lf = (float) logits[v];
                fwrite(&lf, sizeof(float), 1, f);
            }
            if (r + 1u == n_rows) {
                uint32_t top = 0;
                for (uint32_t v = 1; v < V; ++v) {
                    if (logits[v] > logits[top]) {
                        top = v;
                    }
                }
                printf("model_logits rows=%u start=%u top1=%u next=%u logit=%.4f\n", n_rows, start,
                       top, ids[j + 1], (float) logits[top]);
            }
        }
        fclose(f);
        printf("model_dump path=%s tokens=%u layers=%u vocab=%u rows=%u start=%u\n", out_path,
               n_tokens, layer_count, V, n_rows, start);
        rc = 0;
    } while (0);

    qrt_vk_staging_destroy(dev, stg);
    qrt_gguf_close(g);
    qrt_vk_close(dev);
    return rc;
}
