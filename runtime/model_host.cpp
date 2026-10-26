// Host-side weight math of the model run: the q8_0 row decode, the embedding gather and the
// output head over the whole vocabulary. Both arms read the same GGUF rows, so this lives
// apart from the device chain.

#include "model_internal.h"

#include "qrt_half.h"
#include "qrt_util.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

namespace {

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

} // namespace

// token_embd is q8_0 and stays on the host (model.h): one row read per token.
int qrt_model_embed_rows(qrt_gguf * g, const qrt_gguf_tensor * emb, uint32_t E, uint32_t V,
                         const uint32_t * ids, uint32_t n_tokens, float * x, char * err,
                         size_t err_size) {
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

namespace {

// rms norm with the weight applied, in double, for either input precision. eps matches what
// the head has always used so the two arms stay comparable.
template <typename T>
void norm_row_typed(const T * src, const float * norm_w, uint32_t E, double * dst) {
    double ss = 0.0;
    for (uint32_t i = 0; i < E; ++i) {
        ss += (double) src[i] * (double) src[i];
    }
    const double inv = 1.0 / sqrt(ss / (double) E + 1e-6);
    for (uint32_t i = 0; i < E; ++i) {
        dst[i] = (double) src[i] * inv * (double) norm_w[i];
    }
}

} // namespace

// The row math the MTP draft needs, split so one normalisation feeds both the candidate
// head and the next draft step's h input. The fork keeps t_h_nextn as the post-head-norm
// hidden and applies nextn.hnorm to it again, so the double form exists for that second
// norm: rounding h back to f32 between the two would add an error the graph does not have.
void qrt_model_norm_row(const float * src, const float * norm_w, uint32_t E, double * dst) {
    norm_row_typed(src, norm_w, E, dst);
}

void qrt_model_norm_row_d(const double * src, const float * norm_w, uint32_t E, double * dst) {
    norm_row_typed(src, norm_w, E, dst);
}

int qrt_model_gemv_q8_0(qrt_gguf * g, const qrt_gguf_tensor * w, uint32_t n_in,
                        const uint32_t * idx, uint32_t n_out, const double * x, float * y,
                        char * err, size_t err_size) {
    if (g == NULL || w == NULL || x == NULL || y == NULL || n_out == 0 || (n_in % 32u) != 0u) {
        return fail(err, err_size, "the q8_0 gemv needs buffers and a 32-multiple width");
    }
    const uint64_t rows = w->n_dims >= 2 ? w->ne[1] : 1u;
    const uint32_t row_bytes = (n_in / 32u) * 34u;
    if ((uint64_t) rows * row_bytes > w->nbytes) {
        return fail(err, err_size, "%s holds %llu bytes, %llu rows of %u do not fit", w->name,
                    (unsigned long long) w->nbytes, (unsigned long long) rows, row_bytes);
    }
    std::vector<uint8_t> raw((size_t) row_bytes);
    std::vector<float> row_f((size_t) n_in);
    for (uint32_t r = 0; r < n_out; ++r) {
        const uint32_t v = idx != NULL ? idx[r] : r;
        if ((uint64_t) v >= rows) {
            return fail(err, err_size, "row %u (id %u) is outside %s, which has %llu rows", r,
                        v, w->name, (unsigned long long) rows);
        }
        if (qrt_gguf_read_range(g, w->offset + (uint64_t) v * row_bytes, raw.data(), row_bytes,
                                err, err_size) != 0) {
            return -1;
        }
        q8_0_row(raw.data(), n_in, row_f.data());
        double acc = 0.0;
        for (uint32_t i = 0; i < n_in; ++i) {
            acc += (double) row_f[i] * x[i];
        }
        y[r] = (float) acc;
    }
    return 0;
}

// Final rms norm of the dumped rows, then output.weight over the whole vocabulary, in one
// pass. Both arms go through the same vocab row, so output.weight is read once per run:
// at 512 rows the per-row version would re-read 1.7 GB for every row. `b` and `la` may be
// NULL when that arm is not wanted.
int qrt_model_head_rows(qrt_gguf * g, const qrt_gguf_tensor * outw, const float * onorm,
                        const float * xa, const float * xb, uint32_t E, uint32_t V,
                        uint32_t n_tokens, uint32_t n_rows, uint32_t start,
                        std::vector<float> * la, std::vector<float> * lb,
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
