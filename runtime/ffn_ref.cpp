// Host-side FFN reference: the routed expert chain against the CPU double implementation.
// It lives apart from the production FFN so the reference cannot share the device TU's
// helpers or compile-time decisions by accident.

#include "ffn.h"

#include "ffn_internal.h"
#include "kernels/moe_expert.h"
#include "qrt_half.h"
#include "qrt_util.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

// ---------------------------------------------------------------- reference

namespace {

// Q8_0 x Q8_1 row in double over the canonical 34-byte blocks.
void q8_0_ref_row(const unsigned char * raw_row, uint32_t k, const signed char * a_qs,
                  const uint32_t * a_ds, uint32_t tokens, std::vector<double> * out) {
    out->assign(tokens, 0.0);
    const uint32_t nb = k / 32u;
    for (uint32_t t = 0; t < tokens; ++t) {
        double acc = 0.0;
        for (uint32_t b = 0; b < nb; ++b) {
            const unsigned char * blk = raw_row + (size_t) b * 34u;
            const double dw = (double) f16_to_f32((uint32_t) blk[0] |
                                                      ((uint32_t) blk[1] << 8));
            const double aq = (double) f16_to_f32(a_ds[(size_t) t * nb + b] & 0xFFFFu);
            double sub = 0.0;
            for (uint32_t e = 0; e < 32u; ++e) {
                sub += (double) (int8_t) blk[2 + e] *
                       (double) a_qs[((size_t) t * nb + b) * 32u + e];
            }
            acc += dw * aq * sub;
        }
        (*out)[t] = acc;
    }
}

float silu(float x) { return x / (1.0f + expf(-x)); }

} // namespace

int qrt_ffn_ref_cpu(qrt_ffn * f, const float * xpost, const signed char * xpost_qs,
                    const uint32_t * xpost_ds, uint32_t ref_tokens, float * logits,
                    uint32_t * ids, float * weights, float * sgate, float * shexp,
                    float * moe, char * err, size_t err_size) {
    if (f == NULL || xpost == NULL || ref_tokens == 0 || ref_tokens > f->cfg.tokens) {
        return fail(err, err_size, "the FFN reference needs a chunk and its post norm row");
    }
    const uint32_t E  = f->cfg.n_embd;
    const uint32_t FF = f->cfg.ffn;
    const uint32_t NE = f->cfg.n_expert;
    const uint32_t K  = f->cfg.top_k;
    const uint32_t T  = ref_tokens;

    // the same Q8_1 rule the device applies to the post norm row, on the device's own planes
    // when the caller passes them
    std::vector<signed char> qs((size_t) T * E);
    std::vector<uint32_t> ds((size_t) T * (E / 32u));
    if (xpost_qs != NULL && xpost_ds != NULL) {
        memcpy(qs.data(), xpost_qs, qs.size());
        memcpy(ds.data(), xpost_ds, ds.size() * sizeof(uint32_t));
    } else {
        qrt_norm_quant_cpu(T, E, xpost, nullptr, qs.data(), ds.data());
    }

    std::vector<float> probs((size_t) T * NE);
    std::vector<float> picked((size_t) K);
    for (uint32_t t = 0; t < T; ++t) {
        const float * row = xpost + (size_t) t * E;
        double mx = -1e300;
        for (uint32_t e = 0; e < NE; ++e) {
            const float * w = f->h_router.data() + (size_t) e * E;
            double acc = 0.0;
            for (uint32_t i = 0; i < E; ++i) {
                acc += (double) w[i] * (double) row[i];
            }
            probs[(size_t) t * NE + e] = (float) acc;
            if (acc > mx) {
                mx = acc;
            }
        }
        double sum = 0.0;
        for (uint32_t e = 0; e < NE; ++e) {
            sum += exp((double) probs[(size_t) t * NE + e] - mx);
        }
        // selection: strict greater on the softmax value, so ties keep the lower id
        std::vector<unsigned char> taken(NE, 0);
        float norm = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            float best = -3.0e38f;
            uint32_t bi = 0;
            for (uint32_t e = 0; e < NE; ++e) {
                const float p = (float) (exp((double) probs[(size_t) t * NE + e] - mx) / sum);
                if (taken[e] == 0 && p > best) {
                    best = p;
                    bi = e;
                }
            }
            taken[bi] = 1;
            picked[k] = best;
            if (ids != NULL) {
                ids[(size_t) t * K + k] = bi;
            }
            norm += best;
        }
        const float denom = norm > 6.103515625e-5f ? norm : 6.103515625e-5f;
        if (weights != NULL) {
            for (uint32_t k = 0; k < K; ++k) {
                weights[(size_t) t * K + k] = picked[k] / denom * f->cfg.scale;
            }
        }
        if (logits != NULL) {
            memcpy(logits + (size_t) t * NE, probs.data() + (size_t) t * NE,
                   (size_t) NE * sizeof(float));
        }
    }

    // shared expert: q8_0 gate/up, silu*up, the Q8_1 rule again, q8_0 down
    std::vector<float> gh((size_t) T * FF), uh((size_t) T * FF);
    std::vector<double> row;
    for (uint32_t r = 0; r < FF; ++r) {
        q8_0_ref_row(f->h_gate.data() + (size_t) r * (E / 32u) * 34u, E, qs.data(),
                     ds.data(), T, &row);
        for (uint32_t t = 0; t < T; ++t) {
            gh[(size_t) t * FF + r] = (float) row[t];
        }
        q8_0_ref_row(f->h_up.data() + (size_t) r * (E / 32u) * 34u, E, qs.data(), ds.data(),
                     T, &row);
        for (uint32_t t = 0; t < T; ++t) {
            uh[(size_t) t * FF + r] = (float) row[t];
        }
    }
    std::vector<float> h((size_t) T * FF);
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = silu(gh[i]) * uh[i];
    }
    std::vector<signed char> hqs((size_t) T * FF);
    std::vector<uint32_t> hds((size_t) T * (FF / 32u));
    qrt_norm_quant_cpu(T, FF, h.data(), nullptr, hqs.data(), hds.data());
    std::vector<float> gate_out((size_t) T);
    for (uint32_t t = 0; t < T; ++t) {
        double acc = 0.0;
        for (uint32_t i = 0; i < E; ++i) {
            acc += (double) f->h_sgate_w[i] * (double) xpost[(size_t) t * E + i];
        }
        gate_out[t] = (float) acc;
    }
    if (sgate != NULL) {
        memcpy(sgate, gate_out.data(), (size_t) T * sizeof(float));
    }
    if (shexp != NULL) {
        for (uint32_t r = 0; r < E; ++r) {
            q8_0_ref_row(f->h_down.data() + (size_t) r * (FF / 32u) * 34u, FF, hqs.data(),
                         hds.data(), T, &row);
            for (uint32_t t = 0; t < T; ++t) {
                shexp[(size_t) t * E + r] = (float) row[t];
            }
        }
    }

    // routed experts over the pairs of the first ref_tokens tokens
    if (moe != NULL) {
        memset(moe, 0, (size_t) T * E * sizeof(float));
        std::vector<std::vector<uint32_t>> tok_of(NE), slot_of(NE);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = ids[(size_t) t * K + k];
                tok_of[e].push_back(t);
                slot_of[e].push_back(k);
            }
        }
        unsigned char * blk = (unsigned char *) malloc((size_t) qrt_weights_max_expert_bytes(f->store));
        std::vector<signed char> rqs((size_t) T * E);
        std::vector<uint32_t> rds((size_t) T * (E / 32u));
        std::vector<float> rh((size_t) T * FF);
        std::vector<float> rgh((size_t) T * FF), ruh((size_t) T * FF);
        std::vector<signed char> rhqs((size_t) T * FF);
        std::vector<uint32_t> rhds((size_t) T * (FF / 32u));
        std::vector<float> sums((size_t) T * E);
        int rc = blk == NULL ? -1 : 0;
        for (uint32_t e = 0; e < NE && rc == 0; ++e) {
            const uint32_t n = (uint32_t) tok_of[e].size();
            if (n == 0) {
                continue;
            }
            qrt_expert_loc loc;
            if (qrt_weights_locate(f->store, f->layer, e, &loc, err, err_size) != 0 ||
                qrt_weights_read_expert(f->store, f->layer, e, blk, (size_t) loc.bytes, err,
                                        err_size) != 0) {
                rc = -1;
                break;
            }
            const int t_gate = qrt_gemm_type_of(FF, E, loc.gate);
            const int t_up   = qrt_gemm_type_of(FF, E, loc.up);
            const int t_down = qrt_gemm_type_of(E, FF, loc.down);
            if (t_gate < 0 || t_up < 0 || t_down < 0) {
                rc = fail(err, err_size, "reference: expert %u has an unknown type", e);
                break;
            }
            for (uint32_t r = 0; r < n; ++r) {
                memcpy(rqs.data() + (size_t) r * E, qs.data() + (size_t) tok_of[e][r] * E,
                       E);
                memcpy(rds.data() + (size_t) r * (E / 32u),
                       ds.data() + (size_t) tok_of[e][r] * (E / 32u), (E / 32u) * 4);
            }
            const unsigned char * wg = blk;
            const unsigned char * wu = blk + loc.gate;
            const unsigned char * wd = blk + loc.gate + loc.up;
            const uint32_t nb_g = E / 256u;
            const uint32_t nb_d = FF / 256u;
            for (uint32_t r = 0; r < FF; ++r) {
                qrt_moe_ref_row(wg + (size_t) r * nb_g * qrt_gemm_block_bytes((qrt_quant) t_gate),
                                (qrt_quant) t_gate, (const unsigned char *) rqs.data(),
                                rds.data(), E, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    rgh[(size_t) j * FF + r] = (float) row[j];
                }
                qrt_moe_ref_row(wu + (size_t) r * nb_g * qrt_gemm_block_bytes((qrt_quant) t_up),
                                (qrt_quant) t_up, (const unsigned char *) rqs.data(),
                                rds.data(), E, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    ruh[(size_t) j * FF + r] = (float) row[j];
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t r = 0; r < FF; ++r) {
                    rh[(size_t) j * FF + r] =
                        silu(rgh[(size_t) j * FF + r]) * ruh[(size_t) j * FF + r];
                }
            }
            qrt_norm_quant_cpu(n, FF, rh.data(), nullptr, rhqs.data(), rhds.data());
            for (uint32_t r = 0; r < E; ++r) {
                qrt_moe_ref_row(wd + (size_t) r * nb_d * qrt_gemm_block_bytes((qrt_quant) t_down),
                                (qrt_quant) t_down, (const unsigned char *) rhqs.data(),
                                rhds.data(), FF, n, row.data());
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t t = tok_of[e][j];
                    const uint32_t k = slot_of[e][j];
                    moe[(size_t) t * E + r] +=
                        weights[(size_t) t * K + k] * (float) row[j];
                }
            }
        }
        free(blk);
        if (rc != 0) {
            return -1;
        }
    }
    return 0;
}


#endif // QRT_HAVE_VULKAN
