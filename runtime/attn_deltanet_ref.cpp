// Host-side reference of the DeltaNet legs (conv1d + silu, L2 norm, the gated recurrence and
// the gated RMS norm). Kept apart from the kernel launches in attn_deltanet.cpp so the two
// cannot share a mistake through file-local helpers.

#include "attn_deltanet.h"

#include "qrt_util.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <vector>




void qrt_dn_gnorm_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * y,
                      const float * z, const float * norm_w, float eps, float * out) {
    const uint32_t hd = sh->head_dim;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t h = 0; h < sh->n_v_heads; ++h) {
            const float * yrow = y + (size_t) t * sh->d_inner + (size_t) h * hd;
            float * orow = out + (size_t) t * sh->d_inner + (size_t) h * hd;
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; ++i) {
                ss += (double) yrow[i] * (double) yrow[i];
            }
            const double scale = 1.0 / sqrt(ss / (double) hd + (double) eps);
            for (uint32_t i = 0; i < hd; ++i) {
                const double g = (double) z[(size_t) t * sh->n_v_heads * hd + (size_t) h * hd + i];
                orow[i] = (float) ((double) yrow[i] * scale * (double) norm_w[i] *
                                   (g / (1.0 + exp(-g))));
            }
        }
    }
}

void qrt_dn_gate_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * xnorm,
                     const float * alpha_w, const float * beta_w, const float * dt,
                     const float * a, float * decay, float * beta) {
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t h = 0; h < sh->n_v_heads; ++h) {
            double alpha = 0.0;
            double braw = 0.0;
            for (uint32_t c = 0; c < sh->n_embd; ++c) {
                const double xv = (double) xnorm[(size_t) t * sh->n_embd + c];
                alpha += xv * (double) alpha_w[(size_t) c + (size_t) h * sh->n_embd];
                braw  += xv * (double) beta_w[(size_t) c + (size_t) h * sh->n_embd];
            }
            const double sp = log1p(exp(alpha + (double) dt[h]));
            decay[(size_t) t * sh->n_v_heads + h] = (float) exp((double) a[h] * sp);
            beta[(size_t) t * sh->n_v_heads + h] = (float) (1.0 / (1.0 + exp(-braw)));
        }
    }
}

void qrt_dn_prep_cpu(const qrt_dn_shapes * sh, uint32_t tokens, const float * qkv,
                     const float * conv_w, const float * conv_state_in, float l2_eps,
                     float * q, float * k, float * v, float * conv_state_out) {
    const uint32_t taps = sh->conv_taps;
    const uint32_t ch = sh->conv_ch;
    const uint32_t qk = sh->qk_ch;
    const uint32_t hd = sh->head_dim;
    const uint32_t heads = qk / hd;
    const double eps_term = (double) l2_eps * (double) hd;
    std::vector<double> conv((size_t) ch);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < ch; ++c) {
            double acc = 0.0;
            for (uint32_t j = 0; j < taps; ++j) {
                const int src = (int) t - (int) taps + 1 + (int) j;
                const double val = src < 0
                    ? (double) conv_state_in[(size_t) (taps - 1 + (uint32_t) src) * ch + c]
                    : (double) qkv[(size_t) src * ch + c];
                acc += (double) conv_w[(size_t) c * taps + j] * val;
            }
            conv[c] = acc / (1.0 + exp(-acc));      // silu
        }
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = half * qk + h * hd;
                double ss = 0.0;
                for (uint32_t i = 0; i < hd; ++i) {
                    const double val = conv[(size_t) off + i];
                    ss += val * val;
                }
                const double inv = 1.0 / sqrt(ss + eps_term);
                float * dst = half == 0 ? q : k;
                for (uint32_t i = 0; i < hd; ++i) {
                    dst[(size_t) t * qk + (size_t) h * hd + i] = (float) (conv[(size_t) off + i] * inv);
                }
            }
        }
        for (uint32_t c = 0; c < sh->d_inner; ++c) {
            v[(size_t) t * sh->d_inner + c] = (float) conv[(size_t) 2 * qk + c];
        }
    }
    (void) conv_state_out;
}

void qrt_dn_recur_cpu(const qrt_dn_shapes * sh, uint32_t tokens,
                      const float * q, const float * k, const float * v,
                      const float * decay, const float * beta,
                      double * state, float * y) {
    const uint32_t sd = sh->head_dim;
    const uint32_t ratio = sh->n_v_heads / sh->n_k_heads;
    const double qscale = 1.0 / sqrt((double) sd);
    std::vector<double> kcol((size_t) sd), qcol((size_t) sd), acc((size_t) sd);
    for (uint32_t h = 0; h < sh->n_v_heads; ++h) {
        double * s = state + (size_t) h * sd * sd;
        const uint32_t kh = h % (sh->n_v_heads / ratio);
        for (uint32_t t = 0; t < tokens; ++t) {
            const double d = (double) decay[(size_t) t * sh->n_v_heads + h];
            const double b = (double) beta[(size_t) t * sh->n_v_heads + h];
            const float * qrow = q + ((size_t) t * sh->qk_ch + (size_t) kh * sd);
            const float * krow = k + ((size_t) t * sh->qk_ch + (size_t) kh * sd);
            const float * vrow = v + ((size_t) t * sh->d_inner + (size_t) h * sd);
            for (uint32_t i = 0; i < sd; ++i) {
                kcol[i] = (double) krow[i];
                qcol[i] = (double) qrow[i];
            }
            // s *= d; sk[j] = sum_i s[i][j] k[i]
            // the state is stored column major, the same layout the kernel keeps in
            // registers, so a saved state can be handed to either side unchanged
            for (uint32_t j = 0; j < sd; ++j) {
                double sum = 0.0;
                for (uint32_t i = 0; i < sd; ++i) {
                    double & e = s[(size_t) j * sd + i];
                    e *= d;
                    sum += e * kcol[i];
                }
                acc[j] = ((double) vrow[j] - sum) * b;
            }
            float * yrow = y + ((size_t) t * sh->d_inner + (size_t) h * sd);
            for (uint32_t j = 0; j < sd; ++j) {
                const double delta = acc[j];
                double sum = 0.0;
                for (uint32_t i = 0; i < sd; ++i) {
                    double & e = s[(size_t) j * sd + i];
                    e += kcol[i] * delta;
                    sum += e * qcol[i];
                }
                yrow[j] = (float) (sum * qscale);
            }
        }
    }
}
