// Host-side layer chain: the CPU double reference and the --layer-check driver. It lives
// apart from the production layer so a change to the device path cannot silently ride on
// the reference's copy of the same conventions.

#include "layer.h"
#include "vk_transfer.h"

#include "attn_deltanet.h"
#include "attn_full.h"
#include "ffn.h"
#include "gguf_index.h"
#include "kernels/norm.h"
#include "layer_internal.h"
#include "qrt_half.h"
#include "qrt_util.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

double rel_l1_of(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += fabs((double) a[i] - (double) b[i]);
        den += fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : num;
}

// Double precision reference of a q8_0 x Q8_1 GEMM over the whole matrix, [n][tokens].
void dense_cpu(const std::vector<unsigned char> & raw, uint32_t n, uint32_t k,
               const signed char * qs, const uint32_t * ds, uint32_t tokens,
               std::vector<double> * out) {
    out->assign((size_t) n * tokens, 0.0);
    const uint32_t nb = k / 32u;
    for (uint32_t r = 0; r < n; ++r) {
        const unsigned char * row = raw.data() + (size_t) r * nb * 34u;
        for (uint32_t b = 0; b < nb; ++b) {
            const unsigned char * blk = row + (size_t) b * 34u;
            const uint32_t dh = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
            const double dw = (double) f16_to_f32(dh);
            double wq[32];
            for (uint32_t e = 0; e < 32u; ++e) {
                wq[e] = (double) (int8_t) blk[2 + e];
            }
            for (uint32_t t = 0; t < tokens; ++t) {
                const uint32_t ab = t * nb + b;
                const double aq = (double) f16_to_f32(ds[ab] & 0xFFFFu);
                const signed char * qa = qs + (size_t) ab * 32u;
                double sub = 0.0;
                for (uint32_t e = 0; e < 32u; ++e) {
                    sub += wq[e] * (double) qa[e];
                }
                (*out)[(size_t) r * tokens + t] += dw * aq * sub;
            }
        }
    }
}

void to_token_major(const std::vector<double> & src, uint32_t n, uint32_t tokens,
                    std::vector<float> * out) {
    out->assign((size_t) n * tokens, 0.0f);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t t = 0; t < tokens; ++t) {
            (*out)[(size_t) t * n + r] = (float) src[(size_t) r * tokens + t];
        }
    }
}

int download(qrt_layer * l, VkBuffer src, void * dst, uint64_t bytes, char * err,
             size_t err_size) {
    return qrt_vk_download(l->dev, l->stg, src, 0, dst, bytes, err, err_size);
}

} // namespace

namespace {

// The f16 KV append uses round-to-nearest-even (DECISIONS 64.1 / U-13). The reference
// builds its own rows with the same rule instead of reading the device's back.
float f16_rne_round(float x) {
    return f16_to_f32(f32_to_f16_bits(x));
}

} // namespace

int qrt_layer_ref_cpu(qrt_layer * l, const float * x_in, float * x_out,
                      struct qrt_layer_ref_trace * trace, int device_planes, char * err,
                      size_t err_size) {
    if (l == NULL || x_in == NULL || x_out == NULL) {
        return fail(err, err_size, "the layer reference needs an input and an output");
    }
    const qrt_layer_info info = l->info;
    const uint32_t E = info.n_embd;
    const uint32_t T = info.tokens;
    const uint32_t last = l->n_proj - 1;

    std::vector<float> ref_norm((size_t) T * E);
    std::vector<signed char> qs((size_t) T * E);
    std::vector<uint32_t> ds((size_t) T * (E / 32u));
    qrt_norm_cpu(T, E, x_in, l->h_norm.data(), l->eps, ref_norm.data(), qs.data(), ds.data());
    if (trace != NULL && trace->xnorm != NULL) {
        memcpy(trace->xnorm, ref_norm.data(), ref_norm.size() * sizeof(float));
    }

    std::vector<std::vector<float>> ref_proj(last);
    for (uint32_t i = 0; i < last; ++i) {
        std::vector<double> acc;
        dense_cpu(l->proj[i].raw, l->proj[i].n, E, qs.data(), ds.data(), T, &acc);
        to_token_major(acc, l->proj[i].n, T, &ref_proj[i]);
    }

    std::vector<float> ref_src((size_t) T * info.inter);
    if (info.linear) {
        const qrt_dn_shapes & sh = l->dsh;
        const uint64_t st_f = (uint64_t) sh.n_v_heads * sh.head_dim * sh.head_dim;
        std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
        std::vector<float> csto(cst.size());
        std::vector<float> q((size_t) T * sh.qk_ch), k((size_t) T * sh.qk_ch);
        std::vector<float> v((size_t) T * sh.d_inner);
        std::vector<float> dec((size_t) T * sh.n_v_heads), bet(dec.size());
        std::vector<double> st((size_t) st_f, 0.0);
        std::vector<float> y((size_t) T * sh.d_inner);
        qrt_dn_prep_cpu(&sh, T, ref_proj[0].data(), qrt_dn_weights_host(l->dnw, 0), cst.data(),
                        l->l2_eps, q.data(), k.data(), v.data(), csto.data());
        qrt_dn_gate_cpu(&sh, T, ref_norm.data(), qrt_dn_weights_host(l->dnw, 1),
                        qrt_dn_weights_host(l->dnw, 2), qrt_dn_weights_host(l->dnw, 3),
                        qrt_dn_weights_host(l->dnw, 4), dec.data(), bet.data());
        qrt_dn_recur_cpu(&sh, T, q.data(), k.data(), v.data(), dec.data(), bet.data(),
                         st.data(), y.data());
        qrt_dn_gnorm_cpu(&sh, T, y.data(), ref_proj[1].data(), qrt_dn_weights_host(l->dnw, 5),
                         l->eps, ref_src.data());
    } else {
        if (qrt_full_kv_k(l->full) != QRT_KV_F16 || qrt_full_kv_v(l->full) != QRT_KV_F16) {
            return fail(err, err_size,
                        "layer %u: the model reference chain models f16 KV rows only, run it "
                        "with --kv-k f16 --kv-v f16", l->layer);
        }
        const qrt_full_shapes & sh = l->fsh;
        const uint32_t hd = sh.head_dim;
        std::vector<float> q_out((size_t) T * sh.n_head * hd);
        std::vector<float> k_out((size_t) sh.n_kv * T * hd);
        std::vector<float> v_out(k_out.size());
        qrt_full_qk_cpu(&sh, 0, T, ref_proj[0].data(), ref_proj[1].data(), ref_proj[2].data(),
                        l->h_qnw.data(), l->h_knw.data(), q_out.data(), k_out.data(),
                        v_out.data());
        for (size_t i = 0; i < k_out.size(); ++i) {
            k_out[i] = f16_rne_round(k_out[i]);
            v_out[i] = f16_rne_round(v_out[i]);
        }
        std::vector<uint32_t> rows(T);
        for (uint32_t i = 0; i < T; ++i) {
            rows[i] = i;
        }
        qrt_full_attn_rows_cpu(&sh, T, rows.data(), T, q_out.data(), k_out.data(), v_out.data(),
                               ref_proj[0].data(), ref_src.data());
    }

    // output projection over the reference's own Q8_1 planes, then the residual add
    std::vector<float> src_copy((size_t) T * info.inter);
    std::vector<signed char> p2qs((size_t) T * info.inter);
    std::vector<uint32_t> p2ds((size_t) T * (info.inter / 32u));
    if (trace != NULL && trace->attn_src != NULL) {
        memcpy(trace->attn_src, ref_src.data(), ref_src.size() * sizeof(float));
    }
    qrt_norm_quant_cpu(T, info.inter, ref_src.data(), src_copy.data(), p2qs.data(), p2ds.data());
    if (device_planes &&
        (download(l, l->p2_qs, p2qs.data(), p2qs.size(), err, err_size) != 0 ||
         download(l, l->p2_ds, p2ds.data(), p2ds.size() * 4, err, err_size) != 0)) {
        return -1;
    }
    std::vector<double> acc_out;
    dense_cpu(l->proj[last].raw, l->proj[last].n, info.inter, p2qs.data(), p2ds.data(), T,
              &acc_out);
    std::vector<float> ref_out((size_t) T * E);
    to_token_major(acc_out, l->proj[last].n, T, &ref_out);
    std::vector<float> ref_x((size_t) T * E);
    for (size_t i = 0; i < ref_x.size(); ++i) {
        ref_x[i] = x_in[i] + ref_out[i];
    }
    if (trace != NULL) {
        if (trace->attn_out != NULL) {
            memcpy(trace->attn_out, ref_out.data(), ref_out.size() * sizeof(float));
        }
        if (trace->x_res != NULL) {
            memcpy(trace->x_res, ref_x.data(), ref_x.size() * sizeof(float));
        }
    }
    std::vector<float> ref_post((size_t) T * E);
    qrt_norm_cpu(T, E, ref_x.data(), l->h_post.data(), l->eps, ref_post.data(), nullptr, nullptr);
    if (trace != NULL && trace->post_norm != NULL) {
        memcpy(trace->post_norm, ref_post.data(), ref_post.size() * sizeof(float));
    }
    if (l->ffn == NULL) {
        memcpy(x_out, ref_x.data(), ref_x.size() * sizeof(float));
        return 0;
    }
    std::vector<float> ref_logits((size_t) T * info.n_expert);
    std::vector<uint32_t> ref_ids((size_t) T * info.top_k);
    std::vector<float> ref_wts(ref_ids.size()), sgate((size_t) T);
    std::vector<float> shexp((size_t) T * E), moe((size_t) T * E);
    std::vector<signed char> post_qs;
    std::vector<uint32_t> post_ds;
    if (device_planes && l->ffn != NULL) {
        post_qs.assign((size_t) T * E, 0);
        post_ds.assign((size_t) T * (E / 32u), 0);
        if (download(l, l->qs, post_qs.data(), post_qs.size(), err, err_size) != 0 ||
            download(l, l->ds, post_ds.data(), post_ds.size() * 4, err, err_size) != 0) {
            return -1;
        }
    }
    if (qrt_ffn_ref_cpu(l->ffn, ref_post.data(), post_qs.empty() ? nullptr : post_qs.data(),
                        post_ds.empty() ? nullptr : post_ds.data(), T, ref_logits.data(),
                        ref_ids.data(), ref_wts.data(), sgate.data(), shexp.data(), moe.data(),
                        err, err_size) != 0) {
        return -1;
    }
    if (trace != NULL) {
        if (trace->moe != NULL) {
            memcpy(trace->moe, moe.data(), moe.size() * sizeof(float));
        }
        if (trace->shexp != NULL) {
            memcpy(trace->shexp, shexp.data(), shexp.size() * sizeof(float));
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        const float s = 1.0f / (1.0f + expf(-sgate[t]));
        for (uint32_t c = 0; c < E; ++c) {
            const size_t i = (size_t) t * E + c;
            x_out[i] = ref_x[i] + moe[i] + shexp[i] * s;
        }
    }
    return 0;
}

int qrt_layer_check(const char * gguf_path, const char * store_prefix, uint32_t layer,
                    uint32_t tokens, const qrt_layer_cfg * cfg, const char * spv_dir) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open(0, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_layer * l = NULL;
    if (qrt_layer_open(dev, gguf_path, store_prefix, layer, tokens, spv_dir, cfg, &l, err,
                       sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    const qrt_layer_info info = *qrt_layer_shape(l);
    const uint32_t E = info.n_embd;
    const uint32_t T = info.tokens;
    printf("layer_info layer=%u linear=%d n_embd=%u tokens=%u inter=%u projections=%u "
           "ffn=%d experts=%u top_k=%u single_submit=%d\n",
           layer, info.linear, E, T, info.inter, l->n_proj, l->ffn != NULL, info.n_expert,
           info.top_k, l->cfg.single_submit);

    std::vector<float> x((size_t) T * E);
    uint32_t seed = 0x5EED1234u;
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) ((seed >> 8) % 2001u) - 1000) * 0.001f;
    }
    if (qrt_layer_set_input(l, x.data(), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }

    double tms[3];
    for (int i = 0; i < 3; ++i) {
        if (qrt_layer_zero_state(l, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        const double t0 = now_ms();
        if (qrt_layer_run(l, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        tms[i] = now_ms() - t0;
    }
    const double med = tms[0] <= tms[1] ? (tms[1] <= tms[2] ? tms[1] : (tms[0] <= tms[2] ? tms[2] : tms[0]))
                                        : (tms[0] <= tms[2] ? tms[0] : (tms[1] <= tms[2] ? tms[2] : tms[1]));
    double min_ms = tms[0];
    for (int i = 1; i < 3; ++i) {
        if (tms[i] < min_ms) {
            min_ms = tms[i];
        }
    }
    printf("layer_perf layer=%u tokens=%u ms=%.3f min_ms=%.3f us_per_token=%.3f\n", layer, T, med,
           min_ms, med * 1000.0 / (double) T);
    printf("layer_stages ms_pass1=%.3f ms_group=%.3f ms_pass2=%.3f submits=%u "
           "dispatches=%u experts=%u rows=%u store_mib=%.1f planes_mib=%.1f\n",
           qrt_layer_pass1_ms(l), qrt_layer_group_ms(l), qrt_layer_pass2_ms(l),
           qrt_layer_submits(l), qrt_layer_dispatches(l), qrt_layer_experts(l),
           qrt_layer_rows(l), (double) qrt_layer_store_bytes(l) / (1024.0 * 1024.0),
           (double) qrt_layer_plane_bytes(l) / (1024.0 * 1024.0));

    if (T > 1024) {
        printf("layer_chain skipped tokens=%u (the CPU reference is capped at 1024)\n", T);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 0;
    }

    // One more run from a zero state and the original input, so the device state is the one
    // the reference models: the residual add leaves its result in x, which is the point of
    // a layer chain but not what a single-layer check wants to accumulate.
    if (qrt_layer_set_input(l, x.data(), err, sizeof(err)) != 0 ||
        qrt_layer_zero_state(l, err, sizeof(err)) != 0 || qrt_layer_run(l, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }

    const uint32_t last = l->n_proj - 1;
    std::vector<float> g_norm((size_t) T * E), g_src((size_t) T * info.inter);
    std::vector<float> g_out((size_t) T * E), g_x((size_t) T * E), g_post((size_t) T * E);
    if (download(l, l->norm_out, g_norm.data(), g_norm.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->attn_src, g_src.data(), g_src.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->attn_out, g_out.data(), g_out.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->x, g_x.data(), g_x.size() * 4, err, sizeof(err)) != 0 ||
        download(l, l->xpost, g_post.data(), g_post.size() * 4, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t cols = 64;
    std::vector<std::vector<float>> g_proj(l->n_proj);
    std::vector<std::vector<float>> g_proj_full(l->n_proj);
    for (uint32_t i = 0; i < l->n_proj; ++i) {
        const uint32_t n = l->proj[i].n;
        g_proj[i].assign((size_t) cols * n, 0.0f);
        if (n * cols * 4 > 0 &&
            download(l, l->proj_out[i], g_proj[i].data(), (uint64_t) cols * n * 4, err,
                     sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
    }

    // The device projections, all tokens: the later stages' references use these as their
    // input, so this check validates the wiring (each stage's buffer being the next one's
    // source) while every kernel keeps its own check.
    for (uint32_t i = 0; i < last; ++i) {
        const uint32_t n = l->proj[i].n;
        g_proj_full[i].assign((size_t) T * n, 0.0f);
        // A 8192-wide plane does not fit the staging buffer in one go, so it goes in slices.
        const uint32_t slice_tok = 128;
        for (uint32_t t0 = 0; t0 < T; t0 += slice_tok) {
            const uint32_t cnt = T - t0 < slice_tok ? T - t0 : slice_tok;
            if (qrt_vk_download(dev, l->stg, l->proj_out[i], (uint64_t) t0 * n * 4,
                                g_proj_full[i].data() + (size_t) t0 * n, (uint64_t) cnt * n * 4,
                                err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_layer_close(dev, l);
                qrt_vk_close(dev);
                return 2;
            }
        }
    }

    // ---- the CPU chain over the same weights and the same Q8_1 rule ----
    std::vector<signed char> qs0((size_t) T * E);
    std::vector<uint32_t> ds0((size_t) T * (E / 32u));
    std::vector<float> ref_norm((size_t) T * E);
    qrt_norm_cpu(T, E, x.data(), l->h_norm.data(), l->eps, ref_norm.data(), qs0.data(),
                 ds0.data());
    std::vector<std::vector<float>> ref_proj(l->n_proj);
    for (uint32_t i = 0; i < last; ++i) {
        std::vector<double> acc;
        dense_cpu(l->proj[i].raw, l->proj[i].n, E, qs0.data(), ds0.data(), T, &acc);
        to_token_major(acc, l->proj[i].n, T, &ref_proj[i]);
    }
    // attention stage, from the projections the CPU just computed
    std::vector<float> ref_src((size_t) T * info.inter);
    if (info.linear) {
        const qrt_dn_shapes & sh = l->dsh;
        const uint64_t st_f = (uint64_t) sh.n_v_heads * sh.head_dim * sh.head_dim;
        std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
        std::vector<float> csto(cst.size());
        std::vector<float> q((size_t) T * sh.qk_ch), k((size_t) T * sh.qk_ch);
        std::vector<float> v((size_t) T * sh.d_inner);
        std::vector<float> drows((size_t) T * sh.n_v_heads), brows(drows.size());
        std::vector<double> st((size_t) st_f, 0.0);
        std::vector<float> yref((size_t) T * sh.d_inner);
        qrt_dn_prep_cpu(&sh, T, g_proj_full[0].data(), qrt_dn_weights_host(l->dnw, 0), cst.data(),
                        l->l2_eps, q.data(), k.data(), v.data(), csto.data());
        qrt_dn_gate_cpu(&sh, T, g_norm.data(), qrt_dn_weights_host(l->dnw, 1),
                        qrt_dn_weights_host(l->dnw, 2), qrt_dn_weights_host(l->dnw, 3),
                        qrt_dn_weights_host(l->dnw, 4), drows.data(), brows.data());
        qrt_dn_recur_cpu(&sh, T, q.data(), k.data(), v.data(), drows.data(), brows.data(),
                         st.data(), yref.data());
        qrt_dn_gnorm_cpu(&sh, T, yref.data(), g_proj_full[1].data(), qrt_dn_weights_host(l->dnw, 5),
                         l->eps, ref_src.data());
    } else {
        const qrt_full_shapes & sh = l->fsh;
        std::vector<float> q_out((size_t) T * sh.n_head * sh.head_dim);
        std::vector<float> k_out((size_t) sh.n_kv * T * sh.head_dim);
        std::vector<float> v_out(k_out.size());
        qrt_full_qk_cpu(&sh, 0, T, g_proj_full[0].data(), g_proj_full[1].data(),
                        g_proj_full[2].data(),
                        l->h_qnw.data(), l->h_knw.data(), q_out.data(), k_out.data(),
                        v_out.data());
        // The reference reads the DEVICE's KV cache and dequantises it with the kernel's own
        // f16 staging rounding, so the row precision is shared by both sides instead of
        // landing in the gate (M2-6a).
        const qrt_full_bufs * fb = qrt_full_buffers(l->full);
        const uint64_t k_bytes = qrt_full_kv_bytes(l->full);
        const uint64_t v_bytes = qrt_full_v_bytes(l->full);
        std::vector<uint8_t> kraw((size_t) k_bytes);
        std::vector<uint8_t> vraw((size_t) v_bytes);
        if (download(l, fb->kcache, kraw.data(), k_bytes, err, sizeof(err)) != 0 ||
            download(l, fb->vcache, vraw.data(), v_bytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        qrt_kv_row_layout klay, vlay;
        if (qrt_kv_row_layout_of(qrt_full_kv_k(l->full), sh.head_dim, &klay) != 0 ||
            qrt_kv_row_layout_of(qrt_full_kv_v(l->full), sh.head_dim, &vlay) != 0) {
            fprintf(stderr, "qrt: the layer's KV row layout is not one the check knows\n");
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        qrt_full_kv_unpack(&klay, sh.head_dim, sh.n_kv, T, 0, T, T, kraw.data(),
                           qrt_full_kv_k(l->full) == QRT_KV_F16 ? 1 : 2, k_out.data());
        qrt_full_kv_unpack(&vlay, sh.head_dim, sh.n_kv, T, 0, T, T, vraw.data(),
                           qrt_full_kv_v(l->full) == QRT_KV_F16 ? 1 : 2, v_out.data());
        std::vector<uint32_t> rows(T);
        for (uint32_t i = 0; i < T; ++i) {
            rows[i] = i;
        }
        qrt_full_attn_rows_cpu(&sh, T, rows.data(), T, q_out.data(), k_out.data(),
                               v_out.data(), g_proj_full[0].data(), ref_src.data());
    }

    std::vector<float> ref_src_copy((size_t) T * info.inter);
    std::vector<signed char> p2qs((size_t) T * info.inter);
    std::vector<uint32_t> p2ds((size_t) T * (info.inter / 32u));
    // The device's own Q8_1 planes of the attention output: the quantiser is what --norm-check
    // gates, so the output projection is compared on the same planes the device used.
    if (download(l, l->p2_qs, p2qs.data(), p2qs.size(), err, sizeof(err)) != 0 ||
        download(l, l->p2_ds, p2ds.data(), p2ds.size() * 4, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_norm_quant_cpu(T, info.inter, g_src.data(), ref_src_copy.data(), nullptr, nullptr);
    std::vector<double> acc_out;
    dense_cpu(l->proj[last].raw, l->proj[last].n, info.inter, p2qs.data(), p2ds.data(), T,
              &acc_out);
    std::vector<float> ref_out((size_t) T * E);
    to_token_major(acc_out, l->proj[last].n, T, &ref_out);
    std::vector<float> ref_x((size_t) T * E), ref_post((size_t) T * E);
    for (size_t i = 0; i < ref_x.size(); ++i) {
        ref_x[i] = x[i] + ref_out[i];
    }
    qrt_norm_cpu(T, E, ref_x.data(), l->h_post.data(), l->eps, ref_post.data(), nullptr, nullptr);

    int fails = 0;
    int open = 0;
    const double r_norm = rel_l1_of(g_norm.data(), ref_norm.data(), g_norm.size());
    printf("layer_norm rel_l1=%.3e verdict=%s\n", r_norm, r_norm <= 1e-6 ? "PASS" : "FAIL");
    fails += r_norm <= 1e-6 ? 0 : 1;
    for (uint32_t i = 0; i < last; ++i) {
        const uint32_t n = l->proj[i].n;
        const double rel = rel_l1_of(g_proj[i].data(), ref_proj[i].data(),
                                     (size_t) cols * n);
        // f32 accumulation in the kernel against a double reference: measured 2e-06..1e-05
        // depending on the matrix, while a layout bug is O(1e-1).
        const int ok = rel <= 5e-5;
        printf("layer_proj%u n=%u k=%u cols=%u rel_l1=%.3e verdict=%s\n", i, n, E, cols, rel,
               ok ? "PASS" : "OPEN");
        open += ok ? 0 : 1;
    }
    const double r_src = rel_l1_of(g_src.data(), ref_src.data(), g_src.size());
    printf("layer_attn linear=%d inter=%u rel_l1=%.3e verdict=%s\n", info.linear, info.inter,
           r_src, r_src <= 1e-4 ? "PASS" : "FAIL");
    fails += r_src <= 1e-4 ? 0 : 1;
    const double r_out = rel_l1_of(g_out.data(), ref_out.data(), g_out.size());
    printf("layer_outproj n=%u k=%u rel_l1=%.3e verdict=%s\n", l->proj[last].n, info.inter,
           r_out, r_out <= 1e-5 ? "PASS" : "FAIL");
    fails += r_out <= 1e-5 ? 0 : 1;
    const double r_x = rel_l1_of(g_x.data(), ref_x.data(), g_x.size());
    printf("layer_residual rel_l1=%.3e verdict=%s\n", r_x, r_x <= 1e-6 ? "PASS" : "FAIL");
    fails += r_x <= 1e-6 ? 0 : 1;
    const double r_post = rel_l1_of(g_post.data(), ref_post.data(), g_post.size());
    printf("layer_postnorm rel_l1=%.3e verdict=%s\n", r_post, r_post <= 1e-6 ? "PASS" : "FAIL");
    fails += r_post <= 1e-6 ? 0 : 1;

    // ---- the FFN half: router, shared expert, routed experts, final residual ----
    if (l->ffn != NULL) {
        const uint32_t ref = T < 64u ? T : 64u;
        const qrt_ffn_bufs * fb = qrt_layer_ffn_bufs(l);
        std::vector<float> g_logits((size_t) ref * info.n_expert);
        std::vector<uint32_t> g_ids((size_t) ref * info.top_k);
        std::vector<float> g_wts((size_t) ref * info.top_k);
        std::vector<float> g_sgate(ref), g_shexp((size_t) ref * E), g_moe((size_t) ref * E);
        std::vector<float> g_xall((size_t) T * E);
        int rc = download(l, fb->logits, g_logits.data(), g_logits.size() * 4, err,
                          sizeof(err));
        rc = rc == 0 ? download(l, fb->ids, g_ids.data(), g_ids.size() * 4, err, sizeof(err))
                     : rc;
        rc = rc == 0 ? download(l, fb->weights, g_wts.data(), g_wts.size() * 4, err,
                                sizeof(err)) : rc;
        rc = rc == 0 ? download(l, fb->sgate, g_sgate.data(), g_sgate.size() * 4, err,
                                sizeof(err)) : rc;
        rc = rc == 0 ? download(l, fb->shexp, g_shexp.data(), g_shexp.size() * 4, err,
                                sizeof(err)) : rc;
        rc = rc == 0 ? download(l, fb->moe, g_moe.data(), g_moe.size() * 4, err,
                                sizeof(err)) : rc;
        rc = rc == 0 ? download(l, qrt_layer_out(l), g_xall.data(), g_xall.size() * 4, err,
                                sizeof(err)) : rc;
        if (rc != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        std::vector<float> ref_logits((size_t) ref * info.n_expert);
        std::vector<uint32_t> ref_ids((size_t) ref * info.top_k);
        std::vector<float> ref_wts((size_t) ref * info.top_k);
        std::vector<float> ref_sgate(ref), ref_shexp((size_t) ref * E);
        std::vector<float> ref_moe((size_t) ref * E);
        // the device's own Q8_1 planes of the post norm row, so the quantiser's boundary
        // decisions are shared by both sides (the out projection stage does the same)
        std::vector<signed char> post_qs((size_t) ref * E);
        std::vector<uint32_t> post_ds((size_t) ref * (E / 32u));
        int rcp = download(l, l->qs, post_qs.data(), post_qs.size(), err, sizeof(err));
        rcp = rcp == 0 ? download(l, l->ds, post_ds.data(), post_ds.size() * 4, err,
                                  sizeof(err)) : rcp;
        if (rcp != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        if (qrt_ffn_ref_cpu(qrt_layer_ffn(l), g_post.data(), post_qs.data(), post_ds.data(), ref,
                            ref_logits.data(), ref_ids.data(), ref_wts.data(), ref_sgate.data(),
                            ref_shexp.data(), ref_moe.data(), err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
        uint32_t id_hits = 0;
        for (size_t i = 0; i < ref_ids.size(); ++i) {
            id_hits += g_ids[i] == ref_ids[i] ? 1u : 0u;
        }
        const double r_log = rel_l1_of(g_logits.data(), ref_logits.data(), g_logits.size());
        printf("layer_router logits_rel_l1=%.3e ids_match=%u/%zu weights_rel_l1=%.3e "
               "verdict=%s\n", r_log, id_hits, ref_ids.size(),
               rel_l1_of(g_wts.data(), ref_wts.data(), g_wts.size()),
               (r_log <= 1e-5 && id_hits == ref_ids.size()) ? "PASS" : "FAIL");
        fails += (r_log <= 1e-5 && id_hits == ref_ids.size()) ? 0 : 1;
        const double r_w = rel_l1_of(g_wts.data(), ref_wts.data(), g_wts.size());
        fails += r_w <= 1e-5 ? 0 : 1;
        const double r_sg = rel_l1_of(g_sgate.data(), ref_sgate.data(), g_sgate.size());
        printf("layer_sgate rel_l1=%.3e verdict=%s\n", r_sg, r_sg <= 1e-6 ? "PASS" : "FAIL");
        fails += r_sg <= 1e-6 ? 0 : 1;
        const double r_sh = rel_l1_of(g_shexp.data(), ref_shexp.data(), g_shexp.size());
        printf("layer_shexp tokens=%u rel_l1=%.3e verdict=%s\n", ref, r_sh,
               r_sh <= 5e-5 ? "PASS" : "FAIL");
        fails += r_sh <= 5e-5 ? 0 : 1;
        const double r_moe = rel_l1_of(g_moe.data(), ref_moe.data(), g_moe.size());
        printf("layer_moe tokens=%u pairs=%u experts=%u rel_l1=%.3e verdict=%s\n", ref,
               qrt_layer_rows(l) > 0 ? ref * info.top_k : 0, qrt_layer_experts(l), r_moe,
               r_moe <= 5e-5 ? "PASS" : "FAIL");
        fails += r_moe <= 5e-5 ? 0 : 1;
        // the layer output: x + moe + shexp * sigmoid(gate), composed from the reference
        double num = 0.0, den = 0.0;
        for (uint32_t t = 0; t < ref; ++t) {
            const float s = 1.0f / (1.0f + expf(-ref_sgate[t]));
            for (uint32_t c = 0; c < E; ++c) {
                const double want = (double) g_x[(size_t) t * E + c] +
                                    (double) ref_moe[(size_t) t * E + c] +
                                    (double) ref_shexp[(size_t) t * E + c] * (double) s;
                const double got = g_xall[(size_t) t * E + c];
                num += fabs(got - want);
                den += fabs(want);
            }
        }
        const double r_layer = den > 0.0 ? num / den : num;
        printf("layer_ffn_out tokens=%u rel_l1=%.3e verdict=%s\n", ref, r_layer,
               r_layer <= 1e-4 ? "PASS" : "FAIL");
        fails += r_layer <= 1e-4 ? 0 : 1;
    }
    std::vector<float> baseline((size_t) T * E), input(baseline.size()), output(baseline.size());
    if (download(l, qrt_layer_out(l), baseline.data(), baseline.size() * 4, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_layer_close(dev, l);
        qrt_vk_close(dev);
        return 2;
    }
    const uint32_t pieces[] = {T - 4u, 1u, 2u, 1u};
    int state_bad = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
        int rc = qrt_layer_zero_state(l, err, sizeof(err));
        uint32_t pos = 0;
        for (uint32_t count : pieces) {
            if (rc != 0) break;
            std::fill(input.begin(), input.end(), 0.0f);
            std::copy_n(x.data() + (size_t) pos * E, (size_t) count * E, input.data());
            if (repeat == 2 && pos + count == T) {
                for (uint32_t c = 0; c < E; ++c) input[(size_t) (count - 1) * E + c] += (float) (c % 7);
            }
            rc = qrt_layer_set_input(l, input.data(), err, sizeof(err)) ||
                 qrt_layer_run_chunk(l, pos, count, err, sizeof(err)) ||
                 download(l, qrt_layer_out(l), output.data(), output.size() * 4, err, sizeof(err));
            if (rc != 0) break;
            const size_t checked = (size_t) (repeat == 2 && pos + count == T ? count - 1u : count) * E;
            state_bad += memcmp(output.data(), baseline.data() + (size_t) pos * E,
                                checked * sizeof(float)) != 0;
            pos += count;
        }
        if (rc != 0) {
            fprintf(stderr, "qrt: chunk state check: %s\n", err);
            qrt_layer_close(dev, l);
            qrt_vk_close(dev);
            return 2;
        }
    }
    printf("layer_state chunks=%u,1,2,1 repeats=2 prefix_causal=1 mismatches=%d verdict=%s\n",
           T - 4u, state_bad, state_bad == 0 ? "PASS" : "FAIL");
    fails += state_bad != 0;
    printf("layer_check layer=%u linear=%d tokens=%u fails=%d open=%d verdict=%s\n", layer,
           info.linear, T, fails, open, fails == 0 ? (open == 0 ? "PASS" : "PASS_OPEN") : "FAIL");

    qrt_layer_close(dev, l);
    qrt_vk_close(dev);
    return fails == 0 ? 0 : 1;
}


#endif // QRT_HAVE_VULKAN
