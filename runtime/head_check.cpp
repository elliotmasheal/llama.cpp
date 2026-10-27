// The device output head's own gate (M3-7): the head runs over deterministic hidden rows,
// and an independent CPU double chain re-does the dot product from the raw output.weight rows
// and the Q8_1 planes the device packed. Sharing the packing decision is the point: the gate
// measures the head's arithmetic, not the rounding of the activation quantiser, which
// --norm-check gates separately (DECISIONS#review-refactor-decisions, E-CHAIN1 (a)).
//
// The candidate mask gets its own counterexample: a mask that leaves the true winner out has
// to return the best of what is left, so a broken mask cannot pass by ignoring the mask.

#include "qrt_checks.h"

#include "gguf_index.h"
#include "head.h"
#include "kernels/gemm.h"
#include "qrt_half.h"
#include "qrt_util.h"
#include "vk_dev.h"
#include "vk_transfer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

// One vocab row of the head, in double, from the raw q8_0 weight row and the packed planes.
double head_ref_dot(const uint8_t * wrow, const uint8_t * arow, const uint32_t * drow,
                    uint32_t k) {
    double acc = 0.0;
    for (uint32_t b = 0; b < k / 32u; ++b) {
        const float dw = f16_to_f32((uint32_t) wrow[b * 34u] |
                                    ((uint32_t) wrow[b * 34u + 1u] << 8));
        const float da = f16_to_f32(drow[b] & 0xFFFFu);
        int32_t sub = 0;
        for (uint32_t e = 0; e < 32u; ++e) {
            sub += (int32_t) (int8_t) wrow[b * 34u + 2u + e] *
                   (int32_t) (int8_t) arow[b * 32u + e];
        }
        acc += (double) dw * (double) da * (double) sub;
    }
    return acc;
}

} // namespace

int qrt_check_head(const qrt_config * cfg, const char * gguf_path, const char * spv_dir,
                   uint32_t rows) {
    if (cfg == NULL || gguf_path == NULL || spv_dir == NULL) {
        fprintf(stderr, "qrt: --head-check needs a model, a shader directory and rows\n");
        return 2;
    }
    if (rows == 0 || rows > QRT_HEAD_ROWS) {
        fprintf(stderr, "qrt: --head-rows must be 1..%u\n", (uint32_t) QRT_HEAD_ROWS);
        return 2;
    }
    char err[QRT_ERR_SIZE] = {0};
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    const qrt_gguf_tensor * outw = qrt_gguf_find(g, "output.weight");
    const qrt_gguf_tensor * emb = qrt_gguf_find(g, "token_embd.weight");
    std::vector<float> onorm;
    if (outw == NULL || outw->type != 8 || outw->n_dims != 2 || emb == NULL ||
        qrt_gguf_read_f32(g, "output_norm.weight", &onorm, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: head needs output.weight (q8_0), token_embd and output_norm\n");
        qrt_gguf_close(g);
        return 2;
    }
    const uint32_t E = (uint32_t) outw->ne[0];
    const uint32_t V = (uint32_t) outw->ne[1];
    if (onorm.size() != E || emb->ne[0] != (uint64_t) E) {
        fprintf(stderr, "qrt: output_norm is %zu wide, output.weight is %u x %llu\n",
                onorm.size(), E, (unsigned long long) emb->ne[1]);
        qrt_gguf_close(g);
        return 2;
    }
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_gguf_close(g);
        return 2;
    }
    int rc = 2;
    qrt_head * h = NULL;
    qrt_vk_staging * stg = NULL;
    VkBuffer x = VK_NULL_HANDLE;
    VkDeviceMemory x_mem = VK_NULL_HANDLE;
    std::vector<unsigned char> raw((size_t) outw->nbytes);
    std::vector<unsigned char> prepared((size_t) qrt_head_weight_bytes(E, V));
    std::vector<float> hx((size_t) rows * E, 0.0f);
    do {
        if (qrt_head_open(dev, spv_dir, E, V, &h, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        if (qrt_gguf_read_tensor(g, outw, raw.data(), raw.size(), err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: reading output.weight: %s\n", err);
            break;
        }
        qrt_gemm_q8k_prepare(raw.data(), V, E, prepared.data());
        if (qrt_head_set_weights(h, prepared.data(), prepared.size(), err, sizeof(err)) != 0 ||
            qrt_head_set_norm(h, 0, onorm.data(), err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: head load: %s\n", err);
            break;
        }
        // Deterministic hidden rows: the same generator the GEMM check uses, so nothing here
        // depends on what the model happens to have computed.
        uint32_t seed = 24641u;
        for (uint32_t i = 0; i < rows * E; ++i) {
            seed = seed * 1664525u + 1013904223u;
            hx[i] = ((float) (int32_t) (seed >> 8) / 8388608.0f) - 8.0f;
        }
        const uint64_t xbytes = (uint64_t) rows * E * 4;
        if (qrt_vk_alloc(dev, xbytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST, &x,
                         &x_mem, err, sizeof(err)) != 0 ||
            qrt_vk_staging_create(dev, xbytes, &stg, err, sizeof(err)) != 0 ||
            qrt_vk_upload(dev, stg, x, 0, hx.data(), xbytes, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        std::vector<uint32_t> id(rows, 0u);
        std::vector<float> best(rows, 0.0f), second(rows, 0.0f);
        const double t0 = now_ms();
        if (qrt_head_run(h, x, 0, rows, 0, 0, id.data(), best.data(), second.data(), err,
                         sizeof(err)) != 0) {
            fprintf(stderr, "qrt: head run: %s\n", err);
            break;
        }
        const double run_ms = now_ms() - t0;
        // The planes the device packed are the shared part of the reference.
        std::vector<unsigned char> qs((size_t) rows * E);
        std::vector<uint32_t> ds((size_t) rows * (E / 32u));
        std::vector<float> lg((size_t) V);
        if (qrt_head_planes(h, rows, qs.data(), ds.data(), err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: head planes: %s\n", err);
            break;
        }
        double num = 0.0, den = 0.0, max_abs = 0.0;
        uint32_t ids_equal = 1, first_diff = 0;
        int bad = 0;
        const uint32_t row_bytes = (E / 32u) * 34u;
        for (uint32_t r = 0; r < rows; ++r) {
            if (qrt_head_logits_row(h, r, lg.data(), err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: head logits row %u: %s\n", r, err);
                bad = 1;
                break;
            }
            uint32_t ref_id = 0;
            double ref_best = -INFINITY;
            for (uint32_t v = 0; v < V; ++v) {
                const double acc = head_ref_dot(raw.data() + (size_t) v * row_bytes,
                                                qs.data() + (size_t) r * E,
                                                ds.data() + (size_t) r * (E / 32u), E);
                const double d = fabs((double) lg[v] - acc);
                num += d;
                den += fabs(acc);
                if (d > max_abs) {
                    max_abs = d;
                }
                if (acc > ref_best) {   // ties keep the lower index, as the shader does
                    ref_best = acc;
                    ref_id = v;
                }
            }
            if (ref_id != id[r]) {
                ids_equal = 0;
                first_diff = r;
            }
            printf("head_check row=%u id=%u ref_id=%u logit=%.4f ref_logit=%.4f margin=%.4f\n",
                   r, id[r], ref_id, best[r], ref_best, best[r] - second[r]);
        }
        if (bad != 0) {
            break;
        }
        const double rel = den > 0.0 ? num / den : num;
        printf("head_check vocab=%u n_embd=%u rows=%u tile=%u rel_l1=%.3e max_abs=%.3e "
               "ids_equal=%u first_diff=%u ms=%.2f verdict=%s\n",
               V, E, rows, qrt_head_tile(rows), rel, max_abs, ids_equal, first_diff, run_ms,
               (rel <= 1e-6 && ids_equal) ? "PASS" : "FAIL");
        rc = (rel <= 1e-6 && ids_equal) ? 0 : 1;

        // The mask arms: the restricted head must land on the best of what the mask allows.
        std::vector<uint32_t> cands(2);
        cands[0] = id[0] / 2u;
        cands[1] = id[0];
        if (cands[0] == cands[1]) {
            cands[0] = id[0] + 1u < V ? id[0] + 1u : 0u;
        }
        if (qrt_head_set_candidates(h, cands.data(), 2, err, sizeof(err)) != 0 ||
            qrt_head_run(h, x, 0, 1, 0, 1, id.data(), best.data(), second.data(), err,
                         sizeof(err)) != 0) {
            fprintf(stderr, "qrt: head candidate arm: %s\n", err);
            rc = 2;
            break;
        }
        const double a0 = head_ref_dot(raw.data() + (size_t) cands[0] * row_bytes, qs.data(),
                                       ds.data(), E);
        const double a1 = head_ref_dot(raw.data() + (size_t) cands[1] * row_bytes, qs.data(),
                                       ds.data(), E);
        const uint32_t want = a1 >= a0 ? cands[1] : cands[0];
        printf("head_mask c0=%u a0=%.4f c1=%u a1=%.4f id=%u want=%u verdict=%s\n",
               cands[0], a0, cands[1], a1, id[0], want, id[0] == want ? "PASS" : "FAIL");
        if (id[0] != want) {
            rc = 1;
            break;
        }
        // Counterexample: drop the winner from the mask and the head must move to the other.
        const uint32_t only = want == cands[1] ? cands[0] : cands[1];
        if (qrt_head_set_candidates(h, &only, 1, err, sizeof(err)) != 0 ||
            qrt_head_run(h, x, 0, 1, 0, 1, id.data(), best.data(), second.data(), err,
                         sizeof(err)) != 0) {
            fprintf(stderr, "qrt: head counterexample arm: %s\n", err);
            rc = 2;
            break;
        }
        printf("head_mask_single only=%u id=%u verdict=%s\n", only, id[0],
               id[0] == only ? "PASS" : "FAIL");
        rc = id[0] == only ? rc : 1;
    } while (false);
    if (x_mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, x, x_mem);
    }
    if (stg != NULL) {
        qrt_vk_staging_destroy(dev, stg);
    }
    qrt_head_close(dev, h);
    qrt_vk_close(dev);
    qrt_gguf_close(g);
    return rc;
}

#else  // QRT_HAVE_VULKAN

int qrt_check_head(const qrt_config * cfg, const char * gguf_path, const char * spv_dir,
                   uint32_t rows) {
    (void) cfg; (void) gguf_path; (void) spv_dir; (void) rows;
    fprintf(stderr, "qrt: this build has no Vulkan, --head-check needs the device\n");
    return 2;
}

#endif // QRT_HAVE_VULKAN
