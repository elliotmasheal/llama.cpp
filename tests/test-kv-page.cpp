// test-kv-page.cpp - CPU reference for the paged K/V path (bench/THREE-TIER.md F10, M1).
//
// Cells are addressed through a table of 64-bit block addresses, with the logical blocks
// placed in shuffled physical order. Two properties are checked:
//   1. FLASH_ATTN_EXT reads every cell through the table and never touches the anchor;
//   2. SET_ROWS writes each row through the table and leaves the anchor untouched.
// The attention reference is computed on the host in double precision.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int HSK  = 128;
static const int HSV  = 128;
static const int NHKV = 2;
static const int DK   = HSK*NHKV;
static const int DV   = HSV*NHKV;
static const int NQ   = 4;
static const int GQA  = 4;
static const int NH   = NHKV*GQA;
static const int NC   = 256;   // cells
static const int BS   = 32;    // cells per block
static const int NBLK = NC/BS; // 8 blocks
static const int NT   = 64;    // tokens written by the SET_ROWS test

static int phys_block(int b) {
    return (b*5 + 3) % NBLK;   // 5 is coprime with 8, so this is a permutation
}

static uint16_t f2h(float f) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

static float h2f(uint16_t h) {
    ggml_fp16_t v;
    memcpy(&v, &h, sizeof(v));
    return ggml_fp16_to_fp32(v);
}

static float k_val(int c, int d) {
    return std::sin(0.017f*c + 0.031f*d) * 0.5f;
}

static float v_val(int c, int d) {
    return std::cos(0.013f*c + 0.029f*d) * 0.5f;
}

static float q_val(int h, int t, int d) {
    return std::sin(0.007f*(h*97 + t*13 + d));
}

static int compare(const char * name, const std::vector<float> & got, const std::vector<float> & want) {
    if (got.size() != want.size()) {
        std::printf("%s: size mismatch %zu vs %zu\n", name, got.size(), want.size());
        return 1;
    }

    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) {
            std::printf("%s: NaN at %zu\n", name, i);
            return 1;
        }
        max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
    }

    const bool ok = max_abs <= 2e-2;   // sanity bound: the CPU kernel works in f16 K/V
    std::printf("%s: max|diff| = %.3e  %s\n", name, max_abs, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::printf("cannot init the CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);

    const ggml_init_params ip = {
        /*.mem_size   =*/ 8*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // the blocks are their own context: in production each block is its own allocation
    ggml_context * ctx_blocks = ggml_init(ip);

    std::vector<ggml_tensor *> k_blk(NBLK);
    std::vector<ggml_tensor *> v_blk(NBLK);
    for (int b = 0; b < NBLK; ++b) {
        k_blk[b] = ggml_new_tensor_2d(ctx_blocks, GGML_TYPE_F16, DK, BS);
        v_blk[b] = ggml_new_tensor_2d(ctx_blocks, GGML_TYPE_F16, DV, BS);
    }

    if (!ggml_backend_alloc_ctx_tensors(ctx_blocks, backend)) {
        std::printf("cannot allocate the block tensors\n");
        return 1;
    }

    // table: logical block b lives in k_blk[phys_block(b)] / v_blk[phys_block(b)]
    std::vector<uint64_t> k_tab(NBLK);
    std::vector<uint64_t> v_tab(NBLK);
    for (int b = 0; b < NBLK; ++b) {
        k_tab[b] = (uint64_t) (uintptr_t) k_blk[phys_block(b)]->data;
        v_tab[b] = (uint64_t) (uintptr_t) v_blk[phys_block(b)]->data;
    }

    // fill the blocks in logical order, written into the physical block that the table names
    {
        std::vector<uint16_t> buf;

        for (int b = 0; b < NBLK; ++b) {
            buf.assign((size_t) BS*DK, 0);
            for (int r = 0; r < BS; ++r) {
                for (int d = 0; d < DK; ++d) {
                    buf[(size_t) r*DK + d] = f2h(k_val(b*BS + r, d));
                }
            }
            ggml_backend_tensor_set(k_blk[phys_block(b)], buf.data(), 0, buf.size()*sizeof(uint16_t));

            buf.assign((size_t) BS*DV, 0);
            for (int r = 0; r < BS; ++r) {
                for (int d = 0; d < DV; ++d) {
                    buf[(size_t) r*DV + d] = f2h(v_val(b*BS + r, d));
                }
            }
            ggml_backend_tensor_set(v_blk[phys_block(b)], buf.data(), 0, buf.size()*sizeof(uint16_t));
        }
    }

    // inputs and anchors get their own buffers, like the KV cache does in production;
    // the scheduler then only allocates the op results and the intermediates
    ggml_context * ctx_in = ggml_init(ip);

    ggml_tensor * k_anchor = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, DK, NC);
    ggml_tensor * v_anchor = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, DV, NC);
    ggml_tensor * k_contig = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, DK, NC);
    ggml_tensor * v_contig = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, DV, NC);
    ggml_tensor * q        = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, HSK, NQ, NH);
    ggml_tensor * mask     = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, NC, NQ, 1, 1);
    ggml_tensor * k_cur    = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, DK, NT);
    ggml_tensor * idxs     = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, NT);
    ggml_tensor * k_tab_t  = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I64, NBLK);
    ggml_tensor * v_tab_t  = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I64, NBLK);

    if (!ggml_backend_alloc_ctx_tensors(ctx_in, backend)) {
        std::printf("cannot allocate the input tensors\n");
        return 1;
    }

    ggml_backend_tensor_set(k_tab_t, k_tab.data(), 0, k_tab.size()*sizeof(uint64_t));
    ggml_backend_tensor_set(v_tab_t, v_tab.data(), 0, v_tab.size()*sizeof(uint64_t));

    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);

    int rc = 0;

    // --- FLASH_ATTN_EXT through the table ---
    {
        ggml_context * ctx = ggml_init(ip);

        // in production the anchors are views of a zero-size buffer that the kernels never
        // read; here they are real and filled with NaN so any access would show up in the output

        // views shaped exactly like the ones llama_kv_cache::get_k/get_v hand to build_attn
        ggml_tensor * k_view = ggml_view_4d(ctx, k_anchor, HSK, NHKV, NC, 1,
                ggml_row_size(GGML_TYPE_F16, HSK), ggml_row_size(GGML_TYPE_F16, DK), ggml_row_size(GGML_TYPE_F16, DK)*NC, 0);
        ggml_tensor * v_view = ggml_view_4d(ctx, v_anchor, HSV, NHKV, NC, 1,
                ggml_row_size(GGML_TYPE_F16, HSV), ggml_row_size(GGML_TYPE_F16, DV), ggml_row_size(GGML_TYPE_F16, DV)*NC, 0);

        k_view = ggml_permute(ctx, k_view, 0, 2, 1, 3);
        v_view = ggml_permute(ctx, v_view, 0, 2, 1, 3);

        const float scale = 1.0f/std::sqrt((float) HSK);

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k_view, v_view, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_paged(out, k_tab_t, v_tab_t, BS);

        // the same attention over a contiguous K/V: the paged result must match it bit for bit
        ggml_tensor * k_cview = ggml_view_4d(ctx, k_contig, HSK, NHKV, NC, 1,
                ggml_row_size(GGML_TYPE_F16, HSK), ggml_row_size(GGML_TYPE_F16, DK), ggml_row_size(GGML_TYPE_F16, DK)*NC, 0);
        ggml_tensor * v_cview = ggml_view_4d(ctx, v_contig, HSV, NHKV, NC, 1,
                ggml_row_size(GGML_TYPE_F16, HSV), ggml_row_size(GGML_TYPE_F16, DV), ggml_row_size(GGML_TYPE_F16, DV)*NC, 0);

        k_cview = ggml_permute(ctx, k_cview, 0, 2, 1, 3);
        v_cview = ggml_permute(ctx, v_cview, 0, 2, 1, 3);

        ggml_tensor * out_ref = ggml_flash_attn_ext(ctx, q, k_cview, v_cview, mask, scale, 0.0f, 0.0f);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_build_forward_expand(gf, out_ref);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            std::printf("cannot allocate the FA graph\n");
            return 1;
        }

        {
            std::vector<uint16_t> buf((size_t) DK*NC, f2h(NAN));
            ggml_backend_tensor_set(k_anchor, buf.data(), 0, buf.size()*sizeof(uint16_t));
            buf.assign((size_t) DV*NC, f2h(NAN));
            ggml_backend_tensor_set(v_anchor, buf.data(), 0, buf.size()*sizeof(uint16_t));

            buf.assign((size_t) DK*NC, 0);
            for (int c = 0; c < NC; ++c) {
                for (int d = 0; d < DK; ++d) {
                    buf[(size_t) c*DK + d] = f2h(k_val(c, d));
                }
            }
            ggml_backend_tensor_set(k_contig, buf.data(), 0, buf.size()*sizeof(uint16_t));

            buf.assign((size_t) DV*NC, 0);
            for (int c = 0; c < NC; ++c) {
                for (int d = 0; d < DV; ++d) {
                    buf[(size_t) c*DV + d] = f2h(v_val(c, d));
                }
            }
            ggml_backend_tensor_set(v_contig, buf.data(), 0, buf.size()*sizeof(uint16_t));

            std::vector<float> qf((size_t) HSK*NQ*NH);
            for (int h = 0; h < NH; ++h) {
                for (int t = 0; t < NQ; ++t) {
                    for (int d = 0; d < HSK; ++d) {
                        qf[(size_t) (h*NQ + t)*HSK + d] = q_val(h, t, d);
                    }
                }
            }
            ggml_backend_tensor_set(q, qf.data(), 0, qf.size()*sizeof(float));

            std::vector<uint16_t> mf((size_t) NC*NQ, f2h(0.0f));
            ggml_backend_tensor_set(mask, mf.data(), 0, mf.size()*sizeof(uint16_t));
        }

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            std::printf("FA compute failed\n");
            return 1;
        }
        // ne = [HSV, n_head, n_query, 1]
        std::vector<float> got((size_t) HSV*NQ*NH);
        ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));

        std::vector<float> ref(got.size());
        ggml_backend_tensor_get(out_ref, ref.data(), 0, ref.size()*sizeof(float));

        int same = 0;
        double max_abs_ref = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (memcmp(&got[i], &ref[i], sizeof(float)) == 0) {
                ++same;
            }
            max_abs_ref = std::max(max_abs_ref, (double) std::fabs(got[i] - ref[i]));
        }
        std::printf("paged vs contiguous: %d/%zu identical, max|diff| = %.3e  %s\n",
                same, got.size(), max_abs_ref, same == (int) got.size() ? "ok" : "FAIL");
        rc |= same == (int) got.size() ? 0 : 1;

        // host reference: softmax(q.k/sqrt(DK)) . v, in double
        std::vector<float> want(got.size());
        for (int h = 0; h < NH; ++h) {
            const int hk = h/GQA;
            for (int t = 0; t < NQ; ++t) {
                std::vector<double> s(NC);
                double smax = -1e30;
                for (int c = 0; c < NC; ++c) {
                    double acc = 0.0;
                    for (int d = 0; d < HSK; ++d) {
                        acc += (double) h2f(f2h(q_val(h, t, d))) * (double) h2f(f2h(k_val(c, hk*HSK + d)));
                    }
                    s[c] = acc*scale;
                    smax = std::max(smax, s[c]);
                }
                double sum = 0.0;
                for (int c = 0; c < NC; ++c) {
                    s[c] = std::exp(s[c] - smax);
                    sum += s[c];
                }
                for (int d = 0; d < HSV; ++d) {
                    double acc = 0.0;
                    for (int c = 0; c < NC; ++c) {
                        acc += s[c]*(double) h2f(f2h(v_val(c, hk*HSV + d)));
                    }
                    want[(size_t) (h + NH*t)*HSV + d] = (float) (acc/sum);
                }
            }
        }

        rc |= compare("flash_attn_ext (contiguous) vs host reference", ref, want);
        ggml_free(ctx);
        ggml_backend_sched_reset(sched);
    }

    // --- SET_ROWS through the table ---
    {
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * out = ggml_set_rows(ctx, k_anchor, k_cur, idxs);
        ggml_set_rows_set_paged(out, k_tab_t, BS);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            std::printf("cannot allocate the SET_ROWS graph\n");
            return 1;
        }

        std::vector<uint16_t> rows((size_t) NT*DK);
        for (int t = 0; t < NT; ++t) {
            for (int d = 0; d < DK; ++d) {
                rows[(size_t) t*DK + d] = f2h(0.5f + 0.001f*(float) (t*DK + d));
            }
        }
        ggml_backend_tensor_set(k_cur, rows.data(), 0, rows.size()*sizeof(uint16_t));

        std::vector<int32_t> ids(NT);
        for (int t = 0; t < NT; ++t) {
            ids[t] = (t*37) % NC;   // spread across blocks, out of order
        }
        ggml_backend_tensor_set(idxs, ids.data(), 0, ids.size()*sizeof(int32_t));

        std::vector<uint16_t> anchor((size_t) DK*NC, f2h(NAN));
        ggml_backend_tensor_set(k_anchor, anchor.data(), 0, anchor.size()*sizeof(uint16_t));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            std::printf("SET_ROWS compute failed\n");
            return 1;
        }

        // every written row must be readable at its physical block, and nothing else may change
        std::vector<uint16_t> got((size_t) BS*DK);
        int bad = 0;
        for (int t = 0; t < NT; ++t) {
            const int c    = ids[t];
            const int r    = c%BS;
            const int phys = phys_block(c/BS);
            ggml_backend_tensor_get(k_blk[phys], got.data(), (size_t) r*DK*sizeof(uint16_t), (size_t) DK*sizeof(uint16_t));
            for (int d = 0; d < DK; ++d) {
                if (got[(size_t) d] != rows[(size_t) t*DK + d]) {
                    ++bad;
                    break;
                }
            }
        }
        std::printf("set_rows paged: %d/%d rows at the wrong place  %s\n", bad, NT, bad == 0 ? "ok" : "FAIL");
        rc |= bad == 0 ? 0 : 1;

        // the anchor itself must be untouched
        ggml_backend_tensor_get(k_anchor, anchor.data(), 0, anchor.size()*sizeof(uint16_t));
        int changed = 0;
        for (size_t i = 0; i < anchor.size(); ++i) {
            if (anchor[i] != f2h(NAN)) {
                ++changed;
            }
        }
        std::printf("set_rows paged: anchor halves changed = %d  %s\n", changed, changed == 0 ? "ok" : "FAIL");
        rc |= changed == 0 ? 0 : 1;

        ggml_free(ctx);
        ggml_backend_sched_reset(sched);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_blocks);
    ggml_backend_free(backend);

    std::printf("kv page test: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
