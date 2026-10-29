#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

// failed checks; main exits non-zero on any failures
static int n_failed = 0;

static float * gen_rand_f32(int64_t n) {
    float * data = (float *) malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        data[i] *= 5.0f;
    }
    return data;
}

// updates pointers to max_err and rms_err
static void compare_f32(const float * ref, const float * out, int64_t n, float * max_err, float * rms_err) {
    *max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float err = fabsf(ref[i] - out[i]);
        if (err > *max_err) {
            *max_err = err;
        }
        sum_sq_err += (double)err * err;
    }
    *rms_err = sqrt(sum_sq_err / n);
}

// Fill from a flat row-major f32 source: rows * cols floats.
static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t rows, int64_t cols, ggml_type qtype) {
    GGML_ASSERT(t->ne[0] == cols);
    GGML_ASSERT(rows * cols == ggml_nelements(t));
    if (qtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        return;
    }
    void * q = malloc(ggml_nbytes(t));
    // some iq types (iq2_xxs, iq2_xs, iq1_s) steer their code choice with an imatrix; the
    // quantized bytes are shared by the reference and tiled paths, so a dummy imatrix is enough
    const float * imatrix = NULL;
    float * im = NULL;
    if (ggml_quantize_requires_imatrix(qtype)) {
        im = (float *) malloc(rows * cols * sizeof(float));
        for (int64_t i = 0; i < rows * cols; i++) im[i] = 1.0f;
        imatrix = im;
    }
    ggml_quantize_chunk(qtype, src, q, 0, rows, cols, imatrix);
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
    free(im);
}

// The CPU-specific control API is resolved through the backend registry: with
// GGML_BACKEND_DL the CPU backend is a runtime-loaded module and its symbols
// are not available at link time
static void cpu_set_use_ref(ggml_backend_t backend, bool use_ref) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_use_ref)(ggml_backend_t, bool) =
        (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!set_use_ref) {
        fprintf(stderr, "ggml_backend_cpu_set_use_ref not available\n");
        exit(1);
    }
    set_use_ref(backend, use_ref);
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

static void test_matmul(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    srand(0xBEEF);

    float * src1_ref = gen_rand_f32(M * N);
    float * src0_ref = gen_rand_f32(N * K);
    float * dst_out = (float *) malloc(M * K * sizeof(float));
    float * dst_tiled = (float *) malloc(M * K * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    // ggml_mul_mat(t0, t1) computes t1 * t0^T: t0 (src0, quant) is N wide x
    // K high, t1 (src1, f32) is N wide x M high, dst = (K, M).
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst   = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    fill_tensor(src1, src1_ref, M, N, GGML_TYPE_F32);
    fill_tensor(src0, src0_ref, K, N, quant_type);

    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_out, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // std vs tiled: identical quantized inputs, so any large difference here is a bug in the tiled kernel
    float max_err, rms_err;
    compare_f32(dst_out, dst_tiled, M*K, &max_err, &rms_err);
    float tol = 1e-3f;

    printf("TEST %lldx%lld * %lldx%lld (%s): %s (max_err: %f, rms: %f, tolerance: %f)\n",
           (long long)M, (long long)N, (long long)N, (long long)K,
           ggml_type_name(quant_type), (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, tol);

    // if the tiled kernel deviates from std beyond quantization tolerance,
    // dump a few offenders
    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < M*K && shown < 8; ++i) {
            float err = fabsf(dst_out[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld (m=%lld k=%lld) std=%f tiled=%f err=%f\n",
                       (long long)i, (long long)(i/K), (long long)(i%K), dst_out[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(src1_ref); free(src0_ref); free(dst_out); free(dst_tiled);
}

// Higher-dim (ne[2], ne[3] > 1) check, std (ggml_mul_mat) is trusted as the reference
//
// src1 (F32)  : [N, M, src1_2, src1_3]   ne0=N (reduction), ne1=M (out0)
// src0 (quant): [N, K, src0_2, src0_3]   ne0=N (reduction), ne1=K (out1)
// dst = ggml_mul_mat(src0, src1) : [K, M, src1_2, src1_3]
static void test_matmul_highdim(ggml_backend_t backend, int64_t M, int64_t N, int64_t K,
                         int64_t src1_2, int64_t src1_3,
                         int64_t src0_2, int64_t src0_3,
                         ggml_type quant_type) {
    srand(0xBEEF);

    const int64_t n_src1 = N*M*src1_2*src1_3;
    const int64_t n_src0 = N*K*src0_2*src0_3;
    const int64_t n_dst = K*M*src1_2*src1_3;

    float * src1_ref   = gen_rand_f32(n_src1);
    float * src0_ref   = gen_rand_f32(n_src0);
    float * dst_std   = (float *) malloc(n_dst * sizeof(float));
    float * dst_tiled = (float *) malloc(n_dst * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_src1[4] = { N, M, src1_2, src1_3 };
    int64_t ne_src0[4] = { N, K, src0_2, src0_3 };
    struct ggml_tensor * src1  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_src1);
    struct ggml_tensor * src0 = ggml_new_tensor(ctx, quant_type,  4, ne_src0);

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst   = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // the batch dims fold into the row count (rows are contiguous, ne0 fastest)
    fill_tensor(src1, src1_ref, M * src1_2 * src1_3, N, GGML_TYPE_F32);
    fill_tensor(src0, src0_ref, K * src0_2 * src0_3, N, quant_type);

    // reference = stock path (use_ref keeps the op off the tiled hook), tiled
    // = the gated path; same weights, same op, run twice into separate buffers
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_std, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // max |dst|: scale for the quantization tolerance
    float scale = 0.0f;
    for (int64_t i = 0; i < n_dst; ++i) {
        scale = fmaxf(scale, fabsf(dst_std[i]));
    }

    // std vs tiled: identical quantized inputs, so a large difference is a bug
    float max_err, rms_err;
    compare_f32(dst_std, dst_tiled, n_dst, &max_err, &rms_err);
    float tol = (quant_type == GGML_TYPE_F32) ? 1e-4f : fmaxf(1e-3f, 1e-3f*scale);

    // which path did the tiled op take (mirrors the hard-constraint gate; the
    // harness runs with force on, so the profitability check is bypassed)
    bool tiled_kernel = (N % 256 == 0) && (src0_2 > 0) && (src0_3 > 0)
        && (src1_2 % src0_2 == 0) && (src1_3 % src0_3 == 0);

    printf("TEST %lldx%lldx%lldx%lld * %lldx%lldx%lldx%lld (%s, %s): %s (max_err: %f, rms: %f, scale: %f)\n",
           (long long)M, (long long)N, (long long)src1_2, (long long)src1_3,
           (long long)K, (long long)N, (long long)src0_2, (long long)src0_3,
           ggml_type_name(quant_type), tiled_kernel ? "tiled-kernel" : "stock",
           (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, scale);

    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < n_dst && shown < 8; ++i) {
            float err = fabsf(dst_std[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld std=%f tiled=%f err=%f\n",
                       (long long)i, dst_std[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(src1_ref); free(src0_ref); free(dst_std); free(dst_tiled);
}

// MUL_MAT_ID (MoE) check, std (vec_dot) is trusted as the reference
//
// src0 (as, quant): [K, R, n_experts]   ne0 = K (reduction), ne1 = R rows per expert, ne2 = experts
// src1 (b, F32)   : [K, b_slots, batch]  ne0 = K (reduction), ne1 = b_slots (b rows, broadcast over k columns)
// ids             : [k, batch]           expert picked per (slot, batch row); k % b_slots == 0
// dst = ggml_mul_mat_id(as, b, ids) : [R, k, batch]; column (id, t) = GEMV of src1 row (id % b_slots + t*b_slots)
// against expert ids[t*k+id]. b_slots = 1 is the common MoE case (all top-k experts see one input row).
static void test_mul_mat_id(ggml_backend_t backend, int64_t K, int64_t R, int64_t n_experts,
                            int64_t k, int64_t b_slots, int64_t batch, ggml_type quant_type,
                            bool src1_strided = false) {
    srand(0xBEEF);

    const int64_t n_as  = K * R * n_experts;
    const int64_t n_b   = K * b_slots * batch;
    const int64_t n_dst = R * k * batch;

    float * as_ref  = gen_rand_f32(n_as);
    float * b_ref   = gen_rand_f32(n_b);
    int32_t * ids   = (int32_t *) malloc(k * batch * sizeof(int32_t));
    float * dst_ref = (float *) malloc(n_dst * sizeof(float));
    float * dst_tiled = (float *) malloc(n_dst * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_as[4]  = { K, R, n_experts, 1 };
    int64_t ne_b[4]   = { K, b_slots, batch, 1 };
    int64_t ne_ids[4] = { k, batch, 1, 1 };
    struct ggml_tensor * src0  = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * ids_t = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne_ids);
    struct ggml_tensor * src1;
    struct ggml_tensor * src1_base = NULL;
    if (src1_strided) {
        // strided view: nb[1] = 4 bytes, the tiled path reads src1 through strides into wdata
        src1_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, b_slots * batch);
        src1 = ggml_view_3d(ctx, src1_base, K, b_slots, batch, 4, (size_t) b_slots * K * 4, 0);
    } else {
        src1 = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_b);
    }

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat_id(ctx, src0, src1, ids_t);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic balanced routing: each expert gets ~ k*batch/n_experts rows
    for (int64_t t = 0; t < batch; t++) {
        for (int64_t e = 0; e < k; e++) {
            ids[t * k + e] = (int32_t) ((t * k + e) % n_experts);
        }
    }

    fill_tensor(src1, b_ref, b_slots * batch, K, GGML_TYPE_F32);
    fill_tensor(src0, as_ref, R * n_experts, K, quant_type);
    ggml_backend_tensor_set(ids_t, ids, 0, ggml_nbytes(ids_t));

    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_ref, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // same quantized inputs, so a large difference is a bug in the tiled path
    float max_err, rms_err;
    compare_f32(dst_ref, dst_tiled, n_dst, &max_err, &rms_err);
    float tol = 1e-3f;

    printf("TEST mul_mat_id %lldx%lldx%lld * %lldx%lldx%lld ids[%lld,%lld] (%s%s): %s (max_err: %f, rms: %f, tolerance: %f)\n",
           (long long)K, (long long)R, (long long)n_experts,
           (long long)K, (long long)b_slots, (long long)batch,
           (long long)k, (long long)batch,
           ggml_type_name(quant_type), src1_strided ? ", strided-b" : "",
           (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, tol);

    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < n_dst && shown < 8; ++i) {
            float err = fabsf(dst_ref[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld (row=%lld slot=%lld batch=%lld) std=%f tiled=%f err=%f\n",
                       (long long)i, (long long)(i % R), (long long)((i / R) % k), (long long)(i / (R * k)),
                       dst_ref[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(as_ref); free(b_ref); free(ids); free(dst_ref); free(dst_tiled);
}

static double time_graph_compute(ggml_backend_t backend, struct ggml_cgraph * gf) {
#if defined(_WIN32)
    // high-res timer; clock_gettime is not portable across Windows toolchains
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    ggml_backend_graph_compute(backend, gf);
    QueryPerformanceCounter(&t1);
    return (double) (t1.QuadPart - t0.QuadPart) / (double) freq.QuadPart;
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ggml_backend_graph_compute(backend, gf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
#endif
}

// Warm up, then time n runs and return the best
static double time_graph_compute_best(ggml_backend_t backend, struct ggml_cgraph * gf, int n,
                                      void * flush_buf = NULL, size_t flush_size = 0) {

    time_graph_compute(backend, gf); // warmup
    double best = 1e30;
    for (int i = 0; i < n; ++i) {
        // Evict L3 cache first
        if (flush_buf) { memset(flush_buf, 0xAB, flush_size); }
        const double t = time_graph_compute(backend, gf);
        if (t < best) best = t;
    }
    return best;
}

// Fetch the repack extra buffer type through the public proc-address API
static ggml_backend_buffer_type_t get_cpu_repack_buft(void) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return NULL;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    ggml_backend_dev_get_extra_bufts_t get_extra =
        (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return NULL;
    }
    ggml_backend_buffer_type_t * bufts = get_extra(cpu_dev);
    // the only extra buffer type the CPU backend exposes is CPU_REPACK
    return (bufts && *bufts) ? *bufts : NULL;
}

struct bench_row {
    const char * name;
    double time_std, time_repack, time_tiled;
    float max_err_repack, rmse_repack;
    float max_err_tiled, rmse_tiled;
    bool have_repack;
};

// MUL_MAT three-way bench: std (default optimized GEMM) vs repack (CPU_REPACK buffer) vs
// tiled kernel. std is timed in-process with use_ref=true (bypasses the tiled gate); tiled
// and repack with use_ref=false (repack's tiled gate declines on the buffer's extra). repack
// is n/a where no repack kernel exists for the type. Errors are vs the vec-only reference (use_ref).
static bench_row bench_three_way(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    bench_row row;
    row.name = ggml_type_name(quant_type);
    row.time_std = row.time_repack = row.time_tiled = 0.0;
    row.max_err_repack = row.rmse_repack = row.max_err_tiled = row.rmse_tiled = 0.0f;
    row.have_repack = false;

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * src1     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0_std = ggml_new_tensor_2d(ctx, quant_type,  N, K);
    struct ggml_tensor * src0_rep = ggml_new_tensor_2d(ctx, quant_type,  N, K);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0_std, src1);
    ggml_build_forward_expand(gf, dst);

    struct ggml_cgraph * gf_repack = ggml_new_graph(ctx);
    struct ggml_tensor * dst_repack = ggml_mul_mat(ctx, src0_rep, src1);
    ggml_build_forward_expand(gf_repack, dst_repack);

    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    ggml_backend_buffer_t buf_rep = NULL;
    if (repack_buft && N % 8 == 0 && K % 8 == 0) {
        buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(src0_rep));
        src0_rep->buffer = buf_rep;
        src0_rep->data   = ggml_backend_buffer_get_base(buf_rep);
        ggml_backend_buffer_init_tensor(buf_rep, src0_rep);
        row.have_repack = (src0_rep->extra != NULL);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    srand(0xBEEF);
    float * src1_data = gen_rand_f32(M * N);
    float * src0_data = gen_rand_f32(N * K);
    fill_tensor(src1, src1_data, M, N, GGML_TYPE_F32);
    fill_tensor(src0_std, src0_data, K, N, quant_type);
    if (row.have_repack) {
        fill_tensor(src0_rep, src0_data, K, N, quant_type);
    }
    free(src1_data); free(src0_data);

    // one warmup + N timings; best (min) wins. The flush buffer is larger than the
    // whole-chip cache, so its memset evicts L3 and each timed iteration streams DRAM.
    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);
    const int n_reps = 5;

    // std in-process (use_ref bypasses the tiled gate; the vec_dot is the same optimized one)
    cpu_set_use_ref(backend, true);
    row.time_std = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);

    // tiled in-process (the parent holds TILED_MM=1 forced on)
    cpu_set_use_ref(backend, false);
    row.time_tiled = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);

    // repack in-process (the tiled gate declines on the buffer's extra)
    if (row.have_repack) {
        row.time_repack = time_graph_compute_best(backend, gf_repack, n_reps, flush_buf, flush_size);
    }
    free(flush_buf);

    // errors vs the vec-only reference (use_ref)
    float * out_ref    = (float *) malloc(M * K * sizeof(float));
    float * out_tiled  = (float *) malloc(M * K * sizeof(float));
    float * out_repack = (float *) malloc(M * K * sizeof(float));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_ref, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_tiled, 0, ggml_nbytes(dst));
    compare_f32(out_ref, out_tiled, M * K, &row.max_err_tiled, &row.rmse_tiled);
    if (row.have_repack) {
        ggml_backend_graph_compute(backend, gf_repack);
        ggml_backend_tensor_get(dst_repack, out_repack, 0, ggml_nbytes(dst_repack));
        compare_f32(out_ref, out_repack, M * K, &row.max_err_repack, &row.rmse_repack);
    }
    free(out_ref); free(out_tiled); free(out_repack);

    ggml_backend_buffer_free(buf_rep);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return row;
}

static void print_bench_table(int64_t M, int64_t N, int64_t K, const bench_row * rows, size_t n_types) {
    const double flops = 2.0 * M * N * K;

    printf("\nBENCH dense %lldx%lld * %lldx%lld (std vs repack vs tiled), min of 5 timings, 8 threads\n",
           (long long)M, (long long)N, (long long)N, (long long)K);
    printf("%-8s %10s %12s %12s %11s %11s %17s %17s %17s %17s\n",
           "type", "std TF", "repack TF", "tiled TF", "repack/std", "tiled/std",
           "max_err(repack)", "rmse(repack)", "max_err(tiled)", "rmse(tiled)");
    for (size_t i = 0; i < n_types; ++i) {
        const bench_row * r = &rows[i];
        if (r->have_repack) {
            printf("%-8s %10.3f %12.3f %12.3f %11.2f %11.2f %17.5e %17.5e %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   flops / (r->time_repack * 1e12), flops / (r->time_tiled * 1e12),
                   r->time_std / r->time_repack, r->time_std / r->time_tiled,
                   r->max_err_repack, r->rmse_repack, r->max_err_tiled, r->rmse_tiled);
        } else {
            printf("%-8s %10.3f %12s %12.3f %11s %11.2f %17s %17s %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   "n/a", flops / (r->time_tiled * 1e12), "n/a",
                   r->time_std / r->time_tiled,
                   "n/a", "n/a", r->max_err_tiled, r->rmse_tiled);
        }
    }
}

struct bench_row_mmid {
    const char * name;
    double time_std, time_repack, time_tiled;
    float max_err_repack, rmse_repack;
    float max_err_tiled, rmse_tiled;
    bool have_repack;
};

// MUL_MAT_ID (MoE) three-way bench: std (default GEMM, use_ref) vs repack (CPU_REPACK buffer)
// vs tiled kernel (default; TILED_MM_FORCE is set in main so ragged experts take it too).
// Column (id, t) = GEMV of src1 row (id % b_slots + t * b_slots) against expert ids[t*k+id];
// cne1 = k * batch / n_experts rows per expert on average. repack is n/a where no repack
// kernel exists. Errors are vs the vec-only reference (use_ref).
static bench_row_mmid bench_mul_mat_id(ggml_backend_t backend, int64_t K, int64_t R, int64_t n_experts,
                                       int64_t k, int64_t b_slots, int64_t batch, ggml_type quant_type,
                                       bool timing_only = false) {
    bench_row_mmid row;
    row.name = ggml_type_name(quant_type);
    row.time_std = row.time_repack = row.time_tiled = 0.0;
    row.max_err_repack = row.rmse_repack = row.max_err_tiled = row.rmse_tiled = 0.0f;
    row.have_repack = false;

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_as[4]  = { K, R, n_experts, 1 };
    int64_t ne_b[4]   = { K, b_slots, batch, 1 };
    int64_t ne_ids[4] = { k, batch, 1, 1 };
    struct ggml_tensor * src1     = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_b);
    struct ggml_tensor * src0_std = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * src0_rep = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * ids_t    = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne_ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat_id(ctx, src0_std, src1, ids_t);
    ggml_build_forward_expand(gf, dst);

    struct ggml_cgraph * gf_repack = ggml_new_graph(ctx);
    struct ggml_tensor * dst_repack = ggml_mul_mat_id(ctx, src0_rep, src1, ids_t);
    ggml_build_forward_expand(gf_repack, dst_repack);

    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    ggml_backend_buffer_t buf_rep = NULL;
    if (repack_buft && R % 8 == 0) {
        buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(src0_rep));
        src0_rep->buffer = buf_rep;
        src0_rep->data   = ggml_backend_buffer_get_base(buf_rep);
        ggml_backend_buffer_init_tensor(buf_rep, src0_rep);
        row.have_repack = (src0_rep->extra != NULL);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic balanced routing: each expert gets ~ k*batch/n_experts rows
    int32_t * ids = (int32_t *) malloc(k * batch * sizeof(int32_t));
    for (int64_t t = 0; t < batch; t++) {
        for (int64_t e = 0; e < k; e++) {
            ids[t * k + e] = (int32_t) ((t * k + e) % n_experts);
        }
    }
    srand(0xBEEF);
    float * b_data  = gen_rand_f32(K * b_slots * batch);
    float * as_data = gen_rand_f32(K * R * n_experts);
    fill_tensor(src1, b_data, b_slots * batch, K, GGML_TYPE_F32);
    fill_tensor(src0_std, as_data, R * n_experts, K, quant_type);
    if (row.have_repack) {
        fill_tensor(src0_rep, as_data, R * n_experts, K, quant_type);
    }
    ggml_backend_tensor_set(ids_t, ids, 0, ggml_nbytes(ids_t));
    free(b_data); free(as_data);

    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);
    const int n_reps = timing_only ? 7 : 5;

    // the scaling sweep only wants the production path: the scalar reference takes minutes at 256 experts
    if (timing_only) {
        cpu_set_use_ref(backend, false);
        row.time_std = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);
        free(flush_buf);
        ggml_backend_buffer_free(buf_rep);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return row;
    }

    // std in-process (use_ref bypasses the tiled gate; the vec_dot is the same optimized one)
    cpu_set_use_ref(backend, true);
    row.time_std = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);

    cpu_set_use_ref(backend, false);
    row.time_tiled = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);

    // repack in-process (the tiled gate declines on the buffer's extra)
    if (row.have_repack) {
        row.time_repack = time_graph_compute_best(backend, gf_repack, n_reps, flush_buf, flush_size);
    }
    free(flush_buf);

    // errors vs the vec-only reference (use_ref)
    float * out_ref    = (float *) malloc(R * k * batch * sizeof(float));
    float * out_tiled  = (float *) malloc(R * k * batch * sizeof(float));
    float * out_repack = (float *) malloc(R * k * batch * sizeof(float));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_ref, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_tiled, 0, ggml_nbytes(dst));
    compare_f32(out_ref, out_tiled, R * k * batch, &row.max_err_tiled, &row.rmse_tiled);
    if (row.have_repack) {
        ggml_backend_graph_compute(backend, gf_repack);
        ggml_backend_tensor_get(dst_repack, out_repack, 0, ggml_nbytes(dst_repack));
        compare_f32(out_ref, out_repack, R * k * batch, &row.max_err_repack, &row.rmse_repack);
    }
    free(out_ref); free(out_tiled); free(out_repack); free(ids);

    ggml_backend_buffer_free(buf_rep);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return row;
}

static void print_mmid_table(int64_t K, int64_t R, int64_t n_experts, int64_t k,
                             int64_t b_slots, int64_t batch, const bench_row_mmid * rows, size_t n_types) {
    const double flops = 2.0 * (double) K * R * k * batch;

    printf("\nBENCH mmid %lldx%lldx%lld k=%lld b_slots=%lld batch=%lld (cne1=%lld, std vs repack vs tiled), min of 5 timings, 8 threads\n",
           (long long)K, (long long)R, (long long)n_experts, (long long)k,
           (long long)b_slots, (long long)batch, (long long)(k * batch / n_experts));
    printf("%-8s %10s %12s %12s %11s %11s %17s %17s %17s %17s\n",
           "type", "std TF", "repack TF", "tiled TF", "repack/std", "tiled/std",
           "max_err(repack)", "rmse(repack)", "max_err(tiled)", "rmse(tiled)");
    for (size_t i = 0; i < n_types; ++i) {
        const bench_row_mmid * r = &rows[i];
        if (r->have_repack) {
            printf("%-8s %10.3f %12.3f %12.3f %11.2f %11.2f %17.5e %17.5e %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   flops / (r->time_repack * 1e12), flops / (r->time_tiled * 1e12),
                   r->time_std / r->time_repack, r->time_std / r->time_tiled,
                   r->max_err_repack, r->rmse_repack, r->max_err_tiled, r->rmse_tiled);
        } else {
            printf("%-8s %10.3f %12s %12.3f %11s %11.2f %17s %17s %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   "n/a", flops / (r->time_tiled * 1e12), "n/a",
                   r->time_std / r->time_tiled,
                   "n/a", "n/a", r->max_err_tiled, r->rmse_tiled);
        }
    }
}

// thread scaling of the std MUL_MAT_ID path at the qwen35moe decode shape: 256 experts, top-8, one token.
// Reports GB/s of expert bytes read, so "concurrency" and "clock" can be told apart with an outside
// frequency sampler (bench/RESULTS.md 0.15)
// mode 0: thread sweep at batch 1; mode 1: batch sweep at one thread count (per-op fixed cost probe)
static void run_mmid_scale(ggml_backend_t backend, int n_threads_run, int mode) {
    struct scale_shape { const char * name; int64_t K, R; } shapes[] = {
        // qwen35moe: expert_feed_forward_length 512, n_embd 2560; gate, up and down are all [n_embd, n_ff]
        { "expert", 2560, 512 },
    };
    const ggml_type types[] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K };
    const int thread_counts[] = { 1, 2, 4, 5, 8 };
    const int batch_counts[]  = { 1, 2, 4, 8, 16 };

    if (mode == 1) {
        printf("\nMMID batch sweep, E=256 experts, 8 used per token, thr=%d (per op fixed cost probe)\n", n_threads_run);
        printf("%-8s %5s %10s %10s %10s %10s\n", "type", "batch", "ms", "GB/s", "ms/64MB", "GB/s/thread");
    } else {
        printf("\nMMID thread scaling, E=256 experts, 8 used per token, batch=1 (std vec_dot path)\n");
        printf("%-10s %-8s %4s %10s %10s %12s\n", "shape", "type", "thr", "ms", "GB/s", "GB/s/thread");
    }
    for (size_t it = 0; it < sizeof(types)/sizeof(types[0]); ++it) {
        for (size_t is = 0; is < sizeof(shapes)/sizeof(shapes[0]); ++is) {
            const int64_t K = shapes[is].K;
            const int64_t R = shapes[is].R;
            if (mode == 1) {
                const int nt = n_threads_run > 0 ? n_threads_run : 5;
                cpu_set_n_threads(backend, nt);
                for (size_t ib = 0; ib < sizeof(batch_counts)/sizeof(batch_counts[0]); ++ib) {
                    const int batch = batch_counts[ib];
                    // the tokens of a batch reuse the same 8 experts, so the bytes are batch independent
                    const size_t bytes = (size_t) ggml_row_size(types[it], K) * (size_t) R * 8;
                    bench_row_mmid row = bench_mul_mat_id(backend, K, R, 256, 8, 1, batch, types[it], true);
                    const double ms = row.time_std * 1e3;
                    printf("%-8s %5d %10.3f %10.2f %10.3f %10.2f\n",
                           ggml_type_name(types[it]), batch, ms,
                           bytes / row.time_std / 1e9, ms / batch, bytes / row.time_std / 1e9 / nt);
                }
                fflush(stdout);
                continue;
            }
            const size_t bytes = (size_t) ggml_row_size(types[it], K) * (size_t) R * 8;
            for (size_t ic = 0; ic < sizeof(thread_counts)/sizeof(thread_counts[0]); ++ic) {
                const int nt = thread_counts[ic];
                if (n_threads_run > 0 && nt != n_threads_run) {
                    continue;
                }
                cpu_set_n_threads(backend, nt);
                bench_row_mmid row = bench_mul_mat_id(backend, K, R, 256, 8, 1, 1, types[it]);
                const double ms = row.time_std * 1e3;
                printf("%-10s %-8s %4d %10.3f %10.2f %12.2f\n",
                       shapes[is].name, ggml_type_name(types[it]), nt, ms,
                       bytes / row.time_std / 1e9, bytes / row.time_std / 1e9 / nt);
            }
            fflush(stdout);
        }
    }
}

// one graph_compute with one MUL_MAT_ID node over k experts: the per node cost as the expert count grows
static double time_mmid_node(ggml_backend_t backend, int64_t k, int64_t K, int64_t R, ggml_type type,
                             void * flush_buf, size_t flush_size) {
    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_as[4] = { K, R, k, 1 };
    struct ggml_tensor * src0 = ggml_new_tensor(ctx, type, 4, ne_as);
    struct ggml_tensor * src1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    struct ggml_tensor * dst  = ggml_mul_mat_id(ctx, src0, src1, ids);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "error: failed to allocate the node buffers\n");
        exit(1);
    }

    float * b  = gen_rand_f32(K);
    float * as = gen_rand_f32(K * R * k);
    int32_t * i32 = (int32_t *) malloc(k * sizeof(int32_t));
    fill_tensor(src1, b, 1, K, GGML_TYPE_F32);
    fill_tensor(src0, as, R * k, K, type);
    for (int64_t e = 0; e < k; ++e) {
        i32[e] = (int32_t) e;
    }
    ggml_backend_tensor_set(ids, i32, 0, k*sizeof(int32_t));
    free(i32);
    free(as);
    free(b);

    const double t = time_graph_compute_best(backend, gf, 7, flush_buf, flush_size);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return t;
}

// A 64 element add costs nothing, so whatever it measures is the harness itself: the 256 MB L3 flush
// leaves dirty lines whose writeback lands inside the next timed region.
static double time_tiny(ggml_backend_t backend, void * flush_buf, size_t flush_size) {
    struct ggml_init_params ip = { 1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_add(ctx, a, b));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const double t = time_graph_compute_best(backend, gf, 7, flush_buf, flush_size);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return t;
}

// Is the per-op cost of a MUL_MAT_ID a per graph_compute cost (fork/join) or a per node cost?
// N independent nodes reading N disjoint expert sets run in one graph, then as N calls of one node.
// A fused op only removes the per node share; the fork/join stays. qwen35moe decode shape, batch 1.
static void run_mmid_chain(ggml_backend_t backend, int n_threads_run) {
    const int64_t K = 2560;
    const int64_t R = 512;
    const int64_t k = 8;
    const int     counts[] = { 1, 2, 4 };
    const ggml_type type = GGML_TYPE_IQ4_XS;
    const int nt = n_threads_run > 0 ? n_threads_run : 5;
    cpu_set_n_threads(backend, nt);

    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);

    printf("\nMMID chain, K=%lld R=%lld k=%lld, thr=%d, %s: one graph with N nodes vs N calls\n",
           (long long) K, (long long) R, (long long) k, nt, ggml_type_name(type));
    printf("%6s %16s %12s %16s %12s\n", "nodes", "ms one graph", "ms/node", "ms N calls", "ms/call");

    cpu_set_use_ref(backend, false);

    double t_first = 0.0, t_last = 0.0;
    int    n_first = 0, n_last = 0;

    for (size_t ic = 0; ic < sizeof(counts)/sizeof(counts[0]); ++ic) {
        const int N = counts[ic];

        struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);

        int64_t ne_as[4] = { K, R, 8*N, 1 };
        struct ggml_tensor * src0 = ggml_new_tensor(ctx, type, 4, ne_as);
        struct ggml_tensor * src1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);

        struct ggml_tensor * ids[4] = { nullptr };
        struct ggml_tensor * dst[4] = { nullptr };
        for (int i = 0; i < N; ++i) {
            ids[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
            dst[i] = ggml_mul_mat_id(ctx, src0, src1, ids[i]);
        }

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int i = 0; i < N; ++i) {
            ggml_build_forward_expand(gf, dst[i]);
        }
        struct ggml_cgraph * gf_one[4] = { nullptr };
        for (int i = 0; i < N; ++i) {
            gf_one[i] = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf_one[i], dst[i]);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            fprintf(stderr, "error: failed to allocate the chain buffers\n");
            exit(1);
        }

        float * b  = gen_rand_f32(K);
        float * as = gen_rand_f32(K * R * 8 * N);
        fill_tensor(src1, b, 1, K, GGML_TYPE_F32);
        fill_tensor(src0, as, R * 8 * N, K, type);
        for (int i = 0; i < N; ++i) {
            int32_t i32[8];
            for (int e = 0; e < k; ++e) {
                i32[e] = (int32_t) (i*8 + e);
            }
            ggml_backend_tensor_set(ids[i], i32, 0, k*sizeof(int32_t));
        }
        free(as);
        free(b);

        const double t_graph = time_graph_compute_best(backend, gf, 7, flush_buf, flush_size);
        double t_calls = 0.0;
        for (int i = 0; i < N; ++i) {
            t_calls += time_graph_compute_best(backend, gf_one[i], 7, flush_buf, flush_size);
        }
        printf("%6d %16.3f %12.3f %16.3f %12.3f\n",
               N, t_graph*1e3, t_graph*1e3/N, t_calls*1e3, t_calls*1e3/N);
        fflush(stdout);

        if (N == counts[0]) { t_first = t_graph*1e3; n_first = N; }
        if (N == counts[sizeof(counts)/sizeof(counts[0]) - 1]) { t_last = t_graph*1e3; n_last = N; }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // time(N) = fixed + marginal*N over the two endpoints: fixed is the per graph_compute share
    const double marginal = (t_last - t_first) / (n_last - n_first);
    printf("fit: per node %.3f ms, per graph_compute %.3f ms (fork/join + graph setup)\n",
           marginal, t_first - marginal*n_first);

    // one node, more experts: the same total bytes spread over 1, 2 or 3 expert groups tells whether the
    // cost follows the data or carries a per node share that a fused gate/up/down would pay only once
    printf("\nMMID one node, K=%lld R=%lld thr=%d %s: expert sweep\n",
           (long long) K, (long long) R, nt, ggml_type_name(type));
    printf("%6s %10s %10s %10s\n", "k", "MiB", "ms", "GB/s");
    const int64_t ks[] = { 8, 16, 24 };
    double t_k8 = 0.0, t_k24 = 0.0;
    for (size_t ik = 0; ik < sizeof(ks)/sizeof(ks[0]); ++ik) {
        const double t = time_mmid_node(backend, ks[ik], K, R, type, flush_buf, flush_size);
        const double mib = (double) ggml_row_size(type, K) * (double) R * (double) ks[ik] / 1048576.0;
        printf("%6lld %10.1f %10.3f %10.2f\n", (long long) ks[ik], mib, t*1e3, mib*1048576.0/(t*1e9));
        if (ks[ik] == 8)  { t_k8  = t*1e3; }
        if (ks[ik] == 24) { t_k24 = t*1e3; }
    }
    printf("3 nodes of 8 = %.3f ms vs 1 node of 24 = %.3f ms (fused gate/up/down upper bound)\n",
           3.0*t_k8, t_k24);

    // what the harness itself adds: the tiny graph must be microseconds, anything above is the flush
    const double t_h = time_tiny(backend, flush_buf, flush_size);
    const double t_n = time_tiny(backend, NULL, 0);
    printf("harness: tiny graph %.3f ms with the L3 flush, %.3f ms without (flush writeback = %.3f ms)\n",
           t_h*1e3, t_n*1e3, (t_h - t_n)*1e3);
    fflush(stdout);
    free(flush_buf);
}

// The fused expert feed forward against the four node sequence the layer builds today, at the qwen35moe
// decode shape: gate/up [n_embd, n_ff], down [n_ff, n_embd], 8 experts, one token, two slots masked off.
static void run_mmid_fused(ggml_backend_t backend, int n_threads_run) {
    const int64_t n_embd = 2560;
    const int64_t n_ff   = 512;
    const int64_t k      = 8;
    const ggml_type type = GGML_TYPE_IQ4_XS;
    const int nt = n_threads_run > 0 ? n_threads_run : 5;
    cpu_set_n_threads(backend, nt);

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_w[4] = { n_embd, n_ff, k, 1 };
    int64_t ne_d[4] = { n_ff, n_embd, k, 1 };
    struct ggml_tensor * up   = ggml_new_tensor(ctx, type, 4, ne_w);
    struct ggml_tensor * gate = ggml_new_tensor(ctx, type, 4, ne_w);
    struct ggml_tensor * down = ggml_new_tensor(ctx, type, 4, ne_d);
    struct ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, 1);
    struct ggml_tensor * w    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, 1);

    auto combine = [&](ggml_tensor * down_out) {
        ggml_tensor * w_cpu = ggml_mul(ctx, down_out, w);
        ggml_tensor * sum   = nullptr;
        for (int64_t i = 0; i < k; ++i) {
            ggml_tensor * view = ggml_view_2d(ctx, w_cpu, n_embd, 1, w_cpu->nb[2], i*w_cpu->nb[1]);
            sum = sum == nullptr ? view : ggml_add(ctx, sum, view);
        }
        return sum;
    };

    // the four node sequence of src/llama-moe-cache.cpp register_hybrid
    ggml_tensor * up_r   = ggml_mul_mat_id_mask(ctx, up,   x, ids, mask);
    ggml_tensor * gate_r = ggml_mul_mat_id_mask(ctx, gate, x, ids, mask);
    ggml_tensor * act_r  = ggml_swiglu_split(ctx, gate_r, up_r);
    ggml_tensor * dn_r   = ggml_mul_mat_id_mask(ctx, down, act_r, ids, mask);
    ggml_tensor * out_r  = combine(dn_r);

    // the fused node
    ggml_tensor * fused  = ggml_mul_mat_id_swiglu(ctx, up, gate, down, x, ids, mask);
    ggml_tensor * out_f  = combine(fused);

    struct ggml_cgraph * gf_r = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_r, out_r);
    struct ggml_cgraph * gf_f = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_f, out_f);

    // which node carries the time: the gate/up dot is 2560 long over 512 rows, the down dot is 512
    // long over 2560 rows, so the two do the same bytes with 5x the number of vec_dot calls
    struct { const char * name; ggml_tensor * out; } per_op[] = {
        { "up",      ggml_mul_mat_id_mask(ctx, up,   x, ids, mask) },
        { "gate",    ggml_mul_mat_id_mask(ctx, gate, x, ids, mask) },
        { "swiglu",  act_r },
        { "down",    dn_r },
        { "mul+sum", out_r },
    };

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "error: failed to allocate the fused test buffers\n");
        exit(1);
    }

    float * x_data = gen_rand_f32(n_embd);
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    free(x_data);

    float * w_up   = gen_rand_f32(n_embd*n_ff*k);
    float * w_gate = gen_rand_f32(n_embd*n_ff*k);
    float * w_down = gen_rand_f32(n_ff*n_embd*k);
    fill_tensor(up,   w_up,   n_ff*k, n_embd, type);
    fill_tensor(gate, w_gate, n_ff*k, n_embd, type);
    fill_tensor(down, w_down, n_embd*k, n_ff, type);
    free(w_up);
    free(w_gate);
    free(w_down);

    const int32_t ids_data[8]  = { 3, 1, 7, 0, 5, 2, 6, 4 };
    const int32_t mask_data[8] = { 1, 1, 1, 0, 1, 1, 0, 1 };
    float w_data[8];
    for (int64_t i = 0; i < k; ++i) {
        w_data[i] = 0.125f*(float) (i + 1);
    }
    ggml_backend_tensor_set(ids,  ids_data,  0, k*sizeof(int32_t));
    ggml_backend_tensor_set(mask, mask_data, 0, k*sizeof(int32_t));
    ggml_backend_tensor_set(w,    w_data,    0, k*sizeof(float));

    cpu_set_use_ref(backend, false);

    float * ref = (float *) malloc(n_embd*sizeof(float));
    float * got = (float *) malloc(n_embd*sizeof(float));
    ggml_backend_graph_compute(backend, gf_r);
    ggml_backend_tensor_get(out_r, ref, 0, ggml_nbytes(out_r));
    ggml_backend_graph_compute(backend, gf_f);
    ggml_backend_tensor_get(out_f, got, 0, ggml_nbytes(out_f));

    float max_err = 0.0f, rms = 0.0f;
    compare_f32(ref, got, n_embd, &max_err, &rms);
    printf("\nMMID fused gate/up/swiglu/down, n_embd=%lld n_ff=%lld k=%lld, thr=%d, %s\n",
           (long long) n_embd, (long long) n_ff, (long long) k, nt, ggml_type_name(type));
    printf("4 nodes vs 1 fused node: max_err %.3e, rms %.3e\n", max_err, rms);

    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);

    // alternate the two graphs rep by rep: the machine drifts by more than the effect, so only paired
    // runs are comparable, and the minimum of each arm still sees cold data
    const int n_rep = 15;
    double best_r = 1e30, best_f = 1e30;
    double ratio[15];
    time_graph_compute(backend, gf_r);
    time_graph_compute(backend, gf_f);
    for (int i = 0; i < n_rep; ++i) {
        memset(flush_buf, 0xAB, flush_size);
        const double t_r = time_graph_compute(backend, gf_r);
        memset(flush_buf, 0xAB, flush_size);
        const double t_f = time_graph_compute(backend, gf_f);
        if (t_r < best_r) { best_r = t_r; }
        if (t_f < best_f) { best_f = t_f; }
        ratio[i] = t_r/t_f;
    }
    // median of the paired ratios, robust to the odd rep that lands on a core migration
    for (int i = 1; i < n_rep; ++i) {
        const double key = ratio[i];
        int j = i - 1;
        while (j >= 0 && ratio[j] > key) { ratio[j + 1] = ratio[j]; --j; }
        ratio[j + 1] = key;
    }
    const double t_r = best_r;
    const double t_f = best_f;

    // which node carries the time: the gate/up dot is 2560 long over 512 rows, the down dot is 512
    // long over 2560 rows, so the two do the same bytes with 5x the number of vec_dot calls
    for (size_t i = 0; i < sizeof(per_op)/sizeof(per_op[0]); ++i) {
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, per_op[i].out);
        const double t = time_graph_compute_best(backend, g, 7, flush_buf, flush_size);
        const size_t bytes = (per_op[i].out->op == GGML_OP_MUL_MAT_ID_MASK) ? ggml_nbytes(per_op[i].out->src[0]) : 0;
        if (bytes > 0) {
            printf("  %-8s %8.3f ms  %.1f MiB  %.2f GB/s\n", per_op[i].name, t*1e3,
                   bytes/1048576.0, (double) bytes/(t*1e9));
        } else {
            printf("  %-8s %8.3f ms\n", per_op[i].name, t*1e3);
        }
    }
    free(flush_buf);

    const double mib = (double) ggml_row_size(type, n_embd)*n_ff*k*2.0/1048576.0 +
                       (double) ggml_row_size(type, n_ff)*n_embd*k/1048576.0;
    printf("4 nodes %.3f ms (%.2f GB/s), 1 fused %.3f ms (%.2f GB/s), min-ratio %.2fx, median paired %.2fx\n",
           t_r*1e3, mib*1048576.0/(t_r*1e9), t_f*1e3, mib*1048576.0/(t_f*1e9), t_r/t_f, ratio[n_rep/2]);
    fflush(stdout);

    free(ref);
    free(got);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Best of n of a block read over a fixed buffer, scattered or linear. The hybrid worker reads the
// expert weights from the host buffer of the GPU backend, so this compares that memory with plain RAM.
static double read_blocks_best(const char * base, size_t bytes, size_t block, size_t n_blocks,
                               int n_threads, bool scatter, int reps) {
    const size_t n_in = bytes / block;
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        std::atomic<uint64_t> acc{0};
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ths;
        for (int i = 0; i < n_threads; ++i) {
            ths.emplace_back([&, i] {
                std::mt19937_64 rng(0x9e3779b97f4a7c15ull*(i + 1) + r);
                uint64_t sum = 0;
                const size_t per = n_blocks / n_threads;
                size_t blk = scatter ? (size_t)(rng() % n_in) : (size_t) i*per;
                for (size_t b = 0; b < per; ++b) {
                    const char * p = base + blk*block;
                    for (size_t off = 0; off < block; off += 64) {
                        sum += *(const uint64_t *)(p + off);
                    }
                    blk = scatter ? (size_t)(rng() % n_in) : (blk + 1) % n_in;
                }
                acc.fetch_add(sum, std::memory_order_relaxed);
            });
        }
        for (auto & t : ths) {
            t.join();
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) {
            best = dt;
        }
        if (acc.load() == 0xdeadbeefull) {
            printf(" ");
        }
    }
    return best;
}

static void probe_host_buft() {
    const size_t size_mib  = 512;
    const size_t block_kib = 1928;
    const size_t n_blocks  = 320;
    const int    reps      = 3;

    printf("\nCPU read of the host buffer of every device: %zu MiB, %zu KiB blocks, %zu blocks per run\n",
           size_mib, block_kib, n_blocks);

    for (size_t id = 0; id < ggml_backend_dev_count(); ++id) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(id);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft == nullptr) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size_mib << 20);
        if (buf == nullptr) {
            printf("  %-24s alloc failed\n", ggml_backend_buft_name(buft));
            continue;
        }
        char       * base  = (char *) ggml_backend_buffer_get_base(buf);
        const size_t bytes = size_mib << 20;
        memset(base, 0x5a, bytes); // fault in every page first, write-combined memory needs a fair start
        printf("  %-24s (%s)\n", ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
        for (int t : {1, 2, 4, 5}) {
            const double ds = read_blocks_best(base, bytes, block_kib << 10, n_blocks, t, true,  reps);
            const double dl = read_blocks_best(base, bytes, block_kib << 10, n_blocks, t, false, reps);
            printf("    %d threads scatter %6.2f GB/s, linear %6.2f GB/s\n", t,
                   (double) n_blocks*(block_kib << 10)/ds/1e9,
                   (double) n_blocks*(block_kib << 10)/dl/1e9);
        }
        ggml_backend_buffer_free(buf);
    }
}

int main(int argc, char ** argv) {
    bool run_bench = false;
    bool probe_buft = false;
    bool run_fuzz = false;
    int  mmid_scale = -1; // -1 = off, 0 = all thread counts, >0 = one count
    int  mmid_batch = -1; // -1 = off, 0 = default thread count, >0 = that thread count, batch sweep
    int  mmid_chain = -1; // -1 = off, 0 = default thread count, >0 = that thread count, graph vs calls probe
    int  mmid_fused = -1; // -1 = off, 0 = default thread count, >0 = that thread count, fused op check
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else if (strcmp(argv[i], "--fuzz") == 0) {
            run_fuzz = true;
        } else if (strcmp(argv[i], "--mmid-scale") == 0) {
            mmid_scale = (i + 1 < argc) ? atoi(argv[++i]) : 0;
        } else if (strcmp(argv[i], "--mmid-batch") == 0) {
            mmid_batch = (i + 1 < argc) ? atoi(argv[++i]) : 0;
        } else if (strcmp(argv[i], "--mmid-chain") == 0) {
            mmid_chain = (i + 1 < argc) ? atoi(argv[++i]) : 0;
        } else if (strcmp(argv[i], "--mmid-fused") == 0) {
            mmid_fused = (i + 1 < argc) ? atoi(argv[++i]) : 0;
        } else if (strcmp(argv[i], "--host-buft") == 0) {
            probe_buft = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    // Enable tiled MM, also force tiled MM even when unprofitable for benchmarks. The scaling mode
    // measures the production path, so it leaves both switches alone
    if (mmid_scale < 0 && mmid_batch < 0 && mmid_chain < 0 && mmid_fused < 0) {
#if defined(_WIN32)
        _putenv("GGML_CPU_TILED_MM=1");
        _putenv("GGML_CPU_TILED_MM_FORCE=1");
#else
        setenv("GGML_CPU_TILED_MM", "1", 1);
        setenv("GGML_CPU_TILED_MM_FORCE", "1", 1);
#endif
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }

    if (probe_buft) {
        probe_host_buft();
        ggml_backend_free(backend);
        return 0;
    }

    if (mmid_fused >= 0) {
        run_mmid_fused(backend, mmid_fused);
        ggml_backend_free(backend);
        return 0;
    }

    if (mmid_chain >= 0) {
        run_mmid_chain(backend, mmid_chain);
        ggml_backend_free(backend);
        return 0;
    }

    if (mmid_batch >= 0) {
        run_mmid_scale(backend, mmid_batch, 1);
        ggml_backend_free(backend);
        return 0;
    }

    if (mmid_scale >= 0) {
        run_mmid_scale(backend, mmid_scale, 0);
        ggml_backend_free(backend);
        return 0;
    }

    cpu_set_n_threads(backend, 8);
    // base case, smoke each quant type
    test_matmul(backend, 256, 512, 256, GGML_TYPE_Q6_K);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_Q5_K);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_Q4_K);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_Q3_K);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_Q2_K);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ4_XS);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ2_XXS);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ2_XS);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ2_S);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ3_XXS);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ3_S);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ1_S);
    test_matmul(backend, 256, 512, 256, GGML_TYPE_IQ1_M);
    test_mul_mat_id(backend,  256,  256,   2, 8, 1,  64, GGML_TYPE_Q4_K);  // single block, cne1 = 256
    test_mul_mat_id(backend,  512,  300,   4, 2, 1, 150, GGML_TYPE_IQ2_XXS); // iq type, imatrix quant

    if (run_fuzz) {
        printf("Running fuzz tests.\n");
        // ragged edges, both subblock lengths (Q5_K = 32, Q6_K = 16)
        test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q6_K);   // long K, int32 accumulation
        test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q6_K);    // ragged M and K
        test_matmul(backend, 16, 1024, 16, GGML_TYPE_Q5_K);
        test_matmul(backend, 18, 1024, 7, GGML_TYPE_Q5_K);       // ragged M tail, ragged K in first subblock
        test_matmul(backend, 8, 256, 8, GGML_TYPE_Q5_K);
        test_matmul(backend, 18, 1024, 256, GGML_TYPE_Q5_K);
        test_matmul(backend, 9, 256, 256, GGML_TYPE_Q5_K);
        test_matmul(backend, 8, 256, 8, GGML_TYPE_Q6_K);         // tiny M, single QK_K block
        test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q3_K);
        test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q3_K);
        test_matmul(backend, 8, 256, 8, GGML_TYPE_Q3_K);
        test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q3_K);     // K past a full 256 tile
        test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q3_K);     // K past a full microtile
        test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q2_K);
        test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q2_K);
        test_matmul(backend, 8, 256, 8, GGML_TYPE_Q2_K);
        test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q2_K);
        test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q2_K);
        test_matmul(backend, 357, 1024, 137, GGML_TYPE_IQ4_XS);    // ragged M and K
        test_matmul(backend,   8,  256,   8, GGML_TYPE_IQ4_XS);    // tiny M, single QK_K block
        test_matmul(backend,  17, 1024, 257, GGML_TYPE_IQ4_XS);    // K past a full 256 tile

        // fuzz: M/K around the microtile (16) and tile (256) boundaries
        // Q4_K = subblock 32, Q6_K = subblock 16
        test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q4_K);
        test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q4_K);
        test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q4_K);      // M and K in the first microtile
        test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q4_K);      // M and K past the first microtile
        test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q4_K);    // M and K one short of a tile
        test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q4_K);    // M and K one past a tile
        test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q4_K);    // 1 tile + 15
        test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q4_K);    // 1 tile + 16
        test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q4_K);    // 2 tiles - 1
        test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q4_K);    // 2 tiles + 1
        test_matmul(backend, 17, 512, 257, GGML_TYPE_Q4_K);      // ragged M and K, N = 2 blocks

        test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q6_K);
        test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q6_K);
        test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q6_K);
        test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q6_K);
        test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q6_K);
        test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q6_K);
        test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q6_K);
        test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q6_K);
        test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q6_K);
        test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q6_K);
        test_matmul(backend, 17, 512, 257, GGML_TYPE_Q6_K);

        // MUL_MAT_ID (MoE): K = reduction (tiled gate needs K % 256 == 0), R = output rows per expert,
        // k = top-k slots, b_slots = b rows (1 = broadcast MoE, k = i11-diverse gather), cne1 = k*batch/E
        test_mul_mat_id(backend,  512,  300,   4, 2, 1, 150, GGML_TYPE_Q4_K);  // ragged R, broadcast, cne1 = 75
        test_mul_mat_id(backend,  512,  256,   4, 2, 2, 150, GGML_TYPE_Q5_K);  // i11-diverse gather, HAS_MIN
        test_mul_mat_id(backend,  512,  256,   4, 2, 1, 100, GGML_TYPE_Q6_K);  // ragged cne1 = 50
        test_mul_mat_id(backend,  768,  256,   4, 2, 1, 150, GGML_TYPE_Q3_K);  // K = 3 slabs
        test_mul_mat_id(backend, 1024, 2048, 128, 8, 1, 2048, GGML_TYPE_Q4_K); // large, cne1 = 128
        test_mul_mat_id(backend,  512,  256,  32, 2, 1,   8, GGML_TYPE_Q4_K);  // tiny cne1 (~0.5), forced only

        // cne1 > 256: the k-outer ring sweeps multiple 256-row windows per expert (the multi-window path)
        test_mul_mat_id(backend,  256,  256,   8, 8, 1,  512, GGML_TYPE_Q4_K);  // cne1 = 512, two full windows
        test_mul_mat_id(backend,  512,  256,   8, 8, 1,  300, GGML_TYPE_Q4_K);  // cne1 = 300, 2nd window ragged (44 rows)
        test_mul_mat_id(backend,  256,   64,   8, 8, 1,  257, GGML_TYPE_Q4_K);  // cne1 = 257, 1-row tail window, R = 1 group
        test_mul_mat_id(backend,  512,  128,   8, 8, 1,  511, GGML_TYPE_Q4_K);  // cne1 = 511, 15-row ragged tail
        test_mul_mat_id(backend,  768,  300,   8, 8, 1,  300, GGML_TYPE_Q4_K);  // 3 K slabs x 2 windows, ragged R and cne1
        test_mul_mat_id(backend,  768,  300,   8, 2, 2,  300, GGML_TYPE_Q5_K);  // cne1 = 600, 3 slabs, 3 windows (last ragged), i11-diverse, HAS_MIN

        // cne1 = 255: one short of a full ring, last ring row zero-padded by the unpack
        test_mul_mat_id(backend,  512,  256,   4, 4, 1,  255, GGML_TYPE_Q4_K);

        // long K: 8 slabs per ring row
        test_mul_mat_id(backend, 2048,  300,   8, 2, 1,  400, GGML_TYPE_Q4_K);  // cne1 = 100

        // R group edges: 64 = TILED_MMID_GROUP (ceil div, ragged group tail, single-group early return)
        test_mul_mat_id(backend,  512,   64,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ngroups = 1, most threads idle
        test_mul_mat_id(backend,  512,   65,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ceil-div boundary, 1-row tail group
        test_mul_mat_id(backend,  512,   63,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ragged group tail
        test_mul_mat_id(backend,  512,    1,   4, 1, 1,  256, GGML_TYPE_Q4_K);  // R = 1 row, cne1 = 256

        // strided (non-contiguous) F32 src1: nb[1] = 4 bytes, the tiled path reads through strides
        test_mul_mat_id(backend,  512,  256,   8, 8, 1,  128, GGML_TYPE_Q4_K, true);  // cne1 = 128

        // narrow path (cne1 in [8,16], K a multiple of k_extent): the weight is read as long per-row K streams
        test_mul_mat_id(backend, 1024,  256,   2, 2, 1,  16, GGML_TYPE_Q4_K);  // cne1 = 16
        test_mul_mat_id(backend, 1024,  256,   2, 1, 1,  16, GGML_TYPE_Q4_K);  // cne1 = 8
        test_mul_mat_id(backend,  512,  128,   2, 3, 1,   8, GGML_TYPE_Q5_K);  // cne1 = 12

        // higher-dim (ne[2], ne[3] > 1)
        // equal-batch and broadcast shapes (src1_2=2*src0_2, src1_3=2*src0_3)
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 2, 1, GGML_TYPE_Q4_K); // 3D, tiled
        test_matmul_highdim(backend, 1024, 1024, 1024, 1, 2, 1, 2, GGML_TYPE_Q4_K); // 3D, tiled
        test_matmul_highdim(backend,  512, 1024,  512, 2, 2, 2, 2, GGML_TYPE_Q4_K); // 4D, tiled
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 1, 1, GGML_TYPE_Q4_K); // broadcast r2=2 (src1_2>src0_2)
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 1, GGML_TYPE_Q4_K); // broadcast r3=2 (src1_3>src0_3)
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 2, 1, GGML_TYPE_Q6_K); // 3D, tiled, q6_K
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 2, GGML_TYPE_Q3_K); // 4D, tiled, q3_K
        test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 2, GGML_TYPE_Q2_K); // 4D, tiled, q2_K
    }


    // benchmark: std (use_ref) vs repack (CPU_REPACK buffer) vs tiled kernel, over all tiled
    // types. std is timed with use_ref=true (bypasses the tiled gate); tiled and repack with
    // use_ref=false. Errors are vs the vec-only reference (use_ref).
    if (run_bench) {
        printf("Starting benchmark runs.\n");
        const ggml_type bench_types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                          GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                                          GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M };
        const size_t n_types = sizeof(bench_types) / sizeof(bench_types[0]);

        struct { int64_t M, N, K; } shapes[] = {
            { 4096, 4096, 4096 },
            { 4096, 4096,   64 },
            { 4096, 4096,   32 },
            { 4096, 4096,   24 },
            { 4096, 4096,   16 },
            { 4096, 4096,   8 },
            { 4096, 4096,   1 },
        };
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
            bench_row rows[n_types];
            for (size_t i = 0; i < n_types; ++i) {
                rows[i] = bench_three_way(backend, shapes[s].M, shapes[s].N, shapes[s].K, bench_types[i]);
            }
            print_bench_table(shapes[s].M, shapes[s].N, shapes[s].K, rows, n_types);
        }

        // MUL_MAT_ID (MoE): K must be a multiple of 256 (the tiled slab). cne1 = k*batch/E.
        struct { int64_t K, R, E, k, b_slots, batch; } mmid_shapes[] = {
            { 1024, 1024,  8,   8, 1,    1 },  // cne1 = 1, single-token decode
            { 1024, 1024,  4,   2, 1,   16 },  // cne1 = 8, narrow path
            { 1024, 1024,  4,   2, 1,   32 },  // cne1 = 16, narrow path
            {  512,  512,  8,   2, 1,  128 },  // cne1 = 32
            { 1024, 1024, 16,   8, 1,   64 },  // cne1 = 32
            { 1024, 1024, 16,   8, 1,  256 },  // cne1 = 128
            { 1024, 2048, 32,   8, 1,  512 },  // cne1 = 128, wide experts
            { 2048, 1024, 16,   8, 1, 1024 },  // cne1 = 512, long dot
        };
        for (size_t s = 0; s < sizeof(mmid_shapes) / sizeof(mmid_shapes[0]); ++s) {
            bench_row_mmid rows[n_types];
            for (size_t i = 0; i < n_types; ++i) {
                rows[i] = bench_mul_mat_id(backend, mmid_shapes[s].K, mmid_shapes[s].R, mmid_shapes[s].E,
                                           mmid_shapes[s].k, mmid_shapes[s].b_slots, mmid_shapes[s].batch, bench_types[i]);
            }
            print_mmid_table(mmid_shapes[s].K, mmid_shapes[s].R, mmid_shapes[s].E, mmid_shapes[s].k,
                             mmid_shapes[s].b_slots, mmid_shapes[s].batch, rows, n_types);
        }
    }

    ggml_backend_free(backend);
    return n_failed ? 1 : 0;
}
