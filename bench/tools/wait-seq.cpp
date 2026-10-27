// Validation for the sequence-addressed transfer wait (ggml_backend_vk_prefetch_submit_seq / _wait_seq):
// - a batch on the prefetch timeline can be waited by its own number, later batches stay in flight
// - the control (prefetch_wait) waits everything, so the two timings must differ by the big transfer
// - data of every waited batch is correct, and a compute issued in between is not blocked by them
// build: g++ -O2 -std=c++17 -fopenmp bench/tools/wait-seq.cpp -Iggml/include -o build-vulkan/bin/wait-seq.exe \
//        -Wl,--start-group build-vulkan/ggml/src/ggml.a build-vulkan/ggml/src/ggml-base.a \
//        build-vulkan/ggml/src/ggml-cpu.a build-vulkan/ggml/src/ggml-vulkan/ggml-vulkan.a \
//        -Wl,--end-group D:/Scoop/apps/vulkan/current/Lib/vulkan-1.lib -lwinmm
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main() {
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) { printf("no GPU\n"); return 1; }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft_dev  = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    printf("device: %s\n", ggml_backend_dev_description(dev));

    // a small expert-like payload and a whole-layer one, both uploaded on the prefetch timeline
    const size_t n_small = 1 << 18;                        // 1 MiB of f32
    const size_t n_big   = (size_t) 1 << 28;               // 1 GiB of f32
    const size_t n       = 1024;

    ggml_context * ctx = ggml_init({ ggml_tensor_overhead()*16 + ggml_graph_overhead()*2, nullptr, true });
    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_small);
    ggml_tensor * big = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) n_big);
    ggml_tensor * x   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * y   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * sum = ggml_add(ctx, x, y);
    ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft_dev);

    ggml_context * ctx_h = ggml_init({ ggml_tensor_overhead()*8, nullptr, true });
    ggml_tensor * h_small = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, n_small);
    ggml_tensor * h_big   = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, (int64_t) n_big);
    ggml_tensor * h_n     = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, n);
    ggml_backend_buffer_t buf_h = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, buft_host);
    if (buf_h == nullptr || h_small->data == nullptr) { printf("pinned host alloc failed\n"); return 1; }

    float * ps = (float *) h_small->data;
    float * pb = (float *) h_big->data;
    float * pn = (float *) h_n->data;
    for (size_t i = 0; i < n_small; ++i) { ps[i] = (float) (i % 13) + 0.5f; }
    for (size_t i = 0; i < n_big;   ++i) { pb[i] = (float) (i % 7); }
    // x and y are staged separately: reusing the payload buffer would overwrite what is prefetched
    for (size_t i = 0; i < n; ++i) { pn[i] = 0.0f; }
    ggml_backend_tensor_set(x, pn, 0, ggml_nbytes(x));
    for (size_t i = 0; i < n; ++i) { pn[i] = 1.0f; }
    ggml_backend_tensor_set(y, pn, 0, ggml_nbytes(y));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);

    auto check_small = [&](const char * tag) {
        std::vector<float> got(n_small);
        ggml_backend_tensor_get(a, got.data(), 0, ggml_nbytes(a));
        bool ok = true;
        for (size_t i = 0; i < n_small; i += 4093) {
            if (std::fabs(got[i] - ((float) (i % 13) + 0.5f)) > 1e-4f) { ok = false; break; }
        }
        printf("RESULT: %s small-data=%s\n", tag, ok ? "OK" : "MISMATCH");
        return ok;
    };
    auto check_big = [&](const char * tag) {
        std::vector<float> got(1 << 20);
        ggml_backend_tensor_get(big, got.data(), 0, sizeof(float) * got.size());
        bool ok = true;
        for (size_t i = 0; i < got.size(); i += 8191) {
            if (got[i] != (float) (i % 7)) { ok = false; break; }
        }
        printf("RESULT: %s big-data=%s\n", tag, ok ? "OK" : "MISMATCH");
        return ok;
    };

    // 1. seq-addressed wait: the 1 MiB batch must be taken without waiting for the 1 GiB one behind it
    if (!ggml_backend_vk_prefetch_tensor(backend, a, ps, 0, ggml_nbytes(a))) { printf("prefetch_tensor(a) refused\n"); return 1; }
    const uint64_t s1 = ggml_backend_vk_prefetch_submit_seq(backend);
    if (!ggml_backend_vk_prefetch_tensor(backend, big, pb, 0, ggml_nbytes(big))) { printf("prefetch_tensor(big) refused\n"); return 1; }
    const uint64_t s2 = ggml_backend_vk_prefetch_submit_seq(backend);
    printf("seq: small=%llu big=%llu (must be 1 and 2)\n", (unsigned long long) s1, (unsigned long long) s2);

    const double t0 = now_ms();
    ggml_backend_vk_prefetch_wait_seq(backend, s1);
    const double t_small = now_ms() - t0;

    // a compute that touches neither tensor, issued while the 1 GiB batch is still in flight
    const double t1 = now_ms();
    ggml_backend_graph_compute_async(backend, gf);
    const double t2 = now_ms();
    ggml_backend_synchronize(backend);
    const double t3 = now_ms();

    const double t4 = now_ms();
    ggml_backend_vk_prefetch_wait_seq(backend, s2);
    const double t_big = now_ms() - t4;

    printf("TIMING: wait_seq(small)=%.2f ms | compute submit=%.2f ms sync=%.2f ms | wait_seq(big)=%.2f ms\n",
           t_small, t2 - t1, t3 - t2, t_big);
    check_small("seq");
    check_big("seq");

    // 2. control: one submit pair again, but wait the newest like the existing caller does
    ggml_backend_vk_prefetch_tensor(backend, a, ps, 0, ggml_nbytes(a));
    ggml_backend_vk_prefetch_submit(backend);
    ggml_backend_vk_prefetch_tensor(backend, big, pb, 0, ggml_nbytes(big));
    ggml_backend_vk_prefetch_submit(backend);
    const double t5 = now_ms();
    ggml_backend_vk_prefetch_wait(backend);
    printf("TIMING: wait(newest, control)=%.2f ms  <- must be close to the 1 GiB transfer, not to wait_seq(small)\n",
           now_ms() - t5);
    check_small("control");
    check_big("control");

    // 3. the verdict: waiting a number must be much cheaper than waiting everything
    const bool pass = s1 == 1 && s2 == 2 && t_small + 5.0 < t_big && t_big > 10.0;
    printf("VERDICT: seq-addressed wait %s (small %.2f ms vs big %.2f ms)\n", pass ? "OK" : "FAIL", t_small, t_big);

    ggml_backend_free(backend);
    return pass ? 0 : 2;
}
