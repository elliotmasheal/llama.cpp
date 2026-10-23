// STAGE 0 validation for the deferred-copy mechanism (GGML_VK_DEFERRED_COPIES):
// - a deferred batch signals its own semaphore and is not waited by a compute that only depends on earlier copies
// - ggml_backend_wait_deferred makes the next compute wait for the deferred batch
// - deferred data arrives correctly, and repeated deferred writes to one buffer stay ordered
// build: g++ -O2 -std=c++17 -fopenmp bench/deferred-copy.cpp -Iggml/include -o build/bin/deferred-copy.exe \
//        -Wl,--start-group build/ggml/src/ggml.a build/ggml/src/ggml-base.a build/ggml/src/ggml-cpu.a \
//        build/ggml/src/ggml-vulkan/ggml-vulkan.a -Wl,--end-group <vulkan-1.lib>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_dev_description(d), "6700S")) {
            dev = d;
        }
    }
    if (dev == nullptr) { printf("no device\n"); return 1; }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft_dev  = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);

    const int n = 1024;
    const int n_big = 1 << 20;

    ggml_init_params gp = { ggml_tensor_overhead()*64 + ggml_graph_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * big = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_big);
    ggml_tensor * add = ggml_add(ctx, a, b);
    ggml_backend_buffer_t buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft_dev);
    if (buf_dev == nullptr) { printf("device buffer alloc failed\n"); return 1; }

    // pinned host staging
    ggml_context * ctx_h = ggml_init({ ggml_tensor_overhead()*8, nullptr, true });
    ggml_tensor * hsrc = ggml_new_tensor_1d(ctx_h, GGML_TYPE_F32, n_big);
    ggml_backend_buffer_t buf_h = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, buft_host);
    if (buf_h == nullptr) { printf("host buffer alloc failed\n"); return 1; }
    float * h = (float *) hsrc->data;

    for (int i = 0; i < n; ++i) { h[i] = (float) (i % 7) + 1.0f; }
    ggml_backend_tensor_set(a, h, 0, ggml_nbytes(a));
    for (int i = 0; i < n; ++i) { h[i] = (float) (i % 5) + 2.0f; }
    ggml_backend_tensor_set(b, h, 0, ggml_nbytes(b));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, add);

    auto compute = [&]() {
        ggml_backend_graph_compute_async(backend, gf);
        ggml_backend_synchronize(backend);
    };
    auto check = [&](ggml_tensor * t, const std::vector<float> & expect, const char * tag) {
        std::vector<float> got(n);
        ggml_backend_tensor_get(t, got.data(), 0, ggml_nbytes(t));
        bool ok = true;
        for (int i = 0; i < n; ++i) { if (std::fabs(got[i] - expect[i]) > 1e-3f) { ok = false; break; } }
        printf("RESULT: %s correctness=%s\n", tag, ok ? "OK" : "MISMATCH");
        return ok;
    };

    std::vector<float> va(n), vb(n), vsum(n);
    for (int i = 0; i < n; ++i) { va[i] = (float) (i % 7) + 1.0f; vb[i] = (float) (i % 5) + 2.0f; vsum[i] = va[i] + vb[i]; }
    std::vector<float> va2(n), vsum2(n);
    for (int i = 0; i < n; ++i) { va2[i] = va[i] + va[i]; }

    // A: baseline normal copies
    ggml_backend_tensor_set_async(backend, a, va.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set_async(backend, b, vb.data(), 0, ggml_nbytes(b));
    compute();
    check(add, vsum, "A-normal");

    // B: deferred batch is not waited by a compute that does not use it
    ggml_backend_tensor_set_async(backend, a, va.data(), 0, ggml_nbytes(a));
    for (int i = 0; i < n_big; ++i) { h[i] = 1.0f; }
    ggml_backend_set_tensor_async_deferred(backend, big, h, 0, ggml_nbytes(big));
    ggml_backend_tensor_set_async(backend, b, va.data(), 0, ggml_nbytes(b));  // b := a, so add == a+a
    compute();
    check(add, va2, "B-deferred-unrelated");
    { std::vector<float> gb(n_big); ggml_backend_tensor_get(big, gb.data(), 0, ggml_nbytes(big));
      bool ok = true; for (int i = 0; i < n_big; i += 97) { if (gb[i] != 1.0f) { ok = false; break; } }
      printf("RESULT: B-deferred-data correctness=%s\n", ok ? "OK" : "MISMATCH"); }

    // C: explicit wait before the consumer
    ggml_backend_tensor_set_async(backend, a, va.data(), 0, ggml_nbytes(a));
    ggml_backend_set_tensor_async_deferred(backend, b, vb.data(), 0, ggml_nbytes(b));
    ggml_backend_wait_deferred(backend);
    compute();
    check(add, vsum, "C-deferred-consumer");

    // D: repeated deferred writes to the same buffer stay ordered
    for (int i = 0; i < n_big; ++i) { h[i] = 3.0f; }
    ggml_backend_set_tensor_async_deferred(backend, big, h, 0, ggml_nbytes(big));
    for (int i = 0; i < n_big; ++i) { h[i] = 4.0f; }
    ggml_backend_set_tensor_async_deferred(backend, big, h, 0, ggml_nbytes(big));
    ggml_backend_wait_deferred(backend);
    compute();
    { std::vector<float> gb(n_big); ggml_backend_tensor_get(big, gb.data(), 0, ggml_nbytes(big));
      bool ok = true; for (int i = 0; i < n_big; i += 97) { if (gb[i] != 4.0f) { ok = false; break; } }
      printf("RESULT: D-deferred-ordered correctness=%s\n", ok ? "OK" : "MISMATCH"); }

    // F: per-call CPU cost
    const int iters = 1000;
    {
        const double t0 = now_s();
        for (int i = 0; i < iters; ++i) { ggml_backend_set_tensor_async_deferred(backend, big, h, 0, 4096); }
        ggml_backend_wait_deferred(backend);
        ggml_backend_synchronize(backend);
        const double dt = now_s() - t0;
        printf("RESULT: F-issue-us-per-call=%.2f\n", dt*1e6/iters);
    }
    {
        const double t0 = now_s();
        for (int i = 0; i < iters; ++i) { ggml_backend_wait_deferred(backend); }
        const double dt = now_s() - t0;
        printf("RESULT: F-wait-us-per-call=%.2f\n", dt*1e6/iters);
    }

    ggml_backend_buffer_free(buf_dev);
    ggml_backend_buffer_free(buf_h);
    ggml_free(ctx);
    ggml_free(ctx_h);
    ggml_backend_free(backend);
    return 0;
}
