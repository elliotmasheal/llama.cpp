// Shape probe for dense MUL_MAT: same (m,k) as the target case, one row of quantization types,
// sweeping the token count. It answers whether q6_K's gap over q4_K is a byte ceiling, an unpack
// ceiling, or a warptile choice - run it with GGML_VK_MM_PIPELINE_TRACE=1 to see which kernel lands.
// build: g++ -O2 -std=c++17 -fopenmp bench/tools/gemm-shape.cpp -Iggml/include -o build-vulkan/bin/gemm-shape.exe \
//        -Wl,--start-group build-vulkan/ggml/src/ggml.a build-vulkan/ggml/src/ggml-base.a \
//        build-vulkan/ggml/src/ggml-cpu.a build-vulkan/ggml/src/ggml-vulkan/ggml-vulkan.a \
//        -Wl,--end-group D:/Scoop/apps/vulkan/current/Lib/vulkan-1.lib -lwinmm
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const uint64_t m = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 8192;
    const uint64_t k = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 2048;

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            strstr(ggml_backend_dev_description(d), "6700S")) {
            dev = d;
        }
    }
    if (dev == nullptr) { printf("no device\n"); return 1; }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft_dev = ggml_backend_dev_buffer_type(dev);

    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16 };
    const uint64_t ns[] = { 512, 1024, 2048, 4096 };
    const int reps = 7;

    printf("%-6s %6s %8s %10s %8s %10s\n", "type", "n", "ms", "TFLOP/s", "GiB/s", "bytes(src0)");
    for (ggml_type t : types) {
        for (uint64_t n : ns) {
            ggml_init_params gp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(gp);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, t, (int64_t) k, (int64_t) m);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
            ggml_tensor * d = ggml_mul_mat(ctx, w, x);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft_dev);
            if (buf == nullptr) { printf("alloc failed\n"); return 1; }

            // zero the payloads: NaN/Inf in uninitialized f16 changes the arithmetic path
            {
                std::vector<uint8_t> z0(ggml_nbytes(w), 0), z1(ggml_nbytes(x), 0);
                ggml_backend_tensor_set(w, z0.data(), 0, ggml_nbytes(w));
                ggml_backend_tensor_set(x, z1.data(), 0, ggml_nbytes(x));
            }

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, d);

            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_ms();
                ggml_backend_graph_compute(backend, gf);
                best = std::min(best, now_ms() - t0);
            }
            // dst is f32 [m, n], so the byte total is weights + activations in + out
            const double bytes = (double) ggml_nbytes(w) + (double) ggml_nbytes(x) + (double) ggml_nbytes(d);
            const double flops = 2.0 * (double) m * (double) n * (double) k;
            printf("%-6s %6llu %8.3f %10.2f %8.1f %10.1f\n",
                   ggml_type_name(t), (unsigned long long) n, best,
                   flops / (best * 1e-3) / 1e12,
                   bytes / (best * 1e-3) / 1e9 / 1.0737,
                   ggml_nbytes(w) / 1073741824.0);

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
