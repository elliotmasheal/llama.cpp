// GGML_VK_VRAM_LIMIT_MIB probe: prints each GPU device's reported memory before and after the
// backend is initialized, so a tier can be checked without loading a model.
// build: g++ -O2 -std=c++17 -fopenmp bench/tools/vklimit-probe.cpp -Iggml/include -o probe.exe
//   build-vulkan/ggml/src/{ggml,ggml-vulkan,ggml-cpu,ggml-base}.a %VULKAN_SDK%/Lib/vulkan-1.lib -lws2_32
#include "ggml-backend.h"
#include <cstdio>

int main() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        size_t f0 = 0, t0 = 0;
        ggml_backend_dev_memory(dev, &f0, &t0);

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

        size_t f1 = 0, t1 = 0;
        ggml_backend_dev_memory(dev, &f1, &t1);

        printf("%-10s before: free %8.1f MiB / total %8.1f MiB   after init: free %8.1f MiB\n",
               ggml_backend_dev_name(dev), f0/1024.0/1024.0, t0/1024.0/1024.0, f1/1024.0/1024.0);

        if (backend) {
            ggml_backend_free(backend);
        }
    }
    return 0;
}
