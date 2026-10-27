#include "llama-store-io.h"

// bind one backend's transfer entry points into this io unit (2c). Resolution is exactly what the
// MoE cache did before (moe-cache.cpp init_prefetch_api): proc addresses through the device's
// backend reg, so both paths share one set of Vulkan entries and there is no second registration.
bool llama_store_io_bind_backend(struct llama_store_pf_api & pf, ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    pf.tensor     = (llama_store_pf_tensor_fn)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_tensor");
    pf.submit     = (llama_store_pf_submit_fn)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_submit");
    pf.wait       = (llama_store_pf_wait_fn)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_wait");
    pf.submit_seq = (llama_store_pf_submit_seq_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_submit_seq");
    pf.wait_seq   = (llama_store_pf_wait_seq_fn)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_wait_seq");
    pf.bound      = backend;
    return pf.ok();
}

// counter sum for the wiring sanity print; zero transfers are real data until the swaps land
int64_t llama_store_io_total(const llama_store_io * io) {
    if (io == nullptr) {
        return 0;
    }
    return io->total();
}
