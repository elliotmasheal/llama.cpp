#include "llama-moe-exec.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double moe_exec_now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

struct llama_moe_exec::impl {
    const llama_model & model;
    ggml_backend_t      backend = nullptr;
    int32_t             il      = -1;
    int32_t             n_slots = 0;
    int64_t             n_embd  = 0;
    int64_t             n_expert = 0;
    int64_t             n_used  = 0;

    ggml_context_ptr      ctx;
    ggml_backend_buffer_ptr buf;

    ggml_tensor * router   = nullptr; // F32 [n_embd, n_expert]
    ggml_tensor * bank_g   = nullptr;
    ggml_tensor * bank_u   = nullptr;
    ggml_tensor * bank_d   = nullptr;
    ggml_tensor * x        = nullptr;
    ggml_tensor * ids_slot = nullptr;
    ggml_tensor * out      = nullptr;
    ggml_tensor * logits_t = nullptr; // router matmul output, kept for the axis probe
    ggml_tensor * scratch  = nullptr; // stands in for one layer's misses: the average upload bytes per layer
    const ggml_tensor * src_host = nullptr; // pinned host expert weights, the upload source
    std::vector<uint8_t> scratch_host;

    ggml_cgraph * graph_router = nullptr;
    ggml_cgraph * graph_moe    = nullptr;

    std::vector<int32_t> ids;

    double router_ms = 0.0;
    double expert_ms = 0.0;
    bool   ready     = false;

    impl(const llama_model & model, ggml_backend_t backend, int32_t il, int32_t n_slots) :
            model(model), backend(backend), il(il), n_slots(n_slots) {
        const auto & hparams = model.hparams;
        n_embd   = hparams.n_embd;
        n_expert = hparams.n_expert;
        n_used   = hparams.n_expert_used_max();
        if (il < 0 || il >= (int32_t) model.layers.size() || n_expert <= 0 || n_used <= 0) {
            return;
        }
        if (n_slots <= 0 || n_slots > n_expert) {
            n_slots = (int32_t) n_expert;
        }
        const llama_layer & layer = model.layers[il];
        if (layer.ffn_gate_inp == nullptr || layer.ffn_gate_exps == nullptr ||
                layer.ffn_up_exps == nullptr || layer.ffn_down_exps == nullptr) {
            LLAMA_LOG_WARN("%s: layer %d has no routed experts, executor disabled\n", __func__, il);
            return;
        }

        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
        ctx.reset(ggml_init(ip));

        const ggml_tensor * t_router = layer.ffn_gate_inp;
        const ggml_tensor * t_gate   = layer.ffn_gate_exps;
        const ggml_tensor * t_up     = layer.ffn_up_exps;
        const ggml_tensor * t_down   = layer.ffn_down_exps;
        src_host = t_gate;
        for (const ggml_tensor * t : { t_router, t_gate, t_up, t_down }) {
            if (t->data == nullptr) {
                LLAMA_LOG_WARN("%s: %s is not loaded yet, executor self test skipped\n", __func__, t->name);
                return;
            }
        }

        router   = ggml_new_tensor_2d(ctx.get(), t_router->type, t_router->ne[0], t_router->ne[1]);
        bank_g   = ggml_new_tensor_3d(ctx.get(), t_gate->type, t_gate->ne[0], t_gate->ne[1], n_slots);
        bank_u   = ggml_new_tensor_3d(ctx.get(), t_up->type, t_up->ne[0], t_up->ne[1], n_slots);
        bank_d   = ggml_new_tensor_3d(ctx.get(), t_down->type, t_down->ne[0], t_down->ne[1], n_slots);
        x        = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
        ids_slot = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, 1);
        out      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
        // 5.2 MiB per layer = 206 MiB of misses per token / 40 layers
        scratch  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) (5.2 * 1024 * 1024));
        scratch_host.assign((size_t) ggml_nbytes(scratch), 0x5a);

        // router graph: router weight x input -> probs -> top-k ids
        ggml_tensor * logits = ggml_mul_mat(ctx.get(), router, x);
        logits_t = logits;
        ggml_tensor * probs  = ggml_soft_max(ctx.get(), logits);
        ggml_tensor * ids_t  = ggml_top_k(ctx.get(), probs, (int) n_used);
        graph_router = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph_router, ids_t);

        // expert graph: the routed experts are addressed through slots, like the MoE cache does
        ggml_tensor * g   = ggml_mul_mat_id(ctx.get(), bank_g, x, ids_slot);
        ggml_tensor * u   = ggml_mul_mat_id(ctx.get(), bank_u, x, ids_slot);
        ggml_tensor * act = ggml_mul(ctx.get(), ggml_silu(ctx.get(), g), u);
        ggml_tensor * d   = ggml_mul_mat_id(ctx.get(), bank_d, act, ids_slot);
        // get_rows indexes the ne[1] rows of src0, so the probabilities must be [1, n_expert]
        ggml_tensor * w   = ggml_get_rows(ctx.get(), ggml_reshape_2d(ctx.get(), probs, 1, n_expert), ids_slot);
        ggml_tensor * dw  = ggml_mul(ctx.get(), d, w);
        ggml_tensor * perm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), dw, 1, 0, 2, 3));
        ggml_tensor * sum  = ggml_sum_rows(ctx.get(), perm);
        ggml_tensor * res  = ggml_reshape_2d(ctx.get(), sum, n_embd, 1);
        graph_moe = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph_moe, res);

        buf.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buf) {
            LLAMA_LOG_WARN("%s: cannot allocate the executor buffer\n", __func__);
            return;
        }
        // stage the weights: the router and the first n_slots experts of the layer
        copy_tensor(t_router, router, 0, 0, ggml_nbytes(t_router));
        for (int32_t e = 0; e < n_slots; ++e) {
            copy_expert(t_gate, bank_g, e, e);
            copy_expert(t_up,   bank_u, e, e);
            copy_expert(t_down, bank_d, e, e);
        }
        std::vector<float> hx((size_t) n_embd, 0.01f);
        ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));
        ids.assign((size_t) n_used, 0);
        set_ids_slots();
        ready = true;
    }

    bool is_view(const ggml_tensor * t) const {
        return t->view_src != nullptr;
    }

    void copy_tensor(const ggml_tensor * src, ggml_tensor * dst, size_t src_off, size_t dst_off, size_t size) {
        if (!is_view(src) && src->buffer != nullptr && ggml_backend_buffer_is_host(src->buffer)) {
            ggml_backend_tensor_set(dst, (const uint8_t *) src->data + src_off, dst_off, size);
            return;
        }
        std::vector<uint8_t> tmp(size);
        ggml_backend_tensor_get(src, tmp.data(), src_off, size);
        ggml_backend_tensor_set(dst, tmp.data(), dst_off, size);
    }

    void copy_expert(const ggml_tensor * src, ggml_tensor * dst, int32_t e_src, int32_t e_dst) {
        const size_t size = src->nb[2];
        copy_tensor(src, dst, (size_t) e_src * size, (size_t) e_dst * size, size);
    }

    void set_ids_slots() {
        for (int64_t i = 0; i < n_used; ++i) {
            ids[(size_t) i] = ids[(size_t) i] % n_slots;
        }
        ggml_backend_tensor_set(ids_slot, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    double self_test(int n_layers, int iters) {
        if (!ready) {
            return -1.0;
        }
        // router axis probe: same deterministic input on both models, print the logits per expert
        if (getenv("LLAMA_MOE_EXEC_PROBE") != nullptr) {
            std::vector<float> hx((size_t) n_embd);
            for (int64_t i = 0; i < n_embd; ++i) {
                hx[(size_t) i] = 1.0f + 0.01f * (float) (i % 97);
            }
            ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));
            ggml_backend_graph_compute(backend, graph_router);
            ggml_backend_synchronize(backend);
            std::vector<float> hl((size_t) n_expert);
            ggml_backend_tensor_get(logits_t, hl.data(), 0, hl.size() * sizeof(float));
            fprintf(stderr, "PROBE");
            for (int64_t e = 0; e < n_expert; ++e) {
                fprintf(stderr, " %.6f", hl[(size_t) e]);
            }
            fprintf(stderr, "\n");
            fflush(stderr);

            // optional: fixed slot sequence, to compare the expert tensors between two model files
            const char * ids_env = getenv("LLAMA_MOE_EXEC_IDS");
            if (ids_env != nullptr && ids_env[0] != '\0') {
                std::vector<int32_t> slots((size_t) n_used, 0);
                int parsed = 0;
                const char * p = ids_env;
                while (parsed < (int) n_used && *p != '\0') {
                    slots[(size_t) parsed++] = (int32_t) strtol(p, (char **) &p, 10);
                    if (*p == ',') {
                        ++p;
                    }
                }
                ggml_backend_tensor_set(ids_slot, slots.data(), 0, slots.size() * sizeof(int32_t));
                ggml_backend_graph_compute(backend, graph_moe);
                ggml_backend_synchronize(backend);
                ggml_tensor * moe_res = ggml_graph_node(graph_moe, ggml_graph_n_nodes(graph_moe) - 1);
                std::vector<float> ho((size_t) n_embd);
                ggml_backend_tensor_get(moe_res, ho.data(), 0, ho.size() * sizeof(float));
                fprintf(stderr, "PROBEOUT");
                for (int64_t i = 0; i < 8; ++i) {
                    fprintf(stderr, " %.8f", ho[(size_t) i]);
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
            return -1.0;
        }
        // grab the router top-k tensor to read the ids back: it is the graph's output node
        ggml_tensor * ids_t = ggml_graph_node(graph_router, ggml_graph_n_nodes(graph_router) - 1);
        for (int i = 0; i < 4; ++i) {
            ggml_backend_graph_compute(backend, graph_router);
            ggml_backend_graph_compute(backend, graph_moe);
        }
        ggml_backend_synchronize(backend);

        const int n_steps = iters * n_layers;

        // phase A: router compute + sync, no host readback
        double tA = moe_exec_now_ms();
        for (int i = 0; i < n_steps; ++i) {
            ggml_backend_graph_compute_async(backend, graph_router);
            ggml_backend_synchronize(backend);
        }
        tA = moe_exec_now_ms() - tA;

        // phase B: ids readback + sync, no compute
        double tB = moe_exec_now_ms();
        for (int i = 0; i < n_steps; ++i) {
            ggml_backend_tensor_get_async(backend, ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_synchronize(backend);
        }
        tB = moe_exec_now_ms() - tB;

        // phase C: expert compute + sync
        double tC = moe_exec_now_ms();
        for (int i = 0; i < n_steps; ++i) {
            ggml_backend_graph_compute_async(backend, graph_moe);
            ggml_backend_synchronize(backend);
        }
        tC = moe_exec_now_ms() - tC;

        // phase D: the executor loop: router -> ids readback (one sync) -> expert submit
        const double t_start = moe_exec_now_ms();
        for (int it = 0; it < iters; ++it) {
            for (int l = 0; l < n_layers; ++l) {
                const double r0 = moe_exec_now_ms();
                ggml_backend_graph_compute_async(backend, graph_router);
                ggml_backend_tensor_get_async(backend, ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_synchronize(backend);
                router_ms += moe_exec_now_ms() - r0;
                set_ids_slots();
                const double e0 = moe_exec_now_ms();
                ggml_backend_graph_compute_async(backend, graph_moe);
                expert_ms += moe_exec_now_ms() - e0;
            }
        }
        ggml_backend_synchronize(backend);
        const double total = moe_exec_now_ms() - t_start;

        // phase E: executor shape with misses: router -> ids sync -> upload misses + expert compute, no per-layer sync
        double tE = moe_exec_now_ms();
        for (int it = 0; it < iters; ++it) {
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_graph_compute_async(backend, graph_router);
                ggml_backend_tensor_get_async(backend, ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_synchronize(backend);
                set_ids_slots();
                ggml_backend_tensor_set_async(backend, scratch, src_host->data, 0, scratch_host.size());
                ggml_backend_graph_compute_async(backend, graph_moe);
            }
        }
        ggml_backend_synchronize(backend);
        tE = moe_exec_now_ms() - tE;

        // phase F: upload only, device idle before every upload (isolates the transfer cost)
        double tF = moe_exec_now_ms();
        for (int i = 0; i < n_steps; ++i) {
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_set_async(backend, scratch, src_host->data, 0, scratch_host.size());
        }
        ggml_backend_synchronize(backend);
        tF = moe_exec_now_ms() - tF;

        // phase G: expert compute submitted first, upload after it (upload may overlap the pending compute)
        double tG = moe_exec_now_ms();
        for (int it = 0; it < iters; ++it) {
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_graph_compute_async(backend, graph_router);
                ggml_backend_tensor_get_async(backend, ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_synchronize(backend);
                set_ids_slots();
                ggml_backend_graph_compute_async(backend, graph_moe);
                ggml_backend_tensor_set_async(backend, scratch, src_host->data, 0, scratch_host.size());
            }
        }
        ggml_backend_synchronize(backend);
        tG = moe_exec_now_ms() - tG;

        // phase H: explicit n_threads sweep on the expert graph. The build has GGML_OPENMP on, so
        // ggml_graph_compute uses OMP num_threads(cplan.n_threads) whatever the backend threadpool is.
        // "same" routes all n_used rows to slot 0 (1.8 MiB working set); "distinct" uses n_used slots.
        auto sweep = [&](bool distinct, int n_threads) {
            struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
            ggml_threadpool_t tp = ggml_threadpool_new(&ttp);
            struct ggml_cplan cplan = ggml_graph_plan(graph_moe, n_threads, tp);
            std::vector<uint8_t> wdata(cplan.work_size);
            cplan.work_data = wdata.empty() ? nullptr : wdata.data();
            for (int64_t i = 0; i < n_used; ++i) {
                ids[(size_t) i] = distinct ? (int32_t) (i % n_slots) : 0;
            }
            ggml_backend_tensor_set(ids_slot, ids.data(), 0, ids.size() * sizeof(int32_t));
            for (int i = 0; i < 2; ++i) {
                ggml_graph_compute(graph_moe, &cplan);
            }
            const int reps = 16;
            double best = 1e30;
            for (int i = 0; i < reps; ++i) {
                const double r0 = moe_exec_now_ms();
                ggml_graph_compute(graph_moe, &cplan);
                best = std::min(best, moe_exec_now_ms() - r0);
            }
            double t = best;
            ggml_threadpool_free(tp);
            return t;
        };
        const char * tsw_env = getenv("LLAMA_MOE_EXEC_TSWEEP");
        if (tsw_env != nullptr && tsw_env[0] != '\0') {
            const double flops_layer = 2.0 * (double) n_used *
                ((double) bank_g->ne[0] * bank_g->ne[1] + (double) bank_u->ne[0] * bank_u->ne[1] +
                 (double) bank_d->ne[0] * bank_d->ne[1]);
            const double bytes_dist = (double) n_used * (double) (bank_g->nb[2] + bank_u->nb[2] + bank_d->nb[2]);
            const char * p = tsw_env;
            while (*p != '\0') {
                const int nt = (int) strtol(p, (char **) &p, 10);
                if (*p == ',') {
                    ++p;
                }
                if (nt <= 0) {
                    continue;
                }
                const double ms_same = sweep(false, nt);
                const double ms_dist = sweep(true, nt);
                fprintf(stderr, "[moe-exec] t=%d: same-expert %.3f ms/layer (%.1f GFLOPS) | "
                        "distinct %.3f ms/layer (%.1f GFLOPS, %.1f GB/s)\n",
                        nt, ms_same, flops_layer / ms_same / 1e6, ms_dist, flops_layer / ms_dist / 1e6,
                        bytes_dist / ms_dist / 1e6);
            }
            fflush(stderr);
        }

        // phase P: the promotion path. Issue a batch of expert copies through the prefetch API, then
        // run the expert graph: the compute queue must not wait for that batch. A wait follows.
        const char * pf_env = getenv("LLAMA_MOE_EXEC_PSWEEP");
        if (pf_env != nullptr && pf_env[0] != '\0') {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            using pf_copy_t   = bool (*)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t);
            using pf_submit_t = void (*)(ggml_backend_t);
            auto pf_copy   = (pf_copy_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_tensor");
            auto pf_submit = (pf_submit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_submit");
            auto pf_wait   = (pf_submit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefetch_wait");
            ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
            const int pf_experts = std::max(1, atoi(pf_env));
            if (pf_copy && pf_submit && pf_wait && buft_host != nullptr && pf_experts <= n_slots / 4) {
                const size_t src_bytes = std::max({ (size_t) bank_g->nb[2], (size_t) bank_u->nb[2], (size_t) bank_d->nb[2] });
                ggml_backend_buffer_ptr pinned(ggml_backend_buft_alloc_buffer(buft_host, src_bytes));
                void * src = pinned ? ggml_backend_buffer_get_base(pinned.get()) : nullptr;
                if (src != nullptr) {
                    ggml_backend_tensor_get(src_host, src, 0, src_bytes);
                    for (int64_t i = 0; i < n_used; ++i) {
                        ids[(size_t) i] = (int32_t) (i % n_slots);
                    }
                    ggml_backend_tensor_set(ids_slot, ids.data(), 0, ids.size() * sizeof(int32_t));
                    const int slot0 = n_slots - pf_experts;
                    for (int i = 0; i < 2; ++i) {
                        ggml_backend_graph_compute_async(backend, graph_moe);
                    }
                    ggml_backend_synchronize(backend);
                    double base = 1e30;
                    for (int i = 0; i < 8; ++i) {
                        const double t = moe_exec_now_ms();
                        ggml_backend_graph_compute_async(backend, graph_moe);
                        ggml_backend_synchronize(backend);
                        base = std::min(base, moe_exec_now_ms() - t);
                    }
                    double comp = 1e30;
                    double wait = 0.0;
                    for (int i = 0; i < 8; ++i) {
                        for (int e = 0; e < pf_experts; ++e) {
                            const size_t sg = (size_t) (slot0 + e) * bank_g->nb[2];
                            const size_t su = (size_t) (slot0 + e) * bank_u->nb[2];
                            const size_t sd = (size_t) (slot0 + e) * bank_d->nb[2];
                            pf_copy(backend, bank_g, src, sg, bank_g->nb[2]);
                            pf_copy(backend, bank_u, src, su, bank_u->nb[2]);
                            pf_copy(backend, bank_d, src, sd, bank_d->nb[2]);
                        }
                        pf_submit(backend);
                        double t = moe_exec_now_ms();
                        ggml_backend_graph_compute_async(backend, graph_moe);
                        ggml_backend_synchronize(backend);
                        comp = std::min(comp, moe_exec_now_ms() - t);
                        t = moe_exec_now_ms();
                        pf_wait(backend);
                        wait += moe_exec_now_ms() - t;
                    }
                    const double mib = (double) pf_experts *
                        (double) (bank_g->nb[2] + bank_u->nb[2] + bank_d->nb[2]) / 1048576.0;
                    fprintf(stderr, "[moe-exec] prefetch %d experts (%.1f MiB): compute %.3f ms | "
                            "compute+prefetch %.3f ms | wait %.3f ms\n",
                            pf_experts, mib, base, comp, wait / 8.0);
                    fflush(stderr);
                }
            }
        }

        const double per_layer = total / (double) n_steps;
        fprintf(stderr, "[moe-exec] layer %d, %d layers x %d iters: %.3f ms/layer (router+ids %.3f, expert %.3f)\n",
                il, n_layers, iters, per_layer,
                router_ms / (double) n_steps, expert_ms / (double) n_steps);
        fprintf(stderr, "[moe-exec] phases ms/layer: router+sync %.3f | ids+sync %.3f | expert+sync %.3f | combined %.3f\n",
                tA / (double) n_steps, tB / (double) n_steps, tC / (double) n_steps, per_layer);
        fprintf(stderr, "[moe-exec] phase E (router+ids sync + 5.2 MiB upload + expert, no per-layer sync): %.3f ms/layer\n",
                tE / (double) n_steps);
        fprintf(stderr, "[moe-exec] phase F (upload only, idle before each): %.3f ms/layer | "
                "phase G (expert then upload): %.3f ms/layer\n",
                tF / (double) n_steps, tG / (double) n_steps);
        fflush(stderr);
        return per_layer;
    }
};

llama_moe_exec::llama_moe_exec(const llama_model & model, ggml_backend_t backend, int32_t il, int32_t n_slots) :
        pimpl(new impl(model, backend, il, n_slots)) {
}

llama_moe_exec::~llama_moe_exec() = default;

bool llama_moe_exec::ok() const {
    return pimpl->ready;
}

double llama_moe_exec::self_test(int n_layers, int iters) {
    return pimpl->self_test(n_layers, iters);
}

double llama_moe_exec::last_router_ms() const {
    return pimpl->router_ms;
}

double llama_moe_exec::last_expert_ms() const {
    return pimpl->expert_ms;
}





