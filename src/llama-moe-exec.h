#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>

struct llama_model;

// Stage 2 MoE executor: runs the router and the routed experts of one layer as two small graphs on a
// single backend, so ggml_backend_sched is not involved. The self test measures the per-layer cost
// (router -> ids readback -> experts) with the real model tensors and backend.
class llama_moe_exec {
public:
    llama_moe_exec(const llama_model & model, ggml_backend_t backend, int32_t il, int32_t n_slots);
    ~llama_moe_exec();

    bool ok() const;

    // runs n_layers steps of the executor loop and returns the average per-layer milliseconds
    double self_test(int n_layers, int iters);

    double last_router_ms() const;
    double last_expert_ms() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
