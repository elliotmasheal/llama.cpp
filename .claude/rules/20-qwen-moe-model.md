# Qwen MoE 模型约束

- 目标是 Qwen3.6-35B-A3B，但实际 GGUF metadata 和当前 loader 优先于方案中的理论数字。方案中的 40 个主 block、256 experts、8 routed + shared、混合 Gated DeltaNet/attention 和 MTP 是目标 workload，不是对任意模型的无条件假设。
- `src/models/qwen35moe.cpp` 是 Qwen3.5/3.6 MoE 的 model/graph 入口。`build_layer_ffn` 同时处理 routed MoE、shared expert gate 和 MTP 相关 graph。
- 优化 expert residency、upload 或 dispatch 时，不能改变 router 的输入、Top-K 结果、权重归一化、shared gate、残差顺序或 MTP 语义。
- expert key 至少按 `(layer, expert)` 区分。同一个 expert id 在不同 layer 不是同一个权重对象。
- prediction 只能作为 prefetch hint。不能用预测结果跳过 router，也不能以近似 hidden-state hash 直接复用未经证明等价的 routing result。
- shared expert 常驻只能减少 residency 变化；不能移除 gate 或改变其计算结果。
- Q4_K_M 是开发/质量基线，Q3_K_M 是性能对照，IQ2 等是实验量化。文件更小不等于 Vulkan 更快，必须实测并做 perplexity/输出回归。
