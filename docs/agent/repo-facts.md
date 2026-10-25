# 仓库事实

- backend 只有 **CPU 与 Vulkan**，两者之外的东西已物理删除，不要去找。
- 模型只支持 **qwen35moe**（目标模型就是它），`llama_model_mapping()` 只有一个 case。
- 仍在的组件：
  - `src/`：模型加载 / 图 / KV / 采样 / C API
  - `ggml/src/ggml-cpu`、`ggml/src/ggml-vulkan` 与调度器
  - `common/`：CLI 与 `fit.cpp`
  - `tools/`：cli / server / llama-bench / mtmd / perplexity
- decode 主路径**不走**通用 `llama_context` 图（design 3.3）。可复用：GGUF 解析、Vulkan 设备初始化、tokenizer、块格式定义。
- `llama-moe-cache` 的调度与预取**不复用**（静态驻留不如动态 LRU）。
- 可复用的内核与格式：
  - Vulkan `mul_mat_vec` / `mul_mat_id`（要改成批量 k 版本）
  - Q4_K / Q5_K / Q6_K / Q8_0 的块格式定义（`ggml-common.h`）
- 已删除的 tool：`quantize`、`completion`、`batched-bench`、`fit-params`、`imatrix`、`tokenize`。
- 已删除的后端：除 CPU / Vulkan 之外全部。
- 不改变量化格式；不改变输出：k=1 路径 logits KLD <= 1e-4，推测路径接受的 token 与参考 greedy 一致。
- 需要考虑不同硬件下的表现，不能只适配本机。
- 本仓库不考虑"能否上游化"：它的价值是小 VRAM 设备上指定模型的极致性能，核心目标是重写专用运行时而非优化上游。
