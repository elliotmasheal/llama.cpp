# STAGE 0: 双 transfer semaphore + deferred 批次（2026-10-20，task-28）

> 分支 `moe-gpu-cpu-hybrid` @ `2f9069e5b` + 本次改动。**STAGE 0 通过**：deferred 批次按独立
> timeline semaphore 提交，只被显式 `wait_deferred` 的 compute 等待；数据正确；每调用成本 ~0.5 us；
> 关闭时（默认）行为完全不变。**这是后端提交层的最小改动，尚未接到 MoE 路径（STAGE 1 待做）。**

## 1. 机制（env 门控，默认关）

- `ggml/src/ggml-vulkan`：新增第二个 timeline semaphore `transfer_semaphore_deferred` 和第二个
  transfer 命令上下文 `transfer_ctx_deferred`。`GGML_VK_DEFERRED_COPIES=1` 时：
  - `ggml_backend_set_tensor_async_deferred(backend, tensor, data, off, size)` 的拷贝进入 deferred ctx；
  - `ggml_vk_submit_transfer_ctx` 先提交普通批次（signal A），再提交 deferred 批次（signal B）；
  - `ggml_vk_get_compute_ctx` 照旧等 A；只有 `wait_deferred_pending` 被置位时（即调用过
    `ggml_backend_wait_deferred`）才追加等 B 并清除标志。
- `ggml/src/ggml-backend.cpp` / `ggml-backend.h` / `ggml-backend-impl.h`：backend iface 增加两个
  可选入口 `set_tensor_async_deferred` / `wait_deferred`，以及通用封装：
  不支持的后端自动退化为普通 `set_tensor_async`（正确性不受影响，只是没有重叠）。
- 未设置 `GGML_VK_DEFERRED_COPIES` 时：deferred API 直接走普通路径，B 信号量保持 0，无额外开销。

## 2. 验证（`bench/deferred-copy.cpp`，合成工作负载）

序列：普通拷贝 a/b -> 计算 c=a+b；deferred 大拷贝到 big -> 计算 c=a+a（不依赖 big）；
deferred 拷贝 b -> `wait_deferred` -> 计算 c=a+b；同一 buffer 连续两次 deferred 写；成本测量。
等待关系由 `GGML_VK_LOG_WAITS=1` 逐提交打印（确定性，不依赖计时）。

| 测试 | 特征 ON | 特征 OFF |
|---|---|---|
| A 普通拷贝 + 计算 | correctness OK | OK |
| B deferred 批次 + 无关计算（不应等 B） | OK，日志 `waits A=0 B=0` | OK，B_val=0 |
| B deferred 数据到达 | big 内容正确 OK | OK |
| C `wait_deferred` 后消费者 | OK，日志 `waits B=1`（B_val=2） | OK（走 A 路径） |
| D 同 buffer 连续 deferred 写保序 | OK（最终内容=第二次写入） | OK |
| F 每调用成本（issue） | **0.47-0.52 us/call** | 2.58 us/call（普通路径） |
| F `wait_deferred` 成本 | **~0.00 us/call**（只置标志） | - |
| 重复运行 | 结果一致 | 结果一致 |

关键证据（ON）：无关 compute 的提交 `waits A=0 B=0`（B_val=1 时也不等 B）；`wait_deferred` 之后
的消费者提交 `waits B=1`。这证明"deferred 批次不被无关 compute 等待、显式等待生效"。

## 3. 缓冲区 hazard 规则（STAGE 1 必须遵守）

- deferred 目的缓冲：只有在其**消费者已经完成**之后才可重写（本次测试 D 证明同一 deferred 批次内
  的连续写按提交顺序串行；跨批次重写需要 wait + 消费者完成）。
- STAGE 1 采用 MoE cache 的 **bank 槽**作为 deferred 目的：槽在 token 内被规划为某 expert 专用，
  消费者是该层的 down MUL_MAT_ID；同一 token 内不会重写，跨 token 由现有的 per-token 同步保证。
- 消费者必须在自己的 compute 之前调用一次 `ggml_backend_wait_deferred(backend)`（每个 token 每层
  一次，成本 ~0 us）。

## 4. 状态与下一步

- 改动为 env 门控、默认关闭；`GGML_VK_LOG_WAITS` 仅诊断用。构建通过，合成测试全部通过。
- STAGE 1（把该层 up/down 的 miss 上传排成 deferred 批次，让它们叠在 up/down 计算上）尚未实现，
  需要：`llama_moe_cache::copy()` 把最后一张 expert 张量（down）的 fills 拆成 deferred 批次，
  split 提交后由 scheduler 的 post-submit hook 发车，并在 down split 前调用 `wait_deferred`。
- 计时验收仍被机器污染阻塞（16:38 canonical 182.6/14.71 vs 干净 401/26.4）；STAGE 1 完成后需要
  干净窗口才能给 TG 数字，本轮只报告确定性计数与机制证据。
