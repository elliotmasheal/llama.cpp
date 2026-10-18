# L9: 上传/计算重叠尝试 - ring 目的地实测更慢，真正的串行点是 ids 回读的全设备 drain（2026-10-18，task-15）

> 分支 `moe-gpu-cpu-hybrid` @ `e2efd789c`。结论：**未保留代码**。
> 实现了 N=5 的 ring 目的地 + 批内安全复用（无 per-split event），但实测比基线慢
> （cache 5120：233 t/s 带批 drain、248 t/s 不带批 drain，基线 368）。
> 归因非常明确：`sched_copy_experts` 里 **每层一次的 ids 回读带 `ggml_backend_synchronize`**
> （30 次/eval）是硬串行点；任何流水深度都被它截断到"一层"，所以调度器侧少等的等待
> 全部转移到回调内部，总时间守恒。480 在本轮不可达，需要把该 drain 换成"只等产出 ids 的那个 split"。

## 1. 实现（已回退）

- `ggml-backend.cpp`：`GGML_SCHED_HW_SLOTS = 5` 的 ring，按 backend 分配 5 x max(host weight bytes)
  （210 MiB x 5 = 1050 MiB）；主机权重拷贝写到 ring slot，不再写 gallocr 复用区；
  env `GGML_SCHED_HW_PIPELINE=1` 开启。
- **依赖规则/安全性**：slot k 只允许在"所有已提交的 compute 都完成"之后被重写。实现方式是
  **批**：每次连续 N 个 host-weight 拷贝为一批，批首做一次 `ggml_backend_sched_wait_backend`
  （全设备 drain，于是所有 slot 空闲），批内每个 slot 只写一次，因此任何 slot 的写入都不会与
  仍在执行的 split 争用。这个规则本身是安全可证明的（不需要 per-split event）。
- 实测（cache 5120，r2）：带批 drain = 233 t/s；跳过批首 drain（racy 诊断）= 248 t/s；
  基线（无 ring）= 368 t/s。都更慢。

## 2. 归因：ids 回读的 drain 是硬串行点（数字来自 `GGML_SCHED_TRACE=1`）

| 配置 | copy 总时间 | 其中调度器等待(cpy_sync) | 其中回调/h2d body | pp512 |
|---|---:|---:|---:|---:|
| 基线 cache 5120 | 1360 ms | 1286 ms（94 次 per-copy wait） | 20 ms | 368 |
| ring + 批首 drain | 2160 ms | 534 ms | **1625 ms** | 233 |
| ring，无批首 drain（racy） | 2042 ms | 46 ms | **1997 ms** | 248 |

把调度器的 per-copy 等待去掉后，`cpy_sync` 从 1286 降到 46-534 ms，但 h2d body 从 20 ms
涨到 ~2.0 s。这个 body 不是"发起拷贝"的时间，而是**回调里的 drain 时间**：
`sched_copy_experts` 对每层的 gate 拷贝做 ids 回读（`ggml_backend_tensor_get_async` +
`ggml_backend_synchronize`），每 eval 30 次。当主机不再被 per-copy wait 挡住时，主机跑得更靠前，
这 30 次全设备 drain 每次都要等"整整一层的上传+计算"完成（~50 ms），30 x 50 ms ~= 1.5 s。
所以"总时间守恒"：**30 次全设备 drain 已经把流水深度限制为一层，任何情况下都无法重叠**。

这也重新解释了 L8 的 racy 797：那个诊断同时去掉了 per-copy wait，让 GPU 在数据还没到位的
情况下乱序执行（结果不可信），快的是"没有正确等待"这件事本身，而不是可安全获得的流水收益。
它不能当作正确的上界。

## 3. 要真正做到重叠，必须改掉这个 drain

正确形状（下一步）：
1. 回调需要"只等产出 ids 的那个 split"，而不是全设备同步。最小改动是在 scheduler 里为
   每个 backend 记录一个"最后一个已提交 split 的完成信号"（per-split event 或 timeline
   semaphore，`ggml_backend_event_record` 已有类似实现），提供
   `ggml_backend_sched_wait_split(sched, backend)`；`sched_copy_experts` 用它替换
   `ggml_backend_synchronize`（ids 回读仍然需要，但只等 routing split）。
2. 目的地仍然需要 ring（本轮已实现且安全性可证明），否则 per-copy wait 会回来。
3. 代价预算：per-split event_record 在 L5 实测约 3 ms/split（90 splits -> 0.27 s），
   但本形状只需要**每层一次**（30 次）-> 约 0.09-0.2 s；而收益是隐藏约 0.55 s 上传，
   净收益估算 +100 t/s 级别。若 event_record 有便宜的 queue-submit-only 变体，成本还能更低。

本轮没有落地这两步（第 1 步需要新的 scheduler/backend 原语并对 30 次/evals 的等待语义做
逐点验证，剩余时间不足以完成"正确性优先 + 交叉 r>=3"的验收）；按任务约束回退，树保持
L7 状态。

## 4. 状态

回退后基线与任务基线一致：cache 6144 pp512 **383.3**、tg32 **26.44**；`git status` 对
src/ggml 干净（只有本文档新增）。正确性未受影响（没有代码改动）。

原始日志：`bench/moe-baseline/T15-*.{out,err}`。
