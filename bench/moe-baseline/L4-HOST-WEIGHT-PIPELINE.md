# L4: 双缓冲 host-weight 拷贝目的地 - 实测结论（2026-10-18，task-10）

> 分支 `moe-gpu-cpu-hybrid` @ `edf737a3b`。目标：把 prefill 的 host-weight 拷贝做成正确的流水
> （pp512 >= 700，文本一致，tg 不退化）。结论：**在本任务的写作用域内（只改
> `ggml/src/ggml-backend.cpp`）无法达成**，所有正确变体都比基线慢 35-40%。原因已定位到
> `llama_context::sched_copy_experts` 里的 ids 回读 + 全设备 drain。代码已全部回退，
> `git status` 对 ggml/src 干净。

## 1. 基线（本机复测，r=2-3）

| 配置 | pp512 | tg32 |
|---|---:|---:|
| static `-ncmoe 27 -t 5 -fa on` | 344.0-345.6 | 18.7-19.0 |
| static pp2048 ub2048 | 577.9 | - |
| cache `-ncmoe 41 --moe-cache-mib 6144 -ub 2048` | 358.9 | 26.4 |
| cache pp2048 ub2048 | 577.9 | - |

## 2. 试过的三个正确变体（都 env 门控，`GGML_SCHED_HW_PIPELINE`）

原型：给 host-weight 拷贝在 split backend 上分配 **2 个轮换 slot**（2 x max(nbytes) =
2 x 210 MiB = 420 MiB VRAM），拷贝写入 slot，不再写 gallocr 的复用区；每 splits 的
`input_cpy->buffer/data` 指向 slot；`node->src` 指向的同一 tensor 描述符在 copy 前被改指向，
因此消费该权重的 split 从 slot 读。

| 变体 | 顺序 | pp512 | tg32 | 评价 |
|---|---|---:|---:|---|
| V0 baseline | `wait` -> callback copy -> compute | **344.0** | 18.7 | 参考 |
| V1 slot + event | 跳过 `wait`，写入 slot，split compute 后 record event；2 次拷贝后 sync event | 225.4 | - | 慢 |
| V2 slot，copy 后 drain | 跳过 `wait`，copy -> slot，然后 `ggml_backend_synchronize` | 224.3 | - | 慢 |
| V3 原文直译（callback 提到 wait 之上，保留 wait） | copy（原 gallocr 目的地）-> drain | 252.4 | - | 慢，且仍有数据竞争 |
| V4 不 drain（数据竞争参照） | 跳过 wait 且不 drain | 223.5 | - | 慢（且文本错误） |

关键反直觉点：**即使完全照搬 task-9 的"把 callback 提到 wait 之上"（V3），在 HEAD 上也只有
252 t/s，而不是 920 t/s**。920 这个数字在本树上复现不了（见第 4 节）。

## 3. 根因（实测数据）

`GGML_SCHED_TRACE=3` 的 per-copy 分解（PP512，每次 eval 81 次 host-weight 拷贝）：

| 变体 | gate 拷贝 body | gate 拷贝 sync | up/down body | up/down sync | copy 总计 |
|---|---:|---:|---:|---:|---:|
| V0 baseline | 0.8 ms | 20.4 ms | ~0 | 10.6-11.3 ms | 1.23 s |
| V2 slot，copy 后 drain | **34.4 ms** | 8.1 ms | ~0 | 8.3-11.0 ms | 1.69 s |
| V4 不 drain | **60 ms** | ~0 | ~0 | ~0 | 1.66 s |

- gate 拷贝 = 每层第一个 host-weight 拷贝；`llama_context::sched_copy_experts`
  （`src/llama-context.cpp:2837`，本任务写作用域之外）在 ids 变化时做
  `ggml_backend_tensor_get_async(ids)` + **`ggml_backend_synchronize(backend)`**，为的是从
  设备读回本轮 routing ids 才能决定要拷哪些 expert。
- 基线里这个内部 sync 便宜（0.8 ms），因为 copy_input 顶部的 `wait_backend` 已经把设备排空。
  一旦为了流水跳过顶部 wait，这个内部 drain 就要吞掉所有在途工作：V2 34 ms/次 x 27 层 =
  0.93 s/eval；V4 60 ms x 27 = 1.6 s/eval。**这就是流水化在这个 scheduler 里不划算的原因。**
- 2-slot 本身、以及 copy 之后的 drain 都不构成障碍（slot 内存代价 420 MiB，实测无 OOM）。

## 4. 为什么 920 t/s 不可信/不可复现

- 本树 HEAD 上把 callback 提到 wait 之前（保留 wait，V3）实测 252 t/s；不保留 wait（V4）223 t/s。
  两者都远低于 346，更不用说 920。910-920 需要 0.55 s/eval，即 13.1 GiB 拷贝在 0.55 s 内完成
  （24 GB/s，x16 链路可以）**并且** 27 次内部 drain 完全不发生。这与当前 callback 的实现矛盾。
- 更可能是当时的 racy 版本在竞争下**少拷了 expert**（ids 未就绪 -> `st.used` 偏少 -> 上传量变小），
  或者 base 4cf3d44a9 的图布局不同。需要 author 原始 patch 才能定论。

## 5. GPU 计算上界（实测）

`GGML_VK_PERF_LOGGER=1`，PP512 整轮（2 次 eval）GPU op 时间合计 = 1.54 s -> **0.77 s/eval**
（占 1.48 s eval 的 52%）。另外：
- host-weight 上传 = 27 层 x 498 MiB = 13.1 GiB/eval；实测 12-14 GB/s（x16 链路 24 GB/s 上限）；
- 每 eval 串行链 = 27 层 x [attention(A) + gate/up/down 的 upload+compute]。

即使上传与 compute 完全重叠，PP512 上界 ≈ max(0.77 s GPU, 0.55 s 上传) + 依赖串行 ≈ 0.85-1.0 s
-> **PP512 约 510-600 t/s**，除非降低 GPU 计算量或上传量；PP1000 需要 0.51 s/eval，
低于当前 GPU 计算时间 0.77 s，仅靠调度器不可能达到。

## 6. 建议的下一步（按收益排序）

1. **（前置，收益最大，不在本任务作用域）移除 prefill 的 ids 回读**：
   `llama_context::sched_copy_experts` 在 `ubatch >= 32` 时 512 个 token 的 top-8 通常覆盖全部
   256 个 expert，直接整张量上传即可，不需要读 ids。这样每 eval 省掉 27 次
   `get_async + synchronize`（基线实测 27 x ~20 ms 的 drain，约 0.5 s/eval），并且让拷贝不再有
   设备侧依赖，调度器才有机会做 upload/compute 重叠。
2. **（依赖 1）重新应用双缓冲 slot 流水**：本文件第 2/3 节的 2-slot 设计已验证正确性机制可行
   （内存 +420 MiB），预计 PP512 510-600 t/s；tg32 无 host-weight 拷贝，不受影响。
3. **（独立）GPU MoE GEMM 效率**：MUL_MAT_ID 在 PP512 上 2.6-3.1 TFLOPS（稠密 MUL_MAT
   8.5-9.6），占 GPU 时间 47%；提升它直接降低 0.77 s 的 GPU 下界。

## 7. 原型关键设计（供后续 task 复用）

- sched 新增：`hw_pipeline`（env `GGML_SCHED_HW_PIPELINE`，且 `!parallel`）、每 backend
  `hw_buf`（`ggml_backend_buft_alloc_buffer(buft, 2*align_up(max_host_weight_bytes))`）、
  `hw_buf_size/stride`、`hw_slot`。
- `ggml_backend_sched_split_graph` pass 5：记录每个 backend 的最大 host-weight 输入字节数，
  结束时 reserve slot buffer（增长前 synchronize 所有 backend）。
- `copy_input`：仅当 `is_host_weight(input)`、split 只有一个 host-weight 输入、
  `callback_copy` 存在时，把 `input_cpy->buffer/data` 指到 `slot = hw_slot`，调 callback，
  成功则 `hw_slot ^= 1`（并选择 drain 或 event 等待），失败则恢复原指针走老路径。
- **依赖规则（为保证正确性必须同时满足）**：
  1. 拷贝只写专用 slot，绝不写 gallocr 复用区（否则 upload 会与上一 split 的读竞争 - 这就是
     task-9 文本损坏的原因）；
  2. slot 复用前必须等"上次消费该 slot 的 split"完成（2-slot = 等两次拷贝之前的那个 split）：
     事件版本用 split compute 后 record 的 event；
  3. upload 必须在消费它的 split 的 compute 之前被提交（同一 backend 的 set_tensor_async +
     graph_compute 顺序保证）；
  4. **额外约束（本次发现的真瓶颈）**：host-weight 拷贝不能早于产生 routing ids 的 split
     完成，否则 callback 内部会 drain 全部在途工作。

## 8. 复现命令

```
# 基线
llama-bench -m <model> -dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -t 5 -p 512 -n 32 -r 3
# 逐次拷贝分解
GGML_SCHED_TRACE=3 llama-bench ... -p 512 -n 0 -r 1
# GPU 时间上界
GGML_VK_PERF_LOGGER=1 llama-bench ... -p 512 -n 0 -r 1
# 原型（已回退，重新实现后）
GGML_SCHED_HW_PIPELINE=1 llama-bench ... -p 512 -n 0 -r 1
```

原始日志：`bench/moe-baseline/T10-*.{out,err}`（T10-base*, T10-smoke*, T10-hybrid*, T10-racy,
T10-nodrain, T10-vk, T10-restored）。
