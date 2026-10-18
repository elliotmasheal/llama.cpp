# L10: 目标化 wait 原语已实现且很便宜，但重叠在依赖链上没有空间（2026-10-18，task-16）

> 分支 `moe-gpu-cpu-hybrid` @ `9eae2caac`。结论：**未保留代码**。按任务 STOP 条件给出的有效最终答案：
> **这台 8 GiB 机器在 cache 配置下是 compute+link bound，pp512 383-386；static 约 359。**
> 原因是依赖链而非实现成本：原语实测只要 **0.028 ms/call**（远低于 2 ms 的停止阈值），
> 但把全设备 drain 换成目标化 wait 后仍然 236 t/s（更慢），因为层与层之间根本没有可重叠的空隙。

## 1. 原语（已实现、已验证、已回退）

- `ggml/src/ggml-backend.cpp`：每个 split 提交后 `ggml_backend_event_record` 一个 per-backend
  completion signal（timeline semaphore）；新增
  `ggml_backend_sched_wait_split(sched, backend)`（host 侧 `event_synchronize` = 只等"最后提交的
  split"，不等已排队但未提交的拷贝）和 `ggml_backend_sched_hw_ring_free(sched, backend)`
  （由回调在确认所有已排队 split 完成后释放 ring epoch）。
- 成本实测（`GGML_SCHED_TRACE=1` 新增 hw_sig 计时）：**2.55 ms / 90 splits = 0.028 ms/call**，
  pp512 378.9 vs 382-383（-1%）。原语本身**不是**瓶颈。
- 正确性验证：wait 后 `cpy_sync` 从 1286 ms 降到 **44 ms**，说明它确实没有 drain 掉已排队的拷贝；
  ids 回读改为同一 compute queue 上的同步读（`ggml_backend_tensor_get`，临时 ctx + 自身 fence），
  在"等最后 split"之后只会等这一小段拷贝，顺序正确。

## 2. 但叠上 ring 之后仍然更慢（233-236 vs 368）

`GGML_SCHED_HW_SIGNAL=1 GGML_SCHED_HW_PIPELINE=1`，cache 5120，r2：pp512 = 233-236；
copy body（回调内部时间）仍约 2.1 s，`cpy_sync` 只有 44 ms。逐层看：ids 回读（routing split）
和上传之间没有任何空隙可藏，因为——

## 3. 结构性原因：依赖链让层间重叠不存在

prefill 时每个 CPU-MoE 层的顺序依赖是：

    MoE(L-1) 输出 -> routing(L) 产生 ids -> 上传 L 的权重 -> MoE(L) 计算

- `routing(L)` 必须等 `MoE(L-1)` 算完（hidden state 是 MoE 输出），所以 ids 不可能更早拿到；
- 上传 L 必须等 ids(L)（否则不知道传哪些专家）;
- MoE(L) 必须等上传 L。

因此链上没有任何 slack：wait(routing L) 本身就等于 wait(MoE L-1)，而 compute queue 是顺序执行的。
唯一存在的空隙是**层内**：ids 拿到后，up/down 两个权重的上传（约 2/3 字节）可以和 gate 的 compute
（约 1/3 计算）重叠，理论上限约 17%（383 -> ~450 的理想值），而实现代价（目标化 wait + 安全 ring +
ids 同步读 + 每次 wrap 的回退）实测超过了这点收益（233-236）。

这也彻底解释了 L8 的 racy 797：它让 compute 跑在"routing 还没算完"的数据上，即拿垃圾权重做 GEMM，
快的不是重叠，而是没有正确等待。

## 4. 结论（有效最终答案）

- 原语可用且便宜（0.028 ms/call），以后若换更大显存或改掉"上传哪些专家"的判定方式（例如在 GPU 上
  直接读 host 权重）可以复用。
- 在当前模型/后端/8 GiB 硬件上，prefill 的 PP 上限就是 **cache 383-386 / static ~359**：
  上传 7.3-8.0 GiB/eval 已贴 13.9 GB/s 的 H2D 上限（L8），计算侧已由 L7 的 GEMM 优化压过一轮，
  两者之间的串行由模型的数据依赖决定，不是调度器能重排的。
- 回退后基线复测：cache 6144 pp512 **386.1**、tg32 **26.39**；`git status` 对 src/ggml/include 干净。

## 5. 复现

```
# 原语成本（已回退，代码见 L10 描述）
GGML_SCHED_HW_SIGNAL=1 GGML_SCHED_TRACE=1 llama-bench ... -p 512 -n 0 -r 2 -ncmoe 41 --moe-cache-mib 6144 -ub 2048
# 目标化 wait + ring
GGML_SCHED_HW_SIGNAL=1 GGML_SCHED_HW_PIPELINE=1 llama-bench ... -p 512 -n 0 -r 2 -ncmoe 41 --moe-cache-mib 5120 -ub 2048
```

原始日志：`bench/moe-baseline/T16-*.{out,err}`。
