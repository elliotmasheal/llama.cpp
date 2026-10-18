# L5: prefill 上传下限实测 - ids 回读不是瓶颈（2026-10-18，task-11）

> 分支 `moe-gpu-cpu-hybrid`，起点 `7dbc38834`。目标：去掉 prefill 的 ids 回读（llama-context）+ 2-slot
> 拷入流水（ggml-backend），pp512 >= 480。结论：**两步都实测为负收益，已全部回退**，
> `git status` 对 src/ 与 ggml/ 干净；基线复测 pp512 346.8、tg32 18.7。
> 真正卡住 PP 的是 "专家上传字节 / PCIe 带宽"，且本栈上 transfer queue 与 compute queue 不重叠，
> 所以调度器流水没有收益空间。

## 1. 核心发现：ub512 的专家覆盖率只有 70%，不是"几乎全部"

在 `sched_copy_experts` 里临时统计 `st.used[]`（`LLAMA_EXPERT_COVERAGE=1`，已回退），
单次 pp512（ub=512）27 个 CPU-MoE 层、27 次统计：

| 指标 | 值 |
|---|---:|
| 实际被路由到的 expert | **平均 179.2 / 256 = 70.0%** |
| 最少 / 最多 | 122 / 253 |
| token 数 | 512（top-8 = 4096 次选择） |

所以"512 token 的 top-8 覆盖全部 256 个 expert"的假设不成立（该模型路由很偏斜）。
`-ncmoe 27` 现有按"实际用到"拷贝 = 约 10.4 GiB/eval；改成整张量拷贝 = 13.1 GiB/eval，**+43% 字节**。

## 2. 实测：只做 step 1（去 ids、拷整张量）

| 配置 | pp512 | tg32 | sched_trace copy |
|---|---:|---:|---:|
| baseline（现状） | 346.8-347.3 | 18.7 | 1.23 s |
| step1：跳过 ids 回读，整张量上传 | **294.2** | 18.7（decode 无 host-weight 拷贝，不变） | 1.45 s |

原因：上传字节 +43%，而 prefill 时间几乎全部是上传（见第 4 节串行模型）。**因此 step 1 单独就是 -15% 的退化，已回退。**

## 3. 实测：step 2（2-slot 轮换目的地 + event 门控复用）叠加 step 1

实现（已回退，详见 L4-HOST-WEIGHT-PIPELINE.md 的同一设计）：每个 backend 分配
2 x max(host weight bytes) = 2 x 210 MiB = **420 MiB VRAM** 的 slot buffer；拷贝写入 slot；
写入 slot 前 host-wait "上次消费该 slot 的 split" 的 event（该 split compute 后 record）；
拷完不 drain；权重 tensor 的 `buffer/data` 在 copy 前指到 slot。

| 配置 | pp512 | pp2048 ub2048 |
|---|---:|---:|
| baseline | 346.8-347.3 | 577.9 |
| step1 | 294.2 | - |
| step1 + slot 流水 | **215.1** | **177.9** |
| step1 + slot 流水（诊断：不等 event） | 237.9 | 186.3 |

全部显著退化。原因（都实测）：

1. **没有拿到任何上传/compute 重叠收益**。观察到的耗时与"串行和"一致：
   - pp512：按用到 expert 的上传量 10.4 GiB，若按 13.0 GB/s 算是 0.86 s；copy 实测 1.23 s，
     余下 ~0.37 s 与 GPU compute 量级相符。
   - pp2048 ub2048：copy 实测 3.56 s；若上传 13.1 GiB/13.0 GB/s = 1.08 s，余下 ~2.48 s 为 compute。
   - step1：上传 13.1 GiB/13.0 GB/s = 1.08 s + ~0.37 s = 1.45 s = 实测值。
   两种 batch 下**没有任何变体低于串行和**；注意 U 与 C 没有独立测出（perf logger 的
   "Total time" 汇总会重复累计、不可当每 eval GPU 时间用），所以这里只说"实测没有重叠收益"，
   不下"硬件不能重叠"的强结论。链路本身是 PCIe Gen4 x16（`CurrentLinkWidth=16`），
   host->device 实测约 13 GB/s（step1 的 13.1 GiB/1.08 s），上传侧已接近实际上限。
   `vulkaninfo` 显示本机确有独立 transfer-only queue family（GPU1: family2, count=2）与
   compute family（family1, count=4），但本任务没有观察到它们重叠带来的收益。
2. **per-split event_record 代价高**。流水需要每 split record event（= 结束并提交 compute ctx），
   破坏了 Vulkan backend 的跨 split 命令缓冲/描述符集复用：ub2048 时 `gpu_submit`
   从 baseline 的 ~20-30 ms 涨到 **426-465 ms/eval**。这是 backend 级开销，不是调度器能绕开的。

补充：decode（tg32）没有 host-weight 拷贝（hw=0），两步对它都没有影响（18.7 不变），
但也没有收益。

## 4. 当前实现的 PP512 实测下限

PP512 实测 344-347 t/s（copy 1.23 s，其中按用到 expert 的上传 10.4 GiB；上传按 13 GB/s
已接近 PCIe Gen4 x16 的 host->device 实际上限）。要达到 480 需要 prefill 时间 <= 1.07 s：
在"每次 eval 都要把 10.4 GiB 专家权重从 host 搬到 VRAM、且搬与算没有可观测重叠"的前提下，
只能靠**减少上传字节**（驻留）或**减少 compute**，调度器重排无效。

## 5. 下一步杠杆（测量：MUL_MAT_ID GEMM 效率）

数据来自 `GGML_VK_PERF_LOGGER=1`（PP512 ub512 / PP2048 ub2048，本机 Vulkan1）：

| 算子（shape） | batch | 单次耗时 | TFLOPS |
|---|---:|---:|---:|
| MUL_MAT_ID q4_K gate/up, m=512 n=8 k=2048, 256 experts | 512 | 3.7-4.4 ms | **2.0-2.4** |
| MUL_MAT_ID q6_K down, m=2048 n=8 k=512, 256 experts | 512 | 4.6 ms | **1.87-1.87** |
| MUL_MAT_ID q4_K gate/up, same shape | 2048 | 6.4-6.5 ms | **5.3-5.4** |
| MUL_MAT_ID q6_K down, same shape | 2048 | 7.7 ms | **4.43** |
| 稠密 MUL_MAT q4_K m=8192 n=512 k=2048（共享专家/稠密投影） | 512 | 1.0 ms | **8.6** |
| 稠密 MUL_MAT q4_K m=8192 n=2048 k=2048 | 2048 | - | **9.5-9.6** |

要点：
- batch 512 时 MoE GEMM 只有稠密的 **~27%**（2.3 vs 8.6 TFLOPS）；batch 2048 时约 55%（5.3 vs 9.6）。
  prefill 的 compute 主要就是这些 MoE GEMM（PP512 的 C≈0.37 s 中约 123 ms 是 gate/up/down）。
- 若 batch 512 的 MoE GEMM 提到稠密效率，C 可降到 ~0.28 s -> PP512 ~390（仍不到 480）。
- **PP1000 需要 0.51 s/eval**：仅按用到 expert 的上传就要 0.86 s，因此在"每次都从 host 上传"
  的前提下 PP1000 不可能；必须先让一部分专家 prefill 时也常驻 VRAM（把 U 砍到 <0.3 s），
  再配合 GEMM 效率提升才可能。

## 6. 建议的下一步（按可行性）

1. **让 prefill 也用上 VRAM 驻留（cache 覆盖 prefill 的 used-expert 拷贝）**：当前
   `--moe-cache-mib 6144` 的 pp512 只从 346 到 356（+3%），说明 prefill 并没有真正省下上传字节；
   把 `llama_moe_cache` 的 resident 判定接进 `sched_copy_experts` 的逐层拷贝（resident 的不上传），
   U 每减少 1 GiB 约省 77 ms（PP +~20 t/s）。这是目前唯一能明显推动 PP 的杠杆。
2. **MUL_MAT_ID kernel**：batch 512 下 2.0-2.4 TFLOPS（稠密 8.6），小 m 的专家 GEMM 需要
   tile / split-k / repack 方向优化；GEMM 达到稠密效率可省 ~0.09 s/eval（PP +~2 0 t/s）。
3. **后端队列重叠**：本机确有独立的 transfer-only 队列，但本任务没有测到上传与 compute 重叠的收益
   （所有流水变体都更慢）。若要验证/利用它，需要在 ggml-vulkan 里单独做提交编排实验
   （例如把多个 split 的 upload 合批提交、或用精确 fence 编排），超出本任务作用域。

## 7. 复现命令

```
# baseline
llama-bench -m <model> -dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -t 5 -p 512 -n 32 -r 3
# 串行模型 / copy 归因
GGML_SCHED_TRACE=1 llama-bench ... -p 512 -n 0 -r 1
GGML_SCHED_TRACE=1 llama-bench ... -p 2048 -n 0 -ub 2048 -r 1
# 专家覆盖率（临时插桩，已回退）
LLAMA_EXPERT_COVERAGE=1 llama-bench ... -p 512 -n 0 -r 1
# GEMM 效率
GGML_VK_PERF_LOGGER=1 llama-bench ... -p 512 -n 0 -r 1
GGML_VK_PERF_LOGGER=1 llama-bench ... -p 2048 -n 0 -ub 2048 -r 1
```

原始日志：`bench/moe-baseline/T11-*.{out,err}`（smoke, step1, nowait, coverage, p2048-*, restored, final-base）。
