# L8: cache 配置的 H2D / bank 诊断（2026-10-18，task-14）

> 分支 `moe-gpu-cpu-hybrid` @ `ab7a05e6d`。结论：**没有保留任何代码改动**。
> 三条实测结论推翻了任务里的两个前提：(1) H2D 已经贴在硬件上限（13.3 / 13.9 GB/s = 96%），
> 任务里的 "10.1 GB/s" 是"总字节 / 含 compute 的 copy 时间"的商，不是链路速率；
> (2) bank 没有超订（峰值 8028 MiB < 总显存 8176 MiB），缩小 bank 只会更慢
> （4096→351、5120→368、5632→373、6144→382 单调）。
> 真正的杠杆被实测定位为 **upload 与 compute 的串行化**：去掉 copy 前的 drain（racy 诊断）
> pp512 从 382 直接到 **797**，即 0.55 s 上传可完全隐藏在 compute 下。

## 1. H2D 链路天花板（新建探针 bench/h2d-bw.cpp，未提交）

用 ggml 公共 API + Vulkan backend 测 pinned host -> device buffer 的纯拷贝（3 次 pass，取稳定值）：

| chunk | 总量 | 速率 |
|---:|---:|---:|
| 1.5 MiB x 171 | 256 MiB | 13.6-13.7 GB/s |
| 4 MiB x 128 | 512 MiB | 13.9 GB/s |
| 256 MiB x 1 | 512 MiB | 13.9 GB/s |

所以本机 PCIe Gen4 x16（`CurrentLinkWidth=16`, speed=4）上的**实际可用 H2D 上限约 13.9 GB/s**，
且与 chunk 大小无关（不需要改 chunk/batch 策略）。真实 prefill 的边际速率：
用 cache 5120/6144 两个配置做差 (1438.5-1325.5 ms)/(8805-7298 MiB) = **13.3 GB/s = 上限的 96%**。

## 2. bank 是否超订：没有

`GGML_VK_MEMORY_LOGGER=1`（cache 6144, pp512）：

- device 峰值 = **8028 MiB（7.84 GiB）**：权重 1303 + bank 6143 + compute 505 + RS 63 + staging 10。
- 卡总显存 **8176 MiB**（`Device memory: 8176 MB (7378 MB free)`，7378 是**加载权重前**的空闲值，
  不是总量）-> 8028 < 8176，**装得下**，没有 spill，也没有 write+evict 双倍流量。
- bank 大小扫描（pp512, r2）：4096 = 351.5, 5120 = 367.7, 5632 = 373.3, 6144 = **382.4**：
  单调增，没有"缩小 bank 反而更快"的拐点；再往上 6400 = 57.0、6528 = 56.6（崩），
  说明 6144 正好是这张 8 GiB 卡的满配。
- 线性拟合 copy = 777 ms + host_MiB x 0.075（4096: 1438.5 ms/8805 MiB；6144: 1325.5 ms/7298 MiB），
  两条线的截距**相同（778 / 777 ms）**，说明 6144 没有让 compute/bank-read 变慢，
  L7 观察到的"bank-read op 22.1 ms vs 3.9 ms"没有在 prefill 复现。

结论：**不应改 auto-fit**。若按 `free - weights - compute - 256 MiB`（7378-1303-505-256 = 5314 MiB）
自动缩小 bank，pp512 会从 382 掉到约 370（-12 t/s），是退化。

## 3. 真正的瓶颈：0.55 s 上传与 0.55 s compute 串行

- cache 6144：eval 1.33 s = copy 1.325 s（cpy_sync 1.276 s + h2d body 0.05 s）+ 其余 ~0.01 s。
  copy 里的 host 字节 7298 MiB/eval + bank D2D 889 MiB/eval。
- 每次 host-weight copy 之前 scheduler 都 `ggml_backend_sched_wait_backend()` 做全设备 drain，
  所以 GPU 是 [upload N -> compute N -> drain -> upload N+1 -> ...] 的严格串行。
- 诊断（**racy，已回退**）：在 `ggml_backend_copy_input` 里对 host-weight copy 跳过这个 drain
  （env `GGML_SCHED_HW_NOWAIT=1`），cache 6144 pp512 = **797 t/s（0.64 s/eval）**。
  0.64 s ~= max(0.55 s 上传, 0.55-0.6 s compute)，即上传可以被完全隐藏。
  这个数字不安全（会与上一个 split 的 compute 争用同一个 gallocr 目标），但它是**重叠收益的实证上限**。

安全实现需要 L4 的设计：专用轮换目标（2 x 210 MiB = 420 MiB VRAM，必须从 bank 里让出，
6144 -> 5632 会先损失约 9 t/s）+ event 门控复用（L5 实测 per-split `event_record` 约 3 ms，
90 splits 约 0.25 s）。保守估计 pp512 383 -> **480-550**，需要单独一轮实现与验证（task-15 提案）。

## 4. 为什么本轮不改代码

- auto-fit：退化（见第 2 节）。
- H2D 提速：已在 96% 上限，chunk/pinned staging 都不是瓶颈（13.6-13.9 与 chunk 无关）。
- overlap：需要 420 MiB VRAM + 每 split event record，属于调度器级改动；本轮剩余时间不足以
  完成"正确性优先 + 交叉 r>=3"的验证，按任务约束（revert neutral/unsafe）不落地。

回退后基线复测：cache 6144 pp512 **383.7**、tg32 **26.28**（与任务基线 383 / 26.41 一致）。
树相对 L7 只多了本文档（`git status` 对 src/ggml 干净）。

## 5. 复现

```
# H2D 上限（探针未提交，可重建：g++ 链接 build/ggml/src/*.a + vulkan-1.lib）
build/bin/h2d-bw.exe 1048576 268435456 0
build/bin/h2d-bw.exe 4194304 536870912 0
build/bin/h2d-bw.exe 268435456 536870912 0
# bank 扫描
llama-bench -m <model> -dev Vulkan1 -ngl 99 -fa on -t 5 -p 512 -n 0 -r 2 -ncmoe 41 --moe-cache-mib {4096,5120,5632,6144} -ub 2048
# 串行化归因
GGML_SCHED_TRACE=1 GGML_MOE_COPY_TRACE=1 llama-bench ... -p 512 -n 0 -r 1
# 重叠收益上限（racy 诊断，已回退）
GGML_SCHED_HW_NOWAIT=1 llama-bench ... -p 512 -n 0 -r 1
```

原始日志：`bench/moe-baseline/T14-*.{out,err}`。
