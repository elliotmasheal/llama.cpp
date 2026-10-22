# LFM2.5-24B-A2B (UD-Q4_K_XL) 在 RX 6700S 8 GiB 上的性能实测与最优配置

> 最终报告（取代 `REPORT-DRAFT.md` 的 TBD 部分）。
> 机器：RX 6700S 8 GiB (Vulkan1) + Ryzen 7 6800HS 8C/16T + 39.2 GiB RAM, Windows 11。
> 构建：`E:\fastLlama\build-vulkan\bin`（Release/Ninja、`GGML_VULKAN=ON`、build 11750、commit `a9d2cc8a7`）。
> 模型：`F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`（13.855 GiB，428 张量）。
> 所有数字来自 `bench/lfm25/results/*.csv`（每个文件一个配置，文件名即配置 tag），
> 原始命令见 `cases-r*.txt`，执行器见 `run-sweep.ps1`，汇总见 `dump.py`。

## 0. 结论（TL;DR）

**综合最优（PP 与 TG 兼顾、单一命令）**

```
-m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf \
-dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 \
-fa on -t 6 -ub 2048 -b 2048
```

| 指标（项目口径） | 实测 | 说明 |
|---|---:|---|
| pp2048 `-p 2048 -n 0 -ub 2048 -r 3` | **874.3 t/s** | 长 prefill 主口径 |
| pp512 `-p 512 -n 32 -r 3` | 530.3 t/s | 短 prefill 对照 |
| tg128 `-p 512 -n 128 -r 3` | **31.35 t/s** | decode 主口径 |
| tg32 `-p 512 -n 32 -r 5` | 29.5 t/s | 与历史 tg32 对齐 |
| pp4096 / pp8192 | 837.0 / 753.3 t/s | 长上下文 prefill |
| 显存峰值（device 专用） | 8143 / 8176 MiB | 接近上限，见第 6 节风险 |

- 该配置 = "4 个专家层静态驻留显存 + 5120 MiB 专家 LRU bank + ub 2048 + 6 线程"。
- 相对全 CPU（tg 6.6 t/s）TG 提升 4.8x；相对最优纯静态（tg128 22.8）提升 37%。
- 注意口径：tg32/tg128 是 llama-bench 口径（其 decode 循环喂随机 token），会高估 MoE cache 配置的 decode。
  真实贪心生成（`llama-completion -no-cnv --temp 0`，1203 token prompt）为 24.4 t/s，
  官方 llama.cpp 同条件 24.8 t/s（持平）；见 `COMPARE-OFFICIAL.md` 与第 7.5 节。
- 项目目标 PP1000+/TG200+：PP 已达 **87%**（874/1000）；TG 仅 **15.7%**（31.4/200），
  单靠配置调参无法到 200，需投机解码等 runtime 改动（第 7 节）。

**备选（不要 cache，显存余量更大 / 更保守）**

```
-dev Vulkan1 -ngl 99 -ncmoe 22 -fa on -t 6 -ub 2048 -b 2048
```

| 指标 | 实测 |
|---|---:|
| pp2048 (r=3) | 878.1 t/s |
| pp512 (r=3) | 526.0 t/s |
| tg128 (r=3) | 22.83 t/s |

PP 与综合最优相同（差 0.4%），TG 低 27%。适合不想承担 cache 显存风险、或长上下文需要更多余量的场景。

**关键"不要做"清单**

| 做法 | 后果 | 证据 |
|---|---|---|
| `-ub 512`（默认）跑长 prompt | cache/mixed pp2048 只剩 469（应 874）；static ncmoe20 431 | `r3_pp2048_c_m5120_ub512`, `r3_pp2048_s20_ub512` |
| `-lm mmap` | pp512 530 -> 301，tg 29.5 -> 18.8 | `r8_mmap_mixed` |
| `--moe-cache-mib > 5120` 配 `-ncmoe 36` | m5632 时 pp512 崩到 142（显存溢出到共享内存） | `r4_c36_m5632_t6` |
| `--moe-cache-mib >= 5888` 配 `-ncmoe 40` | 单 VkBuffer > 2 GiB，直接分配失败 | `r3_c40_m5888_t6.err` |
| `-ncmoe <= 19` | 静态驻留 21 层，显存溢出，pp/tg 双降 | `r2_ncmoe19` |
| `-ncmoe 20` + `-ub 2048` | 静态驻留 20 层把显存吃满，pp2048 掉到 193 | `r3_pp2048_s20_ub2048` |
| `-t 8`（bench 默认数学线程数） | 比 `-t 6` 低 4-10% | `r2_thr6` vs `r1_ngl99_ncmoe22` |
| `-t 12/16` | tg 再降 10-17% | `r2_thr12`, `r2_thr16` |
| 双设备（iGPU Vulkan0 + dGPU） | pp 341-354，tg 19-23，全面更差 | `r8_dual` |
| `--moe-hybrid` | 源码自带 "experimental, about 2x slower" 警告，未采用 | `src/llama-context.cpp:617-628` |

## 1. 机器与模型事实

### 1.1 机器

| 项 | 值 |
|---|---|
| dGPU | AMD Radeon RX 6700S，8176 MiB，`gfx1032`（RDNA2 28 CU，无 matrix core，有 int dot） |
| iGPU | AMD Radeon(TM) Graphics（`Vulkan0`，UMA 24 GiB）——基准不使用 |
| CPU / RAM | Ryzen 7 6800HS（8C/16T）/ 39.2 GiB |
| 系统盘 | F: Micron 2450 NVMe SSD |
| 构建 | `E:\fastLlama\build-vulkan`，Release/Ninja，Ninja 全量重建于 2026-10-21 |

### 1.2 模型（GGUF header 解析）

| 项 | 值 |
|---|---|
| arch | `lfm2moe`，`general.size_label = 64x1.4B`，23.97 B 参数 |
| 文件 / 张量 | 14,876,901,632 B = 13.855 GiB，428 张量 |
| block | 40，前 2 个 dense（`leading_dense_block_count=2`），后 38 个 MoE |
| MoE | 64 专家 / 每 token 用 4，`expert_ff=1536`，门控 sigmoid |
| 注意力 | 仅 block 2,6,...,38 共 10 层（32 heads / 8 KV heads / head_dim 64） |
| shortconv | 其余 28 层（`l_cache 3`），FFN 同样是 MoE |
| 专家权重 | 13371 MiB（13.06 GiB，占 94.3%）；36 层各 348.0 MiB，block 7/8 各 421.5 MiB；单专家 5.44 MiB |
| 非专家权重 | 808.8 MiB |
| KV | 20 KiB/token（f16）；ctx 2048 约 41 MiB，ctx 16384 约 320 MiB |
| 上下文 | 131072 |

## 2. 测试方法与口径

- PP 主口径：`llama-bench -p 2048 -n 0 -ub 2048 -r 3`（pp2048）；pp512 作对照。
- TG 主口径：`llama-bench -p 512 -n 128 -r 3`（同条命令附带 pp512）；tg32 作历史对齐（r=5）。
- 所有配置统一加：`-m <model> -dev Vulkan1 -ngl 99 -fa on`；另有说明除外。
- 每配置单独进程（避免前一组的路由/缓存状态污染）；扫描阶段 r=2 定位趋势，决赛 r=3/r=5 确认。
- 测量期间 GPU 上无其它推理进程；早期两次与其它 bench 并发的数据（`r2_mmap`、`r2_c30_m4096`、`lfm_c30_m4096_vram`）已作废并用干净重跑替代。
- 峰值显存用 `bench/moe-baseline/monitor.ps1` 采样 `GPU Process Memory`（device 专用）。
- 交叉核对：同配置 r=2 vs r=3 vs r=5 的偏差在 1-6%（tg32 29.5 r5 / 29.05 r2；tg128 31.17 r5 / 31.35 r3）。

## 3. 实测数据

### 3.1 静态放置（`-ncmoe` 扫描，无 cache，t=8，r=2，`-p 512 -n 32`）

驻留专家层 = 40 - ncmoe（前 2 个 block 无专家张量，故 `-ncmoe N` 实际把 block 0..N-1 的专家放 host）。

| ncmoe | 驻留层 | pp512 | tg32 |
|---:|---:|---:|---:|
| 40 | 0 | 382.8 | 13.45 |
| 38 | 2 | 386.6 | 14.16 |
| 36 | 4 | 392.5 | 15.28 |
| 34 | 6 | 421.2 | 15.14 |
| 32 | 8 | 431.2 | 16.62 |
| 30 | 10 | 452.1 | 17.30 |
| 28 | 12 | 473.2 | 17.36 |
| 26 | 14 | 485.4 | 18.59 |
| 24 | 16 | 503.5 | 18.79 |
| 22 | 18 | 520.8 | 20.54 |
| 21 | 19 | 532.2 | 21.49 |
| **20** | **20** | **545.0** | **21.94** |
| 19 | 21 | 528.7（开始显存溢出） | 21.44 |
| 0（全 CPU） | - | 349.2 | 6.59 |

结论：显存能容纳 20 层专家（6.96 GiB），ncmoe 20 是静态边界；ncmoe 22 是"留余量"的安全点。

### 3.2 纯 cache（`-ncmoe 40`，全部专家在 host，t=6，r=2，`-p 512 -n 32`）

| cache MiB | pp512 | tg32 | 备注 |
|---:|---:|---:|---|
| 2048 | 423.5 | 11.95 | 命中太少 |
| 4096 | 470.1 | 22.20 | |
| 5120 | 492.8 / 492.9 | 25.82 / 25.13 | 稳定 |
| 5376 | 498.3 | 26.12 | |
| 5632 | 495.1 | 26.55 | |
| 5888 | 分配失败 | - | 单 VkBuffer 2091 MiB > 驱动上限 |
| 6144 | 分配失败（干净口径） | - | 同上；用 `GGML_VK_FORCE_MAX_BUFFER_SIZE=6442450944` 可分配，pp 516 / tg 28.92（r=2），仍不如 mixed |

### 3.3 混合放置（部分静态驻留 + cache，t=6，r=2，`-p 512 -n 32`）

| ncmoe | cache MiB | pp512 | tg32 |
|---:|---:|---:|---:|
| 40 | 5120 | 492.9 | 25.13 |
| 38 | 5120 | 507.6 | 27.77 |
| 38 | 5376 | 514.7 | 28.29 |
| 37 | 5120 | 520.1 | 28.60 |
| **36** | **5120** | **532.1** | **29.62** |
| 36 | 5376 | 525.7 | 28.64 |
| 36 | 5632 | 142.2（溢出） | 29.32 |
| 35 | 5120 | 536.7 | 29.08 |
| 34 | 4608 | 542.3 | 28.08 |
| 34 | 4096 | 529.1 | 27.88 |
| 32 | 3072 | 526.3 | 24.24 |
| 30 | 2048 | 520.8 | 19.10 |

最佳点：**ncmoe 36 + cache 5120**（4 层专家静态驻留 + 5120 MiB LRU bank）。再加 cache 或再减 ncmoe 都会越过显存边界。

### 3.4 线程（`-t`）

| 配置 | pp512 | tg32 |
|---|---:|---:|
| ncmoe22, t=5 | 538.8 | 23.26 |
| ncmoe22, **t=6** | 541.8 | **23.74** |
| ncmoe22, t=7 | 545.9 | 22.21 |
| ncmoe22, t=8 | 520.8 | 20.54 |
| ncmoe22, t=12 | 518.9 | 18.82 |
| ncmoe22, t=16 | 518.4 | 17.56 |

`-t 6`（8 核 16 线程的超线程外）最优；PP 对 t 不敏感，TG 敏感。

### 3.5 长 prefill 与 `-ub`（关键）

| 配置 | ub | pp | 备注 |
|---|---:|---:|---|
| mixed ncmoe36 + m5120 | 512 | 469.1 (pp2048) | |
| mixed ncmoe36 + m5120 | 1024 | 703.8 (pp2048) | |
| **mixed ncmoe36 + m5120** | **2048** | **878.4 (pp2048) / 837.0 (pp4096)** | 主结论 |
| mixed ncmoe36 + m5120 | 4096 | 162.7 (pp4096) | 溢出 |
| cache40 + m5120 | 512 / 2048 | 469.1 / 840.2 (pp2048) | |
| static ncmoe20 | 512 / 1024 / 2048 | 431.0 / 141.6 / 193.6 (pp2048) | ub>512 显存溢出 |
| static ncmoe22 | 1024 / 2048 | 698.3 / 879.4 (pp2048) | 留出余量后正常 |
| static ncmoe24 / ncmoe26 | 2048 | 863.2 / 849.5 (pp2048) | 驻留层越少 PP 越低 |
| mixed（f16 KV） | 2048 | 753.3 (pp8192) | |
| mixed（q8_0 KV） | 2048 | 749.9 (pp8192) / 618.4 (pp16384) | 长上下文可用 |

结论：`-ub 2048` 是长 prefill 的决定性参数（相比 ub512 提升 1.9x），但同时要求至少约 1.0 GiB 的显存余量；
静态 ncmoe20 没有余量，所以 ub 一大就溢出。

### 3.6 其它旋钮

| 旋钮 | 结果 |
|---|---|
| `-fa on` vs `off`（ncmoe22, t8） | pp512 516.2 vs 516.2，tg32 21.30 vs 20.54：短上下文差异在噪声内；长上下文/量化 KV 需要 `on`，建议固定 `on` |
| `-ctk/-ctv q8_0`（ncmoe22, t8） | pp512 518.3 / tg32 21.43，与 f16 相同；长上下文（>=16k）建议 q8_0 省显存 |
| `-lm mmap`（mixed，干净重跑） | pp512 301.3 / tg32 18.77，必须保持 resident（默认 auto） |
| 双设备 iGPU+dGPU（`-dev Vulkan0,Vulkan1 -ts 0.72,0.28`，全专家上 GPU） | pp512 341-354 / tg32 19.2-23.2，比单 dGPU 差 |

## 4. 机制解释（为什么是这个最优）

### 4.1 静态模式下 PP 与 TG 走不同路径

- 调度器只把"host 权重 + batch 足够大的算子"放上 GPU（`ggml-backend.cpp:1142-1148`，
  Vulkan 判定 `ggml_vk_get_op_batch_size(op) >= op_offload_min_batch_size`，默认 32，
  `ggml-vulkan.cpp:16245-16248`；MUL_MAT_ID 的 batch = token 数）。
- **TG（1 token）**：host 层的专家算子留在 CPU，直接从 RAM 读 4 个专家（每层约 21.75 MiB），
  不产生 H2D。所以静态模式 TG 的瓶颈是 CPU 专家 matvec（20 层 x 4 专家 x 5.44 MiB）。
- **PP（ub >= 32）**：算子 offload 到 GPU，调度器为每个 host 专家张量在 compute buffer 里
  建全尺寸 copy tensor（`ggml-backend.cpp:1547-1568`），再把用到的专家切片填进去
  （`llama-context.cpp:2854-2952`）。因此 PP 额外吃 `3 x 348 MiB` 级的显存 + 随 ub 增长的激活。
- 这解释了两个现象：(a) `-ncmoe 36 cache 5632` 的 pp512 崩到 142 而 tg32 仍有 29.3；
  (b) 静态 ncmoe20 在 ub>=1024 时 pp 掉到 141-193（分配成功但实际溢出到共享内存）。

### 4.2 为什么混合（4 层静态 + cache 5120）是 TG 最优

- 静态驻留的 4 层：命中率 100%，无 slot 查表、无 H2D。
- cache 覆盖其余 34 个 host 层：slot 数只由 cache 字节决定（每 slot 5.4375 MiB），
  层数越少，每层能分到的 slot 越多。按源码公式复算（`mixed_slots.py`）：
  - ncmoe36 + 5120 MiB：每层约 20.5/64 专家常驻 + 5 层整层驻留 -> decode 命中约 74/152 个专家，
    每 token 缺失约 424 MiB 需 H2D；
  - ncmoe40 + 5120 MiB：命中约 58/152，缺失约 511 MiB。
- H2D 实测天花板 13.9 GB/s：511 MiB/token 对应 24.9-26.0 t/s（实测 cache-only 28.1，已到墙），
  424 MiB/token 对应 30.2-31.2 t/s（实测 mixed 31.35，命中模型基本吻合）。
- 结论：**TG 的上限由"每 token 缺失专家字节 / H2D 带宽"决定**，在 8 GiB 显存下
  31-32 t/s 就是本方案的物理上限；再快需要减少缺失字节（更大 bank 不可得、模型文件重排/预取）
  或绕过 H2D（例如让 miss 专家在 CPU/iGPU 上并行计算）。

### 4.3 为什么 `-ncmoe >= 35` 反而 PP 更高

驻留层越多，PP 时要做的 host->device copy tensor 越少、H2D 字节越少；ncmoe 35-36 附近是
"少 host 层 + 显存仍够 ub2048" 的平衡点。ncmoe 34 以下虽然 PP 略高（542），TG 明显下降。

### 4.4 显存边界

- 总 8176 MiB，空闲 7378 MiB（桌面占用约 800 MiB）。
- 冠军配置：cache 5120 MiB + 4 层专家 1.39 GiB + 非专家 0.81 GiB + compute/激活
  -> 实测 device 峰值 **8143 MiB**（几乎顶满）。
- 单个 VkBuffer 上限约 2 GiB（驱动 `maxBufferSize`，`ggml-vulkan-buffers.cpp:52-54`），
  所以 cache >= 5888 MiB 时 down_exps bank 单次分配 > 2 GiB 直接失败；
  `GGML_VK_FORCE_MAX_BUFFER_SIZE=6442450944` 可绕过，但 6144 的收益（28.9）也不如 mixed 5120。

## 5. 推荐命令

### 5.1 llama-bench（项目口径复现）

```powershell
# PP 主口径
.\build-vulkan\bin\llama-bench.exe -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf `
  -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048 -p 2048 -n 0 -r 3
# TG 主口径
.\build-vulkan\bin\llama-bench.exe -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf `
  -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048 -p 512 -n 128 -r 3
```

### 5.2 llama-server / 实际使用

```powershell
.\build-vulkan\bin\llama-server.exe -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf `
  -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 `
  -fa on -t 6 -ub 2048 -b 2048 -c 4096 --parallel 1
```

- 默认 `-ub` 是 512，不显式加 `-ub 2048` 会让长 prompt 的 prefill 掉到约 469 t/s。
- **务必加 `--parallel 1`**：server 默认 `--parallel 4`，实测同一请求 PP 从 676 掉到 196 t/s、TG 从 24.9 掉到 19.25 t/s。
- 不要加 `-lm mmap`；Windows 下默认 AUTO = resident，保持默认即可。
- 端到端校验（`verify-server.ps1` / `verify-server2.ps1`）：1203 token 的 prompt，
  `--parallel 1` 时 prefill 676 t/s、decode 24.8-24.9 t/s；HTTP 路径比 llama-bench 低约 15-20%（含采样与请求开销）。
- 上下文长度影响：同一配置 tg128 在 ctx 512 时 31.35 t/s，ctx 2048 时 30.04 t/s（`r8b_ctx2048_tg128`），
  长上下文 decode 会随 KV 增长继续下降。
- 长上下文（>= 16k）：追加 `-ctk q8_0 -ctv q8_0 -c 16384`（实测 pp16384 618 t/s、tg 28.5）。
  显存紧张时可把 `--moe-cache-mib` 降到 4096（tg 约 28.1）。
- 若追求显存余量/稳定性（例如同时跑其它 GPU 应用）：去掉 cache，用 `-ncmoe 22`，
  pp2048 878 / tg128 22.8。

## 6. 风险与未验证项

- 冠军配置显存峰值 8143/8176 MiB，几乎没有余量：桌面/浏览器占用 GPU 时会溢出到共享内存
  （表现为 PP 掉到 100-200 t/s）。需要余量时把 cache 降到 4608/4096，或退回 `-ncmoe 22`。
- cache 的收益依赖"同一 prompt 的 decode 路由稳定"：单轮长生成时命中率高（tg128 31.35），
  但多轮对话每轮都会重新 prefill，命中率会周期性重建。实测多 prompt 场景未覆盖。
- `--moe-hybrid`（miss 专家交 CPU）源码自带 "experimental, about 2x slower" 警告，未纳入候选。
- 模型没有 MTP/nextn 头（GGUF 无 `nextn` 张量），投机解码只能走 ngram/独立 draft。
- 两个早期数据点在并发干扰下作废：`r2_mmap`（82/8.03）、`r2_c30_m4096`（179/28.11）；
  本报告已用干净重跑数据（mmap 301/18.77）。
- 未测：`--spec-type ngram-mod` 对本模型的收益、专家重排后 PP 的 GEMM 优化、iGPU 参与专家计算。

## 7. 距离项目目标的差距与下一步

| 目标 | 当前 | 差距 | 下一步 |
|---|---:|---:|---|
| PP1000+ | 874 (pp2048) | -12.6% | 剩差主要在专家 GEMM：ub2048 时 PP 已由算力而非上传决定；需要 MUL_MAT_ID kernel 调优（RDNA2 无 coopmat）或专家重排 |
| TG200+ | 31.4 (tg128) | -84% | 单靠配置无解（H2D 墙 + 每层 ids 回读/drain）；需要投机解码（ngram-mod / 独立 draft）把每步产出提到 2-4 token，或绕过 H2D 的 miss 计算路径 |

## 7.5 与官方 llama.cpp 的对照（补充，完整版见 `COMPARE-OFFICIAL.md`）

对照对象：scoop 安装的官方 `llama.cpp-vulkan`，`0.7.0-dev build 11763 / a01ce9ffc`（上游基底比 fork 新 13 个 build，Clang 构建），
同模型、同 `-dev Vulkan1`，并按同等强度调参（`-ncmoe` 扫描、`-t 5/6/8`、`-ub 512/2048`、`-lm auto/none/mmap`）。

| llama-bench r=3 口径 | 官方最优单配置 | fork static | fork mixed |
|---|---:|---:|---:|
| pp512 | 326.6 | 513.8 | 530.8 |
| pp2048 ub2048 | 739.9 | 878.3 | 874.8 |
| tg128 | 26.05 | 22.38 | 31.20 |

| 真实生成（`llama-completion -no-cnv`，temp 0，1203 token prompt，128 token） | 官方 | fork static | fork mixed |
|---|---:|---:|---:|
| prompt eval | 519.3 t/s | 739.3 t/s | 710.1 t/s |
| eval | 24.84 t/s | 22.61 t/s | 24.42 t/s |

| llama-server（1203 token prompt，128 token，`-c 4096 --parallel 1`） | 官方 | fork mixed |
|---|---:|---:|
| prompt | 261.4 t/s | 686.5 t/s |
| decode | 20.9 t/s | 19.7 t/s |

结论：
- **PP 改进确凿**：bench +18%（pp2048）/ +57~62%（默认 ub512），真实生成 +37%，llama-server 2.6x。
- **TG 改进只在 llama-bench 合成 decode 中出现**（+20%）；真实生成与官方持平（24.4 vs 24.8），
  cache 相对 fork 自己的 static 约 +8%。llama-bench 的 tg 循环喂随机 token，专家路由平稳，命中率被抬高。
- 官方同样必须 `-lm none`（默认 auto 慢 10%）、必须手工 `-ncmoe`（默认放不下）。
- server 场景两边都贴显存上限：fork 的 cache 配置在 server 下比 bench 低约 37%（官方低约 20%），
  上 server 时应把 `--moe-cache-mib` 降到 4096~4608 或退回 static 后单独复测。

## 8. 证据索引

| 文件 | 内容 |
|---|---|
| `SWEEP-PLAN.md` | 研究阶段的旋钮语义、源码行号、显存/带宽模型与扫描计划 |
| `cases-r1..r7.txt` | 各轮原始命令 |
| `results/*.csv` | 每个配置的 llama-bench CSV（文件名 = 配置 tag） |
| `results/*.err` | 失败配置的 stderr（含 OOM/2GiB 上限证据） |
| `results/lfm_mixed_champion_vram.summary.txt` | 冠军配置资源采样（GPU 峰值 8143 MiB、CPU 18.6%、RAM 20.2 GiB） |
| `run-sweep.ps1` / `parse.py` / `dump.py` | 扫描执行与汇总脚本 |
| `verify-server.ps1` / `verify-server2.ps1` | llama-server 端到端校验（HTTP 请求 + timings） |
| `mixed_slots.py` | cache slot/命中率公式复算 |
| `REPORT-DRAFT.md` | 研究阶段的详细机制分析（部分 TBD 已被本报告取代） |
