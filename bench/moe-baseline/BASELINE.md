# MoE 本地运行时 - 基线测试报告（第一阶段 / Baseline）

本文件是"GPU + RAM + SSD 协同运行本地 MoE"项目的**基线测试存档**。目的是在优化前，用可复现的方法
记录当前硬件/软件在目标模型上的真实表现、资源占用、以及各阶段的时间与带宽占比，作为后续每一项优化的对照基准。

> 复现脚本同目录：`monitor.ps1`（资源采样）、`run-case.ps1`（运行 + 采样 + 汇总）。
> 原始产物在 `results/`：每次运行的 `*.bench.log` / `*.metrics.csv` / `*.summary.txt`。
>
> 复现示例：
> ```powershell
> powershell -File .\run-case.ps1 -Tag run1 -Bin llama-bench `
>   -Args '-m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 27 -p 512 -n 128 -fa on -r 3 -o md'
> ```

---

## 1. 测试对象

| 项 | 值 |
|---|---|
| 模型 | `F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf` |
| 架构 | `qwen35moe`（hybrid：全注意力 + 线性注意力/SSM） |
| 规模 | 35.51 B 总参 / A3B 激活 |
| 量化 | Q4_K_M（Q4_K 15.23 GiB + Q6_K 4.90 GiB + F32 0.09 GiB） |
| 文件大小 | 20.21 GiB |
| 层数 | 41（`full_attention_interval=4` -> 约 10 层全注意力 + 31 层 SSM） |
| 专家 | 256 专家 / 8 激活，expert_ffn 512，shared expert 512 |
| MTP | `nextn_predict_layers=1`（有 MTP 头） |
| 注意力 | 16 头 / 2 KV 头，head dim 256 |

## 2. 硬件与软件环境

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 7 6800HS（8C/16T） |
| RAM | 39.2 GB |
| dGPU | AMD Radeon RX 6700S，8176 MiB VRAM（空闲约 7378 MiB），驱动 32.0.21045.5002 |
| iGPU | AMD Radeon(TM) Graphics（6800HS 核显，UMA 24 GB，与 CPU 共享内存带宽） |
| 存储 | Micron_2450 1TB SSD；模型位于 F: |
| 后端 | Vulkan（`ggml-vulkan`），`int dot: 1`、**`matrix cores: none`**（RDNA2 无 cooperative matrix） |
| 构建 | `cmake -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`（MinGW g++ 16.2） |
| 版本 | build `623c16393`（llama.cpp 0.7.0-dev） |

**环境要点**
- Vulkan 枚举到 **2 个设备**：`Vulkan0`=核显、`Vulkan1`=RX 6700S。基线必须 `-dev Vulkan1`，否则会跑到核显上。
- `ggml-vulkan` **未持久化 VkPipelineCache**，每次进程启动都需重新构建 shader（driver 可能另有盘级缓存）。
- 模型用 mmap 加载（`CPU_Mapped model buffer`）。39 GB RAM 可容纳 20 GB 权重，稳态下 host 专家应常驻 RAM。

## 3. Roofline 模型（字节级，来自实测 GGUF 张量）

| 分类 | 大小 | 占比 |
|---|---|---|
| MoE 专家（路由） | 18.65 GiB | 92.3% |
| attn | 0.61 GiB | 3.0% |
| output | 0.39 GiB | 1.9% |
| token_embd | 0.27 GiB | 1.3% |
| SSM + 其他 dense | ~0.28 GiB | 1.4% |

**每 token 激活字节** = 路由专家 8/256（0.58 GiB）+ dense（0.90 GiB）+ output（0.39 GiB）= **1.87 GiB**。

| 指标 | 理论上限（RX 6700S, 224 GB/s） |
|---|---|
| TG（裸，权重全在 VRAM） | **≈111 t/s** |
| TG200（等效） | 需 MTP 接受 ≈1.8 token/前向 |
| PP（纯带宽，B=512） | ≈5285 t/s |
| 一次前向读完全部专家 | 96.9 ms（20.21 GiB） |

> 关键结论：**裸 TG 物理上限 111 t/s，因此 TG200 只能靠 MTP 的等效吞吐达成**。

## 4. 测试方法

- 工具：`llama-bench`（PP/TG 标定）、`llama-cli`（缓存/端到端）、`test-backend-ops`（算子）。
- 统一参数：`-dev Vulkan1 -ngl 99 -fa on`；PP 固定 `-p 512`（缩放另测 2048）；TG `-n 128/256`；`-r` 多次取均值。
- 资源采样：`monitor.ps1` 每 1s 采集：CPU 占用、可用/已用内存、物理磁盘读/写 MB/s、GPU 引擎利用率、
  GPU 显存（dedicated/shared）、llama 进程工作集。
- 分阶段：PP-only（`-n 0`）与 TG-only（`-p 0`）分开跑，隔离两个阶段的活动与带宽。

## 5. 结果

### 5.0 一次基准运行的步骤与时间构成（用例 A，wall 66.3 s）

| 步骤 | 主要活动 | 时间 |
|---|---|---|
| 1 初始化 + shader | 枚举设备、构建 Vulkan 管线（无盘级 pipeline cache，首进程最久） | 若干秒（首次更久） |
| 2 模型加载 | mmap 20 GB、按 buffer type 分配、host 侧常驻 RAM | ~20–30 s（页缓存冷时受 SSD 限制，B/F 更久） |
| 3 资源预留 | `sched_reserve`、compute buffer、KV / recurrent 状态分配 | 秒级 |
| 4 预热 | 1 次前向 | — |
| 5 PP 512×3 | 每批将全部专家过一遍：GPU `MUL_MAT_ID` + CPU(27 层) MoE | ~10.7 s（3×3.55 s） |
| 6 TG 128×3 | 逐 token：output 投影 + 专家 GEMV + CPU MoE | ~27.1 s（3×9.04 s） |

> llama-bench 不直接报告 load time，上表加载时间为 `wall - 评测时间` 的反推值，仅作量级参考。

### 5.1 专家驻留扫描（"尽量上显存"）

`-ncmoe N` = 把 N 层的 MoE 专家放 CPU（其余层专家常驻显存）。N 越小，显存驻留越多。

| `-ncmoe` | 专家在显存的层数 | PP512 (t/s) | TG128 (t/s) |
|---:|---:|---:|---:|
| 41 | 0 | 114.8 | 11.16 |
| 34 | 7 | 64.9* | 12.39 |
| 31 | 10 | 78.5* | 13.22 |
| 30 | 11 | 133.2 | 13.46 |
| 29 | 12 | 146.0 | 13.41 |
| 28 | 13 | 148.5 | 13.70 |
| **27** | **14** | **145.9~148.7** | **14.25~14.36** |
| 26 | 15 | 150.8 | 13.59 |
| 25 | 16 | 150.9 | 13.13 |

\* 该两行为 `-p 256`，与其余 `-p 512` 不可直接比较。

**观察**
- PP 随驻留单调上升并在 `-ncmoe 26` 附近触顶（~151 t/s）。
- **TG 在 ~13–14 t/s 触顶**，且驻留过头（25）反而下降（疑似显存超配/走 shared memory）。
- 即使 14 层专家常驻显存、剩余 27 层在 CPU，TG 也只有 14 t/s —— 存在与"专家驻留"无关的**串行瓶颈**。

### 5.2 资源占用（稳态，来自 `results/*.metrics.csv`，去掉加载阶段）

| 用例 | wall (s) | CPU% avg | GPU util avg / max | 显存 peak ded+shared (MB) | 内存已用 peak (GB) | 磁盘读 avg (MB/s) | 进程 WS peak (GB) |
|---|---|---|---|---|---|---|---|
| A 合并 pp512/tg128 | 66.3 | 14.1 | 43.3 / 65.4 | 8499 + 1424 | 30.3 | 0.00 | 18.3 |
| B PP-only p2048 | 211.2 | 6.8 | 82.6 / 176.9* | 8533 + 1460 | 31.5 | 0.88 | 19.3 |
| C TG-only n256 | 77.6 | 18.8 | 43.3 / 46.1 | 8226 + 1311 | 28.9 | 0.00 | 16.9 |
| D TG profile | 35.8 | 13.7 | 27.4 | 8194 + 1279 | 25.1 | 0.00 | 13.1 |

\* `GPU Engine(*)` 利用率是按引擎累加的，可 >100%。

**关键读数**
- **CPU 与 GPU 都没有跑满**（TG 时 CPU≈19%、GPU≈43%）：时间是**延迟/串行**主导，不是算力或带宽饱和。
- **显存超配**：dedicated ≈ 8.5 GB（上限 8176 MiB）外加 shared ≈ 1.3–1.5 GB，说明驻留已顶到并溢出到共享（系统）内存。
- **磁盘读≈0**：稳态下 20 GB 权重常驻 RAM（进程 WS 16–19 GB），SSD 不参与。仅在冷启动/换出时才走盘。
- 内存已用峰值 ~29–31 GB（共 39 GB）。

### 5.3 分阶段时间与带宽归因

**TG（`-ncmoe 27`，实测 14.16 t/s -> 70.4 ms/token）**

按 roofline 拆分每 token 的字节来源：
- VRAM 服务：dense 0.90 + output 0.39 + 14/41×0.583 = **1.49 GiB**
- CPU（27 层 MoE）服务：27/41×0.583 = **0.38 GiB**

| 路径 | 字节/token | 有效带宽 | 占峰值(224 GB/s) |
|---|---|---|---|
| VRAM（含 output） | 1.49 GiB | 1.49/0.0704 = **21.2 GB/s** | ~9% |
| CPU MoE | 0.38 GiB | 5.4 GB/s | n/a |

GPU 侧单 kernel 并不慢（见 5.5，output 投影达 216 GB/s），所以 9% 的差距来自**kernel 之间的空隙与串行**，
而非单个 kernel 效率。**TG 现状 = "CPU MoE + 层间串行/同步" 主导，GPU 大量空转。**

**PP（`-ncmoe 27`）**

| prompt | PP (t/s) | 单前向耗时 | 聚合权重带宽 |
|---|---|---|---|
| 512 | 144.2 | 3.55 s | 20.21 GiB/3.55s = **6.1 GB/s（峰值 2.7%）** |
| 2048 | 45.5 | 45.0 s | **0.48 GB/s（0.2%）** |

> **PP 随 batch 严重退化**（512->2048 每 token 慢 ~3 倍），且远低于带宽上限 —— 这是 PP1000 的核心障碍，
> 需定位是大 M 的 `MUL_MAT_ID`、注意力、还是 compute buffer 溢出（见 5.6）。

### 5.4 替代方案：MoE cache（`--moe-cache-mib`）实测

`llama-bench` 不支持 `--moe-cache-mib`，用 `llama-cli` 实测（`-ncmoe 41 --moe-cache-mib 4096`）：

| 指标 | 值 |
|---|---|
| 缓存大小 | 4094 MiB / host 专家 18600 MiB |
| 覆盖 | 2 组，各 1126 slots（**22%**） |
| TG 命中率 (ubatch<=8) | **65.4%**（上传 7691 MiB） |
| PP 命中率 (ubatch>8) | **27.5%** |
| TG 结果 | **4.7 t/s**（比无缓存基线 11.2 更慢） |

**原因**：`build_moe_cache_slots()`（`llama-graph.cpp:2518`）把 `slot_map` 放主机内存，`ggml_get_rows` 读它
-> 调度器**每层强制开 split** -> 触发 `llama_moe_cache::copy()`：读回专家 id 到 host、CPU 端 LRU、上传、`synchronize`。
**41 层 = 每 token 41 次 split + host 往返 + 同步**，命中率又不够高（35% 未命中），净结果为负优化。

### 5.5 op 级 GPU profile（`GGML_VK_PERF_LOGGER=1`，TG `-ncmoe 27`）

单 token GPU kernel（节选，长尾略）：

| kernel | 形状 | 单次耗时 | 带宽/备注 |
|---|---|---|---|
| `MUL_MAT_VEC q6_K` | m=248320 n=1 k=2048 | **1931 µs** | output 投影，0.39 GiB -> **216 GB/s（已达峰值）** |
| `MUL_MAT_VEC q4_K` | m=2048 n=1 k=4096 | 80 µs ×10 | 专家 gate/up |
| `MUL_MAT_VEC q4_K` | m=8192 n=1 k=2048 | 79 µs ×8 | 专家 down（8192 = 256×...） |
| `MUL_MAT_VEC q6_K` | m=8192 n=1 k=2048 | 89 µs ×5 | 专家 down q6_K |
| `MUL_MAT_VEC q4_K` | m=512 n=1 k=2048 | 10 µs ×33 | SSM 投影 |
| `TOPK_MOE_EARLY_SOFTMAX_NORM` | - | 19 µs ×13 | 路由 |

**整轮 GPU 时间**（D 用例，`-p 0 -n 128`）：perf logger 逐提交打印，共 **3612 个提交块，累计 3181 ms**
-> **≈24.9 ms/token**（注意 logger 自身约 2x 开销；实际约 12–15 ms/token）。

对照实测 TG ≈ 70.6 ms/token（14.16 t/s）：**GPU 只占 ~15–17%，其余 ~83% 在 CPU 的 MoE 计算、跨后端拷贝与层间串行。**

**结论**：单 kernel 效率不差（output 投影已达带宽峰值，且它本身是天然 21% 的字节成本）；真正的问题在
**kernel 之间的空隙、CPU/GPU 之间的往返（混合 MoE）、以及 41 层逐层串行**。

### 5.6 PP 的 op 级 profile（`GGML_VK_PERF_LOGGER=1`，`-ncmoe 27`）

GPU 侧最重的算子（`batch=512`，即默认 ubatch）：

| kernel | 形状 | 单次 | 吞吐 |
|---|---|---|---|
| `MUL_MAT_ID q4_K` | m=512 n=8 k=2048 n_expert=256 | 2762 µs ×24 | **3109 GFLOPS** |
| `MUL_MAT_ID q6_K` | m=2048 n=8 k=512 n_expert=256 | 3286 µs ×7 | 2611 GFLOPS |
| `MUL_MAT_ID q4_K` | m=2048 n=8 k=512 n_expert=256 | 2906 µs ×6 | 2952 GFLOPS |
| `MUL_MAT_VEC q6_K` | m=248320 n=1 k=2048 | 1928 µs | output 投影（PP 也会跑满 vocab） |

**观察**
- 专家 `MUL_MAT_ID` 在 GPU 上达到 **~3000 GFLOPS**，单算子并不慢；GPU 不是 PP 的短板。
- 但 PP512 整批 ~4.0 s 而 GPU 侧远小于它 -> **PP 同样受 CPU（27 层 MoE）与串行支配**。
- `batch=512` 说明默认 ubatch 只有 512；MUL_MAT_ID 的 `n=8`（=n_expert_used）意味着每个专家平均只有 512×8/256 = 16 个 token -> GEMM 形状偏小。

> 注：perf logger 按提交块打印，大 batch 下 `Total time` 块存在不自洽（F 用例 max block 6.37 s > 整批 wall），
> 因此本表**只用于算子相对成本**，不用于绝对占比。

### 5.7 PP 随 batch 退化的原因：内存压力与换页

对比 p512 与 p2048（均 `-ncmoe 27`）：

| 指标 | pp512 | pp2048 |
|---|---|---|
| PP (t/s) | 144.2 | 45.5 |
| 内存已用峰值 (GB) | 30.3 | **34.7** |
| 磁盘读 avg / max (MB/s) | 0.0 / 0.0 | **3.9 / 46.6** |
| GPU util avg | 43.3 | 82.6 |

**结论**：p2048 时内存用量逼近 39 GB 上限，宿主专家被换出到 **SSD**，随机专家访问变成磁盘读 -> PP 从 144 崩到 45.5。
这说明：**"SSD 分层"在第一阶段的正确用法是"容量层"（冷专家），而一旦热专家被换到 SSD，吞吐会被摧毁**。
即 `--no-mmap`（或 mlock/预取）保证热专家常驻 RAM，是 PP 大 batch 的前提。

### 5.8 每 token 的 kernel 数量与"延迟受限"证据

从 5.5 的 profile 统计单 token 的 GPU kernel 次数（`-ncmoe 27` 混合配置）：

| 类别 | 次数 |
|---|---|
| `MUL_MAT_VEC`（含 output 投影、专家、SSM 投影等） | ~114 |
| `RMS_NORM(_MUL)` | ~62 |
| `SCALE` | 18 |
| `SIGMOID` / `SIGMOID_MUL` | 9 / 18 |
| `SILU_MUL` / `SOFTPLUS_MUL` | 9 / 9 |
| `SSM_CONV_SILU` | 9 |
| `ROPE(_VIEW_SET_ROWS)` / `SET_ROWS` | 8 / 4 |
| `TOPK_MOE_EARLY_SOFTMAX_NORM` | 13 |
| **合计（仅 GPU 部分）** | **~270** |

每 token 有 **~270 个小的 GPU kernel**（SSM 逐元素链贡献大量 3–5 µs 的小核）。batch=1 时每个小核都接近"延迟下限"
而非吞吐，GPU 在核之间空转 —— 这解释了稳态 GPU 利用率仅 ~40%。**这是延迟受限，不是算力/带宽受限。**

与之对应，A/B 已证明（见 `AB-TUNING.md`）：提交批处理、fusion、线程数都不改变结果；
**即使把 CPU MoE 全部移走（双 GPU `-ncmoe 0`）也只涨 ~15%**。

**真正的结构性根因（A/B 批次 4 定位）**：`GGML_VK_SYNC_LOGGER` 统计显示
**每 token 约 436 次 `ggml_vk_sync_buffers()` 全局管线屏障**（平均每 ~3 个节点一次）。
该函数（`ggml-vulkan.cpp:1144`）发出的是对所有内存的保守全屏障，会反复排空 GPU 流水线。
这解释了 GPU 利用率仅 ~40% 与每 token ~40 ms 无法归因的开销。

## 6. 结论

1. **现状最快的合法配置是"最大驻留"**：`-ncmoe 27` -> **PP ~144–149 / TG ~14.2**（`-dev Vulkan1 -fa on`）。
2. **瓶颈不是带宽，也不是算力，而是延迟/串行**：TG 时 CPU≈19%、GPU≈43%，两者都未饱和；
   聚合权重带宽仅峰值的 9%（TG）/ 2.7%（PP512）。
3. **单 kernel 效率其实不差**：output 投影（0.39 GiB/token）达 216 GB/s ≈ 峰值。问题在 kernel 之间的空隙、
   跨后端（CPU/GPU）往返，以及 41 层的逐层串行。
4. **现有 MoE cache 在 Vulkan 上净负优化**（4.7 vs 11.2 t/s），根因是每层 split/sync + 低命中率。
5. **显存已超配**（dedicated 8.5 GB > 8176 MiB + shared 1.3–1.5 GB），驻留没有进一步空间，反而可能损害表现。
6. **磁盘不参与稳态**：20 GB 权重常驻 39 GB RAM，SSD 只在冷启动/换出时使用。
7. **TG200 必须靠 MTP**：裸 TG 物理上限 111 t/s；需先把裸 TG 从 ~14 提到 ~100（全 GPU MoE + 去串行 + 高命中），
   再用 MTP 冲等效 200。
8. **PP 随 batch 严重退化**（p512=144 -> p2048=45.5 t/s），是 PP1000 的核心障碍。

## 7. 后续优化方向（按杠杆，附本基线依据）

1. **消除"混合 CPU/GPU MoE"的层间往返（最高杠杆）**。依据：TG 时 CPU≈19%/GPU≈43% 均未饱和，聚合带宽仅峰值 9%；
   27 层专家在 CPU，每层都要 GPU<->CPU 往返 + 同步。目标：让 MoE 全在 GPU，或批量化 CPU 侧。
2. **重写/替换 MoE cache 的调度耦合**。依据：`--moe-cache-mib` 反而 4.7 t/s；根因是每层 split + host 往返 + 同步（41 次/token）。
3. **提高命中率 + 下一层预取**。依据：现命中率 TG 65% / PP 27%。
4. **保证热专家常驻 RAM（`--no-mmap`/mlock/预取）**。依据：p2048 内存压力导致换页到 SSD，PP 崩到 45.5 t/s。
5. **PP：增大 ubatch + 提高每专家 token 数**。依据：默认 ubatch=512，每专家仅 ~16 token，`MUL_MAT_ID` 形状偏小。
6. **RDNA2 无 coopmat**：PP/GEMM 优化只能走 subgroup + 整数点积（`int dot=1`）；GPU 侧 `MUL_MAT_ID` 现约 3000 GFLOPS 可再优化。
7. **MTP（TG200 的唯一路径）**：裸 TG 上限 111 t/s，需先逼近 ~100 再靠 MTP 冲等效 200；需实测接受率与 MTP 单步成本。
8. **评估双 GPU 并行带宽**：dGPU GDDR(224 GB/s) + iGPU UMA(~76 GB/s) 理论叠加 ~300 GB/s，可抬高裸 TG 上限；
   但 iGPU 与 CPU 共享系统 RAM 带宽，需实测净收益。
9. **Vulkan pipeline cache**：`ggml-vulkan` 未持久化 VkPipelineCache，迭代时每次重编 shader 成本高，建议加盘级缓存。

## 8. 复现命令

```powershell
# 最快基线（合并 PP/TG）
powershell -File .\run-case.ps1 -Tag A -Bin llama-bench `
  -Args '-m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 27 -p 512 -n 128 -fa on -r 3 -o md'

# op 级 profile（TG / PP）
powershell -File .\run-case.ps1 -Tag D -Bin llama-bench -Env 'GGML_VK_PERF_LOGGER=1' `
  -Args '-m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 27 -p 0 -n 128 -fa on -r 1 -o md'
```
