# MoE 本地运行时 - A/B 调优记录

对照基线见同目录 `BASELINE.md`。基线（`-dev Vulkan1 -ngl 99 -ncmoe 27 -fa on`）：**PP512 ≈ 144 / TG128 ≈ 14.2 t/s**。

原始产物在 `results/`（命名 `AB*`）。

---

## 归因前提：TG 有一个 ~50 ms/token 的固定开销

- `-ncmoe 41 -> 27`（CPU 层 41 -> 27）TG 仅 11.2 -> 14.2 t/s。
- 对每层"CPU vs GPU"线性拟合：`C-G ≈ 1.3 ms/层`；再由 `41C+F=89.6ms, 27C+14G+F=70.4ms` 得**固定项 F ≈ 50–60 ms/token**。
- 稳态期间 **GPU 利用率 ≈ 40%、llama 进程 ≈ 2.7 核**（`proc_cpu_pct ≈ 14%`）：两者都空闲 -> **延迟/串行**，不是算力或带宽。

## AB 批次 1：定位这个固定开销（`-dev Vulkan1 -ncmoe 27`，TG n256）

| tag | 改动 | TG (t/s) | CPU% avg/max | GPU util avg | 结论 |
|---|---|---|---|---|---|
| AB1_base | — | 12.37 ± 1.59 | 31.7 / 90 | 41.5 | 参照（方差大） |
| AB2_nommap | `--no-mmap`（**无效参数，失败**） | — | — | — | llama-bench 用 `-lm none` |
| AB3_nodes10k | `GGML_VK_MAX_NODES_PER_SUBMIT=10000` | 12.87 ± 0.54 | 23.9 / 35.8 | 40.9 | **无提升** |
| AB4_nofusion | `GGML_VK_DISABLE_FUSION=1` | 12.78 ± 0.54 | 31.8 / 80 | 40.3 | **无提升**（fusion 非瓶颈） |
| AB5_t8 | `-t 8`（默认） | 11.25 ± 1.70 | 34.0 / 56.1 | 42.6 | 与默认同量级 |
| AB6_t16 | `-t 16` | 10.63 ± 0.62 | 35.8 / 94.7 | 42.4 | **更慢**（线程争用） |

**结论（AB1）**：
- 提交批处理（`max_nodes_per_submit`）、fusion、线程数**都不是**这个固定开销的来源。
- 线程默认（=8）已是最优；增多线程反而变差。
- 因此固定开销是**结构性**的：batch=1 时逐层严格串行，而 MoE 在 CPU、其余在 GPU，导致**每个 CPU-MoE 层都要一次 GPU<->CPU 往返 + 同步**，GPU 在 CPU 段空转、CPU 在 GPU 段空转。
- **方向**：减少/消除"CPU 层"，即让更多层的 MoE 在 GPU 上算。

## AB 批次 2：用核显（UMA 24 GB）作为第二显存，消除 CPU MoE

思路：`-dev Vulkan0/Vulkan1 -sm layer` 把层分到 dGPU(8 GB GDDR, 快) 与核显(24 GB UMA, 走系统 RAM, 慢)，
使 `-ncmoe` 可以降到 0 —— **完全消除 CPU 上的 MoE**，从而消除 GPU<->CPU 逐层往返。

| tag | 配置 | TG (t/s) | 磁盘读 avg (MB/s) | 说明 |
|---|---|---|---|---|
| AB7_2gpu_ncmoe0 | `-dev Vulkan0/Vulkan1 -sm layer -ncmoe 0` | **16.27 ± 0.78** | 284 | 全部层在 2 GPU；**最好**，但测量被 SSD 读污染 |
| AB8_2gpu_ncmoe10 | 同上 `-ncmoe 10` | 11.45 ± 0.61 | 211 | |
| AB9_2gpu_ncmoe20 | 同上 `-ncmoe 20` | 9.98 ± 0.17 | 229 | |
| AB10_dgpu_loadnone | `-dev Vulkan1 -ncmoe 27 -lm none` | 12.79 ± 0.93 | 216 | `-lm none` 首次全量读盘 |

**关键结论（推翻批次 1 的假设）**：
- **完全消除 CPU MoE（AB7，`-ncmoe 0`）只把 TG 从 ~14.2 提到 16.27（+15%）**，远不是预期的数倍。
- 因此那个 ~50 ms/token 固定开销**不是** CPU/GPU 逐层往返。
- 综合批次 1（无旋钮有效、fusion/提交批处理/线程都无效）与批次 2（移除 CPU MoE 收益很小），
  **真正的瓶颈是 batch=1 下的逐 kernel 提交/延迟**：41 层 × 每层几十个小 kernel，GPU 在 kernel 之间空转，
  CPU 也大部分时间在等。这是**延迟受限**，不是算力/带宽受限。
- AB7 的 284 MB/s 磁盘读说明该配置还发生了换页，真实上限应高于 16.27，但结构性问题不变。

## AB 批次 3：MTP（增加"每前向 token 数"，同时摊薄 kernel 启动延迟）

既然 batch=1 延迟受限，理论上投机/MTP 能同时提升"等效吞吐"和"GPU 效率"：一次前向验证多个 token。
模型自带 MTP 头（`nextn_predict_layers=1`，GGUF 含 `blk.<last>.nextn.eh_proj.weight`）。

| tag | 配置 | 生成 t/s | 备注 |
|---|---|---|---|
| AB11_cli_nospec | `-p Hello -n 512 -st` | 10.7 | 生成过短（打招呼即 EOS），噪声大 |
| AB12_cli_mtp | + `--spec-type draft-mtp` | 11.3 | 同上 |
| AB13_cli_nospec_eos | `-p Hello -n 256 -st --ignore-eos` | **12.2** | 强制 256 token，干净对照 |
| AB14_cli_mtp_eos | + `--spec-type draft-mtp` | **12.8** | 仅 **+5%**；输出出现重复（接受后退化） |

**结论（AB3）**：该模型的 MTP 头在本配置下**几乎不带来等效吞吐提升（+5%）**，且 `--ignore-eos` 下 draft 产生重复文本，
说明接受率有限/质量不稳。**MTP 不是本模型上的数量级杠杆**（至少在未调优/未配专门 draft 前）。

## AB 批次 4：同步屏障统计（决定性发现）

`GGML_VK_SYNC_LOGGER=1` 打印每个 `ggml_vk_sync_buffers()` 调用（以及节点列表）：

- 一次 64-token 评测：**`sync` 行 51870 条**，图节点索引上限 1256，节点行 149425（约 119 次图评测）。
- => **每个 token 约 436 次同步屏障**（平均每 ~3 个节点一次）。

`ggml_vk_sync_buffers()`（`ggml-vulkan.cpp:1144`）发出的是一个**全局管线屏障**：
`shaderRead|shaderWrite|transferRead|transferWrite -> 同样的全集`，即对所有内存、保守地排空整条流水线。

**判定（经批次 5 修正）**：这 436 次/ token 的全局屏障确实存在（来自 `ggml-vulkan.cpp:12377` 的 `overlaps_unsynced`
别名冒险检测），但下面对照实验表明它只值 ~20%，**不是主因**。

## AB 批次 5：屏障成本上界（临时实验）

在 `ggml-vulkan.cpp` 临时加 `GGML_VK_NO_SYNC` 开关，跳过节点循环里的 `ggml_vk_sync_buffers()`（**故意破坏正确性，仅测上界**），
测完即回退（`git diff -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` 为空）。

| tag | 配置 | TG (t/s) |
|---|---|---|
| SYNC1_base | 正常 | 12.69 ± 0.60 |
| SYNC2_nosync | `GGML_VK_NO_SYNC=1`（屏障全关） | **15.30 ± 0.12** |

**结论**：即使把**全部**节点屏障去掉，TG 也只有 15.3（+21%），离 roofline 111 仍差 7 倍。**屏障不是主因。**

## 方案 A 记录：buffer-scoped barrier（评估后暂缓，不予实现）

**思路**：节点循环里的 `ggml_vk_sync_buffers()`（`ggml-vulkan.cpp:1144`）发出的是**全局内存屏障**（对所有 buffer 生效）。
而上游的冒险判定 `overlaps_unsynced` 本身就是"**同一个 `dev_buffer` 内**区间重叠"，因此可把该屏障从"全局"
收窄为"仅对涉事 buffer 的 `vkCmdBufferMemoryBarrier`（`VK_WHOLE_SIZE`）"，从而不再冲刷模型权重等无关 buffer。

**实测上界（批次 5）**：彻底去掉这些屏障 → TG 12.69 → **15.30（+21%）**。scoped 版本只能拿到其中的一部分。

**为何暂缓（不实现）**：
- **收益有限**：上界仅 +21%，与目标所需 ~14x 差太远。
- **复杂度/风险高**：需精确追踪每次 `need_sync` 涉及的全部 buffer 集合，并保证不遗漏 prealloc 暂存与跨 pass 依赖；
  Vulkan 内存屏障的正确性极难用现有测试覆盖，回归风险大。
- **难以推广**：改动深嵌在 `ggml-vulkan` 的调度/屏障核心路径，其它后端（CUDA/Metal）无法复用，不符合"小而可上游化"。

**结论**：作为一次**已验证上限、但性价比不足**的候选记录在案；若未来专门做 ggml-vulkan 系统级优化，可再取用。

## AB 批次 6：P3 expert-centric prefill（ubatch，重大收益）

前提：CPU `MUL_MAT_ID` 已是 **expert-major**（`ggml-cpu.c:1671` 外层按专家 `cur_a`），所以"每专家一次"的计算本就成立；
但 **ubatch 默认 512** 会把大 prompt 拆成多趟 -> 每专家被读多次、且 GEMM 形状偏小。
（参考 Niko1221/Strata：prefill 分块至 8192 且边算边拉；Cintu07：expert-major vs token-major reads 68x↓。）

**A/B（`-dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -n 0`）**

| tag | prompt / ub | PP (t/s) | 备注 |
|---|---|---|---|
| P3_pp512_ub512 | 512 / 512 | 135.4 | 满块 |
| P3_pp2048_ub512 | 2048 / 512 | 43.6 | 默认，拆 4 趟 |
| P3_pp2048_ub1024 | 2048 / 1024 | 64.7 | 拆 2 趟 |
| **P3_pp2048_ub2048** | 2048 / 2048 | **321.1 / 308.2**（复测） | **最优，7.4x vs ub512** |
| P3_pp512_ub2048 | 512 / 2048 | 132.2 | 块比 prompt 大，无额外收益 |
| P3_pp4096_ub2048 | 4096 / 2048 | 82.2 | 注意力 O(n^2) 主导 |
| P3_pp4096_ub4096 | 4096 / 4096 | 81.9 | 同上 + 内存压力 |
| P3_pp8192_ub2048 | 8192 / 2048 | 74.9 | 同上 |

**结论**：
- **让 ubatch = batch（约 2048）使每专家每层只读一次**，PP 从 43.6 -> **~310**（**7.4x**）；纯配置、零代码、可复现。
- **最优在 prompt=ubatch≈2048**。prompt 再大时，10 层全注意力进入 **O(n^2)**，t/s 随 n 回落（4096 -> 82，8192 -> 75）。
- 因此 PP 的下一步瓶颈是**注意力 O(n^2)**（FA 调优）与剩余的 CPU MoE，而不是"每专家读几次"。
- 实用结论：**prefill 用 `-ub 2048`**，且把大 prompt 按 ~2048 分块读入最省。

## AB 批次 7：TG 阶段的 GPU / 带宽利用率量化（决定是否转预取）

### 7.1 量化（`-ncmoe 27 -fa on`，TG n256）

`TGq_monitor`（无 logger）与 `TGq_profile`（`GGML_VK_PERF_LOGGER=1`, n96）：

| 指标 | 值 |
|---|---|
| TG | 13.2-13.6 t/s -> **75.8 ms/token** |
| GPU 引擎利用率（计数器 avg/max） | 36-42% / 52-85% |
| CPU 利用率 | 20-25% |
| 磁盘读 | ~0（稳态 RAM 常驻） |
| 显存峰值 | ~8200 MB（超配到 shared） |
| GPU kernel 执行时间 / token | **~12 ms**（logger 实测 23.7 ms，logger 约 2x 开销） |
| **GPU kernel 执行占比** | **~16%**（其余 ~84% 是核间空隙 / CPU MoE / 等待） |
| 每 token 激活权重 | 1.87 GiB |
| 聚合权重带宽 | 1.87 GiB / 75.8 ms = **24.7 GB/s（峰值 224 GB/s 的 ~11%）** |
| 其中 VRAM 服务 | 1.49 GiB -> 19.7 GB/s（~9%） |
| 单 kernel 带宽 | output 投影 **216 GB/s（≈峰值）** |

**判定**：算力与带宽都不是限制——**单 kernel 已达峰值，损失在 kernel 之间的空隙、CPU MoE 与串行**。
GPU 引擎利用率虽显示 36-42%，但"真正执行"只占 ~16%，其余是 stall。

### 7.2 尝试提高利用率（配置杠杆 A/B，TG n256）

| tag | 配置 | TG (t/s) | GPU util avg | 备注 |
|---|---|---|---|---|
| TG7_base | `-ncmoe 27` | 13.62 ± 0.70 | 36.0 | 参照 |
| TG7_kvq8_nc27 | +`-ctk q8_0 -ctv q8_0` | 13.74 ± 0.71 | 32.1 | 省显存未转化为收益 |
| TG7_kvq8_nc25 | KV q8_0 + 多驻留 2 层 | 13.58 ± 0.65 | 33.8 | 无收益 |
| TG7_faoff | `-fa off` | 13.82 ± 0.69 | 31.1 | 噪声内；出现磁盘读 |
| TG7_nc26 | `-ncmoe 26` | 10.83 ± 0.39 | 43.7 | **反而更差**（与 nc27/nc25 不一致） |

**判定**：**没有任何配置能提高 TG 或 GPU 利用率**（全部落在 ±0.7 噪声内；`-ncmoe 26` 甚至掉到 10.83）。
KV 量化省下的显存没有转化为可用驻留（受 compute buffer/对齐约束）。

### 7.3 结论（步骤 2 结束）

- GPU 利用率（执行口径 ~16%）与带宽利用率（~11%）**都很低，且无法靠配置提高**。
- 根因是 **batch=1 下 41 层的逐 kernel 延迟 + CPU MoE 与 GPU 串行**，与 `BASELINE.md` 5.x / `AB-TUNING.md` 批次 4-5 一致。
- 因此按计划**转入步骤 3：专家预取 / 流式传输**（见 `PREFETCH-DESIGN.md`，P1/P2）。

## AB 批次 8：预取/流式的上限（"全部专家在 GPU 层"）

预取/流式的终极目标就是把专家搬上 GPU 计算，所以它的**上限 = 所有专家都常驻 GPU 层、没有 CPU MoE 时的 TG**。
单张 8 GB dGPU 装不下 18.65 GiB 专家，只能借助核显（UMA 24 GB），即 `-dev Vulkan0/Vulkan1 -sm layer -ncmoe 0`。

| tag | 配置 | 结果 | 备注 |
|---|---|---|---|
| TG8_ceiling_2gpu | 2GPU `-ncmoe 0 -ub 2048` | **加载失败** | 核显 UMA 分配 304 MB OOM |
| TG8b_ceiling_2gpu_n0 | 2GPU `-ncmoe 0 -ub 512` | **加载失败** | 同上，复现性差 |
| TG8b_ceiling_2gpu_n10 | 2GPU `-ncmoe 10 -ub 512` | **TG 12.38** | 比单 GPU 基线更差；SSD 读 239 MB/s |
| AB7（早先） | 2GPU `-ncmoe 0 -ub 512` | TG 16.27 | 含 SSD 换页污染 |
| TG8_base_recheck | 单 GPU `-ncmoe 27 -ub 2048` | **PP 320.4 / TG 14.23** | 基线复测 |

**判定**：
- 即便把**全部专家放到 GPU 层**（预取/流式能达到的极限状态），TG 也只有 **~12–16 t/s**，**并不优于**单 GPU 的 14.2。
- 双 GPU 路径还叠加了核显较慢（UMA 共享内存带宽）、SSD 换页（239 MB/s）与 OOM 不稳定。
- 因此**预取/流式对 TG 的天花板约 +15%**，离 TG200（需 ~14x）差一个数量级以上。
- 结合批次 7（GPU 执行 ~16%、带宽 ~11%）与批次 4/5（屏障/CPU-MoE 各 ~15-20%）：**TG 的瓶颈是 Vulkan 后端的逐 kernel 延迟与串行，不是专家放置**。预取解决的是"专家传输"，而传输本就不在关键路径上（磁盘读≈0、单核已达峰值带宽）。

## AB 批次 9：84% 非-kernel 时间的归因（sched 分片计时）

新增 env 门控 `GGML_SCHED_TRACE`（`ggml/ggml-backend.cpp` `ggml_backend_sched_compute_splits`），
每 token 打印 4 段耗时：跨后端同步 / activation 拷贝 / CPU 后端计算 / GPU 提交。

`-dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -p 0`：

```
sched_trace: splits=56  sync=0.000ms  copy=20.9ms  cpu=53.6ms  gpu_submit=11.6ms
```

每 token ~79 ms（≈ 1/12.6 t/s）拆解为：

| 段 | 值 | 占比 | 含义 |
|---|---|---|---|
| CPU 后端计算 | **~47 ms** | **~60%** | 27 层 MoE 的 `MUL_MAT_ID` 在 CPU 上算（阻塞） |
| activation 拷贝 | **~20 ms** | **~25%** | 56 个 split 的跨后端拷贝（GPU<->CPU 往返） |
| GPU 提交（CPU 侧） | ~12 ms | ~15% | 56 次 `graph_compute_async` 的 CPU 提交开销 |
| 跨后端同步 | 0 ms | 0% | `sync=0`（事件同步未成为瓶颈） |

**判定**：
- 64% 的时间确实是"非 kernel 的串行/等待"，但**主要不是 launch 延迟，而是 CPU MoE 计算本身（47 ms）+ 56 个 split 的拷贝/提交（32 ms）**。
- 这两者都随 **CPU-MoE 层数**增长：27 层 CPU MoE 造成 ~56 个 split 的 GPU<->CPU 摆渡。
- 因此本轮首选 **D3 投机解码（MTP）**：一次前向验证 K 个 token，把"逐 token 固定开销（CPU MoE + 56 split + 提交）"摊薄到 K 个 token；同时 CPU `MUL_MAT_ID` 在 batch=K 下效率更高。次选 D4（MMQID 调参）直接压 CPU MoE 这 47 ms。

## AB 总结（截至批次 8）

| 手段 | 效果 | 判定 |
|---|---|---|
| 专家驻留扫描 `-ncmoe` | PP 114.8→148，TG 11.2→14.2 | 有效但饱和 |
| MoE cache `--moe-cache-mib` | TG 4.7（更差） | 负优化 |
| `GGML_VK_MAX_NODES_PER_SUBMIT` | ≈base | 无效 |
| `GGML_VK_DISABLE_FUSION` | ≈base | 无效 |
| 线程 `-t 8/16` | 16 更差 | 默认最优 |
| 双 GPU `-ncmoe 0`（消除 CPU MoE） | 16.3（+15%，被换页污染） | 收益有限 |
| MTP `--spec-type draft-mtp` | +5% | 收益有限 |
| 屏障全关（上界实验） | 15.3（+21%） | 不是主因 |
| **P3: ubatch=2048（expert-centric prefill）** | **PP 43.6→~310（7.4x）** | **有效（首个数量级收益）** |
| TG: KV q8_0 / `-fa off` / 多驻留 | 均落 ±0.7 噪声内（nc26 反而 10.8） | 无法提高 GPU/带宽利用率 |
| TG 利用率量化 | GPU 执行 ~16%、带宽 ~11%、单核已达峰值 | 瓶颈是延迟/串行，转预取 |
| 预取/流式上限（全专家在 GPU 层） | TG 12–16（≤ 单 GPU 基线 14.2） | 不是杠杆（天花板 ~+15%） |

**统一结论**：TG 的 ~70 ms/token 由**多个 ~15–20% 的小因素叠加**构成（屏障 ~20%、CPU MoE ~15%、命令录制/调度、逐 kernel 延迟），
**没有任何单一杠杆**，也没有一个能把它抬高 5–10x 的可达旋钮。这是 **batch=1 + 8 GB 显存 + RDNA2 Vulkan + 20 GB 模型** 下的
系统性下限，而非某个孤立缺陷。

## 对第一阶段目标可行性的判断

- roofline 硬上限：裸 TG **111 t/s**；实测 **~14 t/s**（13%）。
- PP：实测 ~145 t/s（p512），roofline 带宽上限 ~5285 t/s。
- 目标 **PP1000 / TG200** 相对现状需 **~7x / ~14x**（TG200 还需 MTP 接受 ~1.8）。
- 经 5 批 A/B：**没有任何单一改动能接近目标**。要达成需要**同时**把数百个小 kernel/屏障压掉、重写调度与算子融合，
  并让 MTP 接受 ~1.8 —— 属 ggml-vulkan 的系统级重写，不是"算子优化/专家预取"层面的工作。
- 现实建议：先兑现已见效的叠加（驻留调优 + 双 GPU + 屏障优化的正确版本 ≈ +35%），把 TG 推到 ~18-20；
  再把 TG200 明确定义为"含 MTP 的等效目标"并承认需要长期 kernel 级投入。
