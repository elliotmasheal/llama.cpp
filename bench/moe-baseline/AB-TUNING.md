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

## AB 总结（截至批次 5）

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
