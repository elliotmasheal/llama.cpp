# 专家预取设计（主参考：Niko1221/Strata）

目标：在本 fork 的 `llama-moe-cache` / Vulkan 上实现专家预取，协同 GPU + RAM + SSD。

## 1. 参考实现：Niko1221/Strata

C++/CUDA-HIP 引擎（MIT，复用 ggml/llama.cpp 的 kernel），把单个 MoE checkpoint 拆到 **VRAM / RAM / 只读 NVMe** 三层，
按 checkpoint 自己声明的精度、路由语义、专家数与 top-k 运行。其关键机制：

1. **层内 CPU + GPU 并发算专家**。GPU 算命中的（缓存里的）专家；**CPU 就地算未命中的专家，二者同时进行**，
   谁也不等谁。`--pcie-frac` 决定"未命中专家里有多大比例拷到 GPU 上算"（其余留 CPU），按 PCIe 链路与 CPU 强弱调。
   - 意义：这正是我们现状的病根——`AB-TUNING.md` 显示 TG 时 GPU≈40%、CPU≈20% 都闲着，因为 CPU MoE 与 GPU 是**串行**的。
2. **下一层专家预取**（`STRATA_LOOKAHEAD`）：CPU 算当前层时，**另一线程把"下一层的 router 权重"作用到"本层输入"**，
   预测下一层会用到哪些专家，并提前向 OS 要页 / 经 PCIe 流式拉取。`STRATA_LOOKAHEAD_K` 控制前瞻深度。
   配套 `STRATA_LOOKAHEAD_STATS` 用下一层真实路由打分（top-k / k+4 / k+10 / vote-ranked 的 recall/precision）。
   - **关键点**：它用一个**廉价近似**做预测——不需要未来 token，只需把下一层 router 套到当前层输入上。
   - **我们的优势**：stock llama.cpp 拿不到路由（mehbul/strata 明说因此做不了预取）；
     **本 fork 的 `llama-moe-cache` 已经能读到路由 id**，所以这条路在本仓库可行。
3. **自适应专家缓存**（`--adapt-every` / `--adapt-swaps` / `--adapt-decay` / `--adapt-async`）：缓存**跟随对话**动态换入换出，
   `--adapt-async` 让交换在**后台线程**进行、不阻塞当前窗口。
4. **prefill 分块 + 重叠**：prompt 按最多 **8192 token** 一块读入；**当前层注意力计算时，下一层专家边经 PCIe 流入**。
5. **MTP 投机层常驻 VRAM**（`--spec 4 --mtp`，1.6-1.8x）；KV 在 64K+ 时**分层**（`--kv-resident`，只留被读的部分在 VRAM）。
6. 经验法则：**"更大 VRAM 比更快 GPU 更重要——每多 1 GB 就能多放约 700 个专家，且每个上 GPU 的专家都是 CPU 不必算的。"**

### 另一参考：Cintu07/strata（预取理论）
- **多层前瞻**：NVMe 读延迟 50–150 µs，一层算不住，**需要 3–8 层前瞻**；
- **expert-centric prefill**：每层每专家只读一次（对比朴素 token-major：reads 32768→482，时间 ~85x↓）；
- TinyLFU 准入 + 试用窗；co-activation 布局；"large reads, not contiguous"；queue depth 值 18x。

### 另一参考：mehbul/strata（llama.cpp + ROCm，同款模型）
- **最大化驻留是错的**（全 GPU 18.9 t/s，留 15/41 层在 RAM 反而 42.5）；**prefill 与 decode 想要不同划分**。
- 明确写：**llama.cpp 不暴露路由 → 它无法做预取**。→ 本 fork 的差异化空间。

## 2. 与本仓库的差距（现状 vs Strata）

| 能力 | Niko1221/Strata | 本 fork 现状 |
|---|---|---|
| 层内 CPU/GPU 并发算专家 | 有（GPU 命中 + CPU 未命中并行） | **无**：`MUL_MAT_ID` 只落单一后端，CPU 层与 GPU 层串行 |
| 下一层预取 | 有（next-layer router × current input） | **无**：`llama-moe-cache` 只在**当层**、**按需**上传，且每层 `synchronize` |
| 缓存淘汰 | 自适应、后台交换 | 纯 LRU，关键路径上 |
| prefill 分块重叠 | 有（≤8192，边算边拉） | 无（ubatch 默认 512，逐趟） |
| SSD 层 | 有（mmap experts / resident budget） | 仅 mmap page cache，未主动管理 |

## 3. 分阶段计划

**P3（expert-centric prefill）**
- 确认结论：CPU `MUL_MAT_ID` 已是 **expert-major**（`ggml-cpu.c:1671` 外层按专家 `cur_a`），
  所以"每专家读一次"的**计算**本就成立；可动的是**分块**——默认 ub=512 会把 2048 的 prompt 拆 4 趟、每专家读 4 次。
- 因此 P3 的首个增量 = **ubatch 调优**（`-ub` 1024/2048/4096），A/B 中（见 `results/P3_*`）。
- 若要"边算注意力边流下一层专家"，则依赖 P2 的预取基础设施（Strata 的 0.1.13 也是把 prefill 与预取一起做的）。

**P1（重叠上传 / 去除关键路径同步）**
- 给 `llama-moe-cache` 加**异步、双缓冲**的专家上传，把 `copy()` 里的 `synchronize` 移出关键路径。

**P2（下一层预取）**
- 实现 Strata 的**廉价预测**：CPU/GPU 算第 L 层时，用**第 L+1 层的 router 权重 × 第 L 层输入**预测 L+1 的专家，
  提前预取（`STRATA_LOOKAHEAD_K` 式前瞻）。
- 先用 `STRATA_LOOKAHEAD_STATS` 式的**离线打分**验证 recall/precision，再决定接入。

## 4. 最相关的一条（我们独有）

**层内 CPU/GPU 并发算专家**（Strata 机制 1）直接治本仓库最大的结构性问题（CPU MoE 与 GPU 串行、两侧都闲置）。
但实现它需要让同一层的 `MUL_MAT_ID` **按专家拆到两个后端并发执行**——这是 `llama-graph`/调度层的大改，
超出"算子优化"范畴，需要单独评估。

## 5. 诚实预期

- P3（ubatch）是**便宜的、可立即量化**的一步，但收益上限取决于 PP 是否真是"每专家被读多次"所致（实测见 `results/P3_*`）。
- P1/P2 才能把预取做成 Strata 那样；它们价值和风险都更高，且需要 `llama-moe-cache`/调度层的实质改动。
- 无论哪条，都**绕不开** `BASELINE.md` 记录的 batch=1 逐 kernel 延迟/屏障——那是 Vulkan 后端的系统级问题。
