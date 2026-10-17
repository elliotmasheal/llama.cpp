# 下一步方向与计划 - PP / TG 提速（research 记录）

> 记录时间: 2026-10-17。**不含代码改动**，仅调研梳理 + 验证计划。
> 前置确认: `fastLlama` 工作区干净，HEAD = `8afb3354d`（含 `396ef1cf7`，见 §0.3）。
>
> 本文补充（不是重复）以下**已完成/已验证**的清单：
> 专家驻留 `-ncmoe`、expert-centric prefill `-ub 2048`（PP 43.6 -> ~310，7.4x）、
> MoE cache（负优化）、全局屏障上界（+21%）、专家预取/流式 P1/P2（TG 上限 ~+15%）、
> 双 GPU 上限、GPU/带宽利用率量化、内置 MTP 头（+5%）。
> 依据见 `BASELINE.md` / `AB-TUNING.md` / `PREFETCH-DESIGN.md` / `tuner/`。

---

## 0. 背景约束（本机，用于判断每个方向是否值得做）

1. **硬件**: RX 6700S 8 GB，RDNA2（28 CU，subgroup 32，**无 coopmat**，有 int dot），Vulkan。
2. **模型**: Ornith-1.5-35B-A3B（MoE，fork 测试模型）；Spark-X2.5-4B（dense）与
   Ternary-Bonsai-2-27B（dense/三元）是实际在用的 dense 模型。
3. **已量化的根因**:
   - TG 是 **batch=1 逐 kernel 延迟/串行**受限：GPU 执行占比 ~16%、聚合带宽 ~11%、
     每 token ~436 次全局管线屏障、~270 个小 kernel；单 kernel 已达峰值（output 投影 216 GB/s）。
   - PP 的 ubatch 拐点已解（`M* ≈ 1814` -> 取 2048）；**下一个 PP 瓶颈是注意力 O(n^2) 与专家 GEMM 效率**。
   - **长上下文 TG 随深度衰减**（本机实测）：Spark 6K 56 -> 20K 49.5 -> 24K 46 t/s；
     Bonsai 6K 20 -> 20K 14 t/s。这段损失是**注意力**，与 MoE 放置无关，现有清单未覆盖。
   - 超配时溢出到 shared/系统内存：Bonsai 256K 有 ~2.7 GB 在 shared（256K -> 150K TG +13%）。

> 结论性判断: TG 没有单一 5-10x 旋钮；能拿的是**多个 15-50% 的叠加**，外加
> **投机解码**（唯一能把等效 TG 推过裸带宽上限 111 t/s 的路径）。

---

## D3. ngram / prompt-lookup 投机（零 draft 模型）  [先做]

- **原理**: 用上下文/已生成文本里的 n-gram 直接当 draft，命中即一次前向验证多个 token。
  把 batch=1 的逐 kernel 固定开销按命中长度摊薄，**不需要任何 draft 模型**。
- **为什么本机有效**: 本 fork 的 `common/arg.cpp` 已内置
  `--spec-type ngram-mod | ngram-simple | ngram-map-k | ngram-map-k4v`
  （`--spec-ngram-*-size-n/-m/-min-hits`），server 还支持静态/动态 lookup cache
  （`-lcs` / `-lcd`）；零模型成本、零显存。agent 场景（重复 system prompt、编辑式代码、
  工具结果回显）n-gram 命中率高，正是 batch=1 延迟受限最受益的场景。
- **验证（小实验）**:
  1. 造两组 prompt：A = 高重复（长 system + 反复出现的片段）；B = 全新文本。
  2. 对照组 `-c 262144 --parallel 1 -fa on`；实验组加
     `--spec-type ngram-mod --spec-ngram-mod-n-max 8`（或 `ngram-simple` + `-lcd cache.bin`）。
  3. 从 server 日志读 draft acceptance，再比 TG。
- **预期指标**: 重复型文本 TG **x2-x4**；新颖文本 **x1.0**（不应退化 >5%）。
- **风险**: 仅在重复工作负载有效；ngram 长度/命中阈值需调；接受率低会略降。
- **前置**: 无（纯配置）。

## D1. 长上下文注意力：FA 的 KV-split / row-split 调优  [针对 256K 用例]

- **原理**: decode 时每个 KV head 的 attention reduction 沿 **key 维在一个工作组内串行**；
  把 K/V 维切成多块到多个工作组（split-K），各出部分 `(m, l)` 统计再归约，可提升占用与吞吐。
  prefill 侧对应 O(n^2) 的 tiling 上限。
- **为什么本机有效**: 实测 TG 随深度衰减（见 §0.3），这部分与 MoE 无关。本机 Vulkan FA
  已有 `row_split` / `d_split` 参数（`ggml-vulkan.cpp:1215-1299`），但
  `row_split = (n_rows < 4 || hsk <= 64) ? 1 : 4` -- **decode（n_rows=1）恒取 1**，
  即长 KV 下不跨工作组切分；RDNA2 28 CU 无 coopmat，占用本就低。
- **验证（小实验）**:
  1. 先量化：`GGML_VK_PERF_LOGGER=1`，在 20K / 64K 深度分别测 FA kernel 的绝对耗时占比
     （Spark 9 层全注意力 vs Bonsai 16 层）。
  2. 改 split 启发式（让长上下文 decode 也 row_split/d_split，或加沿 KV 的 split-K），
     AB 同深度 TG（256K ctx 已分配，20K prompt）。
  3. 正确性先用 `test-backend-ops -o FLASH_ATTN_EXT`（覆盖 GQA 4 KV heads + q4_0 KV）。
- **预期指标**: 长上下文 TG **+20-60%**（Bonsai 比 Spark 更敏感）；短上下文基本不变；
  prompt > 2048 的 PP 也可受益（O(n^2) 主导）。
- **风险**: FA kernel 正确性（GQA、量化 KV、shmem staging）；需重建 + 覆盖测试。
- **前置**: 无。

## D5. 宿主可见内存路径：Resizable BAR / SAM + shared-KV 分层  [先花 10 分钟查]

- **原理**: 超配时（Bonsai 256K 溢出 2.7 GB 到 shared；Ornith 的 27 层 host 专家 + KV
  也大量经 host 访问）性能取决于 **host-visible VRAM 的访问带宽**。若 dGPU 的 BAR 只有
  256 MB，GPU 访问 pinned/host 缓冲要走小窗；启用大 BAR（ReBAR/SAM）后，这些访存更快。
- **为什么本机有效**: 已实测 Bonsai 256K 有 2.68 GB 落在 shared，缩 ctx 立刻 +13% TG
  -- 说明 shared 路径确实在关键路径上。若能把 BAR 从 256 MB 放大，这类场景直接受益。
- **验证（小实验）**:
  1. 查 BAR 大小：GPU-Z / `dxdiag` / Vulkan 设备属性（`maxMemoryAllocationSize`、BAR region）。
  2. 若 BIOS 支持 ReBAR 且当前关闭 -> 开启后重测 Bonsai 256K（20K prompt）TG 与 Ornith TG。
  3. 交叉验证：`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` / `GGML_VK_ALLOW_SYSMEM_FALLBACK`
     行为对比。
- **预期指标**: 若原为小 BAR，开启后 shared 路径 **+10-50%**（Bonsai 256K、Ornith host 专家路径）；
  若本就是大 BAR -> 无变化（快速排除）。
- **风险**: 笔记本 BIOS 可能不提供 ReBAR；属系统/驱动层，非代码；开启后需回归稳定性。
- **前置**: BIOS 选项 + 一次重启。

## D2. 独立小 draft 模型做投机解码（区别于内置 MTP 头）

- **原理**: 把 k 个 token 合成一次 batch=k 验证，把每 token 的固定/串行开销摊到 k 个 token；
  有效 TG ≈ (接受数+1) / 前向。这是**唯一能把等效 TG 推过裸带宽上限 111 t/s 的路径**。
- **为什么本机有效**: 内置 MTP 头只 +5%（接受率差）；独立 draft（同 tokenizer、小 5-10x）
  通常有更高接受率。本机 TG 的瓶颈正是固定/串行开销，且 `-md` / `--spec-type draft-*` 已就绪
  （`common/arg.cpp:4287`）。
- **验证（小实验）**:
  1. `-md <draft.gguf> --spec-draft-n-max K`（K=3/5/8），先读 draft acceptance。
  2. 接受率 >= 0.6 再进 TG AB；同时 8 GB 显存下测 draft 量化（q4_0 KV，draft 也量化）。
- **预期指标**: 接受率 >= 0.6 时 TG **x1.6-x2.2**。
- **风险**: **前置严格** -- 需要与目标模型 **同 tokenizer/vocab** 的小 draft；Ornith
  （qwen35moe）与 Spark/Bonsai 都不一定有现成匹配 draft；8 GB 放 draft + 主模型很紧。
- **前置**: 同 tokenizer 的 draft 模型（否则不可用，直接跳过）。

## D4. RDNA2 专属的 MMQ / MMQID warp-tile 与 wg_denoms 调参（专家 GEMM 效率）

- **原理**: PP 已进入算力平台（`FLOPS_eff 2.88 TFLOP/s ≈ 峰值 18%`），瓶颈是 `MUL_MAT_ID`
  的 GEMM 效率。Vulkan **已为 MoE 提供 int8 路径**（`mul_mmq.comp`、`warptile_mmqid`、
  `mmqid_wg_denoms`，`mul_mat_id_funcs.glsl`，`mul_mmq BK_STEP=1 for MUL_MAT_ID`），
  但 tile 常量按通用 "pre-turing / no-coopmat" 值设，**未针对 RDNA2**（28 CU、subgroup 32）调。
- **为什么本机有效**: `MUL_MAT_ID q4_K` 实测 ~3000 GFLOPS；提升 tile 形状/占用可推高，
  直接抬升 PP 平台（当前 ~310 t/s）。服务 PP1000 目标。
- **验证（小实验）**:
  1. 微基准（秒级）：`test-backend-ops -o MUL_MAT_ID -b Vulkan1`，扫 tile 组合。
  2. 命中后回归 `-p 2048 -ub 2048 -n 0` 的 PP（对照 ~310 / ~320）。
- **预期指标**: `MUL_MAT_ID` **+30-80%** -> PP 平台 **+20-50%**。
- **风险**: 需重建/多次编译；可能 regress 其它 shape 或 GPU；tile 常量当前为编译期。
- **前置**: 一个可快速迭代的 tile 覆盖（env 或生成式模板）。
- **注**: 仅对 MoE（Ornith）有效；Spark/Bonsai 是 dense，不适用。

---

## 排序（按性价比 = 预期收益 / 成本，含成功概率）

| 序 | 方向 | 收益（预期） | 成本 | 成功概率 | 风险/前置 |
|---|---|---|---|---|---|
| 1 | D3 ngram/prompt-lookup | 重复负载 TG x2-x4 | 极低（配置） | 中（依赖负载） | 低；无前置 |
| 2 | D5 ReBAR / shared 路径 | 溢出场景 +10-50% | 极低（先查） | 中（看 BIOS） | 系统层；需 BIOS 支持 |
| 3 | D1 FA KV-split | 长上下文 TG +20-60% | 中（kernel） | 中高 | FA 正确性；无前置 |
| 4 | D2 独立 draft 投机 | TG x1.6-2.2 | 中 | 中（看 draft 可用性） | 需同 tokenizer draft |
| 5 | D4 MMQID tile 调参 | PP 平台 +20-50% | 中高（重建） | 中高 | 仅 MoE；需微基准 |

## 下一步计划（执行顺序）

1. **D3 探路（0.5 天）**: 配置级 AB，先确认本工作负载的 n-gram 命中/接受率值不值得做。
   判定线：重复型 prompt TG >= 1.5x 才继续。
2. **D5 速查（10 分钟）**: 查 dGPU BAR 大小；若可开 ReBAR 且当前为小 BAR，开一档再测。
3. **D1 量化 + 原型（1-2 天）**: 先用 perf logger 确认长上下文 FA 占比与当前 split 行为；
   若占比显著，改 split 启发式 + `test-backend-ops` 回归，再 AB 20K/64K 深度 TG。
4. **D2 可行性筛查（0.5 天）**: 确认目标模型是否有同 tokenizer 的小 draft；有则测接受率。
   无则跳过（不阻塞其它项）。
5. **D4（择机）**: 建 `MUL_MAT_ID` 微基准，扫 tile；仅在 PP 目标优先级回正时投入。

> 与既有路线的衔接：D1/D2 直接攻 TG 的"延迟/串行"根因（与预取/流式是不同层次）；
> D3 是零成本机会项；D4 服务 PP1000；D5 解决长上下文溢出场景。
> 所有项都不改变"TG200 需 MTP/投机 + 裸 TG 逼近 ~100"的既有判断。
