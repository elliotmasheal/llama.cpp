# LFM2.5-24B-A2B (UD-Q4_K_XL) on RX 6700S 8 GiB (Vulkan) - tuning report (DRAFT)

> Status: SUPERSEDED by `REPORT.md` (final numbers, r=3 validation). Kept for the mechanism analysis.
> Measured numbers come only from `bench/lfm25/results/*.csv` (r=2 unless noted).
> Every source claim was verified against the tree at `a9d2cc8a7` by the research pass.
> No benchmark was run while writing this document.

## 1. 机器与模型事实

### 1.1 机器

| 项 | 值 |
|---|---|
| dGPU | AMD Radeon RX 6700S, 8176 MiB, `gfx1032` (RDNA2, 28 CU, `matrix cores: none`, `int dot: 1`) |
| iGPU | AMD Radeon(TM) Graphics (`Vulkan0`, `uma: 1`, shared 24 GiB) |
| CPU / RAM | Ryzen 7 6800HS 8C/16T / 39.2 GB |
| OS | Windows 11 10.0.26300 |
| 后端 | Vulkan, `-dev Vulkan1` (Vulkan0 是核显) |
| 构建 | `E:\fastLlama\build-vulkan\bin` (Release/Ninja, `GGML_VULKAN=ON`) |

### 1.2 模型（由 GGUF header 解析，非估算）

| 项 | 值 |
|---|---|
| arch / 类型 | `lfm2moe` / `LLM_TYPE_24B_A2B`（block_count==40 时命中） |
| 文件 | 14,876,901,632 B = 13.855 GiB，428 张量 |
| block | 40；`leading_dense_block_count=2`（block 0,1 dense，ffn 11776，Q8_0） |
| MoE 层 | block 2..39 = **38 层**，64 专家 / 用 4，`expert_ff=1536` |
| 注意力 | 仅 block 2,6,10,...,38 = **10 层**（32 heads / 8 KV heads / head_dim 64） |
| shortconv | 其余 **28 层**（l_cache 3），但 FFN 同样是 MoE |
| 专家权重 | `ffn_*_exps` = **13371.0 MiB (13.06 GiB)**，占 94.3% |
| 非专家权重 | **808.8 MiB**（token_embd Q8_0 265.6、shortconv 270.7、attn 106.6、dense ffn 146.6、gate_inp 19.0） |
| 每层专家字节 | 36 层：gate Q4_K 108 + up Q4_K 108 + down Q5_K 132 = **348.0 MiB**；block 7,8：132+132+157.5 = **421.5 MiB** |
| layout group | **2 组**（36 层一组、block 7/8 一组），决定 MoE cache 的 bank 分组 |
| KV cache | 20 KiB/token（10 层 × 8 KV head × 64 × K/V × F16）→ ctx 2048 ≈ 41 MiB |

## 2. 旋钮语义（源码事实）

### 2.1 `-ncmoe N` = 前 N 个 **block** 的专家权重留在 host RAM

- 展开为 `blk.0..blk.(N-1).ffn_(up|down|gate|gate_up)_(ch|)exps`（`common/common.h:1243-1262`，
  `common/arg.cpp:2770-2779`）；匹配用 `std::regex_search`、first-match-wins
  （`src/llama-model-loader.cpp:1238-1259`）。
- 前 2 个 block 是 dense、没有 `_exps` 张量 → **实际进 CPU 的 MoE 层数 = max(0, N-2)**；
  `-ncmoe 40` = 全部 38 层，`-ncmoe 0/1/2` 等价（专家全在 GPU，放不下）。
- **驻留层数 = 40 - N**（不是 N-2！）：`-ncmoe 20` → 20 层专家常驻显存（6.80 GiB）。

### 2.2 `-ncmoe` 只改"权重放哪"，不改"算子在哪跑"

本 fork 的静态模式在 PP 和 TG 两条路径上是**不同**的，这是本报告最重要的机制：

- 调度器对"host 权重 + GPU 算子"有 op-offload 判定：`ggml-backend.cpp:1142-1148` 只在
  `ggml_backend_offload_op(dev, op)` 为真时把算子放到 GPU；Vulkan 的实现是
  `ggml_vk_get_op_batch_size(op) >= op_offload_min_batch_size`（`ggml-vulkan.cpp:16245-16248`，
  默认 32，`ggml-vulkan.cpp:16383`），而 MUL_MAT_ID 的 batch = `op->ne[2]` = token 数
  （`ggml-vulkan-debug.cpp:770-782`）。
- **TG（1 token < 32）**：专家算子留在 CPU，**不产生 host→device 权重拷贝**（每层 4 个专家，
  21.75 MiB 从 RAM 读，实测每层 1.2-1.7 ms ≈ 45-62 GFLOPS CPU 计算；
  若走 PCIe 拷贝则至少 21.75/13.9 = 1.56 ms，已到带宽线，故可判定为 CPU 计算）。
- **PP（ubatch ≥ 32）**：算子被 offload 到 GPU，调度器为每个 host 专家张量在 **compute buffer 里
  建一个全尺寸 copy tensor**（`ggml-backend.cpp:1547-1568`，`ggml_dup_tensor_layout`，
  即 108/108/132 MiB 的完整 64 专家张量），由回调只填"用到的专家切片"
  （`src/llama-context.cpp:2854-2952`；cache 的 bank 命中走 D2D，
  `src/llama-moe-cache.cpp:792-826`）。
- 推论：**静态模式 PP 吃显存（copy tensors + 随 ub 增长的激活），TG 不吃**。
  这解释了为什么 `-ncmoe 36 --moe-cache-mib 5632` 的 pp512 崩到 142 t/s 而 tg32 仍有 29.32 t/s。

### 2.3 `--moe-cache-mib N` = 把 host 专家的一部分常驻显存

- 只缓存"全部专家都在 host 内存"的层，按 (device, layout) 分组（`llama-moe-cache.cpp:274-296`）。
- 每组 slot 数按字节预算算：`slot = 1 个专家 × 该层 3 个张量`；
  本模型 A 组 5.4375 MiB/slot、B 组 6.585 MiB/slot。
- 每组建 bank，前 `n_full = min(5, slots/64)` 层**整层驻留**（读静态视图，无 slot 查表、
  无 ids 回读），其余 slots 组成 LRU 池（`llama-moe-cache.cpp:358-369`）。
- **slot/LRU 快路径只在 ubatch ≤ 32 token 时启用**（`llama-moe-cache.cpp:206,551-573`）：
  TG 恒走；PP 不走（PP 只是"整层驻留 + LRU 池预填充切片"减少 host 字节，
  且每层仍要一次 ids 回读 + 全设备 drain，`llama-context.cpp:2881-2901`）。
- `--moe-hybrid`（未命中交 CPU）源码自带 "experimental, ~2x slower" 警告
  （`src/llama-context.cpp:617-628`），不作为候选。

### 2.4 显存硬边界：单个 VkBuffer 上限 ≈ 2 GiB（驱动 `maxBufferSize`）

`--moe-cache-mib 5888/6144` 在 `-ncmoe 40` 下失败，报
`Requested buffer size exceeds device buffer size limit`（`results/r3_c40_m5888_t6.err`）。
这不是"总显存不够"：当时权重仅 809 MiB + bank 才 ~2/3 装完，总占用远低于 8176 MiB。
失败的是**单个 bank 张量**（down_exps：`(slots+1) × 2,062,688 B`）：

| 配置 | A 组 down bank 单次分配 | 结果 |
|---|---:|---|
| m5632 | 2000.6 MiB | OK |
| m5888 | 2091.3 MiB | 失败（与 .err 的 2,192,965,632 B 完全吻合） |
| m6144 | 2182.4 MiB | 失败（与 .err 的 2,288,123,904 B 完全吻合） |

代码：`ggml-vulkan-buffers.cpp:52-54`（`size > device->max_buffer_size` 直接抛），
上限来自 `VkPhysicalDeviceMaintenance4Properties::maxBufferSize`（`ggml-vulkan.cpp:4249-4257`），
可用 `GGML_VK_FORCE_MAX_BUFFER_SIZE` 覆盖（`:4249-4252`）。**已验证**：置
`GGML_VK_FORCE_MAX_BUFFER_SIZE=6442450944` 时 `--moe-cache-mib 6144` 可分配并运行
（ncmoe40：pp512 516.0 / tg32 28.92，`r6_env_c40_m6144`），但仍不如 mixed 5120。

### 2.5 其他旋钮（实测/源码确认）

| 旋钮 | 结论 |
|---|---|
| `-t` | 只影响 CPU 节点（TG 的 CPU MoE、host 侧工作）。实测 static ncmoe20/22 最优 t6（23.74 / 22.61），t8 已降，t12/t16 更差；cache 配置对 t 不敏感 |
| `-fa on/off` | 只影响 10 个 attention 层（conv 层不构建 attention，`src/models/lfm2.cpp:311-312`）；实测差异在噪声内 |
| `-ctk/-ctv q8_0` | 本模型 KV 仅 20 KiB/token，实测在噪声内 |
| `-lm mmap` | **明显变慢**：干净重跑 mixed 配置 pp512 301.3 / tg32 18.77（`r8_mmap_mixed`），对比 resident 530/29.5。必须保持 resident（`src/llama-model.cpp:1590-1616`）。早期 `r2_mmap`（82.12 / 8.03）与另一 bench 并发，已作废 |
| `-ub` | 不影响 decode（`r4_c36_m5120_ub2048` tg 29.05 ≈ ub512 的 29.62）；决定 prefill 吞吐 |
| `-ngl 0` | 全 CPU 参考：pp512 349.15 / tg32 6.59 |

## 3. 实测数据（`results/*.csv`，`-dev Vulkan1 -fa on -p 512 -n 32`，r=2）

### 3.1 static `-ncmoe`（t=8）

| ncmoe | 驻留专家层 | pp512 | tg32 |
|---:|---:|---:|---:|
| 40 | 0 | 382.75 | 13.45 |
| 38 | 2 | 386.61 | 14.16 |
| 36 | 4 | 392.49 | 15.28 |
| 34 | 6 | 421.18 | 15.14 |
| 32 | 8 | 431.24 | 16.62 |
| 30 | 10 | 452.13 | 17.30 |
| 28 | 12 | 473.19 | 17.36 |
| 26 | 14 | 485.43 | 18.59 |
| 24 | 16 | 503.45 | 18.79 |
| 22 | 18 | 520.83 | 20.54 |
| 21 | 19 | 532.21 | 21.49 |
| **20** | **20** | **544.96** | **21.94** |
| 19 | 21 | 528.65 | 21.44 |
| （`-ngl 0`） | - | 349.15 | 6.59 |

- 溢出断崖在 19/20 之间：20 层驻留 = 6960 MiB + 808.8 = 7769 MiB，只剩 ~407 MiB 给 compute buffer。
- static 线程（ncmoe20）：t5 23.26 / **t6 23.74** / t7 22.21 / t8 21.94；
  ncmoe22：t6 22.61 / t8 20.54 / t12 18.82 / t16 17.56。

### 3.2 cache-only（`-ncmoe 40`）

| cache MiB | t | pp512 | tg32 |
|---:|---:|---:|---:|
| 0 | 8 | 382.75 | 13.45 |
| 2048 | 8 | 423.46 | 11.95 |
| 4096 | 8 | 470.07 | 22.20 |
| 5120 | 8 | 492.78 | 25.82 |
| 5120 | 6 | 492.94 | 25.13 |
| 5376 | 6 | 498.25 | 26.12 |
| 5632 | 6 | 495.06 | 26.55 |
| 5888 / 6144 | 6/8 | 失败（§2.4 的单次分配上限） | - |

### 3.3 mixed（`-ncmoe` 部分驻留 + cache，t=6）

| 配置 | pp512 | tg32 | 备注 |
|---|---:|---:|---|
| ncmoe38 + m5120 | 507.64 | 27.77 | |
| **ncmoe36 + m5120** | **532.07** | **29.62** | 当前 TG 冠军 |
| ncmoe36 + m5120 (`-ub 2048`) | 531.58 | 29.05 | ub 不影响 decode |
| ncmoe36 + m5376 | 525.74 | 28.64 | 差异在噪声内（sd 1.3-1.5） |
| ncmoe36 + m5632 | **142.18** | 29.32 | PP 溢出（总占用 7827 > 8176-workingset） |
| ncmoe36 + m4608 | 510.54 | 25.88 | bank 太小，TG 明显变差 |
| ncmoe35 + m5120 | 536.71 | 29.08 | |
| ncmoe34 + m4608 | 542.29 | 28.08 | |
| ncmoe34 + m4096 | 529.13 | 27.88 | |
| ncmoe32 + m3072 | 526.26 | 24.24 | |
| ncmoe30 + m4096 | 179.15 | 28.11 | PP 溢出（总占用过高） |
| ncmoe30 + m2048 | 520.77 | 19.10 | |
| ncmoe22 + m2048 | 492.69 | 16.09 | |

### 3.4 长 prefill（`-p 2048/4096 -n 0`）

| 配置 | ub | pp | 备注 |
|---|---:|---:|---|
| static ncmoe20 | 512 | 431.00 | |
| static ncmoe20 | 1024 | **141.56** | 断崖 |
| static ncmoe20 | 2048 | **193.57** | 断崖 |
| cache c40 m5120 | 512 | 469.14 | |
| cache c40 m5120 | 2048 | **840.19** | |
| mixed c36 m5120 | 1024 | 703.82 | |
| **mixed c36 m5120** | **2048** | **878.44** | 当前 PP 冠军 |
| mixed c36 m5120 (`-p 4096`) | 2048 | 837.01 | ctx 4096 |

### 3.5 其它

| 配置 | pp512 | tg32 |
|---|---:|---:|
| `-fa off` @ ncmoe22 / t8 | 516.19 | 21.30 |
| `-fa on` @ ncmoe22 / t8 | 520.83 | 20.54 |
| `-fa off` @ c40+m5120 / t6 | 495.16 | 26.09 |
| `-fa on` @ c40+m5120 / t6 | 492.94 | 25.13 |
| `-ctk/-ctv q8_0` @ ncmoe22 | 518.32 | 21.43 |
| `-lm mmap` @ ncmoe22 | **82.12** | **8.03** |

## 4. 归因分析

### 4.1 static 在 ub ≥ 1024 崩溃的原因

- 实测钉住 compute-buffer 工作集（copy tensors + 激活）的区间：
  - ub512：ncmoe36+m5376 总占用 7573 MiB 仍正常（pp512 525.74），ncmoe36+m5632 总占用 7827 MiB 崩
    → ub512 工作集 ∈ (349, 603] MiB；static ncmoe20（剩 407 MiB）刚好在边界内，所以 pp512 正常。
  - ub2048：mixed c36+m5120 总占用 7314 MiB 正常（878.44），static ncmoe20（7769 MiB）崩
    → ub2048 工作集 ∈ (407, 862] MiB。
- 结论：**static 的高驻留把显存吃到只剩 407 MiB，ub 从 512 提高后激活 + host-weight copy tensors
  超出余量 → 分配成功但设备内存被驱动换页（无异常、性能 3-11x 崩塌）**；
  cache 配置留 2.2 GiB 余量，所以 ub2048 正常（840 t/s）。
- 为什么 ub1024 (141.6) 比 ub2048 (193.6) 更差：换页行为非单调，属驱动侧现象；不是计算量差异。
- 注意：这与"CPU MoE 变慢"无关 —— 同配置 TG（batch<32，走 CPU 路径）仍是 21.94 t/s。
- **推论（待测）**：断崖只出现在"驻留层太多、余量 <~450 MiB"的配置上。static 在 ncmoe22-26
  （余量 1.1-2.5 GiB）时 ub2048 应该能装下，此时 host 层数只有 20-24（upload 0.50-0.60 s、
  drain 20-24 次），PP 有可能超过 mixed 的 878（估算 900-1150）。这是下一步最有价值的测试（§6-A/B/C）。

### 4.2 mixed 击败 cache-only 的原因（TG）

- 驻留层（ncmoe 让 4-6 层专家整层在 GPU）对 TG 是 100% 命中，且**无 slot 查表、无 ids 回读、
  无 per-layer drain**；同时 bank 预算基本不变（host 字节只少 4-6 层的份额），A 组每层 LRU slot 从
  18.1/layer 升到 20.5/layer（§4.3 表）。
- 命中率模型（hit ≈ slot 覆盖率，整层 100%）：cache-only m5120 命中 58.1/152；mixed c36+m5120 命中
  73.9/152 → miss 从 93.9 降到 78.1，即每 token 上传字节 511 → 424 MiB。
- 字节模型（H2D 13.3-13.9 GB/s）：mixed 上限 30.2-31.2 t/s，实测 29.62（95%）；
  cache-only 上限 24.9-26.0，实测 25.82（99-100%，已经贴住 H2D 天花板）。
  即 **mixed 的收益几乎全部来自"少 upload 字节"**，而 cache-only 已无余量。
- 结论：TG 已到 H2D 天花板附近；按每层 4 专家 5.44 MiB 计，只要命中率再涨 X% 就线性涨 X%。
  但命中率受显存限制（bank 大小）与 PP 需求（要留 compute buffer 余量）两头夹，实测平台在 29-30.5 t/s。

### 4.3 slot / 覆盖率表（源码公式复算，`bench/lfm25/mixed_slots.py`；命中/上限为模型值）

| ncmoe | cache | hostA 层 | A slots | 整层 | LRU/层 | A 覆盖 | 驻留层 | bank MiB | 总占用 MiB | 命中/152 | miss/token | H2D 上限 t/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 5120 | 36 | 881 | 5 | 18.1 | 38.2% | 0 | 5119 | 5119 | 58.1 | 94 | 24.9-26.0 |
| 40 | 5632 | 36 | 969 | 5 | 21.0 | 42.1% | 0 | 5623 | 5623 | 63.8 | 88 | 26.4-27.6 |
| 38 | 5120 | 34 | 877 | 5 | 18.4 | 40.3% | 2 | 5110 | 5806 | 65.9 | 86 | 27.1-28.3 |
| **36** | **5120** | **32** | **874** | **5** | **20.5** | **42.7%** | **4** | **5113** | **7314** | **73.9** | **78** | **30.2-31.2** |
| 36 | 5376 | 32 | 918 | 5 | 22.1 | 44.8% | 4 | 5372 | 7573 | 76.9 | 75 | 31.5-32.5 |
| 36 | 5632 | 32 | 961 | 5 | 23.8 | 46.9% | 4 | 5626 | 7827 | 79.8 | 72 | 32.7-33.7 |
| 35 | 5120 | 31 | 872 | 5 | 21.2 | 44.0% | 5 | 5116 | 7665 | 77.9 | 74 | 31.9-32.9 |
| 34 | 4608 | 30 | 783 | 5 | 18.5 | 40.8% | 6 | 4605 | 7741 | 76.1 | 75 | 31.2-32.1 |

（"总占用" = bank + 808.8 非专家 + 驻留层 × 348 MiB；不含 compute buffer / copy tensors。）

### 4.4 双模式特征汇总

| | TG（batch 1 < 32） | PP（ubatch ≥ 32） |
|---|---|---|
| 专家算子在哪 | host 权重层在 **CPU**；驻留层/bank 在 GPU | 全部在 **GPU**（host 权重先拷到 compute buffer） |
| 显存需求 | 权重 + bank + KV + 少量工作集（≤~350 MiB） | 同上 **+ 每层 copy tensor（348 MiB 级同时存活）+ 随 ub 增长的激活** |
| 显存上限（实测） | 总占用可到 ~7827（c36+m5632 的 TG 仍 29.32） | 总占用须 ≲ 7573-7800（ub512），≲ ~7700（ub2048） |
| 主要成本 | H2D miss 字节 + 每层 ids 回读 drain | host 权重上传 + 每层 drain + GPU GEMM |
| 最优配置 | 多驻留层 + 大 bank（同显存预算下权衡） | 中等驻留（保留 compute 余量）+ ub2048 |


## 5. 结论

1. **如果只给一条命令**：`-ncmoe 36 --moe-cache-mib 5120 -ub 2048 -t 6` →
   pp512 531.58 / tg32 29.05 / pp2048-ub2048 878.44（单一配置同时拿到最好的 PP 和接近最好的 TG）。
2. **TG 冠军**：`-ncmoe 36 --moe-cache-mib 5120 -t 6` = 29.62（`-ub` 对 decode 无影响）。
   命中率模型给出 H2D 上限 30.2-31.2 t/s，实测已 95%；再往上需要更大 bank（显存不够）或改替换策略。
3. **PP 冠军**：mixed + ub 2048 = 878.44（优于 cache-only 的 840.19，远优于 static ncmoe20 的 431）。
4. **static 的 ub 断崖是"高驻留把显存吃光"造成的，不是 CPU 变慢**：ncmoe20 只剩 ~407 MiB，
   ub1024/2048 的激活 + copy tensor 超出 → 驱动换页 → 141/193 t/s。
   **中等驻留的 static（如 ncmoe22-26）+ ub2048 预计可行且可能超过 878**（未测，见 §6-B/C）。
5. **cache 上限不是总显存**，而是驱动单次分配上限（~2 GiB，见 §2.4）：m5888/6144 在 ncmoe40 下失败。
6. **must-keep**：resident load（`-lm mmap` 会把 pp512 打到 82）、`-dev Vulkan1`、`-fa on`、
   TG 用 t6、PP 必须 `-ub 2048`。

## 6. TBD（待定/下一步 6 个配置）

优先级：先补 PP 上界（A/B/C：static 中等驻留 + ub2048，从未测过，模型预计可能 > 878），
再确认单一配置冠军（D），最后验证两个机制性假设（E/F）。

| # | 配置（`-dev Vulkan1 -fa on -t 6 -r 2~3`） | 目的 | 预期 |
|---|---|---|---|
| A | `-ncmoe 22 -p 2048 -n 0 -ub 2048` | static 中等驻留 + 大 ub（20 host 层：upload 0.50 s + drain 0.30 s + compute 0.96 s） | **1000-1150**；若溢出则 150-250（一次性判定） |
| B | `-ncmoe 24 -p 2048 -n 0 -ub 2048` | 同上更保守（16 驻留，余量 1.8 GiB） | 900-1050 |
| C | `-ncmoe 26 -p 2048 -n 0 -ub 2048` | 同上更保守（14 驻留，余量 2.5 GiB） | 800-950 |
| D | `-ncmoe 36 --moe-cache-mib 5120 -p 2048 -n 0 -ub 2048 -r 3` | 复现当前 PP 冠军 | 860-895 |
| E | `-ncmoe 36 --moe-cache-mib 5632 -p 512 -n 32` | TG 专用最大 bank（PP 会溢出≈140，TG 不受影响，已验证） | tg 29.0-30.5 |
| F | `$env:GGML_VK_FORCE_MAX_BUFFER_SIZE='6442450944'` + `-ncmoe 40 --moe-cache-mib 6144 -p 512 -n 32` | 验证 ~2 GiB 单次分配上限是否人为 | 成功则 tg 26.5-28；失败则确认硬限 |

补充（有空再跑）：`-ncmoe 36 --moe-cache-mib 5120 -p 4096 -n 0 -b 4096 -ub 4096`
（PP 在 ub4096 的边界，预计 780-850 或溢出）；`-ncmoe 22 -p 512 -n 32`
（static 的 TG 基线 20.54 已知，用于对照 A 的 PP 值）。


## 7. 复现命令（canonical）

```powershell
$B = 'E:\fastLlama\build-vulkan\bin\llama-bench.exe'
$M = 'F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf'
# 单一配置（PP+TG）
& $B -m $M -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048 -p 512 -n 32 -r 3 -o md
# 长 prefill
& $B -m $M -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048 -p 2048 -n 0 -r 3 -o md
# 诊断（看 compute buffer 实际大小与 bank 大小）
& $B -m $M -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -p 512 -n 0 -r 1 -v -o md
```

## 8. 风险与未验证项

- r=2 的差异（m5120 vs m5376 vs m5632 在 ncmoe36 的 TG 差 0.3-1.0 t/s）都在 sd 1.3-1.5 内，
  最终结论必须 r=5 交叉复测。
- "设备内存被驱动换页"是从"无异常 + 3-11x 崩塌 + 显存边界钉住"推断的，未直接观测；
  可用 `GGML_VK_MEMORY_LOGGER=1` 的 `Total device` 峰值佐证。
- `GGML_VK_FORCE_MAX_BUFFER_SIZE` 是否能让 m6144 分配成功未验证。
- 长上下文（32K/128K）未测：KV 会额外占 0.65-2.6 GiB，cache/bank 上限要下调。
- iGPU（Vulkan0）未测；本模型 13.9 GiB 权重理论上可放进 UMA，值得 1-2 次试验。
