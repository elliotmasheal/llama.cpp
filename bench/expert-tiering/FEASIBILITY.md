# 方案 C 可行性论证与预期收益（专家粒度热/冷分层 + 旁路执行器）

日期：2026-10-21。状态：**设计草案，未改任何运行时代码**。按 AGENTS.md「改动与确认」，本文提交用户确认后再动手。

## 0. 结论摘要

- 方案 C 在这台机器上**可实现，但收益结构与原文档的假设不同**：
  - 静态画像 + 价值密度放置：实测命中等同覆盖率（22% 覆盖 -> 22.2% 命中），**单独做没有收益**。decode 的命中率来自 LRU 的
    时间局部性（48% 命中 @ 33% 覆盖），不是来自 profile 排序。
  - 冷专家放 CPU 执行：fork 已有等价实现（`--moe-hybrid`，图内掩码 CPU 分支），实测 **TG 6.1 t/s vs 基线 15.9**，因为每层
    多 2 个 split、每个 CPU split 一次 Vulkan 队列同步，两支从不重叠。真正的层内 CPU/GPU 并发**至今未被成功实现**，
    这是方案 C 唯一未被否证的机制。
  - 流式上传（S 模式）：CUDA 式 host-mapped pinned 在本栈**不可用**（ggml-vulkan 对 AMD 关闭 host buffer import），
    必须显式 staging 拷贝。**Gate 2 已实测：独立 transfer 队列与 compute 队列可以重叠**（隐藏较短臂的 78-99%）。
- **门禁已跑完（2026-10-21）**，两项都通过，见 [GATES.md](GATES.md)：
  - Gate 1：Vulkan 每层边界地板 **0.19-0.21 ms**（预录制+timeline），深度 2 流水 **0.115 ms**；当前调度器是 0.64 ms/split。
  - Gate 2：transfer 与 compute **确实重叠**（90.6 + 104.8 ms 的两臂，串行 195.4 ms，并行 105.5 ms）。
- 预期收益（估算，基于门禁实测值）：
  - TG：26.4 -> **43-51 t/s（+63% ~ +93%）**；上界约 50-60 t/s。
  - PP：`pp2048 ub2048` 578 -> **780-830 t/s**；`pp512` 383 -> 550-680；LFM2.5 `pp2048` 874 -> 1050-1150。
  - **TG200 不是方案 C 能达成的目标**：单 token 前向的稠密权重读地板是 6.41 ms（1.295 GiB / 217 GB/s），
    加上专家字节与边界后单次前向约 20 ms；即使投机解码一次验证 3 个 token，也要每次前向 <= 12.5 ms 才够 200 t/s。
- 建议：按阶段推进（§8）。Stage 2 = 只旁路 MoE 层的执行器原型，先 Qwen3.6；模型重排（§4.8）作为独立一步，
  先做纯排列（不改数值语义），再做冷专家降比特（最后手段，需质量门）。

## 1. 口径与边界

- 目标硬件：RX 6700S 8 GiB（Vulkan，`-dev Vulkan1`）、Ryzen 7 6800HS、39.2 GiB RAM、PCIe Gen4 x16。
  **没有 CUDA 路径**，参考文档里所有 `cudaHostAlloc(mapped)`、`cudaMemcpyAsync`、CUTLASS grouped GEMM 的做法都不能照搬。
- 参考项目 Strata（`E:\Strata`）：其推理引擎是 CUDA/HIP/SYCL/Metal，**`VULKAN_BACKEND_PLAN.md` 明确写「Vulkan：无 LLM 引擎」**，
  Vulkan 只用于图像生成。所以「JR-Strata-Vulkan 从零排 roadmap」在本地源码中无对应实现；本评估不把它当作可移植参考。
  Strata 的实测机是 RTX 5070 12 GB / Ryzen 5 7600 / 64 GB DDR5-5200，模型为 Qwen3.8-Flash-Next（24,576 专家、48 层、top-10），
  与本地 Qwen3.6（256 专家、top-8）不是一个负载区间。
- 本地源码中**未找到**用户原文引用中的「JR-Strata PR #889」「>20 GiB host 分配失败需 4 GiB 分块」「SYCL roadmap 顺序」
  的依据（对 `E:\Strata` 与 `E:\fastLlama` 全文检索 `#889`、`JR-Strata`、`4GiB chunk` 均无命中；Strata 只有一处注释提到
  「one 39 GiB pinned allocation that fails」）。这些说法不作为论据使用。

## 2. 实测基线（只引用可复现来源）

| 项 | 值 | 来源 |
|---|---:|---|
| H2D 实际上限 | 13.9 GB/s（约 96% 利用率） | `bench/moe-baseline/L8-H2D-AND-BANK-DIAGNOSIS.md` §1 |
| 显存 GEMV 带宽 | 217 GB/s（约 97% DRAM 地板） | `bench/moe-baseline/T2-GEMV-CEILING.md` §2 |
| CPU 只读带宽（numpy 探针，4 线程） | 39.7 GB/s | `bench/expert-tiering/membw_probe_result.txt` |
| CPU MoE 有效带宽 | 20.3 GB/s（63 GFLOPS） | `bench/moe-baseline/PROFILE-2026-10-18.md` §2.3 |
| Qwen3.6 canonical pp512 / tg32 / pp2048 | 383-387 / 26.2-26.5 / 578-590 | AGENTS.md 基线表 |
| Qwen3.6 static `-ncmoe 27` pp512 / tg32 | 359-363 / 18.7-19.1 | AGENTS.md 基线表 |
| decode 每 token | 37.6-40.5 ms，32 splits | `bench/moe-baseline/TG-DECODE-BREAKDOWN.md` §1 |
| 其中 copy 总耗时（含边界） | 23-30 ms（cpy_sync 12.3 + h2d body 11-18） | 同上 |
| 每 split 边界成本 | 约 0.37 ms 等待 + 0.27 ms 提交 | `bench/moe-baseline/L2-DECODE-LATENCY.md` §Where |
| decode 每 token 上传字节 / 命中率 | 206 MiB / 60.9% | `bench/moe-baseline/PARADIGM-SLOT-PLAN-GATE.md` §1 |
| ub512 的 touched expert 覆盖率 | 70.0%（179.2/256，122-253） | `bench/moe-baseline/L5-PREFILL-UPLOAD-FLOOR.md` §1 |
| ub2048 上传量 / 占比（10k prompt） | 14.0 GiB/ubatch；上传 21%，计算 79% | `bench/moe-baseline/PARADIGM-HOST-DIRECT-AND-UBATCH.md` §1 |
| 层内重叠理论上限（D1） | 现实 ceiling 29.5-30.5 t/s；理论 41.8 不可达 | `bench/moe-baseline/PARADIGM-D1-GATE.md` §5 |
| PP 上限（cache 配置） | pp512 383-386 | `bench/moe-baseline/L10-DEPENDENCY-CHAIN-LIMIT.md` §4 |

模型维度（本机 GGUF 读出，`bench/expert-tiering/gguf_dims_result.txt`）：

| 模型 | 层 | 专家 | top-k | expert_ff | 专家总字节 | 每专家层 | 每 token 专家字节 | 稠密地板 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B A3B | 41（40 + 1 MTP） | 256 | 8 | 512 | 18.650 GiB | 1.820 MiB | 582 MiB | 6.41 ms |
| LFM2.5 24B A2B | 40（38 MoE） | 64 | 4 | 1536 | 13.058 GiB | 5.498 MiB | 836 MiB | 2.62 ms |

## 3. 原文档前提与实机的差异（这决定了收益结构）

| 原文假设 | 实机事实 | 影响 |
|---|---|---|
| 「512 个 expert 的 packed tensor」 | Qwen3.6 是 **256** 专家（LFM2.5 为 64） | bank/slot 预算按 256 计算 |
| 「80 层，每层一次同步约 50-100 us，合计 4-8 ms/step」 | 32 splits/token，边界实测 **0.64 ms/split，合计约 20.5 ms/token** | 同步是 decode 的第一大项，比原文高 3-5 倍 |
| 「PCIe Gen4 x16 32 GB/s」 | 实测 H2D **13.9 GB/s** | S 模式的带宽预算要打 43% 折扣 |
| 「`cudaHostAlloc(mapped)` 零额外拷贝」 | `ggml_vk_host_buffer_import_usable()` 对 AMD 返回 false（`ggml-vulkan.cpp:4035-4038`，注释：amdgpu 每次提交重缺页所有导入页）；把 host buft 接进 `supports_buft` 的实验直接崩溃 | S 模式必须走 staging；「GPU 直读 host 权重」在本机被驱动堵死 |
| 「热专家高精度、冷专家低比特」 | 8 GiB 下把热专家提到 fp16 会 3 倍占用，反而降低覆盖率 | 方向建议反转：热专家维持 Q4_K，冷专家降比特（需过质量门） |
| 「头部 30% 专家覆盖 60-70% 激活」 | 本模型无此实测；ub512 的 touched 覆盖已达 70%，说明路由分散；静态驻留命中等于覆盖率 | 依赖该假设的放置策略要先验证再投入 |
| 「prefill 时静态放置足够」 | ub2048 每 ubatch 上传 14 GiB，约占专家总量 75% | prefill 几乎触及全部专家，「冷专家」在 PP 里基本不存在 |

## 4. 方案 C 各部件逐项可行性

### 4.1 热度画像与放置决策：**否证（静态部分）**

- fork 已有等价实现：`count_usage()` 在 prefill 大 batch 统计专家选择数，`build_residency()` 按计数为每层上传 top-N 常驻
  （`src/llama-moe-cache.cpp`）。实测命中 22.2%，覆盖 22.0%，即**命中约等于覆盖率**。
- 同一文档指出 65.4% @ 22% 覆盖率的历史数字来自 LRU 的时间局部性，不是静态排序。
- 结论：价值密度公式 `value/cost` 对这个模型没有可分配的空间；应把精力放在**淘汰策略**（LRU -> SLRU/ARC）而不是静态排序。
  淘汰策略不改数值，属于低风险高性价比项。
- 异构量化的方向正确但需反转（冷专家降比特），且改变数值，必须先过 KLD/PPL/生成一致性质量门。

### 4.2 数据布局：**不需要拆张量，fork 已用逻辑视图**

- fork 用「bank + 每层 slot map」实现逻辑拆分：设备上是一块连续 bank，host 侧 slot map 描述每个 expert 在 bank 的槽位，
  MoE 通过 `ggml_get_rows(slot_map, ids)` 间接寻址（`src/llama-moe-cache.cpp`、`src/llama-graph.cpp` `build_moe_cache_slots`）。
- 因此原文的「物理拆分 vs 逻辑拆分」在这里已解决，不是方案 C 的增量。

### 4.3 执行路径（token 分桶 + grouped GEMV）：**有新工作，但不是瓶颈**

- CPU 侧 `MUL_MAT_ID` 已经是 expert-major（`ggml-cpu.c` 外层按专家），GPU 侧走 MMQ + `mul_mat_id` shader。
  decode 时每专家只有 1-几个 token，分组收益有限；真正的耗时项是上传与边界。
- 若要 per-expert 放置到不同后端并发执行，需要新 kernel 与跨后端结果合并；这是方案 C 的实现主体，但不改变总字节与总边界数。

### 4.4 H 模式（冷专家 CPU 执行）：**图内版本已被实测否证，层内真并发未证**

- 图内 hybrid：TG 6.1 vs 15.9，122 splits，`cpy_sync` 72.5 ms/token。根因是两支从不重叠，CPU 分支在 GPU 分支 drain 之后才跑。
- Strata 的该机制能成立，是因为它自建执行器：GPU 算命中的专家、CPU 就地算未命中的专家，两者同时进行。
  本 fork 的执行器是 ggml scheduler，做不到；这正是方案 C「旁路」的合理动机。
- 但本机 CPU 有效带宽只有 20.3 GB/s（我的只读探针上限 39.7 GB/s），冷专家 CPU 计算的时间约为同字节上传时间的 1.3 倍以内：
  206 MiB -> 上传 15.5 ms vs CPU 10.6 ms（20.3 GB/s）。两者接近，H 模式的价值主要在「可与 GPU 数学并发」，
  而不是「比上传快很多」。CPU kernel 若能从 20.3 提到 30+ GB/s（LFM 的 CPU 多线程扩展只有 2.0x，官方 4.0x，见
  `bench/lfm25/COMPARE-OFFICIAL.md` §6.3），H 模式的底子会明显改善，这是独立于方案 C 的便宜杠杆。

### 4.5 S 模式（流式上传）：**机制可用，重叠未证**

- 不可用：host-mapped 导入（AMD 驱动重缺页）。
- 可用：显式 H2D 拷贝 + 双缓冲。fork 的 `--moe-cache-mib` 已经做「按需上传」，L8 测到 H2D 已贴 96% 上限。
- 未证：传输与计算重叠。L5 试过 2-slot 轮换目的地 + event 门控，pp512 215.1（基线 347），失败原因是每 split
  `event_record` 破坏 Vulkan 后端跨 split 的命令缓冲复用（实测 3 ms/split，90 splits 约 0.25 s）；L9 归因于 ids 回读的全设备
  drain 把流水深度限制在一层。L5 明确写「没有观察到重叠收益」，但**没有下「硬件不能重叠」的强结论**，这是方案 C 唯一真正
  值得投入去攻的点。
- 补充：独立传输队列在本机**默认已启用**。`allow_graphics_queue` 默认 false（`ggml-vulkan.cpp:4336`），因此
  `prefers_transfer_queue` = AMD && 非 GCN && 非 UMA && !allow_graphics_queue 成立（`ggml-vulkan.cpp:4924-4935`），
  RX 6700S 会创建独立的 transfer-only queue 且 `async_use_transfer_queue = true`。
  所以 L5 观察不到重叠的原因在调度器的全设备 drain 与驱动实际行为，不是「没有第二个队列」。

### 4.6 同步点：**决定性瓶颈，且方案 C 也不能完全消除**

- 每层一次 ids 回读是硬依赖（ids 依赖上一层 MoE 输出），只要「按需上传」就必须每层同步。
- D1 门禁给出：在现有提交模型下，即使层内重叠全免费，TG 上限 29.5-30.5 t/s；理论极限 41.8 t/s 需要「上传全隐藏 + 零握手」，
  已判定不可达。
- 方案 C 的潜在增量在「握手成本」：把调度器的每 split 分配/提交/drain 换成预录制命令缓冲 + timeline semaphore 定向等待。
  L10 实测定向等待原语本身只要 0.028 ms，但调度器结构决定了收益有限。

## 5. 收益模型（估算，标注假设）

decode 每 token 的实测构成（同一配置 `-ncmoe 41 --moe-cache-mib 6144`）：

```
GPU 数学        约 12 ms（已贴 DRAM 地板，kernel 优化最多 1-2%）
专家上传        约 15.5 ms（206 MiB / 13.9 GB/s，当前串行）
边界/同步       约 20.5 ms（32 x 0.64 ms）
实测合计        37.6-40.5 ms -> 26.4 t/s
```

| 场景 | 假设 | 估算 t/s | 相对 26.4 |
|---|---|---:|---:|
| A 现状（实测） | - | 26.4 | - |
| B 只降边界 | Gate 1 变体 C：0.19 ms/层 x 40 = 7.6 ms，上传仍串行 | 12+15.5+7.6 = 35.1 | +8% |
| C 上传与计算重叠（Gate 2 已证） | max(12, 15.5) + 7.6 = 23.1 ms | **43.3** | **+64%** |
| D 冷专家改由 CPU 并发计算 | max(12, 10.6) + 7.6 = 19.6 ms | **51.0** | **+93%** |
| E 上界（边界用 Gate 1 变体 D） | max(12, 15.5) + 4.6 = 20.1 ms | 49.8 | +89% |

（2026-10-21 修订：上表用 Gate 1/2 实测值替换了原先“边界 20.5 ms、上传不能重叠”的假设，明细见 [GATES.md](GATES.md)。）

prefill（ub2048，cache 配置）：

| 场景 | 依据 | 估算 |
|---|---|---|
| 现状 pp2048 ub2048 | 实测 578（2048 token 用 3.54 s） | 上传 14.0 GiB = 15.0 GB / 13.9 GB/s = 1.08 s（约 31%），计算约 2.46 s |
| 上传全隐藏 | 理论上限 = 计算时间 2.46 s | 578 -> 832（+44%）；10k prompt 的实测占比 21% 给出 +27% 的保守下限 |
| 保守合并 | 隐藏 50-70% 上传 | 578 -> 700-790（+21% ~ +37%） |
| pp512 | L8 的安全估计 480-550（现状 383）；Gate 2 支持上限约 680 | 550-680 |
| PP1000 | 需要 578 -> 1000（+73%） | 方案 C 不够；还需 MoE GEMM 效率（现 2.6-3.1 TFLOPS）或更多驻留 |

关键判断：**PP 的收益在「重叠」，不在「热/冷分层」**。prefill 每 ubatch 上传 14 GiB，已经接近全部专家（75%），
没有可分层空间；而 decode 的收益在「边界 + 命中率 + 冷专家并发」，其中静态分层已被否证，只有动态淘汰与层内并发有效。

## 4.7 基座放置：按层驻留 -> 按专家驻留

现状有两层放置：`-ncmoe N` 决定哪些层的专家放 host（**按层**），`--moe-cache-mib` 在其上做**按专家**的 LRU。
缓存只对有 host 专家的层生效，加载日志显示 20/41 层、1690 slots、33% 覆盖；整层驻留层（每组最多 5 层）走静态视图，
不需要 ids 回读、也不产生边界。

改为按专家驻留的收益与代价：

- 预算利用率更高：256 个专家独立放置，不必成层分配。当前 6144 MiB 下 33% 覆盖已是最优区间；
  旧实验里“每组 6 个整层”会让 TG -17%（`bench/moe-baseline/TG-DECODE-BREAKDOWN.md` §6b），说明整层分配浪费预算。
- 代价：每层都可能有 miss，于是每层都要一次 ids 回读与上传决策。Gate 1 实测该边界 0.19 ms/层，40 层约 7.6 ms/token，可接受；
  但整层驻留层原本是零边界，所以**保留整层驻留作为快路径**：整层放得下就整层静态驻留，放不下才进 per-expert 决策。
- 结论：不是二选一，而是“按专家驻留为默认 + 整层驻留为零边界快路径”，两者共存。
- 解码与 prefill 需要不同划分：prefill 每 ubatch 上传约 75% 的专家，per-expert 决策没有收益；执行器按 ubatch 大小切换策略。

## 4.8 重排模型结构（repack）

现状：专家是 packed 3D tensor `[n_embd, n_ff, n_expert]`，按 expert 连续存放。repack 按风险从低到高有四件事：

1. **层内按热度重排专家 + 同步置换 router 行**（`ffn_gate_inp`）：热专家连续后，一次上传 k 个热专家是 1 段连续拷贝而非 k 段散拷贝，
   也改善预取顺序性。数值影响只有加权求和的累加顺序（FP32 级），需要质量门但不是重新量化。
2. **按共激活分组**：把常被同一 token 选中的专家放相邻，进一步减少散拷贝段数。数据可用现有 `count_usage()` 的共现统计生成。
3. **拆成 per-expert tensor（物理拆分）**：搬运粒度 1.82 MiB，ring 只需几十 MiB（而不是当前整 tensor 210 MiB、双缓冲 420 MiB），
   也避开单 VkBuffer 2 GiB 上限。代价：加载器与 `mul_mat_id` 的 id 索引路径要改（现 kernel 认 packed 布局），改动面最大。
   建议先做 1（纯排列，不改布局），把 3 留到 kernel 工作确认需要时再做。
4. **冷专家降比特（最后手段）**：IQ3 约 3.1 bpw vs Q4_K 约 4.5 bpw，上传字节降约 30%，VRAM 可多放约 40% 专家。
   改变数值，必须走 KLD/PPL/生成一致性质量门并单独确认。

顺序建议：1 -> 2 ->（执行器原型验证后再决定 3/4）。repack 产物写到 `F:\LLM\quants\` 独立目录，记录生成命令与 SHA256。

## 6. 决定性的门禁（已完成，结果见 [GATES.md](GATES.md)）

### Gate 1：边界成本地板（决定方案 C 的 decode 上限）

写一个独立微基准（`bench/expert-tiering/`，不进 llama.cpp）：在 Vulkan1 上每层提交约 14 MiB 权重读的代表性 GEMV
（8 个专家 x 1.82 MiB）+ 1 次 6.7 MiB H2D（216 MB / 32 层），测量 32 次「提交 -> 等待 -> 回读 ids -> 下一迭代」的每边界成本，
分三种实现：
（a）现状等价：每 split fence + 全设备等待；（b）预录制命令缓冲 + 单 fence；
（c）timeline semaphore 定向等待（不等整个设备）。

- 通过线：0.15 ms/split 以下 -> decode 有 +25% 以上空间。
- 不通过（>= 0.4 ms/split）：方案 C 的 decode 收益 <= +12%，**不再投入执行器**，转做淘汰策略与 PP 重叠。
- **实测结果（2026-10-21）**：预录制+timeline 0.191 ms/层，深度 2 流水 0.115 ms/层 -> **通过**（细节见 [GATES.md](GATES.md)）。

### Gate 2：传输/计算重叠是否存在（决定 PP 收益，也是方案 C 的核心假设）

独立微基准：compute 队列上循环跑 4 s 量级的 GEMV，同时在 transfer 队列上持续 H2D 14 GiB，用 timeline semaphore 同步，
测量总时间是否 <= max(4.0, 1.0) + 小量。

- 通过：PP 可拿 +27% ~ +38%，pp512 450-550，方案 C 的 MoE 执行器值得做。
- 不通过（总时间约等于串行和）：PP 收益上限归零，方案 C 只剩边界优化，收益 <= +12%，应停止。
- **实测结果（2026-10-21）**：90.6 ms 计算 + 104.8 ms 传输，串行 195.4 ms，两队列并行 **105.5 ms**（隐藏 99.3%）-> **通过**。

### Gate 3：淘汰策略离线重放（低风险、数值不变）

用现有 hybrid 的 `count_usage()` 落一份路由使用轨迹（加一个环境变量门控的 dump，不改语义），离线对比 LRU / SLRU / ARC /
TinyLFU 在 33% 覆盖率下的命中率与上传字节。目标：60.9% -> 75%+，即上传 15.5 -> 10 ms，TG +14% 左右。

### Gate 4：质量门（任何改变数值的改动）

- 混合量化（冷专家降比特）或 H 模式并发改变 kernel 数值：先跑 KLD + PPL + 生成一致性 + `test-backend-ops` 回归。
- 参考：`bench/moe-baseline/HYBRID-STATUS.md` 的 PPL 表（all-CPU 6.1161 vs LRU 6.1973），说明 CPU/GPU kernel 数值差异已经是
  可测的，不能当作「纯布局等价」。

## 7. 风险与回滚

| 风险 | 说明 | 缓解 |
|---|---|---|
| 重叠不存在 | Gate 2 已实测两队列可重叠（隐藏 78-99%）；但执行器里 40 层连续调用可能引入新的串行点 | 原型阶段先接 1 层再扩到全部 40 层，保持可回退开关 |
| 显存更紧 | 双缓冲需要额外 VRAM；cache 6144 已到上限（>=6400 崩到 57 t/s），单 VkBuffer 上限 2 GiB | 以 expert 粒度（1.82 MiB）做 ring，不要以整 tensor（210 MiB）为单位 |
| 数值回归 | H 模式、混合量化都改数值 | Gate 4；任何改动先留参考 logits |
| 维护两份计算路径 | 只旁路 MoE、attention/dense 仍走 ggml，风险可控；若整个前向旁路则风险高 | 限定范围：仅 MoE 层 |
| 进度 | fork 已有 4 个调度级尝试全部未胜出；执行器是更大改动 | 门禁已通过；原型按阶段交付，每阶段留回退开关 |

回滚：所有原型放在独立分支 + `bench/expert-tiering/` 脚本，不改 `F:\LLM\llm\` 原模型；运行时代码改动以环境变量门控，
默认路径保持当前 canonical 配置。

## 8. 建议路线（分阶段）

1. 第 0 阶段（**已完成**）：Gate 1 + Gate 2 微基准，两项通过（[GATES.md](GATES.md)）。
2. 第 1 阶段（3-5 天，低风险）：Gate 3 淘汰策略重放（LRU -> SLRU/ARC）；若命中率能到 75%+，先在现有 `llama-moe-cache` 换策略。
3. 第 2 阶段（1-2 周，主投入）：只旁路 MoE 层的执行器，先 Qwen3.6。范围见下：
   - 保留 ggml 图负责 attention/dense/GDN/shortconv，MoE 层由执行器接管：router -> 读 ids -> 逐 expert 分派（GPU 命中 / 上传 / CPU）-> 合并。
   - 传输用独立 transfer 队列 + timeline 信号量，预录制命令缓冲；ring 以 expert 粒度（1.82 MiB）为单位。
   - 冷专家先走上传统一模式（staging 上传），CPU 并发作为开关后加，避免一次引入两个变量。
   - 验收：TG 43+ t/s、pp2048 780+ t/s、质量门通过；达不到则按开关回退到 canonical 路径。
4. 第 3 阶段：模型重排（§4.8 的 1 -> 2），然后视需要做 3（物理拆分）与 4（冷专家降比特，最后手段）。
5. 第 4 阶段：视数据决定是否扩展到 LFM2.5、是否做混合量化（需再次确认）。

## 9. 复现

```powershell
# CPU 带宽探针（本文 §2 的 CPU 39.7 GB/s）
python bench\expert-tiering\membw_probe.py 512

# GGUF 维度（本文 §2 的模型表）
python bench\expert-tiering\gguf_dims.py F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf

# Gate 1 / Gate 2（见 GATES.md）
cd bench\expert-tiering\gates
glslc -O gemv.comp -o gemv.spv
glslc -O ids.comp  -o ids.spv
g++ -O2 -std=c++17 vkgate.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o vkgate.exe
.\vkgate.exe boundary 500
.\vkgate.exe overlap 1400 200
```

结果已落盘：`bench/expert-tiering/membw_probe_result.txt`、`bench/expert-tiering/gguf_dims_result.txt`、
`bench/expert-tiering/gates/gate1_boundary_result.txt`、`bench/expert-tiering/gates/gate2_overlap_result.txt`。
