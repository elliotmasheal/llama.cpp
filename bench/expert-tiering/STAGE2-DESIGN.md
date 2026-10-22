# Stage 2 设计：只旁路 MoE 层的 host 驱动执行器

日期：2026-10-21。状态：M0 实现中（用户已确认开始 Stage 2）。目标模型：Qwen3.6 35B A3B。

## 0. 目标与验收（可判定）

| 项 | 现状（实测） | 目标 | 判定方式 |
|---|---:|---:|---|
| TG（tg32，canonical cache 配置） | 26.2-26.5 t/s | **>= 43 t/s** | `llama-bench -p 512 -n 128 -r 3`（AGENTS 口径），交叉 r>=3 |
| PP（pp2048 ub2048） | 578-590 t/s | **>= 780 t/s** | 同口径 |
| 质量 | 参考 = 改动前同一 GGUF | KLD/PPL/生成一致性不劣化 | `--kl-divergence` + `llama-perplexity` + 固定 prompt 贪心 |
| 算子回归 | 945/939/2275/5332 | 不变 | `test-backend-ops`（AGENTS 基线） |
| 回退 | - | 关闭开关后 == 改动前 | 同一命令对比 ±3% |

任何一项不达标即按里程碑回退，不进入下一阶段。

## 1. 架构（ADR）

**决策：只旁路 MoE 层，其余全部留在 ggml 图。**

理由：
- 实测瓶颈集中在 MoE 的两处：每层 ids 回读后的 32 次全设备 drain（decode 37.6-40.5 ms 中 copy/边界占 23-30 ms）和
  每 ubatch 14 GiB 的专家上传（prefill 21% 时间）。attention/dense/GDN/shortconv 的 GPU 数学已贴 DRAM 地板
  （217 GB/s，`bench/moe-baseline/T2-GEMV-CEILING.md`），没有优化空间。
- 全前向旁路会重写 40 层的图构建、KV、位置编码，风险与收益不成比例。
- Gate 1/2 已证：Vulkan 每层边界地板 0.19 ms（流水 0.115），独立 transfer 队列与 compute 队列可重叠（隐藏 78-99%）。

**执行器职责（每层 MoE）**：
```
router 子图（gap_inp matmul + softmax + top-k） -> ids 回读
  -> 逐 expert 分派：
       hot: 从 bank 直接算（设备侧，slot map 间接寻址）
       miss: 异步上传到 bank 的槽位（transfer 队列），或（开关）交 CPU 线程池
  -> 加权合并（GPU）
```

**相邻层不流水**：ids(L+1) 依赖 MoE(L) 的输出，这是模型的数据依赖，不是调度器能重排的
（`bench/moe-baseline/L10-DEPENDENCY-CHAIN-LIMIT.md`）。执行器只消除同一层内的同步与调度开销，
以及层内 miss 上传与 hot expert 计算的 overlap。

## 1.1 M2 进展（2026-10-21）：执行器模块已进入 libllama

- 新增 `src/llama-moe-exec.{h,cpp}`，CMake 已接入。默认不参与任何路径；`LLAMA_MOE_EXEC_SELFTEST=1`
  在首次 decode 的图计算之后跑一次 40 层的执行器循环并打印测量值。
- 实测（Qwen3.6，cache 6144，真实权重）：**0.503 ms/层**，其中 router+ids 0.336 ms、专家提交 0.163 ms；
  40 层折合 **20.1 ms/token** 的 MoE 成本。与独立 PoC（0.51-0.58 ms/层）一致。
- 关键发现：模型权重张量的 `data` 指针在**上下文构造与析构阶段都是 null**，只有在 decode 图计算后才有效
  （`gate_inp` 在 Vulkan1、`gate_exps` 在 Vulkan_Host）。因此执行器必须挂在 decode 路径内，不能在
  llama_context 构造期做任何权重搬运；自测钩子据此放在首次图计算之后。
- 含义：MoE 部分 20.1 ms + attention/dense 约 12 ms（串行）≈ 32 ms，约 31 t/s；要达到 43 t/s，下一步必须
  把 miss 上传/冷专家 CPU 计算与热专家 GPU 计算重叠（并发已实测隐藏 80.4%，见 M1-POC.md 的 M2 风险探针）。
  这需要图分段（每层 attention 段 + 执行器 MoE 段），是 M2 的下一增量。

### 1.2 分阶段耗时拆解（2026-10-21，同一钩子）

| 阶段（ms/层） | 值 |
|---|---:|
| router 计算 + 同步 | 0.112 |
| ids 回读 + 同步 | 0.085 |
| 专家计算 + 同步 | 0.339 |
| 组合循环（router -> ids -> 提交专家） | 0.486 |

结论：**执行器相对现有路径新增的固定开销 = router + ids ≈ 0.197 ms/层 = 7.9 ms/token**；
专家计算 0.339 ms/层在现有路径里同样存在（属于每 token 的 GPU 数学，约 12 ms/40 层）。
所以执行器的收益来源是「用 7.9 ms 的边界开销替换现有 23-30 ms 的 copy/drain 路径」，投影约 20 ms/token（~50 t/s）；
组合循环里 router+ids 显示 0.386 是因为它同时等待上一层的专家计算，真实接入后这段等待由依赖链决定，不可避免。

### 1.3 现有开关能否替代架构改动（A/B，2026-10-21）

同一 session、同一构建、`-p 512 -n 32 -r 2`（当前电源状态基线 pp512 332.7 / tg32 21.89）：

| 配置 | pp512 | tg32 |
|---|---:|---:|
| `-ncmoe 41 --moe-cache-mib 6144`（基线） | 332.74 | 21.89 |
| 同上 + `LLAMA_MOE_DEFER_UPLOAD=1` | 335.47 | 21.91 |

结论：deferred upload 对 decode 没有可测收益（在噪声内）。现有调度/缓存开关已经到顶，
TG 的提升必须来自执行器架构（去掉 copy/drain 路径），而不是调参。

### 1.4 质量门现状（2026-10-21）

| 检查 | 结果 |
|---|---|
| `test-backend-ops -o MUL_MAT_ID -b Vulkan1` | 939/939 通过 |
| `test-backend-ops -o MUL_MAT_ID -b CPU` | 945/945 通过 |
| `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` | 见 bench/expert-tiering/tbo-flashattn-vulkan.log |
| `test-backend-ops -o MUL_MAT -b Vulkan1` | 见 bench/expert-tiering/tbo-mulmat-vulkan.log |

运行时改动（轨迹统计、执行器自测）都是环境变量门控、默认关闭，算子回归与改动前一致。
（FLASH_ATTN_EXT 5332 项耗时过长，本轮中途停止，日志保留在 `tbo-flashattn-vulkan.log`，后续轮次补跑。）

### 1.5 决定性负结果：执行器路线达不到 TG 43（2026-10-21）

在同一自测里加入「每层 5.2 MiB 上传」的相位（206 MiB/token ÷ 40 层），上传源分别用可页内存和
模型 Vulkan_Host 权重（pinned）测：

| 相位（ms/层） | 可页源 | pinned 权重源 |
|---|---:|---:|
| F 只上传，上传前设备空闲 | 1.033 | 0.525 |
| E router+ids 同步 -> 上传 -> 专家提交 | 1.598-1.777 | 1.027 |
| G 先提交专家计算，再上传 | 1.149 | 0.772 |

解读（pinned 列）：
- 5.2 MiB 单层上传实测只有约 **10.4 GB/s**（0.525 ms），低于整批上传的 13.9 GB/s；
- E/G 说明上传与专家计算的重叠很有限：E 只比串行和（0.238+0.525+0.385=1.148）省 0.12 ms；
  G 省约 0.37 ms，但每层仍要 0.772 ms。
- 按 G 的最优形态：40 层 MoE = **30.9 ms/token**，再加 attention/dense 约 0.3 ms/层 → 全 token 约 43 ms，
  也就是约 **23 t/s**，比现状 26 t/s 更差。

根因不是执行器实现，而是**每 token 的专家上传字节本身**：206 MiB/token 在 10-14 GB/s 下就是 15-21 ms，
且它位于每层的关键路径上，可用来重叠的专家计算每层只有约 0.385 ms。执行器能省掉的是调度器 drain，
但省不掉这 15-21 ms；而现有路径的 23-30 ms 里已经包含了这部分上传。

**结论：TG 43 不能靠「旁路 ggml 的执行器」达成。** 要把 token 时间压到 23 ms 以下，必须减少上传字节：
- 提高命中率（淘汰策略 LRU -> SLRU/ARC，Gate 3；60.9% -> 75% 可把上传降到约 10 ms/token）；
- 或减少每层上传的段数/字节（重排 + 热前缀 + 冷专家降比特，M3/M5）；
- 或投机解码摊薄每 token 的前向次数（与本目标正交）。

建议把投入从「图分段执行器」转向「命中率 + 重排」路线；执行器模块与其测量钩子保留（env 门控），
作为后续若换更大显存/更高命中率时的验证工具。

### 1.6 SLRU 淘汰策略：已实现，但在本负载无可测收益（2026-10-21）

用户要求先做 runtime 优化、暂不叠 MTP。检查结果：本仓库所有 ref（master、moe-gpu-cpu-hybrid、
p1-moe-cache-async、wip-master-2026-10-18、origin/master、origin/vulkan/signed-fwht-fusion）**都没有 SLRU**；
上游 GitHub 在本环境不可访问。于是按 SLRU 的标准定义在 `src/llama-moe-cache.cpp` 里实现：

- `LLAMA_MOE_SLRU=1` 打开；`LLAMA_MOE_SLRU_PROT=<0..1>` 设保护段比例（默认 0.5）；
- 新专家先进 probation 段；probation 命中则晋升 protected 段；淘汰取 probation 的 LRU，
  保护段超限时把其中最久未用的降级；保持原 LRU 的 touch 顺序与合并上传逻辑不变。

实测（同 session、同一构建）：

| 配置 | llama-bench tg32 | 真实生成（-n 128, temp 0） |
|---|---:|---:|
| LRU（默认） | 21.89 | 21.2 t/s |
| SLRU 0.5 | 21.94 | - |
| SLRU 0.7 | 22.04 | 21.3 t/s |

结论：**SLRU 在该负载上没有可测收益**（差异在噪声内）。注意 llama-bench 的 tg 用随机 token，
会高估缓存收益（AGENTS 已注明），所以这里以真实生成为准。原因推测：本配置的 LRU pool 只覆盖 20 层、
命中率已由时间局部性主导（60.9%），两段式保护没有额外可挖的层次。该实现保留在代码中（默认关闭），
换负载/换 cache 配置时可再测。

### 1.7 不叠 MTP 的投机解码：ngram-mod 实测（2026-10-21）

`--spec-type ngram-mod --spec-ngram-mod-n-max 16 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-match 16`
（仓库 PROGRESS.md 记录的最优点），真实生成 `-n 128 --temp 0`，同 session：

| prompt | 基线 Generation | ngram-mod Generation |
|---|---:|---:|
| 短 prompt "Hello" | 21.2 t/s | 21.2 t/s |
| 代码型（函数定义续写） | 19.5 t/s | 18.5 t/s |
| 重复型（显式重复指令） | 19.4 t/s | 19.4 t/s |

结论：在这三个 prompt 上 ngram-mod **没有收益**（代码型甚至 -5%），与 PROGRESS.md 的结论一致：
它只在回复真正重复上下文（代码编辑/引用）且命中率高时才有 2.25x 级别的收益。仓库实测里
`n_max=16` 是必要配置、默认 64 是负优化，这里的测试已用最优配置。
（一次代码 prompt 的 Prompt 出现 56.9 vs 17.5，复测后确认是短 prompt 首 token 延迟的run 间波动，不是真实提升。）
因此「纯速度」路线下，ngram-mod 只能作为特定文本类型的开关，不能当作通用 TG 杠杆；
通用提升仍需 MTP/独立 draft（用户已决定后期再叠）。





## 2. 里程碑

| 里程碑 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| M0 | 路由轨迹落盘（`LLAMA_MOE_TRACE`） | - | 二进制轨迹 + 离线分析脚本；重排与淘汰策略的数据基础 |
| M1 | 执行器 PoC（`pocs/moe-exec`，不接运行时） | M0 | 真实权重下每层边界 <= 0.25 ms，与 sched 同口径对比 >= 2x |
| M2 | 运行时接入（decode，开关回退） | M1 | TG >= 43 t/s、质量门通过、开关关闭等价 |
| M3 | 重排（需真实轨迹） | M0 + 真实使用数据 | 纯排列 GGUF + 质量门 |
| M4 | prefill 重叠（可选，二期） | M2 | pp2048 >= 780 t/s |
| M5 | 冷专家降比特（最后手段） | M3 | 上传字节降约 30%，质量门通过 |

## 3. 实现约束

- 复用现有资产：`llama_moe_cache` 的 bank / slot map / LRU / 整层静态驻留、`ggml_backend_tensor_set_async` 的
  异步拷贝、ggml-vulkan 的独立 transfer 队列。不重复实现缓存。
- 执行器不新增后端、不新增 ggml op；用 backend API 直接计算小型子图，绕开 `ggml_backend_sched` 的分 split 分配与 drain。
- 显存：银行 6144 MiB 已到上限（>=6400 崩到 57 t/s），单 VkBuffer 上限 2 GiB。ring 按 expert 粒度（1.82 MiB），
  不用整 tensor（210 MiB）；不新增超过 256 MiB 的常驻缓冲。
- 开关：`cparams.moe_exec` + CLI `--moe-exec`，默认关闭；关闭时不得改变任何现有行为与数值。
- 数值：执行器复用同一 kernel 与量化类型时，结果应与现有路径在 FP 误差内一致；任何布局/量化改动按 Gate 4 过质量门。

## 4. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 真实每层工作比微基准大，边界收益被抵消 | M1 用真实权重测；边界 > 0.25 ms 则停止 |
| 40 层连续调用引入新的串行点（分配、描述符复用） | 预创建每层资源；M1 先 1 层再 40 层 |
| 数值偏差（CPU/GPU 路径或上传时序） | 开关关闭等价测试 + 质量门 |
| VRAM 超限 | 固定 cache 6144 + ring <= 64 MiB，先做内存预算表再编码 |

回滚：所有改动在独立分支；开关默认关闭；`bench/` 下留同口径前后对比。

## 5. 重排的数据前提（用户提出）

现有数据全部来自 bench 语料（静态计数命中≈覆盖、LRU 48%@33%），**没有真实使用轨迹**，因此：
- 不先做重排；先落 M0 的轨迹采集。
- 采集口径：真实使用（llama-server / llama-cli 的日常对话与代码任务），至少 1 万生成 token。
- 用轨迹统计：层内热度分布（Gini / top-k 覆盖）、共激活（同 token 同层共选频次）、跨 token 时间局部性。
- 只有确认“热度显著偏斜且跨会话稳定”时，纯排列重排才有依据；否则重排只会把 LRU 已经拿到的局部性变成静态损失。
