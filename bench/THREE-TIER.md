# THREE-TIER - VRAM + RAM + SSD 的三层协同

数字见 `RESULTS.md`，判决见 `DECISIONS.md`。本文件只答"三层怎么分、缺口在哪、下一步做什么"。

## 1. 三层角色与硬边界

| 层 | 放什么 | 能否进 decode 单步 |
|---|---|---|
| VRAM | dense、命中专家 bank、KV 工作集、compute | 是 |
| RAM | 整模型 resident / pinned（预取前提）、miss 来源、`-nkvo` 的 KV | 是（H2D 14.3 GB/s） |
| SSD | 冷专家容量层 + 权重 / KV 溢出 | **否**：随机 0.81 GB/s，206 MiB/token = 258 ms |

SSD 只允许出现在预取批次与 prefill，且只服务命中率 >= 85% 的冷层。这就是"三层协同"的工程含义。

## 2. 档位 x 量化（2026-10-29 用户口径）

| 显存档 | 主用档位 | bank | resident 门槛 |
|---|---|---|---|
| 8176 | Q4_K 全系列（Q4_K_XL / IQ4_XS） | 2.3-2.9 GiB | 28-30 GiB |
| 6144 | Q3_K | 1.6-1.9 GiB | 约 22 GiB |
| 4096 | Q2_K | 装不下双环，命中趋零 | 约 16 GiB |

档位是文件选择，不是引擎行为；同一二进制换文件即可。resident 是前提，mmap 下 bank 与预取强制 0。

## 3. 现状判定（代码级）

1. 只有两层半：专家必须整层常驻 RAM 才被缓存，全树没有 SSD / file-backed 专家路径。
2. RAM 不足即失去全部加速：mmap / dio / 不够 resident -> bank 0 -> 预取关。
3. RAM 不是计算层（现状）：图内 hybrid 可跑，但 CPU 就地带宽 20-25 GB/s 仍低于"搬运 + 命中"的组合收益，见判决 10。
4. KV 不参与规划：分配顺序 KV -> bank -> compute，score 缓冲与 256K 的 KV 还没进同一预算式（账本 v1 已覆盖，算子在 fit）。
5. 容量不足与运行期失败不再 abort：`llama-moe-cache.cpp` 已无 `GGML_ABORT`，退化为 `n_plan_fallbacks` /
   `promote_fallbacks` / `hybrid_failures` / `n_split_degrades` 计数。
6. 常数是 8 GiB 单机形状：max_batch 32、pf_min_tokens 512、max_full 5、ring 2 槽；有环时 `pf_max_full` 默认 0。
7. 一个开关管两层放置：`-nkvo` 同时把 KV（O(n_ctx)）和 GDN 状态（62.81 MiB 常数）丢到 host。

## 4. 缺口（F0-F13，一行一条）

- **F0 档位可测**：已落地（`--store-vram-mib` + `vklimit-probe`）。
- **F1 容量不足退化**：已落地（0.17.9 / 0.17.10）：`llama-moe-cache.cpp` 无 `GGML_ABORT`，plan / rolling /
  hybrid worker 全部退化 + 计数，触发轮输出可复现（无触发轮逐字同）。
- **F2 KV 与 bank 同一预算式**：已由 `--fit-plan 1` + 账本覆盖，`-nkvo` 验收臂已测。
- **F3 预取门槛自适应**：6144 档已关闭（+29.3%）；4096 档仍未开，属缺陷。
- **F4 RAM 参与计算**：判据重开（20-25 > 14.3 GB/s），缺 Vulkan `MUL_MAT_ID_MASK`；P1 候选。
- **F5 SSD 容量层**：未开工；前置是 repack KLD = 0。
- **F6 质量门**：已落地，`bash bench/tools/quality-gate.sh`。
- **F7 状态与 KV 解耦**：归因完成（1.92 GiB/ubatch），`--rs-offload` 未开工。
- **F8 score 归属与分块（B2）**：已落地（`-fac 256`），8G 档 PP 478 -> 680。
- **F9 RAM 有界热专家窗口**：未开工，是"跑更大模型"的唯一路径。
- **F10 分页 KV**：M0/M1/M2b/L1 已落地，M3/M4 未完成；默认 off。
- **F11 分页 prefill gather**：L0 + L1 已落地，目标口径矩阵已补（6144 378.9 对 385.6）。
- **F12 llama-store**：2a/2b/2c 第一刀已落地，M1 未验收，M2（use_step）未开工。
- **F13 子系统边界**：S0 trace / S1 三单元已定，S2 sched-plan 最小版已落地（中性）。

## 5. 唯一执行顺序（每轮 = 一个改动 + 一次矩阵 + 四份文档）

**P0（跑通 + 命中率，按顺序）**

1. ~~按组启用预取 + `:2157` / `:2189` 改退化~~ 已完成（0.17.9）：PP 703.9 / TG 20.9、40/40 层有 bank、输出逐字同。
2. ~~F1 剩余 abort~~ 已完成（0.17.10）：victim / plan / hybrid worker / rolling 全部退化 + 计数。
3. 4096 / 6144 档目标口径矩阵：最大空白，必须 `status = ok`；4096 档 bank 非零。
4. 4096 档 500 MiB 缺口（margins 地板 256 + 未归因 compute 464）在账本里逐项读。

**P1（同配置最优）**

5. F4：补 Vulkan `MUL_MAT_ID_MASK`，按 20-25 GB/s 重算收益，验收 TG > 20.66。
6. q6_K dense 的 mmq 解包（单 ubatch 22.6%，改 q4_K 约 PP +7.5%）。
7. store M2：use_step 预取，TG >= 15% 才算收益。

**P2（更大模型）**：F9 有界 pinned 窗口 -> F5 SSD 冷层 -> F7 状态解耦。

## 6. F12 store 的边界（常开，无总开关）

- 只接管专家（层 + 专家子对象两级）与 KV 页（128 cells = 2.5 MiB）；dense / compute / staging 只申报。
- 三个整机预算 `--store-vram-mib` / `--store-ram-mib` / `--store-ssd-mib`，互不挪用；`store_split` 进启动日志。
- 两个入口：`store_acquire`（同步，只等本批序号）、`store_prefetch`（立刻返回句柄，池搬运，消费点 `store_wait`）。
- 句柄 = 槽号 + 代际号；搬运期间 pin；淘汰过"代际未变 + 无在途 + 无未消费句柄"三道检查（一个原语）。
- 内部三单元 slab / io / policy，不得合回一个文件；policy 可热换而 slab 不动。
- 加载方式不变（lazy_read / mmap / resident），store 只登记专家区间（`store_attach_mapping`）。
- SSD 对单步直接返回不可用并计数 `store_ssd_sync_rejected`；池不可用退化为同步并计数，不 abort。
- 记账列：`store_sync_waits` / `_ms` / `store_prefetch_wasted` / `store_hits_by_tier` / `store_evict_deferred` /
  `store_inflight_pins` / `store_pool_cancel` / `store_split`。
- M1 = ±1% 不回退门（8176 档 PP 506.7 / TG 17.2）；M2 = TG >= 15% 收益门。

## 7. F13 子系统边界

判据四条（生命周期不同 / 失败模式不同 / 变化频率不同 / 有不变量可脱离 GPU 单测），至少三条才拆。
已定：S0 `llama-trace`（先做）、S1 store 三单元、S2 `ggml-sched-plan`（P0，store M2 前置）、
S3 `kv-seq`（M3 后）、S4 `store-file`（M3 后）。
不拆：ggml 算子 / kernel、量化、CLI 解析；**CPU 计算池与 IO 池不得合并**。
