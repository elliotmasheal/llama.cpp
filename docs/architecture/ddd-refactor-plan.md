# DDD 改造分析与计划 (Proposal)

状态: Slice 0 基线 + Slice 1 接口/单测草案已落盘; 未接运行时; 四处 abort 已在 0.17.10 就地退化
日期: 2026-10-29
范围: qwen35moe / Vulkan + CPU / VRAM-RAM-SSD 三层
配套视图: [fastllama-ddd-architecture.html](fastllama-ddd-architecture.html), [ddd-model.js](ddd-model.js)
配套基线: [ddd-slice0-baseline.md](ddd-slice0-baseline.md)
Slice 1 草案: [common/residency.h](../../common/residency.h), [tests/test-residency-plan.cpp](../../tests/test-residency-plan.cpp)

> 本文件是架构与改造计划, 不是 bench 结论文档。性能结论仍只写进 `bench/` 四份文档; 本文件不新增第五份 bench md。

## 0. 一句话结论

选择性 DDD 有收益, 但收益不在 PP/TG, 而在“容量与驻留规则集中、可测试、store 演进更稳”。全量 DDD 重写不划算; 重点只放在 Residency & Capacity、store、KV 三个区域, 不碰每 token 热路径。

## 1. 问题陈述

当前项目是单进程、单团队、性能优先的推理引擎。真正需要回答的不是“要不要 DDD”, 而是:

1. 哪些规则已经复杂到需要显式边界?
2. 哪些边界可以在不牺牲 PP/TG 的前提下建立?
3. 哪些 DDD 模式在这里是过度设计?

现状的规则分散在:

- `common/fit.*` 的预算账本与规划器;
- `src/llama-moe-cache.cpp` 的分组、SLRU、预取环、静态整层、hybrid、miss mask;
- `src/llama-store*` 的 slab / io / policy / 预算;
- `src/llama-kv-cache.*` 的 KV 工作集、分页、gather;
- `src/llama-context.*` 的 scheduler 回调;
- `common/common.cpp` 的 store 初始化与 adopt;
- `AGENTS.md` 的工程红线与 4 处未消灭的 abort;
- `bench/` 的口径、判决与缺口。

这些规则目前靠文档、约定和零散断言维持。DDD 的价值是把它们变成可命名的边界、可测试的不变量和可替换的策略; 风险是把每 token 路径也包进抽象, 让 PP/TG 退步。

## 2. 现有代码的 DDD 成熟度

| 区域 | 现有证据 | DDD 成熟度 | 判断 |
|---|---|---|---|
| store 边界 | [`src/llama-store.h`](../../src/llama-store.h) 明确只拥有专家与 KV 页、无总开关、三档预算 | 高 | 已经是聚合/仓储/端口的雏形 |
| slab 句柄 | [`src/llama-store-slab.h`](../../src/llama-store-slab.h) 有 class/slot/generation 与三道 busy 检查 | 高 | 已是不变量边界 |
| io / 传输 | [`src/llama-store-io.h`](../../src/llama-store-io.h) 有 pf API、exec mode、计数 | 高 | 已是端口/适配器 |
| policy | [`src/llama-store-policy.h`](../../src/llama-store-policy.h) 有可热换策略接口 | 中高 | 可继续扩展 |
| 容量规划 | [`common/fit.h`](../../common/fit.h) 的统一账本与 `common_moe_plan` | 中高 | 已是领域模型, 缺不变量测试 |
| 运行时装配 | [`common/common.cpp`](../../common/common.cpp) 初始化 store、adopt、commit | 中 | 可提为应用服务 |
| MoE cache | [`src/llama-moe-cache.cpp`](../../src/llama-moe-cache.cpp) 一个类承担分组/淘汰/预取/静态/hybrid/miss mask | 中低 | 最大的一块混杂职责 |
| KV | [`src/llama-kv-cache.cpp`](../../src/llama-kv-cache.cpp) 分页、grow、gather | 中 | 可提为 KvWorkspace 聚合 |
| 调度耦合 | [`src/llama-context.cpp`](../../src/llama-context.cpp) 的 copy / pre-split / stage 回调 | 低 | 逻辑正确但跨层直接耦合 |
| 证据回路 | [`bench/THREE-TIER.md`](../../bench/THREE-TIER.md), [`bench/RESULTS.md`](../../bench/RESULTS.md) | 中 | 已是 observer/账本, 事件化收益有限 |

结论: 不需要“从零 DDD”。需要的是把现有 seam 提升为显式契约, 并把 MoE cache 的规则从机制里抽出来。

## 3. 收益矩阵

| 区域 | DDD 收益 | 代价 / 风险 | 决策 |
|---|---|---|---|
| 容量 / ResidencyPlan | 高: 预算只有一个所有者; 不变量可单测; 4 处 abort 可归零 | 低: 只在启动/规划期, 不碰热路径 | 做 |
| store / Residency 契约 | 中高: M2/M3 更稳; 避免第三条搬运路径 | 低到中: API 演进与测试成本 | 做 |
| MoE cache 拆分 | 中: 放置/淘汰/预取规则更清楚 | 高: 每层 callback 在热路径 | 条件做, A/B 门禁 |
| KV workspace | 中: 页表/工作集不变量更明确 | 中: 分页路径复杂 | 做小步提取, 不重写 |
| Execution / ggml | 低: 抽象不会让 kernel 更快 | 高: 间接层与性能回溯变难 | 不做 |
| Server / 交付 | 低到中: 主要是命名与边界 | 低 | 暂不做 |
| 实验 / 证据 | 低到中: counters 已够用 | 低 | 只补事件命名, 不建事件总线 |

核心判断: 收益与“规则复杂度”成正比, 与“每 token 调用频率”成反比。容量、store 生命周期、降级在批次/层/搬运粒度, 适合 DDD; 图构建、调度、GEMM 在每 token/每张量粒度, 不适合。

## 4. 目标模型与当前代码映射

| DDD 概念 | 当前代码 | 改造方式 |
|---|---|---|
| ResidencyPlan | `common_moe_plan` + `common_moe_ledger` | 扩展为显式预算聚合, 加不变量检查, 不重写 |
| ExpertResidency | `llama_moe_cache` + `llama_store_slab` 句柄 | 先提取驻留表, 再考虑拆策略 |
| KvWorkspace | `llama_kv_cache` / paged KV | 小步提取页表与工作集不变量 |
| ResidencyRepository | `llama-store` 的 adopt/commit/pf | 先做 façade, 不引入持久化语义 |
| TransferPort | `llama_store_pf_api` + Vulkan 五原语 | 保留现有 API, 外面加窄接口 |
| CapacityPlanner | `common_moe_plan_get` | 保留函数, 外面加策略与测试 |
| PlacementPolicy | moe-cache 的分组/静态/ring 规则 | 条件拆分, 先 A/B |
| EvictionPolicy | slab 三道 busy + SLRU | 已是单一检查点, 只补测试 |
| PrefetchScheduler | moe-cache 预取环 + store-io | 保留机制, 提为策略接口 |
| DegradationPolicy | 4 处 abort + mmap fallback + SSD 拒绝 | 先做, 直接提升鲁棒性 |
| 领域事件 | store-io counters + 现有统计 | 只做命名/结构, 不建事件总线 |

注意: `ResidencyRepository` 在这里是沟通用语义, 不是 JPA/数据库意义上的 Repository。`llama-store` 是内存分配 + 搬运执行器; 真正的仓储抽象如果引入, 应只是对外的窄接口, 不引入持久化生命周期。

## 5. 三刀计划

### Slice 0: 基线与 ADR (1-2 天)

目标: 不改代码, 先固定判断标准。基线与盘点已落盘: [ddd-slice0-baseline.md](ddd-slice0-baseline.md)。

- 基线快照: PP 567.4 / TG 20.66 / hit 68.04% / bank 2376.25 MiB (来自 `bench/RESULTS.md` 第 0 节, 本次未重跑)。
- 盘点 4 处 `GGML_ABORT` 的实际位置: `src/llama-moe-cache.cpp` 的 `:129`、`:1843`、`:2184`、`:2216`。
  - 旧文档里的 `:1841` / `:2157` / `:2189` 是行号漂移; 以基线文档为准。
- 盘点跨层搬运调用点: moe-cache、store-io、Vulkan prefetch、common runtime。
- 写 ADR-001 “选择性 DDD 边界”和 ADR-002 “热路径不引入每 token 抽象”。
- 验收: 基线带口径; abort 清单; 搬运调用清单; ADR 通过。

### Slice 1: ResidencyPlan + DegradationPolicy (3-5 天)

目标: 把预算与降级规则收进一个聚合和一个策略, 不碰热路径。

**草案已落盘 (未接运行时)**:

- `common/residency.h` / `common/residency.cpp`: `common_residency_plan` 不变量 solver + `common_degradation_policy` + counters。solver 只是测试 oracle, 不是第二套生产规划器; `common_moe_plan_get` 仍是生产口径的唯一所有者。
- `tests/test-residency-plan.cpp`: 纯 CPU 单测; 已通过 `ctest -R test-residency-plan`。
- 草案没有加入 `common/CMakeLists.txt`, 不会被 `llama` / `llama-common` 链接; 现有二进制不受影响。
- 已固定: 三预算互不挪用; `sum(alloc) <= budget`; 顺序 KV -> bank -> compute; compute 放不下时先牺牲 bank; 负数/超预算返回 status, 不 abort。
- 四种 abort 映射: `DROP_ROLLING`、`DROP_ROLLING`、`DISABLE_CACHE + DISABLE_HYBRID`、`DISABLE_HYBRID + fail_current_decode`。
- 四处 abort 已在 2026-10-29 就地退化并计数（`bench/RESULTS.md` 0.17.10），不依赖本草案; policy 接入运行时尚无排期。
- 验收: PP/TG 不回退; `GGML_ABORT` 4 -> 0; 不变量有测试; 不新增永久开关。

### Slice 2: Residency 契约 + store (1-2 周)

目标: 让 store 的对外语义稳定, 为 M2/M3 留一条路。

- 在 `llama-store` 外增加窄接口 (名称待定): `acquire` / `prefetch` / `wait` / `evict` / `adopt`; 现有 `pf_*` 保留为 adapter。
- 明确句柄语义: 槽号 + 代际; 三道 busy 检查; 过期句柄不可消费。
- 用该接口完成 M2 (`use_step`) 与 M3 (KV 页 adopt) 的前置; 不写第三套跨层搬运。
- 契约测试: 预算 split、evict 拒绝、SSD 同步拒绝、prefetch 完成、代际失效。
- 验收: 搬运入口只剩一条; store counters 可读; M2 若做, 按 `bench/THREE-TIER.md` 的 TG >= 15% 门禁; 不新增总开关。

### Slice 3: MoE cache 拆分 (条件执行, 2-4 周)

目标: 把规则从机制里抽出来, 但只在 A/B 通过时保留。

- 提取 `placement_policy`、`eviction_policy`、`prefetch_scheduler`; scheduler 回调保留为 adapter。
- 提取 `expert_residency` 驻留表; 不引入每 token 虚调用或额外分配。
- 用 `run-matrix.sh -T 8176,6144,4096 -r 2` 做 A/B; TG 权重 2, PP 权重 1。
- 验收: PP/TG 在 r=2 噪声内; 代码职责减少; 不变量测试; 否则回滚, 不保留开关。

## 6. 不做什么

- 不给 tensor、graph、每次 GEMM 套 Repository 或领域事件。
- 不把 `bank / miss / slot` 改名; 这些是性能语言, 保留。DDD 词汇只用在边界和文档。
- 不拆微服务; 不引入跨进程事件总线。
- 不为 DDD 新增永久开关; 开关只用于 A/B。
- 不把 DDD 页面上的 69 个 proposed 节点当成 69 个待建类; 它是分析视图, 不是改造清单。
- 不把本文件当 bench 结论; 性能读数仍以 `bench/` 为准。

## 7. 验收标准

| 维度 | 标准 |
|---|---|
| PP/TG | 目标口径不回退; r=2 可复现; TG 权重 2, PP 权重 1 |
| 鲁棒性 | `GGML_ABORT` 4 -> 0; 容量条件全部退化并计数 |
| 架构 | 预算只有一个所有者; 搬运入口只剩一条; 不新增永久开关 |
| 测试 | 预算/降级/淘汰/代际不变量有单测; 不依赖 GPU |
| 证据 | 改动只更新受影响节点与 `meta`; DDD 模型状态从 proposed 逐步改为 partial/current |
| 文档 | 本文件与 README、DDD 页面同步; 不新增第五份 bench md |

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 热路径抽象导致 PP/TG 回退 | 中 | Slice 1/2 不碰热路径; Slice 3 必须 A/B, 不回退才保留 |
| 范围蔓延成全量重写 | 中 | 三刀切片; 每刀独立验收; 不做 kernel/Repository |
| 命名 churn 影响沟通 | 中 | 保留 bank/miss/slot; DDD 词只用在边界 |
| store API 演进破坏现有路径 | 低到中 | 先加 façade; `pf_*` 保留为 adapter; 逐步迁移 |
| 单测无法覆盖硬件行为 | 中 | 不变量单测 + quality gate + 目标口径矩阵 |
| 证据与图漂移 | 中 | 改完同步 `ddd-model.js` 状态与 sourceRefs; 跑 validator |

## 9. 预期结果

- PP/TG: 预期中性; 不能指望 DDD 带来 t/s 收益。任何抽象都要过 A/B。
- 维护性: 中幅提升; 预算、降级、淘汰规则更集中, 4 个 abort 归零。
- store 演进: 中高; M2/M3 的风险下降, 搬运路径收敛为一条。
- 沟通: 中; DDD 视图让边界更容易解释, 但它是文档收益, 不是运行时收益。
- 全量 DDD 重写: 负 ROI; 更多文件/接口、更难做性能回溯、命名成本高。

## 10. 待确认问题

1. Slice 0 + Slice 1 草案已落盘; 是否批准进入“逐个 abort 接运行时”?
2. `ResidencyRepository` 是否只作为对外 façade, 不引入持久化语义?
3. `store_acquire` / `store_prefetch` 是最终目标 API, 还是继续以 `pf_*` 为底层、外面加窄接口?
4. 4 处 abort 的修复顺序: 先 `:2184` / `:2216`, 还是先 `:129`, 或先做统一的 `DegradationPolicy`?
5. Slice 3 是否值得做, 取决于 Slice 1/2 后 moe-cache 的复杂度是否仍是主要痛点。

## 11. 参考

- [DDD 视图](fastllama-ddd-architecture.html)
- [DDD 模型](ddd-model.js)
- [Slice 0 基线](ddd-slice0-baseline.md)
- [`common/residency.h`](../../common/residency.h)
- [`tests/test-residency-plan.cpp`](../../tests/test-residency-plan.cpp)
- [README](README.md)
- [`src/llama-store.h`](../../src/llama-store.h)
- [`src/llama-store-slab.h`](../../src/llama-store-slab.h)
- [`src/llama-store-io.h`](../../src/llama-store-io.h)
- [`src/llama-store-policy.h`](../../src/llama-store-policy.h)
- [`common/fit.h`](../../common/fit.h)
- [`common/common.cpp`](../../common/common.cpp)
- [`src/llama-moe-cache.cpp`](../../src/llama-moe-cache.cpp)
- [`src/llama-context.cpp`](../../src/llama-context.cpp)
- [`src/llama-kv-cache.cpp`](../../src/llama-kv-cache.cpp)
- [`bench/THREE-TIER.md`](../../bench/THREE-TIER.md)
- [`bench/RESULTS.md`](../../bench/RESULTS.md)
- [`AGENTS.md`](../../AGENTS.md)
