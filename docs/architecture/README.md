# fastLlama 架构图 (交互版)

一句话: 两张可交互视图。技术视图把 qwen35moe 从 HTTP 请求、图构建、MoE bank、KV 串到 VRAM / RAM / SSD; DDD 视图把同一批证据重画成限界上下文、聚合、领域事件与端口。技术视图是当前态, DDD 视图是 proposed 设计。

## 1. 打开方式

直接双击 [fastllama-architecture.html](fastllama-architecture.html) 或 [fastllama-ddd-architecture.html](fastllama-ddd-architecture.html), 无需构建、无需起服务器。建议使用 Chrome / Edge。
两页顶部互相链接。
如果浏览器禁止 `file://` 加载同目录脚本, 在仓库根目录运行:

```
python -m http.server 8000
```

然后打开 `http://localhost:8000/docs/architecture/fastllama-architecture.html` 或 `http://localhost:8000/docs/architecture/fastllama-ddd-architecture.html`。

## 2. 交付内容

| 文件 | 作用 |
|---|---|
| `fastllama-architecture.html` | 技术视图: 当前态组件图 + 运行期数据流 |
| `architecture-model.js` | 技术视图证据模型: 41 节点 / 64 边 / 8 场景 |
| `architecture-app.js` | 技术视图渲染与交互 |
| `fastllama-ddd-architecture.html` | DDD 视图: proposed 上下文地图 + 聚合 / 事件 / 端口 |
| `ddd-model.js` | DDD 视图模型: 69 节点 / 77 边 / 8 场景, 含当前代码映射 |
| `ddd-app.js` | DDD 视图渲染与交互 |
| `ddd-refactor-plan.md` | DDD 改造分析、收益矩阵、三刀切片、验收标准与风险 |
| `ddd-slice0-baseline.md` | Slice 0 基线: 口径、4 处 abort 实际行号、搬运调用点 |
| `validate-model.js` | 校验 ID 重复、悬空端点、场景引用、证据路径是否存在 |

Slice 1 草案不在本目录: 接口在 [`common/residency.h`](../../common/residency.h), 单测在 [`tests/test-residency-plan.cpp`](../../tests/test-residency-plan.cpp)。它没有加入运行时库。
| `README.md` | 本文件: 沟通计划 + 阅读说明 + 交付摘要 |

重建校验 (两个模型分别校验):

```
node docs/architecture/validate-model.js architecture-model.js
node docs/architecture/validate-model.js ddd-model.js
```

当前输出:

```
architecture-model.js: OK: 41 nodes, 64 edges, 8 scenarios, 59 evidence paths
ddd-model.js: OK: 69 nodes, 77 edges, 8 scenarios, 49 evidence paths
```

## 3. 沟通计划 (audience / decision / message)

**受众**: 新加入的工程师、架构评审、需要理解三层协同与 MoE 容量边界的人; 不假设读者熟悉 Vulkan 内核或 MoE 调度细节。

**他们要做的决定**: 先看哪条路径, 哪些机制决定 PP / TG, 哪些是当前代码, 哪些还没落地; 不是逐函数调用清单。

**时间预算**: 总览 3-5 分钟; 分场景 15 分钟左右; 单个节点详情 1-2 分钟。

**消息层次**:

1. 主链是五段: 接入 -> 编排 -> llama core -> ggml -> 硬件三层。每一段只保留决定性能的节点。
2. MoE 的一阶旋钮是命中率与 miss 上传: bank 放设备, 专家整层在 RAM, miss 走 H2D 14.3 GB/s。
3. KV 与专家共享同一套预算账本: KV -> bank -> compute; 256K 的 KV 工作集约 320 MiB, 全分配约 5120 MiB。
4. store 是常开的容量抽象, 不是第四个推理子系统; SSD 只做 prefill / 预取, 绝不进 decode 单步。
5. 实线是当前代码, 虚线是实验 / 部分落地 / 规划; 数字只读 `bench/`, 本页不产生新结论。

**视图层级**: 技术视图是 L2/L3 组件 + 运行期 / 数据拓扑; DDD 视图是上下文地图 + 聚合 / 事件 / 端口。两者都不是 C4 全景, 也不是部署图。

## 4. 阅读顺序

| 场景 | 看什么 |
|---|---|
| 总览 | 层与主链; 粒子只表示当前步骤, 不是完整调用序列 |
| 加载与规划 | GGUF -> mmap / resident -> 统一账本 -> bank / KV / 预取 / 分页 |
| Prefill / PP | prompt -> tokenizer -> ubatch -> 图 -> 调度 -> 专家上传 -> logits |
| Decode / TG | n_tokens = 1; slot map; bank 命中; miss H2D; 单步 58.5 ms 的构成 |
| 长上下文 KV | KV workset、GDN 状态、--kv-page、-nkvo 退化路径 |
| Server 并发 | HTTP -> 队列 -> slots -> 共享 batch -> 流式返回 |
| 三层 store | 三个整机预算 -> adopt / commit -> 预取路由 -> SSD 冷层 (未开工) |
| 验证回路 | run-matrix、quality-gate、op tests、bench/ 四份文档 |

**交互**:

- 点顶部场景按钮或按 `1`-`8` 切换; 空格播放 / 暂停; 左右键切步骤。
- 点节点看职责、关键数字、sourceRefs 和连接; 点连接按钮看关系说明。
- 滚轮缩放, 拖拽平移; 双击节点聚焦; 点"复位"回到全图。
- 搜索支持节点 ID、标签、描述、文件名; 回车跳到第一个命中。
- "跟随步骤"在手动缩放 / 平移后会自动关闭; 勾选后会重新按步骤聚焦。
- "显示实验 / 规划项"关闭后, 虚线节点与边会淡出, 只留当前态。

## 5. 证据与置信度

`architecture-model.js` 与 `ddd-model.js` 分别是两个视图的事实源; 页面只渲染模型。节点 ID 与边 ID 保持稳定, 便于 diff 和引用。DDD 节点保留 `status` 与到当前代码的 sourceRefs, 不把 proposed 概念伪装成已实现 API。

**关键证据**:

| 事实 | 证据 |
|---|---|
| qwen35moe: 40 层 x 256 专家 top-8, 共享专家, 3/4 GDN + 1/4 full attention, 单个 MTP 块 | `src/models/qwen35moe.cpp:4-34`, `src/models/qwen35moe.cpp:54-139`, `AGENTS.md` |
| MoE 图: router -> top-8 -> MUL_MAT_ID -> weighted sum; miss mask / CPU 分支 | `src/llama-graph.cpp:2237-2416` |
| cache 分组、SLRU、预取环、静态整层、hybrid | `src/llama-moe-cache.cpp:641-681`, `:1334-1440` |
| 调度器切分与回调; 异步执行 | `ggml/src/ggml-backend.cpp:2325-2359`, `src/llama-context.cpp:2916` |
| Vulkan 五个预取原语 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:13686-13771`, `:16820-16824` |
| store 的 adopt / commit / bind / pf 入口 | `src/llama-store.h:47-72` |
| store 初始化与专家 adopt 发生在 common 运行时 | `common/common.cpp:1344-1352`, `common/common.cpp:1441-1479` |
| 统一账本与规划器 | `common/fit.h:24-71` |
| KV 工作集 / 全分配、分页、GDN 状态 | `src/llama-kv-cache.cpp:123-166`, `src/llama-kv-cache.cpp:1372-1534`, `bench/RESULTS.md:94` |
| 目标口径数字与机制账本 | `bench/RESULTS.md:8-58` |
| store 的状态与边界 | `bench/THREE-TIER.md:70-81` |

**置信度**: `high` = 有直接代码或文档证据; `medium` = 部分信号一致, 但没有直接来源确认 (例如 speculative 是否启用取决于运行参数); `low` / `unknown` 不用于 current 节点。

## 6. 口径与边界

- 数字口径: 256K ctx + 10K prompt, IQ4_XS, 8176 MiB 档, 2026-10-29。跨文件、跨档位、跨电源态的读数不可比。
- store 当前是 2a-2c: 专家两级 adopt、commit、backend 传输绑定、预取路由已落地; M1 未验收, M2 (use_step) 未开工, SSD 冷层未开工。
- 设计文档里的 `store_acquire` / `store_prefetch` 尚未实现; 现有入口是 `adopt` / `commit` / `bind_backend` / `pf_tensor` / `submit` / `wait`。
- SSD 随机 0.81 GB/s, 206 MiB/token = 258 ms; 图中明确标为 planned, 不进 decode 单步。
- speculative / MTP 与 `llama-moe-exec` 是可选或实验路径, 不计入目标口径主链读数。
- 边是概念级数据 / 控制 / 预算流, 不是每一条函数调用; 精确调用点看节点的 sourceRefs。
- 架构结论不写进本目录, 只读 `bench/` 四份文档; 本页不新增第五份 bench md。

## 7. 维护方式

1. 改技术视图: 编辑 `architecture-model.js` 的 node / edge / scenario; 保持 ID 稳定。
2. 改 DDD 视图: 编辑 `ddd-model.js`; 概念状态用 current / partial / proposed / planned, 并保留当前代码映射。
3. 改渲染: 编辑对应 app; 状态颜色与线型在对应 HTML 的 CSS 里。
4. 新增场景步骤: 确认 `f` 里的 node ID 与 `e` 里的 edge ID 都存在。
5. 跑两个模型的校验; 再跑 `node --check` 检查语法。
6. 如果 bench 数字或状态变化, 只改受影响的节点与 `meta`, 不复制一整份新文档。

## 8. DDD 视图 (proposed)

DDD 页面不是“把技术视图改名”, 而是把同一批证据按边界和不变量重新组织。它回答:
- 哪些规则应该属于聚合 / 策略, 而不是散落在回调里?
- 哪些依赖应该走端口 / ACL, 而不是直接摸 backend 细节?
- 哪些状态变化应该是领域事件, 供 Evidence 观察?

### 8.1 六个限界上下文

| 上下文 | 类型 | 拥有什么 | 当前代码映射 |
|---|---|---|---|
| Model Acquisition | 支撑 / 上游 | ModelArtifact, 量化, 专家几何, 权重来源 | `src/llama-model-loader.cpp`, `src/llama-mmap.cpp`, `src/llama-quant.cpp` |
| Residency & Capacity | 核心域 | ResidencyPlan, ExpertResidency, KvWorkspace; bank / KV / store | `common/fit.*`, `src/llama-moe-cache.*`, `src/llama-store*`, `src/llama-kv-cache.*` |
| Inference Execution | 核心域 | InferenceSession, GraphBuilder, SamplingService | `src/llama-context.*`, `src/llama-graph.cpp`, `src/models/qwen35moe.cpp`, `src/llama-sampler.*` |
| Backend Runtime | 通用子域 | scheduler, Vulkan / CPU / SSD adapter, ACL | `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-vulkan`, `ggml/src/ggml-cpu` |
| Conversation & Delivery | 支撑域 | ChatCompletion, ChatTemplate, HTTP / CLI | `tools/server`, `tools/cli`, `common/chat.cpp` |
| Experiment & Evidence | 支撑域 / 观察者 | Experiment, MatrixRunner, QualityGate, EvidenceLedger | `bench/`, `tools/llama-bench`, `tools/perplexity` |

另有 **Shared Kernel: ggml graph/tensor**, 是 Execution 与 Backend 共享的最小内核。

### 8.2 引入的概念

- **聚合根**: ModelArtifact, ResidencyPlan, ExpertResidency, KvWorkspace, InferenceSession, Experiment。
- **值对象**: Tier, Budget, ExpertKey, TransferTicket, QuantType, ExpertGeometry, Ubatch, Token, Measurement。
- **领域 / 应用服务**: CapacityPlanner, PlacementPolicy, EvictionPolicy, PrefetchScheduler, DegradationPolicy, GraphBuilder, SamplingService, MatrixRunner, QualityGate。
- **端口**: WeightSourcePort, TransferPort, ExecutionPort, ResidencyPort。
- **适配器 / ACL**: Mmap/Resident, GgmlScheduler, Vulkan, CPU, VulkanPrefetchAcl, SsdAdapter。
- **仓储**: ResidencyRepository。
- **领域事件**: PlanSolved, ExpertMiss, PrefetchIssued, PrefetchCompleted, EvictionDeferred, BudgetExhausted, Degraded, KvPageFault, TokenSampled。
- **不变量**: 三预算不可挪用; 容量不足退化不 abort; 淘汰三道 guard; SSD 不进 decode; graph 只引用驻留专家; 口径不可跨比。

### 8.3 状态口径

- `current`: 当前代码已有明确实现。
- `partial`: 已有雏形, 但还没有显式边界 / 契约 (例如 fit 账本 -> ResidencyPlan, moe cache -> ExpertResidency)。
- `proposed`: DDD 提议的新概念 / 事件 / 端口。
- `planned`: 未开工 (例如 SSD 冷层)。

页面用徽标和面板区分状态, 每个节点都有当前代码映射; proposed 不等于已实现。

### 8.4 与技术视图的差别

| 维度 | 技术视图 | DDD 视图 |
|---|---|---|
| 组织 | 技术层: tools / common / core / ggml / hw | 限界上下文 / 子域 |
| 边界 | 头文件、回调、开关 | 端口、ACL、发布语言 |
| 规则 | 红线 + 散落判断 | 聚合不变量 / 策略 / 规约 |
| 变化 | 新 tier / backend 改多处 | 换 adapter / policy |
| 热路径 | 扁平调用 | 仍扁平; 只保留窄端口 |

### 8.5 DDD 场景

| 场景 | 看什么 |
|---|---|
| 上下文地图 | 六个上下文 + 共享内核 + 关系类型 |
| LoadAndPlan | ModelArtifact -> ResidencyPlan -> 专家 / KV 配额 |
| Prefill / DecodeStep | 用例 -> 执行 -> 驻留 -> 后端 |
| Degrade / SwitchTier | 预算、降级、SSD 冷层 |
| Evidence | 观察者回路与口径不变量 |
| 边界原则 | 窄端口、ACL、shared kernel, 不重写热路径 |

### 8.6 不做什么

- 不给 tensor 或每次 GEMM 套 Repository / 领域事件。
- 不拆微服务; 单进程模块化 + 显式契约就够。
- 不把 proposed 概念当成当前 API; 要成为契约还需要接口与 ADR。
- 性能读数仍在技术视图和 `bench/`; DDD 视图不改变口径。

### 8.7 改造计划

收益矩阵、三刀切片、验收标准与风险见 [ddd-refactor-plan.md](ddd-refactor-plan.md)。一句话: 先做 ResidencyPlan + DegradationPolicy, 再做 store 契约, MoE cache 拆分只在 A/B 通过时保留; 全量 DDD 重写不划算。

当前进度: [Slice 0 基线](ddd-slice0-baseline.md) 已落盘; Slice 1 的 [接口](../../common/residency.h) 与 [单测](../../tests/test-residency-plan.cpp) 草案已通过 `ctest -R test-residency-plan`, 但没有接运行时。`src/llama-moe-cache.cpp` 的四处 abort 已在 0.17.10 就地退化并计数, 与本草案无关。ctest 现为 35 项（34/35, `test-model-resolution` 环境性失败）。

## 9. 已知限制

- 全图总览在窄窗口下文字偏小; 用场景聚焦、滚轮缩放或双击节点解决。页面按桌面宽度设计。
- 跨层长边会从节点卡片后面穿过; 为避免挡住阅读, 边标签只在当前步骤 / 悬停 / 选中时显示。
- DDD 上下文地图的主关系线做了绕行; 上下文级关系的标签只在悬停 / 选中时显示, 默认靠颜色 + 图例 + 侧栏关系清单。绕行点是绝对坐标, 移动上下文后要同步改 `edge.route`。
- DDD 视图是 proposed 设计, 不是当前 API; 它的价值是边界与规则, 不是替代技术视图。
- sourceRefs 是仓库相对路径, 依赖 `docs/architecture/` 的位置; 移动目录后需要同步改链接前缀。
- 页面是离线静态页, 不调用后端; 不展示实时遥测, 只展示 2026-10-29 的代码与文档证据。
