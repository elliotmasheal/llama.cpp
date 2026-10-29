/* fastLlama DDD architecture model (proposed).
 * This is a target-state / communication model, not the current API contract.
 * status: proposed = new DDD concept, partial = current code has a precursor,
 * current = explicit current implementation, planned = not started.
 * Every node keeps sourceRefs to the current code or bench docs that inform it.
 */
window.DDD_MODEL = {
  meta: {
    title: "fastLlama DDD 架构 (proposed)",
    branch: "trim-vulkan-cpu-qwen35moe",
    date: "2026-10-29",
    target: "6 个限界上下文 / 6 个聚合 / 9 个领域事件",
    note: "这是 DDD 重画, 不是当前 API 契约; proposed 概念由 current code 映射支撑; 热路径不引入每 token 抽象."
  },

  metrics: [
    { k: "限界上下文", v: "6", u: "", cls: "hot" },
    { k: "聚合根", v: "6", u: "", cls: "" },
    { k: "领域事件", v: "9", u: "", cls: "" },
    { k: "端口", v: "4", u: "", cls: "" },
    { k: "不变量", v: "6", u: "", cls: "warn" }
  ],

  bands: [
    { id: "band.support", x: 20, y: 20, w: 440, h: 680, label: "Supporting Domains", sub: "Model Acquisition / Conversation & Delivery" },
    { id: "band.core", x: 480, y: 20, w: 960, h: 680, label: "Core Domain", sub: "Residency & Capacity / Inference Execution" },
    { id: "band.generic", x: 1460, y: 20, w: 320, h: 680, label: "Generic / Infrastructure", sub: "Backend Runtime" },
    { id: "band.events", x: 20, y: 720, w: 1760, h: 180, label: "Domain Events / Published Language", sub: "proposed vocabulary; mapped to current counters and functions" },
    { id: "band.evidence", x: 20, y: 920, w: 1760, h: 300, label: "Supporting Domain: Experiment & Evidence", sub: "observer / quality gate / ledger" }
  ],

  nodes: [
    {
      id: "ctx.model", label: "Model Acquisition", sub: "支撑域 / 上游", type: "supporting", kind: "context", status: "proposed", confidence: "high",
      x: 40, y: 70, w: 400, h: 280,
      desc: "把 GGUF 变成一个不可变的 ModelArtifact, 提供量化类型、专家几何和权重来源。它只提供事实, 不决定专家放哪。",
      details: ["下游: Inference Execution 与 Residency & Capacity 都消费 ModelArtifact。", "GGUF 是 published language; 下游不反向依赖加载细节。", "当前代码: llama-model-loader / llama-mmap / llama-quant。"],
      refs: ["src/llama-model-loader.cpp", "src/llama-mmap.cpp", "src/llama-quant.cpp", "src/llama-model.cpp"]
    },
    {
      id: "ctx.delivery", label: "Conversation & Delivery", sub: "支撑域 / 交付", type: "supporting", kind: "context", status: "proposed", confidence: "high",
      x: 40, y: 390, w: 400, h: 290,
      desc: "HTTP / CLI / WebUI 的用例入口。它把 messages 变成 ChatCompletion 用例, 再调用 Inference Execution; 不拥有 KV 或专家驻留。",
      details: ["server slots / queue / SSE 都属于交付适配器, 不属于推理领域。", "ChatTemplate 只做文本到 prompt; 多模态 part 在这里汇入。", "当前代码: tools/server, tools/cli, common/chat。"],
      refs: ["tools/server/server-context.cpp", "tools/cli/cli-context.h:21-46", "common/chat.cpp", "tools/ui/README.md"]
    },
    {
      id: "ctx.residency", label: "Residency & Capacity", sub: "核心域 / 预算与驻留", type: "core", kind: "context", status: "proposed", confidence: "high",
      x: 500, y: 70, w: 500, h: 600,
      desc: "决定专家和 KV 放在 VRAM / RAM / SSD, 并保证预算与降级规则。这是 DDD 重画的核心域; 当前由 fit / moe cache / store / KV cache 共同承担。",
      details: ["拥有 ResidencyPlan / ExpertResidency / KvWorkspace 三个聚合。", "预算、放置、淘汰、预取、降级都是策略, 不散落在调度回调里。", "SSD 只做容量层, 不进 decode 单步。"],
      refs: ["common/fit.h:24-71", "src/llama-moe-cache.h:11-88", "src/llama-store.h:7-72", "src/llama-kv-cache.h:99-123"]
    },
    {
      id: "ctx.execution", label: "Inference Execution", sub: "核心域 / 会话与执行", type: "core", kind: "context", status: "proposed", confidence: "high",
      x: 1020, y: 70, w: 400, h: 440,
      desc: "把 ubatch 变成图, 调 ExecutionPort, 采样 token。它依赖 Residency 提供已驻留专家和 KV, 但不拥有放置策略。",
      details: ["InferenceSession 是聚合根; GraphBuilder / SamplingService 是领域服务。", "对 Residency 只通过 ResidencyPort; 对 Backend 只通过 ExecutionPort。", "当前代码: llama-context / llama-graph / models/qwen35moe / sampler。"],
      refs: ["src/llama-context.h:45-69", "src/llama-graph.cpp:2237-2416", "src/models/qwen35moe.cpp:149-542", "src/llama-sampler.cpp"]
    },
    {
      id: "ctx.backend", label: "Backend Runtime", sub: "通用子域 / 基础设施", type: "generic", kind: "context", status: "proposed", confidence: "high",
      x: 1480, y: 70, w: 280, h: 600,
      desc: "后端运行时: 通用/基础设施子域。实现 ExecutionPort 与 TransferPort, 不拥有领域规则。",
      details: ["Vulkan / CPU / SSD 都是 adapter; 设备细节不进入 Residency 或 Execution。", "Vulkan prefetch 五原语通过 ACL 暴露给 TransferPort。", "当前代码: ggml scheduler / ggml-vulkan / ggml-cpu。"],
      refs: ["ggml/src/ggml-backend.cpp:2325-2359", "ggml/src/ggml-vulkan/ggml-vulkan.cpp:13686-13771", "ggml/src/ggml-cpu"]
    },
    {
      id: "ctx.evidence", label: "Experiment & Evidence", sub: "支撑域 / 观察者", type: "supporting", kind: "context", status: "proposed", confidence: "high",
      x: 40, y: 960, w: 1720, h: 250,
      desc: "观察运行时事件与指标, 跑矩阵和质量门, 维护口径与账本。它是 observer, 不参与推理数据流。",
      details: ["Experiment 是聚合根; Measurement 是带口径的值对象。", "EvidenceLedger 是 RESULTS / DECISIONS / THREE-TIER 的单一落点。", "判决和缺口可以回灌 Residency 策略, 但不直接改运行时状态。"],
      refs: ["bench/tools/run-matrix.sh", "bench/tools/quality-gate.sh", "bench/RESULTS.md", "bench/DECISIONS.md", "bench/THREE-TIER.md"]
    },
    {
      id: "shared.ggml", label: "Shared Kernel: ggml graph/tensor", sub: "C API / graph IR / tensor metadata", type: "shared", kind: "shared", status: "current", confidence: "high",
      x: 1020, y: 530, w: 400, h: 100,
      desc: "Execution 与 Backend 共享的最小内核: ggml 张量、图 IR、后端接口。DDD 建议把它保持薄、稳定、少变。",
      details: ["共享内核不承载领域规则; 它只是执行层的公共语言。", "对 Vulkan/CPU 的差异由 adapter 吸收。", "当前代码: ggml 张量库与 backend C API。"],
      refs: ["ggml/include/ggml.h", "ggml/src/ggml.c", "ggml/src/ggml-backend.cpp"]
    },

    {
      id: "agg.model", label: "ModelArtifact", sub: "元数据 / 张量 / 量化", type: "aggregate", kind: "rect", status: "partial", confidence: "high", context: "ctx.model",
      x: 60, y: 150, w: 230, h: 80,
      desc: "不可变模型工件: 元数据、张量布局、量化类型、专家几何。聚合根保证一次加载后不会被重新量化或改布局。",
      details: ["外部只能通过它读取 geometry / quant / tensor refs。", "变更必须产生新的 ModelArtifact。"],
      invariants: ["加载后 quant / shape / layout 不可变", "geometry 是 ResidencyPlan 的唯一模型输入"],
      refs: ["src/llama-model.h", "src/llama-model.cpp", "src/models/qwen35moe.cpp"]
    },
    {
      id: "vo.quant", label: "QuantType", sub: "IQ4_XS / Q4_K", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.model",
      x: 310, y: 150, w: 110, h: 40,
      desc: "量化类型与档位。档位是文件选择, 不是引擎行为。",
      details: ["同一二进制换文件即可切 8176 / 6144 / 4096 档。"],
      refs: ["src/llama-quant.cpp", "bench/THREE-TIER.md:15-23"]
    },
    {
      id: "vo.geom", label: "ExpertGeometry", sub: "layer / bytes / layout", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.model",
      x: 310, y: 200, w: 110, h: 40,
      desc: "专家几何: 每层专家字节、布局 hash、n_expert_used。供 CapacityPlanner 与 cache 分组。",
      details: ["layout hash 决定哪些层共享一个 bank 分组。"],
      refs: ["src/llama-ext.h:92-103", "src/llama-moe-cache.cpp:641-681"]
    },
    {
      id: "svc.load", label: "ModelLoader", sub: "read / place / validate", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.model",
      x: 60, y: 240, w: 160, h: 50,
      desc: "读 GGUF, 决定 mmap / lazy / resident, 创建 ModelArtifact。当前是过程式加载器, DDD 把它作为领域服务。",
      details: ["-lm none 请求 resident, 容量不够退回 mmap。", "mmap 下 bank 与预取强制 0。"],
      refs: ["src/llama-model-loader.cpp:542-567", "src/llama-model-loader.cpp:1438-1465"]
    },
    {
      id: "port.weights", label: "WeightSourcePort", sub: "mmap / resident / lazy", type: "port", kind: "rect", status: "proposed", confidence: "high", context: "ctx.model",
      x: 240, y: 240, w: 180, h: 50,
      desc: "权重来源抽象。适配 mmap、lazy read、resident; 让 Residency 不关心文件格式。",
      details: ["实现: MmapWeightSource / ResidentWeightSource。"],
      refs: ["src/llama-model-loader.cpp:1438-1465", "src/llama-mmap.h"]
    },
    {
      id: "adapter.mmap", label: "Mmap / Resident", sub: "WeightSource adapter", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.model",
      x: 60, y: 300, w: 170, h: 40,
      desc: "实现 WeightSourcePort 的两种加载方式。",
      details: ["mmap 省 RAM 但失去 bank 与预取; resident 是加速前提。"],
      refs: ["src/llama-mmap.cpp", "src/llama-model-loader.cpp:1580-1698"]
    },
    {
      id: "inv.model", label: "Immutable artifact", sub: "quant / shape fixed", type: "invariant", kind: "invariant", status: "current", confidence: "high", context: "ctx.model",
      x: 240, y: 300, w: 180, h: 40,
      desc: "ModelArtifact 在加载后不可变; 量化、形状、布局变化必须产生新 artifact。",
      details: ["这条不变量让 ResidencyPlan 可以安全地缓存 geometry。"],
      refs: ["src/llama-model.cpp", "bench/THREE-TIER.md:15-23"]
    },

    {
      id: "app.chat", label: "ChatCompletion", sub: "RunPrefill / DecodeStep", type: "usecase", kind: "rect", status: "partial", confidence: "high", context: "ctx.delivery",
      x: 60, y: 470, w: 250, h: 56,
      desc: "应用服务: 接收 messages, 渲染模板, 调 RunPrefill / RunDecodeStep, 流式返回。",
      details: ["它是交付层与推理层之间的用例边界。", "不拥有 KV、专家或采样策略。"],
      refs: ["tools/server/server-context.cpp:3431", "tools/server/README-dev.md:78-84"]
    },
    {
      id: "svc.chat", label: "ChatTemplate", sub: "Jinja / PEG / tools", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.delivery",
      x: 60, y: 545, w: 160, h: 54,
      desc: "把消息历史渲染成 prompt; 支持 Jinja、PEG 与工具调用格式。",
      details: ["只做文本到 prompt, 不碰 GPU。"],
      refs: ["common/chat.cpp", "common/chat-peg-parser.cpp"]
    },
    {
      id: "adapter.server", label: "HTTP / CLI", sub: "queue / slots / SSE", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.delivery",
      x: 240, y: 545, w: 180, h: 54,
      desc: "交付适配器: HTTP / CLI / WebUI 把外部请求翻译成 ChatCompletion 用例。",
      details: ["server slots / queue / 流式返回都在这一层。"],
      refs: ["tools/server/server-context.cpp", "tools/cli/cli-context.h:21-46"]
    },
    {
      id: "vo.messages", label: "Messages", sub: "roles / parts", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.delivery",
      x: 60, y: 620, w: 150, h: 40,
      desc: "用例输入: 角色、内容、多模态 parts。",
      details: ["多模态 part 由 mtmd 适配器编码后注入 batch。"],
      refs: ["common/chat.h", "tools/mtmd"]
    },
    {
      id: "vo.stream", label: "TokenStream", sub: "incremental", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.delivery",
      x: 230, y: 620, w: 150, h: 40,
      desc: "用例输出: 增量 token 流。",
      details: ["通过 SSE / CLI 输出。"],
      refs: ["tools/server/README-dev.md", "tools/cli/cli-context.h"]
    },

    {
      id: "agg.plan", label: "ResidencyPlan", sub: "budget / quota / tier split", type: "aggregate", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 520, y: 150, w: 220, h: 90,
      desc: "一次运行的预算与配额: 三个 tier 预算、bank / KV / compute 分配、margin 与降级规则。",
      details: ["聚合根: 所有放置决策先经过它; 模块不能私自分预算。", "当前是 fit 账本 + params; DDD 把它提升为显式聚合。"],
      invariants: ["VRAM / RAM / SSD 三个预算互不挪用", "sum(alloc) <= budget; 顺序 KV -> bank -> compute", "容量不足必须退化, 不得 abort"],
      refs: ["common/fit.h:24-71", "bench/THREE-TIER.md:24-50", "bench/THREE-TIER.md:70-81"]
    },
    {
      id: "agg.expert", label: "ExpertResidency", sub: "expert -> slot / generation", type: "aggregate", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 760, y: 150, w: 220, h: 90,
      desc: "层 + 专家到 slot 的映射, 带 generation / pin / inflight / unconsumed 状态。当前由 moe cache + store slab 承担。",
      details: ["聚合根: 一次专家占用从 acquire 到 consume 是一个一致性边界。", "淘汰必须通过 busy 检查, 调用方不能绕过。"],
      invariants: ["代际未变 + 无在途 + 无未消费句柄才可淘汰", "bank 命中 / 静态整层 / 预取环统一记账"],
      refs: ["src/llama-moe-cache.h:11-88", "src/llama-store-slab.h:20-38"]
    },
    {
      id: "agg.kv", label: "KvWorkspace", sub: "cells / pages / working set", type: "aggregate", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 520, y: 260, w: 220, h: 80,
      desc: "序列的 KV cells / pages 与工作集。当前是 llama_kv_cache + paged KV; DDD 把它作为独立聚合。",
      details: ["聚合根: 页表、cell 分配、gather 工作集。", "分页是内部策略; 对 Execution 只暴露 sequence -> cells。"],
      invariants: ["页表映射有效; 工作集不超过 KV 预算", "SSD 不进 decode 单步"],
      refs: ["src/llama-kv-cache.h:99-123", "src/llama-kv-cache.cpp:123-166", "src/llama-kv-cache.cpp:1372-1534"]
    },
    {
      id: "vo.tier", label: "Tier", sub: "VRAM / RAM / SSD", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 760, y: 260, w: 100, h: 40,
      desc: "放置层: VRAM / RAM / SSD, 互不挪用。",
      details: ["SSD 只做容量层。"],
      refs: ["src/llama-store.h:15-19", "bench/THREE-TIER.md:7-13"]
    },
    {
      id: "vo.budget", label: "Budget", sub: "bytes / auto", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 870, y: 260, w: 110, h: 40,
      desc: "每个 tier 的字节预算; 0 = auto。",
      details: ["来源: --store-*-mib 或运行时 free memory。"],
      refs: ["src/llama-store.h:27-33", "common/common.cpp:1344-1352"]
    },
    {
      id: "vo.expertkey", label: "ExpertKey", sub: "layer + expert", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 760, y: 310, w: 100, h: 40,
      desc: "专家两级身份: (layer, expert), adopt 后 O(1) 索引。",
      details: ["取代按张量名字符串查找。"],
      refs: ["src/llama-store.h:52-53"]
    },
    {
      id: "vo.ticket", label: "TransferTicket", sub: "slot + generation", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 870, y: 310, w: 110, h: 40,
      desc: "一次异步搬运的句柄: 槽号 + 代际; 消费点 wait。",
      details: ["防止过期句柄在淘汰后仍被消费。"],
      refs: ["src/llama-store-slab.h:14-18", "src/llama-store-io.h:36-42"]
    },
    {
      id: "svc.capacity", label: "CapacityPlanner", sub: "ledger solve", type: "service", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 520, y: 360, w: 140, h: 54,
      desc: "量模型与空闲显存, 用统一账本求解 ResidencyPlan。当前是 common/fit。",
      details: ["账本条目: dense / kv_context / compute / fa_score / kv_gather / margin / bank。"],
      refs: ["common/fit.h:37-71", "common/fit.cpp"]
    },
    {
      id: "svc.placement", label: "PlacementPolicy", sub: "bank / static / rings", type: "service", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 680, y: 360, w: 140, h: 54,
      desc: "放置规则: bank / 静态整层 / 预取环 / hybrid。当前在 moe-cache。",
      details: ["按专家布局 hash 分组; 每组独立预算。"],
      refs: ["src/llama-moe-cache.cpp:641-681", "src/llama-moe-cache.cpp:1334-1440"]
    },
    {
      id: "svc.eviction", label: "EvictionPolicy", sub: "SLRU + guards", type: "service", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 840, y: 360, w: 140, h: 54,
      desc: "淘汰策略: SLRU + 三道 busy 检查; 淘汰的唯一入口。",
      details: ["busy() 是唯一的检查点, 不允许第二套 checkpoint。"],
      refs: ["src/llama-store-slab.h:20-38", "src/llama-moe-cache.cpp:241"]
    },
    {
      id: "svc.prefetch", label: "PrefetchScheduler", sub: "rings / lead / slots", type: "service", kind: "rect", status: "partial", confidence: "high", context: "ctx.residency",
      x: 520, y: 430, w: 140, h: 54,
      desc: "预取环、lead、按组门槛; 生成 TransferTicket。",
      details: ["每组至少 512 slots; 大 ubatch 走两层 ring; 小 batch 走 LRU。"],
      refs: ["src/llama-moe-cache.cpp:1334-1440", "src/llama-moe-cache.cpp:1464-1525"]
    },
    {
      id: "svc.degrade", label: "DegradationPolicy", sub: "degrade, never abort", type: "service", kind: "rect", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 680, y: 430, w: 140, h: 54,
      desc: "容量不足时退 mmap / 关预取 / 拒绝 SSD 同步, 不 abort。",
      details: ["剩余 4 处 abort 是待消灭的缺陷; 目标是所有容量条件都退化并计数。"],
      refs: ["bench/THREE-TIER.md:35-50", "bench/THREE-TIER.md:70-81", "src/llama-store-io.h:57-64"]
    },
    {
      id: "port.transfer", label: "TransferPort", sub: "tensor / submit / wait", type: "port", kind: "rect", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 840, y: 430, w: 140, h: 54,
      desc: "跨层搬运端口: tensor / submit / wait / submit_seq / wait_seq。当前是 store-io + Vulkan 五原语。",
      details: ["同步 acquire 与异步 prefetch 都走这一组原语。"],
      refs: ["src/llama-store.h:66-71", "src/llama-store-io.h:13-42"]
    },
    {
      id: "repo.residency", label: "ResidencyRepository", sub: "adopt / acquire / evict", type: "repository", kind: "rect", status: "proposed", confidence: "high", context: "ctx.residency",
      x: 520, y: 500, w: 220, h: 54,
      desc: "聚合的持久/检索入口: adopt、acquire、prefetch、evict。当前是 llama-store。",
      details: ["store 的 slab / io / policy 对应 repository 的内部实现。", "对 Execution 暴露 ResidencyPort, 不暴露 slab 细节。"],
      refs: ["src/llama-store.h:47-72", "src/llama-store-slab.h", "src/llama-store-io.h", "src/llama-store-policy.h"]
    },
    {
      id: "inv.budget", label: "Budget non-fungible", sub: "VRAM / RAM / SSD", type: "invariant", kind: "invariant", status: "partial", confidence: "high", context: "ctx.residency",
      x: 760, y: 500, w: 220, h: 54,
      desc: "三个整机预算互不挪用; 分配顺序 KV -> bank -> compute。",
      details: ["违反时不是调参, 而是聚合不变量被破坏。"],
      refs: ["bench/THREE-TIER.md:70-81", "bench/THREE-TIER.md:24-50"]
    },
    {
      id: "inv.evict", label: "Eviction guards", sub: "gen + inflight + unconsumed", type: "invariant", kind: "invariant", status: "current", confidence: "high", context: "ctx.residency",
      x: 520, y: 570, w: 220, h: 54,
      desc: "只有代际未变、无在途、无未消费句柄时才允许淘汰。",
      details: ["slab.busy() 是唯一实现; 其他 checkpoint 视为缺陷。"],
      refs: ["src/llama-store-slab.h:20-38", "bench/THREE-TIER.md:70-81"]
    },
    {
      id: "inv.ssd", label: "SSD not in decode", sub: "prefill / prefetch only", type: "invariant", kind: "invariant", status: "current", confidence: "high", context: "ctx.residency",
      x: 760, y: 570, w: 220, h: 54,
      desc: "SSD 只允许出现在 prefill 与预取; decode 单步的同步请求直接拒绝并计数。",
      details: ["随机 0.81 GB/s, 206 MiB/token = 258 ms。"],
      refs: ["bench/THREE-TIER.md:7-13", "bench/THREE-TIER.md:70-81", "bench/RESULTS.md:28"]
    },

    {
      id: "agg.session", label: "InferenceSession", sub: "context / batch / graph", type: "aggregate", kind: "rect", status: "partial", confidence: "high", context: "ctx.execution",
      x: 1040, y: 150, w: 240, h: 90,
      desc: "一次会话的 context / batch / graph / KV handle; 聚合根。",
      details: ["对 Residency 只通过 ResidencyPort; 对 Backend 只通过 ExecutionPort。", "当前是 llama_context + sched reserve。"],
      invariants: ["graph 只引用已驻留专家; miss mask 是显式例外", "batch 前分配 KV cells"],
      refs: ["src/llama-context.h:45-69", "src/llama-context.cpp:2916"]
    },
    {
      id: "svc.graph", label: "GraphBuilder", sub: "qwen35moe layer graph", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.execution",
      x: 1040, y: 260, w: 170, h: 54,
      desc: "构建 qwen35moe 层图: GDN / full attention + MoE + shared expert。",
      details: ["MoE 子图: router -> top-8 -> MUL_MAT_ID -> sum。"],
      refs: ["src/models/qwen35moe.cpp:276-542", "src/llama-graph.cpp:2237-2416"]
    },
    {
      id: "svc.sampling", label: "SamplingService", sub: "logits -> token", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.execution",
      x: 1230, y: 260, w: 170, h: 54,
      desc: "采样服务: logits -> token; 采样链与 grammar。",
      details: ["server 每个 slot 独立采样。"],
      refs: ["src/llama-sampler.cpp", "common/sampling.cpp"]
    },
    {
      id: "vo.ubatch", label: "Ubatch", sub: "physical batch", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.execution",
      x: 1040, y: 335, w: 140, h: 40,
      desc: "物理批次; 图与 KV 分配的单位。",
      details: ["逻辑 batch 拆成等长 ubatch。"],
      refs: ["src/llama-batch.cpp"]
    },
    {
      id: "vo.token", label: "Token", sub: "sampled", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.execution",
      x: 1200, y: 335, w: 120, h: 40,
      desc: "采样结果; 回到下一次 batch。",
      details: ["decode 循环的单位。"],
      refs: ["src/llama-sampler.h"]
    },
    {
      id: "port.exec", label: "ExecutionPort", sub: "compute graph", type: "port", kind: "rect", status: "proposed", confidence: "high", context: "ctx.execution",
      x: 1040, y: 395, w: 170, h: 54,
      desc: "提交 ggml_cgraph 并等待结果; 当前是 sched_graph_compute_async。",
      details: ["对 Backend 的唯一执行入口。"],
      refs: ["ggml/src/ggml-backend.cpp:2325-2359", "src/llama-context.cpp:2916"]
    },
    {
      id: "port.residency", label: "ResidencyPort", sub: "acquire / prefetch / wait", type: "port", kind: "rect", status: "proposed", confidence: "high", context: "ctx.execution",
      x: 1230, y: 395, w: 170, h: 54,
      desc: "向 Residency 请求专家与 KV 页; 当前是 store / moe cache 的窄接口。",
      details: ["同步 acquire 与异步 prefetch 都返回可等待的句柄。"],
      refs: ["src/llama-store.h:66-71", "src/llama-moe-cache.h:23-36"]
    },
    {
      id: "inv.graph", label: "Graph only resident", sub: "hybrid miss explicit", type: "invariant", kind: "invariant", status: "partial", confidence: "high", context: "ctx.execution",
      x: 1040, y: 460, w: 360, h: 40,
      desc: "图只引用已驻留专家; hybrid miss 分支必须显式标记, 不能偷偷回退。",
      details: ["graph 构建时通过 slot map / miss mask 表达驻留状态。"],
      refs: ["src/llama-graph.cpp:2400-2416", "src/llama-moe-cache.h:32-37"]
    },

    {
      id: "adapter.sched", label: "GgmlSchedulerAdapter", sub: "split / callbacks", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.backend",
      x: 1500, y: 150, w: 240, h: 60,
      desc: "实现 ExecutionPort: 图切分、回调、异步执行。",
      details: ["copy / pre-split / stage 回调由 adapter 吸收。"],
      refs: ["ggml/src/ggml-backend.cpp:2325-2359"]
    },
    {
      id: "adapter.vulkan", label: "VulkanAdapter", sub: "shaders / queues", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.backend",
      x: 1500, y: 230, w: 240, h: 60,
      desc: "Vulkan 设备适配器: shader、队列、显存。",
      details: ["单卡 dense/id GEMM 6.3-6.6 TF; H2D 14.3 GB/s。"],
      refs: ["ggml/src/ggml-vulkan/ggml-vulkan.cpp", "bench/RESULTS.md:89-93"]
    },
    {
      id: "adapter.cpu", label: "CpuAdapter", sub: "GEMM / repack / MASK", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.backend",
      x: 1500, y: 310, w: 240, h: 60,
      desc: "CPU 设备适配器: GEMM、repack、MUL_MAT_ID_MASK。",
      details: ["CPU MoE 就地 20-25 GB/s。"],
      refs: ["ggml/src/ggml-cpu", "bench/RESULTS.md:36-37"]
    },
    {
      id: "acl.vk", label: "VulkanPrefetchAcl", sub: "ACL: 5 primitives", type: "adapter", kind: "rect", status: "partial", confidence: "high", context: "ctx.backend",
      x: 1500, y: 390, w: 240, h: 60,
      desc: "防腐层: 把 TransferPort 翻译成 5 个 Vulkan prefetch 原语, 隔离设备细节。",
      details: ["proc address 解析在 store-io, 调用方不接触 Vulkan 名字。"],
      refs: ["src/llama-store-io.cpp:6-22", "ggml/src/ggml-vulkan/ggml-vulkan.cpp:13686-13771"]
    },
    {
      id: "adapter.ssd", label: "SsdAdapter", sub: "cold tier / planned", type: "adapter", kind: "rect", status: "planned", confidence: "high", context: "ctx.backend",
      x: 1500, y: 470, w: 240, h: 60,
      desc: "SSD 冷专家容量层, 只做 prefill / 预取; F5 未开工。",
      details: ["随机 0.81 GB/s; 单步同步请求拒绝。"],
      refs: ["bench/THREE-TIER.md:44-45", "bench/THREE-TIER.md:70-81"]
    },

    {
      id: "agg.experiment", label: "Experiment", sub: "matrix / quality / ledger", type: "aggregate", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 70, y: 1040, w: 220, h: 80,
      desc: "一次矩阵或质量门运行; 聚合根。",
      details: ["记录读数、口径与门禁结论。", "实验结论只进 bench/ 四份文档。"],
      invariants: ["每个读数带日期 / 文件 / 档位 / bank / hit / status", "跨口径比较视为错误"],
      refs: ["bench/RESULTS.md", "bench/DECISIONS.md", "bench/THREE-TIER.md"]
    },
    {
      id: "vo.measure", label: "Measurement", sub: "caliber-bound number", type: "value", kind: "value", status: "proposed", confidence: "high", context: "ctx.evidence",
      x: 310, y: 1040, w: 150, h: 40,
      desc: "带口径的读数: 日期、文件、ftype、档位、bank、hit、status。",
      details: ["没有口径的数字不是测量。"],
      refs: ["bench/README.md", "bench/RESULTS.md:1-5"]
    },
    {
      id: "inv.caliber", label: "Comparable only with caliber", sub: "no cross-caliber diff", type: "invariant", kind: "invariant", status: "current", confidence: "high", context: "ctx.evidence",
      x: 480, y: 1040, w: 280, h: 80,
      desc: "没有相同口径的读数不可直接比较; 跨口径差异视为错误。",
      details: ["这条不变量保护所有性能结论。"],
      refs: ["bench/README.md", "AGENTS.md"]
    },
    {
      id: "svc.matrix", label: "MatrixRunner", sub: "3 tiers x r=2", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 70, y: 1140, w: 170, h: 50,
      desc: "驱动 A/B 矩阵: -T 8176,6144,4096 -r 2。",
      details: ["每轮只改一个机制; 记录 ΔPP / ΔTG / 代价条件。"],
      refs: ["bench/tools/run-matrix.sh", "AGENTS.md"]
    },
    {
      id: "svc.quality", label: "QualityGate", sub: "PPL / KLD", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 260, y: 1140, w: 170, h: 50,
      desc: "质量门: PPL / KLD / same top p。",
      details: ["动数值路径必跑; 当前 GATE PASS。"],
      refs: ["bench/tools/quality-gate.sh", "tools/perplexity", "bench/RESULTS.md:83"]
    },
    {
      id: "svc.ledger", label: "EvidenceLedger", sub: "RESULTS / DECISIONS", type: "service", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 450, y: 1140, w: 180, h: 50,
      desc: "证据账本: 数字、判决、缺口与口径的唯一落点。",
      details: ["结论不新增第五份 bench md。"],
      refs: ["bench/RESULTS.md", "bench/DECISIONS.md", "bench/THREE-TIER.md", "bench/README.md"]
    },
    {
      id: "adapter.bench", label: "llama-bench", sub: "PP / TG adapter", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 790, y: 1040, w: 160, h: 44,
      desc: "测量适配器: 直接调 llama_decode 跑 PP / TG。",
      details: ["用固定 prompt 与参数。"],
      refs: ["tools/llama-bench"]
    },
    {
      id: "adapter.ppl", label: "perplexity", sub: "PPL adapter", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 790, y: 1100, w: 160, h: 44,
      desc: "质量适配器: 计算 PPL / KLD。",
      details: ["质量门的一部分。"],
      refs: ["tools/perplexity"]
    },
    {
      id: "adapter.docs", label: "bench/ docs", sub: "four documents", type: "adapter", kind: "rect", status: "current", confidence: "high", context: "ctx.evidence",
      x: 790, y: 1154, w: 160, h: 40,
      desc: "文档适配器: 四份 bench 文档。",
      details: ["只读/追加到既定文件, 不新增第五份。"],
      refs: ["bench/RESULTS.md", "bench/DECISIONS.md", "bench/THREE-TIER.md", "bench/README.md"]
    },
    {
      id: "svc.observe", label: "EventObserver", sub: "events -> metrics", type: "service", kind: "rect", status: "proposed", confidence: "medium", context: "ctx.evidence",
      x: 990, y: 1040, w: 200, h: 60,
      desc: "订阅领域事件并转成 metrics / 账本; 观察者不反向驱动运行时。",
      details: ["当前对应 store-io counters 与 llama perf。"],
      refs: ["src/llama-store-io.h:44-74", "src/llama-context.h:195-202"]
    },
    {
      id: "adapter.telem", label: "TelemetryAdapter", sub: "counters / perf", type: "adapter", kind: "rect", status: "proposed", confidence: "medium", context: "ctx.evidence",
      x: 990, y: 1140, w: 200, h: 50,
      desc: "把运行时 counters / perf 暴露给 Evidence。",
      details: ["不改运行时行为。"],
      refs: ["src/llama-store-io.h", "src/llama-context.cpp"]
    },

    {
      id: "evt.plansolved", label: "PlanSolved", sub: "plan output", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 50, y: 800, w: 170, h: 46,
      desc: "CapacityPlanner 产出 ResidencyPlan 后发布; 下游用它初始化 bank / KV / store。",
      details: ["当前映射: common_moe_plan_get 的输出。"],
      refs: ["common/fit.h:58-71", "common/fit.cpp"]
    },
    {
      id: "evt.expertmiss", label: "ExpertMiss", sub: "miss mask", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 240, y: 800, w: 170, h: 46,
      desc: "本层选中的专家不在 bank; 触发同步拷贝或 CPU 分支。",
      details: ["当前映射: get_miss_mask / count_usage。"],
      refs: ["src/llama-moe-cache.h:32-37", "src/llama-graph.cpp:2400-2416"]
    },
    {
      id: "evt.prefetchissued", label: "PrefetchIssued", sub: "submit", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 430, y: 800, w: 170, h: 46,
      desc: "预取批次已提交到设备队列, 计算队列不等待。",
      details: ["当前映射: llama_store_pf_submit / Vulkan submit。"],
      refs: ["src/llama-store.h:66-69", "src/llama-store-io.h:36-42"]
    },
    {
      id: "evt.prefetchcompleted", label: "PrefetchCompleted", sub: "wait", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 620, y: 800, w: 170, h: 46,
      desc: "预取完成, 消费点可以读该层专家。",
      details: ["当前映射: llama_store_pf_wait。"],
      refs: ["src/llama-store.h:66-71"]
    },
    {
      id: "evt.evictiondeferred", label: "EvictionDeferred", sub: "busy guards", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 810, y: 800, w: 170, h: 46,
      desc: "淘汰被三道 busy 检查挡住, 延后处理。",
      details: ["当前映射: store_evict_deferred 计数。"],
      refs: ["src/llama-store-slab.h:20-38", "src/llama-store-io.h:57-64"]
    },
    {
      id: "evt.budgetexhausted", label: "BudgetExhausted", sub: "capacity", type: "event", kind: "event", status: "proposed", confidence: "medium",
      x: 1000, y: 800, w: 170, h: 46,
      desc: "某层预算不足; 必须进入降级, 不能 abort。",
      details: ["当前映射: 退化/拒绝计数; 4 处 abort 待消灭。"],
      refs: ["bench/THREE-TIER.md:35-50", "bench/THREE-TIER.md:70-81"]
    },
    {
      id: "evt.degraded", label: "Degraded", sub: "mmap / SSD reject", type: "event", kind: "event", status: "proposed", confidence: "medium",
      x: 1190, y: 800, w: 170, h: 46,
      desc: "系统进入容量退化模式: 退 mmap、关预取、拒绝 SSD 同步。",
      details: ["当前映射: mmap fallback / prefetch off / store_ssd_sync_rejected。"],
      refs: ["bench/THREE-TIER.md:23-33", "bench/THREE-TIER.md:70-81"]
    },
    {
      id: "evt.kvpagefault", label: "KvPageFault", sub: "page grow", type: "event", kind: "event", status: "proposed", confidence: "medium",
      x: 1380, y: 800, w: 170, h: 46,
      desc: "KV 工作集需要新的 page / block; 触发 grow 与 gather。",
      details: ["当前映射: paged KV grow / prefill gather。"],
      refs: ["src/llama-kv-cache.cpp:1440-1534"]
    },
    {
      id: "evt.tokensampled", label: "TokenSampled", sub: "sampler", type: "event", kind: "event", status: "proposed", confidence: "high",
      x: 1570, y: 800, w: 170, h: 46,
      desc: "采样出下一个 token; 回到 batch, 形成 decode 循环。",
      details: ["当前映射: llama_sampler / common_sampling。"],
      refs: ["src/llama-sampler.cpp", "common/sampling.cpp"]
    }
  ],

  edges: [
    { id: "e.model_exec", from: "ctx.model", to: "ctx.execution", label: "Published Language: ModelArtifact", type: "published", status: "proposed", fromSide: "top", toSide: "top", route: [[240, 60], [1220, 60]] },
    { id: "e.model_res", from: "ctx.model", to: "ctx.residency", label: "upstream: geometry / quant", type: "upstream", status: "proposed", fromSide: "right", toSide: "left", route: [[470, 210], [470, 370]] },
    { id: "e.delivery_exec", from: "ctx.delivery", to: "ctx.execution", label: "Customer-Supplier: ChatCompletion", type: "cs", status: "proposed", fromSide: "right", toSide: "top", toOffset: -120, route: [[470, 535], [470, 68], [1100, 68]] },
    { id: "e.exec_res", from: "ctx.execution", to: "ctx.residency", label: "Customer-Supplier: ResidencyPort", type: "cs", status: "proposed", fromSide: "left", toSide: "right", route: [[1010, 290], [1010, 370]] },
    { id: "e.res_backend", from: "ctx.residency", to: "ctx.backend", label: "Customer-Supplier + ACL: TransferPort", type: "acl", status: "proposed", fromSide: "right", toSide: "left", route: [[1010, 370], [1010, 64], [1470, 64], [1470, 370]] },
    { id: "e.evidence_res", from: "ctx.evidence", to: "ctx.residency", label: "observer / policy input", type: "observer", status: "proposed" },
    { id: "e.evidence_exec", from: "ctx.evidence", to: "ctx.execution", label: "observer: metrics", type: "observer", status: "proposed" },
    { id: "e.shared_exec", from: "shared.ggml", to: "ctx.execution", label: "shared kernel", type: "shared", status: "proposed" },
    { id: "e.shared_backend", from: "shared.ggml", to: "ctx.backend", label: "shared kernel", type: "shared", status: "proposed" },

    { id: "e.load_model", from: "svc.load", to: "agg.model", label: "creates", type: "contract", status: "proposed" },
    { id: "e.weights_load", from: "port.weights", to: "svc.load", label: "reads", type: "contract", status: "proposed" },
    { id: "e.mmap_weights", from: "adapter.mmap", to: "port.weights", label: "implements", type: "contract", status: "current" },
    { id: "e.model_quant", from: "agg.model", to: "vo.quant", label: "has", type: "contract", status: "proposed" },
    { id: "e.model_geom", from: "agg.model", to: "vo.geom", label: "has", type: "contract", status: "proposed" },
    { id: "e.model_inv", from: "inv.model", to: "agg.model", label: "guards", type: "contract", status: "current" },
    { id: "e.model_session", from: "agg.model", to: "agg.session", label: "weights / geometry", type: "published", status: "proposed" },
    { id: "e.model_plan", from: "agg.model", to: "svc.capacity", label: "geometry / quant", type: "upstream", status: "proposed" },

    { id: "e.server_chat", from: "adapter.server", to: "app.chat", label: "invokes", type: "contract", status: "current" },
    { id: "e.chat_template", from: "app.chat", to: "svc.chat", label: "renders", type: "contract", status: "current" },
    { id: "e.chat_session", from: "app.chat", to: "agg.session", label: "RunPrefill / DecodeStep", type: "cs", status: "proposed" },
    { id: "e.chat_messages", from: "svc.chat", to: "vo.messages", label: "reads", type: "contract", status: "current" },
    { id: "e.chat_stream", from: "app.chat", to: "vo.stream", label: "emits", type: "contract", status: "current" },

    { id: "e.capacity_plan", from: "svc.capacity", to: "agg.plan", label: "solves", type: "contract", status: "proposed" },
    { id: "e.plan_expert", from: "agg.plan", to: "agg.expert", label: "allocates", type: "contract", status: "proposed" },
    { id: "e.plan_kv", from: "agg.plan", to: "agg.kv", label: "reserves", type: "contract", status: "proposed" },
    { id: "e.placement_expert", from: "svc.placement", to: "agg.expert", label: "placement", type: "contract", status: "proposed" },
    { id: "e.eviction_expert", from: "svc.eviction", to: "agg.expert", label: "evict guards", type: "contract", status: "proposed" },
    { id: "e.prefetch_transfer", from: "svc.prefetch", to: "port.transfer", label: "issues", type: "contract", status: "proposed" },
    { id: "e.transfer_expert", from: "port.transfer", to: "agg.expert", label: "fills / reads", type: "contract", status: "proposed" },
    { id: "e.degrade_plan", from: "svc.degrade", to: "agg.plan", label: "degrade", type: "contract", status: "proposed" },
    { id: "e.plan_budget", from: "agg.plan", to: "vo.budget", label: "uses", type: "contract", status: "proposed" },
    { id: "e.plan_tier", from: "agg.plan", to: "vo.tier", label: "uses", type: "contract", status: "proposed" },
    { id: "e.expert_key", from: "agg.expert", to: "vo.expertkey", label: "keyed by", type: "contract", status: "proposed" },
    { id: "e.transfer_ticket", from: "port.transfer", to: "vo.ticket", label: "ticket", type: "contract", status: "proposed" },
    { id: "e.repo_plan", from: "repo.residency", to: "agg.plan", label: "rehydrates / persists", type: "contract", status: "proposed" },
    { id: "e.repo_expert", from: "repo.residency", to: "agg.expert", label: "stores slots", type: "contract", status: "proposed" },
    { id: "e.repo_kv", from: "repo.residency", to: "agg.kv", label: "stores pages", type: "contract", status: "proposed" },
    { id: "e.inv_budget", from: "inv.budget", to: "agg.plan", label: "guards", type: "contract", status: "proposed" },
    { id: "e.inv_evict", from: "inv.evict", to: "agg.expert", label: "guards", type: "contract", status: "proposed" },
    { id: "e.inv_ssd", from: "inv.ssd", to: "svc.degrade", label: "guards", type: "contract", status: "proposed" },
    { id: "e.transfer_acl", from: "port.transfer", to: "acl.vk", label: "ACL: 5 primitives", type: "acl", status: "proposed" },
    { id: "e.kv_session", from: "agg.kv", to: "agg.session", label: "KV cells / pages", type: "cs", status: "proposed" },

    { id: "e.session_graph", from: "agg.session", to: "svc.graph", label: "builds", type: "contract", status: "proposed" },
    { id: "e.session_sampling", from: "agg.session", to: "svc.sampling", label: "samples", type: "contract", status: "proposed" },
    { id: "e.session_ubatch", from: "agg.session", to: "vo.ubatch", label: "has", type: "contract", status: "proposed" },
    { id: "e.graph_exec", from: "svc.graph", to: "port.exec", label: "submits", type: "contract", status: "proposed" },
    { id: "e.session_residency", from: "agg.session", to: "port.residency", label: "acquire / prefetch", type: "contract", status: "proposed" },
    { id: "e.sampling_token", from: "svc.sampling", to: "vo.token", label: "produces", type: "contract", status: "proposed" },
    { id: "e.port_repo", from: "port.residency", to: "repo.residency", label: "implemented by", type: "cs", status: "proposed" },
    { id: "e.exec_shared", from: "port.exec", to: "shared.ggml", label: "uses graph IR", type: "shared", status: "proposed" },
    { id: "e.inv_graph", from: "inv.graph", to: "agg.session", label: "guards", type: "contract", status: "proposed" },

    { id: "e.sched_vk", from: "adapter.sched", to: "adapter.vulkan", label: "dispatches", type: "contract", status: "current" },
    { id: "e.sched_cpu", from: "adapter.sched", to: "adapter.cpu", label: "dispatches", type: "contract", status: "current" },
    { id: "e.sched_shared", from: "adapter.sched", to: "shared.ggml", label: "consumes", type: "shared", status: "proposed" },
    { id: "e.vk_acl", from: "adapter.vulkan", to: "acl.vk", label: "prefetch", type: "contract", status: "partial" },
    { id: "e.acl_ssd", from: "acl.vk", to: "adapter.ssd", label: "cold tier (F5)", type: "contract", status: "planned" },
    { id: "e.exec_adapter", from: "port.exec", to: "adapter.sched", label: "implemented by", type: "cs", status: "proposed" },

    { id: "e.bench_matrix", from: "adapter.bench", to: "svc.matrix", label: "runs", type: "contract", status: "current" },
    { id: "e.ppl_quality", from: "adapter.ppl", to: "svc.quality", label: "runs", type: "contract", status: "current" },
    { id: "e.docs_ledger", from: "adapter.docs", to: "svc.ledger", label: "publishes", type: "contract", status: "current" },
    { id: "e.matrix_exp", from: "svc.matrix", to: "agg.experiment", label: "records", type: "contract", status: "current" },
    { id: "e.quality_exp", from: "svc.quality", to: "agg.experiment", label: "records", type: "contract", status: "current" },
    { id: "e.ledger_exp", from: "svc.ledger", to: "agg.experiment", label: "ledger", type: "contract", status: "current" },
    { id: "e.caliber_exp", from: "inv.caliber", to: "agg.experiment", label: "guards", type: "contract", status: "current" },
    { id: "e.observe_exp", from: "svc.observe", to: "agg.experiment", label: "observes", type: "contract", status: "proposed" },
    { id: "e.telem_observe", from: "adapter.telem", to: "svc.observe", label: "feeds", type: "contract", status: "proposed" },
    { id: "e.event_token", from: "evt.tokensampled", to: "svc.observe", label: "observed", type: "event", status: "proposed" },
    { id: "e.event_budget", from: "evt.budgetexhausted", to: "svc.observe", label: "observed", type: "event", status: "proposed" },

    { id: "e.evt_plansolved", from: "svc.capacity", to: "evt.plansolved", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_expertmiss", from: "agg.expert", to: "evt.expertmiss", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_prefetchissued", from: "svc.prefetch", to: "evt.prefetchissued", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_prefetchcompleted", from: "port.transfer", to: "evt.prefetchcompleted", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_evictiondeferred", from: "svc.eviction", to: "evt.evictiondeferred", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_budgetexhausted", from: "svc.degrade", to: "evt.budgetexhausted", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_degraded", from: "svc.degrade", to: "evt.degraded", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_kvpagefault", from: "agg.kv", to: "evt.kvpagefault", label: "publishes", type: "event", status: "proposed" },
    { id: "e.evt_tokensampled", from: "svc.sampling", to: "evt.tokensampled", label: "publishes", type: "event", status: "proposed" }
  ],

  scenarios: [
    {
      id: "contextmap", name: "上下文地图", key: 1,
      steps: [
        {
          t: "DDD 视图: 六个限界上下文 + 一个共享内核。实线/虚线表示上下文关系; 点上下文看聚合、服务和不变量; 点 2-8 看用例与事件流。",
          f: ["ctx.model", "ctx.delivery", "ctx.residency", "ctx.execution", "ctx.backend", "ctx.evidence", "shared.ggml"],
          e: ["e.model_exec", "e.model_res", "e.delivery_exec", "e.exec_res", "e.res_backend", "e.shared_exec", "e.shared_backend"]
        }
      ]
    },
    {
      id: "loadplan", name: "LoadAndPlan", key: 2,
      steps: [
        {
          t: "LoadAndPlan: ModelLoader 创建不可变 ModelArtifact; geometry / quant 进入 CapacityPlanner, 求解 ResidencyPlan。",
          f: ["ctx.model", "ctx.residency", "ctx.execution", "agg.model", "svc.load", "port.weights", "inv.model", "svc.capacity", "agg.plan", "agg.expert", "agg.kv", "repo.residency", "agg.session", "vo.quant", "vo.geom"],
          e: ["e.load_model", "e.weights_load", "e.model_plan", "e.capacity_plan", "e.plan_expert", "e.plan_kv", "e.model_session", "e.evt_plansolved", "e.model_quant", "e.model_geom"]
        },
        {
          t: "ResidencyPlan 把预算分给 ExpertResidency 与 KvWorkspace; ResidencyRepository 负责 adopt / 记账; 发布 PlanSolved。",
          f: ["agg.plan", "agg.expert", "agg.kv", "repo.residency", "vo.budget", "vo.tier", "inv.budget"],
          e: ["e.plan_budget", "e.plan_tier", "e.repo_plan", "e.repo_expert", "e.repo_kv", "e.inv_budget"]
        }
      ]
    },
    {
      id: "prefill", name: "Prefill", key: 3,
      steps: [
        {
          t: "ChatCompletion 把 messages 渲染成 prompt, 调 InferenceSession; Delivery 只依赖用例, 不依赖 GPU。",
          f: ["ctx.delivery", "ctx.execution", "ctx.residency", "app.chat", "svc.chat", "adapter.server", "agg.session", "svc.graph", "port.exec", "port.residency", "repo.residency", "agg.expert", "agg.kv"],
          e: ["e.server_chat", "e.chat_template", "e.chat_session", "e.session_graph", "e.graph_exec", "e.session_residency", "e.port_repo", "e.kv_session"]
        },
        {
          t: "GraphBuilder 产出 qwen35moe 图; ExecutionPort 提交给 Backend; Residency 已通过 ResidencyPort 准备专家与 KV。",
          f: ["svc.graph", "port.exec", "adapter.sched", "adapter.vulkan", "agg.expert", "agg.kv"],
          e: ["e.graph_exec", "e.exec_adapter", "e.sched_vk", "e.exec_shared", "e.kv_session"]
        }
      ]
    },
    {
      id: "decode", name: "DecodeStep", key: 4,
      steps: [
        {
          t: "Decode: 本层选中的专家不在 bank -> ExpertMiss; PrefetchScheduler 生成 TransferTicket, TransferPort 异步预取。",
          f: ["agg.expert", "svc.prefetch", "port.transfer", "vo.ticket", "acl.vk", "adapter.vulkan", "agg.session", "svc.sampling", "inv.evict"],
          e: ["e.evt_expertmiss", "e.evt_prefetchissued", "e.prefetch_transfer", "e.transfer_acl", "e.transfer_ticket", "e.transfer_expert", "e.evt_prefetchcompleted"]
        },
        {
          t: "EvictionPolicy 只在三道 guard 都空闲时淘汰; 采样出 TokenSampled, 回到下一次 batch。",
          f: ["svc.eviction", "agg.expert", "inv.evict", "svc.sampling", "evt.evictiondeferred", "evt.tokensampled"],
          e: ["e.eviction_expert", "e.inv_evict", "e.evt_evictiondeferred", "e.sampling_token", "e.evt_tokensampled"]
        }
      ]
    },
    {
      id: "degrade", name: "Degrade", key: 5,
      steps: [
        {
          t: "预算不足 -> BudgetExhausted; DegradationPolicy 进入 Degraded: 退 mmap、关预取、拒绝 SSD 同步, 不 abort。",
          f: ["agg.plan", "svc.degrade", "inv.budget", "inv.ssd", "inv.evict", "agg.kv", "adapter.ssd", "evt.budgetexhausted", "evt.degraded"],
          e: ["e.degrade_plan", "e.inv_budget", "e.inv_ssd", "e.evt_budgetexhausted", "e.evt_degraded", "e.acl_ssd"]
        }
      ]
    },
    {
      id: "switchtier", name: "SwitchTier", key: 6,
      steps: [
        {
          t: "SwitchTier: 三个 Budget/Tier 值对象进入 ResidencyPlan; Repository 重新放置专家与 KV; SSD 冷层仍是 planned。",
          f: ["vo.budget", "vo.tier", "agg.plan", "repo.residency", "agg.expert", "agg.kv", "inv.budget", "adapter.ssd"],
          e: ["e.plan_budget", "e.plan_tier", "e.repo_plan", "e.repo_expert", "e.repo_kv", "e.inv_budget", "e.acl_ssd"]
        }
      ]
    },
    {
      id: "evidence", name: "Evidence", key: 7,
      steps: [
        {
          t: "Experiment & Evidence 观察事件与指标, 跑矩阵和质量门; 读数必须带口径, 结论只进 bench/ 四份文档。",
          f: ["ctx.evidence", "agg.experiment", "svc.matrix", "svc.quality", "svc.ledger", "inv.caliber", "svc.observe", "adapter.telem", "evt.tokensampled", "evt.budgetexhausted"],
          e: ["e.bench_matrix", "e.ppl_quality", "e.docs_ledger", "e.matrix_exp", "e.quality_exp", "e.ledger_exp", "e.caliber_exp", "e.event_token", "e.event_budget", "e.evidence_res", "e.evidence_exec"]
        }
      ]
    },
    {
      id: "boundaries", name: "边界原则", key: 8,
      steps: [
        {
          t: "边界原则: 热路径只跨窄端口; Vulkan 细节走 ACL; ggml 是 shared kernel; 规则只升到聚合/策略, 不引入每 token 抽象。",
          f: ["ctx.execution", "ctx.residency", "ctx.backend", "port.exec", "port.residency", "port.transfer", "acl.vk", "repo.residency", "shared.ggml", "inv.evict", "inv.ssd"],
          e: ["e.exec_res", "e.res_backend", "e.exec_adapter", "e.session_residency", "e.port_repo", "e.transfer_acl", "e.exec_shared", "e.shared_backend", "e.inv_evict", "e.inv_ssd"]
        }
      ]
    }
  ]
};
