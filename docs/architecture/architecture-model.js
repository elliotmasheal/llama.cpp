/* fastLlama architecture evidence model.
 * Stable IDs, sourceRefs and confidence are kept here so the interactive view can be
 * regenerated and diffed. Source references are repo-relative paths with optional line
 * ranges; the renderer turns them into ../../<path> links.
 */
window.ARCH_MODEL = {
  meta: {
    title: "fastLlama / qwen35moe 架构",
    branch: "trim-vulkan-cpu-qwen35moe",
    date: "2026-10-29",
    target: "256K ctx + 10K prompt, IQ4_XS, 8176 MiB 档",
    note: "current 只表示代码路径存在; 性能结论与口径以 bench/ 四份文档为准。"
  },

  metrics: [
    { k: "PP", v: "567.4", u: "t/s", cls: "hot" },
    { k: "TG", v: "20.66", u: "t/s", cls: "hot" },
    { k: "专家命中", v: "68.04%", u: "", cls: "" },
    { k: "bank", v: "2376 MiB", u: "", cls: "" },
    { k: "H2D", v: "14.3 GB/s", u: "", cls: "" },
    { k: "SSD 随机", v: "0.81 GB/s", u: "", cls: "warn" }
  ],

  bands: [
    { id: "band.tools", x: 18, y: 10, w: 770, h: 446, label: "接入与编排 (tools/ + common/)", sub: "请求入口 / 模板 / 规划 / 采样" },
    { id: "band.core", x: 804, y: 10, w: 486, h: 712, label: "llama.cpp core (src/)", sub: "加载 / 图 / 记忆 / 专家 / store" },
    { id: "band.ggml", x: 1306, y: 10, w: 356, h: 582, label: "ggml 运行时", sub: "scheduler / backends / ops" },
    { id: "band.hw", x: 18, y: 746, w: 1644, h: 286, label: "硬件与三层存储", sub: "8 GiB VRAM / 40 GiB RAM / SSD" },
    { id: "band.verify", x: 18, y: 1038, w: 1644, h: 140, label: "验证与账本", sub: "A/B 矩阵 / 质量门 / 门禁 / 判决" }
  ],

  nodes: [
    {
      id: "ext.user", label: "用户 / API 客户端", sub: "浏览器 / OpenAI 兼容调用", type: "ext", status: "current", confidence: "high",
      x: 36, y: 56, w: 150, h: 52,
      desc: "请求的起点。浏览器走 WebUI, 程序走 HTTP 的 OpenAI 兼容接口; 两者最终都进入 llama-server。",
      details: ["聊天、补全、流式 SSE 都走同一 HTTP 入口。", "CLI 在没有 --server-base 时也会在本地拉起一个 server 子进程, 客户端始终说 HTTP。"],
      refs: ["tools/server/README.md", "tools/cli/cli-context.h:21-46"]
    },
    {
      id: "ext.gguf", label: "GGUF 模型文件", sub: "Qwen3.6-35B-A3B-UD-IQ4_XS", type: "ext", status: "current", confidence: "high",
      x: 36, y: 130, w: 150, h: 52,
      desc: "单一权重文件。档位是文件选择, 不是引擎行为: 同一二进制换文件即可在 8176 / 6144 / 4096 档之间切换。",
      details: ["目标口径: 40 层 x 256 路由专家, top-8; IQ4_XS resident 约 16.96 GiB。", "Q4_K_XL 约 21.28 GiB, 过不了 27.6 GiB resident 门槛, 会退回 mmap。"],
      refs: ["bench/THREE-TIER.md:15-23", "bench/RESULTS.md:8-18", "AGENTS.md"]
    },
    {
      id: "ext.prompts", label: "口径 prompts", sub: "prompt10k / prompt20k", type: "ext", status: "current", confidence: "high",
      x: 36, y: 204, w: 150, h: 46,
      desc: "固定长度测量输入: prompt10k 约 10428 token, prompt20k 约 19758 token, 保证 PP/TG 读数跨轮可比。",
      details: ["所有矩阵读数必须带日期 / 文件 / ftype / 档位 / bank / hit / status。", "跨口径比较视为错误。"],
      refs: ["bench/prompts/", "bench/README.md", "bench/RESULTS.md:1-5"]
    },

    {
      id: "tool.cli", label: "llama-cli", sub: "交互客户端, 本地起 server", type: "tool", status: "current", confidence: "high",
      x: 210, y: 56, w: 180, h: 52,
      desc: "交互式命令行客户端。维护会话、模板、多模态附件与工具调用展示; 没有 --server-base 时拉起本地 llama-server 子进程, 通过 HTTP 客户端对话。",
      details: ["cli_client 始终存在; cli_server 只在未指定 --server-base 时初始化。", "启动后拉取 server props, 判断 vision / audio / video 能力。"],
      refs: ["tools/cli/cli-context.h:21-46", "tools/cli/main.cpp"]
    },
    {
      id: "tool.server", label: "llama-server", sub: "HTTP + 队列 + slots + 连续批处理", type: "tool", status: "current", confidence: "high",
      x: 210, y: 128, w: 180, h: 58,
      desc: "OpenAI 兼容 HTTP 服务。请求进队列, 每个 slot 维护独立会话; update_slots() 把所有活跃 slot 拼成一个 batch 调 llama_decode, 采样后流式返回。",
      details: ["连续批处理: 多个请求共享一次 decode, 指标 n_busy_slots_per_decode。", "另有 MCP 客户端, 可连接外部工具 server。"],
      refs: ["tools/server/server-context.cpp:3431", "tools/server/README-dev.md:78-84", "tools/server/server-mcp.h:15-38"]
    },
    {
      id: "tool.ui", label: "WebUI", sub: "Svelte 聊天界面", type: "tool", status: "current", confidence: "high",
      x: 210, y: 206, w: 180, h: 44,
      desc: "浏览器聊天界面, 由 server 托管; 与 OpenAI 兼容 API 共用同一个 HTTP 入口。",
      details: ["前端代码在 tools/ui, 构建产物嵌入 server。", "支持流式消息与会话管理。"],
      refs: ["tools/ui/README.md", "tools/ui/svelte.config.js"]
    },
    {
      id: "tool.bench", label: "llama-bench", sub: "PP / TG 矩阵", type: "tool", status: "current", confidence: "high",
      x: 210, y: 270, w: 180, h: 44,
      desc: "以固定 prompt 和参数重复测量 PP/TG。bench/tools/run-matrix.sh 驱动 -T 8176,6144,4096 -r 2 的三档矩阵。",
      details: ["每轮只改一个机制, 跑完矩阵后把 ΔPP / ΔTG / 代价条件记入 RESULTS 2.1。", "数字必须带口径, 原始日志按 git 历史复核。"],
      refs: ["tools/llama-bench", "bench/tools/run-matrix.sh", "AGENTS.md"]
    },
    {
      id: "tool.mtmd", label: "mtmd", sub: "图像 / 音频 / 视频输入", type: "tool", status: "current", confidence: "high",
      x: 210, y: 334, w: 180, h: 44,
      desc: "多模态前端: 把媒体编码成 embedding 注入 batch。CLI 通过 server 暴露 vision / audio / video 能力。",
      details: ["媒体 embedding 与文本 token 在 batch 层汇合。", "模型是否支持由 server props 与模型元数据决定。"],
      refs: ["tools/mtmd", "tools/cli/cli-context.h:30-34", "tools/cli/cli-context.h:68-70"]
    },
    {
      id: "tool.perplexity", label: "perplexity", sub: "PPL / KLD 质量门", type: "tool", status: "current", confidence: "high",
      x: 210, y: 398, w: 180, h: 44,
      desc: "质量门的一部分。quality-gate.sh 用它对比 PPL 与 KLD; 动注意力数值 (含 -fac) 或量化路径时必须过。",
      details: ["当前门禁读数: PPL 2.1676 对 2.1676, KLD -0.000000, same top p 100%。", "质量门通过记 GATE PASS。"],
      refs: ["tools/perplexity", "bench/tools/quality-gate.sh", "bench/RESULTS.md:83", "bench/RESULTS.md:100"]
    },

    {
      id: "common.chat", label: "Chat 模板", sub: "Jinja / PEG / 工具调用", type: "common", status: "current", confidence: "high",
      x: 420, y: 56, w: 170, h: 52,
      desc: "把消息历史渲染成模型 prompt。支持 Jinja 模板、PEG 解析与工具调用格式; CLI 与 server 共用。",
      details: ["模板在 prompt 进入 tokenizer 之前生效。", "工具调用解析与 JSON schema 约束也在 common/ 下。"],
      refs: ["common/chat.cpp", "common/chat-peg-parser.cpp", "common/json-schema-to-grammar.cpp"]
    },
    {
      id: "common.arg", label: "参数与预设", sub: "三类用户旋钮", type: "common", status: "current", confidence: "high",
      x: 620, y: 56, w: 150, h: 52,
      desc: "解析 CLI 参数与 preset。用户可见只有三类: 档位与预算 (--store-*-mib)、容量 (-c / -ncmoe / -lm)、标注 experimental 的开关。",
      details: ["--store-vram-mib / --store-ram-mib / --store-ssd-mib 是三个互不挪用的预算。", "开关只用于 A/B, 赢了折叠进自适应, 输了删除, 不做永久产品旋钮。"],
      refs: ["common/arg.cpp:2433-2943", "AGENTS.md"]
    },
    {
      id: "common.fit", label: "规划器 + 内存账本", sub: "统一预算式, 决定 bank / KV / store", type: "common", status: "current", confidence: "high",
      x: 420, y: 128, w: 170, h: 70,
      desc: "先量模型几何与空闲显存, 再用统一账本 (dense / kv_context / compute / fa_score / kv_gather / margin / bank) 求解; 结果决定 -ncmoe、bank 大小、预取环与 --kv-page。store 的三个整机预算另由 --store-*-mib 或 auto 进入 common 运行时。",
      details: ["账本是预算算术的唯一所有者, 求解消费条目, 不重新推导。", "分配顺序: KV -> bank -> compute; score 缓冲与 256K KV 在同一预算式里。", "resident 门槛: n_bytes x 1.25 + 1 GiB。"],
      refs: ["common/fit.h:24-71", "common/fit.cpp", "bench/THREE-TIER.md:24-50", "AGENTS.md"]
    },
    {
      id: "common.sampling", label: "采样链", sub: "top-k/p, temp, grammar", type: "common", status: "current", confidence: "high",
      x: 620, y: 128, w: 150, h: 52,
      desc: "采样器链: penalties、top-k、top-p、min-p、temperature、grammar。server 每个 slot 独立采样。",
      details: ["采样输入是 context 产出的 logits。", "grammar 约束可与 JSON schema 组合。"],
      refs: ["common/sampling.cpp", "tools/server/README-dev.md:84"]
    },
    {
      id: "common.speculative", label: "投机 / NextN", sub: "可选 draft 路径", type: "common", status: "experimental", confidence: "medium",
      x: 420, y: 220, w: 170, h: 48,
      desc: "提供 draft 模型与 NextN / MTP 投机解码接口。qwen35moe 支持单个 MTP 块 (graph_mtp); 是否启用取决于参数, 不属于目标口径主链。",
      details: ["qwen35moe 的 MTP 断言只支持单个 NextN 块。", "本图默认主链不把投机算进 PP/TG 读数。"],
      refs: ["common/speculative.cpp", "src/models/qwen35moe.cpp:544-548", "src/llama-ext.h:127-130"]
    },
    {
      id: "common.runtime", label: "公共运行时装配", sub: "store / context / adopt", type: "common", status: "current", confidence: "high",
      x: 620, y: 220, w: 150, h: 48,
      desc: "common/common.cpp 负责模型到 context 的装配: 读取三个整机预算初始化 store, 创建 llama_context; context 建好后按 memory breakdown 申报 dense/compute, 再按 expert geometry 逐 (layer, expert) adopt, 最后 commit 打印 store_split。",
      details: ["store 初始化在 common/common.cpp:1344-1352, adopt/commit 在 1441-1479。", "adopt 是记账与 O(1) 索引, 不是放置权威; transfer executor 2c 已接, M1/M2 未验收。", "KV page adoption 留到 M3。"],
      refs: ["common/common.cpp:1344-1352", "common/common.cpp:1441-1479", "src/llama-store.h:47-57"]
    },

    {
      id: "core.vocab", label: "Tokenizer", sub: "BPE + Qwen3.5 预切分", type: "core", status: "current", confidence: "high",
      x: 824, y: 60, w: 150, h: 48,
      desc: "文本 -> token id。Qwen3.5 有专用预切分正则; chat 模板的输出在这里变成 token。",
      details: ["词表与 pre-tokenizer 类型由 GGUF 元数据决定。", "unicode 正则与 C++ 实现保持逐字一致。"],
      refs: ["src/llama-vocab.cpp:399", "src/llama-vocab.cpp:2532-2533", "src/unicode.cpp:777-779"]
    },
    {
      id: "core.batch", label: "Batch / ubatch", sub: "逻辑批 -> 物理 ubatch", type: "core", status: "current", confidence: "high",
      x: 982, y: 60, w: 150, h: 48,
      desc: "把多序列 batch 拆成等长 ubatch, 分配 KV cell 索引。server 的连续批处理在这里汇合成一次 decode。",
      details: ["ubatch 是图构建与 KV 分配的单位。", "多模态 embedding 与 token 在 batch 层合流。"],
      refs: ["src/llama-batch.cpp", "src/llama-context.h:145-156"]
    },
    {
      id: "core.model_loader", label: "模型加载器", sub: "GGUF / mmap / lazy / resident", type: "core", status: "current", confidence: "high",
      x: 824, y: 126, w: 150, h: 56,
      desc: "读 GGUF 元数据与张量, 决定 mmap、lazy read 或 resident; 按 -ncmoe 把专家留在 host, dense 放设备。加载方式是 bank 与预取的容量前提。",
      details: ["-lm none 请求整模型 resident, 容量不够时退回 mmap。", "mmap 下 bank 与预取强制 0, 这是容量退化路径。", "store 的 adopt 发生在 common 运行时, 不在加载器; 加载器只决定张量落在哪。"],
      refs: ["src/llama-model-loader.cpp:542-567", "src/llama-model-loader.cpp:1438-1465", "src/llama-model-loader.cpp:1580-1698", "bench/THREE-TIER.md:23-33"]
    },
    {
      id: "core.model", label: "模型与张量", sub: "qwen35moe, 40 x 256 experts", type: "core", status: "current", confidence: "high",
      x: 982, y: 126, w: 150, h: 56,
      desc: "架构元数据、hparams 与每层张量。qwen35moe: 40 层, 256 路由专家 top-8, 共享专家; 默认每 4 层一个 full attention, 其余 GDN 线性注意力; 可选 1 个 MTP 块。",
      details: ["层内: attn_norm -> [GDN linear | full attn] -> post_norm -> MoE FFN + shared expert -> residual。", "MoE: router softmax -> top-8 -> MUL_MAT_ID -> weighted sum。", "MTP 层是 full attention + MoE, 带 nextn.eh_proj / enorm / hnorm。"],
      refs: ["src/models/qwen35moe.cpp:4-34", "src/models/qwen35moe.cpp:54-104", "src/models/qwen35moe.cpp:488-542", "src/llama-arch.cpp:44", "AGENTS.md"]
    },
    {
      id: "core.context", label: "llama_context", sub: "调度器 / 图执行 / 输出", type: "core", status: "current", confidence: "high",
      x: 900, y: 206, w: 180, h: 56,
      desc: "推理上下文。创建 backend 与 scheduler, 管理 memory / moe_cache / store, 组织 ubatch 图, 调 ggml_backend_sched_graph_compute_async, 处理 logits / embedding 与 graph reuse。",
      details: ["构造时按能力选择 backend, 建立 compute buffer 与回调。", "moe_cache 在 context 构造时创建, 大小来自 cparams.moe_cache_size。", "图执行入口: llama-context.cpp:2916。"],
      refs: ["src/llama-context.h:45-69", "src/llama-context.cpp:614-661", "src/llama-context.cpp:2916"]
    },
    {
      id: "core.graph", label: "图构建", sub: "attention + MoE 子图", type: "core", status: "current", confidence: "high",
      x: 900, y: 280, w: 180, h: 56,
      desc: "生成 ggml 计算图。qwen35moe 每层先做 GDN 线性注意力或 full attention, 再做 MoE FFN (router -> top-8 -> MUL_MAT_ID -> sum) 和共享专家; 图内插入 MoE cache 的 miss mask 与 CPU 分支。",
      details: ["build_moe_ffn 负责 router、top-k、权重归一与专家乘。", "hybrid 模式: get_miss_mask 标出不在 bank 的专家, CPU 分支复用同一组 ids。", "图可按上次的 backend 分配复用, 减少 realloc。"],
      refs: ["src/models/qwen35moe.cpp:276-355", "src/models/qwen35moe.cpp:357-486", "src/llama-graph.cpp:2237-2416"]
    },
    {
      id: "core.memory", label: "KV / 状态记忆", sub: "KV cells / paged KV / GDN state", type: "core", status: "current", confidence: "high",
      x: 824, y: 360, w: 166, h: 74,
      desc: "KV cache 与 recurrent state。256K 的 KV 工作集约 320 MiB, 全分配约 5120 MiB; -kvg -1 按工作集增长; 可选 --kv-page 以 2048-cell block 分页并做 prefill gather; -nkvo 把 KV 放 host, 同时把 GDN 状态 (62.81 MiB) 也丢到 host。",
      details: ["分配顺序: KV -> bank -> compute。", "默认路径 GDN state 常驻设备; 它与 ctx 无关, 是常数开销。", "分页 KV 6144 档 PP -1.7%, 是容量选配, 默认 off。"],
      refs: ["src/llama-kv-cache.h:99-123", "src/llama-kv-cache.cpp:123-166", "src/llama-kv-cache.cpp:1372-1534", "bench/RESULTS.md:56", "bench/RESULTS.md:94"]
    },
    {
      id: "core.moe_cache", label: "MoE 专家缓存", sub: "bank / SLRU / 预取环 / hybrid", type: "core", status: "current", confidence: "high",
      x: 1000, y: 360, w: 166, h: 80,
      desc: "把 host 常驻专家按热度放进设备 bank。按专家张量布局哈希分组; 组内 SLRU 淘汰; 大 ubatch 走两层整层预取环 (每组至少 512 slots); 静态整层每池最多 5 层; decode 时 copy callback 只上传本层用到的 miss 专家; 可选 hybrid 把 miss 交给 CPU worker。",
      details: ["目标口径 bank 2376 MiB / 1552 slots, hit 68.04%。", "预取环在 40 层被拆成 37/2/1 组后关闭, 后两组过不了 512 slots 门槛。", "静态整层目标口径设为 0 层: 无环比 4 层静态更好 (TG 19.06 -> 20.66)。", "hybrid exec 默认关: 旧口径 TG -52%, 按 20-25 GB/s 需重测 (F4)。"],
      refs: ["src/llama-moe-cache.h:11-88", "src/llama-moe-cache.cpp:641-681", "src/llama-moe-cache.cpp:1334-1440", "src/llama-moe-cache.cpp:1464-1525", "bench/RESULTS.md:47-54"]
    },
    {
      id: "core.moe_exec", label: "MoE 执行器", sub: "router -> ids -> experts", type: "core", status: "experimental", confidence: "high",
      x: 824, y: 456, w: 166, h: 48,
      desc: "Stage 2 执行器: 把一层的 router 与 routed experts 拆成两个小图, 在同一 backend 上跑, 绕过 ggml_backend_sched; 用于测量每层 router / expert 成本。不是默认主链。",
      details: ["self_test 用真实模型张量与 backend 测每层成本。", "与主链的 hybrid CPU 分支不是同一条路径。"],
      refs: ["src/llama-moe-exec.h:10-25", "src/llama-moe-exec.cpp"]
    },
    {
      id: "core.store", label: "llama-store", sub: "2a-2c 已落地 / M1 未验收 / M2 未开工", type: "core", status: "partial", confidence: "high",
      x: 1000, y: 462, w: 166, h: 70,
      desc: "常开的三层抽象, 只拥有专家 (层 + 专家子对象) 与 KV 页两类对象。当前落地 2a/2b/2c: common 运行时在 context 创建后做专家两级 adopt, commit 打印 store_split; backend 传输绑定与预取路由已接。M1 未验收, M2 (use_step) 未开工, SSD 冷层未开工。",
      details: ["设计入口 store_acquire / store_prefetch 尚未实现; 现有入口是 adopt / commit / bind_backend / pf_tensor / submit / wait。", "内部三单元 slab / io / policy 不得合并; policy 可热换而 slab 不动。", "SSD 同步请求直接拒绝并计数 store_ssd_sync_rejected, 不 abort。", "三个预算互不挪用; store_split 进启动日志。"],
      refs: ["src/llama-store.h:7-72", "src/llama-store-slab.h", "src/llama-store-io.h", "src/llama-store-policy.h", "bench/THREE-TIER.md:70-81"]
    },
    {
      id: "core.quant", label: "量化与 repack", sub: "IQ4_XS / Q4_K / Q3_K / Q2_K", type: "core", status: "current", confidence: "high",
      x: 824, y: 536, w: 166, h: 46,
      desc: "量化类型与反量化 / repack 路径。档位是文件选择; q6_K dense 的 mmq 解包是已知瓶颈; expert-repack 只重排专家载荷, 不改类型。",
      details: ["q6_K dense 只有 3.6 TF, q4_K 有 6.3-6.6 TF。", "GPU id GEMM 6.04 TF; CPU MoE 20-25 GB/s。"],
      refs: ["src/llama-quant.cpp", "bench/RESULTS.md:69", "bench/RESULTS.md:93"]
    },
    {
      id: "core.sampler", label: "Sampler", sub: "llama_sampler 链", type: "core", status: "current", confidence: "high",
      x: 1000, y: 552, w: 166, h: 46,
      desc: "词汇表 logits -> token。common 采样链的底层实现; grammar 约束也挂在这里。",
      details: ["server 每个 slot 独立采样。", "输出 token 回到 batch, 形成 decode 循环。"],
      refs: ["src/llama-sampler.cpp", "src/llama-sampler.h"]
    },

    {
      id: "ggml.sched", label: "后端调度器", sub: "图切分 / 回调 / 异步执行", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 60, w: 316, h: 64,
      desc: "按 backend 能力切分图并放置张量; 暴露 pre-split / split / copy 回调; ggml_backend_sched_graph_compute_async 驱动执行。",
      details: ["MoE cache 的 copy / stage_inputs / wait_deferred / pre_split 都挂在调度回调上。", "单卡事件等待 (--sched-parallel on) 是中性开关, 待删。"],
      refs: ["ggml/src/ggml-backend.cpp:2325-2359", "src/llama-context.cpp:841-852", "src/llama-context.cpp:2965"]
    },
    {
      id: "ggml.alloc", label: "图分配器", sub: "compute buffer 复用", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 140, w: 316, h: 48,
      desc: "图内临时张量的分配与复用。context 按 backend 预留 compute buffer; 账本里的 compute 项来自这里。",
      details: ["realloc 计数分 same / diff 两类。", "长跨距 score 缓冲与 paged gather 工作区由 fit 账本预留。"],
      refs: ["ggml/src/ggml-alloc.c", "src/llama-context.h:412-416", "src/llama-context.h:379-396"]
    },
    {
      id: "ggml.ops", label: "张量 / 算子 / 量化", sub: "MUL_MAT_ID + MUL_MAT_ID_MASK", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 204, w: 316, h: 48,
      desc: "ggml 张量库与算子。本 fork 的关键算子: MUL_MAT_ID (MoE 路由乘) 与只有 CPU 实现的 MUL_MAT_ID_MASK (miss 掩码); Vulkan 侧缺 MASK, 是 F4 的 P1 前置。",
      details: ["门禁: MUL_MAT_ID CPU 945/945, Vulkan 939/939, MASK CPU 4/4。", "FLASH_ATTN_EXT Vulkan 5332/5332。"],
      refs: ["ggml/include/ggml.h", "ggml/src/ggml.c", "bench/THREE-TIER.md:41-42", "bench/RESULTS.md:96-100"]
    },
    {
      id: "ggml.cpu", label: "CPU backend", sub: "GEMM / repack / MASK", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 270, w: 316, h: 64,
      desc: "CPU 后端: dense 的 host 部分、hybrid miss、MUL_MAT_ID_MASK 与全部 CPU 门禁; 就地算 miss 的有效带宽 20-25 GB/s。",
      details: ["高于 PCIe 14.3 GB/s, 所以 F4 (CPU 算 miss) 重新成为候选。", "线程数 -t 影响带宽: -t 5 约 21.4, -t 4 约 24.2 GB/s。"],
      refs: ["ggml/src/ggml-cpu/", "bench/RESULTS.md:36-37", "bench/RESULTS.md:91"]
    },
    {
      id: "ggml.vulkan", label: "Vulkan backend", sub: "shaders + prefetch seq", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 350, w: 316, h: 80,
      desc: "Vulkan 后端与 shader 集合; 暴露 prefetch_tensor / submit / wait / submit_seq / wait_seq 五个原语, 供 MoE 预取与 store-io 使用; 单卡 dense / id GEMM 6.3-6.6 TF。",
      details: ["目标设备 RX 6700S 8 GiB, -dev Vulkan1。", "流式预取把整层专家拷贝排进设备队列, 计算队列不等待它。", "5 个原语在 ggml-vulkan.cpp:16820-16824 注册。"],
      refs: ["ggml/src/ggml-vulkan/ggml-vulkan.cpp:13686-13771", "ggml/src/ggml-vulkan/ggml-vulkan.cpp:16820-16824", "bench/RESULTS.md:93"]
    },
    {
      id: "ggml.threads", label: "Threadpool", sub: "CPU 计算线程", type: "ggml", status: "current", confidence: "high",
      x: 1326, y: 446, w: 316, h: 44,
      desc: "CPU 侧线程池, 支持 batch / 单 token 两种线程数; -t 影响 CPU 就地带宽与 hybrid 分支表现。",
      details: ["context 可挂载或分离 threadpool。", "CPU 计算池与 IO 池不得合并 (F13)。"],
      refs: ["ggml/src/ggml-threading.cpp", "src/llama-context.h:404-410", "bench/THREE-TIER.md:88"]
    },

    {
      id: "hw.vram", label: "VRAM 8 GiB", sub: "bank 2376 / KV 320 / compute", extra: "RX 6700S / 6.3-6.6 TF / H2D 14.3 GB/s", type: "hw", status: "current", confidence: "high",
      x: 1040, y: 800, w: 580, h: 160,
      desc: "Radeon RX 6700S 的显存。目标口径: bank 2376.25 MiB, hit 68.04%, 单次运行上传 9727.62 MiB; KV 工作集约 320 MiB; compute 与 score 缓冲由 fit 账本预留。",
      details: ["桌面占 0.5-1.4 GiB, 预算必须留 margin。", "设备总量 8176 MiB; resident 门槛 n_bytes x 1.25 + 1 GiB。", "H2D 14.3 GB/s, 全双工上限约 23.3 GB/s。"],
      refs: ["bench/RESULTS.md:8-13", "bench/RESULTS.md:89-94", "bench/THREE-TIER.md:7-13", "AGENTS.md"]
    },
    {
      id: "hw.ram", label: "RAM 40 GiB", sub: "resident 16.96 GiB / staging / -nkvo KV", extra: "Ryzen 7 6800HS / CPU MoE 20-25 GB/s", type: "hw", status: "current", confidence: "high",
      x: 440, y: 800, w: 580, h: 160,
      desc: "host 内存。8 GiB 档 resident 基准 IQ4_XS 16.96 GiB; 专家整层常驻是 bank 与预取的前提; -nkvo 时 KV 也在这里; staging 与预取缓冲在这里。",
      details: ["resident 不是免费: IQ4_XS 16.96 GiB, Q4_K_XL 21.28 GiB 过不了 27.6 GiB 门槛。", "RAM 不是计算层 (现状); CPU 就地算 miss 20-25 GB/s, 是 F4 候选。", "DRAM 读 37.0 GB/s 多线程 / 39.7 GB/s @4T。"],
      refs: ["AGENTS.md", "bench/THREE-TIER.md:7-13", "bench/RESULTS.md:89-91"]
    },
    {
      id: "hw.ssd", label: "SSD 冷层", sub: "0.81 GB/s / 只做 prefill 与预取 / F5 未开工", extra: "随机 1.95 MiB / 206 MiB/token = 258 ms", type: "hw", status: "planned", confidence: "high",
      x: 440, y: 972, w: 1180, h: 56,
      desc: "未开工的容量层。随机 1.95 MiB 只有 0.81 GB/s, 206 MiB/token = 258 ms, 所以绝不允许进 decode 单步; 设计上只服务命中率 >= 85% 的冷专家。",
      details: ["前置: expert-repack KLD = 0。", "SSD 对单步直接返回不可用并计数, 池不可用退化为同步并计数, 不 abort。"],
      refs: ["bench/THREE-TIER.md:7-13", "bench/THREE-TIER.md:44-45", "bench/RESULTS.md:28"]
    },

    {
      id: "tool.matrix", label: "run-matrix.sh", sub: "A/B 矩阵", type: "meta", status: "current", confidence: "high",
      x: 40, y: 1080, w: 260, h: 64,
      desc: "每轮改一个机制后跑 -T 8176,6144,4096 -r 2 矩阵; ΔPP / ΔTG / 代价条件记入 RESULTS 2.1。",
      details: ["TG 权重 2, PP 权重 1; 同权重先看 r=2 可复现性。", "status 非 ok 记入未结项。"],
      refs: ["bench/tools/run-matrix.sh", "AGENTS.md"]
    },
    {
      id: "tool.quality_gate", label: "quality-gate.sh", sub: "PPL / KLD", type: "meta", status: "current", confidence: "high",
      x: 320, y: 1080, w: 260, h: 64,
      desc: "动注意力数值 (含 -fac) 或量化路径时必跑; 记录 PPL、KLD 与 same top p。当前 GATE PASS。",
      details: ["质量门与性能矩阵是两条独立门禁。", "KLD 0 是 expert-repack 与 store 冷层的前置。"],
      refs: ["bench/tools/quality-gate.sh", "bench/RESULTS.md:83", "bench/RESULTS.md:100"]
    },
    {
      id: "tool.tests", label: "门禁测试", sub: "op tests + ctest", type: "meta", status: "current", confidence: "high",
      x: 600, y: 1080, w: 260, h: 64,
      desc: "MUL_MAT_ID CPU 945/945、Vulkan 939/939, MASK CPU 4/4, MUL_MAT Vulkan 2275/2275, FLASH_ATTN_EXT Vulkan 5332/5332, ctest 33/34。",
      details: ["test-model-resolution 是本机环境性失败。", "动 Vulkan / 调度 / GEMM 必须跑回归。"],
      refs: ["AGENTS.md", "bench/RESULTS.md:96-100"]
    },
    {
      id: "docs", label: "bench/ 四份文档", sub: "RESULTS / DECISIONS / THREE-TIER / README", type: "meta", status: "current", confidence: "high",
      x: 880, y: 1080, w: 300, h: 64,
      desc: "数字、判决、缺口与口径的唯一落点; 结论不新增第五份 md。架构页只读它们, 不写回。",
      details: ["RESULTS: 数字与机制收益账本。", "DECISIONS: 判决、已证伪、硬约束、未结项。", "THREE-TIER: 三层分工、F0-F13 缺口、唯一执行顺序。"],
      refs: ["bench/README.md", "bench/RESULTS.md", "bench/DECISIONS.md", "bench/THREE-TIER.md"]
    },
    {
      id: "tool.expert_repack", label: "expert-repack", sub: "热度重排 / F5 前置", type: "meta", status: "planned", confidence: "high",
      x: 1200, y: 1080, w: 260, h: 64,
      desc: "纯置换 GGUF 专家与 router 载荷, 让每层热专家排在前面; 量化类型、形状、元数据不变, 输出 perm 映射, 用质量门确认 KLD = 0。",
      details: ["输入是 LLAMA_MOE_STATS 统计文件。", "F5 SSD 冷层的前置, 目前未接入执行顺序。"],
      refs: ["tools/expert-repack/expert_repack.py:1-13", "bench/THREE-TIER.md:44-45"]
    }
  ],

  edges: [
    { id: "e.user_cli", from: "ext.user", to: "tool.cli", label: "聊天 / 补全", type: "call", status: "current" },
    { id: "e.user_server", from: "ext.user", to: "tool.server", label: "HTTP / SSE", type: "call", status: "current" },
    { id: "e.ui_server", from: "tool.ui", to: "tool.server", label: "HTTP", type: "call", status: "current" },
    { id: "e.cli_server", from: "tool.cli", to: "tool.server", label: "HTTP (本地子进程)", type: "call", status: "current" },
    { id: "e.server_chat", from: "tool.server", to: "common.chat", label: "messages", type: "call", status: "current" },
    { id: "e.chat_vocab", from: "common.chat", to: "core.vocab", label: "prompt 文本", type: "data", status: "current" },
    { id: "e.server_batch", from: "tool.server", to: "core.batch", label: "共享 batch", type: "data", status: "current" },
    { id: "e.vocab_batch", from: "core.vocab", to: "core.batch", label: "token ids", type: "data", status: "current" },
    { id: "e.batch_context", from: "core.batch", to: "core.context", label: "ubatch", type: "data", status: "current" },
    { id: "e.context_graph", from: "core.context", to: "core.graph", label: "构图", type: "call", status: "current" },
    { id: "e.graph_memory", from: "core.graph", to: "core.memory", label: "K/V 读写", type: "data", status: "current" },
    { id: "e.graph_cache", from: "core.graph", to: "core.moe_cache", label: "selected ids / miss mask", type: "data", status: "current" },
    { id: "e.context_sched", from: "core.context", to: "ggml.sched", label: "ggml_cgraph / async", type: "call", status: "current" },
    { id: "e.sched_vulkan", from: "ggml.sched", to: "ggml.vulkan", label: "split", type: "call", status: "current" },
    { id: "e.sched_cpu", from: "ggml.sched", to: "ggml.cpu", label: "split", type: "call", status: "current" },
    { id: "e.sched_cache", from: "ggml.sched", to: "core.moe_cache", label: "copy / pre-split 回调", type: "call", status: "current" },
    { id: "e.context_sampler", from: "core.context", to: "core.sampler", label: "logits", type: "data", status: "current" },
    { id: "e.sampler_common", from: "common.sampling", to: "core.sampler", label: "采样链", type: "call", status: "current" },
    { id: "e.common_server", from: "common.sampling", to: "tool.server", label: "token", type: "data", status: "current" },
    { id: "e.server_user", from: "tool.server", to: "ext.user", label: "SSE", type: "data", status: "current" },
    { id: "e.mtmd_server", from: "tool.mtmd", to: "tool.server", label: "media parts", type: "data", status: "current" },
    { id: "e.spec_context", from: "common.speculative", to: "core.context", label: "draft decode", type: "call", status: "experimental" },

    { id: "e.gguf_loader", from: "ext.gguf", to: "core.model_loader", label: "read", type: "data", status: "current" },
    { id: "e.loader_model", from: "core.model_loader", to: "core.model", label: "tensors / meta", type: "data", status: "current" },
    { id: "e.loader_ram", from: "core.model_loader", to: "hw.ram", label: "resident / mmap", type: "tier", status: "current" },
    { id: "e.loader_vram", from: "core.model_loader", to: "hw.vram", label: "dense", type: "tier", status: "current" },
    { id: "e.model_context", from: "core.model", to: "core.context", label: "weights / geometry", type: "data", status: "current" },
    { id: "e.model_fit", from: "core.model", to: "common.fit", label: "几何 / 用量", type: "data", status: "current" },
    { id: "e.arg_fit", from: "common.arg", to: "common.fit", label: "参数", type: "config", status: "current" },
    { id: "e.arg_context", from: "common.arg", to: "core.context", label: "cparams", type: "config", status: "current" },
    { id: "e.fit_loader", from: "common.fit", to: "core.model_loader", label: "-ncmoe / 放置", type: "config", status: "current" },
    { id: "e.fit_cache", from: "common.fit", to: "core.moe_cache", label: "bank 预算", type: "config", status: "current" },
    { id: "e.fit_memory", from: "common.fit", to: "core.memory", label: "KV 预算 / kv_page", type: "config", status: "current" },
    { id: "e.arg_runtime", from: "common.arg", to: "common.runtime", label: "params / store 预算", type: "config", status: "current" },
    { id: "e.runtime_context", from: "common.runtime", to: "core.context", label: "llama_init_from_model", type: "call", status: "current" },
    { id: "e.runtime_store", from: "common.runtime", to: "core.store", label: "初始化 + adopt 专家", type: "call", status: "current" },

    { id: "e.memory_vram", from: "core.memory", to: "hw.vram", label: "KV 工作集 320 MiB", type: "tier", status: "current" },
    { id: "e.memory_ram", from: "core.memory", to: "hw.ram", label: "-nkvo / 容量退化", type: "tier", status: "current" },
    { id: "e.cache_vram", from: "core.moe_cache", to: "hw.vram", label: "bank 2376 MiB", type: "tier", status: "current" },
    { id: "e.cache_ram", from: "core.moe_cache", to: "hw.ram", label: "host 专家", type: "tier", status: "current" },
    { id: "e.ram_vram", from: "hw.ram", to: "hw.vram", label: "miss 上传 H2D 14.3 GB/s", type: "tier", status: "current" },
    { id: "e.cache_store", from: "core.moe_cache", to: "core.store", label: "prefetch 路由 (2c)", type: "call", status: "current" },
    { id: "e.store_vram", from: "core.store", to: "hw.vram", label: "VRAM 预算", type: "tier", status: "current" },
    { id: "e.store_ram", from: "core.store", to: "hw.ram", label: "RAM 预算", type: "tier", status: "current" },
    { id: "e.store_ssd", from: "core.store", to: "hw.ssd", label: "SSD 冷层 (F5)", type: "tier", status: "planned" },
    { id: "e.exec_cpu", from: "core.moe_exec", to: "ggml.cpu", label: "独立执行器", type: "call", status: "experimental" },

    { id: "e.alloc_sched", from: "ggml.alloc", to: "ggml.sched", label: "compute buffers", type: "call", status: "current" },
    { id: "e.ops_cpu", from: "ggml.ops", to: "ggml.cpu", label: "kernels", type: "call", status: "current" },
    { id: "e.ops_vulkan", from: "ggml.ops", to: "ggml.vulkan", label: "kernels", type: "call", status: "current" },
    { id: "e.cpu_threads", from: "ggml.cpu", to: "ggml.threads", label: "threadpool", type: "call", status: "current" },
    { id: "e.vulkan_vram", from: "ggml.vulkan", to: "hw.vram", label: "compute + bank", type: "tier", status: "current" },
    { id: "e.cpu_ram", from: "ggml.cpu", to: "hw.ram", label: "host 专家", type: "tier", status: "current" },

    { id: "e.prompts_bench", from: "ext.prompts", to: "tool.bench", label: "10K / 20K", type: "data", status: "current" },
    { id: "e.bench_context", from: "tool.bench", to: "core.context", label: "llama_decode", type: "call", status: "current" },
    { id: "e.matrix_bench", from: "tool.matrix", to: "tool.bench", label: "驱动", type: "call", status: "current" },
    { id: "e.quality_perplexity", from: "tool.quality_gate", to: "tool.perplexity", label: "PPL / KLD", type: "call", status: "current" },
    { id: "e.perplexity_context", from: "tool.perplexity", to: "core.context", label: "logits", type: "call", status: "current" },
    { id: "e.matrix_docs", from: "tool.matrix", to: "docs", label: "ΔPP / ΔTG", type: "data", status: "current" },
    { id: "e.quality_docs", from: "tool.quality_gate", to: "docs", label: "门禁读数", type: "data", status: "current" },
    { id: "e.tests_cpu", from: "tool.tests", to: "ggml.cpu", label: "op tests", type: "validation", status: "current" },
    { id: "e.tests_vulkan", from: "tool.tests", to: "ggml.vulkan", label: "op tests", type: "validation", status: "current" },
    { id: "e.docs_fit", from: "docs", to: "common.fit", label: "判决 / 缺口 -> 规则", type: "config", status: "current" },
    { id: "e.repack_ssd", from: "tool.expert_repack", to: "hw.ssd", label: "冷专家重排 (F5 前置)", type: "tier", status: "planned" },
    { id: "e.repack_gguf", from: "tool.expert_repack", to: "ext.gguf", label: "out.gguf", type: "data", status: "planned" }
  ],

  scenarios: [
    {
      id: "overview", name: "总览", key: 1,
      steps: [
        {
          t: "主链从接入层进入, 在 llama core 构图, 由 ggml 调度到 Vulkan / CPU; 专家走 bank, KV 走 memory, 三层由规划器与 store 约束。点击任一节点查看职责、关键数字与代码证据; 点上方场景按钮看分步数据流。",
          f: [],
          e: ["e.batch_context", "e.context_graph", "e.graph_memory", "e.graph_cache", "e.context_sched", "e.sched_vulkan", "e.cache_vram", "e.ram_vram", "e.memory_vram", "e.store_ram", "e.store_ssd", "e.loader_ram"]
        }
      ]
    },
    {
      id: "load", name: "加载与规划", key: 2,
      steps: [
        {
          t: "GGUF 进入加载器: reader 解析元数据与张量, 决定 mmap / lazy / resident。目标口径用 -lm none 请求 resident, 容量不够才退回 mmap。",
          f: ["ext.gguf", "core.model_loader", "core.model"],
          e: ["e.gguf_loader", "e.loader_model"]
        },
        {
          t: "按 -ncmoe 放置: dense 放 VRAM, 专家整层留在 RAM。mmap 下 bank 与预取被强制关闭, 这是容量退化路径。",
          f: ["core.model_loader", "hw.vram", "hw.ram"],
          e: ["e.loader_vram", "e.loader_ram"]
        },
        {
          t: "规划器读取模型几何与空闲显存, 用统一账本求解 dense / kv_context / compute / fa_score / kv_gather / margin / bank, 再回填参数。",
          f: ["common.fit", "common.arg", "core.model"],
          e: ["e.model_fit", "e.arg_fit"]
        },
        {
          t: "规划结果落到 bank 大小与 KV 策略, 预取环与分页也在这里定。分配顺序是 KV -> bank -> compute。",
          f: ["common.fit", "core.moe_cache", "core.memory"],
          e: ["e.fit_cache", "e.fit_memory", "e.fit_loader"]
        },
        {
          t: "common 运行时按 (layer, expert) 两级 adopt 专家并 commit, 打印 store_split。当前 2a-2c 已落地, M1 未验收, SSD 冷层未开工。",
          f: ["core.store", "common.runtime", "hw.ram", "hw.vram", "hw.ssd"],
          e: ["e.runtime_store", "e.store_ram", "e.store_vram", "e.store_ssd"]
        }
      ]
    },
    {
      id: "prefill", name: "Prefill / PP", key: 3,
      steps: [
        {
          t: "请求 -> 模板 -> 分词: server 拿到 messages, chat 模板渲染成 prompt, tokenizer 产出 token ids。",
          f: ["tool.server", "common.chat", "core.vocab"],
          e: ["e.server_chat", "e.chat_vocab"]
        },
        {
          t: "逻辑批拆成 ubatch, 分配 KV cell 索引。10K prompt 的 prefill 就按 ubatch 连续推进。",
          f: ["core.batch", "core.memory", "core.context"],
          e: ["e.vocab_batch", "e.batch_context"]
        },
        {
          t: "构图: 每层先走 GDN 线性注意力或 full attention, 再走 MoE FFN + 共享专家; 图里同时写入 KV、读取专家 ids。",
          f: ["core.context", "core.graph", "core.memory", "core.moe_cache"],
          e: ["e.context_graph", "e.graph_memory", "e.graph_cache"]
        },
        {
          t: "调度器把图切给 Vulkan / CPU; dense 与命中专家在显存, miss 专家从 host 上传 (H2D 14.3 GB/s); 大 ubatch 走整层预取环。",
          f: ["ggml.sched", "ggml.vulkan", "ggml.cpu", "hw.vram", "hw.ram"],
          e: ["e.context_sched", "e.sched_vulkan", "e.sched_cpu", "e.cache_vram", "e.ram_vram"]
        },
        {
          t: "logits -> 采样 -> token。目标口径 PP 567.4 t/s, 10K prompt 约 3.6 s。",
          f: ["core.context", "core.sampler", "common.sampling", "tool.server"],
          e: ["e.context_sampler", "e.sampler_common", "e.common_server"]
        }
      ]
    },
    {
      id: "decode", name: "Decode / TG", key: 4,
      steps: [
        {
          t: "上一步的 token 回到 batch, 这一步 n_tokens = 1; KV 只追加一个 cell。",
          f: ["core.batch", "core.context", "common.sampling"],
          e: ["e.batch_context"]
        },
        {
          t: "MoE cache 用 slot map 把本层命中的专家映射到 bank; copy callback 只上传本层缺的专家, 不整层扫描。",
          f: ["core.graph", "core.moe_cache", "hw.vram"],
          e: ["e.context_graph", "e.graph_cache", "e.cache_vram"]
        },
        {
          t: "miss 专家 RAM -> VRAM。单步 58.5 ms 里 cpy_sync 42.3 ms, 拷贝本体只有 2.7 ms; 命中率是一阶旋钮。",
          f: ["hw.ram", "hw.vram", "ggml.sched", "ggml.vulkan"],
          e: ["e.ram_vram", "e.sched_vulkan"]
        },
        {
          t: "Vulkan 计算 -> logits -> 采样 -> 下一 token。目标口径 TG 20.66 t/s; 命中率每低 10 点约多 60 MiB / 4 ms。",
          f: ["core.sampler", "common.sampling", "core.context"],
          e: ["e.context_sampler", "e.sampler_common"]
        }
      ]
    },
    {
      id: "kv", name: "长上下文 KV", key: 5,
      steps: [
        {
          t: "attention 每层写入 K/V。256K 下 KV 工作集约 320 MiB, 全分配约 5120 MiB; -kvg -1 让它按工作集增长。",
          f: ["core.graph", "core.memory", "hw.vram"],
          e: ["e.graph_memory", "e.memory_vram"]
        },
        {
          t: "GDN 线性注意力层有 recurrent state, 默认常驻设备; 它约 62.81 MiB, 与 ctx 长度无关, 是常数开销。",
          f: ["core.memory", "core.model"],
          e: []
        },
        {
          t: "可选 --kv-page: 2048-cell block + page table, prefill gather 把工作集搬进设备; 6144 档 PP -1.7%, 是容量选配, 默认 off。",
          f: ["core.memory"],
          e: []
        },
        {
          t: "-nkvo 把 KV 与 GDN 状态一起丢到 host, 用 RAM 换 VRAM; 这是容量退化路径, 不是性能路径。",
          f: ["core.memory", "hw.ram"],
          e: ["e.memory_ram"]
        }
      ]
    },
    {
      id: "server", name: "Server 并发", key: 6,
      steps: [
        {
          t: "HTTP 请求进入队列, 每个 slot 维护独立会话与采样状态。",
          f: ["ext.user", "tool.server"],
          e: ["e.user_server"]
        },
        {
          t: "update_slots() 把所有活跃 slot 拼成一个 batch, 调一次 llama_decode; 这就是连续批处理。",
          f: ["tool.server", "core.batch", "core.context"],
          e: ["e.server_batch", "e.batch_context"]
        },
        {
          t: "采样后流式返回。CLI 本地运行时就是自己的 server 子进程; WebUI 与 API 共用入口。",
          f: ["common.sampling", "tool.server", "tool.cli"],
          e: ["e.common_server", "e.cli_server", "e.server_user"]
        },
        {
          t: "MCP 客户端可连接外部工具 server; 工具调用结果再回到 chat 模板。",
          f: ["tool.ui", "tool.server"],
          e: ["e.ui_server"]
        }
      ]
    },
    {
      id: "store", name: "三层 store", key: 7,
      steps: [
        {
          t: "三个互不挪用的整机预算来自 --store-vram-mib / --store-ram-mib / --store-ssd-mib (0 = auto); common 运行时用它们初始化 store, store_split 进启动日志。",
          f: ["common.arg", "common.runtime", "core.store"],
          e: ["e.arg_runtime", "e.runtime_store"]
        },
        {
          t: "common 运行时按 (layer, expert) 两级 adopt 专家; adopt 后在 RAM 层 O(1) 索引; commit 打印 store_split。",
          f: ["common.runtime", "core.store", "hw.ram"],
          e: ["e.runtime_store", "e.store_ram"]
        },
        {
          t: "MoE cache 的整层预取经 store-io 路由到 Vulkan 五个原语; 当前 2c 已落地, M1 未验收。",
          f: ["core.moe_cache", "core.store", "ggml.vulkan"],
          e: ["e.cache_store", "e.store_vram"]
        },
        {
          t: "SSD 冷层 (F5) 未开工: 只允许 prefill 与预取, 单步同步请求计数拒绝; 前置是 expert-repack KLD = 0。",
          f: ["core.store", "hw.ssd", "tool.expert_repack"],
          e: ["e.store_ssd", "e.repack_ssd"]
        },
        {
          t: "内部三单元: slab 管句柄与三道 busy 检查, io 管 backend / file 搬运, policy 可热换; 三单元不得合并。",
          f: ["core.store"],
          e: []
        }
      ]
    },
    {
      id: "quality", name: "验证回路", key: 8,
      steps: [
        {
          t: "每轮只改一个机制, run-matrix.sh 驱动 llama-bench 跑 3 档位 x r=2 矩阵。",
          f: ["tool.matrix", "tool.bench", "ext.prompts"],
          e: ["e.matrix_bench", "e.prompts_bench"]
        },
        {
          t: "ΔPP / ΔTG / 代价条件记入 RESULTS 2.1; 非 ok 的 status 进未结项。",
          f: ["tool.matrix", "docs"],
          e: ["e.matrix_docs"]
        },
        {
          t: "动数值路径跑 quality-gate (PPL / KLD); 动 Vulkan / 调度 / GEMM 跑 op tests 与 ctest。",
          f: ["tool.quality_gate", "tool.perplexity", "tool.tests"],
          e: ["e.quality_perplexity", "e.perplexity_context", "e.quality_docs", "e.tests_cpu", "e.tests_vulkan"]
        },
        {
          t: "判决与缺口回灌规划器; 架构页只读 bench/ 四份文档, 不新增第五份 md。",
          f: ["docs", "common.fit"],
          e: ["e.docs_fit"]
        }
      ]
    }
  ]
};
