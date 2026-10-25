# Instructions for llama.cpp (qwen36 专用 MoE 运行时)

除术语外中文回复；解释求简，结合目标给进度与影响。
本仓库是剪裁 + 二次优化的 llama.cpp fork，仅供参考：核心目标是**重写**专用运行时，不是优化上游。

## 0. 入口（每次会话按序）

1. `python bench/tools/state_check.py` 退出 0，否则停。
2. 读下面 STATE 块，取自己 lane 的队列头。
3. `python bench/tools/state_check.py --brief <task-id>`：拿任务、**必读清单**、`done_when`。
4. 按第 1 节路由表读对应文档。不要通读历史日志。

## 1. 路由表（"要做什么 -> 先读什么"）

| 你要做 | 先读 |
|---|---|
| 写或改 `runtime/` 代码 | `runtime/AGENTS.md` |
| 跑数、报读数、填口径 | `bench/README.md` |
| 判 pass/fail、动门禁 | `docs/agent/gates.md` |
| 走完一轮、写报告、加节锚点 | `docs/agent/cycle.md` |
| 编排多 agent、抢资源、开 worktree | `docs/agent/resources.md` |
| 重试、要停问用户 | `docs/agent/failures.md` |
| 构建、ctest、找不到 DLL | `docs/agent/build.md` |
| 仓库还剩什么、能复用哪些内核 | `docs/agent/repo-facts.md` |
| 两处说法冲突、找不到该读哪篇 | `docs/agent/INDEX.md` |

## 2. 目标（唯一；模型细节与开关默认值在 design）

`F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，`qwen35moe`，40 主层 + 1 MTP，256 专家 top-8）。
RX 6700S 8 GiB（`-dev Vulkan1`）+ Ryzen 7 6800HS + 40172 MiB RAM，平衡电源 + AC 供电。

| 指标 | 硬地板 | 口径 | 里程碑 |
|---|---|---|---|
| PP | >= 1000 tok/s | 10.4K prompt（10428 tok），ctx 256K，chunk 2048 | M2 |
| TG | >= 130 tok/s | 单流 greedy，10K 之后 128 tok，k=4，21.9% 缓存，q4 KV | M3 |

每轮另报 0.85 x 绑定腿 roofline 作参照，差距分解成具体优化项。
非目标：其他模型 / 量化 / 后端、多用户服务化、上游化。跨口径数字不可比。

## 3. 当前状态

<!-- STATE:BEGIN -->
phase: M2
phase_status: in_progress
decision_gate: M1 closed
queues: bench=- impl=M2-6a data=Q-KV2 io=- user=-
last_gate: G-REG unknown (2026-10-23)
blockers: none
<!-- STATE:END -->

`queues` 只写任务 id；"为什么是它、做到什么算完"在 `state.json`。
本节摘要最多 3 行：历史与读数只在 `RESULTS.md` / `DECISIONS.md`，`--lint` 会拦 dated bullet 与任务表。
焦点：M2-6a 的 q4_0 KV 与常驻窗口分页；层链（含 FFN、两次合并提交）已 closed，见 `RESULTS` `m2-6-ffn-two-submits`。

## 4. 铁律

1. 并行 = 写代码并行、执行串行；同一资源同时只有一个持有者，锁由 `state.json` 的 `doing + resources` 派生。
2. 数字只认 `RESULTS.md`；跨口径不可比；同口径 r >= 3，相对极差 > 15% 该批作废，报中位与极差。
3. 门禁不许为过门改判据、缩 prompt、跳档位、改 seed；失败留复现命令与已排除项。
4. 改数值路径合入需 L-KERN（rel_L1 <= 1e-6）；要成为默认值需 L-MODEL（KLD <= 1e-4）。
5. 一轮一个提交，含读数 + 判决 + 状态；`state_check.py` 必须回 0 才能交接。
6. 结论只进三份 bench 文档；`design/` 由用户改，agent 只读。
7. 代码只用 ASCII（`-` `->` `x` `...`）；注释 1-2 行；新行为必须带 CLI 开关。
8. 同一任务第 5 次尝试前先写"失败假设 + 排查思路"；连续两次没有新读数即视为无效尝试。
9. 待用户决策一律进 `state.json` 的 `escalations`，不写在散文里等用户看到。
10. 资源冲突写 `blockers` 等 bench 队首；不许 force push，不许改别人的提交。
11. 不用 `GGML_ABORT` 处理运行期条件；SSD 不进 decode 单步；不许第二套调度器 / 缓存 / 预算器。
12. 必须停问用户的六类见 `docs/agent/failures.md`（改目标口径 design、换模型后端、删实测数字、超时超大、联网）。

## 5. 文档权威（冲突裁决顺序）

1 `design/qwen36-moe-runtime-design.md`（用户改，agent 只读） 2 `AGENTS.md` 3 `bench/state/state.json`
4 `bench/RESULTS.md` 5 `bench/DECISIONS.md` 6 `bench/README.md`

`RESULTS.md` 的实测优先于 design 的估计；design 的数字必须在 `RESULTS.md` 有回链。
完整表与"谁能改"在 `docs/agent/INDEX.md`；分层理由（L0 常驻 / L1 作用域 / L3 按需）与迁移记录在 `docs/agent/LAYOUT.md`。
