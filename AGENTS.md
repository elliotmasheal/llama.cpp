# Instructions for llama.cpp (qwen36 专用 MoE 运行时)

除术语外中文回复；解释求简，结合目标给进度与影响。
本仓库是剪裁 + 二次优化的 llama.cpp fork，仅供参考：核心目标是***重写**专用运行时释放硬件性能以追求**最佳性能**，不是优化上游。
高内聚低耦合、最佳性能是两个最重要的约束，不要留技术债。
design仅供参考，有充分的自由度可以探索更优方案。

## 0. 入口（每次会话按序）

1. 运行 `python bench/tools/state_check.py`。非 0 先诊断；能在授权内修复的一致性问题自行修复并重跑，未回 0 不领任务、不执行实验。
2. 用户明确指定的工作优先；泛指"继续推进"时先续做本人 `doing`，否则取本 lane 可执行队首。STATE 是摘要，任务与依赖查 `state.json`；不得抢别人任务。
3. 队列任务运行 `python bench/tools/state_check.py --brief <task-id>`，读必读清单与 `done_when`；直接指定的文档维护不冒领运行时任务。
4. 实施前读 `docs/agent/cycle.md`、`docs/agent/failures.md`，再按路由读取。历史按任务 evidence/slug 定位；不通读日志，也不能跳过关联失败证据。

## 0.1 单次会话的执行契约

- 实施请求默认授权在范围内连续完成**研究 -> 修改 -> 运行 -> 排错 -> 回归 -> 验收**。计划、一次修改、一次实验或一次提交都不是停止点；不要用"是否继续/下一步做什么"交还可自行决定的工作。
- 一次会话可包含多轮单变量实验。优先解决当前验收阻塞；宽泛推进请求在任务完成后继续同阶段已就绪的相关任务，具体请求完成后不扩展到无关任务。
- 先证明最小真实链路正确，再扩规模/优化。独立参考、失败反例与生产入口验证必须对应；编译成功、检查器 PASS、格式正确或局部变好均不等于功能完成。
- 仅在目标已验收、用户叫停，或确无授权内可执行的安全工作时结束；越权只暂停受影响分支，资源等待先做无冲突研究/测试准备。上下文紧张就压缩并续做，不用半成品换交接。
- 结束时明确已验证什么、尚缺什么、阻塞的证据及恢复条件；不得用计划代替结果。停止与升级边界见 `docs/agent/failures.md`。

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
| 查架构审计结论与修复顺序 | `docs/agent/ARCH-REVIEW-2026-10-26.md` |
| 查设计 vs 实现的性能差距归因 | `docs/agent/PERF-GAP-2026-10-26.md` |
| 两处说法冲突、找不到该读哪篇 | `docs/agent/INDEX.md` |

## 2. 目标（唯一；模型细节与开关默认值在 design）

`F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，`qwen35moe`，40 主层 + 1 MTP，256 专家 top-8）。
RX 6700S 8 GiB（`-dev Vulkan1`）+ Ryzen 7 6800HS + 40172 MiB RAM，平衡电源 + AC 供电。

| 指标 | 硬地板 | 口径 | 里程碑 |
|---|---|---|---|
| PP | >= 1000 tok/s | 10.4K prompt（10428 tok），ctx 256K，chunk 2048 | M2 |
| TG | >= 130 tok/s | 单流 greedy，10K 之后 128 tok，k=4，21.7% 缓存，q8_0 K + q4_0 V | M3 |

每轮另报 0.85 x 绑定腿 roofline 作参照，差距分解成具体优化项。
非目标：其他模型 / 量化 / 后端、多用户服务化、上游化。跨口径数字不可比。

## 3. 当前状态

<!-- STATE:BEGIN -->
phase: M2
phase_status: in_progress
decision_gate: M1 closed
queues: bench=- impl=M3-3b data=Q-KV2 io=- user=-
last_gate: G-REG unknown (2026-10-23)
blockers: none
<!-- STATE:END -->

`queues` 只写任务 id；原因、验收条件与证据查 `state.json`。本人在途任务看 `running`，不要把已前移的队首当作续跑入口。
不在 STATE 外复制当前队首或写第二份进度；历史与读数只在 `RESULTS.md` / `DECISIONS.md`。

## 4. 铁律

1. 并行 = 写代码并行、重执行串行；锁由 `state.json` 的 `doing + resources` 派生。本人持锁可续跑，不能要求全机 `free`。
2. 数字只认 `RESULTS.md`；跨口径不可比；正式性能批次 r >= 3，相对极差 > 15% 作废，报中位与极差。诊断证据不可冒充正式跑数。
3. 门禁不许为过门改判据、缩 prompt、跳档位、改 seed；失败留复现命令与已排除项。
4. 改数值路径合入需 L-KERN（rel_L1 <= 1e-6）；要成为默认值需 L-MODEL（KLD <= 1e-4）。测试标签不得替代实测值与覆盖范围。
5. 一个可验收增量一个提交（有提交授权时），可含多轮实验；未授权不提交、不因此停工。`state_check.py` 回 0 只证明一致性，不证明代码可用。
6. 结论只进三份 bench 文档；`design/` 由用户改，agent 只读。设计中的估计、草案不是已验证实现。
7. 代码只用 ASCII（`-` `->` `x` `...`）；注释只写非显然原因。新实验/优化行为带 CLI 开关；恢复既定语义的修复不保留错误路径。
8. 首次排错就列假设、区分性实验与预期证据；两次无新增证据必须换定位方法，不机械重跑，不把重试次数当退出配额。
9. 待用户决策进 `state.json` 的 `escalations`，同时明确告知用户；不把普通实现选择升级成用户决策。
10. 资源冲突写 `blockers`，优先无冲突工作；不抢占、不凭超时夺锁，不许 force push，不许改别人的提交。
11. 不用 `GGML_ABORT` 处理运行期条件；SSD 不进 decode 单步；不许第二套调度器 / 缓存 / 预算器。
12. 越权动作先停问，见 `docs/agent/failures.md` 六类；仅暂停受影响分支，不把局部受阻当整场会话结束。

## 5. 文档权威（冲突裁决顺序）

1 `design/qwen36-moe-runtime-design.md`（用户改，agent 只读） 2 `AGENTS.md` 3 `bench/state/state.json`
4 `bench/RESULTS.md` 5 `bench/DECISIONS.md` 6 `bench/README.md`

`RESULTS.md` 的实测优先于 design 的估计；design 的数字必须在 `RESULTS.md` 有回链。
完整表与"谁能改"在 `docs/agent/INDEX.md`；分层理由（L0 常驻 / L1 作用域 / L3 按需）与迁移记录在 `docs/agent/LAYOUT.md`。
