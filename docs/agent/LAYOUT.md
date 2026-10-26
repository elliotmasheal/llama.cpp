# agent 文档布局与维护

本文件解释布局，不是第二份执行入口。当前规则以 `AGENTS.md`、`docs/agent/INDEX.md` 及其路由为准；不保存可直接覆盖入口的历史模板。

## 1. 设计取舍

- 根 `AGENTS.md` 常驻，只放自主执行契约、硬目标、关键边界、STATE 和路由。不复制任务看板或实验日志。
- 子目录 `AGENTS.md` 只放该子树始终需要的约定；当前 runtime 的重点是语义契约、独立参考和生产链验收。
- `docs/agent/` 按动作加载；`cycle.md`/`failures.md` 在实施前读取，其余由任务触发。`INDEX.md` 只导航，不重写规则。
- 必读清单是研究下限，不是阅读上限。少读无关文档，完整读相关实现、调用者、参考与失败证据；不要用小上下文预算为浅研究辩护。
- 一处维护一种事实：目标在 design，任务在 state，实测在 RESULTS，判决在 DECISIONS；其它文档引用，不建第四份进度/证据账本。

## 2. 加载与预算

历史布局针对 `@deepseek-ai/dsh-agent-instructions` 的根到 cwd 注入链设计；其它 harness 的自动加载时机不保证相同，所以始终以根路由显式读取兜底，不假设子树规则已经注入。

| 层 | 用途 | 内容 |
|---|---|---|
| L0 常驻 | 每次会话 | 根 `AGENTS.md`，`CLAUDE.md` 仅指针 |
| L1 作用域 | 进入子树 | `runtime/AGENTS.md` |
| L2 索引 | 查找时 | `docs/agent/INDEX.md` |
| L3 事件 | 命中动作时 | cycle/gates/resources/failures/build/repo-facts、bench 口径、skills |

预算唯一来源为 `bench/state/state.json` 的 `doc_budget`；不调 harness 的截断上限，也不靠新增文件分散重复规则。`AGENTS.local.md` 只用于确需私有的机器值，不建空文件。

`skills/*/SKILL.md` 是普通仓库文件，通过 read 读取，不假设存在同名可调用工具。`skills/code-review/SKILL.md` 面向本项目，不再把上游投稿手续当实现准入门槛。

## 3. 校验能力的真实边界

`python bench/tools/state_check.py` 校验状态和文档；`--lint` 只校验文档；`--brief` 打印任务，不执行完整校验；`--selftest` 跑隔离回归。

| 机械检查 | 边界 |
|---|---|
| 根字节/行数、dated bullet、M 数字任务表；预算表内文件字节 | 不度量研究深度或摘要质量 |
| INDEX 名字覆盖及部分路径存在性 | 不证明实际阅读；根路径/悬空 slug 有些只告警 |
| 重复 slug、部分引用检查 | 不要求历史节补 slug，不验证产物内容 |
| 状态字段、队首、doing/done 前置依赖、资源/路径冲突、STATE 六项镜像 | 不证明 done_when 真值，不检测所有依赖环 |
| 部分 owner/时间/escalation 字段 | 不提供 CAS、自动心跳、进程存活或全局跨 worktree 锁 |

尚未实现：实际 reads 对账、散文与 escalation 双向对账、README 过时任务自动检测。由主 agent 人工核实；禁止把计划中的功能写成现有防线。规则修改必须连同索引/相关入口检查，脚本修改必须先有能失败的回归。

## 4. 维护规则

- 优先替换冲突条文，删除重复模板，不在旧规则旁堆叠例外。自主推进集中在根入口，实验闭环在 cycle，越界判断在 failures。
- 研究/验证要求写成可观察动作：看哪条真实链、哪种独立证据、什么反例、未验证时是什么状态；不要只写"深入研究"或"保证质量"。
- 校验退出 0 仅是文档/状态一致；交付仍逐项核对 done_when、生产路径与数值门禁。
- 新增文档才扩展路由和预算；不能靠提高预算解决重复内容。文档维护不改性能目标、质量阈值、已有实测或 design。

## 5. 历史迁移记录（非当前字节与通过证明）

首次迁移将根入口的日志/任务表移回 bench 与 state，保留脚本名 `state_check.py`；没有实现原提案中的 `state_lint` 别名、`--take/--release` 或 CAS。4 小时是校验窗口，不是自动接管授权。

以下保留 2026-10-25 的布局读数，不用于宣称当前预算或测试已通过。

| 文件 | 字节 | 行数 | 预算 |
|---|---:|---:|---:|
| `AGENTS.md` | 4315 | 59 | 8192 B / 140 行 |
| `CLAUDE.md` | 130 | 1 | - |
| `runtime/AGENTS.md` | 3345 | 35 | 4096 B |
| `docs/agent/INDEX.md` | 3698 | 47 | 4608 B |
| `docs/agent/cycle.md` | 3375 | 44 | 4096 B |
| `docs/agent/gates.md` | 3309 | 36 | 4096 B |
| `docs/agent/resources.md` | 4059 | 42 | 5120 B |
| `docs/agent/failures.md` | 1999 | 25 | 2560 B |
| `docs/agent/build.md` | 1403 | 16 | 2048 B |
| `docs/agent/repo-facts.md` | 1388 | 18 | 2048 B |
| `bench/README.md` | 5510 | 59 | 6144 B |

常驻成本 40,857 B -> **4,445 B（-89.1%）**；进入 `runtime/` 时再加 3,345 B。
`state_check.py --selftest` 17/17；`state_check.py` 退出 0（含四条文档 lint）。
净删除：根现状摘要 193 行 + 任务表 43 行，均为 `RESULTS.md` / `DECISIONS.md` 的副本。
