# agent 文档索引（L2：只被查，不被通读）

## 1. 冲突裁决顺序

| 优先级 | 文件 | 只放什么 | 谁能改 |
|---:|---|---|---|
| 1 | `design/qwen36-moe-runtime-design.md` | 目标、非目标、硬约束、性能模型、里程碑与验收阈值 | 用户；agent 只读 |
| 2 | `AGENTS.md` | 契约卡：目标阈值、铁律、门禁 id、路由表、STATE 块 | agent 每轮更新 STATE 块 |
| 3 | `bench/state/state.json` | 任务图、依赖、资源、队列头、必读清单、当前状态 | agent 领任务/交接时改 |
| 4 | `bench/RESULTS.md` | 实测数字的唯一来源（含口径头与 slug） | 有新读数时追加 |
| 5 | `bench/DECISIONS.md` | 判决、已证伪、开关表、未结项 | 有判决时追加 |
| 6 | `bench/README.md` | 口径常量、必报列、测量陷阱 | 口径变化时才改 |

规则：

- `RESULTS.md` 的实测优先于 design 的估计；design 里的数字必须在 `RESULTS.md` 有回链。
- design 与本套文档冲突时停下，走 `docs/agent/failures.md` 的升级流程。
- 跨口径的数字不可比。

## 2. 全量索引

根 `AGENTS.md` 的路由表是本表的子集；本表必须覆盖全集（`state_check.py` 校验覆盖率）。

| 文档 | 何时读 | 预算 | 层 |
|---|---|---:|---|
| `docs/agent/INDEX.md` | 找文档、查权威表、判"该读哪篇" | 4608 B | L2 |
| `runtime/AGENTS.md` | 写或改 `runtime/` 下任何代码之前（进入该子树后自动上链，仍要确认） | 4096 B | L1 |
| `docs/agent/cycle.md` | 走一轮、写每轮报告、加节锚点 | 4096 B | L3 |
| `docs/agent/gates.md` | 判 pass/fail、动门禁、查 G-REG 固化数字、定质量门 | 4096 B | L3 |
| `docs/agent/resources.md` | 多 agent 编排、抢资源、开 worktree、领用任务 | 5120 B | L3 |
| `docs/agent/failures.md` | 重试、要停问用户、写 escalations | 2560 B | L3 |
| `docs/agent/build.md` | 构建、ctest、找不到 DLL | 2048 B | L3 |
| `docs/agent/repo-facts.md` | 查仓库还剩什么、能复用哪些内核或格式 | 2048 B | L3 |
| `docs/agent/LAYOUT.md` | 维护这套文档体系本身（分层理由与迁移记录） | 不限 | L3 |
| `bench/README.md` | 每次跑数之前 | 6144 B | 事件 |
| `skills/*/SKILL.md` | 命中各自触发条件（见第 4 节） | 不限 | L3 |

字节预算由 `bench/state/state.json` 的 `doc_budget` 定义，`state_check.py --lint` 强制：超上限直接报错，不告警。
新增文档要同时改 `state.json`、本表与根 `AGENTS.md` 路由表。

## 3. 分层约定

- **L0 常驻**：根 `AGENTS.md`。每次会话注入一次，预算 8192 B / 140 行。
- **L1 作用域**：子目录 `AGENTS.md`。首次操作该子树后由 harness 自动上链并常驻。只放"写这批文件时一直需要"的约定。
- **L2 索引**：本文件。只被查。
- **L3 事件**：其余 md。命中触发条件才 `read`，不命中成本为零。

判据一句话：问"写这段代码时是否一直需要" -> 是则 L1，否则 L3。默认走 L3。

**本机私有覆盖**：需要放不进 git 的本机值（另一台机器的模型路径、专门的构建目录）时，建 `AGENTS.local.md`；它在候选清单里、是 additive overlay，且已被 `.gitignore` 忽略。平时不存在。

## 4. skills 目录

`skills/` 下的 `SKILL.md` 是**普通文件**，不在本会话 `skill` 工具的目录里（该目录只含 harness 内置项）。要读就 `read`，不要指望用 `skill` 工具加载。

| 文件 | 触发条件 |
|---|---|
| `skills/code-review/SKILL.md` | 写代码或改内核之前 |
| `skills/add-new-model/SKILL.md` | 新增模型映射时（本项目非目标，仅参考） |
| `skills/slice-model/SKILL.md` | 切片/裁剪模型时 |
| `skills/update-fusion-table/SKILL.md` | 改 Vulkan 融合表时 |

## 5. 外部参考

- 构建：`docs/build.md`
- 解析 / 模板：`docs/development/parsing.md`、`common/jinja/README.md`
- 新增模型：`docs/development/HOWTO-add-model.md`
