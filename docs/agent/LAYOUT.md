# agent 文档布局：按需加载与瘦身方案（2026-10-25）

**范围**：只改仓库里的文档布局与校验脚本。不改 harness 配置、不覆盖 `maxBytes`、不新增外部依赖。
**目标**：第一次会话注入的字节从 ~40 KB 降到 ~7 KB（-82%），信息一条不丢，全部转为按需读取。

---

## 0. 一句话

现在 426 行里 **45%（193 行）是 `RESULTS.md` / `DECISIONS.md` 的副本**，直接删除不损失信息；
剩下 55% 按"作用域 / 事件"重新分配：进哪个子树才需要的放子目录 `AGENTS.md`（harness 自动上链），
只在某个动作发生时才需要的放 `docs/agent/*.md`（靠路由表 + `read`）。

---

## 1. 加载机制的事实（这决定架构，不是猜的）

`@deepseek-ai/dsh-agent-instructions`（`dsh-base` 默认启用）的行为：

| 事实 | 含义 |
|---|---|
| 注入链 = `$DSH_HOME/AGENTS.md` -> 项目根（`.git`）-> ... -> **session cwd** 每一级的 `AGENTS.md`/`CLAUDE.md` | 当前 cwd 是仓库根，所以只有根文件常驻 |
| 候选名固定 `AGENTS.md` / `CLAUDE.md`；溢出覆盖 `*.local.md`；单文件上限 1 MiB | `AGENTS.local.md` 天然适合放本机私有值（不进 git） |
| **成功的文件系统操作会发现新出现的嵌套文件**并上链；cwd 变化在下一次请求前重载链 | 子目录 `AGENTS.md` 是原生懒加载钩子，但触发时机不完全确定，**必须有显式路由表兜底** |
| 字节预算 65536；超限时"**更宽的文件先被省略，最具体的文件最后才被截断**" | 调小 `maxBytes` = 拿截断当节流，等于丢弃。**本方案不这么做** |
| 已上链的文件在该会话内 durable 保留 | 放进子目录 `AGENTS.md` 的内容会一直占着，所以"作用域绑定"才放那里 |

**推论（关键）**：nested 自动加载是"**作用域**"机制，不是"**事件**"机制。
把"跑数前必读"塞进 `bench/AGENTS.md` 会导致每次碰 `bench/` 都常驻 5 KB —— 那是把省钱变成长驻开销。
事件型内容一律放 `docs/agent/`，用路由表触发一次性 `read`。

---

## 2. 分层原则

| 层 | 何时进上下文 | 放什么 | 判据 |
|---|---|---|---|
| **L0 常驻** | 每次会话，1 次 | 目标阈值、STATE 块、铁律、门禁 id、路由表 | 每个任务都要用，且一屏能容 |
| **L1 作用域** | 首次操作该子树后常驻 | 该子树专属、进去必然需要的约定 | "写这批文件时是否一直需要" |
| **L2 索引** | 需要导航时 | 全量索引 + 文档权威表 | 只被查，不被读 |
| **L3 事件** | 命中触发条件时 read | 循环、门禁细则、资源编排、构建、失败、仓库事实 | "只在某个动作发生时要" |

判据一句话：**问"写这段代码时是否一直需要"→ 是则 L1，否则 L3。**
默认走 L3，因为 L3 的成本是 0，L1 的成本是常驻。

---

## 3. 目标文件树与字节预算

```
E:\fastLlama\
├─ AGENTS.md                 L0  契约卡 + 路由表 + STATE 块（bytes/lines 上限见 doc_budget）
├─ CLAUDE.md                 L0  仅兼容指针，无实质内容
├─ AGENTS.local.md           按需存在：本机私有覆盖（.gitignore 已忽略，平时不存在）
├─ design/                   用户只读，不动
├─ runtime/AGENTS.md         L1  内核约定、L-KERN 门、check 清单、已踩过的坑
├─ bench/
│  ├─ README.md              事件：口径常量表 + 必报列 + 陷阱（不写进度）
│  ├─ RESULTS.md  DECISIONS.md           不变，仅补 slug 锚点规则
│  └─ state/state.json                   schema 3：doc_budget / lanes.reads / evidence
├─ skills/*/SKILL.md                     不变，靠 read（见 7.c）
└─ docs/agent/
   ├─ INDEX.md               L2  权威表 + 全量索引 + 每份文档的触发条件
   ├─ cycle.md               L3  每轮 8 步 + 报告格式 + slug 规矩
   ├─ gates.md               L3  六道门 + 两级质量门 + 目录归属
   ├─ resources.md           L3  资源表、lane、worktree、领用协议
   ├─ failures.md            L3  失败假设、升级 6 类、escalations 流程
   ├─ build.md               L3  构建、ctest、PATH 陷阱、机器事实
   └─ repo-facts.md          L3  仓库事实、可复用内核、已删组件
```

**预算的唯一来源是 `state.json` 的 `doc_budget`**（`agents_md_bytes` / `agents_md_lines` / `docs{}`），
`state_check.py --lint` 对每一份逐个强制，超上限报错而不是告警。本文件不复制具体数字，避免第三处漂移。

常驻成本：**根 8 KB**（现在 40 KB）。进 `runtime/` 时 +6 KB。其余按需，不命中就是 0。

---

## 4. 根 AGENTS.md 骨架（可直接替换，约 85 行 / 6 KB）

```md
# Instructions for llama.cpp (qwen36 专用运行时 fork)

除术语外中文回复；解释求简，结合目标给进度与影响。
本仓库是剪裁 + 二次优化的 llama.cpp fork，仅供参考；核心目标是重写专用运行时，不是优化上游。

## 0. 入口（每次会话按序）

1. `python bench/tools/state_check.py` 退出 0，否则停。
2. 读下面 STATE 块，取自己 lane 的队列头。
3. `python bench/tools/state_check.py --brief <task-id>` 拿任务、**必读清单**、`done_when`。
4. 按第 1 节路由表读对应文档。不要通读历史日志。

## 1. 路由表（"要做什么 -> 先读什么"，不是"参考资料"）

| 你要做 | 先读 |
|---|---|
| 写/改 `runtime/` 代码 | `runtime/AGENTS.md` |
| 跑数、报读数、填口径 | `bench/README.md` |
| 判 pass/fail、改门禁 | `docs/agent/gates.md` |
| 走完一轮、写报告 | `docs/agent/cycle.md` |
| 多 agent 编排、抢资源、开 worktree | `docs/agent/resources.md` |
| 重试、要停问用户 | `docs/agent/failures.md` |
| 构建、ctest、找不到 DLL | `docs/agent/build.md` |
| 仓库还剩什么、能复用哪些内核 | `docs/agent/repo-facts.md` |
| 两处说法冲突、找文档 | `docs/agent/INDEX.md` |

## 2. 目标（唯一，细节在 design）

模型 `F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，qwen35moe，40 主层 + 1 MTP，256 专家 top-8）。
硬件 RX 6700S 8 GiB（`-dev Vulkan1`）+ Ryzen 7 6800HS + 40172 MiB RAM，平衡电源 + AC。

| 指标 | 硬地板 | 口径 | 里程碑 |
|---|---|---|---|
| PP | >= 1000 tok/s | 10.4K prompt（10428 tok），ctx 256K，chunk 2048 | M2 |
| TG | >= 130 tok/s | 单流 greedy，10K 之后 128 tok，k=4，21.7% 缓存，q8_0 K + q4_0 V | M3 |

非目标：其他模型 / 量化 / 后端、多用户服务化、上游化。数值与开关默认值见 design 第 6/7 节。

## 3. 当前状态

<!-- STATE:BEGIN -->
phase: M2
phase_status: in_progress
decision_gate: M1 closed
queues: bench=- impl=M2-6 data=- io=- user=- 
last_gate: G-REG unknown (2026-10-23)
blockers: none
<!-- STATE:END -->

焦点：M2-6 整层链路（见 `state.json` 的 note 与 `state_check.py --board`）。
摘要不超过 3 行；历史与读数只在 `RESULTS.md` / `DECISIONS.md`。

## 4. 铁律（12 条，每条一行）

1. 并行 = 写代码并行、执行串行；同一资源同时只有一个持有者，锁状态由 `state.json` 派生。
2. 数字只认 `RESULTS.md`；跨口径不可比；同口径 r >= 3，相对极差 > 15% 该批作废。
3. 门禁不得为过门改判据、缩 prompt、跳档位、改 seed；失败留复现命令与已排除项。
4. 改数值路径合入需 L-KERN（rel_L1 <= 1e-6）；要成为默认值需 L-MODEL（KLD <= 1e-4）。
5. 一轮一提交，含读数 + 判决 + 状态；`state_check` 必须回 0 才能交接。
6. 结论只进三份 bench 文档；根文件禁止 dated bullet 与任务表（`state_lint` 会拦）。
7. 代码只用 ASCII（`-` `->` `x` `...`）；注释 1-2 行；新行为必须带 CLI 开关。
8. 同一任务第 5 次尝试前先写"失败假设 + 排查计划"；连续两次无新读数即无效尝试。
9. 待用户决策一律进 `state.json` 的 `escalations`，禁止写在散文里等用户看到。
10. 资源冲突写 `blockers` 等 bench 队首，不自行抢占；不许 force push、不许改别人的提交。
11. `GGML_ABORT` 不用作运行期回退（容量/内存/路由过散一律降级 + 计数）；SSD 不进 decode 单步。
12. 必须停问用户的 6 类见 `docs/agent/failures.md`（改目标口径 design、换模型后端、删实测数字、超时/超大、联网）。

## 5. 文档权威（冲突裁决顺序）

1 `design/qwen36-moe-runtime-design.md`（用户改，agent 只读）
2 `AGENTS.md` 3 `bench/state/state.json` 4 `bench/RESULTS.md` 5 `bench/DECISIONS.md` 6 `bench/README.md`

`RESULTS.md` 的实测优先于 design 的估计；design 的数字必须在 `RESULTS.md` 有回链。
完整表格与不可变规则见 `docs/agent/INDEX.md`。
```

对照：原 426 行 -> 约 85 行；原 40 KB -> 约 6 KB，仍有 2 KB 余量给未来。

---

## 5. 原 426 行逐条去处

| 原内容 | 规模 | 去向 | 说明 |
|---|---|---|---|
| 头部引用块 | 8 行 | 根（压到 3 行） | 方向性内容常驻 |
| §0 怎么开始 + 权威表 | 22 行 | 根 12 行；表细节 -> `INDEX.md` | 表本质是路由 |
| §1 目标 | 36 行 | 根 14 行（阈值表 + 工况 + 非目标）；读数清单 / 开关默认值 / 显存预算 -> `gates.md` 与 design | 目标常驻，参数可查 |
| §2 STATE 块 | 8 行 | 根，**位置不动** | `state_check` 解析依赖 |
| §2 现状摘要 | **193 行** | **删除** | 100% 是 RESULTS/DECISIONS 副本 |
| §3 循环 | 约 30 行 | `cycle.md`（根留 1 行） | 事件型 |
| §4 并行与资源 | 约 45 行 | `resources.md`（根留铁律 1/10） | 只在编排时需要 |
| §5 任务看板 + M1 表 | 约 43 行 | **删除**，留 `--board` 一行 | 15 行全是已 done 的死内容 |
| §6 门禁 | 约 42 行 | 根 6 行表（id / 命令 / 判据）；G-REG 六组数字 -> `gates.md` | 门禁 id 常驻，数字事件 |
| §7 报告格式 | 约 23 行 | `cycle.md`（与循环合并） | 同一件事，别再分两处 |
| §8 失败与升级 | 约 19 行 | `failures.md`（根留铁律 8/9/12） | 事件型 |
| §9 红线 | 约 21 行 | 根 8 条压缩为铁律 11 + 细节进 `gates.md` | 常驻但可瘦 |
| §10 机器与构建 | 约 26 行 | `build.md` | 事件型 |
| §11 仓库事实 | 约 13 行 | `repo-facts.md` | 一次性 |
| §12 参考 | 5 行 | `INDEX.md` | 导航 |

**净删除 236 行（§2 摘要 + §5 表），搬家约 180 行，根文件保留约 85 行。**

---

## 6. 与已评审发现的 15 个问题合并修复

搬家会把矛盾一起搬走，所以逐条在新结构里定医嘱：

| # | 问题 | 落位 | 新条文 |
|---|---|---|---|
| 1 | §2 193 行日志化，且已漂移（写 51 项，实际 60） | 根 §3 | 摘要 <= 3 行；`state_lint` 拦 `^- .*20\d\d-\d\d-\d\d` |
| 2 | §5 复制全表，与"本文不复制全表"矛盾 | 删除 | 唯一视图 `--board`；lint 拦 `^\| M\d` 形式的任务表 |
| 3 | G-REG 触发词含 `vendoring`，同节又说 `runtime/` 不走 G-REG | `gates.md` | 改成**目录归属表**：`src/ ggml/ common/ tools/` -> G-REG；`runtime/` -> L-KERN + X-4；`bench/` -> 只 lint |
| 4 | G-QUAL"不许合入"否决了当前唯一能推进的工作 | `gates.md` | 两级：**L-KERN**（内核 rel_L1 <= 1e-6 + 逐位复现）允许合入；**L-MODEL**（整模型 KLD <= 1e-4 + 逐 token）才允许成为默认值 |
| 5 | `paths_overlap` 实测漏报 `runtime/attn.*` vs `runtime/attn.cpp` | `state_lint` | 段级 `fnmatch`，通配符吃掉剩余段才算重叠；6 条用例进 selftest |
| 6 | 任务无 `owner`，"谁在跑"不可见，4 小时可被接管 | `resources.md` + schema 3 | `doing` 必填 `owner` / `worktree` / 带时区 `claimed_at`；领用走 CAS；心跳 30 min 才可接管 |
| 7 | 资源声明与 `quiet-*` 双轨，impl 跑 GPU 计时不声明 `gpu` | `resources.md` + 脚本 | 删 `quiet-*` 文件；锁由 `doing + resources` 派生，`--take/--release` 子命令；跑 GPU 计时必须声明 `gpu` |
| 8 | `escalations` 空转，而散文里挂着"请你定" | 根 §4-9 + `failures.md` | 待决必须进 `escalations`；lint 双向查（散文出现"请你定/等用户"而无对应条目即报错；条目关闭必须留判决指针） |
| 9 | `RESULTS` 节号乱序且缺 59，全靠"第 N 节"引用 | `cycle.md` | 用 **slug 锚点**：节标题下加 `<!-- slug: m2-6b-q8-gemm -->`，引用写 slug；历史标题与数字**一字不动**（不违反"删改实测数字"禁令）；lint 查悬空 slug |
| 10 | "一次提交只改一层"与"一轮一提交"互斥 | 根 §4-5 | 删前者，明确一轮一提交（读数 + 判决 + 状态） |
| 11 | `bench/README.md` 过期（还写"运行时尚未落地 M1"、"受限词表未测"） | `bench/README.md` 瘦身 | 只留口径常量 + 陷阱；lint：README 提到已 done 的任务 id 即告警 |
| 12 | `r >= 2` 无方差口径，1.18-2.11 的抖动仍被当依据 | 根 §4-2 + `gates.md` | 同口径 r >= 3，相对极差 > 15% 该批作废；报中位 + 极差，禁报单次最优 |
| 13 | "10K prompt"实际 10428 token | 根 §2 + `bench/README.md` | 根文件写 `10.4K (10428 tok)`；口径常量集中在 README 一张表 |
| 14 | §8"> 2 小时需停问"会把 G-REG 整组重跑判成需停机的事件 | `failures.md` | 白名单：G-REG 整组 / repack / 全量构建不算；其余首次命令预估 > 2 h 才停问 |
| 15 | `CLAUDE.md` 只是"去读 AGENTS.md"，而后者已自动注入；`skills/` 里 4 个 SKILL 不在 skill catalog | `CLAUDE.md` + `INDEX.md` | `CLAUDE.md` 降为一行兼容指针；`INDEX.md` 写明这些是**普通文件用 `read` 打开**，不是 `skill` 工具条目，并按触发条件列出 |

---

## 7. 防漏读：三道机械保障

按需加载唯一的真实风险是**跳过**。三件事必须同时上：

**(a) 条件式路由，不用"参考资料"措辞。**
根文件第 1 节写成"你要做 X -> 先读 Y"。条件句比"见 Y"在行为上可靠得多。

**(b) 必读清单变成任务数据，不是记忆。**
`state.json` 每个任务加 `reads: ["bench/README.md", "runtime/AGENTS.md"]`；
`state_check.py --brief <id>` 打印"本任务必读：…"，与 `done_when` 一起输出。
领用协议第 2 步改成"先按 `reads` 顺序读，再改 status"。这样"该读什么"跟着任务走，
新任务不会因为人忘了写索引而失联。

**(c) `state_lint` 校验覆盖率与字节预算。**
- 根 `AGENTS.md` <= 8192 B、行数 <= 140、无 dated bullet、无任务表
- 根与 `INDEX.md` 中出现的每个路径必须存在
- `docs/agent/*.md` 全集必须 100% 出现在 `INDEX.md`（漏一个就报错）
- 每份 L1/L3 文档 <= 各自预算，超了报错而不是警告（防第二个 40 KB）
- 报告格式新增一行 `reads: <本轮实际读的文档>`，lint 对账 `reads` 字段

---

## 8. 配套脚本改动（`state_check.py` -> `state_lint`，保留旧名别名）

| 优先级 | 改动 | 理由 |
|---|---|---|
| P0 | 修 `paths_overlap`（段级 fnmatch + selftest） | 现有实现漏报，所有权机制形同虚设 |
| P0 | AGENTS 比对从 4 个字段扩到全部 STATE 键（含 `last_gate` / `blockers`） | 现有 4 字段恰好没变，所以漂移全被放过 |
| P0 | 字节预算 / dated bullet / 任务表 / 索引覆盖率四条 lint | 让第 7 节可执行 |
| P1 | schema 3：`owner` / `worktree` / `reads` / 带时区 `claimed_at`；删 `quiet-*` 支持 | 单一锁语义 |
| P1 | `--brief <id>` 输出必读清单；`--take/--release` | 领用协议机械化 |
| P2 | slug 解析与悬空引用检查；`escalations` 双向检查 | 引用可靠性 |
| P2 | README 过期任务 id 告警 | 文档新鲜度 |

---

## 9. 迁移四步（每步独立可回退）

| 步 | 动作 | 字节变化 | 风险 |
|---|---|---|---|
| **0** | 新建 `docs/agent/` 七份文档 + `INDEX.md`，内容从现 AGENTS.md 原样搬迁（此时根文件不动） | 0 | 无，纯新增 |
| **1** | 删 §2 现状摘要（193 行）与 §5 M1 表（43 行）；`--board` 补一句 | 40 KB -> 约 12 KB | 极低：删的全是副本 |
| **2** | 根文件换成第 4 节骨架；搬走的章节从根文件移除 | 12 KB -> 约 6 KB | 低：路由表已就位 |
| **3** | 上 `state_lint`（P0 四条）+ `runtime/AGENTS.md` + schema 3 | 不变 | 中：需一次全量校验 |
| **4** | `CLAUDE.md` 降级；本机私有值进 `AGENTS.local.md`；`bench/README.md` 瘦身 | -0.1 KB | 无 |

第 1 步单独就能拿到 **-70%**，且不改变任何机制。要更快见效就先只做第 1 步。

---

## 10. 验收标准

1. `wc -c AGENTS.md` <= 8192；行数 <= 140。
2. `state_lint` 退出 0，新四条 lint 全绿。
3. 根文件里 grep 不到 `20\d\d-\d\d-\d\d` 形式的 bullet，grep 不到 `| M2-` 任务表。
4. `docs/agent/*.md` 100% 出现在 `INDEX.md`。
5. 会话启动注入字节：40,708 -> <= 8,200（-80%）；每个 teammate 同样省（成本是乘性的）。
6. 一次典型"改 runtime 内核"的会话，实际进上下文的文档总量 <= 8 份 / 20 KB。
7. `state_check.py --brief M2-6` 能打印必读清单，且清单里的文件都存在。
8. 所有 `RESULTS 第 N 节` 旧引用仍可用（slug 是新增锚点，不重排历史）。

---

## 11. 明确不做的事

- **不调 `maxBytes`**。超限语义是"先省略更宽的文件、再截断最具体的文件"，调小等于拿截断当节流，
  直接违反"不是丢弃"。预算护栏放在 lint 里，行为可预测。
- **不给 `bench/` 建 `AGENTS.md`**。nested 上链是作用域机制，塞进去会变成常驻开销；
  跑数纪律本来就是 `bench/README.md` 的职责，靠路由表触发一次 `read`。
- **不重排 `RESULTS.md` / `DECISIONS.md` 的历史节号**。只用新增 slug 注释锚点。
- **不把 `design/` 纳入这套体系**。它是用户的只读规格，索引指向章节即可。
- **不新增第四份 bench 文档**。`docs/agent/` 是契约的拆分，不是证据文档。

---

## 12. 实施记录（方案与落地的差异）

按本方案已完整落地。以下是与原文的偏离，逐条记原因：

| 项 | 方案原本 | 实际落地 | 原因 |
|---|---|---|---|
| 脚本名 | 改名 `state_lint`，保留旧名别名 | **保留 `state_check.py`**，新增 `--lint` / `--brief` / `--selftest` | 全仓 40+ 处引用这个名字；改名会让两套名字长期并存，反而制造漂移 |
| `AGENTS.local.md` | 目标树里列为一个文件 | **按需创建**，不建空壳；`.gitignore` 已忽略 | 空壳会被注入 L0 却不含信息，纯成本 |
| 接管窗口 | 心跳 30 分钟可接管 | **仍为 4 小时** | 30 分钟会把正在跑的长任务立刻判成可接管；`claim_ttl_hours` 可再调 |
| `--take/--release` | P1 实现 | **不实现**，改为手工编辑 + 校验 | 用脚本写 JSON 会重排整份 1243 行文件，diff 不可审计；协议改由 `resources.md` 第 4 节约束 |
| slug 迁移 | "节标题下加 slug" | **只对新节强制**，历史节不补 | 历史 71 + 66 节逐个起名要人工判断，且违反"不改已有条目"；lint 只查被引用 slug 的存在性与唯一性 |
| `owner` 校验 | `doing` 必填 | 缺失报 **warning**，不阻断 | 迁移期现有 `doing` 任务没有 owner；补齐后可升为 error |
| 字节预算 | 写在第 3 节树里 | **唯一来源是 `state.json` 的 `doc_budget`** | 三处写死必然漂移；现在 lint 逐文件强制，超限报错 |

### 实测结果（2026-10-25）

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
净删除：§2 现状摘要 193 行 + §5 任务表 43 行，均为 `RESULTS.md` / `DECISIONS.md` 的副本。
