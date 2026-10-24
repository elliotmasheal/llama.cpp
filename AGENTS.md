# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> 本仓库是 `elliotmasheal/llama.cpp` fork，基线是 ggml-org/llama.cpp 的上游 master。
> `origin` 指向本 fork，未配置 `upstream` remote。**本项目不考虑"能否上游化"，追求小 VRAM 设备下的指定模型的极致性能。**
> 只有一条主线：`design/qwen36-moe-runtime-design.md`（qwen36 专用 MoE 运行时，里程碑 M0-M5）。

## 0. 怎么开始

本文件是 agent 的唯一入口与执行契约。性能规格与数学模型在 `design/qwen36-moe-runtime-design.md`（下称 design）。
任务图与资源占用在 `bench/state/state.json`，与第 2 节的 STATE 块必须一致，由
`python bench/tools/state_check.py` 校验（退出码 0 才允许开跑）。

**启动顺序**：`state_check.py` -> 本文件第 2 节 -> 自己 lane 的队列头 -> design 里与该任务对应的章节。
看全量任务用 `state_check.py --board`，只看一条 lane 用 `--lane impl`。不要通读历史日志。

**文档权威（冲突时按此顺序裁决）**

| 优先级 | 文件 | 只放什么 | 谁能改 |
|---:|---|---|---|
| 1 | `design/qwen36-moe-runtime-design.md` | 目标、非目标、硬约束、性能模型、里程碑与验收阈值 | 用户；agent 只读 |
| 2 | `AGENTS.md` | 执行契约、并行规则、门禁、循环 | agent 每轮更新状态块 |
| 3 | `bench/state/state.json` | 任务图、依赖、资源、队列头、当前状态 | agent 领任务/交接时改 |
| 4 | `bench/RESULTS.md` | 实测数字的唯一来源（含口径头） | 有新读数时追加 |
| 5 | `bench/DECISIONS.md` | 判决、已证伪、开关表、未结项 | 有判决时追加 |
| 6 | `bench/README.md` | 口径、必报列、测量陷阱 | 口径变化时才改 |

规则：`RESULTS.md` 的实测优先于 design 的估计；design 里的数字必须在 `RESULTS.md` 有回链。
design 与本文件冲突时停下，按第 8 节升级。跨口径的数字不可比。
一次提交只改一层：先数字来源，再判决，再状态。

## 1. 目标（唯一）

模型 `F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，arch `qwen35moe`，40 主层 + 1 MTP 层，
256 专家 top-8）。硬件：RX 6700S 8 GiB（`-dev Vulkan1`）+ Ryzen 7 6800HS + 40172 MiB RAM，
笔记本「平衡」电源 + AC 供电。

统一工况：10K prompt，ctx 256K，greedy，单流。

| 指标 | 目标 | 口径 | 判定里程碑 |
|---|---|---|---|
| PP | >= 1000 tok/s | 10K prompt，ctx 256K，batch 512 | M2 |
| TG | >= 130 tok/s | 单流，greedy，10K prompt 之后 128 token | M3 |

**硬约束（M0 实测，运行时绕不过）**

- 稠密权重每 token 必读 2.17 GB；显存带宽地板 217 GB/s -> 单步前向上限 **100 tok/s**。
  突破唯一出路是一次前向处理多个 token（MTP 草稿 + k=8 批量验证）。
- H2D 14.3 GB/s 是硬墙；CPU 4 线程读 40 GB/s；SSD 随机读 0.81 GB/s。
- 不改变量化格式；不改变输出：k=1 路径 logits KLD <= 1e-4，推测路径接受的 token 与参考 greedy 一致。
- 非目标：其他模型 / 量化 / 后端、多用户服务化（design 1.2）。

## 2. 当前状态

<!-- STATE:BEGIN -->
phase: M1
phase_status: in_progress
decision_gate: M1 open
queues: bench=M1-1 impl=M1-4a data=M1-6 io=- user=U-1
last_gate: G-REG unknown (2026-10-23 record, not re-run since 44bd79305)
blockers: none
<!-- STATE:END -->

`queues` 是每条 lane 的下一个可跑任务，`-` 表示该 lane 暂停。只改这一块与 `state.json`，两者必须同一次提交。
队列头只写"任务 id"，"为什么是它、做到什么算完"在 `state.json` 里，避免两处叙述漂移。

**现状摘要（人读，不参与机器校验）**

- M0 完成：张量账、10K + 128 token 路由 trace、局部性与命中率模拟（`RESULTS.md` 第 1-2 节）。
- M1 进行中：bench 队首是 trace 缺口，data 队首是 KV 预算，impl 队首是 Vulkan 计时原型。
- M1 是**决策门**：受限词表 E 或周期时间不达标就停下问用户是否下调 TG 目标（design 第 9 节）。
- 已定：完整词表草稿是基线（理论约 125，低于 130）；受限词表是达标手段，其 E 未测。
- U-1（运行时归属）必须在 M2 开工前由用户回答；M1 不受它阻塞。

## 3. 每轮自主循环

一轮 = 一个单变量改动（或一次测量）+ 一次证据采集 + 三份 bench 文档的必要更新 + 状态同步 + 一个提交。

0. **preflight**（任一不过就停，不跑数）
   - `python bench/tools/state_check.py` 退出码 0
   - `git status --short` 只剩本轮预期改动；不带着上一轮的半成品开跑
   - 自己 lane 的队列头没有被别人领走（`state.json` 里该任务不是 doing）
   - 没有别人的 quiet 锁（`state_check.py` 的 `quiet` 行是 `free`，或是本轮自己的）
   - 机器条件：平衡电源 + AC、dGPU 上无其它推理进程、空闲 RAM 满足当前臂
1. **领任务**：取自己 lane 的队首，按第 4 节第 5 条的领用协议改状态；不许跨 lane 抢任务。
2. **读规格**：design 对应章节 + `README.md` 口径 + `DECISIONS.md` 已证伪清单。
3. **单变量改动**：改动必须可回退、必须有开关、必须能用一个命令复现。
4. **采证**：用 `bench/tools/` 的脚本，不写一次性命令进报告；原始产物落 `bench/raw/`（不进 git）。
5. **判定**：按第 6 节逐条给 `pass / fail / n-a`。禁止"看起来更快"、"趋势一致"这类结论。
6. **落文档**：`RESULTS.md` 加读数（带口径头）、`DECISIONS.md` 加判决/开关/未结项、
   本文件状态块 + `state.json` 同步。
7. **提交**：小而可回退。提交信息写"改了什么 + 跑了哪些门禁 + 读数出处"。
8. **交接**：把任务置 `done`，队列头前移，`state_check.py` 必须回到 0；下一个人只读状态与证据文件。

## 4. 并行推进与资源独占

本机只有一张 GPU、40 GiB 内存、一块盘。**"并行"指写代码与写文档并行，执行（跑数、编译、repack）串行。**
任何时刻，同一资源最多一个持有者；冲突由 `state_check.py` 拦截，不靠口头约定。

**资源（互斥量，容量都是 1）**

| 资源 | 含义 | 典型任务 |
|---|---|---|
| `gpu` | dGPU 与显存，一次只能一个进程 | 任何跑模型或内核的测量 |
| `host-ram` | 驻留 21.28 GiB 模型（约 28 GiB 空闲内存） | bench 跑 llama-cli / 运行时 |
| `machine-quiet` | 计时、`-j8` 全量构建、内存/磁盘探针、repack | bench/io 整段持有；impl 构建时用 quiet 锁瞬时占用 |
| `disk-io` | 超过 1 GiB 读写或 SSD 带宽测量 | repack 与校验 |
| `trace-bin` | 重新生成路由 trace 文件 | M1-1、M1-2b |
| `prompt-file` | 修改 `bench/prompts/` 下的文件（只读共享） | M1-2a |

**lane（固定编组，任务在 `state.json` 里指定）**

| lane | 默认占用 | 并发度 | 说明 |
|---|---|---|---|
| `bench` | gpu + host-ram + machine-quiet | 1 | 只做测量；全项目只有这一条执行队，串行消费 |
| `impl` | quiet 锁（构建/微基准时） | 2-3 | 写代码；路径不重叠时真正并行，编译要等 bench 空闲 |
| `data` | 无 | 任意 | 语料、词表、GGUF 元数据、汇总、文档；不阻塞任何人 |
| `io` | machine-quiet + disk-io | 1 | 重组与逐字节校验，和 bench、构建互斥 |
| `user` | 无 | 1 | 只有用户能做的决定；agent 准备选项后停 |

**跑数纪律**：bench 任务运行时不许编译、不许 repack、不许跑第二个模型。
impl agent 在 bench 运行期间只写代码与测试；要编译或跑微基准时先取 quiet 锁：

```
touch bench/state/quiet-<task>     # 取锁，跑完立刻删
rm bench/state/quiet-<task>
```

`state_check.py` 会在"quiet 锁与 bench/io 任务同时存在"或"锁超过 2 小时"时报错。
bench/io 任务的 `doing` 状态本身就是整台机器的独占。

**工作树与构建隔离**

- 每个 impl agent 一个 worktree：`git worktree add ../fastLlama-<id> -b <id>/<task>`。
- 每个 worktree 一个构建目录 `build-<id>`；**不许共用 `build-vulkan`**，也不许两个 agent 同时构建。
- 共享只读：模型文件、`bench/prompts/`、已有 `bench/raw/` 证据、`design/`。

**路径所有权与合入**

- `task.paths` 是该任务的写权限，glob 前缀重叠即冲突；改别人的路径前先改 `state.json` 的归属并记录原因。
- 合入顺序按 `deps`；一个任务一个提交；冲突由后合入者解决，不许 force push，不许改别人的提交。
- 发现任务拆分不对：先改 `state.json`（拆任务、改 deps/paths），跑通 `state_check.py`，再动手。

**领用协议（每个任务照做）**

1. `state_check.py` 退出 0，且该任务 `status` 是 `todo` / `partial`。
2. 把它改成 `doing`；同一 `resources` 或重叠 `paths` 的另一个 doing 必须已经结束。
3. 把本 lane 的队列头前移到下一个可跑任务（没有就写 `-`）。
4. 再跑一次 `state_check.py`，退出 0 才开始。
5. 完成时 `status = done`，读数进 `RESULTS.md`，判决进 `DECISIONS.md`，队列头前移，再校验、再提交。

**排班建议**：1 个 bench agent + 2-3 个 impl agent + 1 个 data agent。data 先产出 bench 需要的输入
（非技术语料、受限词表、KV 预算），bench 才不会空转；impl 在决策门未过时只做工具与原型。

## 5. 任务与看板

任务图的唯一来源是 `bench/state/state.json`（39 项，M1-M5 + U/X），本文不复制全表。
查看方式：

```
python bench/tools/state_check.py --board         # 全部，按 lane 分组
python bench/tools/state_check.py --lane bench    # 只看一条 lane
```

**M1（决策门，进行中）**

| id | lane | 内容 | 验收要点 |
|---|---|---|---|
| M1-1 | bench | 补齐 10 个 full-attention 层与 layer 39 的解码路由 | 40 层 x 128 token 都有记录；根因进 DECISIONS |
| M1-2a | data | 非技术语料：对话 / 代码 / 多语言各 10K | token 实测并写进 RESULTS |
| M1-2b | bench | 三类文本局部性复测（技术类已完成） | 与 M0 偏差 <= 5 pp 或已解释 |
| M1-3a | data | 受限草稿词表（16K）与生成脚本 | 覆盖验证集 >= 95% |
| M1-3b | bench | 受限词表 MTP 接受率 E | E >= 2.1 / 2.3，tokens 与 greedy 一致 |
| M1-4a | impl | Vulkan 计时原型（k=1/2/4/8） | 每 pass 的 submit/sync 时间可测 |
| M1-4b | bench | 周期时间测量 | 与 design 2.6 偏差 <= 10%，开销 < 1 ms |
| M1-5 | bench | 带宽同口径复测 | 偏差 <= 5% 或更新 RESULTS |
| M1-6 | data | full-attention 层数与 256K KV 预算 | 层数、每 token 字节、显存占比 |
| M1-GATE | user | 决策门评审 | 三条全过才进 M2，否则走 U-2 |

**决策门 M1 的通过条件**（design 第 9 节）：M1-1/3b/4b 全过，M1-2b 偏差 <= 5 pp，M1-5 与 M1-6 已回填。
任一项不过：`phase_status=blocked`，问用户是否下调 TG 目标，不许自行改阈值。

**门后**：M2 prefill（M2-0..8）、M3 decode（M3-0..6）、M4 调优（M4-1..5）、M5 重组（M5-1..4）、
横切工具（X-1 质量门、X-2 汇总器）。它们已登记 deps 与路径，未过门前只允许写工具与微基准。

## 6. 门禁

| id | 触发条件 | 命令 | 通过标准 | 不通过怎么办 |
|---|---|---|---|---|
| **G-CONS** | 每轮 | `python bench/tools/state_check.py`；`git diff` 自查 ASCII 与注释规范 | 退出码 0，diff 只含本轮改动 | 不提交 |
| **G-EVID** | 每轮有读数 | `bench/tools/` 里的统一脚本 | 必报列齐全，`status = ok`，r >= 2，口径头齐全 | 该批作废，重建口径后重跑 |
| **G-REG** | 改 fork 或复用其中的 ggml 内核 | 见下方六条 | 全部 100% 通过 | 回退改动，记未结项 |
| **G-QUAL** | 改数值路径、精度、注意力实现 | 逐 token 对比参考实现；k=1 路径另测 logits KLD | token 全同且 KLD <= 1e-4（工具由 X-1 交付） | 不许合入 |
| **G-DEC** | 里程碑 | design 第 8/9 节 | 阈值全过且证据可复核 | 停，问用户 |

**G-REG 的固化数字**（2026-10-23 记录，`build-vulkan/bin/` 下；动 fork 代码前必须重跑）：

- `test-backend-ops -o MUL_MAT_ID -b CPU` = 945/945
- `test-backend-ops -o MUL_MAT_ID_MASK -b CPU` = 4/4；`-b Vulkan1` = 0/0（Vulkan 侧没有这个 op）
- `test-backend-ops -o MUL_MAT_ID -b Vulkan1` = 939/939
- `test-backend-ops -o MUL_MAT -b Vulkan1` = 2275/2275
- `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` = 5332/5332
- `ctest --test-dir build-vulkan` = 32/32

**门禁纪律**

- 门禁数字由 design 或本节规定，agent 不得为了过门而改判据、缩 prompt、跳档位、改 seed。
- 门禁失败必须留复现命令与已排除项，"配置不对"不是结论。

## 7. 每轮报告格式

固定字段，缺一行视为本轮未完成：

```
round   : <date> <commit>
lane    : <bench|impl|data|io|user> [resources]
task    : <task id>
change  : <one line, single variable>
口径    : model | ctx | prompt tokens | n_gen | k | draft vocab | cache MiB | VRAM tier | load | power
证据    : <command or raw path> | r=<n>
结果    : <必报列>
门禁    : G-CONS pass | G-EVID pass/fail/n-a | G-REG pass/fail/n-a | G-QUAL pass/fail/n-a
判决    : adopt | revert | partial(<why>)
文档    : RESULTS <section> | DECISIONS <item> | queue=<lane>:<next id>
阻塞    : none | <what blocks, unblock condition>
```

## 8. 失败处理与升级

- 同一任务第 3 次尝试前，必须先写"失败假设 + 排查计划"进 `DECISIONS.md` 未结项，再动手。
- 失败留三样：复现命令、已排除项、下一步判据。
- **必须停下来问用户的 8 类**：改目标或阈值；改 design；换模型 / 量化 / 后端；删改已有实测数字；
  改口径；新增 design 第 5.1 节未登记的模块或第二套调度器；预计 > 2 小时或 > 20 GiB 磁盘；需要联网下载。
- 不得用"跳过某个臂""只跑一次""把两档平均"之类的手段让门禁变绿。
- 资源冲突时不许自行抢占：把冲突写进 `state.json` 的 `blockers`，由 bench 队首优先。

## 9. 工程红线

1. **不得用 `GGML_ABORT` 处理运行期条件**（容量不足、内存不足、路由过散）。必须回退到慢但正确的路径并计数告警。
2. **SSD 只做容量层，绝不进 decode 单步关键路径**：随机 1.95 MiB = 0.81 GB/s，206 MiB/token 要 258 ms。
3. **新行为一律带开关**：CLI 参数 + `DECISIONS.md` 开关表；默认值由目标工况实测决定；
   只靠 env 而不上 CLI 的行为不得成为默认路径。
4. **新模块必须先登记**：design 第 5.1 节列出允许的模块。不允许出现未登记的第二套调度器、缓存或预算器。
5. **数字必须带口径**：日期、提交、电源态、显存档位、k、草稿词表。跨口径比较视为错误。
6. **结论只写进三份 bench 文档**，design 由用户改（agent 只读）；原始产物写 `bench/raw/`（不进 git）。
7. **代码只用 ASCII**：不要 `—`、`→`、`×`、`…`，改用 `-`、`->`、`x`、`...`。
8. **注释 1-2 行**，只写不显然的约定；不复述代码；不按固定列宽硬换行；先写代码后写注释。

## 10. 机器事实与构建

- RX 6700S 8 GiB（`-dev Vulkan1`），设备总量 8176 MiB，桌面常占 0.5-1.4 GiB。
- Ryzen 7 6800HS，RAM 40172 MiB（`bench/env/machine.txt`），电源「平衡」（`bench/env/power.txt`，AC）。
- resident 门槛约 `21.28 x 1.25 + 1 = 27.6 GiB` 空闲内存；不足时 mmap，性能不可比。
- 构建（MinGW + Ninja，不是 MSVC preset；目录按第 4 节每 agent 一份）：

  ```
  cmake -B build-<id> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe \
    -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
  cmake --build build-<id> -j 8
  ```

- Git Bash 里跑 `build-*/bin/*.exe` 前要 `export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"`，
  否则找不到 libstdc++/libgcc DLL，直接 exit 127。
- 性能一律 Release/RelWithDebInfo。测试：`ctest --test-dir build-<id> --output-on-failure -C Release`。

## 11. 仓库事实

- backend 只有 **CPU 与 Vulkan**，两者之外的东西已物理删除，不要去找。
- 模型只支持 **qwen35moe**（目标模型就是它），`llama_model_mapping()` 只有一个 case。
- 仍在的组件：`src/` 的模型加载 / 图 / KV / 采样 / C API，`ggml/src/ggml-{cpu,vulkan}` 与调度器，
  `common/` 的 CLI 与 `fit.cpp`，`tools/` 的 cli / server / llama-bench / mtmd / perplexity。
- 可复用的内核与格式：Vulkan `mul_mat_vec` / `mul_mat_id`（要改成批量 k 版本）、
  Q4_K / Q5_K / Q6_K / Q8_0 的块格式定义（`ggml-common.h`）。
- tools 里的 `quantize`、`completion`、`batched-bench`、`fit-params`、`imatrix`、`tokenize` 等已删除。

## 12. 参考

- 新增模型：`docs/development/HOWTO-add-model.md`。
- 构建：`docs/build.md`；解析/模板：`docs/development/parsing.md`、`common/jinja/README.md`。
- 可复用工作流：`skills/` 目录，其中 `skills/code-review/SKILL.md` 在写代码前必读。
