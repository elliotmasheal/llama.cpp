# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> 本仓库是 `elliotmasheal/llama.cpp` fork，基线是 ggml-org/llama.cpp 的上游 master。
> `origin` 指向本 fork，未配置 `upstream` remote。**本项目不考虑"能否上游化"，追求小 VRAM 设备下的指定模型的极致性能。**
> 2026-10-24 起只有一条主线：`design/qwen36-moe-runtime-design.md`（qwen36 专用 MoE 运行时，里程碑 M0-M5）。

## 0. 怎么开始

本文件是 agent 的唯一入口与执行契约。性能规格与数学模型在 `design/qwen36-moe-runtime-design.md`（下称 design）。
机器可读状态在 `bench/state/state.json`，与第 2 节的 STATE 块必须一致，由
`python bench/tools/state_check.py` 校验（退出码 0 才允许开跑）。

**启动顺序**：`state_check.py` -> 本文件第 2 节 -> design 里与任务对应的章节 -> 该任务的证据文件。
不要通读历史日志，也不要靠回忆历史结论。

**文档权威（冲突时按此顺序裁决）**

| 优先级 | 文件 | 只放什么 | 谁能改 |
|---:|---|---|---|
| 1 | `design/qwen36-moe-runtime-design.md` | 目标、非目标、硬约束、性能模型、里程碑与验收阈值 | 用户；agent 只读 |
| 2 | `AGENTS.md` | 执行契约、当前状态、任务队列、门禁、循环 | agent 每轮更新状态块 |
| 3 | `bench/RESULTS.md` | 实测数字的唯一来源（含口径头） | 有新读数时追加 |
| 4 | `bench/DECISIONS.md` | 判决、已证伪、开关表、未结项 | 有判决时追加 |
| 5 | `bench/README.md` | 口径、必报列、测量陷阱、目录 | 口径变化时才改 |

规则：`RESULTS.md` 的实测优先于 design 的估计；design 里的数字必须在 `RESULTS.md` 有回链，
没有回链的按估计处理。design 与本文件冲突时停下，按第 7 节升级。跨口径的数字不可比。
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
next: M1-1
decision_gate: M1 open
last_gate: G-REG unknown (2026-10-23 record, not re-run since 44bd79305)
blockers: none
<!-- STATE:END -->

只改这一块和 `bench/state/state.json`，两者必须同一次提交。状态块只写阶段与下一个任务，
"为什么是它、做到什么算完"写在 `state.json` 里，避免两处叙述漂移。

**现状摘要（人读，不参与机器校验）**

- M0 完成：张量账、10K + 128 token 路由 trace、局部性与命中率模拟（`RESULTS.md` 第 1-2 节）。
- M1 进行中：M1-1 / M1-3 / M1-4 未闭环，M1-2 已部分完成（技术类两份），M1-5 待补测。
- M1 是**决策门**：受限词表 E 或周期时间不达标就停下问用户是否下调 TG 目标（design 第 9 节）。
- 已定：完整词表草稿是基线（理论约 125，低于 130）；受限词表是达标手段，其 E 未测。

## 3. 每轮自主循环

一轮 = 一个单变量改动（或一次测量）+ 一次证据采集 + 三份 bench 文档的必要更新 + 状态同步 + 一个提交。
不允许"先在短上下文验证，之后再外推"。

0. **preflight**（任一不过就停，不跑数）
   - `python bench/tools/state_check.py` 退出码 0
   - `git status --short` 只剩本轮预期改动；不带着上一轮的半成品开跑
   - 机器条件：平衡电源 + AC、dGPU 上无其它推理进程、空闲 RAM 满足当前臂
   - 二进制比源码新；计时臂必须与计时段一致（接受数、周期数、命中率来自同一次运行）
1. **领任务**：只做 `state.json` 的 `next`。要改优先级，先在 `DECISIONS.md` 记一条判决，再改 `next`。
2. **读规格**：design 对应章节 + `README.md` 口径 + `DECISIONS.md` 已证伪清单（避免重做死路）。
3. **单变量改动**：改动必须可回退、必须有开关、必须能用一个命令复现。
4. **采证**：用 `bench/tools/` 的脚本，不写一次性命令进报告；原始产物落 `bench/raw/`（不进 git）。
5. **判定**：按第 5 节逐条给 `pass / fail / n-a`。禁止"看起来更快"、"趋势一致"这类结论。
6. **落文档**：`RESULTS.md` 加读数（带口径头）、`DECISIONS.md` 加判决/开关/未结项、
   本文件状态块 + `state.json` 同步。
7. **提交**：小而可回退。提交信息写"改了什么 + 跑了哪些门禁 + 读数出处"。
8. **下一轮**：把 `next` 指到下一个可执行任务；若到决策门，置 `phase_status=blocked` 并等用户。

## 4. 任务队列

只做表内任务；表外需求先落 design（用户批准）或 `DECISIONS.md` 未结项，再进表。

**M1（决策门，进行中）**

| id | 内容 | 验收（done_when） | 证据 |
|---|---|---|---|
| M1-1 | 补齐 10 个 full-attention 层与 layer 39 的解码路由 | 40 层 x 128 token 都有记录；根因写进 `DECISIONS.md` | `bench/raw/m0-qwen36/trace-10k-g128.bin` |
| M1-2 | 非技术文本的局部性复测 | 技术 / 对话或代码 / 多语言三类各一份；w=8 唯一专家与 25% 命中率分文本报；与 M0 偏差 <= 5 pp 或已解释 | `m0-sim-10k.txt`、`m0-sim-multidoc.txt` |
| M1-3 | 受限词表 MTP 接受率 E | 默认深度与 n-max 4 各一份；接受的 token 与完整词表 greedy 一致；E >= 2.1 / 2.3 | `bench/raw/m0-qwen36/mtp-*.out` |
| M1-4 | Vulkan 原型周期时间（k = 1/2/4/8） | 含提交与同步；与 design 2.6 偏差 <= 10%；每 pass 开销 < 1 ms | 待建 |
| M1-5 | 带宽同口径复测 | dense GEMV、H2D、4 线程 CPU 读；偏差 > 5% 要改 design 引用 | `bench/raw/h2d-bw` |

**决策门 M1 的通过条件**（design 第 9 节）：M1-1/3/4 全过，M1-2 局部性偏差 <= 5 pp，M1-5 已回填。
任一项不过：`phase_status=blocked`，问用户是否下调 TG 目标，不许自行改阈值。

**后续里程碑（只登记，不开工）**：M2 prefill（PP >= 1000 + KLD <= 1e-4）、
M3 decode（TG >= 130 + 逐 token 一致）、M4 调优（30 分钟波动 <= 10%）、M5 重组工具与回滚。

**并行纪律**：决策门未过，M2/M3 只允许写测试、工具、微基准，不允许改主路径；
同一个 M 内同时只有一个 `doing` 任务。

## 5. 门禁

| id | 触发条件 | 命令 | 通过标准 | 不通过怎么办 |
|---|---|---|---|---|
| **G-CONS** | 每轮 | `python bench/tools/state_check.py`；`git diff` 自查 ASCII 与注释规范 | 退出码 0，diff 只含本轮改动 | 不提交 |
| **G-EVID** | 每轮有读数 | `bench/tools/` 里的统一脚本 | 必报列齐全，`status = ok`，r >= 2，口径头齐全 | 该批作废，重建口径后重跑 |
| **G-REG** | 改 fork 或复用其中的 ggml 内核 | 见下方六条 | 全部 100% 通过 | 回退改动，记未结项 |
| **G-QUAL** | 改数值路径、精度、注意力实现 | 逐 token 对比参考实现；k=1 路径另测 logits KLD | token 全同且 KLD <= 1e-4（工具随 M2 建） | 不许合入 |
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

## 6. 每轮报告格式

固定字段，缺一行视为本轮未完成：

```
round   : <date> <commit>
task    : <state.json task id>
change  : <one line, single variable>
口径    : model | ctx | prompt tokens | n_gen | k | draft vocab | cache MiB | VRAM tier | load | power
证据    : <command or raw path> | r=<n>
结果    : <必报列>
门禁    : G-CONS pass | G-EVID pass/fail/n-a | G-REG pass/fail/n-a | G-QUAL pass/fail/n-a
判决    : adopt | revert | partial(<why>)
文档    : RESULTS <section> | DECISIONS <item> | next=<id>
阻塞    : none | <what blocks, unblock condition>
```

## 7. 失败处理与升级

- 同一任务第 3 次尝试前，必须先写"失败假设 + 排查计划"进 `DECISIONS.md` 未结项，再动手。
- 失败留三样：复现命令、已排除项、下一步判据。
- **必须停下来问用户的 8 类**：改目标或阈值；改 design；换模型 / 量化 / 后端；删改已有实测数字；
  改口径；新增 design 第 5.1 节未登记的模块或第二套调度器；预计 > 2 小时或 > 20 GiB 磁盘；需要联网下载。
- 不得用"跳过某个臂""只跑一次""把两档平均"之类的手段让门禁变绿。

## 8. 工程红线

1. **不得用 `GGML_ABORT` 处理运行期条件**（容量不足、内存不足、路由过散）。必须回退到慢但正确的路径并计数告警。
2. **SSD 只做容量层，绝不进 decode 单步关键路径**：随机 1.95 MiB = 0.81 GB/s，206 MiB/token 要 258 ms。
   只允许出现在预取批次与 prefill，且服务命中率 >= 85% 的冷层。
3. **新行为一律带开关**：CLI 参数 + `DECISIONS.md` 开关表；默认值由目标工况实测决定；
   只靠 env 而不上 CLI 的行为不得成为默认路径。
4. **新模块必须先登记**：design 第 5.1 节列出允许的模块。不允许出现 design 未列出的第二套调度器、
   缓存或预算器。
5. **数字必须带口径**：日期、提交、电源态、显存档位、k、草稿词表。跨口径比较视为错误。
6. **结论只写进三份 bench 文档**，design 由用户改（agent 只读）；原始产物写 `bench/raw/`（不进 git）。
7. **代码只用 ASCII**：不要 `—`、`→`、`×`、`…`，改用 `-`、`->`、`x`、`...`。
8. **注释 1-2 行**，只写不显然的约定；不复述代码；不按固定列宽硬换行；先写代码后写注释。

## 9. 机器事实与构建

- RX 6700S 8 GiB（`-dev Vulkan1`），设备总量 8176 MiB，桌面常占 0.5-1.4 GiB。
- Ryzen 7 6800HS，RAM 40172 MiB（`bench/env/machine.txt`），电源「平衡」（`bench/env/power.txt`，AC）。
- resident 门槛约 `21.28 x 1.25 + 1 = 27.6 GiB` 空闲内存；不足时 mmap，性能不可比。
- 构建（MinGW + Ninja，不是 MSVC preset）：

  ```
  cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe \
    -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
  cmake --build build-vulkan -j 8
  ```

- Git Bash 里跑 `build-vulkan/bin/*.exe` 前要 `export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"`，
  否则找不到 libstdc++/libgcc DLL，直接 exit 127。
- 性能一律 Release/RelWithDebInfo。测试：`ctest --test-dir build-vulkan --output-on-failure -C Release`。

## 10. 仓库事实

- backend 只有 **CPU 与 Vulkan**，两者之外的东西已物理删除，不要去找。
- 模型只支持 **qwen35moe**（目标模型就是它），`llama_model_mapping()` 只有一个 case。
- 仍在的组件：`src/` 的模型加载 / 图 / KV / 采样 / C API，`ggml/src/ggml-{cpu,vulkan}` 与调度器，
  `common/` 的 CLI 与 `fit.cpp`，`tools/` 的 cli / server / llama-bench / mtmd / perplexity。
- 可复用的内核与格式：Vulkan `mul_mat_vec` / `mul_mat_id`（要改成批量 k 版本）、
  Q4_K / Q5_K / Q6_K / Q8_0 的块格式定义（`ggml-common.h`）。
- tools 里的 `quantize`、`completion`、`batched-bench`、`fit-params`、`imatrix`、`tokenize` 等已删除。

## 11. 参考

- 新增模型：`docs/development/HOWTO-add-model.md`。
- 构建：`docs/build.md`；解析/模板：`docs/development/parsing.md`、`common/jinja/README.md`。
- 可复用工作流：`skills/` 目录，其中 `skills/code-review/SKILL.md` 在写代码前必读。
