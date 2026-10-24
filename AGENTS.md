# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> 本仓库是 `elliotmasheal/llama.cpp` fork，基线是 ggml-org/llama.cpp 的上游 master。
> `origin` 指向本 fork，未配置 `upstream` remote。**本项目不考虑"能否上游化"，追求小 VRAM 设备下的指定模型的极致性能。**

## 0. 本文件是什么

本文件是 agent 的唯一入口与执行契约：目标、当前状态、任务队列、门禁、每轮循环、升级条件。
性能规格与数学模型在 `design/qwen36-moe-runtime-design.md`（下称 design）。
机器可读状态在 `bench/state/state.json`，与第 2 节的 STATE 块必须一致，由
`python bench/tools/state_check.py` 校验（退出码 0 才允许开跑）。

**新会话的启动顺序**（照做，不要先通读整个 `bench/`）：`state_check.py` -> 本文件第 2 节
-> design 里与当前任务对应的章节 -> 该任务的证据文件。上下文不够时以这四步为准，不要靠回忆历史结论。

**文档权威（冲突时按此顺序裁决）**

| 优先级 | 文件 | 只放什么 | 谁能改 |
|---:|---|---|---|
| 1 | `design/qwen36-moe-runtime-design.md` | 目标、非目标、硬约束、性能模型、里程碑与验收阈值 | 用户；agent 只读 |
| 2 | `AGENTS.md` | 执行契约、当前状态、任务队列、门禁、循环 | agent 每轮更新状态块 |
| 3 | `bench/RESULTS.md` | 实测数字的唯一来源（含口径头） | 有新读数时追加 |
| 4 | `bench/DECISIONS.md` | 判决、已证伪、开关表、代码级硬约束、未结项 | 有判决/开关/未结项变化时追加 |
| 5 | `bench/README.md` | 两条口径、必报列、测量陷阱、目录说明 | 口径变化时才改 |
| 6 | `bench/THREE-TIER.md` | 归档线（Ornith 三层协同）的现状与缺口 | 冻结，只读 |

**裁决规则**

1. 数字冲突：`RESULTS.md` 的实测优先于 design 的估计。design 里的实测数字必须在 `RESULTS.md` 有回链，
   没有回链的按估计处理，不得引用为结论。
2. 方向冲突：design 的"目标 / 非目标 / 里程碑"优先于本文件。本文件与 design 冲突时停下，按第 7 节升级。
3. 口径冲突：先看 `bench/README.md` 第 0 节的两条口径表。**跨口径比较一律视为错误**，
   引用归档线数字时必须写明"归档线"。
4. 一次提交只改一层：先数字来源，再判决，再状态。跨层改动拆提交。

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
- H2D 14.3 GB/s 是硬墙；CPU 4 线程读 40 GB/s。SSD 随机读 0.81 GB/s。
- 不改变量化格式；不改变输出：k=1 路径 logits KLD <= 1e-4，推测路径接受的 token 与参考 greedy 一致；
  浮点归约顺序导致的差异按 design 8.1 / R9 记录并量化上报。
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

- M0 完成：张量账、10K + 128 token 路由 trace、局部性与命中率模拟（`RESULTS.md` R 节）。
- M1 进行中：M1-1 / M1-3 / M1-4 未闭环，M1-2 已部分完成（技术类两份），M1-5 待补测。
- M1 是**决策门**：受限词表 E 或周期时间不达标就停下问用户是否下调 TG 目标（design 第 9 节）。
- 已定：完整词表草稿是基线（理论约 125，低于 130）；受限词表是达标手段，其 E 未测。
- 旧线（Ornith-1.5-35B + `llama-cli` + THREE-TIER）自 2026-10-24 起冻结为归档对照，不再追加计划。

## 3. 每轮自主循环

一轮 = 一个单变量改动（或一次测量）+ 一次证据采集 + 四份 bench 文档的必要更新 + 状态同步 + 一个提交。
不允许"先在短上下文验证，之后再外推"。

0. **preflight**（任一不过就停，不跑数）
   - `python bench/tools/state_check.py` 退出码 0
   - `git status --short` 只剩本轮预期改动；不带着上一轮的半成品开跑
   - 机器条件：平衡电源 + AC、dGPU 上无其它推理进程、空闲 RAM 满足当前臂
   - 二进制比源码新；需要 `held` / `reallocs` / `hit` 列时必须带 `-v`
   - 跑批期间不要编辑正在被 bash 读取的脚本（见 `README.md` 4.17）
1. **领任务**：只做 `state.json` 的 `next`。要改优先级，先在 `DECISIONS.md` 记一条判决，再改 `next`。
2. **读规格**：design 对应章节 + `README.md` 口径 + `DECISIONS.md` 已证伪清单（避免重做死路）。
3. **单变量改动**：改动必须可回退、必须有开关、必须能用一个命令复现。
4. **采证**：用 `bench/tools/` 的脚本，不写一次性命令进报告；原始产物落 `bench/raw/`（不进 git）。
5. **判定**：按第 5 节逐条给 `pass / fail / n-a`。禁止"看起来更快"、"趋势一致"这类结论。
6. **落文档**：`RESULTS.md` 加读数（带口径头）、`DECISIONS.md` 加判决/开关/未结项、
   本文件状态块 + `state.json` 同步、`README.md` 仅口径变化时改。
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
| **G-EVID** | 每轮有读数 | 新主线用 `bench/tools/` 的统一脚本（运行时落地前是 m0-* 探针）；归档线用 `bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2` | 必报列齐全，`status = ok`，r >= 2，口径头齐全 | 该批作废，重建口径后重跑 |
| **G-REG** | 改 fork 的 Vulkan / 调度 / GEMM / KV | 见下方六条 | 全部 100% 通过 | 回退改动，记未结项 |
| **G-QUAL** | 改注意力数值、精度、分块、混合精度 | `bash bench/tools/quality-gate.sh` | `GATE PASS`（PPL / mean KLD / same top p 三项都要贴） | 不许合入 |
| **G-DEC** | 里程碑 | design 第 8/9 节 | 阈值全过且证据可复核 | 停，问用户 |

**G-REG 的固化数字**（2026-10-23 记录，`build-vulkan/bin/` 下）：

- `test-backend-ops -o MUL_MAT_ID -b CPU` = 945/945
- `test-backend-ops -o MUL_MAT_ID_MASK -b CPU` = 4/4；`-b Vulkan1` = 0/0（Vulkan 侧没有这个 op）
- `test-backend-ops -o MUL_MAT_ID -b Vulkan1` = 939/939
- `test-backend-ops -o MUL_MAT -b Vulkan1` = 2275/2275
- `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` = 5332/5332
- `ctest --test-dir build-vulkan` = 32/32

**质量门的基线**：PPL 5.798313，logits 基线 2.0 GB 在 `F:/LLM/kld/`（不进 git）。
任何降精度、混合精度、CPU 分支或注意力数值路径的改动，都要贴读数，否则不予验收。

**门禁纪律**

- 门禁数字由 design 或本节规定，agent 不得为了过门而改判据、缩 prompt、跳档位、关 `-v`、改 seed。
- 门禁失败必须留复现命令与已排除项，"配置不对"不是结论。
- 归档线的 `G-REG` 数字是 2026-10-23 的记录，`44bd79305` 之后未重跑；动 fork 代码前先重跑一遍。

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

1. **容量不足必须退化，不得 abort**。`src/llama-moe-cache.cpp` 还有 5 处 `GGML_ABORT`
   （`:128`、`:1740`、`:1910`、`:2013`、`:2045`），其中 `:1910` 把正确性绑在调度器分片启发式上。
   动 fork 代码时必须按 `DECISIONS.md` 第 6 节第 1 条的修法改成回退 + 计数告警。
2. **SSD 只做容量层，绝不进 decode 单步关键路径**：随机 1.95 MiB = 0.81 GB/s，206 MiB/token 要 258 ms。
   只允许出现在预取批次与 prefill，且服务命中率 >= 85% 的冷层。
3. **新行为一律带开关**：CLI 参数 + `DECISIONS.md` 开关表；默认值由目标工况实测决定；
   只靠 env 而不上 CLI 的行为不得成为默认路径。
4. **复用优先，新模块必须先登记**：design 第 5.1 节列出允许的模块。不允许出现 design 未列出的
   第二套调度器、缓存或预算器；借用的 fork 组件按 `THREE-TIER.md` 顶部复用表判定。
5. **数字必须带口径**：日期、提交、电源态、显存档位、bank、k、草稿词表。跨口径比较视为错误。
6. **结论只写进那四份 bench 文档**，design 由用户改（agent 只读）；不新增第五份 md；原始产物写 `bench/raw/`（不进 git）。
7. **代码只用 ASCII**：不要 `—`、`→`、`×`、`…`，改用 `-`、`->`、`x`、`...`。
8. **注释 1-2 行**，只写不显然的约定；不复述代码；不按固定列宽硬换行；先写代码后写注释。

## 9. 机器事实与构建

- RX 6700S 8 GiB（`-dev Vulkan1`），设备总量 8176 MiB，桌面常占 0.5-1.4 GiB。
- Ryzen 7 6800HS，RAM 40172 MiB（`bench/env/machine.txt`），电源「平衡」（`bench/env/power.txt`，AC）。
- 新模型的 resident 门槛约 `21.28 x 1.25 + 1 = 27.6 GiB` 空闲内存；不足时 AUTO 退回 mmap。
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

## 10. 代码地图

- `src/` 核心库 libllama：加载 `llama-model.cpp` / `llama-model-loader.cpp` / `llama-mmap.cpp`；图 `llama-graph.cpp`；
  KV `llama-memory*.cpp`、`llama-kv-cache*.cpp`；MoE cache `llama-moe-cache.cpp`、执行器 `llama-moe-exec.cpp`；
  采样 `llama-sampler.cpp`；C API `include/llama.h`，私有扩展 `src/llama-ext.h`。
- `ggml/` 张量库 + backend（只有 `ggml/src/ggml-cpu` 与 `ggml/src/ggml-vulkan`）；调度器 `ggml/src/ggml-backend.cpp`。
- `common/` 参数/CLI（`arg.cpp`、`common.cpp`）、放置与规划（`fit.cpp`、`fit.h`）、chat/解析、`speculative.cpp`。
- `tools/` `cli`、`server`、`llama-bench`、`mtmd`、`perplexity`、`ui`；`app/` 统一 `llama` 入口。
- `design/` 新主线规格；`bench/state/` 机器可读状态；`bench/tools/` 口径脚本与探针。

## 11. 仓库裁剪范围

本 fork 已物理裁剪（`trim-vulkan-cpu-qwen35moe` 分支），不要去找不存在的东西：

- backend 只有 **CPU 与 Vulkan**；CUDA/Metal/SYCL/OpenCL/MUSA/CANN 等源码、CI、文档、fixture 全部删除。
- 模型只支持 **qwen35moe**（`src/models/qwen35moe.cpp` + `delta-net-base.cpp`），
  `llama_model_mapping()` 只有这一个 case。
- tools 只有 **cli、server、llama-bench、mtmd、perplexity**（外加 `tools/ui`）；
  `quantize`、`completion`、`batched-bench`、`fit-params`、`gguf-split`、`imatrix`、`tokenize`、
  `export-lora`、`results`、`examples/`、`pocs/` 已删除。
- tests 只剩 32 个可跑用例；`test-tokenizer-0` 与 `models/ggml-vocab-*.gguf` 已删除。
- CI 只留 `build-vulkan.yml` + `code-style.yml`；`.devops` 只留 `vulkan.Dockerfile` + `tools.sh`。
- Python 侧保留 `convert_hf_to_gguf.py` + `conversion/` + `gguf-py/` + `requirements/`。

## 12. 归档线（Ornith，冻结）

2026-10-24 之前的全部优化对象是 `Ornith-1.5-35B-Q4_K_M`（20.2 GiB）走 `llama-cli` 单步前向，
记录在 `RESULTS.md` 第 0-8 节与 `THREE-TIER.md`。它现在只提供三样东西：

1. 已付过代价的物理常数：217 GB/s、H2D 14.3 GB/s、CPU 40 GB/s、SSD 0.81 GB/s。
2. 已证伪清单（`DECISIONS.md` 第 3 节）：静态驻留、图内 hybrid、执行器变体、调 kernel 旋钮等。
3. 动 fork 代码时仍然生效的红线（abort、KV 分页、SET_ROWS 融合、`grow()` 扩容）。

**不许做的事**：把归档线的 26.6-27.1 t/s 与新主线的 130 tok/s 放在一起比较或推导；
把归档线的 `-ncmoe` / bank / `-nkvg` 方案直接搬进新主线（新主线的专家缓存在 design 4.3 重定义）。

## 13. 参考

- 新增模型：`docs/development/HOWTO-add-model.md`。
- 构建：`docs/build.md`；server：`tools/server/README.md`、`tools/server/README-dev.md`。
- 解析/模板：`docs/development/parsing.md`、`docs/autoparser.md`、`common/jinja/README.md`。
- 可复用工作流：`skills/` 目录，其中 `skills/code-review/SKILL.md` 在写代码前必读。
