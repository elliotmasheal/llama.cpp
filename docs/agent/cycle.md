# 每轮循环与报告格式

一轮 = 一个单变量改动（或一次测量）+ 一次证据采集 + 三份 bench 文档的必要更新 + 状态同步 + 一个提交。

## 0. preflight（任一不过就停，不跑数）

1. `python bench/tools/state_check.py` 退出码 0。
2. `python bench/tools/state_check.py --brief <本 lane 队首 id>`，按打印的必读清单读完。
3. `git status --short` 只剩本轮预期改动；不带着上一轮的半成品开跑。
4. 本 lane 队首未被别人领走：该任务 `status` 是 `todo` / `partial`，或 `doing` 且 `owner` 是本 agent 且 `claimed_at` 在 4 小时内。
5. 机器条件：平衡电源 + AC、dGPU 上无其它推理进程、空闲 RAM 满足当前臂、`state_check.py` 的 `lock` 行为 `free`（锁由 `state.json` 的 `doing + resources` 派生，没有独立锁文件）。

## 1. 八步

1. **领任务**：按 `docs/agent/resources.md` 第 4 节把队首改成 `doing`，写 `owner`、`worktree`、带时区的 `claimed_at`。
2. **读规格**：`--brief` 的必读清单 + design 对应章节 + `DECISIONS.md` 已证伪清单。
3. **单变量改动**：可回退、带 CLI 开关、一个命令能复现。
4. **采证**：只用 `bench/tools/` 的脚本；原始产物落 `bench/raw/`（不进 git）。
5. **判定**：按 `docs/agent/gates.md` 逐条给 `pass / fail / n-a`。禁止"看起来更快""趋势一致"。
6. **落文档**：`RESULTS.md` 加读数（带口径头与 slug）、`DECISIONS.md` 加判决/开关/未结项、`AGENTS.md` STATE 块与 `state.json` 同步。
7. **提交**：小而可回退。提交信息写"改了什么 + 跑了哪些门禁 + 读数出处"。**一轮一个提交**，含读数、判决与状态。
8. **交接**：任务置 `done`（带 `evidence`）、队列头前移、`state_check.py` 回 0。

## 2. 每轮报告格式

固定字段，缺一行视为本轮未完成：

```
round   : <date> <commit>
lane    : <bench|impl|data|io|user> [resources]
task    : <task id>
change  : <one line, single variable>
口径    : model | ctx | prompt tokens | n_gen | k | draft vocab | cache MiB | VRAM tier | load | power
reads   : <本轮实际读过的 docs/agent 与 bench 文档>
证据    : <command or raw path> | r=<n>
结果    : <必报列> 中位 <x> 极差 <y>%
门禁    : G-CONS pass | G-EVID pass/fail/n-a | G-REG pass/fail/n-a | L-KERN pass/fail/n-a | L-MODEL pass/fail/n-a
判决    : adopt | revert | partial(<why>)
文档    : RESULTS <slug> | DECISIONS <slug> | queue=<lane>:<next id>
阻塞    : none | <what blocks, unblock condition>
```

`reads` 行与 `state.json` 的 `reads` 字段对账；漏读在 `state_check.py --lint` 里报 warning。

## 3. 节锚点（slug）

`RESULTS.md` 与 `DECISIONS.md` 的节号是**显示用序号，不是引用键**：历史编号有重复与缺号（`RESULTS` 缺 59，30 落在 34 之后），按节号引用并不可靠。新引用一律用 slug。

新节写法（标题下一行加注释锚点）：

```
## M2-6b 块主序重排: 3.38 -> 12.89 TFLOPS (2026-10-25)
<!-- slug: m2-6b-repack -->
```

引用写法：`RESULTS.md#<slug>`，或行内简写 `RESULTS <slug>`。

- 历史节不重排、不改标题、不改数字；旧写法"第 N 节"仍可用，但不再作为新引用。
- `state_check.py --lint` 检查：被引用的 slug 必须存在，且每个 slug 只出现一次。
