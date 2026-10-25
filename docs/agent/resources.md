# 资源、lane 与领用

本机只有一张 GPU、40 GiB 内存、一块盘。**并行指写代码与写文档并行；执行（跑数、编译、repack）串行。**

## 1. 资源（互斥量，cap 都是 1）

| 资源 | 含义 | 典型任务 |
|---|---|---|
| `gpu` | dGPU 与显存，一次只能一个进程 | 任何跑模型或内核的测量（含 impl 的 GPU 微基准） |
| `host-ram` | 驻留 21.28 GiB 模型（约 28 GiB 空闲内存） | bench 跑 llama-cli / 运行时 |
| `machine-quiet` | 计时、`-j8` 全量构建、内存/磁盘探针、repack | bench/io 整段持有；impl 构建或跑微基准时声明 |
| `disk-io` | 超过 1 GiB 读写或 SSD 带宽测量 | repack 与校验 |
| `trace-bin` | 重新生成路由 trace 文件 | M1-1、M1-2b |
| `prompt-file` | 修改 `bench/prompts/` 下的文件（只读共享） | M1-2a |

**锁只有一套**：任务 `status = doing` 且 `resources` 含某资源，就是持有该资源。不存在额外锁文件；`bench/state/quiet-*` 已废弃（`state_check.py` 发现即报 warning）。

跑 GPU 计时必须声明 `gpu`。不声明就是隐形占用，`state_check.py` 拦不住，只能靠这条纪律。

## 2. lane

| lane | 默认占用 | 并发度 | 必读 | 说明 |
|---|---|---|---|---|
| `bench` | gpu + host-ram + machine-quiet | 1 | `bench/README.md` | 只做测量；全项目只有这一条执行队，串行消费 |
| `impl` | machine-quiet | 2-3 | `runtime/AGENTS.md` | 写代码；路径不重叠时真正并行，编译要等 bench 空闲 |
| `data` | 无 | 任意 | - | 语料、词表、GGUF 元数据、汇总、文档；不阻塞任何人 |
| `io` | machine-quiet + disk-io | 1 | `runtime/AGENTS.md` | 重组与逐字节校验，和 bench、构建互斥 |
| `user` | 无 | 1 | - | 只有用户能做的决定；agent 准备一页选项后停 |

## 3. 工作树与构建隔离

- 每个 impl agent 一个 worktree：`git worktree add ../fastLlama-<id> -b <id>/<task>`。
- 每个 worktree 一个构建目录 `build-<id>`；不许共用 `build-vulkan`，不许两个 agent 同时构建。
- 共享只读：模型文件、`bench/prompts/`、已有 `bench/raw/` 证据、`design/`。

## 4. 领用协议（每个任务照做）

1. `state_check.py` 退出 0；该任务 `status` 是 `todo` / `partial`。
2. 读 `--brief <id>` 打印的必读清单。
3. 把 `status` 改成 `doing`，写 `owner`（`<agent-id>@<host>`）、`worktree`、带时区的 `claimed_at`（如 `2026-10-25T17:40:00+08:00`）。
4. 同一 `resources` 或重叠 `paths` 的另一个 `doing` 必须已经结束。
5. 把本 lane 队列头前移到下一个可跑任务（没有就写 `null`）。
6. 再跑一次 `state_check.py`，退出 0 才开始。
7. 完成时 `status = done`，补 `evidence`（RESULTS slug + 复现命令），读数进 `RESULTS.md`，判决进 `DECISIONS.md`，队列头前移，再校验、再提交。

**接管**：`doing` 且 `claimed_at` 超过 4 小时、或 `owner` 与自己的 agent id 不一致且已超时，才可以接管；接管时同时改写 `owner` 与 `claimed_at`，并在提交信息里写明接管原因。

**编辑方式**：`state.json` 直接编辑（保持 2 空格缩进），由 `state_check.py` 校验。不提供写 JSON 的工具，避免重排整份文件产生巨大 diff。

## 5. 路径所有权

- `task.paths` 是该任务的写权限；段级 glob 重叠即冲突（`state_check.py` 用 `fnmatch` 逐段比较，通配符吃掉剩余段才算重叠）。
- 改别人的路径前先改 `state.json` 的归属并记录原因。
- 合入顺序按 `deps`；一个任务一个提交；冲突由后合入者解决，不许 force push，不许改别人的提交。
- 发现任务拆分不对：先改 `state.json`（拆任务、改 deps/paths/reads），跑通 `state_check.py`，再动手。

## 6. 排班建议

1 个 bench agent + 2-3 个 impl agent + 1 个 data agent。data 先产出 bench 需要的输入（非技术语料、受限词表、KV 预算），bench 才不会空转；impl 在决策门未过时只做工具与原型。
