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

## 3. 工作树与分工

- 并行写代码的 impl agent 各用独立 worktree 与 `build-<id>`；不共用构建目录，不同时构建。单 agent 不为形式额外建 worktree。
- 只读研究无需工作树；共享只读模型、prompt、已有原始证据和 design。
- 按问题/路径分工，给子 agent 契约、边界和输出证据要求；避免重复搜索。主 agent 负责串联与最终复核，不能把摘要当验收。

## 4. 领取、续做、释放

1. 先找本人 `doing` 并读其 `--brief`；队首已前移，不代表本人任务完成。新领任务须 `todo/partial`、依赖全部验收完成。
2. 读必读清单；核对其他 `doing` 的资源与路径。冲突先登记 blocker，转做无冲突工作，不篡改声明规避锁。
3. 新领写 `status=doing`、`owner=<agent-id>@<host>`、`worktree`、带时区 `claimed_at`；队首前移到下个可跑任务，无则 `null`。同步根 STATE。
4. `state_check.py` 回 0 才执行；所需资源由本人本任务持有即可，不要求全局 `lock=free`。同一任务也不能同时启动两个重执行。
5. 续做/阶段检查点核实本人任务仍活跃，更新 `claimed_at`，在 4 小时校验窗口内续期；续期不是重新领任务，不动别人归属。
6. 全部验收后 `done`，补命令、产物、判决引用等 `evidence`；部分完成不标 done。暂停写 `partial/blocked`、下一命令/判据，确认进程退出再释放资源；重执行仍在跑就不能释放锁。
7. 更新队列与 STATE，再校验；有授权才提交。可连续完成多个相关闭环，不在每次领取后请求下一步。

**超时不是接管授权**：先核对 owner、工作树与进程。原任务仍在运行则等待；确认已停止且收到移交/接管授权后才改 owner/时间并记原因。无法确认就阻塞，不杀进程、不强夺锁。

**状态编辑**：由主 agent 串行编辑 `state.json`，保留 2 空格与小 diff，写前重读，写后校验。脚本没有 CAS/自动心跳；多个工作树的状态副本不是全局互斥，重执行前必须与协调者的最新持锁状态对账。

## 5. 路径与任务边界

- `task.paths` 限定任务写范围；重叠路径不并行。协作变更先获原 owner 移交并记录，不靠改归属抢占。
- 用户明确指定的文档/流程维护可直接开展，先核对在途路径，不冒领 GPU 任务；进入重执行仍须声明资源。
- 范围内缺少必要测试或依赖任务时，主 agent 可补 paths/reads、拆分任务并记录理由，校验后继续；不得降低 done_when、删验收依赖或越过 design。
- 按依赖合入，有授权时一个可验收增量一个提交；保留他人工作，不 force push，不改别人提交。
- bench 保持唯一重执行队；实现和研究可并行，未过决策门不擅自进入被门禁阻断的阶段。
