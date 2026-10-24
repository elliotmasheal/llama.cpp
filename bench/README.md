# bench 索引与测试口径

本目录是这个 fork 唯一的实测记录处。2026-10-23 起从 54 份文档合并成 4 份，职责不重叠：

| 文件 | 只放什么 | 谁在读 |
|---|---|---|
| `bench/README.md` | 测试口径、档位定义、怎么跑、测量陷阱 | 每次跑数之前 |
| `bench/RESULTS.md` | 数字的唯一来源（含日期、分支、电源态、配置） | 报收益、对比基线 |
| `bench/DECISIONS.md` | 判决、已证伪清单、开关表、代码级硬约束、未结项 | 决定下一步做什么 |
| `bench/THREE-TIER.md` | 三层（VRAM+RAM+SSD）与自适应的现状、缺口、步骤 | 动 cache / fit / loader 之前 |

被合并删除的 54 份 md 与原始日志不再维护，需要原文时查 git 历史（提交 `a44e13e08` 及其之前的工作区）。

## 1. 统一目标口径

优化目标：**256K ctx + 10K prompt**，在 qwen35moe（Ornith-1.5-35B-Q4_K_M，20.2 GiB）上追求 PP/TG 最大化，
显存、内存、SSD 三层协同，且**显存与内存都不足时 SSD 必须能当临时缓冲**。

每一轮优化都要在目标环境参数下测，不允许只在短上下文档上测完再外推：

```
llama-cli -m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -fa on \
  -t 5 -b 2048 -ub 2048 -c 262144 -f bench\prompts\prompt10k.txt -n 64 \
  --temp 0 --no-warmup --no-display-prompt -lm none -v
```

- **K/V 默认按工作集分配**（`-kvg -1`，见 `DECISIONS.md` 第 4 节）：256K + 10K 只用 320 MiB，驻设备。
  所以本口径**不再加 `-nkvo`**；`-nkvo` 只用于"工作集超过显存"的对照臂，且它和 `--moe-cache-mib` 非零同时出现会触发
  `llama-moe-cache.cpp:1910`（F1 未修）。要复现旧行为用 `-kvg 0`（一次分配 `n_ctx`，256K 下 5120 MiB）。
- **`--kv-page` 默认仍是 off**：`--kv-page-auto auto` 让 `common/fit.cpp` 判决（扣掉 score 预留后没有余量就开分页，
  3 个档位的实测判决见 `RESULTS.md` 6.5），`--kv-page` / `--no-kv-page` 强制开/关。分页臂必须配 `-K`（KV 在设备），
  gather 的稠密工作区由 `--kv-gather-mib`（默认 256）封顶。默认值不改的原因见 `DECISIONS.md` 第 6 节第 17 条
  （10K prompt 下分页与连续布局的输出不一致，未定性）。
- `used` 里的 KV 只算当前分配，不再等于 `n_ctx x 20 KiB`；比较不同臂时先看日志的 `K/V grown to N cells`
  （连续布局）或 `paged K/V: N blocks of M cells`（分页）。
- 长跨距注意力的 score 缓冲由 `-fac/--fa-chunk-mib`（默认 256）封顶，query 维按预算切块；`-fac 0` 是旧行为
  （整段一块，`RESULTS.md` 0.4 有五个 cap 的消融）。改这个值要同时看 bank 与 `reallocs`。
- 不加 `--moe-cache-mib` 时由内存规划器（`common_moe_plan_get()`）自己算 bank，这就是"自适应 RAM 条件"。
- 显存档位用 `GGML_VK_VRAM_LIMIT_MIB` 实现，见第 2 节。
- 电源：「平衡」方案 + AC 供电（`bench/env/power.txt` 记 GUID），dGPU 上不得有其它推理进程，
  `-lm none` 还要求至少 22 GiB 空闲 RAM。脚本会先检查这三条，不合格直接停。

## 2. 显存与内存档位

| 维度 | 怎么造出来 | 语义 |
|---|---|---|
| VRAM 4G / 6G / 8G | `GGML_VK_VRAM_LIMIT_MIB=4096` / `6144` / 不设（8176） | 设备初始化时一次性 hold back「当前可用 - 目标值」的 device-local 内存。`VK_EXT_memory_budget` 按进程计数，所以 `llama-fit-params`、MoE bank 规划器、后续每次分配看到的都是缩小后的预算 |
| RAM 充足 / 不足 | `-lm none`（resident）/ `-lm mmap`（或 `LLAMA_AUTO_RESIDENT=0`） | resident 门槛是 `n_bytes x 1.25 + 1 GiB`，20.2 GiB 模型要约 26.3 GiB 空闲；不足时 AUTO 自动退回 mmap，而 mmap 下 bank 与预取被守卫强制为 0 |

已知边界：`GGML_VK_VRAM_LIMIT_MIB` 对 UMA（集显）不生效；**设备初始化之前的查询仍会看到完整预算**
（`ggml_backend_dev_memory` 本身不创建设备），所以顺序敏感的调用要确认日志里已经出现 `VRAM limited to ...`。
用 `bench/tools/vklimit-probe.cpp` 可以在不加载模型的情况下核对某个档的报告值。
设备不支持 memory budget 扩展时只缩真实容量、不改报告值。

2026-10-23 实测（`a44e13e08` + 工作区，桌面占用后的空闲 7378.8 MiB）：不设 = 7378.8；
`=6144` -> hold back 1234.8 MiB，init 后报 **6142.9**；`=4096` -> hold back 3282.8 MiB，init 后报 **4094.9**；
Vulkan0（680M，UMA）两次都是 22842.2 MiB，不受影响。

## 3. 每轮优化要报的列

`bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2` 直接产出这些列，缺一不可（参数见脚本 `-h`：
`-T` 显存档 / `-C` ctx / `-B` bank / `-K` KV 在 GPU / `-L` load 模式 / `-r` 重复 / `-E` 附加 env /
`-a` 臂名 / `-A` 该臂的附加 CLI 参数）：

`arm | tier | run | load | ctx | KV | bank | ncmoe | PP t/s | TG t/s | VRAM peak | held | VRAM used | RAM peak | reallocs | hit % | host blk | gather MiB | status`

- `arm` 是 `-a` 给的名字，跨臂比较只看同一 `tier` 下的同名行。
- `host blk` = 落在 host 的 K/V 块数（分页臂的容量指标，对应日志 `K/V block N placed on the host`）；
  `gather MiB` = 分页 prefill 稠密工作区长到的最大值，连续布局恒为 0。
  `gather MiB` 只在 `--kv-page` 下非零，`--kv-gather-mib 0` 是它的对照。

- `VRAM peak` 取**每进程独占**（`\GPU Process Memory(pid_<pid>*)\Dedicated Usage` 的峰值），
  设备总量 8176 MiB 里桌面已占约 0.5-1.4 GiB，看设备总量会误判。
- `held` 是 `GGML_VK_VRAM_LIMIT_MIB` 自己 hold back 的那块，它也计入进程独占峰值，
  所以真实足迹看 `used = peak - held`。设定值高于当时空闲时不会 hold back
  （本机当时空闲 7378.8 MiB，所以 `8176` 档等价于"不限"）。
- `status` 必须是 `ok`。出现 `abort` / `vram-oom` 是缺陷，不是"配置不对"，要当成待修项记进 `DECISIONS.md`。
- 改了注意力图、分块、KV 或任何数值路径的轮次，必须附质量门的读数（`bash bench/tools/quality-gate.sh`，
  输出 `PPL | mean KLD | same top p | PASS/FAIL`）。它不在矩阵脚本里，因为要读 2 GB 的 logits 基线。
- 同档同配置的臂之间比收益；跨档、跨电源态的数字不可直接比。
- `THREE-TIER.md` 第 4 节的 P0 归因（F7 / F8）除上表外还要报四项，它们不在现成的矩阵列里：
  每 eval 的 `splits`（`GGML_SCHED_TRACE=1`）、日志里的 `RS buffer size`、
  `-nkvo` 下长跨距时**设备** compute buffer 的增量（`GGML_VK_MEMORY_LOGGER`）、
  以及 `sched_reserve` 与 teardown 两侧的 compute 归属（设备 `Vulkan1 compute buffer size` 对 host 的
  `Vulkan_Host ...`，出现 `does not match expectation` 的 warning 必须原文抄进报告）。
  这四项决定"状态解耦"与"score / compute 预算归属"值不值得做，缺一项就不允许改代码。

## 4. 测量陷阱

1. 电源方案是乘性因子：混合电源态与「平衡」之间约 x1.33，跨 session 不可比。
2. 热漂移：连跑 pp2048 后几次会掉到约 310 t/s，用 `-GapSec 30` 或 r=2 看方差。
3. `GGML_VK_PERF_LOGGER=1` 把 decode 拖慢约 2.6 倍，prefill 上只汇总到约 77 ms 碎片，与真实不符：
   只能用看占比，不能看绝对时间；kernel 绝对时间用 `test-backend-ops perf` 隔离微基准。
4. `llama-bench` 把 `n_ctx` 设成 `n_prompt + n_gen`（`tools/llama-bench/llama-bench.cpp:1337`），
   所以 `-p 8192` 是 ctx 8192，`-c` 不起作用；目标口径必须走 `llama-cli` 的单轮 prompt 文件法。
5. Git Bash 里 `$!` 不是 Windows pid，按 pid 采显存会得到 0；必须用 `Start-Process -PassThru`。
6. 外部推理进程在 GPU 上时整批数据作废（`ErrorOutOfDeviceMemory` 也可能是 RAM 不足，
   先看 `\Memory\Available MBytes` 再怀疑显存）。
7. 改完代码要先确认二进制里真的含这个改动（strings/构建时间戳），再做 A/B。
8. cache 路径的数字来自 `llama-cli`，与 `llama-bench` 差约 2-5%。
9. `ub=1024` 是病态点（66.79 t/s），不要拿它做对照；`-ub 2048` 必须配 `-b 2048`。
10. 从别的 shell 启动 `build-vulkan/bin/*.exe` 时若 PATH 里没有 MinGW 的 bin，子进程会**秒退且日志 0 字节**，
    看起来像"配置错误"。`run-matrix.sh` 自己 `export PATH=/d/Scoop/apps/mingw/current/bin:$PATH`。
11. `run-matrix.sh` 的 dGPU 独占检查是有意的：只要树里有 `llama-*` 进程就报错退出，不去跑一批作废的数据。
13. **跑批器本身能造出 4 倍差**：2026-10-23 用 PowerShell `Start-Process -RedirectStandardError` 接一个
    带 `-v` 的子进程，decode 从 21.6 掉到 4.8（同一命令行换 bash 重定向就是 21.6），因为 PowerShell 要逐行搬运
    子进程日志。跑批器已改成 bash（`run-matrix.sh`，普通 `> out 2> err`，按进程名找回 Windows pid 采显存）；
    旧的 PowerShell 版退役在 `bench/raw/run-matrix.ps1.superseded`，不要复活它。
    另一个坑：用 `Start-Process` 起 hidden cmd/.bat 时子进程跑完不退出，会占着显存空转。
    任何"引擎回归"的结论，都要先用另一种启动方式复现一次再说。
14. 不带 `-v` 时这个 fork 的 INFO 日志（`common_moe_plan_impl`、`sched_graph_realloc`、`hit rate`、`VRAM limited`）
    一条都不出，所以 `held`/`reallocs`/`hit` 这几列必须有 `-v`。
15. **`bash` 必须是 Git Bash，不能是 WSL 的 `/usr/bin/bash`**。2026-10-23 用 WSL bash 跑 `run-matrix.sh` 时：
    `GGML_VK_VRAM_LIMIT_MIB` 没有传给 interop 起来的 `.exe`（日志里没有 `VRAM limited`，三档 bank 完全一样），
    用 Windows pid 采显存的采样也全为 0（`peak`/`used`/`RAM` 列都是 0），只有 PP/TG 是"不限档"的真实读数。
    判据：日志里必须出现 `VRAM limited to ...`，否则整批 `tier` 列作废。用
    `"C:\Program Files\Git\bin\bash.exe" -c ...` 启动；Git Bash 的 `PATH` 以 `/mingw64/bin` 开头且没有 `/mnt/`，
    WSL 的 `PATH` 以 `/usr/local/sbin` 开头并带 `/mnt/c/...`，一眼可辨。
16. **RAM 刚被腾出来时，`-lm none` 的失败长得像显存缺陷**：2026-10-24 在用户刚结束一个 19 GiB 的 chat 进程后
    立刻开跑，前三次 `status=vram-oom`（日志是 `llama_model_load: vk::Queue::submit: ErrorOutOfDeviceMemory`，
    发生在 18.9 GiB 的 `Vulkan_Host` 专家上传阶段），而同一批后三次全部 `ok`、`ram peak` 19.4 GiB。
    当时 `FreePhysicalMemory` 只有 9.5 GiB，低于 resident 门槛。所以 `vram-oom` 先看 RAM 再看显存；
    脚本已在 preflight 里要求 `-lm none` 至少有 22 GiB 空闲 RAM。
17. 不要在跑批过程中编辑 `run-matrix.sh`：bash 按偏移增量读脚本，改了长度之后尾部会被读成乱码
    （实测报 `syntax error near unexpected token 'done'`，CSV 本身没坏）。要改就先跑完这一批。

## 5. 目录里还留着什么

```
bench/README.md  RESULTS.md  DECISIONS.md  THREE-TIER.md   四份文档
bench/prompts/   prompt10k.txt (10428 tok)  prompt20k.txt (19758 tok)
bench/env/       power.txt  machine.txt                      口径依赖的机器事实
bench/tools/     run-matrix.sh（矩阵） quality-gate.sh（PPL + KLD 门禁）
                 monitor.ps1  run-case.ps1  bench10k.ps1  matrix.ps1
                 ssd_probe.py  membw_probe.py  gguf_dims.py  read_trace.py
                 summarize_perf.py  vkgate.cpp               可复用探针与门禁
bench/raw/       每次跑的 .out/.err/.csv 产物（不入库）
```

原始产物不再进 git。要新证据就重新跑，跑完把结论写进 `RESULTS.md` / `DECISIONS.md`，
不要再往本目录加第五份 md。
