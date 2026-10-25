# bench 索引与测试口径

本目录是项目唯一的实测记录处。三份 md 职责不重叠：

| 文件 | 只放什么 | 谁在读 |
|---|---|---|
| `bench/README.md` | 口径常量、必报列、测量陷阱、目录 | 每次跑数之前 |
| `bench/RESULTS.md` | 数字的唯一来源（含日期、提交、电源态、配置、slug） | 报收益、对比基线 |
| `bench/DECISIONS.md` | 判决、已证伪、开关表、未结项 | 决定下一步做什么 |

规格在 `design/qwen36-moe-runtime-design.md`（agent 只读）；任务图在 `bench/state/state.json`，由
`python bench/tools/state_check.py` 校验。跑数前按 `docs/agent/cycle.md` 的 preflight 走一遍。

## 1. 口径常量（同口径才可比）

| 常量 | 值 |
|---|---|
| 模型 | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，`qwen35moe`，40 主层 + 1 MTP，256 专家 top-8） |
| prompt | `bench/prompts/prompt10k.txt` = **10428 token**；文中写 10.4K，不写"10K" |
| ctx | 256K，由 q8_0 K + q4_0 V + 16384 token 常驻窗口支持（U-10，design 5.5）；窗口外页面在主机，不计入 TG |
| 采样 | greedy，单流；推测解码按 k=4 验收 |
| 目标 | PP >= 1000、TG >= 130（硬地板，用户重设前不变） |
| 环境 | 平衡电源 + AC（`bench/env/power.txt`）、dGPU 无其它推理进程、`-lm none`（resident，>= 27.6 GiB 空闲 RAM）；任一不满足就不跑 |
| 显存档位 | `GGML_VK_VRAM_LIMIT_MIB` 模拟 hold back；本机空闲约 7378 MiB，8176 档等价"不限" |
| 重复 | r >= 3，相对极差 <= 15%；超了该批作废，报中位与极差 |

性能数字只能来自新 runtime。fork 只作参考（路由 / 接受率口径、硬件探针），它的 TG/PP 不作结论。

## 2. 必报列

date | commit | phase | task | model | ctx | prompt tokens | n_gen | k | draft vocab | expert cache MiB | VRAM tier | load | power | PP | TG | E | accept p | hit % | H2D MiB/token | VRAM peak | RAM peak | status

- **E 与 p 是导出量**：E = 生成 token / 周期数，周期数 = 生成数 - 接受数（design 2.5）。必须同时给原始 `draft_n` / `draft_n_accepted`，否则 E 不可复核。交叉校验：k=4 时草稿数/周期约 3.9，k=8 时约 7.7。
- **草稿代价单列**：每个草稿步读一次完整输出头约 0.5 GB / 2.3 ms（design 2.6）。只报总 TG 会把这块藏起来。
- `status` 不是 `ok` 一律按缺陷进 `DECISIONS.md` 未结项。
- 正确性与速度分开报：接受的 token 与参考 greedy 逐 token 一致；`k=1` 另报 logits KLD <= 1e-4。KLD / PPL / same top-1 由 `bench/tools/quality/`（X-1）出，口径见 `RESULTS` 第 51 节。**不许**把 fork 自己打印的 PPL 当门禁读数。
- 计时与计数必须同一次运行：E、周期数、命中率、H2D 字节不能跨运行拼。

## 3. 测量陷阱

1. 稠密地板：2.17 GB/token / 156.5 GB/s = 13.9 ms -> 单步上限 72 tok/s。任何"单步 > 72"先怀疑口径。
2. H2D 14.17 GB/s 已接近 PCIe 4.0 x8 的 90%；与 CPU 读并发时掉到 8.9-11.2，分配必须自适应。
3. MTP 默认草稿深度不是 1（约 3）。把 63.8% 当单草稿接受率会得到错误的 TG 投影。
4. 日志里没有周期计数，只能由"生成 - 接受"推算，并与"草稿数/周期"互相校验。
5. 受限词表只影响速度；验证 pass 永远是完整词表，正确性看接受的 token。
6. `llama-bench` 不可用（它把 `n_ctx` 设成 `p+n`）；用 `llama-cli` 或运行时自己的计时。
7. 电源方案与热漂移是乘性因子：跨电源态、连跑多轮后的读数不可直接比。
8. Git Bash 里跑 `build-*/bin/*.exe` 要先把 `/d/Scoop/apps/mingw/current/bin` 放进 PATH，否则子进程秒退、日志 0 字节。
9. 改完代码先确认二进制含本次改动（看构建时间戳），再做 A/B。
10. 外部推理进程在 GPU 上时整批数据作废；容量类失败先看 RAM 再怀疑显存。
11. 编译、repack、第二个模型进程都会扰动计时：跑数前确认 `state_check.py` 的 `lock` 行是 `free`。
12. KV 精度（默认 q8_0 K + q4_0 V）改动注意力数值路径：不过 L-MODEL 不许当默认；16K 窗口外页面在主机，长上下文 decode 不计入 TG。
13. 两臂对比前必须各跑一次预热 pass：首触缺页会造假加速比（`DECISIONS` 35）。

## 4. 目录

```
design/            qwen36-moe-runtime-design.md            规格（只读）
docs/agent/        INDEX cycle gates resources failures build repo-facts LAYOUT
runtime/AGENTS.md  作用域约定（进 runtime/ 自动上链）
bench/README.md    RESULTS.md  DECISIONS.md                 三份文档
bench/state/       state.json（任务图/资源/必读）            无锁文件，锁由 doing+resources 派生
bench/prompts/     prompt10k.txt (10428 tok)  prompt20k.txt (19758 tok)
                   prompt-multidoc10k.txt (9402 tok)
bench/env/         power.txt  machine.txt                    口径依赖的机器事实
bench/tools/       state_check.py（一致性 + lint + --brief + --selftest）
                   m0-trace.cpp  m0-sim.py  m0-analyze.py  membw_probe.py
                   gguf_dims.py  roofline.py  ssd_probe.py  quality/（G-QUAL）
bench/raw/         每次跑的 .out/.err/.bin 产物（不入库）
```

`bench/tools/` 里只有列出的这些参与新主线；其余历史脚本不在任何门禁里，不要引用。
原始产物不进 git。要新证据就重新跑，跑完把结论写进 `RESULTS.md` / `DECISIONS.md`，不要新开第四份证据文档。
