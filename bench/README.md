# bench 索引与测试口径

本目录是本项目唯一的实测记录处。只有三份 md，职责不重叠：

| 文件 | 只放什么 | 谁在读 |
|---|---|---|
| `bench/README.md` | 口径、必报列、测量陷阱、目录 | 每次跑数之前 |
| `bench/RESULTS.md` | 数字的唯一来源（含日期、提交、电源态、配置） | 报收益、对比基线 |
| `bench/DECISIONS.md` | 判决、已证伪、开关表、未结项 | 决定下一步做什么 |

规格在 `design/qwen36-moe-runtime-design.md`（agent 只读）；机器状态在 `bench/state/state.json`，
由 `python bench/tools/state_check.py` 校验。

## 1. 统一口径

优化目标：`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（21.28 GiB，arch `qwen35moe`），
**10K prompt，greedy，单流**，PP >= 1000、TG >= 130；ctx 256K 的驻留显存不成立
（`RESULTS.md` 第 10 节），口径待 U-3 裁决。

- 运行时尚未落地（M1）。fork 只作参考：路由/接受率用 `llama-cli`，硬件上限用 `bench/tools/` 的探针；
  **性能数字只能来自新 runtime**，fork 的 TG/PP 不作为结论。
- 目标工况的完整矩阵（chunk、k、草稿词表、专家缓存、PP/TG/E）由 R0 的 roofline 曲线与 M2/M3 定义；
  在那之前不得用代理指标外推成结论。
- 环境前置：平衡电源 + AC 供电（`bench/env/power.txt`）、dGPU 上无其它推理进程、
  `-lm none`（resident，至少 27.6 GiB 空闲 RAM）。任一不满足就不跑。
- 计时与计数必须同一次运行：E、周期数、命中率、H2D 字节不能跨运行拼。

**显存档位**用 `GGML_VK_VRAM_LIMIT_MIB` 模拟（设备初始化时 hold back 差额，之后的预算查询都看到缩小值）。
本机空闲约 7378 MiB，所以 `8176` 档等价于"不限"；4G/6G 档只在需要验证退化路径时用。

## 2. 必报列

```
date | commit | phase | task | model | ctx | prompt tokens | n_gen | k | draft vocab |
expert cache MiB | VRAM tier | load | power | PP | TG | E | accept p | hit % |
H2D MiB/token | VRAM peak | RAM peak | status
```

- **E 与 p 是导出量**：E = 生成 token / 周期数，周期数 = 生成数 - 接受数（design 2.5）。
  必须同时给原始 `draft_n` / `draft_n_accepted`，否则 E 不可复核。
  交叉校验：k=4 时草稿数/周期约 3.9，k=8 时约 7.7。
- **草稿代价单列**：每个草稿步读一次完整输出头约 0.5 GB / 2.3 ms（design 2.6）。只报总 TG 会把这块藏起来。
- **两条草稿基线**：完整词表（E 已测 2.84 / 3.24 / 3.82，n-max 默认 / 4 / 8）与受限词表（未测）。
  报受限词表时必须同时报词表大小，以及同 prompt 的完整词表对照。
- `status` 不是 `ok` 一律按缺陷进 `DECISIONS.md` 未结项，不得当成"配置不对"跳过。
- 正确性判据与速度分开报：接受的 token 必须与参考 greedy 逐 token 一致；`k=1` 路径另报 logits KLD <= 1e-4。

## 3. 测量陷阱

1. 稠密地板：2.17 GB/token / 156.5 GB/s = 13.9 ms -> 单步上限 72 tok/s。任何"单步 > 72"的读数先怀疑口径
   （是不是 k>1、是不是复用了上一 pass 的结果）。156.5 是当前软件值（硬件的 70%），不是硬件上限。
2. H2D 14.17 GB/s 接近 PCIe 4.0 x8 的 15.75 GB/s（90%），余量很小；与 CPU 读并发时掉到 8.9-11.2，
   分配必须自适应。
3. MTP 默认草稿深度不是 1（约 3）。把 63.8% 当成单草稿接受率会得到错误的 TG 投影（design 2.4 / 2.5）。
4. 日志里没有周期计数，周期数只能由"生成 - 接受"推算；必须与"草稿数/周期"互相校验后再写进报告。
5. 受限词表只影响速度：验证 pass 永远是完整词表，正确性看接受的 token，而不是草稿的 logits。
6. `llama-bench` 不可用：它把 `n_ctx` 设成 `p+n`，目标口径必须用 `llama-cli` 或运行时自己的计时。
7. 电源方案与热漂移是乘性因子：跨电源态、连跑多轮后的读数不可直接比；用 r=2 看方差。
8. Git Bash 里跑 `build-vulkan/bin/*.exe` 必须先把 `/d/Scoop/apps/mingw/current/bin` 放进 PATH，
   否则子进程秒退、日志 0 字节，看起来像"配置错误"。
9. 改完代码先确认二进制含本次改动（构建时间戳），再做 A/B。
10. 外部推理进程在 GPU 上时整批数据作废；容量类失败先看 RAM 再怀疑显存。
11. 编译、repack、第二个模型进程都会扰动计时：跑数前确认 `state_check.py` 的 `quiet` 行是 `free`，
    跑数期间别人不许构建，规则见 `AGENTS.md` 第 4 节。

## 4. 目录

```
design/           qwen36-moe-runtime-design.md                     规格（只读）
bench/README.md   RESULTS.md  DECISIONS.md                        三份文档
bench/state/      state.json（任务图/资源）  quiet-*（运行锁，不进 git）
bench/prompts/    prompt10k.txt (10428 tok)  prompt20k.txt (19758 tok)
                  prompt-multidoc10k.txt (9402 tok)
bench/env/        power.txt  machine.txt                           口径依赖的机器事实
bench/tools/      state_check.py（状态一致性）  m0-trace.cpp  m0-sim.py  m0-analyze.py（路由探针）
                  membw_probe.py（带宽）  gguf_dims.py（张量账 + KV）  roofline.py（目标推导）  ssd_probe.py（SSD 能力）
bench/raw/        每次跑的 .out/.err/.bin 产物（不入库）
```

`bench/tools/` 里只有列出的这些参与新主线；其余历史脚本不在任何门禁里，不要引用。
原始产物不进 git。要新证据就重新跑，跑完把结论写进 `RESULTS.md` / `DECISIONS.md`，
不要再加第四份 md；每轮收尾同步 `AGENTS.md` 状态块与 `bench/state/state.json`。
