# 官方 llama.cpp（scoop, Vulkan）对照基线 - LFM2.5-24B-A2B

> 目的：用本机 scoop 安装的官方 `llama.cpp-vulkan` 跑同口径 PP/TG，对照本项目 fork，验证改进幅度。
> 机器 / 模型 / 度量口径与 `bench/lfm25/REPORT.md` 相同；数据文件在 `bench/lfm25/results/`（tag = 文件名前缀 `o*` / `c1*` / `realgen*` / `srv_*`）。
> 结论：**PP 改进确凿（+18%~+62% bench，+37% 真实生成，服务器路径 2.6x）；TG 的改进只在 llama-bench 合成 decode 中出现（+20%），真实生成与 llama-server 下与官方持平**。详见第 5 节。

## 1. 对照环境

| 项 | 官方（scoop） | 本项目 fork |
|---|---|---|
| 二进制 | `D:\Scoop\apps\llama.cpp-vulkan\current\llama-bench.exe`（经 `D:\Scoop\shims`） | `E:\fastLlama\build-vulkan\bin\llama-bench.exe` |
| 版本 | `0.7.0-dev build 11763`，commit `a01ce9ffc` | `0.7.0-dev build 11750`，commit `a9d2cc8a7` |
| 编译器 / 链接 | Clang 20.1.8，动态 backend DLL（`ggml-vulkan.dll`、`ggml-cpu-haswell.dll`） | GCC 16.2.0，静态链接 |
| 设备 | `-dev Vulkan1`（RX 6700S，8176 MiB） | 同 |
| 模型 | `F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`（同一文件） | 同 |
| 专有特性 | 无 MoE 专家 cache | `--moe-cache-mib`、`--moe-hybrid`、deferred host-weight 管线等 |

注意：**官方上游基底比 fork 新 13 个 build**（11763 vs 11750），且为 Clang 构建，这两点对官方有利，因此下文 fork 的领先幅度是保守估计。

## 2. 公平性：官方也做了同等强度调参

官方同样需要手工放置专家（默认 `-ngl -1` 把 13.06 GiB 专家全放显存，8 GiB 卡放不下）。对官方做了：

1. `-ncmoe` 扫描 20/22/24/26/28/32/36/40（`-p 512 -n 32 -r 2 -t 8`）；
2. `-t` 5/6/8；`-ub` 512/1024/2048；`-fa on`；
3. `-lm`（load mode）auto / none / mmap —— **官方默认 auto 会慢 10%，必须 `-lm none`（resident）**；
4. 最终按项目口径（pp2048 ub2048、tg128）r=3 复测。

官方 `-ncmoe` 扫描（t8、ub512、lm auto、fa on）：

| ncmoe | 驻留专家层 | pp512 | tg32 |
|---:|---:|---:|---:|
| 40 | 0 | 196.7 | 15.68 |
| 36 | 4 | 210.2 | 16.80 |
| 32 | 8 | 221.2 | 18.23 |
| 28 | 12 | 249.2 | 19.41 |
| 26 | 14 | 266.1 | 20.54 |
| 24 | 16 | 285.1 | 21.72 |
| 22 | 18 | 297.3 | 23.11 |
| 20 | 20 | 302.5 | 24.99 |

其它官方调参结果（r=2）：

| 配置 | pp512 | tg32 / tg128 | 备注 |
|---|---:|---:|---|
| ncmoe20 t6 | 310.2 | 24.99 (tg32) | |
| ncmoe20 t5 | 305.5 | 25.23 (tg32) | |
| ncmoe22 t6 `-lm none` | 300.7 | 24.92 (tg32) | resident 最优 |
| ncmoe22 t6 `-lm mmap` | 274.7 | 23.42 (tg32) | 官方也怕 mmap |
| ncmoe22 t6 `-ub 2048` pp2048 | - | 684.5 (pp2048) | ub512 时仅 302.6 |
| ncmoe20 t6 `-ub 2048` pp2048 | - | **175.6（显存溢出）** | 与 fork 相同：驻留 20 层后没有 ub2048 的余量 |

**官方最优单配置**：`-ncmoe 22 -t 6 -ub 2048 -fa on -lm none`（PP/TG 折中）；**官方 TG 最优**：`-ncmoe 20 -t 6`（tg128 27.00，但长 prompt 不能开 ub2048）。

## 3. llama-bench 同口径对照（r=3，项目度量口径）

| 配置 | pp512 | pp2048（ub2048） | tg128 |
|---|---:|---:|---:|
| 官方 ncmoe22 t6 ub2048 lm none | 326.6 | **739.9** | **26.05** |
| 官方 ncmoe20 t6（TG 最优配置） | 346.9 | 175.6（溢出，不可用） | 27.00 |
| fork static ncmoe22 t6 ub2048 | 513.8 | **878.3** | 22.38 |
| fork mixed ncmoe36 + cache5120 t6 ub2048 | 530.8 | **874.8** | **31.20** |

差值（fork mixed vs 官方最优单配置）：

| 指标 | 官方 | fork mixed | 差值 |
|---|---:|---:|---:|
| pp512 | 326.6 | 530.8 | **+62.5%** |
| pp2048 ub2048 | 739.9 | 874.8 | **+18.2%** |
| tg128 | 26.05 | 31.20 | **+19.8%** |
| tg128（各自 TG 最优配置） | 27.00 | 31.20 | +15.6% |

同配置隔离（都 `-ncmoe 22 -t 6 -ub 2048 -fa on`，唯一差别是运行时）：

| 指标 | 官方 | fork static | 差值 |
|---|---:|---:|---:|
| pp512 | 326.6 | 513.8 | +57.3% |
| pp2048 | 739.9 | 878.3 | +18.7% |
| tg128 | 26.05 | 22.38 | **-14.1%** |

- PP 的 fork 优势在 **ub512（默认）时最大（+57%~+62%）**，ub2048 时收窄到 +18%：说明 fork 的 host-weight 上传/调度管线在"上传受限"的小 ubatch 上收益最大；ub2048 后 PP 已接近算力受限，剩余差距缩小。
- 纯静态（无 cache）的 decode 上，官方反而快 9-14%（官方上游新 13 个 build + Clang 构建；fork 的静态 decode 路径仍有优化空间）。
- fork 的 decode 领先完全来自 `--moe-cache-mib`。

## 4. 真实生成对照（llama-completion `-no-cnv`，temp 0，seed 1234，同一 1203 token prompt，生成 128 token）

`llama-bench` 的 tg 测试喂的是 `std::rand() % n_vocab` 随机 token（`tools/llama-bench/llama-bench.cpp` 的 `test_gen`），不是真实生成。为验证 bench 结论，用 `llama-completion -no-cnv --temp 0 -s 1234` 跑真实贪心生成（同一模型、同一 prompt、同一配置）：

| 配置 | prompt eval（1203 tok） | eval（127 run） |
|---|---:|---:|
| 官方 ncmoe22 t6 lm none | 519.3 t/s | **24.84 t/s** |
| fork static ncmoe22 t6 | 739.3 t/s | 22.61 t/s |
| fork mixed ncmoe36 + cache5120 t6 | 710.1 t/s | **24.42 t/s** |

真实生成下的差值：

| 指标 | 官方 | fork mixed | 差值 |
|---|---:|---:|---:|
| PP | 519.3 | 710.1 | **+36.7%** |
| TG | 24.84 | 24.42 | **-1.7%（在噪声内）** |

**关键发现（度量偏差）**：fork 的 MoE cache 在 llama-bench 随机 token decode 下得到 31.20 t/s（比官方 +19.8%），但真实生成只有 24.42 t/s（与官方持平）。即 **llama-bench 的 tg 口径系统性高估了 MoE-cache 配置的 decode 收益**（本模型中约 +28%）。原因：随机 token 序列的专家路由是平稳的，cache 命中率被抬高；真实生成的路由随文本演化，命中率下降。官方（无 cache）两种口径差异只有 +5%（26.05 vs 24.84）。

## 5. llama-server 端到端（1203 token prompt，生成 128 token，`-c 4096 --parallel 1 -ub 2048`）

| 配置 | prompt | decode |
|---|---:|---:|
| 官方 ncmoe22 t6 lm none | 261.4 t/s | **20.9 t/s** |
| fork mixed ncmoe36 + cache5120 t6 | **686.5 t/s** | 19.7 t/s |

- PP：fork **2.6x**（+163%），与 bench 趋势一致。
- TG：fork 略低 5.7%。两个配置在 8 GiB 卡上都贴着显存上限（bench 峰值 8143/8176 MiB），server 额外分配（更大 KV、slot 结构）会把 fork 的 cache 配置挤进溢出区（fork 比自身 bench 低约 37%，官方低约 20%）。**server 场景建议把 `--moe-cache-mib` 降到 4096~4608 留出余量，或退回 static，并单独复测。**
- 注：`--parallel` 对官方同样敏感（默认 4 时 PP 194 t/s），两边都需 `--parallel 1` 才公平。

## 6. 复盘：官方静态/纯 CPU decode 为什么更快（追加实验，2026-10-21）

### 6.1 先纠正前提：官方不是"TG 全面更快"

- 官方也有 MoE 专家 cache（`--moe-cache-mib`，`common/arg.cpp:2781`），但 **llama-bench 不暴露该参数**
  （fork 给 llama-bench 补了 19 行才暴露，`tools/llama-bench/llama-bench.cpp`）。所以 bench 对照里官方只能跑 static，
  这不是官方的最强配置，但反过来也说明我们的 bench 数字与官方 bench 数字不是同一能力集。
- 用 llama-completion 实测官方开 cache（`-ncmoe 36 --moe-cache-mib 5120 -lm none`）：
  PP 404.5 t/s / TG **16.32 t/s**；5632 MiB 时 PP 崩到 83.0（显存溢出）。**官方 cache 在本机远差于 fork mixed（710/24.42）**。

| 口径 | fork | 官方 |
|---|---|---|
| bench static tg128 | 22.38 | **26.05**（官方 +16%） |
| bench cache tg128 | **31.20** | 不可用（bench 无该参数） |
| 真实生成 TG | **24.42**（cache） | 24.84（static）/ 16.32（cache） |
| 真实生成 PP | **710-739** | 519（static）/ 404（cache） |
| llama-server TG | 19.7 | 20.9 |

官方只在"无 cache 静态 decode"一项上占优；PP 和"开 cache 的真实 decode"都是 fork 更好。

### 6.2 差距定位：纯 CPU 隔离（`-ngl 0`，无任何 Vulkan 张量）

| 口径（纯 CPU） | fork | 官方 | 差 |
|---|---:|---:|---:|
| llama-bench pp512 t6 | 347.3-349.0 | 187.1-193.1 | fork **+81%** |
| llama-bench tg128 t6 | 8.54 | 18.16（lm none） | 官方 **+113%** |
| llama-completion PP（1203 tok） | 475.2 | 306.9 | fork **+55%** |
| llama-completion TG（64 tok） | 8.35 | 16.59 | 官方 **+99%** |

纯 CPU 时没有 GPU 张量、跑不到 MoE cache/放置代码，差距仍然存在且方向相反（PP 我们快、TG 官方快）
-> 说明差距在 **batch=1 的 CPU 单 token 路径**，不在我们的 cache/放置实现。

### 6.3 线程扩展实验（决定性）

纯 CPU、`-p 512`、同一模型/编译选项集，只改 `-t`：

| threads | fork tg (t/s) | 官方 tg (t/s) |
|---:|---:|---:|
| 1 | **4.21** | **4.49** |
| 6 | 8.54 | 18.16 |
| 8 | 7.05 | 14.48 |
| 12 | 5.74 | 16.39 |

- **单线程时两者几乎相等（4.21 vs 4.49）** -> CPU 数学内核没有实质差距。
- 线程扩展：官方 t6/t1 = **4.0x**（接近线性），fork = **2.0x**，且 fork 线程越多越慢。
- 结论：fork 在 batch=1 的 CPU 多线程并行效率只有官方一半。这直接解释了静态 decode 的 -16%
  （20 个 host 层 expert 在 CPU 上算）和纯 CPU 的 -113%。这是**调度/线程/同步路径**问题，不是内核或模型结构问题。

### 6.4 候选根因与验证计划

1. **上游窗口未合**：fork base = `623c16393`（2026-10-16），官方 = `a01ce9ffc`（2026-10-20），
   中间约 80 个上游 commit。与本模型 decode 相关的候选：
   `42b70032b ggml-cpu: vectorize ggml_v_silu and ggml_v_expf (#30791)`（LFM2 用 SiLU）、
   `0878f7c30 ggml: support non-contiguous SUM (CPU/CUDA/Vulkan) (#28851)`、
   `8baabd53b llama: scale moe down projection per expert token (#30861)`、
   `33d6135cb moe-cache: SLRU (#30637)`、`84e66d032 refactor LFM2 shortconv/FFN (#30874)`、
   `eb182d85c llama: place d2t so it can stay on the GPU (#30912)`。
2. **编译器**：官方 Clang 20.1.8，fork GCC 16.2.0；两边 CPU 特性位完全相同
   （AVX2/FMA/F16C/BMI2/LLAMAFILE/OPENMP/REPACK），t1 相等也说明不是内核数学差异，
   但多线程下 OpenMP runtime（libomp vs libgomp）的屏障开销可能不同。
3. **fork 自身改动**：ggml-backend +236 行（deferred copy/第二 timeline semaphore）、llama-context +118、
   llama-graph +66。纯 CPU 下应为惰性，但仍需 profile 排除。

验证实验（按成本）：

| # | 实验 | 成本 | 回答 |
|---|---|---|---|
| 1 | 用 Clang 构建 fork HEAD，跑纯 CPU t1/t6 | ~10 min | 编译器贡献多少 |
| 2 | 临时 worktree 构建上游 `a01ce9ffc`（Vulkan OFF）同口径 | ~10 min | 上游窗口贡献多少 |
| 3 | 合入上游窗口后复测 | 30 min+ | 静态/真实 TG 能否追平 |
| 4 | profile batch=1 CPU 调度（ggml-cpu n_tasks/barrier、GGML_SCHED_TRACE） | 30 min | 并行度丢在哪 |

## 7. 结论

1. **PP（prefill）改进已验证且显著**：同配置 +18%（长 prefill）、+57~62%（默认 ub512）、真实生成 +37%、llama-server +163%。这是 fork 的 deferred host-weight 管线/调度优化的直接效果，且在"官方上游更新 13 个 build"的劣势下取得。
2. **TG（decode）改进部分成立**：
   - llama-bench 口径：fork mixed 31.20 vs 官方 26.05（+20%）。
   - 真实生成口径：fork 24.42 vs 官方 24.84（持平）；fork static 22.61（-9%）。
   - 即 cache 的真实收益约 +8%（相对 fork 自己的 static）；官方更高的解码效率抵掉了其余部分。
3. **官方静态 decode 领先的根因**（见第 6 节）：batch=1 的 CPU 多线程效率（fork 2.0x vs 官方 4.0x，单线程相等），
   叠加未合入的上游窗口与编译器差异；不是我们 cache/放置实现的问题，也不是数学内核问题。
4. **度量建议**：项目把 `tg128` 作为 TG 主口径时，应补充一个"真实 token"参考（`llama-completion -no-cnv --temp 0 -s 1234`），否则 cache 类配置会被高估。PP 口径（pp2048 ub2048）没有这个问题（随机 token 与真实 token 差 <5%）。
5. **下一步方向**（按收益排序）：
   - batch=1 CPU 多线程效率：先做 6.4 的 4 个验证实验，再决定"合上游窗口"还是"改自身调度"。
   - 真实 decode 的命中率/缺失字节：当前真实 TG 24.4 t/s，对应每 token 仍偏多的 H2D。
   - server 场景显存余量配置（cache 4608 或 static），因为 8 GiB 上 server 的额外分配会把 cache 配置挤出。
   - PP：fork 已 +18%~62%，离 PP1000 还差 14%，瓶颈是专家 GEMM 算力（与官方同构）。

## 8. 复现命令

```powershell
# 官方（scoop）PP/TG 主口径
llama-bench -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf -dev Vulkan1 -ngl 99 `
  -ncmoe 22 -fa on -t 6 -ub 2048 -lm none -p 2048 -n 0 -r 3
llama-bench -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf -dev Vulkan1 -ngl 99 `
  -ncmoe 22 -fa on -t 6 -lm none -p 512 -n 128 -r 3

# 真实生成对照（两个 build 各跑一次；官方把 -no-cnv 放在最后）
llama-completion -m F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf -dev Vulkan1 -ngl 99 `
  -ncmoe 22 -fa on -t 6 -ub 2048 -b 2048 -c 2048 -lm none --no-warmup `
  -f bench\lfm25\prompt512.txt -n 128 --temp 0 -s 1234 -no-cnv
```

## 9. 数据索引

| tag / 文件 | 内容 |
|---|---|
| `o1_*` | 官方 ncmoe 扫描（t8, ub512, lm auto） |
| `o2_*` | 官方 t / ub / load-mode 调参 |
| `o3_*` | 官方 r=3（lm auto）：pp2048 658（噪声大）、tg128 23.11、ncmoe20 tg128 24.04 |
| `o4_off_*` | 官方 r=3（lm none，最终）：pp2048 739.9、pp512 326.6、tg128 26.05、ncmoe20 tg128 27.00 |
| `o4_fork_*` | fork mixed / static r=3 对照 |
| `o5_*` | 1203 token 真实上下文 bench（fork 752.7 / 官方 572.7 pp；tg 30.46 / 26.06） |
| `realgen_*`, `realgen2_*` | llama-completion 真实生成（temp 0 与随机 seed、有/无 warmup） |
| `realgen_off_cache*` | 官方 build 开 MoE cache 的真实生成（5120 -> 16.32 t/s；5632 -> PP 83） |
| `cpu_*` | 纯 CPU 隔离实验（t=1/6/8/12，llama-bench 与 llama-completion） |
| `srv_fork_mixed*`, `srv_official*` | llama-server 端到端 |
| `server-compare.txt` | server 结果汇总 |
| `cases-cpu*.txt`, `cases-o*.txt` | 上述各轮的原始命令 |
