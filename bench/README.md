# bench - 口径、跑法、必报列

四份文档只放结论，不存过程：本文件 = 怎么跑，`RESULTS.md` = 数字，`DECISIONS.md` = 判决与红线，
`THREE-TIER.md` = 三层与缺口。旧读数查 git 历史；`bench/raw/` 只放本批产物（2026-10-29 已清空）。

## 1. 目标口径（唯一）

模型 `F:\LLM\llm\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`（16.96 GiB，4.25 bpw；40 层 x 256 专家，top-8）。
`UD-Q4_K_XL`（21.28 GiB）的 resident 门槛 27.6 GiB 高于本机空闲约 25.5 GiB，`-lm none` 会退 mmap
（bank 与预取被强制 0），所以 8 GiB 档的 resident 基准用 IQ4_XS。换文件等于换口径，跨文件 PP/TG 不得混算。

```
llama-cli -m F:\LLM\llm\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -dev Vulkan1 -ngl 99 -fa on `
  -t 5 -b 2048 -ub 2048 -c 262144 -ncmoe 40 -f bench\prompts\prompt10k.txt -n 64 `
  --temp 0 --no-warmup --no-display-prompt -lm none -v

bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2
```

口径五条，改一条就不是同一口径：

1. `-ncmoe 40` 显式给：专家留在 host 才有 bank。缺省时 `-ngl 99` 把专家算进 GPU 权重，规划器判
   "no device memory for a cache"、bank 0。
2. KV 按工作集增长（`-kvg -1`，默认）：256K+10K 只用 320 MiB 且驻设备，所以不加 `-nkvo`。
   `-kvg 0` 复现旧行为（一次 5120 MiB）；`-nkvo` 只用于"工作集超过显存"的对照臂。
3. 长跨距 score 按 query 分块（`-fac 256`，默认）；`-fac 0` 是整段一块的旧行为。
4. 不指定 `--moe-cache-mib` 时 bank 由规划器给（"自适应"），规则由 `--fit-plan 1`（默认）选。
5. 电源「平衡」+ AC，dGPU 上无其它推理进程，`-lm none` 要求 >= 22 GiB 空闲 RAM。

## 2. 档位 = 三个整机预算

| 维度 | 怎么造 | 语义 |
|---|---|---|
| 显存 4G/6G/8G | `--store-vram-mib 4096` / `6144` / `8176` | 整机 VRAM 总预算（dense + bank + KV + compute + staging）。store 把预算外 hold back，日志 `VRAM limited to ...`。旧 env `GGML_VK_VRAM_LIMIT_MIB` 已并入 |
| RAM 总预算 | `--store-ram-mib N`（0 = auto） | resident 权重 + `-nkvo` 的 KV + staging + mmap 页 |
| SSD 冷层 | `--store-ssd-mib N`（0 = 不参与） | 只做容量层，不进 decode |
| resident / mmap | `-lm none` / `-lm mmap` | resident 门槛 `n_bytes x 1.25 + 1 GiB`；mmap 下 pageable 源拷贝慢，bank 与预取强制 0 |

预算 = 对象占用（专家 + KV 页）+ 申报占用（dense / compute / staging）；三个预算互不挪用。
未申报即越权：`GGML_VK_MEMORY_LOGGER` 的进程总量与 `store_bytes_in(vram) + declared(vram)` 差 > 5% 判缺陷。

## 3. 必报列（缺一不可）

```
arm | vram | run | load | ctx | KV | bank | ncmoe | PP t/s | TG t/s | VRAM peak | held | used | RAM peak | reallocs | hit % | status
```

- `VRAM peak` 取每进程独占（`\GPU Process Memory(pid_<pid>*)\Dedicated Usage` 峰值），不看设备总量。
- `held` 是档位自己 hold back 的那块，也计入进程峰值；真实足迹 `used = peak - held`。
- `status` 必须 `ok`；`abort` / `vram-oom` 是缺陷，记入 `DECISIONS.md` 未结项，不许当"配置不对"。
- 动注意力 / KV / 任何数值路径：附 `bash bench/tools/quality-gate.sh` 的 `PPL | mean KLD | same top p | PASS`。
- 收益门：同档同配置配对、r=2；< 15% 的差异不验收（session 台阶未定性）。
- A/B 用的开关不进产品默认；每轮把 ΔPP / ΔTG / 生效条件记入 `RESULTS.md` 2.1 的账本。
- 表格后附一段白话总结（改了什么、快慢、为什么、下一步），术语用类比带过。
- 分页臂另报 `host blk`（落在 host 的块数）与 `gather MiB`（稠密工作区峰值），连续布局恒为 0。
- store 落地后另报 `store_sync_waits` / `store_prefetch_wasted` / `store_hits_by_tier` / `store_split`。

## 4. 测量陷阱

1. 电源是乘性因子（混合电源态与「平衡」约 x1.33），跨 session 不可比。
2. 热漂移：连跑 pp2048 会掉到约 310 t/s；用 r=2 看方差，别用单次。
3. `GGML_VK_PERF_LOGGER=1` 把 decode 拖慢约 2.6x，只能看占比；绝对时间用 `test-backend-ops perf` 隔离。
4. `llama-bench` 把 `n_ctx` 设成 `n_prompt + n_gen`，目标口径必须走 `llama-cli` 的 prompt 文件法。
5. Git Bash 的 `$!` 不是 Windows pid，按 pid 采显存会得到 0；用 `Start-Process -PassThru` 或按进程名找回。
6. 外部推理进程在 GPU 上时整批作废；`ErrorOutOfDeviceMemory` 先看 RAM 再看显存。
7. A/B 前核对二进制时间戳与 `git status`：别人的半成品会让同一命令行差 20%。
8. 跑批器能造出 4 倍差：PowerShell `-RedirectStandardError` 接 `-v` 子进程；矩阵一律用 bash 重定向。
9. `ub=1024` 是病态点（66.79 t/s），不要拿它做对照；`-ub 2048` 必须配 `-b 2048`。
10. PATH 缺 MinGW 时子进程秒退且日志 0 字节；`run-matrix.sh` 自己 export。
11. 不带 `-v` 时 INFO 日志（bank / held / hit / reallocs）一条都不出。
12. `llama-cli` 跑完停在交互提示、留下僵尸（占 RAM、锁 bin），后续臂会静默掉 mmap：stdin 接 NUL，跑完 `Get-Process llama*` 归零。
13. 顺序跑的臂会静默从 resident 掉到 mmap（日志 `mmap, ... prefetch off`）：每臂确认没有这行，否则作废。
14. 不在跑批中编辑 `run-matrix.sh`（bash 按偏移读脚本）。
15. 子进程写文件受沙箱限制（AppContainer ACE）：`llama-perplexity` 的 logits 基线只能写到仓库根下新建的目录；
    `bench/`、`F:` 等旧目录对子进程是 EACCES(13)。`quality-gate.sh` 默认写 `.gate/`（已 gitignore）。

## 5. 目录

```
bench/prompts/  prompt10k.txt（10428 tok）  prompt20k.txt（19758 tok）
bench/env/      power.txt（电源 GUID）  machine.txt（硬件事实）
bench/tools/    run-matrix.sh（矩阵）  quality-gate.sh（PPL + KLD）  探针（ssd / membw / vkgate）
bench/raw/      本批产物，不进 git；已清空，旧批次查 git 历史
```
