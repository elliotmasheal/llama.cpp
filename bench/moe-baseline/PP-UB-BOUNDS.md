# P5 STAGE 1: 长上下文 PP 的 ub/cache 边界与推荐配置（2026-10-20..21，task-29）

> 分支 `moe-gpu-cpu-hybrid` @ `307d47a27`。**STAGE 1 完成（无代码改动）。**
> 结论：配置层最大收益 **+10%（10k prompt 397.9 -> 437.8 t/s，decode 不变）**，推荐
> `-b 4096 -ub 4096 --moe-cache-mib 4608 -c 32768`；256K ctx 仍必须 ub 2048（ub 4096 溢出到 9226 MiB）。
> 到 460 t/s（+12%）的剩余 ~4% 只能来自 compute（MoE GEMM），见第 5 节。

## 1. 干净窗口与分解

- 开工 sanity：pp512 404.55 / tg32 24.11（tg 比 canonical 26.4 低约 10%，pp 正常；本报告 PP 对比用同 session 基线）。
- 分解（10k prompt = 10428 tokens，ub 2048，cache 5120，来自 task-19 的 69.6 GiB 字节计数与 trace）：
  上传 **5.4 s（21%）**，GPU compute **20.1 s（79%）**，bank D2D 可忽略。总 25.5 s = 409 t/s。
- 把 ub 提到 4096（必须同时 `-b 4096`，否则 ub 被逻辑 batch 2048 卡住）：上传字节 **71278 -> 40765 MiB
  （-43%）**，上传时间降到 ~3.1 s -> 总 ~23.1 s -> **437.8 t/s**（实测）。
- `GGML_VK_PERF_LOGGER` 在 prefill 上不可靠（只汇总到 ~77 ms 的碎片，和 3.5 s/ubatch 不符），
  与 Lead 的提醒一致；本报告只用 `GGML_SCHED_TRACE` + `GGML_MOE_COPY_TRACE` 的确定性计数。

## 2. ub 边界（ctx 32768，10k prompt；峰值来自 GGML_VK_MEMORY_LOGGER）

| ub | cache MiB | PP t/s | 峰值 device | 可行 |
|---:|---:|---:|---:|---|
| 2048 | 5120 | 397.9-406.6 | 7731 | 基线 |
| 3072 | 5120 | 433.1 | 7926 | 临界（余 250 MiB） |
| 4096 | 5120 | **100.8（溢出）** | 8110 | 否 |
| 4096 | 4864 | 438.7 | 7854 | 是（余 322 MiB） |
| 4096 | **4608** | **437.8-438.5** | **7598** | 是（余 578 MiB，**推荐**） |
| 4096 | 4096 | 435.4-443.3 | 7086 | 是 |
| 5120 | 4096 | 169.5（溢出） | - | 否 |
| 5120 | 3584 | 421.0 | - | 是（更慢） |
| 6144 | 3072 | 423.7 | - | 是（更慢） |

- **ctx 32768 的最大可行 ub = 4096**（cache <= 4608）；更大的 ub 即使压缩 cache 也更慢（上传摊薄收益
  被 ub 相关的 compute buffer 挤占与 cache 命中下降抵消）。
- **ctx 262144：ub 4096 不可行**（峰值 9226 MiB，PP 崩到 89 t/s）；256K 仍用 ub 2048（峰值 7956 MiB）。

## 3. cache 扫描（ctx 32768）

| prompt | 配置 | PP | decode@depth |
|---|---|---:|---:|
| 10k | ub2048 / cache5120（基线） | 397.9 | 20.2 |
| 10k | ub3072 / cache5120 | 433.1 | 19.9 |
| 10k | ub4096 / cache4608（推荐） | **437.8** | **20.1** |
| 10k | ub4096 / cache4096 | 435-443 | 18.0-18.8 |
| 20k | ub2048 / cache5120（基线） | 332.3 | 18.4-18.9 |
| 20k | ub4096 / cache4096 | **355.6（+7%）** | 17.5-18.0 |

cache 4608 与 5120 的差别主要影响 decode（resident 少 512 MiB -> decode 降 ~1%），PP 基本持平；
4608 是在"PP 最高 + decode 不退化 + 余量 >500 MiB"下的最好折中。

## 4. 推荐配置（长上下文，ctx <= 32K）

```
-ncmoe 41 --moe-cache-mib 4608 -b 4096 -ub 4096 -c 32768 -t 5 -fa on
```
- 10k prompt: **PP 437.8 t/s（较同 session 基线 +10%，较文档基线 412.1 +6.2%）**，decode 20.1（不变）。
- 20k prompt: PP 355.6（+7%），decode ~18。
- 256K ctx：保持 ub 2048、无 cache（`-c 262144 -b 2048 -ub 2048`，峰值 7956）。
- 短 prompt canonical（`--moe-cache-mib 6144 -ub 2048`）不受影响：pp512 399-404 / tg32 23.8-26.5（本 session 波动）。
- 配置变更不改数学：514-token prompt 上 ub2048/cache5120 与 ub4096/cache4608 生成文本逐字节一致。

## 5. STAGE 2 评估：剩余 ~4% 只能来自 compute

- ub 4096 时上传只占 ~13%（3.1 s / 23.1 s），**compute 占 ~87%**；其中 MoE GEMM 是最大项
  （先前 L7 测得 q4_K 5.11 TFLOPS vs dense 8.3-8.6，MUL_MAT_ID 在 batch>=2048 的 tile/layout 是瓶颈）。
- 本轮未做 kernel 改动（时间与风险）；给出可行方向与预期：
  1. MUL_MAT_ID 的 expert-major B-tile 复用/更大 BM：若把 5.1 提到 6.5 TF（+27%），MoE 部分省 ~15% ->
     PP +6-8% -> 可达 460-470。
  2. FA 在长深度（10k->20k -17%）：prefill FA 已按 wave64/large-head 调过，再优化空间有限。
  3. 更大的 ub 已被 VRAM 与 cache 命中限制，不可行。
- 因此"460 t/s"需要一轮 MoE GEMM kernel 工作，本轮交付的是 +10% 的配置收益与其边界。

## 6. 复现

```
# 推荐配置
llama-cli -m <model> -dev Vulkan1 -ngl 99 -fa on -t 5 -ncmoe 41 --moe-cache-mib 4608 \
  -b 4096 -ub 4096 -c 32768 -f bench/longctx/prompt10k.txt -n 32 --temp 0 --no-warmup -st
# 边界/峰值：加 GGML_VK_MEMORY_LOGGER=1，PG 日志取 "Total device:" 最大
# 上传字节：GGML_MOE_COPY_TRACE=1
```
原始日志：`bench/moe-baseline/T29-*.{out,err}`。
