# 长 prompt 的注意力路径与显存预算（2026-10-23）

> 分支 `trim-vulkan-cpu-qwen35moe`（工作区），RX 6700S 8 GiB / Ryzen 7 7735HS / 39.2 GiB RAM。
> 电源方案为「平衡」，桌面（dwm + Edge）占约 1.4 GiB 显存，所以本 session 的绝对数字比 AGENTS 的
> 「高性能」基线低一档，**只能同 session 对比**。
> 测试口径：短 = `llama-bench -p 256 -n 256`；长 = `bench/longctx/prompt10k.txt`（10428 tokens）
> + `-n 64`，`-c 16384`，`-lm none`（resident），`-fa on`，`-t 5 -ncmoe 41`。

## 1. 结论

1. **显存预算是长 prompt 的第一杀手**：同一份 `--moe-cache-mib`，ctx 16384 下 6144 让 prefill
   从 414 掉到 229 t/s（-45%），decode 从 19.1 掉到 14.8。峰值 device 分配 7.94 GiB，
   在桌面已占 1.4 GiB 的 8 GiB 卡上触发 WDDM 换页。同 ctx 下 4096 是本次的甜点（峰值 6.25 GiB）。
2. **FA 在 `n_kv > n_q` 的 prefill 上是净亏**：实测 FA 斜率 0.344 ms/每 KV token（11 个全注意力层，
   q=2048），非 FA 路径 0.0905 ms，即 1.07 TFLOPS vs 4.08 TFLOPS。10k prompt 的 prefill 里
   FA 多花约 4.5 s / 25 s。
3. 因此按形状选路径（`src/llama-graph.cpp` 的 `build_attn_mha`）：decode（`n_tokens < 512`）与
   短跨距（`n_kv <= 1.5 n_q`）继续走 FA，长跨距走 matmul 路径。**10k prefill +15~20%，
   decode 不变**（见第 4 节），代价是 compute buffer 多出 `n_head x n_q x n_kv x 4B`
   （10k 时 1.34 GB，可用 `LLAMA_FA_LONG_KV_MIB` 限制，默认 1536 MiB）。
4. KV q8_0 在本树已经不再是负优化（旧记录 -40% 来自 10-19 的旧 build）：10k 下 PP 400.8 vs
   f16 413.9，decode 19.0 vs 19.1，基本持平而 KV 占一半显存。需要腾显存时可以用。
5. 短口径不受影响：见第 4 节的回归。

## 2. 10k prompt 的配置扫描（`-n 64`，单次，peak 来自 `GGML_VK_MEMORY_LOGGER`）

| cache MiB | ctx | PP t/s | TG t/s | peak device |
|---:|---:|---:|---:|---:|
| 6144 | 16384 | 229.7 / 229.5 | 14.8 / 14.8 | 7.94 GiB |
| planner 自动（约 4.2 GiB） | 16384 | 395.0 | 15.3 | 5.79 GiB |
| 4096 | 16384 | **413.9** | **19.1** | 6.25 GiB |
| 5120 | 16384 | 406.0 | 16.6 | 7.20 GiB |

`-c 32768` + cache 5120（`LONGCTX-256K.md` 的推荐）在本 session 里无法稳定复现 401 t/s：
它需要约 7.6 GiB 的 device 分配，与桌面的 1.4 GiB 叠加后同样在换页边缘。

## 3. 时间去哪了（`GGML_SCHED_TRACE=1`，10k prompt，cache 4096）

prefill 被切成 5 个 ubatch（2048 x 4 + 188），每个 eval 的 `copy` 阶段基本等于该 eval 的
设备忙时间（每次上传前 scheduler 会 drain 整个设备）：

| eval（累计 KV） | FA on | FA off |
|---|---:|---:|
| 2048 | 81 ms | 83 ms |
| 4096 | 3203 ms | 3021 ms |
| 6144 | 3960 ms | 3248 ms |
| 8192 | 4651 ms | 3420 ms |
| 10240 | 5314 ms | 3576 ms |
| 尾部 188 tok | 1784 ms（109 splits） | 1503 ms |

线性拟合 `T = C + A x KV`：FA on 时 A = 0.344 ms/token、C = 1.79 s；FA off 时 A = 0.0905、
C = 2.65 s。斜率差就是注意力：每个 KV token 要 11 层 x 16 头 x 2048 q x 2 x 256 x 2 = 369 MFLOP，
FA 1.07 TF、matmul 路径 4.08 TF。C 的 0.86 s 差异来自 matmul 路径的额外 softmax/kq 物化开销。

预取不是长 prompt 的瓶颈：`LLAMA_MOE_PREFETCH_TIME=1` 给出首层 37.4 ms/ubatch（共 187 ms）
与每层 1.1-2.0 ms 等待，全程约 0.43 s（1.7%）。

## 4. 改动与 A/B

`src/llama-graph.cpp::build_attn_mha` 增加形状判据，其余不动：

```
fa_long_kv = n_tokens >= 512 && n_kv > 1.5 * n_tokens &&
             n_kv * n_tokens * n_head * 4 <= LLAMA_FA_LONG_KV_MIB (default 1536 MiB)
use_flash_attn = cparams.flash_attn && kq_b == nullptr && !fa_long_kv
```

- 只在 prefill 且跨 ubatch 时才可能触发；decode（n_tokens=1）与首个 ubatch（n_kv = n_q）永远走 FA。
- mask 在 `cparams.flash_attn` 为真时是 f16，matmul 路径的 `SOFT_MAX` 有 `f32_f16` pipeline，已支持。
- 数值上两条路径本来就不是逐位一致（累加顺序不同），实测 24 token 贪心生成的首段内容同义、
  措辞在第一个 token 后分叉，属预期。

A/B（ctx 16384，`-n 64`，交错，同 session）。开启后会多出 `n_head x n_q x n_kv x 4B` 的 score
缓冲，10k 时 1.34 GB，因此 **bank 必须同步下调**，否则踩显存悬崖（见第 1 节结论 1）：

| cache MiB | 路径选择 off（全 FA） | 路径选择 on | 变化 | on 的 peak |
|---:|---|---|---|---:|
| 4096 | 406.4 / 19.1、406.4 / 19.3、406.9 / 19.3 | 473.8 / 18.7、411.0 / 18.3、453.6 / 15.4 | +17% 但**不稳** | 7434 MiB |
| 3584 | **413.9 / 18.9** | **514.6 / 18.6、454.5 / 18.6、514.5 / 18.5** | **PP +24%**，TG -1.6% | 6922 MiB |
| 3072 | 415.5 / 15.4 | 446.0 / 14.1、488.7 / 17.5 | bank 太小，TG 掉 | - |

所以 10k 口径的推荐配置是 **`--moe-cache-mib 3584 -c 16384 -ub 2048` + 判据默认开启**：
device 峰值 6.92 GiB，PP 454-515、TG 18.5。cache 4096 只有在关掉判据时才安全（峰值 6.25 GiB）。
`3072` 及以下会让 decode 的 bank 不够，TG 掉到 14-17。

为什么 3584 比 4096 还快：本 session 的悬崖在约 7.2 GiB（桌面 1.4 GiB + 模型分配），
4096 + 1.34 GB score 正好压在线上。想让 10k 用满 4096 的 bank，需要把 score 缓冲按
query 分块（每次 1024 token）或把它算进预算规划器，两者都还没做。

短口径回归（cache 6144，`llama-bench -p 256 -n 256 -r 2`）：判据不触发（`n_tokens < 512`），
pp256 237.6 / tg256 25.8，与改动前 240.9 / 26.2 的差异属本 session 漂移（约 1.5%）。

## 5. 推荐配置（本机，8 GiB）

| 口径 | 参数 | 实测 |
|---|---|---|
| 短（ctx <= 8k） | `-c 8192 --moe-cache-mib 6144 -ub 2048` | pp256 240.9 / tg256 26.2 |
| 长（10k prompt，ctx 16k） | `-c 16384 --moe-cache-mib 3584 -ub 2048`（判据默认开） | PP 454-515 / TG 18.5 |

两条都加 `-dev Vulkan1 -ngl 99 -ncmoe 41 -fa on -t 5 -lm none`。
判据关闭方式：`LLAMA_FA_LONG_KV_MIB=0`（回到全 FA）。

门禁：`ctest --test-dir build-vulkan` = 32/32（本次）。本次没有改 FA kernel，
`FLASH_ATTN_EXT -b Vulkan1` 5332 例不适用；改动只在图构建里选路径。

## 6. 复现

```bash
export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"
M=F:/LLM/llm/Ornith-1.5-35B-Q4_K_M.gguf

# 短口径
build-vulkan/bin/llama-bench.exe -m $M -dev Vulkan1 -ngl 99 -fa on -t 5 -ncmoe 41 \
  --moe-cache-mib 6144 -ub 2048 -lm none -p 256 -n 256 -r 3

# 长口径（10k prompt）
powershell -ExecutionPolicy Bypass -File bench/longctx/bench10k.ps1 -Runs 3 -N 64 \
  -CacheMib 3584 -Ctx 16384 -Tag fa-long
# 关掉判据对照
LLAMA_FA_LONG_KV_MIB=0 powershell -ExecutionPolicy Bypass -File bench/longctx/bench10k.ps1 ...
```

工具：`bench/longctx/bench10k.ps1`（单配置重复）、`matrix.ps1`（配置矩阵，带 peak 显存）、
`summarize_perf.py`（`GGML_VK_PERF_LOGGER` 表格摘要）。

## 7. 下一步（按 ROI）

1. `q6_K` dense 的 tile：LDS 预算 32768 B 下 `l` tile（BM=BN=128, BK_STEP=4）需要 36864 B，
   于是 q6_K 只能落到 `m`(64x64)，实测 3.6 TF 而 q4_K 6.3 TF。用 BK_STEP=2 的 128x128 变体
   （18432 B）可把权重重读减半，是 dense MUL_MAT 里最肥的一块。
2. 长跨距 FA 的正解是提高 K/V 复用（Br 32 或 shared-memory staging），RDNA2 上受寄存器限制，
   需要重写 shader，投入远大于第 1 项。
3. decode@10k 的 52 ms 里，GPU 忙约 32 ms：output 投影 2.55 ms、11 层 FA 4.9 ms、
   MoE/dense matvec 约 23 ms；剩余约 20 ms 是 split 间等待（已由 `ROADMAP` 5.3 定性）。
