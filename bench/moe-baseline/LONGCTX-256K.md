# 长上下文（256K 可行性 + 10k/20k prompt）调参结果（2026-10-19，task-18）

> 分支 `moe-gpu-cpu-hybrid` @ `b10fca3bd`（== master + 文档提交）。机器：RX 6700S 8 GiB（`-dev Vulkan1`），
> Ryzen 7 7735HS，39.2 GiB RAM，resident load，全部 `-fa on`。
> 结论：**短 prompt 的最优配置不能直接用于长上下文**；长上下文有两套推荐配置（见下表），
> `-c 262144` 只在"全部专家放 host、不用 cache"时可行（峰值 7956 MiB / 8176 MiB）。

## 1. 推荐配置与实测（r>=2 交叉，ctx 与 prompt 见列头）

| 场景 | 配置 | prompt tokens | prompt-eval | decode@depth | 峰值 VRAM |
|---|---|---:|---:|---:|---:|
| 长上下文通用（10k-20k，ctx<=32K） | `-ncmoe 41 --moe-cache-mib 5120 -ub 2048 -t 5 -c 32768` | 10428 | **401.3 t/s** | **20.9 t/s** | 7752 MiB |
| 同上 | 同上 | 19758 | **333.9 t/s** | **20.2 t/s** | 7752 MiB |
| 真 256K ctx | `-ncmoe 41 -c 262144 -ub 2048 -t 5`（无 cache） | 10428 | 363.4 t/s | 14.4 t/s | **7956 MiB** |
| 同上 | 同上 | 19758 | 309.9 t/s | 13.7 t/s | **7956 MiB** |
| 省 VRAM 备选（ctx<=32K） | `-ncmoe 34 -c 32768 -ub 2048 -t 5` | 10428 | 387.1 t/s | 15.8 t/s | 5683 MiB |
| 同上 | 同上 | 19758 | 325.0 t/s | 15.5 t/s | 5683 MiB |

decode 深度 = prompt 长度（llama-cli 单轮，n=48；`[ Prompt: X t/s | Generation: Y t/s ]` 口径）。

## 2. 短 prompt 最优配置为何不能用于长上下文（关键结论）

| 配置 | ctx | 峰值 device | 结果 |
|---|---:|---:|---|
| cache `-ncmoe 41 --moe-cache-mib 6144` | 32768 | **8765 MiB** | 溢出，pp 368（短 ctx 是 386） |
| static `-ncmoe 27`（短 prompt 最优） | 32768 | **8837 MiB** | 严重溢出，pp 78.3 / tg 9.1 |
| static `-ncmoe 34` | 131072 | 8120 MiB | 溢出，pp 157 / tg 15.0 |
| static `-ncmoe 34` | 32768 | 5683 MiB | 正常（长期可行到 ~96K 量级） |
| cache 5120 | 32768 | 7752 MiB | 正常（推荐） |
| cache 5120 | 40960 | 7946 MiB | 峰值可用但性能崩（pp 255 / tg 11.0），不要用 |
| static `-ncmoe 41` | 262144 | 7956 MiB | 正常（256K 唯一可行配置） |
| cache 512 | 262144 | 8468 MiB | 溢出，不可行 |

KV 成本实测约 **21-25 KiB/token**（f16；混合架构只有部分层是全注意力），256K 需要 5.4-6.2 GiB。
8 GiB 卡上"专家驻留"与"KV"直接抢显存：短 prompt 最优把 ~7 GiB 给了专家驻留，只剩 ~0.2 GiB 给 KV，
所以 32K ctx 就溢出。长上下文必须把专家搬回 host（`-ncmoe 41`），把显存留给 KV（或在 32K ctx 下
留 5 GiB 给 cache）。

## 3. 粗扫（prompt 13.2k tokens，ctx 32768，n=48，用于选型；±3% 漂移）

| 配置 | pp | tg | 备注 |
|---|---:|---:|---|
| cache 5120, ncmoe 41, ub 2048, t5 | 394.0 | 19.9 | 最优 |
| cache 4096, ncmoe 41, ub 2048, t5 | 386.8 | 19.5 | 次之 |
| static ncmoe 34, ub 2048, t5 | 380.8 | 15.7 | 省显存 |
| static ncmoe 38, ub 2048, t5 | 368.4 | 14.8 | |
| static ncmoe 41, ub 2048, t5 | 365.2 | 14.3 | |
| static 34, ub 4096 | 380.5 | 15.8 | ub 4096 无收益 |
| cache 5120, ub 4096 | 392.5 | 20.7 | ub 4096 无收益 |
| static 34, t 8 | 380.2 | 14.9 | t 8 无收益 |
| cache 6144, ctx 32768 | 368.2 | 19.2 | 溢出（8765 MiB） |
| static 34, `-ctk q8_0 -ctv q8_0` | 227.9 | 15.4 | KV 量化可用但 PP -40%，不要用 |

KV 量化结论更新：`-ctk/-ctv q8_0` 在本 build 上**可以运行**（此前 exit 1 不复现），但 PP 从 ~380 掉到 228，
所以不采用；`q4_0` 未单独测（同类趋势，无收益场景）。

## 4. PP 与深度的关系（验证"长 prompt 上传被摊薄"的假设）

- 10k -> 20k：cache 5120 的 PP 从 401.3 降到 333.9（-17%），static 34 从 387.1 降到 325.0（-16%）。
- 原因：每个 ubatch 的专家上传量不变（按当前 ubatch 的 routing 上传），**上传并没有被摊薄**；
  PP 下降来自全注意力层的 O(depth) 成本。与短 prompt 相比（cache pp512 386、pp2048 578），
  10k prompt 的 401 t/s 只是 ub=2048 下的正常水平，不是"上传摊薄"带来的提升。
- decode@depth：cache 配置 10k/20k 深度下仍有 20.9/20.2 t/s（短 ctx tg32 26.2），下降来自 KV 读取深度；
  static 配置 15.8/15.5（短 ctx 18.7）。

## 5. 复现

```
# prompt 文件（已提交）：bench/longctx/prompt10k.txt = 10428 tokens，prompt20k.txt = 19758 tokens
# 计数：build/bin/llama-tokenize -m <model> -f bench/longctx/prompt10k.txt --ids | ...
# 推荐配置（10k/20k，ctx 32K）
llama-cli -m <model> -dev Vulkan1 -ngl 99 -fa on -t 5 -ncmoe 41 --moe-cache-mib 5120 -ub 2048 \
  -c 32768 -f bench/longctx/prompt10k.txt -n 48 --temp 0 --no-warmup --no-display-prompt -st
# 256K
llama-cli ... -ncmoe 41 -c 262144 -f bench/longctx/prompt20k.txt -n 48 ...
# 峰值 VRAM：加 GGML_VK_MEMORY_LOGGER=1，取日志里 "Total device:" 的最大值
```

原始日志：`bench/longctx/f-*.{out,err}`、`sw-*`、`pk-*`、`ctx-*`。
