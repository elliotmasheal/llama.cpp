# L6: prefill 跳过驻留专家的 host 上传（2026-10-18，task-12）

> 分支 `moe-gpu-cpu-hybrid` @ `3f63d84f7`。改动文件：`src/llama-moe-cache.cpp`、`src/llama-context.cpp`。
> 结论：**正确性通过、PP 有稳定 +3.5% 提升，但 acceptance 的 pp512 >= 460 达不到**；
> 瓶颈是 8 GiB VRAM 下 bank 只能覆盖 32% 的 host 专家字节（6.0 GiB），这是实测上限。

## 1. 改动

1. `llama_moe_cache::impl::prepopulate()`：除了原有的"整层驻留"（10 层，24.4% host 专家），
   把 **LRU 池**（每 group 410 slots，共 820 slots = 1.45 GiB = 7.8% host 专家）用确定性的
   专家切片预填充：写 `slot_map`/`key_of` 并一次性上传到 bank。环境开关
   `GGML_MOE_CACHE_PREFILL=0` 可关闭（A/B 用）。
2. `llama_context::sched_copy_experts`：只加诊断计数（env `GGML_MOE_COPY_TRACE=1` 打印
   每次拷贝的 host/bank 字节），零成本 when unset。prefill 的拷贝循环本身**不需要改**：
   它已经调用 `copy_experts()`，之前 LRU 池为空所以全部走 host，预填充后驻留专家自动改走
   bank（device->device）。

依赖规则：bank 里的字节就是 host 专家字节的精确副本（加载时 `set_async` 上传、预填充前后无
其它写入），LRU 池的 `key_of`/`slot_map` 与 `plan()` 的链表状态一致，因此
`copy_experts()` 的 D2D 拷贝与原来的 host 上传**逐字节等价**；decode 的 LRU 仍会按 recency
淘汰这些 slot，行为不变。

## 2. 实测（cache 配置 `-ncmoe 41 --moe-cache-mib 6144 -ub 2048 -t 5 -fa on`，r=3-5，交叉）

| 测试 | before（PREFILL=0） | after（默认） | 变化 |
|---|---:|---:|---:|
| pp512 | 355.7 / 356.6（356.2） | 367.8 / 369.8 / 366.5 | **+3.5%** |
| tg32 | 26.18 / 26.56（26.37） | 26.30 / 26.34 / 26.26（26.30） | 噪声内（不退化） |
| pp2048 ub2048 | 580.0 | 590.3 | +1.8%（未退化） |
| static `-ncmoe 27` pp512 / tg32 | 348.2 / 19.18 | 348.2 / 19.18（无 cache，不走该路径） | 不变 |

bytes/eval（pp512，`GGML_MOE_COPY_TRACE=1` 逐层求和）：

| | host 上传 | bank D2D |
|---|---:|---:|
| before | 8187 MiB | 0 |
| after | **7298 MiB** | 889 MiB |
| 变化 | **-889 MiB（-10.8%）** | +889 MiB |

VRAM 增量：**0**（只使用本来就空闲的 LRU bank slots）；代价是加载时一次性多上传 1.45 GiB
（~0.1 s）。

## 3. 为什么 460 达不到（实测上限）

- bank 容量被 VRAM 卡死：`--moe-cache-mib 6144` 已是本机上限（7178 MiB 时 pp512 掉到 309，
  7680 MiB 时 58.6，驱动开始抖动）。VRAM 里同时还有 ~1.6 GiB 稠密/attention 权重和
  KV/compute buffer。
- 6.0 GiB bank / 18.6 GiB host 专家 = **32% 覆盖率**，这已经是"全部 VRAM 预算给专家"的上限。
  prefill 总 used 字节约 10.7 GiB/eval，32% 驻留最多省 ~3.5 GiB -> 理论 pp512 上限 ~370，
  实测 367-370，已到顶。
- 整层驻留（100% 利用）与 LRU（实测 61% 利用）的字节效率接近（55% vs 61%），把 slots 从
  LRU 挪给整层不会增加总节省；实测组合配置更差：`-ncmoe 34 + cache 4096/5120` = 56-59 t/s，
  `-ncmoe 30 + cache 5120` = 59 t/s（VRAM 溢出），`-ncmoe 38 + cache 4096` = 344.6。
- 要 460 t/s 需要 eval 从 1.39 s 降到 1.11 s，即再省 ~6 GiB/eval 的 host 字节（当前只剩
  7.1 GiB），需要 ~12 GiB bank —— 8 GiB 显卡上不可能；或者把 prefill compute 大幅降低
  （MUL_MAT_ID 在 batch 512 只有 2.0-2.4 TFLOPS，见 L5）。
- 可选的小增量（未做）：用上一次 prefill 的路由计数重选 LRU 驻留专家，可把 LRU 池利用率从
  61% 提到 ~95%，但 LRU 池只占 7.8% host 字节，估算只多 +4 t/s，不值得增加复杂度。

## 4. 正确性证据

- 文本（llama-cli，cache 配置，`-p "The quick brown fox" -n 32 --temp 0 -st`）：
  PREFILL=1 与 0 生成文本逐字节一致。
- PPL（llama-perplexity，cache 配置，`-f T12-ppl.txt -c 512 --chunks 2`，n_seq=4）：
  `Final estimate: PPL = 4.0688 +/- 0.51413`，ON 与 OFF 完全相同。
- `test-backend-ops -o MUL_MAT_ID -b CPU`：945/945 passed。
- `test-backend-ops -o MUL_MAT -b Vulkan1 -p "q4_K"`：96/96 passed（覆盖 q4_K gemm）。

## 5. 复现命令

```
# before/after（cache 配置）
GGML_MOE_CACHE_PREFILL=0 llama-bench -m <model> -dev Vulkan1 -ngl 99 -fa on -t 5 -p 512 -n 32 -r 3 -ncmoe 41 --moe-cache-mib 6144 -ub 2048
llama-bench ...（同上，不设 env）
# bytes/eval
GGML_MOE_COPY_TRACE=1 llama-bench ... -p 512 -n 0 -r 1
# 正确性
llama-cli ... -ncmoe 41 --moe-cache-mib 6144 -p "The quick brown fox" -n 32 --temp 0 -st
llama-perplexity ... -f bench/moe-baseline/T12-ppl.txt -c 512 --chunks 2
```

原始日志：`bench/moe-baseline/T12-*.{out,err}`（ab-*, tg-*, p2048-*, cli-*, ppl2-*, scan-*, c7168/c7680, final）。
