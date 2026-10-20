# Decode (TG) 分解与候选杠杆（2026-10-20，task-23 中期）

> 分支 `moe-gpu-cpu-hybrid` @ `09cacdfd8`。配置：cache `-ncmoe 41 --moe-cache-mib 6144 -ub 2048 -t 5 -fa on`。
> 本文件里的数字全部来自 12:23 之前的独占窗口；12:23 起机器上有两个外部 `llama-server.exe`
> （PID 800 今起、PID 20452 昨天遗留）占用 GPU，之后的测量无效，priority 3/4 的实测待独占恢复后补。

## 1. 每 token 的分解（cache 6144，n=32，GGML_SCHED_TRACE）

| 项 | 每 token | 说明 |
|---|---:|---|
| splits | 32 | 30 个部分驻留层 + 2 个杂项 |
| copy 总计 | 23-30 ms | 内含下面两项 |
| - cpy_sync | 12.3 ms | 30 次调度器 `wait_backend` 全设备 drain（每个 hw copy 前一次） |
| - h2d body | 11-18 ms | 30 次 ids 回读（各自再带一次全设备 drain）+ 上传发起 |
| gpu_submit | 6.3-10 ms | 32 次提交 |
| GPU op 时间 | ~21.7 ms | 见下表 |
| 实测 | 37.6-40.5 ms/token | 26.6 t/s（llama-bench）/ 24.7（llama-cli 口径） |

## 2. GPU 侧 op 分解（GGML_VK_PERF_LOGGER，按 32 token 归一）

| op | 调用/token | 平均 us | ms/token | 占比 |
|---|---:|---:|---:|---:|
| MUL_MAT_VEC（稠密 matvec） | 407 | 30.7 | 10.34 | **47.6%** |
| MUL_MAT_ID_VEC（MoE matvec） | 78 | 51.9 | 4.05 | 17.1% |
| MUL_MAT_ID_MUL | 39 | 37.9 | 1.48 | 6.9% |
| FLASH_ATTN_EXT | 11 | 128.3 | 1.41 | 6.5% |
| MUL_MAT_ID（MoE） | 11 | 129.6 | 1.43 | 6.2% |
| TOPK_MOE_EARLY_SOFTMAX_NORM | 42 | 27.3 | 1.15 | 5.0% |
| RMS_NORM_MUL | 139 | 5.4 | 0.75 | 3.6% |
| 其它（MULTI_ADD/SIGMOID_MUL/SILU_MUL/SSM 等） | - | - | ~1.1 | 4.8% |

关键事实：**每 token 有 407 次稠密 matvec 调用**（41 层 x ~10 次），平均 30.7 us —— 它们是
带宽受限而不是算力受限：稠密权重约 1.6-1.7 GiB 每 token 必须全部读一遍。
RX 6700S 显存带宽约 224 GB/s -> 理论下限 7.3-7.6 ms，实测 10.3 ms = **74% 效率**；
MoE 侧约 1.4 GB/token 从 bank 读，~7 ms，也是带宽受限。即 GPU 访存下限约 17-18 ms/token，
其余 ~20 ms 是每层 CPU<->GPU 往返（30 层 x 约 0.6-0.9 ms）。

## 3. Priority 2（L10 targeted wait 用在 decode）：实测负收益，已回退

| 版本 | tg32（r=3，交叉） | pp512 |
|---|---:|---:|
| 基线 | 26.36 / 26.23 | 401.2 / 401.3 |
| 目标化 wait（跳过调度器 drain + 回调同步读） | **25.62 / 25.75** | 399-401 |

原因：这些 drain 并不是"多余开销"——它们等的是上一层 MoE 输出的真实 GPU 计算（routing 依赖
上一层 hidden state），把全设备 drain 换成 targeted wait 后，回调里剩余的那次 drain 会吸收同样的
等待，总时间守恒（与 L9/L10 的结论一致）。**结论：decode 的每层同步不可再压缩。**

## 4. Priority 1（预取）评估：同专家预取与 LRU 重复，收益存疑

- 现状：LRU 已经把"本 token 用过的专家"留在 bank 里；下一 token 的 miss 是"最近没用过"的专家，
  用"本 token 的 routing"预测它们恰好预测不到。
- 更强的信号（top-(k+8) 候选）在图中确实存在：`build_moe_ffn` 先算 `probs [n_expert, n_tokens]`
  再做 `argsort_top_k`，候选 probs 可用；但要用它预取需要 (a) 每层一次额外的 ids 候选回读
  （又是一次往返）、(b) 一块 VRAM 环（cache 6144 时只剩 ~148 MiB）。
- 结论：按 Lead 决定降优先级，不实现。

## 5. Priority 3（整层数量 trade）：估算接近中性，已备好 5 分钟 A/B

当前每组 5 个整层（共 10 层）+ 410 个 LRU slot。改成每组 6 个整层（12 层）：
- 收益：少 2 层往返（约 -1.3 ms/token）+ 这 2 层不再 miss（约 -0.7 ms/token）；
- 代价：LRU slot 410 -> 154，剩余 28 层的命中率下降（粗估 +2 ms/token 的 miss 上传）。
- 净：约 0，需要实测。已加 env 门控（默认关闭、行为不变）：
  `GGML_MOE_CACHE_MAX_FULL=6`（配合 `GGML_MOE_CACHE_LRU_DIV=19` 允许 LRU 只剩 154 slot），
  代码在 `src/llama-moe-cache.cpp` 的 `max_full` / LRU 下限处。
  待跑：`GGML_MOE_CACHE_MAX_FULL=6 GGML_MOE_CACHE_LRU_DIV=19` 的 tg32/pp512/pp2048，
  对照组不设 env；一次 `llama-bench -p 512 -n 32 -r 3` + `-p 2048 -n 0 -ub 2048 -r 2`。

## 6. Priority 4（投机解码，CLI-only）：命令已备好

模型自带 MTP/nextn 头（加载时会报 `blk.40.nextn.shared_head_norm.weight` 未使用，说明文件里有）。

```
# MTP
llama-cli -m <model> -dev Vulkan1 -ngl 99 -fa on -t 5 -ncmoe 41 --moe-cache-mib 6144 -ub 2048 \
  -p "<prompt>" -n 128 --temp 0 --spec-type draft-mtp
# ngram
llama-cli ... --spec-type ngram-mod --spec-ngram-mod-n-max 16 -n 128 --temp 0
# 读 "Generation: X t/s" 作为 effective rate；接受率用 -v 的 spec 统计行（若该 build 打印）
```

## 6b. Priority 3/4 实测结果（独占恢复后补测）

Priority 3（整层数量 trade，r=3，cache 6144）。`GGML_MOE_CACHE_MAX_FULL` 为每组整层数（共 2 组）：

| 每组整层 | 整层数 | LRU slot/组 | pp512 | tg32 |
|---:|---:|---:|---:|---:|
| 6 | 12 | 154 | 406-408 | **21.9**（-17%） |
| 5（默认） | 10 | 410 | 402.4-402.6 | 26.41-26.44 |
| 4 | 8 | 666 | 390.3 | 27.01 |
| 3 | 6 | 922 | 387.1 | 27.12 |
| 2 | 4 | 1178 | 384.8 | 26.99 |
| 1 | 2 | 1434 | 375.0 | 27.10 |
| 0 | 0 | 1690 | 372.8 | 26.07 |

结论：TG 在 ~27.1 饱和（+2.6%），代价是 pp512 -4~-7%（低于 395 的验收线）；
默认的 10 整层仍是 PP/TG 综合最优。整层越多 TG 越差（12 整层时 -17%），因为 LRU 池被压缩。

Priority 4（投机解码，CLI-only，本机实测）：
- 基线（短 prompt，n=64）：Generation 24.8 t/s
- `--spec-type draft-mtp`：14.6 t/s（更慢；模型确实带 MTP 头，但在这套硬件/配置上没有收益）
- `--spec-type ngram-mod --spec-ngram-mod-n-max 16`：16.2 vs 基线 16.1 t/s（该文本无命中，无收益）
=> 投机解码不能作为有效 TG 的可行路径（至少在当前 prompt 集与配置下）。

## 6c. 最终验收表（树无改动，r=2-3）

| 指标 | 目标 | 实测 | 结论 |
|---|---:|---:|---|
| tg32（cache 6144） | >= 30 | **26.24** | 未达（天花板 26.2-27.1） |
| pp512 | >= 395 | 401.07 | 通过 |
| pp2048 ub2048 | >= 575 | 578.18 | 通过 |
| long-ctx 10k prefill（cache5120, ctx32768） | >= 405 | 412.3 | 通过 |
| long-ctx 10k decode | >= 20.6 | 21.3 | 通过 |
| 256K ctx decode（10k prompt） | >= 14.0 | 14.4 | 通过 |
| 256K ctx prefill | - | 372.5 | - |

正确性：本轮最终未保留任何引擎改动（priority 2 的目标化 wait 与 priority 3 的 env 开关都已回退），
树与 task-17 已验证的 `09cacdfd8` 一致，因此"字节一致文本/PPL/test-backend-ops"沿用已通过的结论。

## 7. 结论（当前设计下的 TG 天花板）

- 26.6 t/s 是"每层一次 CPU<->GPU 往返 + 稠密 matvec 74% 带宽效率"下的引擎上限；
  要 30 t/s 需要下列之一：
  1. 投机解码（MTP/ngram）拿 effective TG —— 只测配置，不保证基线 tg32 提升；
  2. 消除 30 次/ token 的 slot-map 往返（把 LRU 计划搬到 GPU，或按预测 ids 预生成全部 slot map
     后一次提交）——这是设计级改动，风险高；
  3. 把稠密 matvec 从 74% 提到 ~90% 带宽效率（-2.7 ms/token，需改 Vulkan GEMV kernel，超出本任务
     允许的改动范围）。
- 待独占恢复后要补的实测：priority 3 A/B、priority 4 MTP/ngram、以及验收表（tg32/pp512/pp2048/
  long-ctx/256K）。
