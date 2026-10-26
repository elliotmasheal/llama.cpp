# 架构审计与修复清单（2026-10-26）

- 范围：`runtime/` 的 prefill/decode 执行模型、专家存储、数值门禁与证据链；不含 fork 其他代码。
- 证据基线：工作区 `b8ce846fe + 未提交 diff`；数字引用 `bench/RESULTS.md` 当时读数，不新增性能口径。
- 性质：架构审计，不是正式 PP/TG 批次，不替代 `RESULTS.md` 的实测。
- 用户决策（2026-10-26）：三层数值门；短期允许每层 2 次 submit + 跨层 pipeline；批准 repack v2 内部布局；code 仍要求 TG 130。

## 0. 结论

组件不弱，系统未闭环。当前最大的三个风险不是单核 TFLOPS，而是：

1. prefill 的 “layer-first” 没有真正保持层内专家常驻，10.4K 诊断层循环约 194 tok/s；正确层链的组件拼盘约 1010 tok/s。
2. decode 没有执行引擎；现有 `qrt_layer` 是 prefill 形状，k=4 的专家行仍按 32 行补齐，专家 FLOPs 放大约 20 倍。
3. 数值门只在“逐层隔离 + 共享 Q8_1 打包决策”下成立；真实 40 层链、独立打包的 logits rel_L1 为 1.387e-1，门禁不能证明模型正确。

修复顺序：先修两个量级错误（X-5、X-6），再把数值门改成可证伪的三层门（X-7），然后才做 M3-2 decode 和 repack v2（M5-5）。

## 1. P0-1：prefill 重复读取整层专家

### 证据

- `qrt_ffn_open` 为整层 256 个专家分配 slot，并注释“Each expert of the layer is read once”；`qrt_ffn_group` 却每个 chunk 都重新读盘并 H2D 整层专家。见 `runtime/ffn.cpp:500`、`runtime/ffn.cpp:707`。
- `model_forward` 在层内对每个 chunk 调 `qrt_layer_run_chunk`，而 `layer_run_pass` 每次调 `qrt_ffn_group`。见 `runtime/model.cpp:510`、`runtime/layer.cpp:1089`。
- 2048-token chunk 会命中全部 256 个专家；10.4K prompt 有 6 个 chunk，等于同一层专家被搬约 6 次。
- 10.4K 诊断：40 层合计 53.6 s，约 194 tok/s；组件拼盘 10.3 s，约 1010 tok/s。见 `bench/RESULTS.md:3177`、`bench/RESULTS.md:2545`。
- 3436 tok/s / 7.1 TFLOPS 的 prefill 专家腿只计时、不累加输出；专家输出轮转在 8 个 slice，未合回 token 行。见 `runtime/README.md:293`、`runtime/kernels/moe_expert.cpp:395`。正确层链 FFN 在 10K 是 5.4 TFLOPS，见 `bench/RESULTS.md:2543`。

### 影响

- PP 1000 当前不可达；不是 kernel 差 1.5 倍，而是端到端集成差约 5 倍。
- design 6.4 的“专家 2.98 s”来自计时壳，不能作为正确 prefill 路径的预算。

### 修复（X-5）

1. `qrt_ffn_open` 为所有专家预先计算固定 `dev_off[e]` 和 `slot_bytes`，作为整层布局。
2. 增加 per-expert loaded bitmap；`qrt_ffn_group` 只读/传未加载的专家。连续未加载专家可合并为一次 H2D；已加载的后续 chunk 传 0 字节。
3. 保留整层 slot 跨 chunk 存活；chunk 只更新路由表、gather 表和 row table。
4. 增加 `uploaded_bytes` 计数并打印每层/每 chunk 的 store 字节，作为“只读一次”的可观察证据。
5. 输出必须与修改前逐字节一致；用 2048 chunk + tail 的 golden dump 对比。

验收：

- 10.4K 层循环每层 `uploaded_bytes` 收敛到该层 store 字节（约 458-560 MiB），不随 chunk 数线性增长；每个专家最多上传一次，首 chunk 后只补新路由的专家。
- 与修改前同 KV 档的 golden logits 逐字节一致。
- `--model-check` / `--chunk-check` 不回归。
- 报修改前后的 10.4K 层循环 ms 与 uploaded bytes；这是诊断读数，不冒充 M2-8。

**2026-10-26 进展（X-5，done）**：10.4K 诊断 40 层 ms 67.56 s -> 38.9 s，输出 `cmp=0`；
层 0 上传 464.0 MiB、层 39 上传 559.8 MiB，均等于该层被路由专家字节；`model-check` 64 token
PASS（logits rel_L1 6.403e-07），`chunk-check` 512/128 PASS。证据
`RESULTS.md#x-5-x-6-prefill-slot-decode-tile`。

## 2. P0-2：decode 的 32 行专家 padding

### 证据

- `rows_e = (cnt + 31) / 32 * 32`，每个被路由专家补齐到 32 行。见 `runtime/ffn.cpp:678`。
- k=4 时层 3 的 30 个专家实际 32 个 pair，却生成 960 行。见 `bench/RESULTS.md:3257`。
- 32 行时每个专家约 201 MFLOP；7 TFLOPS 下约 29 us，而权重读约 12 us，已进入计算受限区。k=4 每 pass 20-34 ms 的专家计算，超过 21 ms 周期预算。

### 影响

- 单独这一项把 TG 压在约 100 以下；compact tile 是 decode 的入场券，不是优化项。

### 修复（X-6）

1. routed-only 路径的专家行对齐从 32 改为 8（K-quant 内核的 `Q4K_TOKENS`）。
2. `max_rows`、gather 平面、accumulate grid 同步按 8 行对齐；prefill chunk 2048 的 64 行对齐行为不变。
3. 用 `--moe-batch-check` 在层 0/3/34 对 k=1/2/4/5/8 复测 ids、union、weights、MoE 输出，必须与 k=1 串行和 CPU 双精度参考一致。
4. 记录每 k 的 rows 与诊断耗时；k=4 层 3 应从 960 行降到 240 行，k=1 从 256 行降到 64 行。

验收：

- 行数下降 4 倍；输出一致；prefill 全量层链不回归。
- `ctest -L qrt` 通过；`--layer-check` 层 3 回归通过。
- 报诊断耗时；不宣称 TG 达标（decode 尚未接线）。

**2026-10-26 进展（X-6，done）**：k=4 层 3 rows 960 -> 240，k=1 256 -> 64；层 0/3/34 的
k=1/2/4/5/8 全部 PASS（ids、union、weights、routed sum vs 串行与 CPU 参考）；10.4K prefill
输出 `cmp=0`；`layer-check` 与 `ctest -L qrt` 通过。M3-2 的 grouped/indirect dispatch 仍未做。
证据 `RESULTS.md#x-5-x-6-prefill-slot-decode-tile`。

**2026-10-26 进展（M3-2，done：k=1 持久会话切片）**：新增 `--generate`，40 层只开一次，
先 32-token chunk prefill，再逐 token decode；40 层 32+8 的 8 个生成 token 与 fresh
chunked prefill 的 top1/logit 8/8 一致。wall 42 s，peak 2740.5 MiB，2 submits/layer，
1 top-k 回读/layer；默认 q8/q4 32+2 也 2/2 一致。新增 Q8_0 m=1 内核、共享 FFN slot、
动态 ffn run_tokens。仍未接 expert cache/miss split、MTP、GPU head/argmax 或跨层
pipeline；decode 1.9-2.0 s/token，不是 TG。证据 `RESULTS.md#m3-2-persistent-generate`。

## 3. P0-3：数值门是自洽，不是可证伪

### 证据

- 交付的 5.929e-7 是 `reseed on + planes device` 的逐层隔离，不是完整链。见 `bench/RESULTS.md:3051`。
- `model.cpp` 自己写的 acceptance 是 reseed 0（参考链喂自己）。见 `runtime/model.cpp:400`。
- 真实 40 层链、独立打包：logits rel_L1 1.387e-1，KLD 3.01e-2，FAIL。见 `bench/RESULTS.md:3052`。
- 4 层独立链 KLD 已到 1.8881e-4。见 `bench/RESULTS.md:3173`。
- 跨引擎 f16 对 fork：8.38e-3 KLD / 87.5% top-1；fork 自身有 1.5e-2 路径地板。见 `bench/RESULTS.md:3001`、`bench/RESULTS.md:2691`。
- Q8_1 是阶跃量化：只切换打包决策来源，层输出从 1.236e-3 掉到 2.077e-6。见 `bench/RESULTS.md:3071`。
- `--norm-check` 长期存在 `qs_bad=80/655360`，说明 CPU 参考与 shader 的 Q8_1 并非逐位一致。
- Q6_K/DeltaNet 五处语义与竞争错误曾全部通过局部检查，因为参考复制了同一约定；只有跨引擎才暴露。见 `bench/RESULTS.md:2966`。
- f16 KV 的 append 仍用 `packHalf2x16`（向零取整），而 `common.glsl` 要求 RNE。见 `runtime/kernels/attn_full_qk.comp:197`、`runtime/kernels/attn_full_qk.comp:249`、`runtime/kernels/common.glsl:48`。

### 三层门（X-7）

- **A 量化器逐位一致**：Q8_1 与 KV 行格式在全部边界、次正规、ties 上 CPU 参考与 shader 逐位一致；`qs_bad=0`。这一层不允许“共享决策”。
- **B 算术链可证伪**：完整 40 层链，参考从原始输入独立计算；在共享同一批量化字节的前提下 logits rel_L1 <= 1e-4、top-1 100%。必须用 `reseed off` 的完整链，而不是逐层隔离。
- **C 模型质量外部锚**：runtime f16/f16 对 fork f16 的 KLD <= fork 自身 1.5e-2 地板、top-1 >= 85%；并补一个高精度（无 Q8_1 激活）短切片读数，量出“量化本身”的误差。跨引擎数字只作外部 sanity，不作为 1e-4 的算术门。

注意：C 的 1.5e-2 是外部锚的上界，不是把 L-MODEL 放宽到 1.5e-2；B 仍是默认数值路径的 1e-4 判据。若 B 在独立打包下无法达到 1e-4，先修量化器/边界，不得直接改用共享打包过门。

验收：

- A：`--norm-check` 的 `qs_bad=0`；新增/复用边界测试覆盖 ties、denormal、±0、block max。
- B：`--model-check --model-check-reseed off` 完整 40 层，logits rel_L1 <= 1e-4；报告是否使用共享平面，并说明输入 norm 平面如何处理。
- C：`qgate.py compare` 对 fork f16 的 KLD/top-1；高精度切片读数；两者写入 RESULTS。
- f16 KV 改 RNE 后重跑 C；f16 是 A 侧，不能被向零取整污染。

**2026-10-26 进展（X-7，done）**：

- A：纯量化器 `norm_quant` qs_bad/ds_bad/sum_bad 全 0（含 tie/次正规/±0）；norm 路径的
  3/32768 差异是 RMS float/double 归约，单列诊断。
- B：64 token、40 层、`reseed off`、共享设备 Q8_1 平面，logits rel_L1 **4.122e-05**、
  top-1 16/16；输入 norm 平面由 CPU 计算（层 39 xnorm 3.208e-05），已在 RESULTS 标注。
- C：同 512 context、同 fork token 流、255 行；runtime 对 fork KLD **1.0867e-02**、
  top-1 **97.255%**，优于 fork 自身 `-fa on/off` 的 1.4681e-02 / 93.701%。7 个
  margin>=0.05 分叉只报（fork 自身 11 个），最差 0.959 仍是残余。
- f16 KV RNE 后 `full_attn` 1.428e-06、`layer-check`/`ctest` 通过；高精度 dense 切片
  `norm_gemm_f32` 3.989e-03。
- 残余：7 个分叉未归因；默认 q8/q4 的 dequant staging 仍向零取整；全模型 f32 激活锚未做。
  由 X-8 跟踪。证据 `RESULTS.md#x-7-three-tier-gate`、`DECISIONS.md#x-7-three-tier-gate-decisions`。

## 4. P1：cycle 模型与余量

- 21 ms 周期里 GPU 腿约 20.8 ms；任何 32 行 padding、千级 dispatch、host 往返都会破线。
- cache 不是越多越快：命中的专家也占显存带宽。design 表中 30% cache 的 TG 132 低于 25% 的 135（`design/qwen36-moe-runtime-design.md:243`）。k=4 的平衡命中率约 60% 出头，建议 M4 增加 15-22% cache 档，并按边际周期成本放置专家。
- M3-1a 门禁取预热后首轮 0.0575/0.0587，20 轮中位 0.0665（`bench/RESULTS.md:968`）；按中位 code 约 120，差 8%。code 仍是 130（用户 2026-10-26），需要用布局/内核把中位压到 0.059，不能把首轮当正式值。
- PP 9.73 s 拼盘不含 router/norm；10.3 s 层链含 router 但假设 4.4 s 专家读取完全重叠。正式 PP 必须单次、全阶段、r>=3。
- 短期允许每层 2 次 submit + 跨层 pipeline（用户 2026-10-26）；长期仍以 GPU 侧 indirect dispatch 为目标。当前 ring 固定 0、timeline semaphore 未使用，先改执行模型。

## 5. P1：存储、内核与预算

- Dense Q8_0 块主序重排已到 12.89 TFLOPS；专家 Q4_K/Q5_K/Q6_K 仍是 GGML 行主序，正确层链 FFN 5.4 TFLOPS。repack v2（已批准）做块主序、gate/up 交织、scale SoA，不改量化值。
- prefill 每专家 dispatch + barrier；改为 grouped/persistent GEMM，gate/up 融合、swiglu 融合、输出缓冲 + 单 combine。
- full attention 的 GQA 未吃干净：8 个 query head 共享一个 KV head 却重复读 KV；decode 只有 16 个 workgroup，CU 闲置。见 `runtime/kernels/attn_full_attn.comp:76`。
- router 10K 约 16 ms/层，0.7 TFLOPS；x tile 每 row block 重读。
- `qrt_cache_trace_weights` 不读 trace id，只写 `decode_tokens*top_k`（`runtime/expert_cache.cpp:699`），所以 79.9% 不是 traffic-aware；SLRU 只比 LRU 高 0.14 pp；同层 pending 会让 acquire 失败（`runtime/expert_cache.cpp:381`）。decode 前补 per-expert pending、generation+fence、真实 miss 权重、频率准入。
- repack 信任链未闭合：`weights` 解析 sha 后不用，`.bin` 头不校验，按 `64 + offset` 读（`runtime/weights.cpp:349`）；repack 头 `payload_bytes` 为 0，`n_experts` 写 +40（`tools/expert-repack/repack.cpp:285`）。M5-4 前不能把 sidecar 当可信输入。
- `qrt_vk_alloc` 只检查单次分配，不检查累计（`runtime/vk_dev.cpp:509`）；plan 只剩约 15 MiB，且没算 FFN slot、prefill 双缓冲、staging ring、host KV。
- decoupling: KV 分页与 pass mode 不兼容；10K 目标窗口够用，但 256K 质量/长上下文需要单独路径。

## 6. P2：工程与证据

- 生产与 CPU 参考同 TU（`layer.cpp`、`ffn.cpp`），routed expert 数学有 2 份 GPU、4 份 CPU 实现，submit 路径 6 处。参考移入独立模块，避免共模。
- CI 未跑 `ctest -L qrt`；`--logits-dump/--model-check/--fullattn-check` 无自动覆盖；`bench/raw` 被 ignore，golden 无法入库；X-4 仍 todo。
- 隐藏 `QRT_LAYER_STOP_PROJ` 会静默返回半层（`runtime/layer.cpp:993`）；`--layer-pass` 在 model 路径被硬编码。
- state/queue note 有过期；M2-8/M3-6 的 `done_when` 写 r=2 与 gates 的 r>=3 冲突；state_check 不查新鲜度。
- 没有 tokenizer/chat template/sampler；对当前 benchmark 不是阻塞，但离可部署 runtime 还差接口。

## 7. 修复顺序与任务映射

| 顺序 | 任务 | 内容 | 状态 |
|---:|---|---|---|
| 1 | X-5 | prefill 专家 slot 持久化，消除每 chunk 重读 | done：67.56 -> 38.9 s，golden cmp=0 |
| 2 | X-6 | decode 专家行 32 -> 8 | done：k=4 960 -> 240，全 k PASS |
| 3 | X-7 | 三层数值门 + f16 KV RNE | done：A/B pass；C KLD/top-1 pass，7 flips 残余 |
| 3b | X-8 | C 残余分叉归因、全模型 f32 锚、q8/q4 staging RNE | todo |
| 4 | M3-2 | 持久 40 层 k=1 prefill+decode；2 submits/layer | done：32+8 状态 8/8 一致；cache/MTP/TG 未做 |
| 5 | M5-5 | repack v2 内部布局 | approved, pending |

## 8. 不要做的事

- 不要用 5.929e-7 逐层隔离读数宣称完整模型正确。
- 不要用 fork 的 1.5e-2 地板把 1e-4 判据直接放宽；B 层仍是 1e-4。
- 不要把 k=1 的 32+8 短档贯通当成 TG：cache/miss split、MTP、GPU head/argmax 仍未接。
- 不要把 3436 tok/s 的 prefill 计时壳当成可交付 prefill；它没有累加。
- 不要为过门改 prompt、seed、档位或阈值；code 仍是 130。
