# DECISIONS - 判决、已证伪、开关、硬约束

与 `RESULTS.md` 冲突处以 `RESULTS.md` 的数字为准；本文件只回答"这意味着什么、还能不能做"。

## 1. 可达性判决

| 目标 | 物理边界 | 判决 |
|---|---|---|
| 短 ctx PP1000 | ub2048 时 eval 需 <= 2.05 s；活跃参数约 2.2B，需约 4.4 TFLOP/s；实测 dense 8.6 TF、id GEMM 渐近 6.04 TF | **可达**，需把 GPU compute 砍 35-40% |
| TG200（单步前向） | 每 token 至少读 1.81 GB，224 GB/s → 8.1 ms → **123 t/s** | **本机不可达**，不要用调度或 kernel 去填这个缺口 |
| TG200（effective） | 每次前向产出 2-4 token | 只能靠投机解码；内置 MTP 与 ngram 默认参数已实测为无效或负优化 |
| 单步 TG 工程上限 | 命中率 + split 间整设备等待 | 实测天花板 **26.6-27.1 t/s**；执行器变体的结构下限 20-22 |
| 256K ctx 与专家驻留 | KV 需 **5.0 GiB**（日志实测 5120.00 MiB / 262144 cells，f16、10 个注意力层 = 20 KiB/token；旧口径 5.4-6.2 GiB 作废），设备总 8176 MiB | 互斥。256K 唯一可行配置是 `-ncmoe 41 -c 262144` 无 cache（峰值 7956 MiB）；任何 bank 都溢出 |
| 4G/6G 档 | 未测 | 未知，这是当前最大的空白 |

## 2. 结构性结论（不要重新推导）

1. **resident 是前提**：pp512 336.17 对 mmap 171.31（2.06x）。mmap 下 pageable 源拷贝慢数倍，
   所以规划器在 `-lm mmap` 时强制 bank = 0。
2. **ub<=512 的 PP 是上传受限，ub2048 起是 compute 受限**（NOOP 探针同速）；PP 的余量在 kernel 不在搬运。
3. **MoE 层间依赖链**：`MoE(L-1) -> routing(L)/ids -> upload(L) -> MoE(L)`，链上没有空隙，
   调度层重叠与"目标化 wait"全部无效（省下的是等待，不是工作）。层内只剩 up/down 与 gate 的约 17% 余量。
4. **命中率来自时间局部性**，不来自驻留计数：静态 prefill 计数放置是 hit ~= coverage（22.0% 覆盖 → 22.2% 命中），
   而 warmed LRU 在 33% 覆盖下有 48-68% 命中。
5. **回读是唯一真实的可攻项**：ids 的 D2H + 整设备 drain 每层一次，并入 split 后 +3.8% tg32；
   上传路径本身（pinned 源、在途深度）早已满足，只占约 0.3 ms/token。
6. **H2D 14.3 GB/s 是硬墙**，与 chunk、队列、在途深度、缓存类型全无关；唯一余量是全双工（约 23.3 GB/s 合计），
   但 AMD 上 buffer ownership transfer 未实现，且省不掉 fence 往返。
7. **执行器/图内 hybrid 单独做都是负收益**：CPU/GPU 并发能隐藏 80.4%，但每个 CPU split 都要 device sync，
   两个分支从不重叠；绕开 `ggml_backend_sched` 是前置条件，而即便如此 ids 往返仍不可约。
8. **bank 单调变好直到设备容量上限**（6144），再大就崩；bank 还必须受单次分配上限（本机 2 GiB）约束，
   这与 1 GiB 的打包提示是两回事。
9. **长 KV 的注意力路径不是 FA**：`n_kv > n_q` 的跨块 prefill 上 FA 只有 1.07 TFLOPS，matmul 路径 4.08，
   所以判据切到 matmul；代价是 score 缓冲 +1276.7 MiB。decode（n_tokens=1）与首个 ubatch 必须仍走 FA。
10. **GPU 算命中 / CPU 算 miss 目前做不到**：`GGML_OP_MUL_MAT_ID_MASK` 只有 CPU 实现，
    Vulkan 的 `supports_op` 落到 `default: return false`，图内 hybrid 会把 GPU 分支一起拽到 CPU。
11. **`-nkvo` 在 256K 上是 x2.4-x2.7 的税，但成因是 CPU 注意力 + 专家上传，不是 KV 搬运**（2026-10-23 归因，`RESULTS.md` 0.2）：
    同一 10k prompt，32K ctx + KV 在 GPU + bank 1719 是 PP 483.6 / TG 15.4，256K + `-nkvo` 只有 PP 172-178 / TG 6.4。
    归因结果：prefill 期间 KV **仅约 210 MiB** 跨总线（全程）；每 ubatch 是 18.1 GiB 专家上传（1.85 s，bank 0 的后果）
    + 6.0-6.3 s CPU 注意力（随 n_kv 线性增长，单个 ubatch 内 5.3 -> 13.5 s）。
    原写的"瓶颈是 KV 搬运"作废。真实足迹 `used` 3.13 GiB、4 GiB 档余量约 0.9 GiB 仍然有效。
    另：`splits 40 -> 291` 里 bank 也有份（32K + KV 在 GPU + bank 0 就是 121，不是 40），不要把它单独记在 KV 放置账上。
12. **256K 口径与 MoE cache 互斥，而且触发条件是 `-nkvo` + 非零 bank，与 ctx 长短无关**（2026-10-23 两轮复现）：
    `-nkvo` + 256K 下 8176 档（planner 1547）在 layer 1、6144 档在 layer 0 抛 `llama-moe-cache.cpp:1910`；
    32K + `-nkvo` + `--moe-cache-mib 1719` 同样在 layer 1 抛同一句。只有 bank 0 能跑完。
    所以 F1 是所有 host-KV 臂的前置项，不只是 4G/6G 档。修 F1 之前不要在这条口径上谈 bank。
13. **"FA/decode 回归"是跑批器伪影，不是引擎问题**（2026-10-23 追查全过程见 `RESULTS.md` 第 0 节）：
    PowerShell `Start-Process -RedirectStandardError` 接一个带 `-v` 的子进程会把 decode 拖慢约 4.5 倍
    （同命令行 bash 重定向 = PP 671.3 / TG 21.6，PS 重定向 = TG 4.8）。由此产生的
    "TG 只有 4-5"、"悬崖说"、"bank 阶梯否证悬崖"、"`-fa off` 定位到 FA"四条推论全部作废，
    FA kernel 今天反而更快（317-323 us/层 对 445）。
    **规则**：任何"引擎回归"的结论，必须先用另一种启动方式复现一次；`run-matrix.ps1` 现在由 cmd 拥有重定向。
    今天表里所有 TG 列（3.0-4.9）都是这个回归的读数，不能用来评价 cache 或档位——
    32K 三臂里"cache 开的 3.2 反而低于 cache 关的 4.1"就是同一个症状。
14. **KV 分配大小本身有代价，与用不用得到无关**（2026-10-23，`RESULTS.md` 0.2 表 B）：
    同 bank 0、同 n_kv（约 10.4K）、同 splits、同 CPU 时间，KV 在 GPU 时 256K 的 PP 只有 239.5 而 32K 是 472.3；
    差别是每 ubatch 专家上传 3.7 s -> 7.4 s（峰值 13.1 s）。5 GiB 的常驻 KV 把专家 staging 挤出了快路径。
    这是"KV 按工作集增长"的最强证据；205 MiB 的真实工作集对 5120 MiB 的分配是 4% 的利用率。
15. **设备 compute 有一笔随 n_ctx 走、与 KV 放置无关的预留**（2026-10-23，`RESULTS.md` 0.2 表 A）：
    `-nkvo`（设备 KV = 0）下 `Vulkan1 compute` 从 16K 的 578.02 涨到 256K 的 1712.02，+1134.0 MiB；
    host 侧 compute 精确等于 `96.18 + 0.2 x KV`，256K 时 1120.18 MiB，且实测运行中会超（teardown warning）。
    所以 4 GiB 档的 bank 0 不是"容量真的不够"，而是这笔预留没被归因。
16. **F7 的状态搬运量级要按工作张量算，不是按 RS buffer 算**（2026-10-23，`RESULTS.md` 0.2 表 C/D）：
    prefill 是 `conv_state_last` 1.92 GiB/ubatch（30 层 x 64 MiB），decode 是 144.8 MiB/token。
    `CPU RS buffer size = 62.81 MiB` 只是常驻状态，不参与每次传输。
17. **K/V 应该按工作集分配，不是按 `n_ctx` 一次分配**（2026-10-23 落地并实测，`RESULTS.md` 0.3）：
    256K + 10K prompt 的真实工作集是 320 MiB（16384 cells），一次分配是 5120 MiB（4% 利用率）。
    改按工作集增长后，256K 三档 PP 447-478 / TG 12-14，`used` 3.6-5.1 GiB，RAM 峰值 -7.2 GiB；
    增长后 reserve 的 mask 自动按当前 `n_kv` 变小（设备 compute 1490 -> 708 MiB），planner 首次在 256K 给出 bank 1481 MiB。
    所以 `-nkvo` 不再是 256K 的必需项，`:1910` 那条 abort 也被绕开（KV 留在设备上）。
    限制：真把 256K 填满时会增长回 5120 MiB 并重新撞容量悬崖，这是预期行为。
18. **长跨距 score 缓冲按 query 分块是 8G 档 PP 的最大单项杠杆**（2026-10-23 落地并实测，`RESULTS.md` 0.4）：
    把一次 1280 MiB 的 score 换成若干 256 MiB 的块后，8176 档 bank 1481 -> 2879 MiB、每组 slots 406 -> 791，
    真正跨过"双环 512 slots/组"的门槛，splits 114 -> 40、hit 6.3% -> 57.6%，PP 478 -> 680、TG 13.7 -> 20.4。
    6144 档 bank 246 -> 1644（451 slots），仍差 61 slots 没进预取，PP 只 +8.6%。
    4096 档 score 从 1536 降到 256 仍不够，planner 继续判 bank 0，这一档与 score 无关。
    数值等价：KLD -0.000000、same top p 100%（`RESULTS.md` 7.1）。
    **cap 不是越小越好**：64 MiB 时 bank 最大（3969 MiB）但块数 32 抵消收益，PP 比 256 低 20 t/s。
19. **规划器的 "prefetch available" 不等于 cache 真的开了预取**（2026-10-23 两批日志对照）：
    规划器只比较总 bank 与 930 MiB 双环，cache 侧却按 group（q6_K / q4_K 各 20 层）要求每组 `2 x n_expert = 512` slots。
    0.3 的 8176 臂 bank 1481 MiB = 406 slots/组，实际是关的（`prefetch needs 512 slots per group`），
    当时文档写"预取打开"作废；0.4 的 cap 256 之后才是 791 slots/组、真的打开。
    以后引用预取状态必须以 cache 的 `prefetch needs ...` 行与 `splits` 为准，不读规划器那一句。

## 3. 已证伪清单（勿重做）

| 方向 | 实测 |
|---|---|
| BN=32 的 id tile | 一致的 BN=32 使 q4_K 5.11 → 4.89，早先 7.40 TF 是配置不一致的假象 |
| id m/l tile 上 BK_STEP=4 | pp2048 565.04 → 557.33（-1.36%），LDS 3840 → 14592 B，常驻 WG 8 → 2 |
| 单一 SPIR-V 内全 tile BK4 | batch2048 5.11 → 4.51（-12%），q6_K 5.03 → 3.79 |
| id GEMM 结构性重写 | 只占 wall 18.4%，dense 平价也只有 474 对 448.6，低于 485 门 |
| q6_K dense `_bk2`（工作区默认开） | bank 6144 pp512 -37%、bank 3584 pp2048 -5.8%、pp8192 -4.5%，两个方向都是负的 |
| 跨 token routing 预测预取 | top-8 recall 33.6-55.6%，token 回退率 100%，superset 驻留需 1.27-3.18 GB/token → 4.4-16.0 t/s |
| GPU 侧 slot plan（单独） | Option A 上限 32-35 t/s；B 非法（ids(L) 依赖 MoE(L-1)）；C 在 AMD 上不可能 |
| 层内重叠（不改 backend） | 上限 29.5-30.5 t/s：每层上传 0.50 ms 是可重叠计算 0.16 ms 的 3 倍 |
| 目标化 wait / 5 槽 ring | pp512 233-248 对 368；decode 25.6-25.8 对 26.4，时间只是被搬走 |
| 把 copy 回调挪到 wait_backend 之前 | pp512 346 → 920、TG 15 → 36，但输出是垃圾（目的缓冲复用竞争）；racy 的 797 t/s 同理 |
| 跳过 ids 回读、整张量上传（prefill） | coverage 只 179.2/256 = 70%，字节 +43%，pp512 346.8 → 294.2 |
| ubatch 级预取 | 上传只占长 prefill 的 21%，上限 +27%、现实 +10-15%，且没有 350-500 MiB 第二缓冲 |
| `VK_EXT_external_memory_host` | AMD 上 `ggml_vk_host_buffer_import_usable()` = false，强行启用 0xC0000005 崩 |
| SLRU（probation/protected 分段） | 无可测收益：tg32 21.89/21.94/22.04 |
| 缩小 bank / 改 auto-fit | 单调变慢：4096=351.5、5120=367.7、5632=373.3、6144=382.4，6400 崩到 57 |
| 每组 6 整层 | tg32 21.9（-17%），LRU 池 410 → 154 |
| 稠密 GEMV 的所有 Vulkan 旋钮 | rows/WG 1→2/4/8、大 workgroup（-17%）、MMVQ 开关：全部中性或更差；已在 DRAM 地板 97% |
| 内置 MTP `--spec-type draft-mtp` | +5% 或更慢（24.8 → 14.6；np4 45.6 → 18.1），verify 是整前向且翻倍每层专家并集 |
| ngram 投机（默认参数） | `n_max 64` = 0.69x；只有复制型文本 + `n_max 16` 才 2.17-2.25x |
| decode 侧 deferred 上传 | 只有 34% 字节可隐藏，+1.5% TG，默认关 |
| 图内 `--moe-hybrid` | TG 6.1 对 15.9，splits 42 → 122 |
| 执行器 + 滚动驻留 + M1b | 18.26 对基线 21.24 |
| 静态 prefill 计数驻留 | 命中 == 覆盖 |
| 给 CPU MoE 加线程 | 4 线程顶点，8 线程比 4 慢 38% |
| `-nopo 1`（prefill MoE 走 CPU） | pp512 345 → 169.3 |
| 长上下文 KV q8_0 | 10-19 判 PP -40%（227.9）；10-23 复测与 f16 平价（400.8 对 413.9），改用"平价可用"结论 |
| ub=1024 | 病态 66.79 t/s |
| 全 barrier 关闭（racy） | +21% TG，是 barrier 优化的上界，不是收益 |
| RDNA2 上的 cooperative matrix | 驱动侧显式只放行 RDNA3/RDNA4，cm2 是 NV 专属 |
| FA decode 的 split-K 与 block_cols 调参 | auto 28 是局部最优（438 us），4/8/14/32/42/56/224 全更慢；bc64 无差异 |
| `test-backend-ops perf FLASH_ATTN_EXT` 当 decode 基线 | 1717 us 对 in-situ 445 us（kv_view 非连续） |
| B1 lazy 专家单独做 | NO-GO：`TENSOR_READ_LAZY` 对 `ffn_*_exps` 无调用点，`--lazy-mode on` 等价 mmap；mmap+cache 5.8/5.2 对 mmap 无 cache 18.0/10.2 |
| SSD 进 decode 关键路径 | 0.81 GB/s x 206 MiB/token = 258 ms → 3.9 t/s |
| `-t 16`、`GGML_VK_MAX_NODES_PER_SUBMIT`、`DISABLE_FUSION` | 10.63 / 12.87 / 12.78 对基线 12.x |
| 双 GPU（`-ncmoe 0`） | 16.27 后被证为 SSD 污染，复测 12.38 + 304 MB OOM |

## 4. 开关表

| 开关 | 默认 | 作用 | 实测 |
|---|---|---|---|
| `LLAMA_MOE_PREFETCH` | 开 | 大 ubatch 的整层预取环（2 槽） | 关：pp2048 728 → 580、pp8192 187 → **111** |
| `LLAMA_MOE_PREFETCH_MIN_TOKENS` / `_MAX_FULL` / `_TIME` / `_DEBUG` | 512 / 4 / - | 门槛与诊断 | 诊断不影响性能 |
| `LLAMA_MOE_STAGE_IDS` | 开 | ids 回读并入产生它的 split | 关：tg32 29.08 → 27.84，h2d 2.3 → 4.3 ms/token |
| `LLAMA_FA_LONG_KV_MIB` | 1536 | 长 KV 走 matmul 判据的 score 缓冲上限，0 = 关 | 关：pp8192 187 → 293.9；10k@16K 上开 = PP +24% |
| `GGML_VK_Q6K_BK2` | 开 | AMD q6_K dense l-tile BK_STEP=2 | 两向皆负，**建议默认关**（保留前提：MUL_MAT 2275/2275） |
| `LLAMA_MOE_DEFER_UPLOAD` + `GGML_VK_DEFERRED_COPIES` | 关 | down 的 miss 走第二条 transfer 时间线 | +1.5% tg32 |
| `--moe-hybrid`（cparam） | 关，且 `llama-context.cpp` 里硬置 false | 图内 CPU 算 miss | 6.1 对 15.9，负优化 |
| `LLAMA_MOE_HYBRID_EXEC` / `_T` / `_POLL` | 关 / 1 / - | 脱离调度器的常驻 worker 执行器 | 18.26 对 21.24 |
| `LLAMA_MOE_TRACE` / `LLAMA_MOE_STATS`(_DECAY) | 未设 | 路由 trace / 衰减权重文件（给 repack 用） | 离线分析 |
| `GGML_MOE_CACHE_PREFILL` | 开 | prefill 是否走 cache | 关：pp512 356.2 → 367.8（+3.5%） |
| `LLAMA_AUTO_RESIDENT` | 开（仅 `_WIN32`） | 0 = 强制 mmap | mmap 下 PP 腰斩 |
| `LLAMA_MOE_PLAN` | 开 | 0 = 不由规划器决定 bank | - |
| `GGML_VK_VRAM_LIMIT_MIB` | 未设 | **显存档位模拟**，见 `README.md` 第 2 节 | 2026-10-23 报告值已验证；各档性能未测 |
| `-kvg` / `--kv-grow` | **-1（auto）** | K/V 初始 cells：-1 = 按工作集增长（上限 `n_ctx`，设备失败回退 host）、0 = 一次分配 `n_ctx`、>0 = 指定初始 cells | 256K+10K：PP 178.5 -> 477、TG 6.4 -> 13.8、RAM 峰值 -7.2 GiB；`0` 时 PP 239.4（`RESULTS.md` 0.3） |
| `-fac` / `--fa-chunk-mib` | **256** | 长跨距 prefill 注意力每块 score 的上限；query 维按预算切块，0 = 整段一块（旧行为）。规划器的 score 预留同步取这个值 | 8176 档：0 -> PP 458、64 -> 657、128 -> 673、**256 -> 680**、512 -> 627（`RESULTS.md` 0.4） |
| `--kv-page` / `--kv-block` / `--no-kv-page` / `--kv-page-auto`（**M1/M2b/L1 已进树**，见 `THREE-TIER.md` F10/F11） | **off**；auto 需显式 `--kv-page-auto auto`；`--kv-block` 2048 | K/V 改为固定大小块 + 块表：扩容零拷贝、峰值 = 1 块、块级设备/host 放置。auto 由 `common/fit.cpp` 判决：扣掉 score 预留后设备没有余量就开分页，`-nkvo` 下恒关 | M1（CPU 内核）逐位相同；**M2b 设备路径已打开并验收**（`RESULTS.md` 6.4）：PPL 2.1676 对 2.1676、4535 token 逐字相同、质量门 KLD -0.000000 / top-p 100%、FA 5332/5332、SET_ROWS 319/319、`ctest` 33/33。三档矩阵见 `RESULTS.md` 6.5；10K prompt 的输出一致性未过（第 6 节第 17 条） |
| `--kv-gather-mib` / `LLAMA_ARG_KV_GATHER_MIB`（F11 L1） | **256**；0 = 关 | 分页 prefill 的稠密 K/V 工作区上限：长跨距注意力先把块 gather 进工作区，再走 `-fac` 的 chunked matmul；按工作集 2 倍增长，失败退回分页 FA 并计数 | 正确性：gather 与连续布局逐字节相同（`test-kv-page`，CPU+Vulkan）。目标口径 PP（r=2 均值）：6144 档 400.3 对连续 391.3、4096 档 390.4 对 374.5；`--kv-gather-mib 0` 是 349.5 / 343.2（`RESULTS.md` 6.5） |
| `GGML_SCHED_TRACE` / `GGML_MOE_COPY_TRACE` / `GGML_VK_TIME_COPIES` / `GGML_VK_MEMORY_LOGGER` / `GGML_VK_PERF_LOGGER` | 关 | 诊断；PERF_LOGGER 拖慢 decode 2.6x | 只看占比 |
| `GGML_VK_FA_TRACE` | 未设 | 每个不同的 FA 形状打一行（path/Br/Bc/D_split/sparse/dequant_kv），F10 M0b 用它定案 | 诊断，只打几种形状 |
| `GGML_SCHED_HW_SIGNAL` / `_HW_PIPELINE` / `_HW_NOWAIT`、`GGML_MOE_CACHE_MAX_FULL` / `_LRU_DIV`、`LLAMA_ROUTING_DIAG` | 已回退，不在树里 | 见第 3 节 | - |

## 5. 代码级硬约束（改之前先读）

- `max_batch = 32`（`src/llama-moe-cache.cpp:290`）：slot_map 与 miss_mask 只覆盖 <= 32 token，
  超出就退回 host 权重路径。`ids_host`、`miss_mask` 的容量都按它分配。
- 预取环只有 2 槽、提前量 1 层（`pos % 2`），hybrid 模式下环被禁用、改用预取批次做 promotion。
- 每组整层驻留上限 5（`max_full`），并保留 >= 1/5 的 slot 给 LRU。
- 零槽固定在 `n_slots`、分配时预清 0xFF/0，miss 读它并贡献 0。
- `ggml_vk_get_compute_ctx` 会等最新 `transfer_semaphore`，所以任何在途上传都会挡住下一次 compute；
  `prefetch_semaphore` 批次不被 compute 等，这是 hybrid promotion 的前提；`prefetch_wait` 是阻塞的
  `waitSemaphores(UINT64_MAX)`。
- 预取要求：单 GPU、专家在 pinned host buft（`-lm none`）、后端暴露 proc address；任一不满足打 warning 退回。
- 容量不足目前是 **abort 而不是退化**：`GGML_ABORT` 在 `llama-moe-cache.cpp` 的 :128、:1910、:2013、:2045、:1740。
  :1910 还把正确性绑在调度器分片启发式上。**小显存路线必须先把它们改成回退**，否则 4G/6G 档一跑就崩。
- **分页 `SET_ROWS` 不得进 ROPE/RMS_NORM 的融合**（2026-10-24 定位，`RESULTS.md` 6.4）。融合把 rope 输出
  直接写进 SET_ROWS 的 dst，分页下那是 256 B 的锚而不是块：块永远为 0，`-c 512` 输出错、`-c 8192` 触发
  `DeviceLostError`。两个判据（`ggml_vk_can_fuse_rope_set_rows`、`ggml_vk_can_fuse_rms_norm_set_rows`）
  已经加了 `ggml_set_rows_get_paged` 检查。**以后任何"把 SET_ROWS 融合进前一个 kernel"的优化都要过这条门**。
- 分页 prefill 的 gather 是**纯拷贝**：块的 cell 布局与连续缓存同构（`[n_embd, BS]`，`nb[1] = row_size(n_embd)`），
  每块一条 `ggml_cpy` 即可，不要为此新增 op 或 shader。工作区与 `-fac` 的 score 缓冲是两笔预算，不能互相替代。
- `llama_context` 析构里 `wait_promotion()` 必须在后端释放之前调用。
- `llama_moe_cache_is_host_weight()` 只看 buffer 是不是 host，lazy 张量会被当成 host 专家但因非 pinned 而关掉预取。
- hybrid worker 的 `ggml_context` 必须含 `ggml_graph_overhead_custom`，否则 `register_hybrid` SIGSEGV。

## 6. 未结项

优先级按 `THREE-TIER.md` 第 4 节：P0 = 本口径跑通与解耦，P1 = 同配置最优，P2 = 跑更大模型。

1. **P0，F1**：目标工况当前跑不通。8 GiB 可见 + `256K ctx + -nkvo` + 规划器 bank 时，第一个 prefill
   ubatch 在 layer 1 抛 `llama-moe-cache.cpp:1910` 的分片不变量 abort（6144 档在 layer 0 同样复现）。
   **2026-10-23 补充**：32K + `-nkvo` + `--moe-cache-mib 1719` 同样在 layer 1 抛，所以触发条件是
   `-nkvo` + 非零 bank，不是长上下文；F1 是所有 host-KV 臂的前置项。
   修它之前，4G/6G/8G 三档只能拿 `--moe-cache-mib 0` 的数，那不是三层协同。
   修法约束：producer/lookup 的边界要在**调度分片阶段**解决。调度器留给外层的两个钩子都在 split 计算的前后
   （`ggml/src/ggml-backend.cpp:2079` 的 pre、`:2127` 的 post），签名是 `void`（`ggml/include/ggml-backend.h:386`、`:391`），
   没有错误恢复通道，也不能把 producer 挪进前一个 split；在 copy 回调里补同步等于去等一个本 split 尚未提交的节点。
   其余 4 处 `GGML_ABORT` 同批处理：`:2013` 可只跳过本次 promotion 并计数；`:2045` 要退回原始权重路径，
   运行中切换需重建图与恢复状态，不能 `return` 后继续读 bank；`:1740` 先同步重算 detached CPU 图，
   仍失败则禁用 worker 并上报可恢复错误；`:128` 属映射不变量故障，按 `:154` 的前置条件它应不可达。
   另有一处读码发现要一起修：`src/llama-moe-cache.cpp:1562` 用 group 索引 `l.ig` 去索引 `devices`，
   单 GPU 多 group 时会取错 backend，bank 拆组会扩大暴露面。
2. **P0，F7**：`-nkvo` 同时把 GDN 状态（O(n_seq_max)，实测 62.81 MiB 常数，与 n_ctx 无关）和 KV（O(n_ctx)，256K 下 5120.00 MiB）
   丢到 host，30 个 recurrent 层的 `SSM_CONV` / `GATED_DELTA_NET` 与注意力子图一起被拖到 CPU
   （`src/llama-model.cpp:2904`/`:2915`/`:2918`；`src/llama-memory-hybrid.cpp:34-65`；
   `src/llama-graph.cpp:3189-3192`；日志 `bench/raw/m-256k-bank0/vram8176_c262144_b0_r1.err:2132` 起 30 行 `dev = CPU`、
   `:2172` 的 `CPU RS 62.81 MiB`）。
   **归因已于 2026-10-23 完成**（`RESULTS.md` 0.2 表 C/D）：prefill 期间 KV 跨总线仅约 210 MiB（全程），每 ubatch 是
   `conv_state_last` 1.92 GiB（30 x 64 MiB）+ 专家 18.1 GiB；decode 是 KV 205.7 MiB/token + 状态 144.8 MiB/token。
   所以"状态解耦"要省的不是 RS buffer 本身，而是这 1.92 GiB/ubatch 的工作张量与 decode 的 144.8 MiB/token。
   实现口径不变（`--rs-offload` / `-nrs` 正交开关，字节进同一预算式），收益预期按上面的量级重估。
3. **P0，F8**：`common/fit.cpp:954-960` 从**设备** surplus 里扣长跨距 score 缓冲（默认 1536 MiB），
   不区分它这次落 GPU 还是 host。目标口径（`-nkvo`）上注意力在 CPU，所以这笔预留不占显存。
   由三档日志反解（`RESULTS.md` 0.1）：修归属的真实收益在 6144（bank 311 -> 约 1848，且首次满足 930 MiB 双环 -> 预取开）
   和 8176（1547.9 -> 约 3084），**不在 4096**：那一档摘掉预留也只有 56.0 MiB surplus，slack 规则仍然判 0。
   4 GiB 档的钥匙是另一条还没查的归属：`sched_reserve` 的设备 compute 是 1712.02 MiB，三档一致，而 32K 那批只有 772.0 MiB。
   **2026-10-23 已补测**（`RESULTS.md` 0.2 表 A，同一 `-nkvo` 放置、16K..256K 七点）：设备 compute 从 578.02 涨到 1712.02
   （+1134.0 MiB），而设备 KV 恒为 0；host 侧 compute 精确等于 `96.18 + 0.2 x KV`。所以这笔预留确实随 `n_ctx` 走，
   不是 KV 本身，且比先前假设的"约 940 MiB"更大。修法是按 `n_kv` 工作集而不是 `n_ctx` 建 reserve 图（或按 query 分块），
   归属与量级都已确认，**已由 A 落地**（`RESULTS.md` 0.3：device compute 1490.02 -> 708.02 MiB，planner 在 256K 给出 bank 1481 MiB）。
   另一条腿是按 query 分块（保持 `-ub 2048` 与完整历史范围），将 KV 驻 GPU 口径的 bank 从 3584 用回 4096；它改变数值结果，等 F6。
    **2026-10-23 已落地并验收**（判决 18、`RESULTS.md` 0.4）：F6 先行，分块数值等价（KLD 0），
    8176 档 bank 2879 MiB、预取真开、PP 680 / TG 20.4；6144 档 451 slots/组差 61 开不了预取（F3）；
    4096 档 bank 仍 0，与本条"这一档缺的不是 score"的结论一致。
4. **P1**：目标口径矩阵的 r=2 方差已补（`RESULTS.md` 0.3，三档各 2 次）；
   旧的 256K bank 阶梯（3072/2048/1024，为了绕 `:1910`）**已作废**，现在 planner 直接给 1481 MiB 且不 abort。
   `-ctk/-ctv q8_0` 对搬运税的实测也降级：prefill 不搬 KV，只剩 decode 的 205.7 MiB/token。
   FA hunk 的二分**取消**，没有引擎回归（判决 13）。
5. **P2，F9**：RAM 侧是全有或全无（`common/fit.cpp:987-995` + `src/llama-moe-cache.cpp:208-213`），
   跑更大模型要换成有界 pinned 窗口 + 批量填充；SSD 仍是冷层，且**不得**把 85% 命中率当成 decode 不等盘的保证。
6. `FLASH_ATTN_EXT -b Vulkan1` **5332/5332 已补齐**（2026-10-23，B2 落地那轮，0 FAIL）；`MUL_MAT` 2275/2275、
   `MUL_MAT_ID` Vulkan1 939/939 / CPU 945/945、`MUL_MAT_ID_MASK` CPU 4/4（Vulkan 0/0）、`ctest` 32/32。
   这条门禁关闭。
7. repack 的 KLD != 0 未修，这是混合精度分层路线（唯一能改 TG 屋顶线的选项）的前置门。
8. 独立小 draft 模型（需要同 tokenizer/vocab 且受 arch 裁剪限制）与 ReBAR/SAM 两项可行性筛查至今没做。
9. q6_K dense 的真实缺口（1.4x）还没定位到 MMQ 的哪一段；GEMV 向量化已被判无空间。
10. **F6 已关闭**（2026-10-23）：`tools/perplexity/` 从 git 历史恢复，`bench/tools/quality-gate.sh` 一条命令给出
    PPL 与 KLD（基线 PPL 5.798313、logits 基线 2.0 GB 在 `F:/LLM/kld/`），B2 用它验收过（`RESULTS.md` 7.1）。
    限制：10K 文本最多 `-c 4096`（要求 >= 2 x n_ctx tokens），质量门验证算法等价，不是 256K 数值。
11. **A 第一阶段的遗留**（`RESULTS.md` 0.3）：增长是按 2 倍翻，`reallocs` 3 次；长 prefill 时会多一次重预留。
    要收掉这两项需要 (a) 按 `n_ubatch` 预增长而不是越界才长，(b) 增长后跳过当前 eval 的重预留、推迟到下一个 eval。
    两者都只影响常数，不影响正确性，属于下一轮的小优化。
12. A 的第二阶段（设备段 + host 段 + LSE 合并）**由 F10 取代**（2026-10-23）：不用 LSE 合并，改为分块 KV + 块表间接寻址，
    所有块在同一注意力内统一寻址。F10 的容量收益来自块级放置与零拷贝扩容，decode 速度仍随 host 字节线性下降。
13. **P1，F10（分块 KV，块表间接寻址）**：设计、里程碑 M0-M4 与验收见 `THREE-TIER.md` F10，
    **M0 已完成（2026-10-23，读数 `RESULTS.md` 6.1），M1 已完成（6.2），M2b 部分完成（6.3）**。
    四个决定已定：BS=2048、先 M2a 后 M2b、M4 接受 host 块作容量档、表通过 `op_params` 传（不占 `src` 槽）。
    M0 的三条实现约束：目标工况 `FA_SCALAR` + `Bc=32`（所以 BS 必须是 32 与 256 的公倍数）；
    `use_dequant_kv` 在分页时必须显式关闭（锚的虚拟连续 `nb` 会让 `is_dense_kv_cache()` 误判）；
    `use_sparse` 目标工况实测为 0，第一版遇到就退回连续路径并计数。
    M1 的落地形态与遗留：
    (a) 块 = 一次后端分配（`init_pages` / `add_block`），块表 `uint64[2*n_layer_kv*n_blocks]` 在设备上，
        经 `ggml_backend_vk_buffer_get_device_address`（proc address）取地址；CPU 时表项是主机指针。
    (b) 锚是"虚拟"张量：`data` 指向 256 B 的占位 buffer，调度器据此判 backend，内核从不读它。
        它必须是 `data != NULL`，否则 `ggml_gallocr` 会去分配 256 MiB 的虚拟张量。
    (c) 分页下 `get_can_shift()` = false（K-shift 逐块化留到 M3），`state_write/read` 已按块拆分。
    (d) **设备 K/V 仍被守卫挡下**（`device K/V is not verified yet`）：FA 与 SET_ROWS 的分页内核都已进树，
        算子级在 CPU 与 Vulkan 上都逐位正确（`tests/test-kv-page.cpp` 两后端），但模型级 Vulkan 路径
        PPL 4.1412 对 6.0684，未定位（`RESULTS.md` 6.3 列了已排除项与下一步）。
    (e) ~~长跨距 matmul 注意力不能分页~~ **2026-10-23 已决定：每层每 ubatch gather 到稠密工作区**
        （设计见 `THREE-TIER.md` F11）。`fa_span` 里的 `!paged` 保留为兜底，gather 成功时按连续布局走
        chunked matmul。否决 `RESULTS.md` 6.3 的另两条：(a) 接受回退等于放弃 `-fac 256` 的 PP 680；
        (c) "只在 host 块 > 0 时启用"在全驻显存时同样需要，因为 FA prefill 每层要按 `n_q/Br = 256` 个行组
        重复读 K/V（`ggml-vulkan.cpp:8332-8345` 的 gqa 合并只对 `N <= 8` 生效，prefill 的 `gqa_ratio = 1`）。
        M2a（B-lite，gather 之后喂 FA）的验证目的已由 L0 与 `test-kv-page` 覆盖，不再单独做；
        它的 `ggml_cpy` 机制并入 F11 L1，但目标是 matmul 而不是 FA。
        **2026-10-24 已落地**（L0 + L1，`RESULTS.md` 6.4）：设备 K/V 打开（PPL 与连续布局相同、4535 token 逐字相同）、
        `--kv-gather-mib` 默认 256 的稠密工作区 + 逐块 `ggml_cpy` 进树、`test-kv-page` 新增 gather 用例。
        剩余是目标口径的 PP/TG 矩阵（本轮 planner 判 deficit、bank 0，测不出 gather 的收益）。
        **2026-10-24 更正**：当天早些时候写过"取 (c)、gather 暂不实现"，随后用户确认 (b)，已按 (b) 落地
        （`RESULTS.md` 6.4：paged + gather 530.48 对连续 509.94、对 paged FA 449.20）。
        (c) 保留的含义只剩**默认模式选择**：`--kv-page auto` 由 fit 预算决定初始是否开分页，**该 auto 已于同日落地**
        （本节第 16 条，默认值就是 auto）；
        它不再影响 `!paged` 这个兜底条件的作用域。
    上下文压缩（agent / 多轮）是 M5 的延伸，**已决定不做**：丢中间段会让输出偏离预期，
    三条硬约束仍留在 `THREE-TIER.md` F10 9.5 供将来参考：
    (a) hybrid 的 `seq_rm` 删不了头部（recurrent 只支持全删与尾回滚，`src/llama-memory-recurrent.cpp:184-232`），
    要新增只动 attention 段的操作；(b) server 的命中判定只看 token LCP、不看 `seq_pos_min`
    （`tools/server/server-context.cpp:3972`），压缩后必须补校验，否则静默漏历史；
    (c) 只留洞不降 `n_kv`（`src/llama-kv-cache.cpp:1494-1508`），必须借块表压缩 cell 编号，否则 PP/TG 都不省。
    压缩改数值，验收走已落地的 F6 质量门。
14. **P1，F3 现在是 6144 档的钥匙**：B2 之后这一档 bank 1644 MiB = 451 slots/组，双环要 512，
    差 61 slots（约 220 MiB）。两条候选：规划器在决定 bank 时先把双环预留算进去（优先保预取，牺牲 LRU slots），
    或者把"每组 2 x n_expert"的门槛改成分块层预取 / 允许 1 环 + 同步上传。
    **已量化收益**（`RESULTS.md` 0.4 读数 5）：同档强给 `--moe-cache-mib 1920` -> PP 490 -> 678（+38%）、
    TG 14.5 -> 19.2、hit 14.6% -> 55.2%、splits 114 -> 42、used 5161 MiB（余约 1 GiB），`status = ok`。
    所以这条不是"要不要"，而是"怎么让规划器自己给出 1920"。
15. **P1，4096 档 bank 仍是 0**：`-fac 256` 之后 score 预留只有 256 MiB，surplus 仍然不够（`RESULTS.md` 0.4 读数 6）。
    这一档要么论证"0 就是最优"，要么继续查设备 compute 的其它归属（0.2 表 A 的 +1134 MiB 已在 A 里按工作集收掉，
    剩下的固定项还没拆）。
16. **`--kv-page-auto` 已落地，默认仍 off**（2026-10-24）：`common_params.kv_page` 是三态（`-1` auto / `0` / `1`，默认 0），
    `common_moe_plan` 新增 `kv_page` 与 `kv_init_bytes`，判据是 `surplus - fa_score`（规划器给 bank 的那笔余量）
    是否为负，且要求 `offload_kqv`（`-nkvo` 下恒关）。日志两行：
    `common_moe_plan_impl: ... left after the initial K/V ... -> kv-page on/off` 与
    `K/V cache: paged|contiguous layout from the memory plan (initial ... MiB)`。
    实测判决：8176 档 +1900 MiB、6144 档 +1644 MiB -> 连续；4096 档 -147.8 MiB -> 分页。
    `LLAMA_MOE_PLAN=0` 时自动判决失效，退回连续并打 warning。
    **默认没有改成 auto**，因为第 17 条的 10K 输出差异还没定性。
17. **P0，分页 KV 在 10K prompt 上与连续布局的输出不一致**（2026-10-24 发现，未结项）。证据（同一二进制、
    `-c 16384 -ub 2048 -f prompt10k.txt -n 16 --temp 0 --moe-cache-mib 0`，`bench/raw/l1d-20261024/`）：
    - 分页（gather 开）与分页（`--kv-gather-mib 0`，走 paged FA）**逐字相同**（md5 `8a940649`，两种预填 kernel 同结果），
      说明差异不在预填的 matmul/FA 选择上；
    - 连续布局（`--no-kv-page`）是 `b5b211df`，与分页差在**第 1 个生成 token**，之后的重合；
    - 把两边的分配都固定成 12288 cells（`-kvg 12288` 对分页默认增长到 12288）仍然不同，所以不是 mask/分配尺寸；
    - 各自 3 次重复都稳定（分页 3/3 相同），排除采样或竞态噪声；
    - `-c 4096 --chunks 2` 的质量门两边一致（`RESULTS.md` 6.4 第 5 条），所以差异只在长 n_kv 路径。
    下一步：用 `GGML_VK_FA_TRACE` 比较两臂 decode 时的 FA 形状（path/Br/Bc/D_split），再对 10K 的
    prompt logits 做逐层对比（`llama-perplexity` 不能跑 `-c > 5214`，需要临时探针）。
    **在定性之前**：`--kv-page` 不做默认、F11 的"逐字相同"验收只覆盖 `n_kv <= 8192`（`RESULTS.md` 6.4 的 4535 token）。

新增行为（F7 的 offload 开关、F8 的分块开关）落地时必须进第 4 节开关表，默认值由目标口径实测决定；
`-kvg/--kv-grow` 已按此规则登记（默认 -1，实测见 `RESULTS.md` 0.3）。
