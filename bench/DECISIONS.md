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
    **2026-10-27 已修**（判决 20）：判据搬进规划器，按 group 算 1860 MiB，与 cache 侧一致。
20. **6 GiB 档的瓶颈是预算归属，不是容量**（2026-10-27，`RESULTS.md` 0.5 A）：
    `margins` 从固定 1024 改成 `max(256, min(margins, free/8))` 后，6144 档 bank 1644 -> 1908 MiB，
    跨过 per-group 预取门槛 1860 MiB，hit 14.6% -> 55.1%、splits 114 -> 42 量级，
    **PP 393.9 -> 509.5（+29.3%）、TG 12.35 -> 15.8（+28.3%）**，臂内方差 < 1%。
    8176 档中性（只多 110 MiB，预取本来已开）；4096 档见判决 21。开关 `--fit-plan`（默认 1）。
21. **小 bank 是负优化，必须设地板**（2026-10-27，`RESULTS.md` 0.5 C）：
    4096 档在 margins 缩放后得到 116 MiB 的 bank（64 slots），hit 0.0%、TG 12.4 -> 10.3（-17%），
    因为每层多了一次 lookup split 却没有任何命中。地板取 `n_expert_used * per_expert * n_layer_expert / 4`（145 MiB）。
    所以"4 GiB 档 bank 非零可达"要成立，必须先腾出 500 MiB 以上（margins 地板 256 + 未归因 compute 464 是仅有的两块）。
22. **方阵跨度的注意力留在 FA 是对的**（2026-10-27，`RESULTS.md` 0.5 读数 4）：
    `n_kv == n_q`（一个 prompt 的第一个 ubatch）换到 chunked matmul 后 total +2.5-5%、SOFT_MAX +37-72 ms、PP 更低。
    FA 在 hsk 256 上 Br=8 要重读 K/V 256 遍，但这一档每层 K/V 只有 2 MB，落在 Infinity Cache 里；
    matmul 路径的 score 缓冲有 4 遍物化往返，反而更贵。`LLAMA_FA_SPAN_SQUARE` 默认关。
23. **官方 b11763 在目标工况上没有可用路径**（2026-10-27，`RESULTS.md` 0.9）：f16 KV 在设备 + 非零 bank 是
    `ErrorDeviceLost`（5120 MiB KV 与 2986 MiB bank 争 8 GiB），`-nkvo` + 非零 bank 仍是 `:577` abort，
    q8_0 KV 能把 KV 压到 2720 MiB 但 PP 只有 147-149、TG 14.4，小 bank（512）再倒扣到 105/7.8。
    同旋钮对照（都关 cache、都 f16）是 PP 385 对 147、TG 12.2 对 6.9；而 PP 的差在 `llama-bench pp2048`
    （432 对 233）和 32K 档（386 对 149）就存在，所以主因是宿主专家上传的重叠，不是 KV/ctx。
24. **store 不预测路由、不碰序列语义**（2026-10-27，F12 §1 / F13）：预取需要"下一步要哪些专家"，
    但这是路由预测，属于调用方；store 只按 `(对象, use_step)` 排 deadline。KV 页没有"路由"这个概念，
    一旦 store 学会猜，两类对象就没法共用一套接口。同理 `seq↔cell`、shift、state 恢复留在 `kv-seq`，
    store 只管页的物理位置。
25. **子系统判据四条**（2026-10-27，F13）：生命周期不同 / 失败模式不同 / 变化频率不同 / 有不变量可脱离 GPU 单测，
    至少满足三条才拆；拆完要互相回调的，不算解耦。明确不拆：ggml 算子与 kernel、量化、CLI 解析；
    **CPU 计算池与 IO 池也不得合并**（GEMM 长任务会把搬运饿死）。
26. **调度 drain 是预取收益的上限**（2026-10-27，`RESULTS.md` 0.7）：单步 58.5 ms 里 `cpy_sync` 占 42.3 ms（72%），
    所以 `ggml-sched-plan` 必须排在 store M2 之前 - 预取能重叠多少，取决于调度器肯不肯让搬运与计算同时在飞。
27. **架构评审三结论（2026-10-27）**：方向不推翻，三处"主权分裂"要收，顺序以 `THREE-TIER.md` 第 0 节 0.0 为准。
    (a) **D 升 P0（记账先于搬运）**：统一预算式先落 `common/fit.cpp` 产出 Placement 表，store 第 2 步接管时
    整体搬入、fit 收薄壳（未结项 25）。(b) **F12 句柄状态机的三道检查（代际 / 在途 / 未消费）必须收敛成
    一个原语**，不许在解引用、淘汰、回收三条路径各写一份（本文件第 5 节、`THREE-TIER.md` F12 第 8 节）。
    (c) **同步归属单一**：分页 KV 的表更新等待归调度层（序号等待，未结项 19 / 第 24 条），store 不自建
    整设备等待。另两处验收口径同步修正：F9 加"同 RAM 压力下无窗口"第三臂（未结项 5）；
    F10 冻结等价性回归集（`tests/test-kv-page` 显式清单，未结项 13）。

## 3. 已证伪清单（勿重做）

| 方向 | 实测 |
|---|---|
| BN=32 的 id tile | 一致的 BN=32 使 q4_K 5.11 → 4.89，早先 7.40 TF 是配置不一致的假象 |
| id m/l tile 上 BK_STEP=4 | pp2048 565.04 → 557.33（-1.36%），LDS 3840 → 14592 B，常驻 WG 8 → 2 |
| 单一 SPIR-V 内全 tile BK4 | batch2048 5.11 → 4.51（-12%），q6_K 5.03 → 3.79 |
| id GEMM 结构性重写 | 只占 wall 18.4%，dense 平价也只有 474 对 448.6，低于 485 门 |
| q6_K dense `_bk2`（工作区默认开） | bank 6144 pp512 -37%、bank 3584 pp2048 -5.8%、pp8192 -4.5%，两个方向都是负的。**机制已知**（未结项 9 / `RESULTS.md` 0.8）：缺口在 mmq 的解包指令数，改 BK/tile 几何不会碰到它 |
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
| 方阵跨度（`n_kv == n_q`）改走 chunked matmul | total +2.5-5%、SOFT_MAX +37-72 ms、PP 更低：2 MB/层的 K/V 在 Infinity Cache 里，256 次重读比重物化 score 便宜（`RESULTS.md` 0.5 读数 4） |
| 按真实每专家行数把 id GEMM 的 tile 从 `bn=32` 换成 `bn=64` | 探针实测分布 `mean 64.3`、分桶 `10/10/60/91/59/25`（`RESULTS.md` 0.6 D）：tile 数 589->364 省约 0.6 ms 的 A 重读，但填充行数 18,848->23,296 多约 1.6 ms 计算，**净负**。`n_sel` 被 821 个 slot 摊薄是真，但它不是这 3.58 TF 缺口的来源 |
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
| `GGML_VK_Q6K_BK2` | **不在树里**（2026-10-27 复核：HEAD `44bd79305` 与工作区的 `git grep` 都只命中本文档与 `RESULTS.md`，源码 0 处） | 原记：AMD q6_K dense l-tile BK_STEP=2 | 原记的"两向皆负，**建议默认关**"**无法复现也无法回归**：代码里没有这个开关。`RESULTS.md` 那行"关 `_bk2` -> pp512 417.1 对 262.9"要么按 BK_STEP 的真实实现位置重测，要么标作废 |
| `LLAMA_MOE_DEFER_UPLOAD` + `GGML_VK_DEFERRED_COPIES` | 关 | down 的 miss 走第二条 transfer 时间线 | +1.5% tg32 |
| `--moe-hybrid`（cparam） | 关，且 `llama-context.cpp` 里硬置 false | 图内 CPU 算 miss | 6.1 对 15.9，负优化 |
| `LLAMA_MOE_HYBRID_EXEC` / `_T` / `_POLL` | 关 / 1 / - | 脱离调度器的常驻 worker 执行器 | 18.26 对 21.24 |
| `LLAMA_MOE_TRACE` / `LLAMA_MOE_STATS`(_DECAY) | 未设 | 路由 trace / 衰减权重文件（给 repack 用） | 离线分析 |
| `GGML_MOE_CACHE_PREFILL` | 开 | prefill 是否走 cache | 关：pp512 356.2 → 367.8（+3.5%） |
| `LLAMA_AUTO_RESIDENT` | 开（仅 `_WIN32`） | 0 = 强制 mmap | mmap 下 PP 腰斩 |
| `LLAMA_MOE_PLAN` | 开 | 0 = 不由规划器决定 bank | - |
| `--store-vram-mib` | 未设（= 自动，设备可用量） | **整机 VRAM 总预算**（dense + 专家 bank + KV + compute + staging），同时就是显存档位（4096 / 6144 / 8176）；store 把预算之外 hold back，日志 `VRAM limited to ...`。旧 env `GGML_VK_VRAM_LIMIT_MIB` **已并进来** | **CLI 已实现（2a 骨架，2026-10-27）**：参数在 init 前写进旧 hold-back 场，`--store-vram-mib 4096` 不加档位臂 = `VRAM limited to 4096.0, held 3282.8`，随后规划器 ledger 与 4096 档臂逐字相同、`status = ok`（`RESULTS.md` 0.11 读数 3）；旧 env 在迁移期继续可用（`run-matrix.sh -T`） |
| `-kvg` / `--kv-grow` | **-1（auto）** | K/V 初始 cells：-1 = 按工作集增长（上限 `n_ctx`，设备失败回退 host）、0 = 一次分配 `n_ctx`、>0 = 指定初始 cells | 256K+10K：PP 178.5 -> 477、TG 6.4 -> 13.8、RAM 峰值 -7.2 GiB；`0` 时 PP 239.4（`RESULTS.md` 0.3） |
| `-fac` / `--fa-chunk-mib` | **256** | 长跨距 prefill 注意力每块 score 的上限；query 维按预算切块，0 = 整段一块（旧行为）。规划器的 score 预留同步取这个值 | 8176 档：0 -> PP 458、64 -> 657、128 -> 673、**256 -> 680**、512 -> 627（`RESULTS.md` 0.4） |
| `--fit-plan N` / `LLAMA_FIT_PLAN` | **1** | 内存规划器规则。1 = margins 随档缩放（`max(256, min(margins, free/8))`）、score 预留跟着注意力放置（`offload_kqv`）、预取环按 expert group 判、145 MiB 小 bank 地板；0 = 旧规则（固定 1024 margin、score 无条件扣设备、930 MiB 总 bank 判据） | 6144 档 **PP +29.3% / TG +28.3%**（bank 1644 -> 1908，hit 14.6 -> 55.1%）；8176 中性；4096 加地板后与旧规则逐位相同（`RESULTS.md` 0.5） |
| `GGML_VK_CONCAT_FAST` | **开** | CONCAT 在 `dim==0` 且三方 `is_contiguous_rows`、非量化、unit 2/4/8 时走"一 WG 一行、两段平铺"的快路径，删掉每 4 B 三次 idiv；0 = 原标量路径 | CONCAT 合计 268 -> 226 ms（perf logger，-16%），端到端 -1.4% eval；`test-backend-ops -o CONCAT` 198/198、质量门 PASS（`RESULTS.md` 0.5 读数 5） |
| `LLAMA_FA_SPAN_SQUARE` | **关** | 1 = 方阵跨度（`n_kv == n_q`）也走 chunked matmul，而不是留给 FA | **负**：total +2.5-5%、SOFT_MAX +37-72 ms（`RESULTS.md` 0.5 读数 4、判决 22）；保留只为复核 |
| `--kv-page` / `--no-kv-page` / `--kv-page-auto`（**M1/M2b/L1 已进树**，见 `THREE-TIER.md` F10/F11） | **off**；auto 需显式 `--kv-page-auto auto` | K/V 改为**固定大小页** + 块表：扩容零拷贝、峰值 = 1 页、页级设备/host 放置。页固定 **128 cells = 2.5 MiB**（FA tile 整数倍），`--kv-block` **已移除**（2026-10-27）。auto 由 `common/fit.cpp` 判决：扣掉 score 预留后设备没有余量就开分页，`-nkvo` 下恒关 | M1（CPU 内核）逐位相同；**M2b 设备路径已打开并验收**（`RESULTS.md` 6.4）：PPL 2.1676 对 2.1676、4535 token 逐字相同、质量门 KLD -0.000000 / top-p 100%、FA 5332/5332、SET_ROWS 319/319、`ctest` 33/33。三档矩阵见 `RESULTS.md` 6.5（2048/256 cells 口径）；改 128 后要重跑同一套读数 |
| `--kv-gather-mib` / `LLAMA_ARG_KV_GATHER_MIB`（F11 L1） | **256**；0 = 关 | 分页 prefill 的稠密 K/V 工作区上限：长跨距注意力先把块 gather 进工作区，再走 `-fac` 的 chunked matmul；按工作集 2 倍增长，失败退回分页 FA 并计数 | 正确性：gather 与连续布局逐字节相同（`test-kv-page`，CPU+Vulkan）；10K prompt 下分页与连续布局的输出逐字相同（`RESULTS.md` 6.6）。目标口径 PP（r=2 均值）：6144 档 378.9 对连续 385.6、4096 档 360.6 对 379.5；`--kv-gather-mib 0` 是 311.6 / 307.6（`RESULTS.md` 6.5） |
| `LLAMA_MOE_IDS_TRACE` | 未设（关） | 探针：在 slot lookup 处回读 ids，按 bank slot 打印每层的路由分布（`touched/mean/max/分桶`）。**会强制一次整设备同步**，只用于测量分布 | 已测出 prefill 真实分布 `touched 252-256 mean 64.3 max 322-734`（`RESULTS.md` 0.6 D）；带它跑 PP 502.8 对不带 502.4 |
| `GGML_VK_ID_TILE_TRACE` | 未设（关） | 探针：每个 MUL_MAT_ID 打印 `m/n/k/n_as/n_sel/bn/max_tiles`，即 tile 选择器按平均行数选完的结果 | bank 路径 `n_as=821 -> n_sel=20 -> bn=32`，非 bank `n_as=256 -> n_sel=64 -> bn=64`（`RESULTS.md` 0.6 D） |
| `GGML_VK_MM_PIPELINE_TRACE` | 未设（关） | 探针：dense MUL_MAT 打印 `a=类型 m/n/k aligned 选中的 pipeline 名`，用来判断某个量化类型走没走 mmq | 定位 q6_K 缺口的关键一步：q4_K/q5_K/q6_K 分别是 `matmul_q{4,5,6}_k_q8_1_1`，无 dequant 回退（`RESULTS.md` 0.8） |
| `GGML_SCHED_TRACE` / `GGML_MOE_COPY_TRACE` / `GGML_VK_TIME_COPIES` / `GGML_VK_MEMORY_LOGGER` / `GGML_VK_PERF_LOGGER` | 关 | 诊断；PERF_LOGGER 拖慢 decode 端到端约 2.6x（这是**端到端**系数，不是每个 kernel 的系数：大 prefill kernel 的读数接近真值） | 看份额要用 `bench/tools/summarize_perf.py <log> --agg`；按精确 key 求和会漏掉所有带形状后缀的 GEMM/FA 行（见未结项 21） |
| `--store-ram-mib` / `--store-ssd-mib` / `--store-threads` | 0 / 0（= auto，SSD 不参与）/ 2 | 整机 RAM 总预算（resident 权重 + `-nkvo` KV + staging + mmap 页估算）、SSD 冷层容量、共享池大小。**子系统本体没有总开关**，A/B 只能靠这几个参数复现旧行为 | **CLI 已接线（2a 骨架）**：参数进 store、auto 由 plan ledger/free 推（ram 的 auto 分母 = free - 4 GiB 留量常数，默认值待 M1 复核）；ssd 0 = 关时全路径零 SSD；`--store-threads` 只在 2c 的 io 池创建时生效（`RESULTS.md` 0.11） |
| `--store-exec` / `LLAMA_ARG_STORE_EXEC` | **0** | MoE 整层预取的传输路由：0 = cache 自己的路径；1 = 经 store-io 路由（原语不变，等价验证）；2 = 1 + 层内等待只等本批序号（0.6 E 序号管线接主路径；等待时机的收益本体在 2g/M2）。promotion/回填路径两种模式下都保持 raw drain（发布 slot map 前必须清空传输队列） | **A/B/A 已测（2026-10-27，`RESULTS.md` 0.12）**：mode 1/2 与 0 的决策/hit/内存/行为行逐字相同、PP 差 < 0.2%，震荡在 session 台阶内；默认 0，M2 读数后决定 |
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
- **分页扩容必须用 `llama_kv_cells::grow()`，不能用 `resize()`**（2026-10-24，`RESULTS.md` 6.6）：
  `resize()` 会 `reset()` 全部 cell 的 pos/ext/seq，等于丢掉已缓存 K/V 的簿记；连续布局的 `realloc_cache`
  也是 `grow()`。以后新增的扩容点（state 恢复、块回收后再分配）都按这条写。
- `llama_context` 析构里 `wait_promotion()` 必须在后端释放之前调用。
- `llama_moe_cache_is_host_weight()` 只看 buffer 是不是 host，lazy 张量会被当成 host 专家但因非 pinned 而关掉预取。
- hybrid worker 的 `ggml_context` 必须含 `ggml_graph_overhead_custom`，否则 `register_hybrid` SIGSEGV。
- `llama-store` 落地后：句柄一律是"槽号 + 代际号"，不得在调用方缓存裸设备地址；搬运期间槽被 pin；
  池内线程禁止阻塞等待（要等就在调用线程上等序号）；SSD 对单步路径的同步取由代价表直接拒绝并计数；
  三个预算是**整机总量**、互不挪用，手设值是硬顶，超了降级不 abort；
  专家与 KV 页必须经 store，其他分配必须申报（未申报即越权，差 > 5% 判缺陷）；
  加载路径（`lazy_read` / mmap / resident）**不得改动**，store 只登记专家区间。
  `store_declare` 是**硬边界**：debug 对账失败直接 abort（开发期不变量，不是运行期容量条件），
  release 只告警并计数 `store_undeclared_bytes`；不得把它降级成"建议值"。
  store 内部三单元（slab / io / policy）不得合回一个文件，policy 必须能热换而 slab 不动。
  这些是"违反即错"，不做 warning 容忍（`THREE-TIER.md` F12 第 7-8 节、第 12 节）。
- **F12 句柄判忙只有一个原语**（2026-10-27 架构评审，判决 27）：代际未变 + 无 in-flight + 无未消费句柄的
  三道检查在解引用、淘汰、回收三条路径上共用同一个实现（一次持锁或一个序号范围判定）；slab 的无 GPU 单测
  必须证明淘汰走的就是解析用的那个判定。分三处各写一份即缺陷。
- **预算解算只有一个所有者**（判决 27）：K/V、score、compute、dense、RS、bank、staging 的预算只在一处解算
  （当前为 `common/fit.cpp` 的统一式子，store 落地后归 store）；其余模块只读表或申报，不得各自再算。
  迁移规则：fit -> store 时式子整体搬入，`fit.cpp` 收成申报薄壳；过渡期两处并存即第二套算术，按本条处理。

## 6. 未结项

优先级按 `THREE-TIER.md` 第 4 节：P0 = 本口径跑通与解耦，P1 = 同配置最优，P2 = 跑更大模型。

**2026-10-27 复核后的执行顺序**（编号不重排，避免打乱本节内部"第 N 条"的交叉引用；来源
`.research/review-20261027/README.md` 第 4/5 节）：

| 序 | 事项 | 本节条目 |
|---|---|---|
| 1 | ~~补 256K + `GGML_SCHED_TRACE=1` 的握手实测~~ **已补**（`RESULTS.md` 0.7：`cpy_sync` 42.3 ms = 单步 72%，气泡约 20 ms）；仍缺 `session id` / `n_kv` 两列与台阶定性 | 第 20 条 |
| 2 | ~~修 PERF_LOGGER 的 GEMM/FA 行~~ 已证为脚本误判并修好（`--agg`）；改用新排名：**prefill 口径 MUL_MAT 39.1% + MUL_MAT_ID 31.7%** | 第 21 条（已关闭） |
| 3 | 给单 GPU 调度加**目的缓冲深度**（`n_copies = 2`，事件随之存在），并把 ids 可见性协议从"靠 drain"改成靠 event / 序号等待；这一项归 `ggml-sched-plan`（`THREE-TIER.md` F13 S2），报告要多报 `skp_splits` / `skp_copies` / `skp_cpy_sync` | 第 19 条、第 24 条 |
| 4 | ~~`:1910` 改回退 + 计数~~ **已关闭**（2026-10-27，`RESULTS.md` 0.6 A/B）；F1 剩下的 4 处 abort 转成第 1 条的尾巴 | 第 1 条（F1） |
| 5 | 算子轴改靶：id GEMM 的 tile 宽度已按真实分布算成净负（第 3 节），现在排第一的是 **dense MUL_MAT 39.1%** 里的 q6_K 段（`m=8192 n=2048 k=2048` 单 key 4.7%，3.6 TF 对 q4_K 6.3-6.6 TF） | 第 9 条 |
| 6 | 464 MiB 未归因的设备 compute（4096 档 bank 要非零的唯一剩余来源） | 第 15 条 |
| 7 | 补 PP 份额账缺的 281 ms，否则不能用"差 964 ms"排 ROI | 第 22 条 |
| 8 | Vulkan `MUL_MAT_ID_MASK`（TG 破 27 t/s 平台的唯一路径） | 本节没有，见 `THREE-TIER.md` F4 |
| 9 | 统一预算式（D 项）升 P0：一个式子产出 Placement 表，先落 `common/fit.cpp` 不动搬运（记账先于搬运） | 第 25 条（新增，判决 27） |

`--fit-plan 1`（判决 20）已经把原顺序里"预算归属"那一批（F3 与 F8 的归属腿）做完并实测，所以本表不再列；
F7 状态解耦、F9 有界 pinned 窗口仍按 `THREE-TIER.md` 第 4 节的位置。

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
   **2026-10-27：`:1910` 这一处已关闭**（`RESULTS.md` 0.6 A/B）——`-nkvo` 且不在整层预取路径上时图上不再发
   slot lookup（`src/llama-graph.cpp:2708`），ids 与 lookup 同侧时改成整层 ring 上传 + 计数（`-v` 里的
   `llama_moe_cache: split fallbacks ...`）。原必崩配置（32K + `-nkvo` + `--moe-cache-mib 1719`）现在 `exit=0`
   且输出连贯；目标工况 `split fallbacks = 0`，PP/TG/hit/bank 与 0.5 一致，`ctest` 33/33。
   **F1 仍未结**：其余 4 处 `GGML_ABORT`（`:128`、`:2013`、`:2045`、`:1740`）与 `:1562` 的 group 索引错。
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
   **2026-10-27 评审补充验收口径**（判决 27）：原 "(c) pp2048 >= resident 的 70%" 只证明"窗口有用"，
   不证明"窗口是好方案"（RAM 压力测试本身会改变 CPU 侧，pinned pool 会挤掉 page cache）。
   加第三臂：同 RAM 压力下"无窗口、纯 mmap 退化"的对照，结论以内比较为准，70% 留作下界
   （`THREE-TIER.md` F9 验收已同步改）。
6. `FLASH_ATTN_EXT -b Vulkan1` **5332/5332 已补齐**（2026-10-23，B2 落地那轮，0 FAIL）；`MUL_MAT` 2275/2275、
   `MUL_MAT_ID` Vulkan1 939/939 / CPU 945/945、`MUL_MAT_ID_MASK` CPU 4/4（Vulkan 0/0）、`ctest` 32/32。
   这条门禁关闭。
7. repack 的 KLD != 0 未修，这是混合精度分层路线（唯一能改 TG 屋顶线的选项）的前置门。
8. 独立小 draft 模型（需要同 tokenizer/vocab 且受 arch 裁剪限制）与 ReBAR/SAM 两项可行性筛查至今没做。
9. **已定位（2026-10-27，`RESULTS.md` 0.8）：q6_K dense 慢 1.5x 的原因是 mmq 内循环的解包算术**，不是字节、
   不是 tile、不是走错 kernel。三条独立证据：q8_0（1.78x 字节）与 f16（3.6x 字节）都跑 6.0-6.3 TF，
   与 q4_K 同档；`GGML_VK_MM_PIPELINE_TRACE` 显示 q4_K/q5_K/q6_K 各自落在 `matmul_q{4,5,6}_k_q8_1_1`，
   没有 dequant 回退；掉速只发生在"高位平面 + 双级 scale"的 q5_K/q6_K 上，而完全不解包的 f16 matmul
   反而比 q4_K 慢（6.04 对 6.40 TF），说明限制在解包而不是乘加。
   in-situ 份额：`MUL_MAT q6_K m=8192 n=2048 k=2048` 每 eval 56 次 = 单 ubatch 的 22.6%，改 q4_K 约 **PP +7.5%**。
   两条出路各自的门：(a) 共享专家重排成 q4_K——被未结项 7（repack KLD != 0）阻塞，且共享专家每 token 都用，
   精度代价比稀疏专家高；(b) 改成每 tile 解一次进 LDS——是 kernel 改写，第 3 节那三条调几何的实验对它无效
   （现在有了机制解释：它们没减少解包指令数）。
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
    **2026-10-27 评审补充（等价性回归集，判决 27）**：分页改动的回归风险在"看似无关的 opt-in 路径"绕过守卫
    （SET_ROWS 融合门发生过一次，本文件第 5 节）。把 `THREE-TIER.md` F10 第 7 节的自动退回条件与第 5.8 节
    的两条 opt-in 冲突（`use_dequant_kv` 分页必须显式关、`use_sparse` 退回连续）冻结成 `tests/test-kv-page`
    的显式用例清单，每个退回条件一条；动 FA / SET_ROWS / 图复用必跑该清单，不靠文档记忆。
14. **P1，F3 已关闭**（2026-10-27）：6144 档缺的 61 slots/组不再靠强给 `--moe-cache-mib 1920`，
    而是由 `--fit-plan 1` 的 margins 缩放自己给到 1908 MiB（判决 20、`RESULTS.md` 0.5 A）。
    实测方向与 0.4 的前瞻臂一致（PP 393.9 -> 509.5、TG 12.35 -> 15.8，同 session 配对）。
15. **P1，4096 档 bank 仍是 0，且 116 MiB 量级已证负**（2026-10-27 更新）：`-fac 256` 之后 score 预留 256 MiB，
    margins 缩放再给 512 MiB，surplus 只到 372 MiB，减去 256 MiB slack 得 116 MiB = 64 slots -> hit 0%、TG -17%
    （`RESULTS.md` 0.5 C），所以加了 145 MiB 地板把它归零。要继续推进这一档，必须再腾出 500 MiB 以上：
    候选只剩 margins 地板（256 MiB）与未归因的 464 MiB compute（`RESULTS.md` 0.4 与 0.3 同 n_kv 的 720 MiB 差值）。
16. **`--kv-page-auto` 已落地，默认仍 off**（2026-10-24）：`common_params.kv_page` 是三态（`-1` auto / `0` / `1`，默认 0），
    `common_moe_plan` 新增 `kv_page` 与 `kv_init_bytes`，判据是 `surplus - fa_score`（规划器给 bank 的那笔余量）
    是否为负，且要求 `offload_kqv`（`-nkvo` 下恒关）。日志两行：
    `common_moe_plan_impl: ... left after the initial K/V ... -> kv-page on/off` 与
    `K/V cache: paged|contiguous layout from the memory plan (initial ... MiB)`。
    实测判决：8176 档 +1900 MiB、6144 档 +1644 MiB -> 连续；4096 档 -147.8 MiB -> 分页。
    `LLAMA_MOE_PLAN=0` 时自动判决失效，退回连续并打 warning。
    **默认没有改成 auto**，因为第 17 条的 10K 输出差异还没定性。
17. **已解决（2026-10-24）：分页 KV 在 10K prompt 上与连续布局的输出不一致**。根因是
    `llama_kv_cache::ensure_blocks()` 扩容后调 `v_cells[s].resize()`，而 `llama_kv_cells::resize()` 会 `reset()`
    全部 cell（`src/llama-kv-cells.h:69-76`），丢掉了已缓存 K/V 的 pos/seq；连续布局的 `realloc_cache` 一直用
    `grow()`。改成 `grow()` 之后，默认 `--kv-block 2048` 与 `--kv-block 256` 在 10K prompt 上与连续布局
    **逐字相同**（`RESULTS.md` 6.6 的 md5 表）。修复前的 paged 读数（`bench/raw/l1d-20261024/paged2/`）作废。
18. **作废（2026-10-27）：`--kv-block` 已移除，页固定 128 cells = 2.5 MiB**，本条"块 > 2048 在 gather 下
    与连续布局不同"不再适用。128 是 FA tile（64/128）的整数倍，分页内核不用改；M3 只需重跑旧读数
    （`test-kv-page` / 10K 逐字 / `FA 5332` / `SET_ROWS 319`）。
    原记录（2026-10-24）：4096 与 8192 只在 `--kv-gather-mib` 非零时不同，`--kv-gather-mib 0` 时与连续布局逐字相同，
    问题在 `build_paged_dense` 对大于 2048 的块的假设上（未定位）。默认 2048 不受影响。

19. **P1，去掉每层整设备 drain**（`.research/review-20261027/README.md` 2.1，2026-10-27 复核）：目标工况是单 GPU，
    `pipeline_parallel` 恒 false 的唯一有效阻挡是 `cparams.moe_cache_size == 0`（`src/llama-context.cpp:600-606`，
    注释即 "not supported by the MoE cache"）；多设备那条其实是满足的（`-ncmoe 41` 让专家权重留在 host，
    日志 `llama_context: backend_ptrs.size() = 2`）。于是 `n_copies = 1`、events 全 NULL
    （`ggml/src/ggml-backend.cpp:2174`、`:2207-2211`），每次等待都落到 `ggml_backend_synchronize`
    （`:1909-1914`、`:1920-1925`）。这与第 3 节那五项"调度重排"不是一件事：那五项在 drain 模型内搬时间，本条要去掉 drain。
    两条路径：给单 backend 建 fence ring，或把 ids 回读改成显式 timeline 等待。
    **必须一起改的三处协议**：post-split 回调只在 `!parallel` 时注册（`src/llama-context.cpp:836-840`）；
    ids staging 的可见性依赖"调度器在下一个 copy 回调前等过 backend"（`src/llama-moe-cache.cpp:1524-1525`）；
    未 staging 的分支是无条件 `get_async + synchronize`（`:1928-1929`）。
    收益量级只有 32K 口径的 `cpy_sync 12.3 ms/token`（`RESULTS.md:448`），256K 口径未测，要在第 20 条之后重估。
    **2026-10-27 后端原语已落地**（`RESULTS.md` 0.6 E，F12 的第一步）：`ggml_backend_vk_prefetch_submit_seq` 返回批次
    序号、`ggml_backend_vk_prefetch_wait_seq(seq)` 只等到该序号，命令缓冲按序号排队回收，多批可同时在飞；
    实测等小批 2.64 ms 时 1 GiB 那批仍在飞，而等最新是 76.51 ms。**仍未结**：调度器还没有用它——
    `ggml_backend_sched` 在 `n_copies == 1` 时仍走整设备 fence（`:1909-1914`、`:1920-1925`），
    MoE cache 的 ids 回读也仍带一次 `ggml_backend_synchronize`（`src/llama-moe-cache.cpp:1928-1929`）。
    把这两处换成序号等待，才算真正去掉每层 drain。
    **2026-10-27 目标口径实测后改靶**（`RESULTS.md` 0.7）：光换等待粒度不会赢——单队列上宿主等 split i 时
    队列里只有 0..i-1，event 等待与整设备 fence 等的是同一个点（这也解释了第 3 节"目标化 wait / 5 槽 ring"
    为何只是搬时间）。真正的缺口是**目的缓冲深度**：`n_copies == 1` 时下一条 split 的拷贝必须等旧的用完，
    而"把回调挪到等待之前"能到 pp512 920 却输出垃圾，正是同一件事的反证。所以本条要做的是：
    单 GPU 也给 `n_copies = 2`（事件随之被创建），并把 ids 的可见性协议从"靠 drain"改成靠 event /
    序号等待（`post_split` 现只在 `!parallel` 注册，`src/llama-context.cpp:836-840`）。
    收益上限按 0.7 是约 **20 ms/token（TG 17.1 -> 约 26）**，风险是数值正确性，验收必须含输出逐字对比。
20. **P0，目标口径的调度读数缺失**（2026-10-27）：splits / copy / cpy_sync 三个数仍然只有 32K 口径
    （`RESULTS.md` 3：2026-10-20、`09cacdfd8`、cache 6144、tg32、hit 60.9%），而统一工况是 256K + 10K + hit 57.6%。
    要补：一条 `256K + GGML_SCHED_TRACE=1` 的 run，跑批器必报列加 `session id` 与 `n_kv`（`bench/README.md` 第 3 节）。
    同时给 session 级台阶定性：`RESULTS.md` 0.5 的 8176 档是 511.7，0.4 是 679.5（低约 25%，同臂 r1/r2 只差 1%）。
    **在定性之前，任何 <15% 的收益都不能验收**（0.5 读数 6 已禁止两节混算）。
    **2026-10-27 目标口径读数已补**（`RESULTS.md` 0.7）：解码单步 58.5 ms 里 `cpy_sync` 占 **42.3 ms（72%）**、
    38 次宿主权重回调每次约 1.1 ms、拷贝本体只有 2.7 ms；prefill 每 ubatch 墙钟 4.03 s 里 `cpy_sync` 2.70 s。
    纯串行气泡约 **20 ms/token**，抹平的终点约 26 t/s，与第 3 节整层扫实测的 27.1 t/s 平台一致。
    仍缺：跑批器必报列的 `session id` 与 `n_kv`，以及 session 级台阶（-25%）的定性。
21. **已关闭（2026-10-27）：所谓"PERF_LOGGER 不输出 GEMM 与 FA 行"是汇总脚本的误判，不是 profiler 坏了**。
    `GGML_VK_PERF_LOGGER` 的 key 带形状后缀（`MUL_MAT_VEC q4_K m=2048 n=2 k=4096`、`MUL_MAT_ID ... n_expert=821 batch=2048`、
    `FLASH_ATTN_EXT ...`），按"纯 op 名"过滤求和会把 GEMM/FA 整类丢掉。已给 `bench/tools/summarize_perf.py` 加
    `--agg`（按基名聚合 + top 精确 key）。用 `bench/raw/p0-20261027/concat/fast1-1.err`（目标工况 prefill 口径）重算：
    **MUL_MAT 39.1% + MUL_MAT_ID 31.7% = 70.8%**，单一最大 key `MUL_MAT_ID q4_K m=512 n=8 k=2048 n_expert=821 batch=2048`
    就占 14.6%（反推 3.58 TF，与 in-situ 3.5-3.7 TF 吻合），FA 只有 3.9%。
    推论：算子轴的验收手段一直是好的，第 8/9 条（id GEMM tile、q6_K 缺口）**没有被挡住**；
    而 PP 的份额账要按这份新排名重写，第 3/4 节那批 FA 为主的锚点已过期（判决 22 同向）。
22. **P1，PP 份额账不平**（2026-10-27）：`RESULTS.md` 4 的每 ubatch 份额按行相加是 2733 ms，表合计写 3014 ms
    （= 实测 `2048/680`），**缺 281 ms 未列项**。补齐之前，"对 2.05 s 的 PP1000 门槛差 964 ms"不能用来排 ROI。
23. **P1，`llama-store` 未开工（2026-10-27 需求定稿）**：它是**本机 VRAM/RAM/SSD 的抽象层**（分配、驻留、搬运、记账），
    但第一版**只接管两类对象** - 专家（**层 + 专家子对象**两级，搬运单位为专家）与 KV 页
    （页固定 128 cells = 2.5 MiB，`--kv-block` 已移除）；
    dense 权重、compute buffer、staging 的分配路径与**加载方式都不改**，只 `store_declare` 申报占用。
    核心目标是专家预取，其余模块不许被影响；**MoE cache 也交给它**（cache 只按句柄取专家接进图）。
    子系统**常开，无总开关**。对外只有三个**整机总预算**参数（`--store-vram-mib` / `--store-ram-mib` /
    `--store-ssd-mib`，默认 0 = fit/auto），预算 = 对象占用 + 申报占用；VRAM 那个同时就是显存档位
    （4096 / 6144），旧 env `GGML_VK_VRAM_LIMIT_MIB` 并进来；内部分配由 store 自己决定并打 `store_split`。
    **2026-10-27 进展（2a 骨架 + 2b 申报部分 + 2c 第一刀完成，`RESULTS.md` 0.11/0.12）**：`src/llama-store*`
    六文件三单元落地（slab/io/policy），常开挂 `common_init_result`（析构先于 context）；`--store-*` 四 CLI
    进树，vram 参数经旧 F0 场 hold back（4096 参数臂 = `held 3282.8`、规划器 ledger 与 4096 档逐字同）；
    专家两级注册 10240 个进 registry（物理字节还在 dense buffer，`co-located` 记账防双计）；
    `store_split` 进启动日志与 plan ledger 对得上；`--store-exec 0/1/2` 把 MoE 整层预取的提交/等待路由进
    store-io（A/B/A 零行为差：决策/hit/内存/行为行逐字同、PP 差 < 0.2%，`RESULTS.md` 0.12）；
    `ctest` **34/34**（增加 slab 单测）。剩余：io 池（2g/M2 的 use_step 预取）、staging 申报与 5% 越权对账、
    promotion 路径入计数、KV 页 adopt = 3a/M3、M2 的 TG >= 15% 收益验收。
    两个入口：`store_acquire` 同步阻塞、只等这一批（序号等待，不得整设备 fence）返回 ready 句柄；
    `store_prefetch` 立刻返回句柄（槽号 + 代际号，经间接表解析），由进程内唯一的共享池搬运，消费点 `store_wait`。
    句柄三条约束、记账列、M1-M4 验收见 `THREE-TIER.md` F12 第 5-12 节；当前只有 M0 等待原语。
    M1 是 ±1% 不回退门（8176 档 PP 506.7 / TG 17.2），M2 才是收益步（TG >= 15%），A/B 只能用预算参数复现旧行为，
    不许加总开关。与第 19 条的重叠：同步取必须落在序号等待上，否则 store 会把要删的整设备 drain 再固化一遍，
    所以第 19 条先做。加载边界也已定：loader 的 `lazy_read` / mmap / resident 全照旧，
    只把含专家的映射区间 `store_attach_mapping` 登记，store 只控专家区间的常驻（`THREE-TIER.md` F12 第 12 节）。
24. **P0，`ggml-sched-plan` 未开工（2026-10-27，F13 S2）**：目标口径单步 58.5 ms 里 `cpy_sync` 占 42.3 ms（72%），
    而分片决策、拷贝调度、event / 序号可见性现在分散在 `ggml/src/ggml-backend.cpp`（2400 行）、
    `src/llama-context.cpp`（3982 行）、`src/llama-graph.cpp`（3638 行），没有一个模块对"这一步有几次拷贝、
    谁等谁、什么时候可见"负责。抽出来后 store 只给 deadline、拿回"要等到的序号"。
    验收：`cpy_sync` < 20 ms、输出逐字、`splits` 不劣化，TG 收益按 15% 门槛。
    它是 store M2 收益的上限，必须排在 M2 之前（判决 26）。与第 19 条（去掉每层 drain）是同一件事的运行期侧。
25. **P0，统一预算式（D 项升 P0，2026-10-27 架构评审，判决 27）**：预算现在是分段扣减且主权分裂。
    KV 的分配在 `src/llama-kv-cache.cpp:233`、bank 在 `src/llama-context.cpp:639`、compute 在 `:642`，
    三处互不可见，`common/fit.cpp` 靠扣减逼近；score 预留按设备扣的记错账（判决 18 之前的状态）是它的已发生后果。
    把 KV（`-kvg` 工作集口径）、score（`-fac` 块口径）、设备 compute（`RESULTS.md` 0.2 表 A 的 n_ctx 回归）、
    host compute（`96.18 + 0.2 x KV`）、dense、RS、bank、staging 放进同一个式子，输出
    `Placement[资源][层] -> {vram, ram, ssd}` 表 + 总账，实现先落在 `common/fit.cpp` 内部。
    **不动任何搬运代码**：`--moe-cache-mib` / `-kvg` / `-fac` / `--fit-plan` 行为不变；`llama-store` 第 2 步
    （`THREE-TIER.md` F12 第 4 节）接管数值决策时把式子整体搬进 store、fit 收成申报薄壳，不写第二套算术，
    迁移冲突同第 5 节"预算解算只有一个所有者"。
    验收：`RESULTS.md` 0.4/0.5 的三档读数在 r=2 噪声内复现、`reallocs` 不增；4096 档的 500 MiB 缺口（第 15 条）
    在表分解里逐项可读（margins 地板 256 + 未归因 compute 464），不再靠日志反解。
    与第 24 条的关系：sched-plan 解决"能不能重叠"，本条解决"给多少"，两者都是 store M2 的前置，都不在搬运侧。
    **2026-10-27 进展（1a-1d 完成）**：`common_moe_ledger` 已落地（`common/fit.{h,cpp}`，挂 `common_moe_plan::ledger`），
    解算只剩一个式子；同 session 新旧二进制 A/B 三档规划器决策**逐字相同**、4096 档 500 MiB 缺口已逐项可打印，
    读数与协议在 `RESULTS.md` 0.10；账本此后就是本条与第 23 条（store M1）的共同消费面。
    v1 界定：RS 未拆分（context 类并入 KV+RS，F7 落 `--rs-offload` 时拆）、分层放置表随 store 2a/3a 进入；
    检查清单（1a-3c）见 `THREE-TIER.md` 第 0 节 0.0.1，进度以那份为准。

新增行为（F7 的 offload 开关、F8 的分块开关）落地时必须进第 4 节开关表，默认值由目标口径实测决定；
`-kvg/--kv-grow` 已按此规则登记（默认 -1，实测见 `RESULTS.md` 0.3）。
