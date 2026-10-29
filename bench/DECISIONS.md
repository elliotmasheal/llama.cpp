# DECISIONS - 判决、已证伪、开关、硬约束

与 `RESULTS.md` 冲突时以数字为准；本文件只回答"这意味着什么、还能不能做"。编号供代码注释引用。

## 1. 判决（一行一条）

1. resident 是前提：pp512 336 对 mmap 171（2.06x）；mmap 下 bank 与预取强制 0。
2. ub<=512 的 PP 是上传受限，ub2048 起是 compute 受限；余量在 kernel 不在搬运。
3. MoE 层间依赖链 `MoE(L-1) -> routing(L) -> upload(L) -> MoE(L)` 没有空隙，调度重叠无效。
4. 命中率来自时间局部性：静态 prefill 计数放置 hit ~= coverage（22%），warmed LRU 33% 覆盖有 48-68%。
5. 回读是唯一真实可攻项：ids 的 D2H + 每层整设备 drain，并入 split 后 +3.8% tg32。
6. H2D 14.3 GB/s 是硬墙；唯一余量是全双工约 23.3，但 AMD 上 buffer ownership 未实现。
7. 执行器 / 图内 hybrid 单独做都是负收益：每个 CPU split 要 device sync，两个分支从不重叠。
8. bank 单调变好直到设备容量上限（6144），再大就崩；还受单次分配上限 2 GiB 约束。
9. 长 KV 的注意力不是 FA：`n_kv > n_q` 时 FA 1.07 TF、matmul 4.08 TF；代价 score +1276.7 MiB。
10. CPU 算 miss 的判据是带宽：`B_cpu > B_pcie` 才成立。干净重测 20-25 > 14.3，**判据成立**（0.15.5）；
    旧读数 10.1 作废。方向重新开放，但要先有 Vulkan `MUL_MAT_ID_MASK`。
11. `-nkvo` 在 256K 上是 x2.4-2.7 的税，成因是 CPU 注意力 + 专家上传，不是 KV 搬运。
12. 256K 与 MoE cache 冲突的触发条件是 `-nkvo` + 非零 bank，与 ctx 长短无关（`:1910` 已修）。
13. "FA / decode 回归"是跑批器伪影（PowerShell 重定向），不是引擎问题。
14. KV 分配大小本身有代价：同 n_kv 下 256K 的 PP 239.5 对 32K 472.3。
15. 设备 compute 有一笔随 n_ctx 走的预留：16K 578 -> 256K 1712（+1134 MiB），与 KV 放置无关。
16. F7 的状态搬运量级按工作张量算：prefill `conv_state_last` 1.92 GiB/ubatch，decode 144.8 MiB/token。
17. K/V 按工作集分配（`-kvg -1`）：320 MiB 对 5120 MiB；256K 三档 PP 447-478 / TG 12-14。
18. 长跨距 score 按 query 分块是 8G 档 PP 的最大单项杠杆：PP 478 -> 680、TG 13.7 -> 20.4；
    cap 不是越小越好（64 MiB 比 256 低 20 t/s）。
19. 规划器的 "prefetch available" 不等于 cache 真开预取；判据看 cache 侧 `prefetch needs ... slots`。
20. 6 GiB 档的瓶颈是预算归属，不是容量：`--fit-plan 1` 后 PP +29.3% / TG +28.3%。
21. 小 bank 是负优化，必须设地板：4096 档 116 MiB -> hit 0%、TG -17%；地板 145 MiB。
22. 方阵跨度（`n_kv == n_q`）留在 FA 是对的：换 matmul 后 total +2.5-5%。
23. 官方 b11763 在目标工况没有可用路径（f16 KV + bank 冲突 / abort / PP 147）。
24. store 不预测路由、不碰序列语义：只按 `(对象, use_step)` 排 deadline。
25. 子系统判据四条（生命周期 / 失败模式 / 变化频率 / 可单测），至少三条才拆；CPU 池与 IO 池不得合并。
26. 调度 drain 是预取收益的上限：单步 58.5 ms 里 `cpy_sync` 42.3 ms（72%）。
27. 架构评审三结论：记账先于搬运（D 升 P0）、句柄三道检查收敛成一个原语、同步归属单一（序号等待）。
28. 单卡事件等待对 TG 中性（+0.9%）：等待点就是该层真实工作量；继续在"等"上做没有收益。
29. 预取与缓存是方向不是可选项：8 GiB 档 bank 装不下 40 层 x top8；命中率是一阶旋钮。
    SSD 不进 decode（0.81 GB/s）；"miss 交 CPU 算"按 20-25 GB/s 重开（判决 10）。
30. 无预取环时静态整层是负收益，默认 0 层：TG 19.06 -> 20.66、hit 52.88 -> 68.04%。
    只在 2376 MiB / 1552 slots 这一格成立，8176 档 Q4_K_M 必须复测。
31. hybrid 的 CPU 侧成本是图的节点数，不是带宽：随机 1.883 MiB 专家块读 1 线程 24.5、4 线程 42.3 GB/s，
    而 worker 每层 0.738 ms 只算 5.1 MiB（6.9 GB/s）。combine 的 8 个节点值 0.32 ms/层，拿掉后
    worker 1.056 -> 0.738 ms、TG 12.3 -> 14.7（8176 档、bank 2380、T=4、2.72 miss/层）。
32. hybrid 的结构上限是每层 0.74 ms 全串行（combine wait 0.704 对 worker 0.738，零重叠），
    加删除后的拷贝/提交 32 + 14 ms，落在 14-15 t/s，低于同口径基线 20.0；CPU 侧要 < 0.3 ms/层
    才可能翻盘。当前定位是容量退路，不是 8 GiB 档的性能路径。
33. `Vulkan_Host` 的 CPU 读与普通 RAM 同带宽：随机 1.883 MiB 块 1 线程 24.3、4 线程 36.6 GB/s
    （`test-tiled-mulmat --host-buft`）。worker 的 6.9 GB/s 是算子结构（4 个节点 + down 的 512 短 dot），
    不是内存类型。hybrid 在 8 GiB 档 NO-GO 为性能路径。
34. IQ4_XS 的 down 短 dot（K=512、2560 行、batch 1）在 AVX2 上已到 12-13 GB/s：tiled、x2、换 Q4_K
    三种都更差（见已证伪表）。CPU 侧没有 2 倍路径，hybrid 的 14.7 就是 8 GiB 档的上限。
35. hybrid 每层的墙钟是 59 us（GPU split）对 745 us（CPU worker），`combine wait` 0.710 ms 就是 GPU
    做完后在等 CPU。重叠最多省 59 us，即使 CPU 无限快，每 token 也只到 48 ms（20.8 t/s），因为
    hybrid 只省下 3 ms 上传，却把基线里被 GPU 计算掩盖的异步上传（h2d 1.04 ms/token）换成了同步
    CPU 计算（25 ms/token）。方向本身不可能赢，不是没调好。
36. 容量不变（1264 slots、63 token trace）时命中率上限 76.08%（Belady），生产 SLRU 68.04%，
    可实现的"每层 pin 24 + 共享 LRU"70.45% 是事后频率、在线约 69-70。净空间 1-2 点（TG +1-2%），
    不是一阶杠杆：标准 LRU 只有 61.5%，SLRU 已经把时间局部性吃干净；LFU 58.7% 更差（无衰减会饿死新专家）。
    相邻 token 同层重复率 38.5%、每层 63 token 内中位 99 个不同专家（256 里），1264 slots 平均 31.6/层
    只覆盖约 1/3。工具 `bench/tools/moe_trace_policy.py`。
37. 预取环的价值在 prefill，不在 decode：纯环（0 整层）r=2 交替 PP 560 -> 705（+25%）、TG 20.5 -> 19.3（-6%）、
    hit 67.6% 对 68.04%、输出逐字同。**这 6% 不是环本身**：把环的阈值调到永不触发，TG 仍是 19.3，
    而"环关"是 20.5；差别在层数——环开时 3 个布局不同的小组被踢出 bank（40/40 -> 37/40 层有 bank），
    它们每 token 都要从 host 拿专家。0.16 判"环是负收益"混了 4 个整层，那组 PP +33% / TG -9% / hit 51.66%。
    按 PP 权 1 / TG 权 2 加权，纯环净 +13%。
38. per-group 环兑现判决 37 的账：37 层组双环、2/1 层小组留在 bank 走纯 LRU，环的 512 slots 在 decode
    前 release 给 LRU。8176 档 PP 555.2/550.8 -> 703.9/700.8、TG 20.5/20.2 -> 20.9/20.9（r=2），
    hit 68.04 -> 66.60%、上传 +428 MiB；6144 档 PP 709.6 -> 702.4（-1%）、TG 17.7 -> 19.3（+9%）、
    hit 50.1 -> 55.0；4096 档 bank 0（规划器 deficit），读数噪声内。三档输出 64 token 逐字同。
    `pf_max_full` 默认 0；环槽的 map 不进 `key_of`，release 必须按地址扫（0.17.9）。
39. F1 全部落地：`victim()` 返回 -1、`plan()` 先算 misses / 可驱逐槽再改状态（失败不留半成品）；hybrid
    worker 失败先在 combine 线程内联重跑，二次失败清零该层 CPU 贡献 + `hybrid_off`（后续 graph 走图内
    CPU 分支）；rolling 的 plan / 上传失败用 `rolling_off` 延迟到下一个 lookup 再切静态驻留（当前 token
    仍用 promote_map），切换前 drain。`llama-moe-cache.cpp` 已无 `GGML_ABORT`。8176 档 r=2 无触发轮读数与输出逐字同（0.17.10）。

## 2. 已证伪（勿重做）

| 方向 | 实测 |
|---|---|
| BN=32 的 id tile / BK_STEP=4 / 单一 SPIR-V 全 tile | 5.11 -> 4.89 / pp2048 -1.36% / batch2048 -12% |
| id GEMM 结构性重写 | 只占 wall 18.4%，dense 平价 474 对 448.6（低于 485 门） |
| q6_K dense `_bk2` | bank 6144 pp512 -37%、3584 pp2048 -5.8%，两向皆负 |
| 跨 token routing 预测预取 | top-8 recall 33.6-55.6%，token 回退率 100%，superset 需 1.27-3.18 GB/token |
| GPU 侧 slot plan / 层内重叠 / 目标化 wait / 5 槽 ring | 上限 32-35 / 29.5-30.5 / 233-248 对 368 |
| 把 copy 回调挪到 wait 之前 | pp512 920 但输出是垃圾（目的缓冲复用竞争） |
| 跳过 ids 回读、整张量上传 | coverage 70%，字节 +43%，pp512 346.8 -> 294.2 |
| ubatch 级预取 | 上限 +27%、现实 +10-15%，且没有第二缓冲 |
| `VK_EXT_external_memory_host` / SLRU 分段 / 每组 6 整层 | 0xC0000005 / tg32 21.9 无差 / -17% |
| 缩小 bank / 改 auto-fit | 4096=351.5、6144=382.4，6400 崩到 57 |
| 稠密 GEMV 的 Vulkan 旋钮 | 全部中性或更差，已在 DRAM 地板 97% |
| 内置 MTP / ngram 默认参数 | +5% 或更慢 / 0.69x |
| decode 侧 deferred 上传 | 只有 34% 字节可隐藏，+1.5% TG |
| 图内 `--moe-hybrid` / 执行器 + 滚动驻留 | 6.1 对 15.9 / 18.26 对 21.24 |
| 给 CPU MoE 加线程 / `-nopo 1` | 4 线程顶点，8 线程慢 38% / pp512 169.3 |
| 长上下文 KV q8_0 | 与 f16 平价（400.8 对 413.9），不是速度杠杆 |
| 方阵跨度改 chunked matmul | total +2.5-5% |
| 按真实行数把 tile 换 bn=64 | 净负（省 0.6 ms 重读，多 1.6 ms 计算） |
| B1 lazy 专家单独做 / SSD 进 decode | NO-GO（无调用点）/ 258 ms -> 3.9 t/s |
| `-t 16` / 调 `GGML_VK_MAX_NODES_PER_SUBMIT` / `DISABLE_FUSION` / 双 GPU | 全部低于基线 12.x；双 GPU 复测 12.38 + 304 MB OOM |
| 强制 tiled 跑 batch=1 / `__builtin_prefetch` | 3.3 对 10.1 GB/s / -13% |
| 融合 `MUL_MAT_ID_SWIGLU` 当端到端杠杆 | 单线程 1.00x、13 节点图 -4%、瘦图 0（14.8 对 14.7）；收益只来自节点边界的线程池摊销 |
| down 短 dot 的三种重写 | tiled -4 倍（worker 3.11 对 0.738 ms）、x2 -45%（1.072 对 0.738）、换 Q4_K 更慢（12.71 对 13.72 GB/s） |

## 3. 实验开关（A/B only，不是产品旋钮）

- 开关只为一次性 A/B 存在：赢了折叠进自适应（规划器 / 代价表 / 预算式），输了删除。
- 生产不读 env。用户可见只有三类：档位与预算、容量（`-c` / `-ncmoe` / `-lm`）、标注 experimental 的开关。
- "去向"列：折叠 = 已进默认或自适应，生产不再碰；实验 = 只用于 A/B；待删 = 结论已负。

| 开关 | 默认 | 作用 | 关键实测 | 去向 |
|---|---|---|---|---|
| `LLAMA_MOE_PREFETCH` | 开 | 整层预取环（2 槽，提前 1 层） | 关：pp2048 728 -> 580、pp8192 187 -> 111 | 折叠 |
| `LLAMA_MOE_PREFETCH_MAX_FULL` | 0 | 有环时每组整层驻留上限 | 4 层：TG 18.3 / hit 51.66%（0.16）；0 层：TG 20.9 | 实验（默认 0） |
| `LLAMA_MOE_STAGE_IDS` | 开 | ids 回读并入产生它的 split | 关：tg32 29.08 -> 27.84 | 折叠 |
| `LLAMA_FA_LONG_KV_MIB` | 1536 | 长 KV 走 matmul 的 score 上限，0 = 关 | 关：pp8192 187 -> 293.9 | 折叠（`-fac` 接管） |
| `GGML_VK_CONCAT_FAST` | 开 | CONCAT 快路径 | CONCAT -16%，端到端 -1.4% | 折叠 |
| `--kv-gather-mib` | 256 | 分页 prefill 的稠密工作区 | 0 是 311.6 / 307.6；非 0 是 378.9 / 360.6 | 折叠 |
| `--fit-plan` | 1 | 规划器规则（margins / score 归属 / 组预取 / 地板） | 6144 档 PP +29.3% / TG +28.3% | 实验（A/B 对照），生产默认 1 |
| `--sched-parallel` | auto | 单卡也开双拷贝槽 + 事件等待 | 中性（+0.9% / -2.0%） | 中性，待删（auto 已够） |
| `--store-exec` | 0 | 预取提交 / 等待的路由 | 0/1/2 零行为差 | 实验，待 M2 定 |
| `LLAMA_MOE_HYBRID_FUSED` | 关 | worker 用 `MUL_MAT_ID_SWIGLU` 代替四节点 | 瘦图 0（14.8 对 14.7），13 节点图 -4% | 中性，保留 A/B |
| `--moe-free-frozen-host` | 关 | 冻结整层在 host 的 pinned 专家只读一次后释放 | Host -1424 MiB，PP/TG 噪声内；不可逆 | 实验（容量杠杆） |
| `LLAMA_FA_SPAN_SQUARE` | 关 | 方阵跨度也走 matmul | 负（total +2.5-5%） | 待删 |
| `--store-vram-mib` | 未设 | 整机 VRAM 总预算 + 显存档位 | 4096 档 `held 3282.8`，ledger 逐字同 | 产品参数 |
| `-kvg` / `--kv-grow` | -1 | K/V 初始 cells（-1 按工作集） | PP 178.5 -> 477、TG 6.4 -> 13.8 | 产品参数 |
| `-fac` / `--fa-chunk-mib` | 256 | 长跨距 score 每块上限 | 0 -> 458、64 -> 657、256 -> 680、512 -> 627 | 产品参数 |
| `--kv-page` / `--kv-page-auto` | off | 分页 KV（128 cells = 2.5 MiB） | 等价性已过；4096 档比连续低 3-5% | 产品参数 |
| `--store-ram-mib` / `--store-ssd-mib` / `--store-threads` | 0 / 0 / 2 | RAM 预算、SSD 冷层、池大小 | 子系统无总开关 | 产品参数 |

## 4. 代码级硬约束（改之前先读）

- `max_batch = 32`：slot_map / miss_mask 只覆盖 <= 32 token，超出退回 host 权重路径。
- 预取环只有 2 槽、提前 1 层（`pos % 2`）；hybrid 模式下环禁用，改用预取批次做 promotion。
- 环按组启用：放不下双环的组留在 bank 走纯 LRU；pf 批次只给有环层 slot map，其余层在 graph 侧绑 host。
- 每组整层驻留上限 5（`max_full`），并保留 >= 1/5 slot 给 LRU；有环时 `pf_max_full` 默认 0，无环时 `n_full = 0`。
- 零槽固定在 `n_slots`，分配时预清 0xFF/0，miss 读它并贡献 0。
- `ggml_vk_get_compute_ctx` 会等最新 transfer semaphore；`prefetch_semaphore` 批次不被 compute 等。
- 预取前提：单 GPU、专家在 pinned host buft（`-lm none`）、后端暴露 proc address；缺一打 warning 退回。
- **容量不足与运行期失败一律退化，不得 abort**：`llama-moe-cache.cpp` 已无 `GGML_ABORT`。计数：
  `n_plan_fallbacks`（LRU plan）、`promote_fallbacks`（rolling）、`hybrid_failures` / `hybrid_missing`（worker）、
  `n_split_degrades`（ids 在 lookup 自己的 split）。
- 分页 `SET_ROWS` 不得进 ROPE / RMS_NORM 融合（会把 256 B 锚写成块，输出错 / DeviceLost）。
- 分页扩容必须用 `llama_kv_cells::grow()`，不得用 `resize()`（会清空 cell 的 pos/seq）。
- `llama_context` 析构里 `wait_promotion()` 必须在后端释放之前调用。
- store 落地后：句柄 = 槽号 + 代际号；搬运期间 pin；池内线程禁止阻塞等待；三预算互不挪用；
  `store_declare` 是硬边界（debug abort，release 计数 `store_undeclared_bytes`）；三单元不得合回一个文件。
- 预算解算只有一个所有者（当前 `common/fit.cpp`，store 落地后整体搬入），其余模块只读表或申报。

## 5. 未结项（按优先级）

1. **P0** 4096 / 6144 档目标口径矩阵（最大空白）；4096 档 bank 必须非零且 `status = ok`。
2. **P0** 4096 档 500 MiB 缺口（margins 地板 256 + 未归因 compute 464）在账本里逐项读。
3. **P1** F4：hybrid 已接进图（`cpu_out`），combine 出图后 CPU 侧 0.738 ms/层 = 6.9 GB/s。
   `Vulkan_Host` 读已证与 RAM 同带宽（24.3 / 36.6 GB/s），差在算子结构（4 个节点 + down 的 512 短 dot），
   要翻盘得重写小 batch 的 MMID kernel；验收 TG > 20.9，否则 hybrid 只做 4G/6G 容量退路。
   探针：`test-tiled-mulmat --host-buft`、`bench/tools/moe_rand_bw.cpp`。
4. **P1** q6_K dense 的 mmq 解包（单 ubatch 22.6%，改 q4_K 约 PP +7.5%）。
5. **P1** repack KLD != 0（混合精度分层与 F5 的前置门）。
6. **P1** store M2：use_step 预取，`store_prefetch_wasted` 降 >= 50% 且 TG >= 15%。
7. **P2** F9 有界 pinned 窗口（三臂验收：无窗口 / 窗口 / resident）。
8. **P2** F7 `--rs-offload`（状态 1.92 GiB/ubatch 与 KV 解耦）。
9. `ctest` 34/35：`test-model-resolution` 本机环境性失败，定性前基线按 34/35。
10. 独立小 draft 模型与 ReBAR / SAM 两项可行性筛查未做。
