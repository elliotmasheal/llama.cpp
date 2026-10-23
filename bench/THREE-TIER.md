# THREE-TIER - VRAM + RAM + SSD 协同与自适应

回答"小显存、小内存、大上下文这台机器该怎么分层、下一步做什么"。
数字与已证伪方向分别见 `RESULTS.md` 与 `DECISIONS.md`，本文件不重复判决。

## 0. 最高优先级优化方案（2026-10-23 评审，覆盖下面的顺序）

本节是后续优化的最高优先级，下面第 4 节的顺序以本节为准。方案按"预期收益 x 证据把握"排序。
每一条都是先测后改，都必须在 256K + 10K prompt 口径跑矩阵，并按 `README.md` 第 3 节报全列。
下列数字除非标了实测，都是待验证的假设，不是结论。

### 0.1 根因重述（为什么 A 排第一）

2026-10-23 的归因测量（`RESULTS.md` 0.2）把原先的三条假设换成了下面五条实测事实：

- KV 按 `n_ctx` 整块分配（`src/llama-kv-cache.cpp:233-234`，`ggml_new_tensor_3d(..., kv_size, ...)` 且 `kv_size = n_ctx`）。
  256K 下 5120 MiB 必须一次放进某个 backend，装不进设备就整体 `-nkvo`。
- 真实工作集只有约 205 MiB：10428 token 的 prompt + 64 token 生成约占 10.5K cell（10.5K x 20 KiB）。
  利用率 4%。**分配大小本身有代价**：同 bank 0、同 n_kv、同 splits 下，KV 在 GPU 时 256K 的 PP 只有 239.5，
  32K 是 472.3，差别是每 ubatch 专家上传从 3.7 s 劣化到 7.4 s（`RESULTS.md` 0.2 表 B）。这条是 A 的最强证据。
- `-nkvo` 把注意力子图拖到 CPU（`src/llama-graph.cpp:3189-3192`），prefill 的 CPU 时间从 1.7 ms 涨到 6.0-6.3 s/ubatch
  （单个 ubatch 内 5.3 -> 13.5 s，随 n_kv 线性）。**KV 在 prefill 期间几乎不跨总线**（整个 10K prefill 约 210 MiB，
  专家是 18.1 GiB/ubatch），所以税不是"搬 KV"，而是"在 CPU 上算注意力"。
  splits 291 里有 bank 的份（32K + KV 在 GPU + bank 0 也是 121），不要全记在 KV 放置账上。
- decode 反过来是 KV 在搬：205.7 MiB/token 的 KV 视图上传（decode 走 FA，K/V 必须在设备侧），
  加状态 144.8 MiB/token。A 做完这两项直接归零或退回小常数。
- 设备 compute 有一笔随 `n_ctx` 走、与 KV 放置无关的预留：`-nkvo` 下从 16K 的 578.02 涨到 256K 的 1712.02（+1134.0 MiB），
  而 host 侧 compute 精确等于 `96.18 + 0.2 x KV`。**A 落地后这笔预留自动跟着工作集走**（1490.02 -> 708.02 MiB），
  因为 reserve 图读的是 `init_full()` 报的当前容量，而不是 `n_ctx`。

### A（P0）：KV 按工作集增长 + 设备优先放置（**已落地并实测**）

- 实现（第一阶段，不分段）：K/V 初始只分配工作集大小的 cells，越界时按 2 倍增长，增长用 device-to-device 拷贝已用区域，
  上限仍是 `n_ctx`；设备分配失败时回退 host 并计数告警，不 abort（红线）。
  开关 `-kvg/--kv-grow N`（-1 = 自动、0 = 旧行为、>0 = 指定初始 cells），已登记进 `DECISIONS.md` 第 4 节。
- 实测（`RESULTS.md` 0.3，256K + 10K，三档 r=2）：**PP 447-478、TG 12-14、全部 `status = ok`**，
  对照基线 PP 178.5 / TG 6.4 是 **x2.67 / x2.16**；RAM 峰值 26554 -> 19390 MiB。
  初始 160 MiB -> 增长到 320 MiB（16384 cells）一次，`reallocs` 3 次。
- 数值等价：同一二进制 `-kvg 0` 对 `-kvg -1`，`temp 0` 生成文本逐字相同；`ctest` 32/32。
- **A 自动带出了 B1**：增长后 `ensure_capacity` 触发重预留，reserve 的 mask 按当前 `n_kv` 计算，
  设备 compute 1490.02 -> 708.02 MiB、host 1088.09 -> 128.09 MiB；planner 首次在 256K 给出 bank 1481 MiB 且预取打开。
- **F1 被绕开**：`:1910` 的触发条件是 `-nkvo` + 非零 bank。目标工况不再需要 `-nkvo`，所以 bank 非零也不会撞。
  F1 仍然要做（4G/6G 档与超过显存的深度需要 host KV），但不阻塞本方向。
- 限制：真把 256K 填满时会增长回 5120 MiB，重新撞 `RESULTS.md` 0.2 表 B 的容量悬崖；这是预期行为。
  要覆盖"真长上下文 + 小显存"仍需 F1 + 分段（原第二阶段）。
- 第二阶段（只有第一阶段证明不够才做）：按 cell 块分段，前段设备、尾段 host，用 LSE 合并两段注意力。
  这要动 FA/matmul 注意力路径，风险大，先不做。
- 验收：256K + 10K prompt、8176/6144/4096 三档全部 `status = ok`，KV 驻设备部分约 205 MiB，
  splits 从 291 降到 40-121（取决于 bank，bank 0 时 121）；PP、TG 相对同 n_kv 的 32K + KV-on-GPU 对照在 r=2 噪声内持平。
  **已达成**（`RESULTS.md` 0.3）：KV 驻 320 MiB，PP 447-478（对照 32K + KV-on-GPU 的 472.3），TG 12-14（对照 12.9-15.4）。
  预期（按 `RESULTS.md` 0.2 表 B 的同批对照，bank 0）：PP 178 -> 约 470、TG 6.4 -> 约 13；
  若 A 同时让 bank 非零，PP 还会再高（bank 1719 的对照是 483.6 / 15.4）。

### B（P0）：注意力按 query 分块 + 预留随块大小，不随 n_ctx（即 `THREE-TIER.md` F8 的实现形态）

**B1（预留随工作集）已由 A 自动达成**：`RESULTS.md` 0.3 的 1490.02 -> 708.02 MiB 就是它，不需要单独改代码。
下面只剩 B2（按 query 分块），它改变数值结果，等 F6 或 top-1 一致性对比再动。

- 证据一：matmul 注意力的 score 缓冲在 10K 时 +1276.7 MiB（`RESULTS.md` 2.2），由设备 surplus 里固定扣 `LLAMA_FA_LONG_KV_MIB=1536`（`common/fit.cpp:954-960`），挤掉 bank。
- 证据二（2026-10-23 新增，`RESULTS.md` 0.2 表 A）：reserve 图里 `attn_inp_kq_mask` 的 ne 就是 `n_ctx`（16K 时是 16384），
  于是设备 compute 在**设备 KV 为 0** 时仍从 16K 的 578.02 涨到 256K 的 1712.02（+1134.0 MiB），
  host compute 为 `96.18 + 0.2 x KV`（256K 时 1120.18 MiB）。
  这就是 4 GiB 档 bank 0 的真正原因，比先前按三档反解出的"约 940 MiB"更大。
- 做法：`src/llama-graph.cpp:3079-3086` 的长 KV 分支改为沿 query 维分块，每块的 scratch = `n_head x chunk x n_kv x 4B`。
  chunk 必须由预算反解，不能写死：256K 时 `n_kv = 262144` 下 chunk 256 就要 4.2 GB，所以 chunk = budget / (n_head x n_kv x 4B)。
  外层 `-ub 2048` 与完整历史注意力范围不变（`THREE-TIER.md` F8 第 2 条的约束继续有效）。
- 连带收益（待测）：score 与 ubatch 解耦后，8 GiB 档可以用 `-ub 4096`。`RESULTS.md` 2.3 的 10k 读数 ub4096/4608 PP 437.8 对 ub2048/5120 的 398-407，
  并且上传从 71278 MiB 降到 40765 MiB（-43%）。这是 PP 的第二个主要杠杆，和 A 独立叠加。
- 代价：分块后 dispatch 变多，分块 softmax 与整段路径不逐位一致。先有 F6 或至少 top-1 一致性对比，再验收。
- 门禁：改 FA/graph 后必跑 `-o FLASH_ATTN_EXT -b Vulkan1`（5332，需先补完 5310 之后的约 22 例）与 `-o MUL_MAT -b Vulkan1`（2275/2275）。
- 验收：8176 档 planner bank 达到约 3084 MiB（6 层）且预取开；4096 档 `-ub 4096` 不溢出；`reallocs`、`hit %` 进表。

### C（P1）：前缀 KV 检查点落 SSD，只在 prefill 复用（合规的 SSD 用法）

- 证据：10428 token 的 prefill 在 PP 479 时约 22 s，在 `-nkvo` 的 PP 178 时约 58 s。A 完成后 KV 约 205 MiB，
  顺序读 2.3 GB/s 时约 0.1 s。
- 做法：prefill 结束后用现有 `llama_state_seq_get_data` / `llama_state_seq_set_data`（`include/llama.h:891`、`:901`）或 `*_file` 版本
  存整个状态（KV + GDN 的 RS 62.81 MiB 一并存），key = 模型文件 + ctx 参数 + token 前缀哈希。
  同一份 10K prompt 反复提问、或同一系统提示复用，都直接命中。
- 合规：只在 prefill 阶段读盘，decode 单步路径读盘次数必须为 0（`GGML_SCHED_TRACE` 证明）。
  **不能把 KV 本身放 SSD 做 decode 访问**：decode 每 token 要读全部历史 KV，10K 时约 205 MiB/token，与 `RESULTS.md` 第 5 节的 SSD 258 ms/token 同量级，直接不可行。
- 验收：同 prompt 第二次运行，prefill 段时间 < 1 s，`temp 0` 下输出与首次运行逐字一致。
- 报表规则：PP 必须分成"冷 PP"（无检查点）与"热复用"两列，不得混写成一个 PP。

### D（P1）：预算解算器一体化（F2 + F7 + F8 合并成一个式子）

- 把 dense、KV（按 A 的增长口径）、score 缓冲（按 B 的块口径）、compute、bank、RS 状态放进 `common/fit.cpp` 同一个式子，
  输出 KV 放哪、RS 放哪、ub、bank 四项，而不是现在的分段扣减。
- 复用 `common_moe_plan_get()`，不新增规划器。
- 验收：在三档 x 每档 3 个 bank 的消融里，规划器推荐值与实测最优 bank 的 argmax 一致。

### E（P1）：ub 与 bank 联合调参

- B 完成后，`-ub` 不再被 score 缓冲卡住。按 `RESULTS.md` 2.3 的数字，ub 对上传字节影响最大（-43%），
  但 ub 对 compute 与设备预留也增长。要在 256K + 10K 口径下对 `ub 2048/3072/4096` x 三档做一次完整矩阵，取每档最优。
- 注意 `RESULTS.md` 2.3 的 256K ub4096 = PP 89 / 9226 MiB 是在 KV 放设备、未做 B 的条件下读的，不能直接套用。

### F（P2，门禁）：KV 量化作为容量杠杆，不是速度杠杆

- `-ctk/-ctv q8_0` 把 256K KV 从 5 GiB 降到约 2.66 GiB，q4 约 1.4 GiB。A 完成后它能让 6G 档把整个 256K 放进设备。
- 10K 口径下 q8_0 与 f16 的 PP 平价已测（`DECISIONS.md` 第 3 节），但**质量未测**。没有 F6 的 PPL/KLD 通道前不得作为默认。

### G（P2，门禁）：大模型路径 = 有界 pinned 窗口（F9）+ 冷专家降精度（F5/F6）

- 目标"受限设备跑更大模型"的唯一路径。现在 RAM 侧是全有或全无（`common/fit.cpp:987-995`），20.2 GiB 模型在 40 GiB 机器上只能 resident。
- 降精度（热 Q4_K + 冷 IQ2/IQ3）是唯一能改变 TG 屋顶线的选项，必须先过 F6 与 repack 的 KLD 门（当前 KLD 0.01516 != 0，`RESULTS.md` 7）。

### 已排除，不要再排队

- KV 落 SSD 做 decode 访问（见 C 的说明）。
- 再调投机解码（`DECISIONS.md` 第 3 节已证伪）。
- 线程数调参（4 线程是 CPU 专家顶点，`RESULTS.md` 第 5 节）。
- 只缩 ub 来省 score 缓冲（`THREE-TIER.md` F8 第 2 条已证伪的 ubatch 级方案）。
- 用 `-ctk/-ctv q8_0` 去救 prefill PP：归因显示 prefill 全程 KV 只有约 210 MiB 跨总线，减半没有对象（`RESULTS.md` 0.2 表 C）。
- 把 `-nkvo` 的税记成"KV 搬运"：prefill 价格是 CPU 注意力 + 专家上传，decode 的 KV 上传只占约 9%（同上）。

### 执行顺序与每轮门禁

0. F1 仍是功能前提：目标口径下 bank 非零会在 `:1910` abort。A 之前若要读 bank 相关数字，只能用 `--moe-cache-mib 0`，并在报告里注明。
   **2026-10-23 更新**：触发条件是 `-nkvo` + 非零 bank（32K 也复现），所以 A 落地后可以绕开它；F1 仍要修，但不阻塞 A。
1. 归因测量（已按本节跑完，见 `RESULTS.md` 0.2）：
   (a) splits 构成：prefill 291、decode 282；prefill 期间 KV 跨总线仅约 210 MiB（全程），`conv_state_last` 1.92 GiB/ubatch，专家 18.1 GiB/ubatch。
   (b) 同一 `-nkvo` 下 16K-256K 的 `sched_reserve`：设备 compute 578.02 -> 1712.02（+1134.0 MiB），host compute = `96.18 + 0.2 x KV`。
2. **A + B1 已落地并验收**（`RESULTS.md` 0.3）：PP x2.67、TG x2.16、三档 ok、rm 峰值 -7.2 GiB。
   下一步是 C（前缀检查点）与 D（预算式一体化）；B2（按 query 分块）等 F6。
2. A + B 同一轮实现。只做 A 看不到 compute 预留的收益。
3. C 紧随 A（依赖 KV 变小）。
4. D 收尾统一规划器；E 跟 B 一起调参。
5. F、G 等 F6 质量门。

每轮都要更新四份文档：`RESULTS.md` 矩阵新增对应臂，`DECISIONS.md` 登记开关与未结项，`README.md` 若口径变化则同步。

## 1. 三层的角色

| 层 | 放什么 | 允许出现在关键路径上吗 |
|---|---|---|
| VRAM | dense 权重、命中专家 bank、KV 工作集、compute buffer | - |
| RAM | 整模型（resident + pinned，预取的前提）、`-nkvo` 时的 KV、miss 专家的就地计算源 | 是（H2D 14.3 GB/s、CPU 读带宽 39.7 GB/s @4T） |
| SSD | **冷专家容量层 + KV/权重溢出缓冲** | **否**。随机 1.95 MiB 只有 0.81 GB/s，206 MiB/token 要 258 ms；只有在 prefill 阶段、且与计算重叠、且只服务命中率 >= 85% 的冷层时才允许 |

也就是说"显存与内存都不足时 SSD 当临时缓冲"这句话在工程上的含义是：
SSD 只能进 **预取批次**（提前若干层、批量顺序读）和 **prefill**，绝不能进 decode 的单步关键路径。

## 2. 现状判定（代码级）

1. **只有两层半**：专家必须整层常驻 RAM 才被缓存（`src/llama-moe-cache.cpp:208-213`），
   全树没有任何 SSD / `RANDBACKED` / file-backed 专家路径，cache 对 mmap 完全不可见（`grep mmap src/llama-moe-cache.cpp` = 0）。
   `lazy_read` 的门槛是单张量 4 GiB（`src/llama-model-loader.cpp:1101-1105`），而单层专家张量约 116-330 MiB，
   `--lazy-mode on` 打不到专家（`TENSOR_READ_LAZY` 无调用点，B1 已证）。
2. **RAM 不足即失去全部加速**：规划器在 `-lm mmap` / `dio` / RAM 不够 resident 时把 bank 强制为 0
   （`common/fit.cpp:977-982`），预取也随之关闭；实测 mmap + cache 5.8/5.2 对 mmap 无 cache 18.0/10.2。
3. **RAM 不是计算层**：`MUL_MAT_ID_MASK` 无 Vulkan 实现，CPU 分支会把 GPU 分支一起降级。
4. **KV 不参与规划**：分配顺序是 KV（`src/llama-context.cpp:561`）-> bank（`:639`）-> compute（`:642`），
   `llama_moe_cache` 看不到 `n_ctx`；规划器算 bank 时只扣 dense/KV/compute/margin，
   但 256K 的 KV（实测 5.0 GiB）与 matmul 注意力的 score 缓冲（+1276.7 MiB）都还没进同一个预算式。
   `-nkvo`（`cparams.offload_kqv = false`）在本口径下的读数已经拿到，见下面 F2。
5. **容量不足是 abort**：见 `DECISIONS.md` 第 5 节。小显存/小内存档一旦路由分散就硬崩。
6. **常数是 8 GiB 单机形状**：`max_batch = 32`、`pf_min_tokens = 512`、`pf_max_full = 4`、`max_full = 5`、
   `LLAMA_FA_LONG_KV_MIB = 1536`、ring 2 槽。
7. **一个开关管两层放置**：`-nkvo` 同时决定 KV 和 GDN（recurrent）状态住在哪。
   三个 hybrid 分支都把 `cparams.offload_kqv` 原样传下去（`src/llama-model.cpp:2878`、`:2898`、`:2918`，本模型走 `:2918`），
   `src/llama-memory-hybrid.cpp:34-65` 再把它同时交给 `llama_kv_cache`（O(n_ctx)）和 `llama_memory_recurrent`；
   而 recurrent 的容量与 `n_ctx` 无关，是个常数：qwen35moe 走 `llama_memory_hybrid` 这一支
   （`src/llama-model.cpp:2904`，`recurrent_kv_size = max(1, cparams.n_seq_max)` 在 `:2915`，offload 在 `:2918`；
   另外两支 `:2875`、`:2895` 同理），两臂日志把这个常数读了出来：
   256K + `-nkvo` = `CPU KV 5120.00 MiB（262144 cells, 10 layers）` + `CPU RS 62.81 MiB`
   （`bench/raw/m-256k-bank0/vram8176_c262144_b0_r1.err:2129`、`:2172`）；
   32K + KV 在 GPU = `Vulkan1 RS 62.81 MiB`、KV 只有 640.00 MiB
   （`bench/raw/ctrl-kvongpu-32k/vram8176_nc41_r1.err:2130`、`:2173`）。
   层数也以此为准：`full_attention_interval = 4`、`n_layer = 40`，KV cache 只建 **10 个全注意力层**，
   其余 **30 个 recurrent 层**的 `SSM_CONV` / `GATED_DELTA_NET` 在目标口径下读写 host 状态
   （同一份日志 `llama_memory_recurrent, layer N: dev = CPU` 共 30 行/次，`:2132` 起）。
   注意力侧同样被钉在 CPU：`src/llama-graph.cpp:3189-3192` 的注释就是"KV store 到注意力输出之间的全部节点跑在 CPU"。
   **host 放置是日志确认的事实**；这 62.81 MiB 的常数换走的 CPU 时间已于 2026-10-23 归因完成
   （`RESULTS.md` 0.2 表 C/D：prefill 里 KV 跨总线仅约 210 MiB（全程），价格是每 ubatch 6.0-6.3 s 的 CPU 注意力
   与 `conv_state_last` 1.92 GiB），见下面 F7。

## 3. 缺口与步骤

按依赖排序，每步都用 `README.md` 的目标口径验收，且必须报 `RESULTS.md` 第 0 节那张表。

### F0 显存档位可测（已落地并验证）

`GGML_VK_VRAM_LIMIT_MIB` 在设备初始化时 hold back 差额。已验证：日志出现 `VRAM limited to ...`；
`vklimit-probe.cpp` 报告 free 等于设定值（6144 档 6142.9 MiB、4096 档 4094.9 MiB）；
跑批器能采到 `held` 并从峰值里扣掉；规划器随档位变化（同口径下 8176 档给 bank 1547、4096 档给 0）。
矩阵首版读数在 `RESULTS.md` 第 0 节。

### F1 容量不足必须退化，不得 abort（**已复现：目标工况当前跑不通**）

把 5 个 `GGML_ABORT` 换成"回退到上游的 host 权重拷贝路径 + 一次性 warning + 计数"，
其中 `:1910` 的分片不变量要么用 event 保证，要么退回非 staging 路径。
这是 4G/6G 档能跑完一个 10k prompt 的前提。

复现（2026-10-23，`-c 262144 -nkvo`，两档都撞）：8 GiB 可见 + planner 给 bank 1547 MiB 时在 **layer 1** 抛，
6144 档在 **layer 0** 抛 `the experts of layer N are selected in the same split as their MoE cache lookup`
（`src\llama-moe-cache.cpp:1910`）；4096 档因为 planner 判 bank 0 而躲过。
对照组（32K ctx + KV 在 GPU + bank 1719）跑通，所以成因是 `-nkvo`/长 ctx 的分片布局，不是"bank 非零"本身。
这条守卫要求 `selected_experts` 的生产者在 cache 查表之前的 split 里，且它在 staging 分支**之前**，
所以 `LLAMA_MOE_STAGE_IDS=0` 绕不过（未单独验证，读码判断）。已验证的唯一绕法是 `--moe-cache-mib 0`。
验收：4096 档 + 256K ctx 全程 0 abort、`status = ok`，回退次数进日志。

### F2 KV 与 bank 的同一预算式 + `-nkvo` 验收

规划器把 KV（`n_ctx`、`-ctk/-ctv`、FA 与否）、matmul 注意力的 score 缓冲、bank、compute buffer 放进
同一次分配计算，并输出"KV 放 GPU 还是放 RAM"的判定。

已观察到的行为（2026-10-23，4096 档 + `-c 262144 -nkvo`，`common_moe_plan_impl` 日志）：
`-nkvo` 把 KV 从设备预算里拿掉（"KV 0.0 GiB at n_ctx 262144"），但 compute 1712.0 MiB 与长 KV score 缓冲
预留的 1536.0 MiB 已经把 4 GiB 吃掉 3.2 GiB，加上 dense 之后 **deficit 1480.0 MiB**，于是规划器直接判
"no device memory for a cache, keep the experts on the host"，同时整层预取也因两个环需要 930.0 MiB 而关闭。
也就是说 4 GiB 档现在退化成"无 cache 的 host 专家路径"，三层协同实际上只剩两层。
预算式必须重解：score 缓冲的上限应该随可见显存缩（现在固定 1536 MiB），且要给 bank 留出空间。
这三档的预算逐项反解（margins 1024 MiB、self 3014 MiB、摘掉预留后各档能给到多少 bank）写在 `RESULTS.md` 0.1，
F8 的归属修法值多少以那张表为准，不要按"1536 MiB 全是可回收的 bank"来估：4096 档摘掉也只有 56 MiB 余量。
256K 口径的实测代价（2026-10-23，bash 跑批器，三档 + mmap）：`-nkvo` + cache 关闭时
PP 178.3-178.5、TG 6.4-6.5、真实足迹 `used` 3.13 GiB（4 GiB 档余量约 0.9 GiB）、每 eval 291 splits；
同一 prompt 在 32K ctx + KV 在 GPU + bank 1719 是 PP 483.6、TG 15.4、121 splits（`RESULTS.md` 0.2 复现）。
即 `-nkvo` 的税是 PP x2.7、TG x2.4。
**归因已改（2026-10-23，`RESULTS.md` 0.2）**：prefill 全程 KV 只有约 210 MiB 跨总线，税来自
(a) 每 ubatch 6.0-6.3 s 的 CPU 注意力（随 n_kv 线性）与 (b) bank 0 下每 ubatch 18.1 GiB 的专家上传。
"瓶颈是 KV 搬运"作废；decode 侧确实有 205.7 MiB/token 的 KV 上传，但只占约 9%。
候选（都要实测，未验）：`-ctk/-ctv q8_0` 只对 decode 的那 205.7 MiB 有效（10-23 已测 prefill 上与 f16 平价）；
`--moe-cache-mib` 阶梯先找出既不 abort 又有覆盖的 bank（注意触发条件是 `-nkvo` + 非零 bank，32K 也撞）。
**顺序修正**：F7/F8 的归因已完成，上面的候选按新归因重排：先做第 0 节的 A（KV 按工作集 + 设备放置），
q8_0 这类只减 KV 字节的方案降级，因为它对 prefill 的主要成本（CPU 注意力 + 专家上传）无效。
另有一条已量化的事实：`-lm mmap`（RAM 不足档）只把 PP 从 178.5 打到 142.5、TG 6.4 -> 6.2，
比短上下文时代记的"mmap 腰斩"温和得多，因为这条口径本来就没在用 bank。

验收：4G/6G/8G 三档在 256K + 10k prompt 下都不溢出、`reallocs` 进表，且每档自动选到当时最优 bank
（4 GiB 档要能给出非零 bank，或明确论证 0 就是最优）。

### F3 预取门槛按显存自适应

`:646` 要求每组至少 `2 x n_expert` slot，小卡直接失去预取（也就失去 +25.5% 的 pp2048）。
允许 1 环 / 分块层预取，`pf_min_tokens` 与 `pf_max_full` 由 bank 与 ubatch 推导。
验收：4096 档 `LLAMA_MOE_PREFETCH_TIME=1` 显示预取生效，等待不超过该档 PP 的 10%。

### F4 RAM 参与计算的前置

Vulkan 补 `MUL_MAT_ID_MASK`（或把 mask 变成 `MUL_MAT_ID` 的 `src[3]`，不动 opcode），
并让 `repack.cpp`、`ggml_prec_set_*`、`ggml_op_alloc_size_may_expand`、`sched_copy_experts` 认识它。
验收：`test-backend-ops -o MUL_MAT_ID_MASK -b Vulkan1` 有量化权重大批用例且全绿；
`--moe-hybrid` 在 4096 档 TG 高于同配置 static。这一条也是 SSD 层"CPU 就地算冷专家"的前置。

### F5 SSD 容量层（B2）

`lazy_read` 暴露专家文件区间 -> 每层 2 个 pinned staging 环 -> 与 `pf.tensor` 批量 H2D 重叠 pread ->
只对预计命中率 >= 85% 的冷层走盘；SSD 只做容量层，热专家一律禁止落盘（会摧毁吞吐）。
混合精度分层（热 Q4_K + 冷 IQ2/IQ3）是同一方向的另一条腿，但必须先修 F5 前置的 repack KLD != 0。
验收：RAM 受限 profile 的 pp2048 >= resident 的 70%；稳态磁盘读为 0；`ctest` 32/32；
decode 关键路径上pread 次数为 0（用 `GGML_SCHED_TRACE` 证明）。

### F6 质量门

恢复一个可重建的 PPL/KLD 通道（本树已删 `llama-perplexity`），
否则 F4/F5 的降精度与 CPU 分支改动无法判断是否伤输出。
验收：一条命令能在 10k 文本上给出 PPL 与 logit KL，并进入每轮的报告。

### F7 状态放置与 KV 放置解耦（**新增，P0，TG/PP 双向**）

机理已核对（本文件第 2 节第 7 条；`DECISIONS.md` 第 2 节第 11 条是它的性能读数）：`-nkvo` 把常数大小的 GDN 状态和 O(n_ctx) 的 KV 一起丢到 host，
于是 30 个 recurrent 层的 `SSM_CONV` / `GATED_DELTA_NET` 与注意力子图被拖到 CPU。
KV 落 RAM 是 256K 唯一装得下的办法，状态落 RAM 不是。

状态字节数已经读到，不用估：`bench/raw/m-256k-bank0/vram8176_c262144_b0_r1.err:2172` 在 `-nkvo` 下报 `CPU RS 62.81 MiB`
（`:2173` 展开为 R 2.81 + S 60.00 MiB），对照臂 `bench/raw/ctrl-kvongpu-32k/vram8176_nc41_r1.err:2173` 报
`Vulkan1 RS 62.81 MiB`（同一常数，与 n_ctx 无关），而同日志 `:2129` 的 KV 是 5120.00 MiB。
注意那条 size 日志的 `40 layers` 字段是 hybrid 的总层数，真正带状态的是 30 层：`src/llama-memory-hybrid.cpp:62-64`
用 `is_recr(il)` 过滤，`full_attention_interval = 4` 下 40 层里 10 层是注意力、30 层是 GDN。
也就是说：状态只有 62.81 MiB，把它推到 host 对 5120.00 MiB 的 KV 预算几乎没有帮助（1.2%），
代价却是 30 层 recurrent 计算与 10 层注意力子图的 CPU 放置。这正是 F7 要解耦的理由。

先测再改（**已完成，2026-10-23，见 `RESULTS.md` 0.2 表 C/D**）：
prefill 每 ubatch 的 D2H 是 `conv_state_last` 1.92 GiB（30 层 x 64 MiB）+ `new_state` 60 MiB + `Qcur` 320 MiB，
KV 跨总线仅约 210 MiB（全程）；decode 每 token 是 KV 205.7 MiB + `state_predelta`/`new_state` 144.8 MiB。splits 为 prefill 291、decode 282。
所以"状态搬运"的量级按 `conv_state_last` 这个工作张量算，不按 62.81 MiB 的 RS buffer 算。

实现口径：给 recurrent 状态一个与 `-nkvo` 正交的开关（建议 `--rs-offload` / `-nrs`），
默认值由目标口径实测决定，落地时同时登记进 `DECISIONS.md` 第 4 节开关表；
状态字节数必须进 `common/fit.cpp` 的同一预算式，挤掉的额度从 bank 里明算。

验收：`RESULTS.md` 第 0 节矩阵里新增 `KV=host + RS=dev` 臂，相对现有 `KV=host + RS=host` 臂，
PP 与 TG 都有 r=2 可复现的差值，并同表记录 splits、`used`、`RS buffer size`、`hit %`。
差值小于口径噪声就把这条降级进 `DECISIONS.md` 第 3 节，不许留在缺口表里当"待测收益"。

### F8 长跨距 score 缓冲的归属与按 query 分块（**新增，P0，PP 与小显存档**）

两个方向的错都来自同一块缓冲：

1. **归属错**：`common/fit.cpp:954-960` 只要 `flash_attn` 没关就从**设备** surplus 扣
   `LLAMA_FA_LONG_KV_MIB`（默认 1536 MiB），不看这块 kq 这次落在哪个 backend（`common/fit.cpp:958` 的条件只有 `flash_attn_type != DISABLED`）。
   目标口径（`-nkvo`）下注意力子图在 CPU（`src/llama-graph.cpp:3189-3192`），这 1536 MiB 就不该记在设备账上。
   把它从设备账上摘掉值多少，`RESULTS.md` 0.1 已经用三档日志反解出来了（margins 1024 MiB、设备 self 3014 MiB）：
   6144 档 surplus 568.0 -> 2104.0，bank 从 311.0 升到约 **1848 MiB** 并且首次满足 930 MiB 的双环 -> 预取打开；
   8176 档 surplus 1803.9 -> 3339.9，bank 从 1547.9 升到约 **3084 MiB**（6 个整层）。
   **但 4096 档不吃这个修法**：不扣之后 surplus 只有 56.0 MiB，而 `common/fit.cpp:973-974` 的 slack 规则最多再让出 `surplus - min(surplus, 256)`，
   算出来还是 0。谁占掉了这 4 GiB 档的余量，是下面第 3 条，不是 F8 第 1 条。
   先做归因测量：`-nkvo` + 10k prompt 下用 `GGML_VK_MEMORY_LOGGER` 看设备 compute buffer 随长跨距是否真的涨、涨多少，
   再改 `llama_fa_long_kv_reserve_bytes()`（`src/llama-graph.cpp:128-137`）的调用条件，host 归属的字节记到 RAM 预算侧。
2. **粒度错**：`src/llama-graph.cpp:3079-3086` 的 `fa_long_kv` 判据按整段 `n_head x n_q x n_kv x 4B` 决定（`n_head = 16`），
   10k 时实测 +1276.7 MiB（`RESULTS.md` 2.2），所以在 KV 驻 GPU 的短 ctx 口径上它确实挤 bank（4096 被迫降到 3584）。
   正解是按 query 分块跑 KQ/softmax/KV（例如每块 1024 token），
   **保持外层 `-ub 2048` 与完整历史注意力范围不变**，不要靠缩小整个 ubatch 省这块（ubatch 级方案已被第 3 节证伪）。
   代价是 dispatch 变多，要用 PP 实测抵掉；分块 softmax 与整段路径不逐位一致，验收前先有 F6。
3. **设备 compute 本身没查过归属**：`sched_reserve` 报的设备 compute 是 **1712.02 MiB**，三档完全一致，
   而 32K ctx 那批只有 772.0 MiB。这两个读数同时变了两个变量（`n_ctx` 与 KV 放置），所以"约 940 MiB 随 `n_ctx` 走"目前只是假设，
   要补的是同一 `-nkvo` 放置下 32K/64K/256K 的 `sched_reserve` 设备 compute 对照。
   **2026-10-23 已闭环**（`RESULTS.md` 0.2 表 A）：同 `-nkvo`、设备 KV 恒为 0 的七点扫描给出
   578.02（16K）-> 690.02（32K）-> 738.02（64K）-> 930.02（96K）-> 1154.02（128K）-> 1328.02（192K）-> 1712.02（256K），
   即 **+1134.0 MiB 确实随 `n_ctx` 走**，比假设的 940 MiB 更大；host 侧 compute 精确等于 `96.18 + 0.2 x KV`。
   同批 4096 档的 teardown warning 同向：host compute 从 1120.18 涨到 1446.35 MiB。
   这就是 4 GiB 档的钥匙：reserve 图要按 `n_kv` 工作集建，而不是按 `n_ctx`。

验收：`-nkvo` + 256K + 10k 下，6144 与 8176 两档的 planner bank 达到上面反解出的量级（约 1848 / 3084 MiB）且预取打开，
4096 档则要求第 3 条的归因结论（要么论证设备 compute 里可摘出 >= 1 GiB，要么明确论证这一档 bank 0 最优）；
且在 32K ctx + KV 在 GPU 上，分块后的 bank 能安全用回 4096（`used`、`reallocs`、`hit %` 进表）。

### F9 RAM 侧改成有界热专家窗口（**新增，P2，但它是"跑更大模型"的唯一路径**）

现状是全有或全无：`common/fit.cpp:987-995` 把 `mmap` / `mlock` / `dio` 一律判成非 pinned -> `bank = 0` -> 预取关；
`src/llama-moe-cache.cpp:208-213` 又要求专家整层常驻。于是 20.2 GiB 模型在 40 GiB 机器上只有"整模型 resident"一条路，
模型再大一点、或桌面/浏览器吃掉内存，就整体退到无 cache。
目标口径已经量过这条退化不是免费的：`-lm mmap` 让 PP 178.5 -> 142.5（本文件 F2 的读数）。

边界要一起尊重：pageable 源拷贝慢数倍（`DECISIONS.md` 判决 1、第 3 节 B1），
所以**不是**解除守卫让 bank 直接从 mmap 页面里读。做法是复用现成件，不加新子系统：

1. 有界 pinned 窗口：专家独享一块固定大小的 pinned host 池，大小由 fit 在 **RAM 预算里解出来**
   （而不是回答"整模型装得下吗"这个布尔问题），loader 用现有区间读 + 批量顺序 pread 填窗口。
2. 三者一起算：RAM 侧 = dense + pinned 窗口 +（`-nkvo` 时的）KV，输出的是"窗口多大、哪些层驻留"。
3. SSD 只当窗口的填充来源，仍受第 1 节约束：只进 prefill 与预取批次，绝不进 decode 单步关键路径。
4. **不许把 85% 命中率当"不会等盘"的前提**：decode 稳态工作集必须装得进 `窗口 + bank`；
   装不下时在规划阶段就把该层标成 host/CPU 或降精度，而不是运行中撞到 miss 才去读盘。

验收：一条 RAM 受限 profile（把可用内存压到 resident 门槛以下）里 (a) bank 非零、
(b) decode 关键路径 pread 次数 = 0（`GGML_SCHED_TRACE` 证明）、(c) pp2048 >= 同配置 resident 的 70%、(d) `ctest` 32/32。
混合精度那条腿（热 Q4_K + 冷 IQ2/IQ3）仍先受 F6 与 repack KLD 的门禁。

## 4. 建议顺序

以第 0 节为最高优先级，本节保留原有的分级依据。

P0 先跑通并解耦：F1（`:1910` 的 producer/lookup 边界要在调度分片阶段解决；调度器的 pre/post 回调都在 split
计算的前后且返回 `void`，不能把 producer 挪进前一个 split，删守卫更不行）+ F7（先归因再解耦）+ F8 的三项归因
（score 预留的设备归属、1712.02 MiB 设备 compute 的归属、按 query 分块的 PP 代价）。
P0 里 F1 是功能前提，F7/F8 是本口径 PP/TG 最大的两块，三者都要在同一轮矩阵里读数。

P1 同配置最优：F8 的实现（预算归属 + 按 query 分块）-> F2 的三档最优 bank -> F3 预取门槛自适应。

P2 跑更大模型：F9 有界 pinned 窗口 -> F5 SSD 冷层 -> F4 Vulkan `MUL_MAT_ID_MASK`（CPU 就地算冷专家的前置）。
F6 是 P2 与 F8 分块改动的验收前提，不是可以并行推的独立任务。

混合精度分层仍是唯一能改变 TG 屋顶线的选项（屋顶线见 `RESULTS.md` 第 5 节），但它的前提是 F6 + repack KLD = 0。
