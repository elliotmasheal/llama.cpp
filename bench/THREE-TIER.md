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
  这要动 FA/matmul 注意力路径，风险大，先不做。**2026-10-23 更新：该方向由 F10（分块 KV + 块表间接寻址）取代，
  不再用 LSE 合并两段，改为所有块在同一注意力内统一寻址。**
- 验收：256K + 10K prompt、8176/6144/4096 三档全部 `status = ok`，KV 驻设备部分约 205 MiB，
  splits 从 291 降到 40-121（取决于 bank，bank 0 时 121）；PP、TG 相对同 n_kv 的 32K + KV-on-GPU 对照在 r=2 噪声内持平。
  **已达成**（`RESULTS.md` 0.3）：KV 驻 320 MiB，PP 447-478（对照 32K + KV-on-GPU 的 472.3），TG 12-14（对照 12.9-15.4）。
  预期（按 `RESULTS.md` 0.2 表 B 的同批对照，bank 0）：PP 178 -> 约 470、TG 6.4 -> 约 13；
  若 A 同时让 bank 非零，PP 还会再高（bank 1719 的对照是 483.6 / 15.4）。

### B（P0）：注意力按 query 分块 + 预留随块大小，不随 n_ctx（即 `THREE-TIER.md` F8 的实现形态）

**B1（预留随工作集）已由 A 自动达成**：`RESULTS.md` 0.3 的 1490.02 -> 708.02 MiB 就是它，不需要单独改代码。

**B2（按 query 分块）已落地并验收**（`RESULTS.md` 0.4）：开关 `-fac/--fa-chunk-mib`（默认 256 MiB），
分块不改变数学（质量门 KLD -0.000000、top-p 100%，见 `RESULTS.md` 7.1）。8176 档 bank 1481 -> 2879 MiB、
每组 slots 406 -> 791（真的跨过双环门槛）、splits 114 -> 40、PP 478 -> 680、TG 13.7 -> 20.4；
6144 档仍差 61 slots/组（F3），4096 档 bank 仍是 0（与 score 无关）。下面保留当时的设计依据。

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
- 代价：分块后 dispatch 变多。**实测分块与整段逐位一致**（`RESULTS.md` 7.1 的 KLD -0.000000、top-p 100%），
  因为 softmax 的归约范围本来就是每条 query 自己那一行，切分不改变它。
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
- 10K 口径下 q8_0 与 f16 的 PP 平价已测（`DECISIONS.md` 第 3 节），但**质量还没测**。
  F6 的通道已就位（`bash bench/tools/quality-gate.sh`），把它当默认之前先补 q8 的 KLD 读数。

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
   **B2 随后落地并验收**（`RESULTS.md` 0.4，F6 先行）：8176 档 PP 478 -> 680、TG 13.7 -> 20.4，三档 ok。
   下一步按收益排：F3（6144 差 61 slots/组就能开预取）-> E（`ub 2048/3072/4096` 联合调参，B2 已解锁）
   -> C（前缀检查点）与 D（预算式一体化）。
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

### F6 质量门（**已落地，2026-10-23**）

`tools/perplexity/` 已从 git 历史恢复，一条命令是 `bash bench/tools/quality-gate.sh`
（`-R` 重录基线 `F:/LLM/kld/qwen35moe-10k-c4096.kld`，2.0 GB，不进 git）。
口径：`prompt10k.txt`、`-c 4096 --chunks 2`、`-ncmoe 41`、`-lm none`、`-fa on`，覆盖长跨距 matmul 路径。
基线 PPL 5.798313 ± 0.237848；门槛是 |mean KLD| <= 1e-4 且 same top p >= 99.5%（脚本自己判）。
B2 用它验收过（KLD -0.000000、top-p 100%，`RESULTS.md` 7.1）。
剩下的局限：10K 文本只能跑 `-c <= 5214`，所以质量门不是 256K 口径；它验证的是算法等价，不是长上下文数值。
F5/G 的降精度仍需这个门。

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
   代价是 dispatch 变多，要用 PP 实测抵掉；分块与整段实测逐位一致（KLD 0）。
3. **设备 compute 本身没查过归属**：`sched_reserve` 报的设备 compute 是 **1712.02 MiB**，三档完全一致，
   而 32K ctx 那批只有 772.0 MiB。这两个读数同时变了两个变量（`n_ctx` 与 KV 放置），所以"约 940 MiB 随 `n_ctx` 走"目前只是假设，
   要补的是同一 `-nkvo` 放置下 32K/64K/256K 的 `sched_reserve` 设备 compute 对照。
   **2026-10-23 已闭环**（`RESULTS.md` 0.2 表 A）：同 `-nkvo`、设备 KV 恒为 0 的七点扫描给出
   578.02（16K）-> 690.02（32K）-> 738.02（64K）-> 930.02（96K）-> 1154.02（128K）-> 1328.02（192K）-> 1712.02（256K），
   即 **+1134.0 MiB 确实随 `n_ctx` 走**，比假设的 940 MiB 更大；host 侧 compute 精确等于 `96.18 + 0.2 x KV`。
   同批 4096 档的 teardown warning 同向：host compute 从 1120.18 涨到 1446.35 MiB。
   这就是 4 GiB 档的钥匙：reserve 图要按 `n_kv` 工作集建，而不是按 `n_ctx`。

**2026-10-23 落地结果（B2，`RESULTS.md` 0.4）**：第 2 条改成按预算分块，预算由 `-fac/--fa-chunk-mib` 给（默认 256 MiB），
`src/llama-graph.cpp` 的 matmul 注意力按 query 维切块、mask 同步切片、输出沿 query 维 concat；
规划器的 score 预留改读同一个上限（`llama_fa_long_kv_reserve_bytes(..., fa_chunk_mib)`）。
8176 档 bank 2879 MiB（>= 反解值 3084 的大半）、预取真的打开（791 slots/组）、splits 40、PP 680、TG 20.4。
6144 档 bank 1644 MiB（451 slots/组），**还差 61 slots** 才开预取 -> 归到 F3。
4096 档第 3 条的结论仍然成立：score 从 1536 降到 256 之后 bank 还是 0，这一档缺的是别的归属。
第 1 条的"host 归属"只在 `-nkvo` 臂上还有意义，目标口径已不需要 `-nkvo`（KV 在设备）。

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

### F10 分块（paged）KV：块表间接寻址，原地扩展（**新增，容量路线，取代 A 第二阶段的 LSE 分段**）

#### 0. 结论先行

- 原地扩展**可以实现**，但不是"在原 buffer 后面接着长"（Vulkan 不能 realloc，且 tensor stride 固定），
  而是把 KV 从"一张 `[n_embd, n_ctx]` 大张量"改成"固定大小的块 + 块表"，即 vLLM PagedAttention 的同一思路：
  - 逻辑 cell 下标不变（`find_slot`、`k_idxs`、kq_mask、`get_n_kv` 全部照旧），只把 `cell -> 物理地址` 交给一张小表；
  - 扩容 = 分配一个新块（BS=2048 时 40 MiB）并追加表项，**不拷贝、不重建已有张量、不重新预留图**；
  - 峰值从 `cur + next`（约 1.5 x 新尺寸）降到 `cur + 1 块`。
- Vulkan 侧的支撑已在树里：所有 storage buffer 在设备支持时带 `eShaderDeviceAddress`（`ggml-vulkan-buffers.cpp:64-69`），
  `bda_addr` 已记录（`:199`）；cm2 FA 已用 `buffer_reference` 解引用 K/V（`flash_attn_cm2.comp:21,44-47`）。
  块表存 64 位设备地址，shader 直接访问，**不需要把各块合并成一张大 buffer**。
- **它解决的是容量，不是 decode 速度**。full attention 每个 token 都读全部历史 cell，没有"热冷"局部性；
  落在 host 的块每 token 多付一次 PCIe（实测 H2D 14.3 GB/s，第 1 节）。成本模型见第 9 节。SSD 仍不进 decode。
- 推荐顺序：M0 spike（确认 BDA、FA 实际路径、表读取开销）-> M1 CPU 参考 -> M2a B-lite 验证 -> M2b 融合 FA。
  M0-M2b 只改寻址、不改 FA 数学，数值应当**逐位相同**，这是最硬的验收手段，**不依赖 F6**。
- **M0 已完成（2026-10-23）**，读数在 `RESULTS.md` 6.1：路径确认为 `FA_SCALAR`（PP 与 TG 同一条），`Bc = 32`，
  目标工况 `sparse = 0`；BDA + `uint64` 块表在 3 个独立分配上读写都正确；按块寻址在 DRAM 规模下差 <= 1%
  （`-5.6% ~ +0.9%`），10K 热缓存点 +3.2%，而真实 FA 每 32 个 cell 才查一次表，所以第 5.7 节的上限成立。
  表通过 `op_params` 传（见 4 节），不占 `src` 槽、不在锚上挂元数据。

#### 1. 问题与动机

A 第一阶段有三个结构性问题，它们都来自"一张连续大张量"：

- 扩容是整块重分配：`realloc_cache` 先为所有层分配新 buffer（`src/llama-kv-cache.cpp:1377-1386`），
  旧 buffer 直到 `:1429` 才释放，峰值是 `cur + next`。
- 失败是整棵 cache 粘住 host：`host_fallback` 是 cache 级标志（`src/llama-kv-cache.h:283`），
  一次设备分配失败后整棵 KV 都在 host（`src/llama-kv-cache.cpp:1283-1291`）。
- state 恢复不扩容：`state_read_meta` 在 `cell_count > cells.size()` 时直接失败（`src/llama-kv-cache.cpp:2575-2577`），
  这是 `DECISIONS.md` 第 6 节未列的缺口，也是 server 的 "No available space in KV cache" 的潜在来源。

分块后三者同时消失（M3 验收）。

#### 2. 目标 / 非目标 / 不变量

目标：

1. 扩容零拷贝，峰值 = 1 块；
2. 放置粒度 = 块（device-local / host-visible），一次设备分配失败只影响那一块；
3. 缩容可行（clear 或显式回收尾块）；
4. 第一版数值逐位等于现有路径。

非目标（第一版）：

- 量化 K/V 的分页（q8_0 等）-> M5；
- 块级前缀共享 / copy-on-write（vLLM prefix caching）-> M5，与 `THREE-TIER.md` C 的检查点合并考虑；
- SSD 块作为 decode 访问层：永久禁止（红线）。

不变量（实现时必须保持）：

- 逻辑 cell 下标语义不变；`llama_kv_cells::grow`（`src/llama-kv-cells.h:79`）原样复用；
- `get_n_kv()` 只看 `used_max`（`src/llama-kv-cache.cpp:1494-1508`），与分配量无关，图拓扑不因分配而变；
- 块表长度固定为 `n_ctx / BS` 项（256K / 2048 = 128 项，1 KiB），图输入尺寸不变，图复用不被打断；
- 容量不足 = 退化而非 abort（红线）。

#### 3. 数据结构

- **块 `kv_block`**：BS 个 cell 的全部注意力层 K/V，是一个 ggml backend buffer（默认 device-local，可选 host-visible）。
  块内布局 `[layer][K|V][cell][row]`，同一层在块内是连续的 `BS x row` 矩阵，FA tile 的局部性不变。
  每 cell 20 KiB（10 层 x K/V x f16，与 `RESULTS.md` 0.3 的 160 MiB / 8192 cells 一致），BS=2048 时块为 40 MiB。
- **块表 `kv_block_table`**：每 stream 一张，`uint64 addr[n_ctx / BS]`，Vulkan 时放在设备 SSBO，CPU 时放在主机数组。
  `addr[b]` 为块 b 的基址（Vulkan 为 BDA，CPU 为指针），层内偏移为 `layer_idx * layer_stride + kv_off`，由 shader 计算。
- **锚张量 `anchor`**：一个不分配数据的 f16 ggml 张量，`ne = [n_embd_k_gqa, n_ctx, n_stream]`，挂在设备 buffer 上，
  只用来让调度器把依赖它的 FA/SET_ROWS 放到正确 backend。`get_k/get_v` 继续返回对锚的 `ggml_view_4d`，
  `ne/nb` 与现在一致，`src/llama-kv-cache.cpp:1510-1550` 的视图代码基本不动。
  待 M0 确认：调度器对"无真实数据的 view"按 `view_src` 的 buffer 分配 backend（读 `ggml-backend.cpp` 的 split 逻辑）。
- **分配器**：空闲链表 + 每块状态 `{addr, placement(dev|host), refcount(M5 预留)}`。
  `ensure_blocks(n_extra)` 取代 `ensure_capacity`：`need = used_max + n_extra`，`while cap < need: alloc_block()`。

#### 3.1 块大小（BS）选型：容量粒度参数，不是性能参数

BS 只影响以下四项，其中没有一项直接作用在 PP/TG 的算子上：

| 影响面 | BS=2048（40 MiB/块） | BS=8192（160 MiB/块） | 对 PP/TG 的量级 |
|---|---|---|---|
| 增长粒度与最坏浪费 | <= 40 MiB | <= 160 MiB | 噪声内。实测 160 对 320 MiB 的 KV 无差（都 477），320 MiB 对 5120 MiB 才有差（477 对 239，`RESULTS.md` 0.2 表 B） |
| host/device 边界粒度 | 剩余 40 MiB 即可多留一块 | 剩余 160 MiB 才多留一块 | 唯一真实影响：边界上最多差 160 MiB 的 host 流量（约 11 ms/token，见 9 节） |
| 表项数 | 128 项 / 1 KiB（256K） | 32 项 / 256 B | 无 |
| FA 表查询次数 | 按 K/V tile 计数，与 BS 无关（BS 只需 >= tile 且为其倍数） | 同 | 无 |
| B-lite gather 的 dispatch 数与 scratch | 更少 / 更小 | 更多 / 更大 | 只影响验证路径（M2a）的 PP |

结论：默认 2048，理由是边界粒度细、浪费小，不是因为更快。8192 只在 M2b 实测证明 dispatch 或表查询开销不可忽略时才启用。

#### 4. ggml 与图层接口

- **SET_ROWS**（`GGML_OP_SET_ROWS`，`ggml/include/ggml.h:538`）：dst 为分页锚时，目标地址 =
  `addr[idx / BS] + layer_off + (idx % BS) * row_bytes`。在 op 参数里置分页标志，表作为额外输入。
- **FLASH_ATTN_EXT**（`ggml/include/ggml.h:2503`）：K/V 为分页锚视图时，每个 K/V tile 的地址由表查出。表的挂法有两个候选：
  (a) 新增一个 `src` 槽（需核对 `ggml_flash_attn_ext_add_sinks` 占用的槽位）；
  (b) 在锚上挂一个分页元数据字段，不占 `src`。优先 (b)：改动小，不影响其他 op。M0 决定。
- **M0 定案（2026-10-23）**：两个候选都不用，改为**把层的表地址写进 `op_params`**：
  FA 用槽 5-6 存 `uint64 table`、槽 7 存 `BS`（槽 0-2 是 scale/max_bias/softcap，3 是 prec，4 是 `n_kv_max`，剩余槽充足）；
  SET_ROWS 用槽 0-1 存表地址、槽 2 存 `BS`。表地址在图的生命周期内不变（块只增不改表基址），
  所以 `op_params` 恒定，图复用不受影响；也不需要新的 `src` 槽或后端扩展字段。
  API 是 `ggml_flash_attn_ext_set_paged(a, table_addr, BS)` / `ggml_set_rows_set_paged(...)`，
  内核从张量的 `nb` 取每 cell 步长：分页后 `kb[1]` 恰好等于 `n_embd_k_gqa * type_size`（锚是虚拟连续布局）。
  表本身是 `uint64[n_layer_kv][2][n_blocks_max]`，每个 (层, K|V) 一段连续数组，
  传进 op 的是该层那一段的地址；层内偏移由"表项指向该层的块内区域"吸收，内核不做层算术。
- 复用：不新增 op，不新增并行子系统（符合 AGENTS "复用现有基础设施"）。

#### 5. Vulkan 专项

5.1 BDA 基础设施（已具备，只需小改）

- `device->buffer_device_address` 为真时（`ggml/src/ggml-vulkan/ggml-vulkan.cpp:4589`）
  storage buffer 带 `eShaderDeviceAddress`（`ggml-vulkan-buffers.cpp:64-69`）。
  需新增访问器 `ggml_backend_vk_buffer_get_device_address(buffer)`，放进 `ggml/include/ggml-vulkan.h`，风格与 `ggml_backend_vk_prefetch_tensor` 一致。
- 扩展已在用：`GL_EXT_buffer_reference2`（`flash_attn_cm2.comp:21`）、`GL_EXT_buffer_reference`（`mul_mm_cm2.comp:17`）；
  `uint64` 表依赖 `device->shader_int64`（`ggml-vulkan.cpp:4588`）。
  **M0c 已在 RX 6700S 上实测**：`bufferDeviceAddress=1`、`shaderInt64=1`、`storageBuffer16BitAccess=1`，
  且 3 个独立 `VkBuffer` 的地址表读写都正确（`bench/tools/vkpage.cpp`，`RESULTS.md` 6.1）。

5.2 FA shader 改点（FA_SCALAR 路径）

- **M0b 定案**：PP 与 TG 都是 `FA_SCALAR`（RDNA2 无 coopmat），没有 Intel Xe 的 decode phase 变体，
  所以只需要改 `flash_attn.comp` 这一条；`flash_attn_decode_phase_*.comp` 不进目标路径。
  实测参数：`Br=8`、`Bc=32`、`D_split=8`、`row_split=4`、subgroup 64、`wg=256`、`aligned=1`、`f32acc=1`。
- 需改的 f16 读取点：`flash_attn.comp:266, 322, 347`（`data_kv4[k_offset/4 + kcol*k_stride/4 + ...]`）及 V 侧对应读取；
  `k_offset` 在 `:182`，`k_stride` 在 `flash_attn_base.glsl:208`。改成按 tile 解析块：`blk = kcol / BS`，`row = kcol % BS`，
  地址 = `addr[blk] + layer_off + row * row_bytes + d`。
- `vulkan-shaders/` 下另有 `flash_attn_decode_phase_1.comp` / `flash_attn_decode_phase_2.comp`：M0 确认 TG 实际走哪一条，生效的那条同样要改。
- 量化 K/V（`dequantize4`、`get_k_qs`，`flash_attn.comp:400-413`，mmq 变体）第一版**不支持**：分页要求 F16 K/V，其他类型退回连续路径并记录原因。

5.3 tile 对齐（硬约束）

- 每个 K tile（`BLOCK_SIZE_K` 或 Bc，M0 查清具体值）必须落在单个块内：`BS % BLOCK_SIZE_K == 0`，且 `BS % 256 == 0`
  （与 `get_n_kv` 的 padding 粒度一致，`src/llama-kv-cache.cpp:1499`）。
- BS 候选 1024 / 2048 / 4096 / 8192，默认 2048。满足约束后，shader 每个 tile 只查一次表，内层循环不做除法。

5.4 表更新与同步

- 追加块是 host 写表（`ggml_backend_tensor_set`，8 字节项）。必须在上一张图完成之后做：入口处 `llama_synchronize(lctx)`，
  照 `update()` 的做法（`src/llama-kv-cache.cpp:863`）。释放块同样只在同步点发生，走延迟释放列表。
- 表项为 0 时 shader 不得解引用：host 侧保证 `need <= cap`。调试开关下 shader 对 `addr == 0` 写 sentinel 并计数，避免 GPU fault 导致 device lost。

5.5 放置与显存限额

- `GGML_VK_VRAM_LIMIT_MIB`（F0）经 hold-back 让新块分配失败。失败时**只把这一块放到 host-visible 内存**，
  而不是整棵 cache 粘住。这是相对现状最大的改进。
- host-visible 块：BDA 同样可用，FA 直接经 PCIe 读。它的作用是容量，不是 TG 加速。

5.6 分配失败的退化路径（已具备）

- `ggml_backend_vk_buffer_type_alloc_buffer` 捕获 `vk::SystemError` 并返回 nullptr（`ggml-vulkan.cpp:13315-13317`），
  不 abort，可直接用作逐块降级的判据。

5.7 性能假设（**M0d 已测，2026-10-23**）

- 标量 FA 在 tile 边界查表后，预期每 tile 增加 1 次表读 + 1 次 64 位加法，开销 < 1%。
- M0d 读数（`RESULTS.md` 6.1，每 cell 查一次表的上界）：DRAM 规模下 +0.4% ~ +0.9%（128K / 256K），
  64 MiB 点 -5.6%，10K 热缓存点 +3.2%。真实 FA 每 `Bc=32` 个 cell 才查一次，所以假设成立。
- **测量陷阱**：同一探针在 `chunk=1024`（10K 只有 10 个 workgroup）给出 +69%，那是占用率不是表寻址。

5.8 与现有 opt-in 路径的冲突（M0b 发现）

- `use_dequant_kv`（`ggml-vulkan.cpp:8296`）在 `n_kv == kv_size` 时会把整段 K/V 拷进 f16 scratch。
  分页后 `is_dense_kv_cache()` 仍会返回真（锚的 `nb` 是虚拟连续布局），因此分页模式必须显式关掉它，
  否则 shader 会去读块 0 的连续内存。
- `use_sparse`（`n_kv_max`）在目标工况实测为 0，分页第一版不实现"经表的 gather"，遇到就退回连续路径并计数。

#### 6. CPU 参考实现（M1）

- `ggml-cpu` 的 FLASH_ATTN_EXT 与 SET_ROWS 按同一张表求地址（指针算术）。它是 M1 的正确性基准，也是 test-backend-ops 的参考实现。
- CPU 块用对齐的 host 内存分配，沿用 `ggml_backend_cpu_buffer_type`。

#### 7. 与现有行为的对接

| 位置 | 现状 | F10 后 |
|---|---|---|
| `src/llama-kv-cache.cpp:1253` `ensure_capacity` | 整块重分配 | `ensure_blocks`，追加块，不拷贝 |
| `src/llama-kv-cache.cpp:1299` `realloc_cache` | 全量重建 | 分页模式下删除该路径 |
| `:1510` / `:1530` get_k / get_v | 对大张量取视图 | 对锚取视图（ne/nb 不变） |
| `:1594` / `:1629` / `:1650` cpy_k / cpy_v | SET_ROWS 写大张量 | SET_ROWS 写锚（分页变体） |
| `:2233` build_graph_shift | 对整张量做 RoPE | 逐块 `ggml_view_2d` 做 RoPE（块内连续） |
| `:2468` state_write_data / `:2747` state_read_data | 按 cell range 读写大张量 | 按块拆分 range，每块一次 `ggml_backend_tensor_get/set` |
| `:2569` state_read_meta | cell 超出即失败 | 先 `ensure_blocks(cell_count)` 再读，修掉 "not enough cells" |
| `src/llama-context.cpp:2043` | 增长后置 `sched_need_reserve` | 分页模式下 n_kv 不变，不再重预留（顺带收掉 `DECISIONS.md` 第 6 节第 11 条的 (b)） |
| `src/llama-memory-hybrid.cpp:35` | 只挂 `llama_kv_cache` | 不变，recurrent 不参与 |

自动退回条件（任一成立就走连续路径，计数告警，不 abort）：`flash_attn` 关闭、K/V 非 F16、`v_trans`、
`other != nullptr`（共享 cells，MTP / qwen4exp）、`layer_share` 复用张量、`hparams.no_alloc`。
与 `ensure_capacity` 现有守卫一一对应（`src/llama-kv-cache.cpp:1254-1265`）。

#### 8. 开关与默认

- `--kv-page auto|on|off`（env `LLAMA_ARG_KV_PAGE`）、`--kv-block N`（默认 2048，必须是 256 的倍数）。
- 默认值：M2b 之前为 off；M2b 过门禁并在目标口径 256K + 10K 三档 r=2 上 PP/TG 不退步后，改 auto，并登记进 `DECISIONS.md` 第 4 节。
- 每次退回都打印一次原因（`kv-page disabled: <reason>`）。

#### 9. 成本模型（M0/M2 实测前的估算基线）

9.0 先纠正一个容易误读的前提

**分页本身不产生 PCIe 流量。** 全部块在 VRAM 时每 token 的 PCIe = 0，与 KV 全驻设备相同；
M2a 的 gather 是显存内 D2D，走显存带宽而不是 PCIe。**只有块落在 host 时才有 PCIe**，
那是"容量装不下"的固有代价，不是分页引入的代价。相对现状 `-nkvo`（100% host，每 token 全量上传）
分页只会减少 PCIe；相对 KV 全驻 VRAM 则是中立的。
唯一必须守住的一条：**块表必须驻设备**（256K 时 1 KiB）。表在 host 会被每个 tile 读一次，那才是真正的灾难。

9.1 TG（单一 token 查询，无内容可重叠，损失是直接的）

口径：20 KiB/cell、256K 满填 = 5120 MiB、PCIe 14.3 GB/s（第 1 节实测）、基线 73 ms/token（13.7 t/s）。

| host 块占比 | 每 token 字节 | PCIe 时间 | 结论 |
|---|---:|---:|---|
| 0（全 VRAM） | 0 | 0 | 基线 |
| 5%（0.25 GiB） | 256 MiB | 18 ms | 可被流水预取隐藏 |
| 10%（0.5 GiB） | 512 MiB | 36 ms | 可被流水预取隐藏 |
| **20%（1 GiB）** | 1 GiB | **72 ms** | **临界：约等于一次完整的单 token 计算预算** |
| 40%（2 GiB） | 2 GiB | 143 ms | 不可隐藏，TG 约 4.7 t/s |
| 100%（现状 `-nkvo`） | 5 GiB | 359 ms | 约 2.3 t/s，且注意力可能被拖到 CPU，实际更差 |

注意上表的基线是 10K 口径。256K 满填时基线本身也会变慢（仅从 VRAM 读 5 GiB KV 就要约 17 ms），
所以表里的 TG 是上限估计，不是预测值。

9.2 PP（可重叠，且摊在整段 prefill 上）

每个 ubatch 都要从 host 读一次当前历史范围内的全部 host 块：

```
host 字节总量 ~= f_host x 20 KiB x N^2 / (2 x n_ubatch)
```

N = 262144、`-ub 2048` 时：f_host = 1 约 320 GiB -> 22 s，占 256K prefill（约 870 s 量级）的 2-3%；
f_host = 0.2 约 4.5 s -> 约 0.5%，且 H2D 与 shader 可并行。
**PP 不是这个方案的问题，TG 才是。**

9.3 缓解梯度（按性价比，全部需要实测）

1. 流水预取（无质量风险，优先做）：host 块按层预取，复用 MoE 的 2 槽环机制（`LLAMA_MOE_PREFETCH`），
   让第 L 层的 H2D 与第 L-1 层的计算重叠。隐藏上限就是 9.1 的临界值：**host 侧 1 GiB 以内基本免费**，
   1-2 GiB 部分隐藏，超过 2 GiB 按字节线性付费。
2. 冷块降精度：host 块按 q8_0 存（10 KiB/cell），读入后反量化到 scratch，PCIe 减半，
   代价是一遍显存内反量化（host 字节的 2 倍，约 14 ms/GiB）。改数值，受 F6 门禁。
3. 减少 n_kv：见 9.4，是唯一从根上把 decode 变便宜的路径。
4. 减少每 cell 字节：`-ctk/-ctv q8_0`（5120 -> 2660 MiB），与 bank 空间互挤，受 F6 门禁。

9.4 显存侧的两个对照量（说明分页把成本从显存挪到了 PCIe，而不是凭空增加）

- 全驻 VRAM 时每 token 的 KV 读取本身有下限：256K 满填 5 GiB / 300 GB/s 约 17 ms。
  即 PCIe 为 0 时，256K 满上下文的 KV 读也要约 17 ms。这是显存带宽的地板，不是分页的成本。
- B-lite（M2a）的 gather 是显存内 D2D，额外约 2 x n_kv x 20 KiB：10K 约 1.4 ms/token（约占 2%）、
  128K 约 18 ms、256K 约 36 ms。**只用于验证，长上下文下不可作为默认。**

9.5 上下文压缩与前缀缓存（agent / 多轮场景的硬约束）

> **2026-10-23 决定不做**：丢中间段会让输出偏离预期，M5 的压缩分支不启动。
> 本节保留作为将来重启时的约束清单；下面的三条硬约束与策略要求都不实现。

压缩必须做在 KV 层，不能做在 token 层。server 的复用判定只看 token 列表：
`n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`（`tools/server/server-context.cpp:3972`），
slot 选择也基于 LCP（`:1812`）。所以"改写 / 摘要 prompt 文本"会让前缀缓存全废、每轮全量重算；
而"只丢 KV cell、token 原文逐字不变"则照常命中，PP 只算新增。

三条代码级约束（实现前必须满足）：

1. **现有 API 删不了头部**。`llama_memory_hybrid::seq_rm` 先问 recurrent（`src/llama-memory-hybrid.cpp:146-153`），
   而 `llama_memory_recurrent::seq_rm` 只支持全删与尾回滚（`n_rs_seq`），删头部直接返回 false
   （`src/llama-memory-recurrent.cpp:184-232`）。所以要新增一个只动 attention 段的压缩操作；
   语义成立，因为 30 层 GDN 的 RS 已经把被丢的那段总结进去了。
2. **命中判定不看 `seq_pos_min`**。压缩后若新请求只匹配被丢掉的前段，LCP 给出很大的 `n_past`，
   但那些 cell 已不存在，会静默漏掉历史。恢复前必须校验 `n_past >= seq_pos_min`，否则重算。
3. **只留洞不省时间**。`get_n_kv()` 取 `used_max`（`src/llama-kv-cache.cpp:1494-1508`），低地址留洞不降 `n_kv`，
   FA 仍遍历整段：只省显存，不省 PP/TG。必须压缩 cell 编号。
   分页让这一步免费：重排 = 改块表顺序，不搬数据；无分页则要 defrag 搬 cell。

策略要求：保留边界必须是"位置 / 固定窗口"的确定性函数，**不得由当前剩余显存决定**，
否则同一 prompt 在不同时刻得到不同状态，缓存不可复用、输出不可复现。逻辑保留集合与物理放置解耦（M4）。
压缩后的状态要能作为缓存对象保存与装载（内存 prompt cache、SSD 检查点，与 C 合并），
键 = 完整 token 前缀哈希 + 压缩策略版本；装载前 `ensure_blocks`（M3）。

PP 成本分三层（256K 口径）：

| 场景 | PP 成本 | 依据 |
|---|---|---|
| 无前缀缓存，每轮全量重算 | 约 550 s/轮 | 10K 实测 22 s、PP 478（`RESULTS.md` 0.3）线性外推，是下界 |
| 有前缀缓存，每轮只算新增 | 每轮几十秒（估算，未实测） | 新增 token 仍 attend 全部历史，成本随 `n_kv` 线性 |
| 有前缀缓存 + 压缩到 N' | 按 `N' / n_kv` 线性下降 | 注意力部分随 `n_kv` 走 |

第二行估算区间：每 token = 专家（常数）+ 注意力（随 `n_kv`）。10K 时 2.09 ms/token，
256K 时注意力放大 26 倍；若注意力在 10K 占 20-40%，每 token 约 13-23 ms，新增 2K token 的一轮约 25-46 s。
**未实测，M4 的 256K 满填臂就是为了量它，不得当结论引用。**

质量保护（压缩一旦启用）：system prompt / 工具定义 / 最近 N 轮钉住不压，只压中间最老的部分，
优先"冷块降精度"（q8_0 存 host、读入反量化），最后才整块丢弃。压缩改数值，必须过质量门
（F6 已落地，`bash bench/tools/quality-gate.sh`，门槛 |mean KLD| <= 1e-4 且 same top p >= 99.5%）。

与压缩正交的 PP 手段：F8/B2 的按 query 分块、更大的 `-ub`、C 的 SSD 前缀检查点。不改变上下文语义，可独立推进。


#### 10. 里程碑与验收

M0 spike（前置，不改主线）**已完成 2026-10-23，读数见 `RESULTS.md` 6.1**

- 0a. ~~补完 `FLASH_ATTN_EXT -b Vulkan1` 的约 22 例~~ **已解除**（2026-10-23：5332/5332 已补齐，`DECISIONS.md` 第 6 节）。
- 0b. ~~用日志确认 RX 6700S 上 PP、TG 实际走的 FA 路径~~ **已确认**：PP 与 TG 都是 `FA_SCALAR`，
  `Br=8 Bc=32 D_split=8 row_split=4 sgs=64 wg=256 aligned f32acc`，没有 decode phase 变体；
  `sparse=0`；`dequant_kv` 只在 `n_kv == kv_size` 的那一批出现。实现方式：`GGML_VK_FA_TRACE=1` 打每个不同形状一行。
- 0c. ~~独立 Vulkan 小测试：两个 buffer，块表（uint64）经 BDA 访问，表内块不连续时结果正确~~ **PASS**：
  3 个独立 `VkBuffer`、8 个逻辑块乱序映射，读经表/写经表/未映射槽位全 0 三项都过（`bench/tools/vkpage.cpp`）。
- 0d. ~~手写一个"读表版"标量 FA 微基准~~ **已测**：DRAM 规模 +0.4%~0.9%，10K 热缓存 +3.2%，
  且真实 FA 的查表密度是它的 1/32。
- 验收：0b 结论落地；0c 通过；0d 开销 < 1%（按 tile 密度折算），结论：**分页寻址不构成 PP/TG 风险**。

M1 CPU 参考 + 分配器（默认 off）**已完成 2026-10-23，读数见 `RESULTS.md` 6.2**

- 块分配器（`init_pages` / `add_block` / `ensure_blocks`）、块表、锚、`get_k/get_v` 的锚视图、
  SET_ROWS / FA 的 CPU 分页实现、`--kv-page` / `--kv-block` 开关、state 按块读写、分页下禁用 K-shift。
- 验收：`tests/test-kv-page.cpp`（进 CTest）里 paged FA 与 contiguous FA **4096/4096 逐位相同**
  （8 个逻辑块乱序映射、`BS=32`），paged SET_ROWS 0/64 错位、锚未被写坏；`ctest` **33/33**；
  CPU 后端整模型 512 cells 两臂输出逐字相同。
- 与原计划的偏差：验收写的是"CPU 上 8K/16K 两点 logits 逐位相同"，实际做的是**算子级 256 cells 逐位 + 模型级 512 cells 逐字**。
  原因：20.2 GiB 模型在 CPU 上 prefill 8K 要小时级，而算子级已经覆盖了表的全部语义（乱序块 + 非连续 cell）。
  真正的长上下文逐位验收放在 M2b 的 256K + 10K 口径。
- 已知遗留：设备 K/V 被守卫挡下（`device K/V needs the paged Vulkan kernel (M2b)`）；
  K-shift 逐块化与 server 的 slot 恢复压力测试留到 M3。

M2a B-lite（验证，不改 FA shader）

- 每层每块 `ggml_cpy` 把块内连续行拷进 scratch，FA 走原连续路径。只用于验证 M1 的表与块语义，性能不作目标。
- 验收：Vulkan 上 `--kv-page on`（B-lite）与 `off` 的生成文本在 temp 0 下逐字相同，10K 与 40K 两个 prompt。

M2b 融合 FA（Vulkan）

- 分页 SET_ROWS 变体；`flash_attn.comp` 的 f16 读取改表（5.2）；分页模式强制 FA_SCALAR（RDNA2 本来就走它）。
- 门禁（AGENTS 第 3 节）：`MUL_MAT_ID` CPU 945/945、`MUL_MAT_ID_MASK` CPU 4/4、`MUL_MAT_ID` Vulkan 939/939、
  `MUL_MAT` Vulkan 2275/2275、`FLASH_ATTN_EXT` Vulkan 5332/5332、`ctest` 32/32。
- 验收：256K + 10K 三档 `status = ok`；逐位相同；PP/TG 与 `off` 在 r=2 噪声内持平；`reallocs` 列为 0（扩容不再触发重分配）。

M3 状态与清理

- state_write / state_read 按块；`state_read` 前调用 `ensure_blocks`；K-shift 逐块；clear 时回收块。
- 验收：server 的 slot save/restore 在"快照 token 数 > 当前容量"时通过；上下文平移（K-shift）测试通过。

M4 放置与三层

- 块级 device/host 放置，显存限额失败时逐块降级；`used`、`held`、host 块数进矩阵。
  回归确认 F1 的触发条件（`-nkvo` + 非零 bank）没有被 host 块误触发。
- 验收：`RESULTS.md` 第 0 节新增"256K 满填"臂（真正填到 262144 cells，而不是 10K），8176 / 6144 / 4096 三档全部 `status = ok`，
  报告 host 块字节数与 decode 额外时间，与第 9 节模型对照。

M5（可选）：量化 K/V 分页；块级前缀共享（seq_cp 的 copy-on-write，与 C 的检查点合并）。

#### 11. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 表项错误或越界 -> GPU fault / device lost | 需重启整机 | host 侧表校验；调试开关；M0c 专测；随机块序测试 |
| 调度器对无数据锚的 split 判断与预期不同 | split 数增加、性能下降 | M1 用 `GGML_SCHED_TRACE` 对照 splits 基线（prefill 291、decode 282） |
| 标量 FA 内循环地址计算拖慢 PP | PP 下降 | M0d 测量；必要时 tile 级预计算表指针 |
| 图复用被打断 | PP/TG 下降 | 锚与表尺寸固定；只有 n_kv 变化才重建图（现状已如此） |
| 块对齐与 FA tile 不匹配 | 数值错误或性能下降 | BS 约束 + M0 确认 `BLOCK_SIZE_K` |
| 与 `-nkvo` / F1 交互 | 复现 abort | M4 专门回归 |
| 守卫分支组合爆炸 | 维护成本 | 只支持 F16 + FA + 非共享 + 非 transposed，其余显式退回 |

#### 12. 需要用户确认的决定（按 AGENTS 要求，动手前先停）

1. ~~是否批准启动 M0~~ **已批准并完成（2026-10-23）**：M0b/0c/0d 全部落地，读数在 `RESULTS.md` 6.1。
2. 块大小默认 2048 还是 8192：**已定 2048**。M0b 给出 `Bc = 32`，2048 是 32 与 256 的公倍数；
   8192 只增加边界浪费，表更小没有收益（128 项 = 1 KiB）。
3. 是否先做 M2a（B-lite）再做 M2b（融合 FA）：**已定先 M2a**（不碰 shader 就能验证整套表与块语义）。
4. M4 是否接受 host 块作为"能跑但慢"的容量档（见第 9 节的 decode 代价）：**接受**，
   但只作为容量档，块级放置的默认策略仍是 device 优先、失败才逐块降级。

## 4. 建议顺序

以第 0 节为最高优先级，本节保留原有的分级依据。

P0 先跑通并解耦：F1（`:1910` 的 producer/lookup 边界要在调度分片阶段解决；调度器的 pre/post 回调都在 split
计算的前后且返回 `void`，不能把 producer 挪进前一个 split，删守卫更不行）+ F7（先归因再解耦）+ F8 的三项归因
（score 预留的设备归属、1712.02 MiB 设备 compute 的归属、按 query 分块的 PP 代价）。
P0 里 F1 是功能前提，F7/F8 是本口径 PP/TG 最大的两块，三者都要在同一轮矩阵里读数。

P1 同配置最优：F8 的实现（预算归属 + 按 query 分块）-> F2 的三档最优 bank -> F3 预取门槛自适应。

P2 跑更大模型：F9 有界 pinned 窗口 -> F5 SSD 冷层 -> F4 Vulkan `MUL_MAT_ID_MASK`（CPU 就地算冷专家的前置）。
F6 是 P2 与 F8 分块改动的验收前提，不是可以并行推的独立任务。

容量路线 F10（分块 KV）：与 P0 的 F1 无依赖，可在 P0 之后独立推进。M0-M2b 只改寻址，数值逐位相同，
因此**不依赖 F6**；只有将来在分页上叠加量化或分块 softmax 时才需要 F6。
F6 已于 2026-10-23 落地（`bash bench/tools/quality-gate.sh`），所以 9.5 的上下文压缩一旦启用，
质量门可以直接用，不再是阻塞项。

混合精度分层仍是唯一能改变 TG 屋顶线的选项（屋顶线见 `RESULTS.md` 第 5 节），但它的前提是 F6 + repack KLD = 0。
