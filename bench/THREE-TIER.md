# THREE-TIER - VRAM + RAM + SSD 协同与自适应

回答"小显存、小内存、大上下文这台机器该怎么分层、下一步做什么"。
数字与已证伪方向分别见 `RESULTS.md` 与 `DECISIONS.md`，本文件不重复判决。

## 0. 最高优先级优化方案（2026-10-23 评审，覆盖下面的顺序）

本节是后续优化的最高优先级，下面第 4 节的顺序以本节为准。方案按"预期收益 x 证据把握"排序。
每一条都是先测后改，都必须在 256K + 10K prompt 口径跑矩阵，并按 `README.md` 第 3 节报全列。
下列数字除非标了实测，都是待验证的假设，不是结论。

### 0.0 架构评审（2026-10-27）：三个结构风险与三步收敛（下面的顺序以此为准）

结论登记在 `DECISIONS.md` 判决 27。评审判定方向不需要推翻：A/B2/F10-M0 的证据链、验收手段与四份文档
的闭环纪律都成立，真正的问题是三处"主权分裂"，都属于"下一步会不会散"，不是"现在快不快"：

1. **放置主权分裂**：KV、score 预留、bank、compute 预留各有一处"放置的所有权"，`common/fit.cpp`
   靠分段扣减逼近，score 预留按设备扣的记错账（判决 18 之前的状态）就是它的已发生后果。
   解法即 D（本评审升 P0）：一个式子 + 一张 Placement 表。
2. **F12 句柄状态机只有文字守卫**：代际未变 / 无 in-flight / 无未消费句柄三道检查若分三处实现
   （解引用、淘汰、回收各写一份），就是下一处 `:1910`（正确性绑在实现形状上）。
   要求收敛成一个原语，判据落在 F12 第 8 节与 `DECISIONS.md` 第 5 节。
3. **同步归属必须单一**：分页 KV 的"表更新在上一张图完成后"是调度层关切；store 若再自建等待就是双主。
   归属不重复立项：等待只有"序号等待"一处，由未结项 19（drain）与 `DECISIONS.md` 第 24 条（F13 S2）负责，
   store 只消费。

三步收敛（依赖自下而上；每步都过 `README.md` 第 3 节全列与既有门禁）：

| 步 | 内容 | 落点 | 验收形态 |
|---|---|---|---|
| 一 | 统一预算式（D，升 P0）：KV / score / compute / dense / RS / bank / staging 一个式子，产出 `Placement[资源][层] -> {vram, ram, ssd}` 表与总账 | `common/fit.cpp`（本期不进 store） | r=2 复现 0.4/0.5 三档读数、`reallocs` 不增；4096 档的 500 MiB 缺口（`DECISIONS.md` 第 6 节第 15 条）在表分解里逐项可读 |
| 二 | `llama-moe-cache` 拆机械侧：保留策略侧（SLRU、整层驻留提示、use_step 观测）；槽分配、pin / 代际、预取环执行、统计进 store 三单元（`F13` 第 2 节点名的现行反例） | `llama-store` M1/M2（F12 第 4、11 节） | M1 用 ±1% 判等 + 越权差 <= 5%；M2 按 15% 门槛 |
| 三 | KV 页接 store：页分配、host-visible 放置、延迟释放全走 store 入口；"host 块预取"与 MoE 共享同一池，不再是复用 | F10 M4 的落地路径 | 同口径读数复现 + `test-kv-page` 全绿；host 块读数借 F10 M4 的 256K 满填臂进矩阵 |

顺序依据：第一步无风险且是后两步的总账前提（先记账再搬运）；第二步是收益主体（消灭平行子系统）；
第三步依赖第二步（KV 进 store 时不能顺带改 store，否则 `tests/test-kv-page` 的等价验收层被打破）。

另一条口径决定（2026-10-27）：默认值变更按证据强度判，A/B2 的位级等价是标准。F10 在等价性回归集冻结
（第 13 条的评审补充）之前不改默认，session 级台阶未定性时尤其不动（`DECISIONS.md` 第 20 条）。

#### 0.0.1 三步收敛执行清单（2026-10-27 生成，勾选式；本清单一行 = 一个改动 + 一次矩阵 + 四份文档更新）

矩阵口径一律 256K + 10K prompt（`README.md` 第 3 节全列）；清单只挂本节，不另立文件。

**第一步 D（P0）：统一预算式，落 `common/fit.cpp`，不动搬运代码**

- [x] 1a. 定结构：`common/fit.h` 新增 Placement 解算结构（输入：dense、KV 按 `-kvg` 工作集口径、score 按
      `-fac` 块口径、设备 compute 按 `RESULTS.md` 0.2 表 A 的 n_ctx 回归、host compute `96.18 + 0.2 x KV`、
      RS、staging、margins；输出：`Placement[资源][层] -> {vram, ram, ssd}` 表 + 总账，以及 KV/RS/ub/bank 四项）。
      现有三处放置改为**只报数**：`src/llama-kv-cache.cpp:233`（KV）、`src/llama-context.cpp:639`（bank）、`:642`（compute）。
      **完成（2026-10-27）**：`common_moe_ledger` 落地并挂在 `common_moe_plan::ledger`；v1 是分层汇总（vram/ram/ssd 三列），
      两处界定记进 `RESULTS.md` 0.10：RS 未拆分（留 F7，条目已留位）、逐层表随 store 2a/3a。
- [x] 1b. `common_moe_plan_get()` 内部改读 Placement 表，删分段扣减；`--fit-plan 0` 保留旧规则作对照臂。
- [x] 1c. 矩阵：三档 x r=2。8176/6144 两档与 `RESULTS.md` 0.4/0.5 读数逐列差 <= r=2 噪声、`reallocs` 不增；
      4096 档日志打印 500 MiB 缺口的逐项分解（margins 地板 256 + 未归因 compute 464）。`ctest` 与当前树一致（34）。
      **完成（2026-10-27，`RESULTS.md` 0.10）**：验收协议升级为同 session 新旧二进制 A/B（`git stash` 回退重编），
      三档规划器决策**逐字相同**（bank 2989.1/1908.4/0）、两臂运行行为逐字相同、配对 PP/TG 三档两档 0.3% 内持平
      （8176 判 session 漂移，读数 1）、12/12 `status = ok`、4096 档缺口逐项分解已在日志；`ctest` 33/33（当前树为 33）。
- [x] 1d. 文档：`RESULTS.md` 新增 1c 臂；开关表核 `--fit-plan` 行；本清单勾选并写日期。
      注：本步不动数值路径，不触发质量门；但 bank 变化会改配布，`status = ok` 是硬验收。
      **完成（2026-10-27）**：0.10 节 + 日期已写；`--fit-plan` 行无语义变化（账本只是产出方式，判据逐项保留，
      0.10 A 为证），无需改开关表。

**第二步：`llama-moe-cache` 拆机械侧（= F12 M1；M2 收益步的前置）**

> 依赖：2a 依赖第一步（总账存在，越权检查才有分母）。M2（2g-2h）另需 `DECISIONS.md` 第 24 条
> `ggml-sched-plan` 先行（判决 26）：sched-plan 管"能不能重叠"，本清单管"给多少、谁来搬"。

- [x] 2a. 骨架：`src/llama-store.{h,cpp}` 三单元（slab / io / policy）+ 薄壳五入口（F13 第 2 节）；
      专家按"层 + 专家子对象"两级注册（层组 41、子句柄约 3 万，O(1) 数组索引，`store_adopt`）；
      挂 `--store-*` 三预算（表见 F12 第 7 节）。slab 纯数据结构、无 GPU 可单测。
      **完成（2026-10-27，`RESULTS.md` 0.11）**：`llama-store.{h,cpp}` + 三单元分文件编译（slab/io/policy），
      常开挂在 `common_init_result::impl`（析构先于 context）；`--store-vram/ram/ssd-mib --store-threads`
      四个 CLI 已进树（vram 参数 -> 旧 F0 hold-back 场直接生效：4096 参数臂 `VRAM limited to 4096.0`、
      `held 3282.8` 与 4096 档臂逐字一致）；`--moe-cache-mib` 0/非 0 两态（0 = 旧行为复现、非 0 照显式值）。
      slab 的"三道守卫一个原语"由 `tests/test-store-slab.cpp` 钉住；adopt 槽不足时分段 grow（不 abort）。
      偏差记录：acquire/prefetch/wait/release 五入口在 2c 随传输执行子落地（骨架期调用面还没人用），
      专家字节与 declared dense 同柜（`co-located` 项，2c repack 后自然并入 accounted sum）。
- [ ] 2b. `store_declare` 接线：dense / compute / staging 报数进总预算；`store_split` 打进启动日志；
      越权对账按 `DECISIONS.md` 第 5 节：差 > 5% release 计 `store_undeclared_bytes`，debug abort 属开发期不变量。
      **部分完成（2026-10-27，随 2a 顺手落地）**：dense/compute 按机/主申报进总预算、`store_split` 进启动日志
      （`RESULTS.md` 0.11，与 plan ledger 对得上）；staging 申报与 5% 越权对账（需 `GGML_VK_MEMORY_LOGGER`
      的对账脚本）随 2c 收口，勾选推迟到那时。
- [ ] 2c. 搬运换执行者（**策略不换**）：SLRU / 整层驻留提示 / 提前一层的 2 槽环照搬，执行归 store-io，
      走 `ggml_backend_vk_prefetch_submit_seq` / `_wait_seq`；池大小 `min(2, max(1, hw/4))`、
      `--store-threads` 可覆盖；池线程禁止阻塞等待；池不可用异步退化同步并计数，不 abort。
      **第一刀完成（2026-10-27，`RESULTS.md` 0.12）**：`--store-exec 0/1/2` 进树（默认 0），
      store-io 绑定面吃下了五个 proc 地址（含 seq），MoE cache 的环路径三条调用路由化，
      A/B/A 验证"路由零行为差"（决策/hit/内存/行为行逐字同、PP 差 < 0.2%）+ 序号管线已接进主路径。
      **未完**：就是 use_step 预取的收益本体和 M2 的 `store_prefetch_wasted`/TG>=15% 验收（2g）；
      promotion/回填路径入 store 计数、staging 申报、5% 对账也在收口清单里。
- [ ] 2d. 删旧执行体：`hybrid_tp`（`src/llama-moe-cache.cpp:471-482`）并入共享池后删除；
      M1 结束同步删除影子模式（F12 第 11 节：留着就是事实上的总开关）。
- [ ] 2e. M1 验收（±1% 不回退门，`RESULTS.md` 新臂）：8176 档 PP 506.7 / TG 17.2 同口径复核；
      `hit %` 按层、按专家两口径各出一张新旧对照表；越权差 <= 5%；temp 0 输出逐字一致。
- [ ] 2f. 文档：开关表 `--store-*` 置"已落地（默认 = fit/auto）"；未结项 23 记进展；勾选。

**M2（收益步）：use_step 驱动预取（前置 = `DECISIONS.md` 第 24 条）**

- [ ] 2g. policy 按 `(对象, use_step)` 排 deadline（store 不预测路由，判决 24）；
      `store_prefetch_wasted` 是唯一自检指标：比 M1 高即 use_step 给错，先修调用方，不调参掩盖。
- [ ] 2h. M2 验收（收益门）：`store_prefetch_wasted` 比 M1 降 >= 50%；**TG 提升 >= 15% 才算收益**
      （未结项 20 规则）；decode 关键路径 0 次整设备 fence（`GGML_SCHED_TRACE` 证明）。

**第三步：KV 页接 store（= F10 M4 的落地路径；依赖第二步）**

- [ ] 3a. `ensure_blocks` / 延迟释放列表 / host-visible 逐块降级全改走 store 句柄；
      逐块降级的判据用现成的 `alloc_buffer` 返回 nullptr（F10 第 5.6 节）；页固定 128 cells，不加新开关
      （页级策略已由 `--kv-page*` 表达）；块表与 `store_acquire` / `store_wait` 同一套。
- [ ] 3b. 验收（等价 + 容量两门）：`tests/test-kv-page` 全绿（含 F10 M1 评审补充冻结的等价性回归集清单）；10K 输出逐字相同；
      `FLASH_ATTN_EXT -b Vulkan1` 5332/5332、`SET_ROWS` 319/319；256K 满填臂（F10 第 10 节 M4）三档
      `status = ok`，host 块字节数与 decode 额外时间进矩阵并与第 9 节成本模型对照；host 块预取与 MoE 同池。
- [ ] 3c. 文档：F10 第 10 节 M4 置完成态；`RESULTS.md` 补满填臂；勾选。

状态（2026-10-27）：**第一步 + 2a 已完成（1a-1d 与 2a，`RESULTS.md` 0.10/0.11）**，2b 部分完成（staging 与
5% 对账随 2c）、2g/3a 未开工。阻塞链：1a/1b -> 2a（已满足）-> 2c（下一个，io 池 + 传输执行子 +
`prefetch_submit_seq/_wait_seq` 接入）-> 2g（另需第 24 条）-> 3a。

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

### D（P0，2026-10-27 评审升级）：预算解算器一体化（F2 + F7 + F8 合并成一个式子）

> 升级理由见 0.0：放置主权分裂是三个结构风险之首，score 预留按设备扣的记错账在判决 18 之前就是它的后果。
> 升级同时是把风险做小：这一步不动任何搬运代码，本期只落在 `common/fit.cpp` 内部。

- 把 dense、KV（按 A 的增长口径）、score 缓冲（按 B 的块口径）、compute（设备按 `RESULTS.md` 0.2 表 A 的
  n_ctx 回归，host 按 `96.18 + 0.2 x KV`）、bank、RS 状态放进 `common/fit.cpp` 同一个式子，
  输出 KV 放哪、RS 放哪、ub、bank 四项，外加一张 `Placement[资源][层] -> {vram, ram, ssd}` 表与总账，
  而不是现在的分段扣减。
- 复用 `common_moe_plan_get()`，不新增规划器。`llama-store` 第 2 步（F12 第 4 节）接管数值决策时
  把该式子**整体搬进** store、fit 收成申报薄壳，不写第二套预算算术。
- 验收：在三档 x 每档 3 个 bank 的消融里，规划器推荐值与实测最优 bank 的 argmax 一致。
  追加（2026-10-27 评审）：0.4/0.5 的三档读数在 r=2 噪声内复现、`reallocs` 不增；
  4096 档的 500 MiB 缺口（`DECISIONS.md` 第 6 节第 15 条：margins 地板 256 + 未归因 compute 464）
  必须在表分解里逐项可读，不再靠日志反解。

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
   下一步按收益排（**2026-10-27 评审重排**，依据见 0.0）：
   ~~F3~~ 已关闭（`DECISIONS.md` 第 6 节第 14 条）；**D 升 P0 先行**（记账先于搬运，第一步不动搬运代码，
   验收形态是复现读数，不是新收益）-> E（`ub 2048/3072/4096` 联合调参，B2 已解锁）-> C（前缀检查点）。
2. A + B 同一轮实现。只做 A 看不到 compute 预留的收益。（已按此执行。）
3. C 紧随 A（依赖 KV 变小）；D 的第一步与 C 无依赖，可在此之前插入。
4. 收敛的第二、三步不再单列编号：拆 moe-cache 机械侧 = `llama-store` M1/M2（F12 第 11 节），
   KV 页接 store = F10 M4 的落地路径。
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

`GGML_VK_VRAM_LIMIT_MIB` 在设备初始化时 hold back 差额（**旧 env，已并进 `--store-vram-mib`**，
语义不变）。已验证：日志出现 `VRAM limited to ...`；
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

### F3 预取门槛按显存自适应（**6144 档已关闭，2026-10-27**）

`:646` 要求每组至少 `2 x n_expert` slot，小卡直接失去预取（也就失去 +25.5% 的 pp2048）。
允许 1 环 / 分块层预取，`pf_min_tokens` 与 `pf_max_full` 由 bank 与 ubatch 推导。
验收：4096 档 `LLAMA_MOE_PREFETCH_TIME=1` 显示预取生效，等待不超过该档 PP 的 10%。

**2026-10-27 结果**：6144 档不需要动 cache 的门槛，只需要规划器把 margins 按档缩放（`--fit-plan 1`）——
bank 1644 -> 1908 MiB，跨过每组 512 slots 对应的 1860 MiB，`RESULTS.md` 0.5 A 实测 PP +29.3% / TG +28.3%。
规划器的预取判据同时改成 per-group（`2 * expert_total / n_layers_of_group`），与 cache 侧口径统一。
**4096 档仍未打开预取**：那一档的 bank 连 145 MiB 的小 bank 地板都过不去（`DECISIONS.md` 判决 21）。

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
(b) decode 关键路径 pread 次数 = 0（`GGML_SCHED_TRACE` 证明）、
(c) pp2048 双对照（2026-10-27 评审改）：(c1) >= 同配置 resident 的 70% 只证明"窗口有用"，不证明"窗口是好方案"，
因为 RAM 压力测试本身会改变 CPU 侧（pinned pool 会挤掉 page cache）；结论以 (c2) 同 RAM 压力下
"无窗口、纯 mmap 退化"的第三臂内比较为准，(c1) 留作下界； (d) `ctest` 32/32。
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

- **页 `kv_page`**：`BS` 个 cell 的全部注意力层 K/V，是一个 ggml backend buffer（默认 device-local，可选 host-visible）。
  块内布局 `[layer][K|V][cell][row]`，同一层在块内是连续的 `BS x row` 矩阵，FA tile 的局部性不变。
  每 cell 20 KiB（10 层 x K/V x f16，与 `RESULTS.md` 0.3 的 160 MiB / 8192 cells 一致）。
- **页大小固定为 128 cells = 2.5 MiB**（2026-10-27，`--kv-block` 移除）：
  `256K / 128 = 2048 页`，正好整除，没有半页。128 是 FA tile（64 / 128）的整数倍，
  **分页 FA / SET_ROWS 的现有假设不用改**，这是选它而不是选"专家大小 1.945 MiB"的唯一理由。
- **不再和专家同尺寸**：单个 Q4_K_M 专家是 `2,039,808 B = 1.945 MiB`（41 层 x 256 专家，每层三元组 498 MiB），
  一页 KV 是 2.5 MiB，两者差 28%。store 的 slab 分配器因此有两个 size class（专家 1.945 MiB / KV 页 2.5 MiB），
  句柄与搬运接口仍是一套。想强行同尺寸就得把页改成 99 cells，那要改 FA tile 假设 - **已否决**（2026-10-27）。
- **块表 `kv_block_table`**：每 stream 一张，`uint64 addr[n_ctx / BS]`（256K 下 2048 项，16 KiB），
  Vulkan 时放在设备 SSBO，CPU 时放在主机数组。
  `addr[b]` 为页 b 的基址（Vulkan 为 BDA，CPU 为指针），层内偏移为 `layer_idx * layer_stride + kv_off`，由 shader 计算。
- **锚张量 `anchor`**：一个不分配数据的 f16 ggml 张量，`ne = [n_embd_k_gqa, n_ctx, n_stream]`，挂在设备 buffer 上，
  只用来让调度器把依赖它的 FA/SET_ROWS 放到正确 backend。`get_k/get_v` 继续返回对锚的 `ggml_view_4d`，
  `ne/nb` 与现在一致，`src/llama-kv-cache.cpp:1510-1550` 的视图代码基本不动。
  待 M0 确认：调度器对"无真实数据的 view"按 `view_src` 的 buffer 分配 backend（读 `ggml-backend.cpp` 的 split 逻辑）。
- **分配器**：空闲链表 + 每块状态 `{addr, placement(dev|host), refcount(M5 预留)}`。
  `ensure_blocks(n_extra)` 取代 `ensure_capacity`：`need = used_max + n_extra`，`while cap < need: alloc_block()`。

#### 3.1 块大小（BS）选型：容量粒度参数，不是性能参数（历史，BS 已固定为 128）

BS 只影响以下四项，其中没有一项直接作用在 PP/TG 的算子上：

| 影响面 | BS=2048（40 MiB/块） | BS=8192（160 MiB/块） | 对 PP/TG 的量级 |
|---|---|---|---|
| 增长粒度与最坏浪费 | <= 40 MiB | <= 160 MiB | 噪声内。实测 160 对 320 MiB 的 KV 无差（都 477），320 MiB 对 5120 MiB 才有差（477 对 239，`RESULTS.md` 0.2 表 B） |
| host/device 边界粒度 | 剩余 40 MiB 即可多留一块 | 剩余 160 MiB 才多留一块 | 唯一真实影响：边界上最多差 160 MiB 的 host 流量（约 11 ms/token，见 9 节） |
| 表项数 | 128 项 / 1 KiB（256K） | 32 项 / 256 B | 无 |
| FA 表查询次数 | 按 K/V tile 计数，与 BS 无关（BS 只需 >= tile 且为其倍数） | 同 | 无 |
| B-lite gather 的 dispatch 数与 scratch | 更少 / 更小 | 更多 / 更大 | 只影响验证路径（M2a）的 PP |

结论（历史）：当时默认 2048，理由是边界粒度细、浪费小，不是因为更快。
**2026-10-27 推翻**：`--kv-block` 移除，BS 固定 128（2.5 MiB），选它的理由变成"FA tile 整数倍 + 2048 页整除 256K"，
不是容量粒度；上表的边界粒度与表项数结论按 128 重算时都更优（表项 2048 项），只有 gather 的 dispatch 数会涨，
由 M3 实测。

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

- `GGML_VK_VRAM_LIMIT_MIB`（F0，**已并进 `--store-vram-mib`**）经 hold-back 让新块分配失败。失败时**只把这一块放到 host-visible 内存**，
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

- `--kv-page` / `--no-kv-page`（env `LLAMA_ARG_KV_PAGE`）、`--kv-page-auto auto|on|off`
  （env `LLAMA_ARG_KV_PAGE_AUTO`）。`--kv-block N` **已移除**（2026-10-27）：页大小固定为单个 Q4_K_M 专家，
  一页 128 cells（见第 3 节），不再对外暴露块大小。下面 3.1 的 BS 选型只作为历史理由保留。
- 默认值：**off**。`--kv-page-auto auto` 让 `common/fit.cpp` 的预算判决（`DECISIONS.md` 第 4 节开关表、
  第 6 节第 16 条；实测 4096 档判分页、6144/8176 判连续）。M2b + L1 的验收本来允许直接改 auto，
  但 `DECISIONS.md` 第 6 节第 17 条的 10K 输出差异未定性，所以默认值先不动。
  （2026-10-27 评审补充：默认值按证据强度判，与 A/B2 的位级等价同一标准。第 17 条的差异已由 `grow()`
  修复解决（该节另有记录）；现在仍不改的理由是收益侧没有长上下文真实读数：目标口径 10K 的 KV 只有
  320 MiB、三档都放得下（F11 第 1 节），4096 档换到的只是"页级降级的安全性"且 PP 低 3-5%；
  session 台阶定性（`DECISIONS.md` 第 20 条）之前不动默认。）
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
- **2026-10-27 评审补充：冻结等价性回归集**。分页改动的回归风险不在新代码，而在"看似无关的 opt-in 路径"
  绕过守卫（SET_ROWS 融合门发生过一次，`DECISIONS.md` 第 5 节）。把第 7 节的自动退回条件与第 5.8 节的
  两条 opt-in 冲突（`use_dequant_kv` 分页必须显式关、`use_sparse` 退回连续）冻结成 `tests/test-kv-page`
  的显式用例清单，每个退回条件一条；以后动 FA / SET_ROWS / 图复用必跑该清单，不靠本文档记忆。

M2a B-lite（验证，不改 FA shader）

- 每层每块 `ggml_cpy` 把块内连续行拷进 scratch，FA 走原连续路径。只用于验证 M1 的表与块语义，性能不作目标。
- 验收：Vulkan 上 `--kv-page on`（B-lite）与 `off` 的生成文本在 temp 0 下逐字相同，10K 与 40K 两个 prompt。

M2b 融合 FA（Vulkan）

- **进行中（2026-10-24）**：FA 与 SET_ROWS 的分页内核都已进树，算子级在 CPU 与 Vulkan 上都逐位正确
  （`tests/test-kv-page.cpp`，两种后端；表以 I64 张量挂在 `src[5]/src[6]` 与 `src[3]`）。
  非分页门禁 `FLASH_ATTN_EXT -b Vulkan1` 5332/5332 不变。
- **模型级 Vulkan 路径**：曾经的 PPL 4.1412 对 6.0684 已定位为 ROPE/RMS_NORM 融合旁路分页 SET_ROWS
  （`RESULTS.md` 6.4 第 1 条），关掉融合后 PPL 与连续布局相同、质量门 KLD 0。设备 K/V 的守卫已改成
  "后端必须能给出 device address"，`--kv-page` 在 Vulkan 上直接可用。
- **gather（F11 L1）已实现**；`--kv-page auto`（按 fit 预算决定默认开不开分页）仍未实现，
  它是"容量档也要 PP"之外的另一件事，见 `DECISIONS.md` 第 6 节。

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

以第 0 节为最高优先级，本节保留原有的分级依据。2026-10-27 评审把 D 升 P0、并给出三步收敛（见第 0 节 0.0），
本节的 P0/P1/P2 归属段以 0.0 为准。

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

## F11 分页 KV 的 prefill 路径：块 -> 稠密工作区 -> 现有 chunked matmul（**L0 + L1 已落地**，P0）

> **2026-10-24 状态**：用户确认取 (b)，gather 已实现并验收（`RESULTS.md` 6.4）。当天早些时候曾记过
> "先按容量档推进（选项 c）、gather 暂不实现"，那条已被本轮落地取代；(c) 只剩**默认模式选择**这一个含义：
> `--kv-page auto` 由 `common/fit.cpp` 的预算判决，初始 KV 装得进设备就保持连续布局。它还没有实现。


### 0. 结论先行

- 现状（`RESULTS.md` 6.3）：`fa_span` 被 `!paged` 关掉（`src/llama-graph.cpp:3094`），
  于是 `--kv-page` 下 prefill 的所有跨度都留在 `FLASH_ATTN_EXT` 上，B2 的 `-fac 256` 换来的 PP 680 用不上，
  同一档 `-fac 0` 只有 458（`DECISIONS.md` 判决 18）。
- 还有一条没人算过的代价：FA 的 prefill 按 query 行组重复读 K/V。`n_tokens = 2048` 时 `Br = 8`、`gqa_ratio = 1`
  （`ggml-vulkan.cpp:8332-8345` 的两段 tuning，gqa 合并只在 `N <= 8` 生效），
  所以 `workgroups_x = ceil(2048/8) = 256`，每个 workgroup 都扫一遍整段 KV（`flash_attn.comp:213-449`）。
  **只要有块落在 host，prefill 的 PCIe 就是 KV 字节的 256 倍**（第 5 节）。
- 方案：分页模式下的 prefill 换成"**按块拷进一块稠密工作区，再喂现有的 chunked matmul 路径**"。
  这就是 `THREE-TIER.md` F10 M2a 已经写下的机制（`ggml_cpy` 逐块），也是 vLLM 长前缀收集的同一思路。
  不新增 op、不新增 shader、不改数值（拷贝是逐字节的），只把 `k`/`v` 的来源从锚换成工作区。
- 预期：PP 458 -> 约 680（拿回 B2 的收益），代价是 10.5K 口径约 7 ms/整段 prefill（0.05%）+ 32 MiB 常驻；
  host 块场景的 PCIe 从"KV 字节 x 256"降到"KV 字节 x 1"。
- 不做：上下文并行（单 GPU）、选择性 KV 加载（dense causal，`use_sparse = 0`），理由见第 4 节。

### 1. 为什么分页 FA 不能当 prefill 的答案

- 分页寻址本身不是问题：M0d 实测表寻址 <= 1%（`RESULTS.md` 6.1），真实 FA 每 `Bc = 32` 个 cell 才查一次表。
- 问题在 tile 形状：AMD 上 `hsk = 256 > 128` 会把 `block_rows` 减半（`ggml-vulkan.cpp:1230-1244`），
  实测 `Br = 8 / Bc = 32 / row_split = 4`；判决 9 的 1.07 TF 对 4.08 TF 就是这个形状造成的。
- 每个行组都要把 `j = 0 .. KV/Bc` 走完，K/V 的读取次数是 `n_q / Br`，与是否分页无关。
  全驻显存时这是 L2/DRAM 带宽问题，有 host 块时就是 PCIe 问题。
- 目标口径的形状（`RESULTS.md` 6.1 的 fa-trace）：prefill 是 `N = 2048 / 198`、`Br = 8`，
  行组数 `ceil(N/Br) = 256`（`N = 8` 的 gqa 合并不适用；那两行是 decode）。最后几个 ubatch 的
  `n_kv` 接近 10496，即 `KV/Bc = 328` 个 K/V tile x 256 个行组，每层每 ubatch 约 5.4 GiB 的 K/V 读取请求。

### 2. 设计

#### L0 前置：SET_ROWS 分页变体（**已落地并验收，2026-10-24，`RESULTS.md` 6.4**）

- 落地时挖出真正的根因：分页 SET_ROWS **被执行了，但结果被 ROPE/RMS_NORM 的融合旁路**（`ROPE_VIEW_SET_ROWS`、
  `RMS_NORM_VIEW_SET_ROWS`）。融合直接把 rope 输出写进 SET_ROWS 的 dst，即 256 B 的锚，块永远是 0；
  `-c 512` 因此输出不同，`-c 8192` 直接 device lost。两个融合判据已加 paged 检查。
- 验收结果：`--kv-page` 与 `off` 的 PPL 完全相同（2.1676 对 2.1676）、4535 token 逐字相同、
  `FLASH_ATTN_EXT` 5332/5332、`SET_ROWS` 319/319、`ctest` 33/33。
- 设备 K/V 的守卫已从"未验证"改成"后端必须能给出 device address"，`--kv-page` 在 Vulkan 上直接可用。

- `copy_to_quant.comp` 的 `PAGED` 分支（:18-26、:336-342）把 `dst_idx` 换成
  `data_qtab[cell / BS]` + `(cell % BS)` 行内偏移；`pipeline_set_rows_paged` 在
  `ggml-vulkan.cpp:3393-3397` 建立，分发在 `:10946-10963`（`PageBS` 走 `param3`）。
- `cell -> 块` 是恒等映射：`add_block` 只追加（`src/llama-kv-cache.cpp:1426-1498`），`ensure_blocks` 只增长，
  所以块 `b` 覆盖 cell `[b*BS, (b+1)*BS)`。**gather 依赖这条不变量**，将来做块回收/压缩（F10 9.5）时必须同步改。
- 验收：`-c 512 --kv-page` 输出与 `off` 逐字相同；`tests/test-kv-page.cpp` 补一条 Vulkan 后端用例；
  `FLASH_ATTN_EXT -b Vulkan1` 5332/5332 不变；`ctest` 34/34。

#### L1 主线：块 -> 稠密工作区 -> chunked matmul（**L1a-L1d 已落地，2026-10-24**）

- 落地形态与设计一致：`--kv-gather-mib`（默认 256，0 = 关）、工作区按工作集 2 倍增长、
  每层每块一条 `ggml_cpy`、`llama_memory_context_i::get_kv_page_dense()` 在 `build_attn_mha` 里接上。
  正确性见 `RESULTS.md` 6.4：gather 与连续布局**逐字节相同**（`test-kv-page`，CPU 与 Vulkan 各一遍），
  4535 token 的生成文本与 `off` 逐字相同。
- **L1d 挖出一个工作区覆盖缺口并已修**：`ensure_capacity` 原来按 `used_max() + n_ubatch` 决定工作区大小，
  而 `used_max()` 在个别路径上比图真正读的范围小一整批，于是 10K prompt 的最后一个 ubatch
  （`n_kv = 10240`）拿不到工作区、静默退回 paged FA。现在 `build_paged_dense()` 记下每个图请求的最大范围
  （`page_dense_need`），`ensure_dense_workspace` 取它与 `used_max() + n_extra` 的较大值。
  判据日志：`GGML_KV_PAGE_TRACE=1` 下 `build_paged_dense: layer N: X cells, workspace Y cells, usable 0/1`。
- 目标工况读数（r=2，`RESULTS.md` 6.5）：三档 x 三臂，gather 把分页 prefill 从 paged FA 拉高 17-32%，
  与连续布局持平（8176 +4.4%、6144 -1.7%、4096 -5.0%）；TG 三者相同。三档 `status = ok`。
- **L1e 已闭环**（`RESULTS.md` 6.6）：10K prompt 下分页与连续布局**逐字相同**，根因是扩容用了
  `v_cells.resize()`（会 `reset()` 全部 cell）而不是 `grow()`；修好后默认 `--kv-block 2048` 与 256 都逐字相同。
  修复前的 paged 读数偏高约 5%，已作废。
- **host 块臂在目标口径造不出来**：10K cells 的 KV 只有 320 MiB，三档的设备都能放下；
  显式 `-B 3072` 只会让 bank 和 compute 互相挤（4096 档 PP 379 -> 169.9），产生不了 host 块。
  host 块只在工作集本身超过设备（`n_kv` 约 10 万以上）时出现，那是 L2(b) 的区间，见 3.1 的区间表。
- ~~仍未做：`--kv-block` > 2048 在 gather 下与连续布局不同~~ **作废（2026-10-27）**：`--kv-block` 已移除，
  页固定为 128 cells（tile 对齐，FA 假设不用改）；默认值仍是 `--kv-page` off（`--kv-page-auto` 可用但 4096 档
  比连续布局低 3-5%，换的是页级降级的安全性）。

(a) 工作区（KV cache 持有）

- `page_scratch_k` / `page_scratch_v`：`[n_embd_k_gqa, cells]` / `[n_embd_v_gqa, cells]` 的 f16 张量。
  行步长必须与块内一致：块的 2D 张量是 `[n_embd_k_gqa, BS]`、`nb[1] = row_size(n_embd_k_gqa)`
  （`src/llama-kv-cache.cpp:1447-1450`），与连续布局同构，所以一次 `ggml_cpy` 就是一次 memcpy。
- 初始 32 MiB（约 16K cells），按 2 倍长到 `--kv-gather-mib`（默认 256 MiB，0 = 关），
  增长点与 `ensure_blocks` 同批；分配失败 -> 本次不 gather，退回分页 FA 并计数告警（红线：不 abort）。
- cap 与 matmul 判据是同一个范围：`-fac 256` 要求 `chunk_q >= 64`，即 `n_kv <= 65536`，
  而 64K cells 的工作区只要 `65536 x 2 KiB = 128 MiB`。256 MiB 的 cap 因此覆盖 matmul 路径的全部有效区间。

(b) 图（`src/llama-graph.cpp:3055` `build_attn_mha`）

- 判据拆两步：先算与分页无关的 `span_ok`（现在的 `n_tok_q >= 512 && n_tok_kv > n_tok_q + n_tok_q/2` 等），
  再对分页缓存调 `mctx->get_kv_page_dense(ctx0, il, n_kv, &kd, &vd)`；成功则 `fa_span = span_ok`，
  失败保持 `fa_span = false`（原行为）。**gather 只在 matmul 路径真会用到时才做**，decode 与短跨距一次都不拷。
- `kd` / `vd` 用与 `get_k` / `get_v`（`src/llama-kv-cache.cpp:1810-1860`）完全相同的 `ggml_view_4d` 数学指向工作区，
  后面的 chunk 切分、mask 切分、`ggml_concat` 一行不改；`v_trans` 也因此与连续布局一致（false）。
- 拷贝节点：每块一条 `ggml_cpy(block.k[i], view_2d(sk, ne0, BS, sk->nb[1], b*BS*sk->nb[1]))`，V 同理。
  10.5K + BS 2048 = 6 块 -> 12 条/层/ubatch；尾块按 `n_kv - b*BS` 截断。
- 不需要新的同步逻辑：Vulkan 侧按 buffer range 自动插 barrier（`ggml-vulkan.cpp:12541-12607` 的
  `unsynced_nodes_written/read`），SET_ROWS 写块与 gather 读块命中同一个范围，这就是现在
  "SET_ROWS 写块 -> FA 经表读块"能正确工作的同一条机制。
- 每层复用同一块工作区（层的节点在图上串行）。若实测出重叠，退化成 2 块轮转（+1 份内存）。

(c) 内存/预算

- `common/fit.cpp:954-960` 现在扣 `llama_fa_long_kv_reserve_bytes()`（等于 `-fac` 的 cap）。
  新增 `llama_kv_gather_reserve_bytes()`，扣**初始值 32 MiB**（不是 cap），并在计划日志里打印一行。
- 只扣初始值的理由：A 已经把"按工作集分配"变成默认（`-kvg -1`），工作区同理。
  扣 cap 会在 6144 档再挤掉 256 MiB，可能把 bank 压到 512 slots/组以下（判决 19 的预取门槛）。

#### L2 只在 L1 实测不够时做（两条独立终态）

(a) **分页 MUL_MAT**：f16 K/V 专用变体，`Paged` / `PageBS` spec constant，表经 `op_params`
    （与 FA 同一套约定，不占 `src` 槽），每 `BN` 行查一次表（要求 `BS % BN == 0`）。
    收益 = 省掉工作区与那 25% 的 K 额外读写；风险 = 动最热 kernel，必须 `MUL_MAT -b Vulkan1` 2275/2275 + 质量门。
(b) **KV 分块 + LSE 合并**：只服务 `n_kv > 64K`（此时 `-fac 256` 的判据本身不触发，matmul 路径本来就不在），
    改数值，受 F6 门禁。**不在本轮**。

#### L3 host 块（不需要单独实现，L1 一开就有）

见第 5 节。它同时是 F10 M4 的前置：没有 L1，块级 host 放置的容量档在 prefill 上不是慢一点，是差两个数量级。

### 3. 容量与判据算术（10.5K / BS 2048 / -ub 2048 / -fac 256）

- `score_row = n_kv x n_head x 4B = 10496 x 16 x 4 = 672 KiB`；
  `chunk_q = 268435456 / 688128 = 390`；`n_chunks = ceil(2048/390) = 6`。
- 工作区 = `n_kv x 2 KiB = 21 MiB`（K+V 合计，每层；10 层共用一块）。块数 6（12288 cells）。
- 每次 gather = 21 MiB 读 + 21 MiB 写 = 42 MiB/层 -> 420 MiB/ubatch -> 2.6 ms @168 GB/s。
- 整段 prompt 重拷 5 次，总量约 `sum(n_kv) x 20 KiB = 614 MiB` -> 7.4 ms，
  对 PP 680（10.5K / 680 = 15.4 s）是 **0.05%**；dispatch 120 条/ubatch（约 2-4 ms）同量级。
- 对照：现有 matmul 路径的 score 流量是 6 chunk x 262 MiB（写+读）x 10 层 = 31 GiB/ubatch；
  gather 是它的 1/70。
- 重拷的 N^2 上限：读写各 `5 x N^2` 字节，合计 `10 x N^2`；N = 32K 是 10.7 GB（64 ms），
  N = 64K 是 43 GB（256 ms，对 96 s 的 prefill 是 0.26%）。所以**不需要增量 gather**，
  也不需要在跨 ubatch 保留工作区。
- host 块：gather 每层只读一次 host，见第 5 节。

#### 3.1 长 prompt 下的三个区间（回答"会不会劣化"）

判据来自 `-fac 256` 的 `chunk_q >= 64`，不是人为设的阈值：

| 区间 | 走哪条 | 相对今天的 paged 模式 | 相对连续布局同 n_kv |
|---|---|---|---|
| `n_kv <= 3072`（`n_tok_kv <= 1.5 n_tok_q`） | FA，不 gather | 不变 | 不变（FA 不看布局） |
| `3072 < n_kv <= 65536` | gather + chunked matmul | **PP 从 458 回到约 680 的量级**，gather 0.05%-0.26% | 平价（长 n_kv 的 score 税两边都付） |
| `n_kv > 65536` | paged FA，不 gather | **不变**（今天就是这条路） | 不变；但 host 块的 256x PCIe 仍在，要 L2(b) |

- 中间的窗口内**不会劣化**：gather 的读写是 `2 x KV 字节/层/ubatch`，而它替掉的是 FA 的
  `n_q/Br = 256 x KV 字节`（第 5 节）；即使不看 FA，它相对 matmul 路径自身 6-32 个 chunk 的 score 流量也只有 1/70。
- 窗口外**也不劣化**：不 gather 就不动数据，行为与今天的 paged FA 完全一致。代价是那一档拿不到 L1 的收益，
  这一条写进 L2(b) 的触发条件，不当成已解决。窗口外 + host 块的量级（估算，供 L2(b) 立项用）：
  `n_kv = 128K`、20% host 时 FA 每 ubatch 约 131 GiB PCIe -> 9 s，而 compute-bound 约 6.8 s（PP 300）；
  `n_kv = 256K` 同比例约 18 s/ubatch。也就是说**真正 10 万 token 以上的 prompt + 块落 host，L1 只能覆盖前 64K**，
  剩下的必须靠 L2(b) 或调大 `-fac`。目标口径（256K ctx + 10K prompt，`n_kv` 最多 10.5K）不涉及这一段。
- 唯一新增的常驻是工作区（10.5K 21 MiB、64K 128 MiB）。它按 2 倍增长，**增长失败只退回 FA，不会去挤 bank**
  （`ggml_backend_vk_buffer_type_alloc_buffer` 返回 nullptr，`ggml-vulkan.cpp:13315`）。
- 想把这个窗口推长只有两条：调大 `-fac`（score 缓冲同步变大，`RESULTS.md` 0.4 里 512 在 8176 档已经更慢），
  或者 L2(b) 的 KV 分块 + LSE。都不作为默认。

### 4. 与业界四条路径的对照（为什么只取第 2 条）

| 路径 | 本项目判定 | 依据 |
|---|---|---|
| 分页感知融合内核（ONNX Runtime WebGPU / flashinfer） | **已就位**（FA 分页分支），但只能当 decode 与短跨距；prefill 用它就是 458 | 判决 9 的 1.07 TF 对 4.08，加上 `Br = 8` 的重复读 |
| 长前缀收集 + 稠密工作区（vLLM Volta） | **本方案主线**（L1）。阈值不是 "Q >= 4096 且 KV >= 8192"，而是"matmul 判据成立"，即 `3072 < n_kv <= 64K` | 判据本来就在 `fa_span`（`src/llama-graph.cpp:3094-3102`） |
| 上下文并行（vLLM CP / ring attention） | **排除**：单 dGPU（RX 6700S），无第二设备可切；历史双 GPU 尝试是 SSD 污染 + OOM | 判决 3 |
| 选择性 KV 加载（CompactAttention / ChunkAttn） | **排除**：10 个全注意力层是 dense causal，M0b 实测 `sparse = 0`，没有 2D 块稀疏掩码可用；共享前缀在本项目走 server LCP + cell 复用（F10 M5），不属于 prefill kernel | `RESULTS.md` 6.1 |
| （补充）分页 GEMM | L2(a)，L1 实测后若工作区/拷贝成为瓶颈再做 | - |

### 5. host 块：L1 顺带解决的硬问题

10.5K 口径、每层 K+V 共 21 MiB、PCIe 14.3 GB/s：

| host 块占比 | FA prefill 每层每 ubatch | gather 后 | 倍数 |
|---|---:|---:|---:|
| 10%（2.1 MiB/层） | 537 MiB | 2.1 MiB | 256x |
| 20%（4.2 MiB/层） | 1.07 GiB | 4.2 MiB | 256x |
| 100%（21 MiB/层） | 5.4 GiB | 21 MiB | 256x |

即 100% host 时 FA 路径每 ubatch 要 3.8 s（10 层 x 377 ms），gather 后 14 ms。
所以**F10 M4 的 host 块必须和 L1 同批验收**，否则那一档的 prefill 数字没有意义。

#### 5.1 L1 之后分页还剩哪些收益

| 收益 | L1 之后 | 依据 |
|---|---|---|
| 块级 device/host 放置，单块失败不粘整棵 cache | **保留，且这是分页唯一不可替代的收益** | 连续布局的 `host_fallback` 是 cache 级标志（`src/llama-kv-cache.h:283`），一次失败整棵 KV 落 host |
| 扩容零拷贝、峰值 = 1 块 | **保留** | `-kvg` 的 2 倍增长要 D2D 拷已用前缀（10K 臂 `reallocs` 3 次，`RESULTS.md` 0.3） |
| prefill PP | 从"分页 = 放弃 `-fac`"变成与连续布局平价 | 3.1 的窗口 |
| decode TG | 约 -1% 的固定税（每 32 cell 查一次表，M0d 在 DRAM 规模 +0.4%~0.9%） | `RESULTS.md` 6.1 |
| host 块的 decode PCIe | 不变，按字节线性（容量档的固有代价，F10 M4 已接受） | 9.1 的成本模型 |

- 结论：分页现在**没有速度理由**，只有容量理由。所以默认策略应是"全驻显存保持连续布局（`--kv-page off`），
  只有工作集装不下连续布局时才开分页"。这与第 4 节否决 `RESULTS.md` 6.3 的候选 (c) 不矛盾：
  (c) 是拿"只在 host 块 > 0 时启用"当 prefill 的替代方案；L1 是让分页在任何时候都不掉 PP 的前提，
  有了它，"按容量开分页"才是纯收益的选择。

### 6. 里程碑与验收

| 里程碑 | 内容 | 状态 |
|---|---|---|
| L0 | SET_ROWS 分页 + 关掉 ROPE/RMS_NORM 对分页 SET_ROWS 的融合；打开设备 K/V | **完成**（`RESULTS.md` 6.4） |
| L1a | 工作区 + 逐块 cpy + `get_kv_page_dense` | **完成**：单测逐字节相同（CPU + Vulkan） |
| L1b | 图切换 + 判据 + 开关 | **完成**：4535 token 逐字相同 |
| L1c | 预算 + 文档 | **完成**：fit 扣 32 MiB、开关表已登记 |
| L1d | 目标口径矩阵：三臂（off / paged + gather / paged + `--kv-gather-mib 0`）x 三档 x r=2 | **完成**（`RESULTS.md` 6.5）：三档 `ok`，gather 比 paged FA 高 17-32%，与连续布局持平；host 块臂在目标口径造不出来（见 L1 节） |
| L1e | 10K prompt 下分页与连续布局的输出一致性 | **完成**（`RESULTS.md` 6.6）：根因是扩容用了 `resize()`，改 `grow()` 后默认块尺寸逐字相同 |
| L2a | 分页 MUL_MAT（可选） | 未开始：L1d/L1e 的读数没有给出理由（>64K 与 host 块臂不在目标口径） |

L1 验收线：

- 数值：**已达成**。gather 与连续布局逐字节相同（`test-kv-page`，CPU + Vulkan）；
  `--kv-page` 与 `off` 的 PPL 相同（2.1676）、4535 token 输出逐字相同；质量门带 `--kv-page` 通过
  （`mean KLD -0.000000`、`same top p 100.000%`，`RESULTS.md` 6.4 第 5 条）；
  10K prompt + `--kv-block 2048/256` 与连续布局逐字相同（`RESULTS.md` 6.6）。
- 性能：paged + gather 的 PP >= 连续布局 `-fac 256` 的 95%（8176 档目标 646 以上）；
  TG 不退步（decode 不走 gather）；三档 `status = ok`；`reallocs` / `hit %` / 工作区 MiB 进表。
- 4096 档若 bank 仍为 0，只要求不退化（这一档分页本来就是容量档）。
- 新增 trace：`GGML_KV_PAGE_TRACE=1` 每 ubatch 一行（`n_kv`、blocks、`dense|fa`、gather MiB、scratch MiB），
  给矩阵提供口径。

### 7. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 每层共用工作区的调度重叠 | 数值错乱且静默 | 先跑 `-c 512` 与 10K 逐字；靠 buffer-range 同步；必要时 2 块轮转 |
| 工作区把 bank 挤到预取门槛下 | PP 反而降 | 只扣 32 MiB 初始值；6144 档先验 512 slots |
| 逐块 cpy 的 dispatch 数 | 长上下文吃 dispatch | 10.5K 120 条/ubatch（2-4 ms）；>64K 走 L2(b)，或合并成单个 `page_gather` kernel |
| 块在 host 时 cpy 变成 H2D | 慢，但只一次 | 正是目的（替代 256 次） |
| 非均匀 head_dim / 多流 / V 转置 | 用不了工作区 | 已有守卫 + 退回分页 FA，计数 |
| 开关组合爆炸 | 维护成本 | 只加 `--kv-gather-mib` 一个数，0 = 关 |

### 8. 需要确认的决定

1. ~~顺序：L1 先行，L2 视测量~~ **已按此执行**（L0 + L1 已落地）。
2. ~~工作区 cap 默认 256 MiB、初始 32 MiB~~ **已按此落地**：`--kv-gather-mib` 默认 256，fit 扣 32 MiB。
3. host 块与 L1 同批进 F10 M4 验收：**仍待做**，和 L1d 的矩阵一起。

## F12 统一存储子系统 llama-store（本机 VRAM/RAM/SSD 的抽象层，常开，预算可手设）

`llama-store` 是**本机三类硬件的抽象层**，但第一版只接管**两类对象**：专家权重（按专家粒度）与 KV 页（按 page）。
核心目标只有一个 - **高性能的专家预取**；其余模块（dense 权重、compute buffer、staging、加载路径）**一律不动**，
只要求把占用申报给 store，否则总预算不成立。对外只做四件事 - 分配、驻留、搬运、记账；
放置策略与搬运执行都在这一层。这一条取代 F5 的"SSD 单独一套"与 F9 的"RAM 单独一套"。

### 1. 边界（解耦的是决策，不是信息）

- **归子系统**：专家对象与 KV 页的分配、驻留、搬运、预取、淘汰、记账；三个预算及其内部分配。
- **不归它**：怎么把结果接进计算图。图只发一个请求、拿一个句柄，不感知层数与柜子名。
- **其他模块不改**：
  - dense 权重 / compute buffer / staging 的分配仍走现有路径（loader、调度器、`llama_memory`），
    只在初始化与扩容时向 store 申报（`store_declare(kind, bytes)`），由 store 从总预算里扣掉。
  - 加载方式不变：`lazy_read`、mmap、resident 判定、`lazy_read` 的 4 GiB 门槛全部照旧；
    专家读进 host 之后按层分组、按专家注册给 store（`store_adopt(kind = expert, il, ie, bytes)`）。
- **KV 按页**：页固定 **128 cells = 2.5 MiB**（tile 对齐，`--kv-block` 已移除），页是句柄与搬运单位，
  复用 F10/F11 的块表间接寻址；专家（1.945 MiB）与 KV 页是 store 的两个 size class。
- **调度者必须随请求给的信息**：`要哪些对象`、`在哪一步用`（层号 / ubatch 序号）、`能不能等`。
  缺了"在哪一步用"，子系统只能事后反应：要么过度预取把热点挤出去，要么预取不足退化成同步等待。
  现在的"预取只提前 1 层"就是没有这条信息时写死的折中。
- **store 不猜下一步**：路由预测（rolling promotion、按历史命中率提升）留在调用方，
  store 只按收到的 use_step 排 deadline。否则它迟早要理解"序列"与"路由"，KV 页就没法干净地共用一套接口了。
- **常开**：没有"关掉 store"的开关（用户要求，见第 11 节）。要复现旧行为就走预算参数与新参数，
  不靠旁路回退；store 建不起来（无 GPU、无 SSD 预算）时自动降级为"只有 RAM 一层"并告警，不得 abort。

### 2. 两种取物与它们的硬约束

- `能等 = 否`（同步交物）：这类请求永远存在，因为"用了哪几个专家"本身就是 GPU 在这层的中途算出来的
  （F1 的 `:1910` 就是这个性质的后果，不是 bug）。同步交物必须**只等这一次搬运**，
  不能退化成整设备 fence——这是子系统的前置原语，也就是 `DECISIONS.md` 未结项 19。
- `能等 = 是`（异步预取）：**立刻**返回一个可用的句柄，不是"稍后要等的凭据"；数据是否到位由句柄状态说，
  消费前在消费点等。语义见第 5 节。
- **句柄要能在执行期改变**：图建好之后钉死的是设备地址，柜子一换就错。
  复用 F10/F11 已经造好的间接寻址（块表经 `op_params` 传、内核按表寻址、长 prefill 先 gather），
  不另设计一套。

### 3. 红线写进代价，而不是写进调用方的记性

每层给出 `(带宽, 延迟, 一次能取多小, 能否异步)` 四个数，判据由代价算出：
SSD 随机读 0.81 GB/s、1.95 MiB 一次 2.4 ms，解码单步要 206 MiB -> 258 ms，
所以它对"单步关键路径"返回**不可用**，只对"预取批次"和"前缀 KV 复用"返回可用。
这条约束是数据结构，不是文档里的一句话。

### 4. 开工顺序

1. 先做等待原语（序号化：提交返回序号，可按序号等待，多批同时在飞）——它与接口无关，做了不白做。
   **已落地**（2026-10-27，`RESULTS.md` 0.6 E）：`ggml_backend_vk_prefetch_submit_seq` / `_wait_seq`，
   验证工具 `bench/tools/wait-seq.cpp`。调度器与 MoE cache 还没用它，那是 `DECISIONS.md` 未结项 19 剩下的半条。
2. 骨架 + 预算：句柄表、共享池、三个预算参数先立起来，并接管 `common_moe_plan_get()` 的**数值决策**
   （搬运暂时仍走旧 MoE cache 路径）。这一步不改行为，只换"谁算"，所以能用 ±1% 判等。
   与 D 的对齐（2026-10-27 评审）：D 的统一预算式先落 `common/fit.cpp`（`THREE-TIER.md` 第 0 节 0.0 的第一步），
   本步接管时把那个式子**整体搬进** store，`fit.cpp` 收成申报 + 打印的薄壳；过渡期两处并存即第二套算术，
   按 `DECISIONS.md` 第 5 节"预算解算只有一个所有者"处理。
3. MoE cache 变客户端：搬运交给 store，策略（SLRU / 整层驻留 / 预取环）先留在原处，等读数说话再移。
4. SSD 冷层进预算（`--store-ssd-mib` > 0）：它当前占用 0 字节，是最安全的一刀，放在这里而不是最前面，
   因为"常开"意味着不能先发布一个半截的 store。
5. KV 分页块与前缀 KV 段接入，`common/fit.cpp` 收成解析 + 打印的薄壳。里程碑与验收见第 11 节。

### 5. 两种取物的接口语义（2026-10-27 需求定稿）

子系统只有两个入口，调用方用"能不能等"选，其余全部由子系统决定：

| | 同步取 `store_acquire` | 异步取 `store_prefetch` |
|---|---|---|
| 返回时机 | 数据到位后才返回 | **立刻**返回 |
| 返回物 | ready 句柄，可被内核解引用 | 句柄立即可用于传参/进图，数据未必到位 |
| 谁等 | 调用线程阻塞，**只等这一个对象所属的批次** | 不阻塞；由消费点的 `store_wait` 等 |
| 典型调用点 | decode 里"GPU 刚算出这层用了哪几个专家" | prefill 的下一层、下一 token 的可预测需求、前缀 KV 复用 |

- 同步取永远存在，且必须只等序号（`ggml_backend_vk_prefetch_wait_seq`），不得退化成 `ggml_backend_synchronize`：
  否则 F12 就把未结项 19 想删的整设备 drain 又固化了一遍。
- 异步取"立刻返回可访问句柄"的前提是**句柄不是裸地址**：句柄 = 槽号 + 代际号，由子系统持有的间接表
  解析成当前落点。执行期换柜子只改表，不改图、不重建图。
- 未 ready 的句柄被消费属于未定义行为，`store_wait(handle)` 是唯一的合法消费点。建图阶段就检查
  "每个被解引用的句柄前面有 wait"：缺了不许以异步状态接进图，退回同步取后重试并计数，
  不留给调用方自觉。
- 一次 `store_acquire` 只保证这一个对象到位；同层其它对象要一次到位就合并成一次批量请求，
  批量之间用序号排序，先到先可用。

### 6. 线程池（异步取的执行者）

- **进程内只有一个**共享池，归子系统所有。复用 `ggml_threadpool` 那套实现（MoE cache 现在的
  `hybrid_tp` 是它的私有版本，`src/llama-moe-cache.cpp:471-482`），不新增每 cache 一个池，
  也不占用 CPU 计算池（会被 GEMM 的长任务堵住）。
- 池线程只做**发起与登记**：H2D/S2H 走 Vulkan 传输队列（`ggml_backend_vk_prefetch_submit_seq`），
  SSD 走重叠读，RAM<->VRAM 走 pinned staging。真正的搬运在拷贝引擎或内核里，所以池不需要大：
  默认 `min(2, max(1, hw/4))`，`--store-threads` 可覆盖。
- **池内线程绝不允许阻塞等待**。池任务只提交搬运并挂序号/完成回调；要等的人在调用线程上等。
  否则同步取会排在预取后面，decode 单步直接死掉。
- 同步取插队：同步请求进池的高优先级队列（后进先出）；在途预取允许在粗检查点取消
  （对象已被淘汰、层已过、目标槽被复用）。取消不是错误，只计数 `store_pool_cancel`。
- 池建不起来、池满、后端不暴露 prefetch 接口时，异步**退化为同步**（调用线程自己搬）+ 一次性 warning，
  **不得 abort**（`DECISIONS.md` 第 5 节）。

### 7. 三个预算参数（对外唯一的旋钮）

| 参数 | 默认 | 语义 |
|---|---|---|
| `--store-vram-mib` | 0 = 自动（设备可用量） | **整个运行期的 VRAM 总预算**：dense 权重、专家 bank、KV、compute buffer、staging 全算在内。手设 4096 / 6144 就是档位 |
| `--store-ram-mib` | 0 = 自动（空闲 RAM - 系统留量） | **整个运行期的 RAM 总预算**：resident 权重、`-nkvo` 的 KV、staging、mmap 页的驻留采样值 |
| `--store-ssd-mib` | 0 = 关 | 冷层容量上限；路径用 `--store-ssd-path` |

- **预算的构成** = 对象占用（专家 + KV 页）+ 申报占用（dense / compute / staging，由既有分配器报数）。
  store 不接管别人的分配器，只要求报数，这样"不影响其他模块"和"总预算成立"能同时满足。
- **只有一个 VRAM 参数**：`--store-vram-mib 4096` 就等于"这台机器跑在 4G 上"，不再有额外的设备档位旋钮。
  旧 env `GGML_VK_VRAM_LIMIT_MIB` 已移除，语义并进这一个参数。
- 预算是**硬顶**：store 初始化时把预算之外的部分 hold back（日志 `VRAM limited to ...` 保留），
  这样即使某条路径没走 store，也吃不掉那部分；最终态由 store 自己的分配器保证，hold back 只是兜底。
- **不允许绕过 store 访问**：专家与 KV 页只能经 store 分配/取用；其他分配（dense、compute、staging）
  必须在初始化与扩容时 `store_declare` 申报，未申报即越权。
  申报是**硬边界不是礼貌**：debug 构建对账失败直接 `GGML_ABORT`（这是开发期不变量，不是运行期容量条件），
  release 只告警并计数 `store_undeclared_bytes`；申报值必须能被 `GGML_VK_MEMORY_LOGGER` 独立复核，
  对账式：进程总量 vs `store_bytes_in(vram) + declared(vram)`，差 > 5% 即缺陷。
- **mmap 页不是账外资源**：loader 照旧建映射、照旧 `lazy_read`，但专家所在的映射区间用
  `store_attach_mapping` 把句柄登记给 store；store 只对**专家区间**控常驻
  （`madvise(MADV_DONTNEED)` / `PrefetchVirtualMemory`），驻留页数定期采样
  （Windows `QueryWorkingSetEx`、Linux `mincore`）记进 RAM 预算。不复制数据、不自管 `pread`。

- 三个预算**互不挪用**：显存不够不会去借 SSD，也不会把 RAM 预算折算成显存。跨层只走搬运，不走预算合并。
- 手设值是**硬上限**：超了由 store 自行降级（缩 bank / 关 SSD 冷层 / 退回 mmap）并计数，
  不 abort，warning 一次性 + 次数。
- 内部分配（专家 bank 多少、KV 工作集多少、staging 多少、预取环几槽）由 store 自己算，**不暴露成 CLI**；
  但结果必须在启动日志与记账里可见（`store_split: vram expert/KV/staging = ...`），否则无法复核。
- 现有 CLI 的关系：`--moe-cache-mib`、`-nkvo` / `-kvg`、`--kv-page*`、`-ncmoe` / `--cpu-moe` 全部降级为
  **强制覆盖**（用户显式给了就听用户的），不给时由 store 按 fit/auto 决定。`--moe-cache-mib` 解释为
  "专家块的建议上限"：未手设 `--store-vram-mib` 时完全照用；手设预算且建议值超出时才 clamp 并打一行 warning。
  这样旧臂的 bank 值仍可逐位复现，clamp 只发生在新的手设预算臂上。
- 第一版策略**照搬现状**：bank 大小与 KV 页数按 `common_moe_plan_get()` 现在算出的值，
  预取用现有 SLRU + 2 槽环（提前 1 层），只换执行者。margins、score 预留、resident 门槛仍由 `fit.cpp` 算并申报，
  不搬进 store。要换新策略（含 M2 的 `use_step` 预取）必须先有 A/B 读数，按第 11 节的门槛验收。

### 8. 句柄的三条硬约束

1. 搬运期间目标槽被 pin 住，不可淘汰。否则会出现"搬完发现落地处已经被别人占用"。
2. 淘汰只允许发生在代际号未变、无 in-flight、无未消费句柄时；不满足就跳过并计数 `store_evict_deferred`。
3. 释放显式化（`store_release` 或作用域句柄）。漏释放等于槽泄漏，按层打泄漏计数进日志。

- **单原语要求（2026-10-27 架构评审）**：第 2 条的三道检查（代际未变 / 无 in-flight / 无未消费句柄）在
  解引用、淘汰、回收三条路径上必须共用**同一个原语**（一次持锁或一个序号范围判定）；slab 的无 GPU 单测
  必须证明"淘汰走的就是解析用的那个判忙函数"。分三处各写一份即缺陷，那会重演 `:1910` 的形状：
  正确性绑在实现形状上而不是数据结构上（`DECISIONS.md` 判决 27 与第 5 节）。

### 9. 与现有组件的关系（不许长成平行子系统）

| 现有 | 变化 |
|---|---|
| `llama_moe_cache`（SLRU / 整层驻留 / 2 槽预取环 / bank 大小 / 搬运） | 整体交给 store：**专家的放置、预算、预取都归 store**，cache 只剩"按句柄取专家并接进图"。整层驻留保留为放置提示 + 批次单位 + 统计维度 |
| `common/fit.cpp` 的 `common_moe_plan_get()` | 只保留"非 store 占用"的计算并申报（`store_declare`）；bank / KV 的分给 store（M4） |
| 分页 KV（F10/F11） | 页就是句柄；页固定 128 cells = 2.5 MiB，`--kv-block` 已移除；块表与 `store_acquire` / `store_wait` 是同一套 |
| `hybrid_thread` / `hybrid_tp`（`src/llama-moe-cache.cpp:394-403`） | 并入共享池，删掉私有版本 |
| `llama-model-loader` / `llama-mmap` | **加载方式不变**（`lazy_read` / mmap / resident 全照旧）；只把含专家的映射区间 `store_attach_mapping` 登记（第 12 节） |

### 10. 记账（矩阵列之外，第一版就要有）

`store_sync_waits` / `store_sync_wait_ms` / `store_prefetch_issued` / `store_prefetch_lead_steps` /
`store_prefetch_wasted`（发了但没在 use_step 前被消费）/ `store_hits_by_tier`（vram / ram / ssd）/
`store_evict_deferred` / `store_inflight_pins` / `store_ssd_sync_rejected` / `store_pool_cancel` /
`store_split`（三个预算各自分给了谁）/ `store_bytes_in(vram|ram|ssd)`（越权检查的分母）/
`store_file_resident_mib`（mmap 驻留采样值，file 层用）。

`store_prefetch_wasted` 是这套接口唯一的自检指标：它高说明 use_step 给错了，不是硬件慢。
`store_sync_wait_ms` 必须能对上现在的 `cpy_sync`（`RESULTS.md` 0.7：单步 58.5 ms 里 42.3 ms），
对不上就是记账口径错了，不是收益。

### 11. 常开、A/B 与里程碑

- **常开怎么 A/B**：store 没有关掉的开关，所以对照靠预算参数复现旧行为（`--store-vram-mib` 设成与旧档位相同的
  总值，或直接用旧 `--moe-cache-mib` 覆盖值），而不是旁路回退。任何"必须先关 store 才能跑通"的设计都算设计错误。
- 这与 `AGENTS.md` 的"新行为一律带开关"不冲突：常开的是 store 这个**基础设施**，不是某项新策略；
  store 内部每项新策略（新淘汰算法、新预取提前量、SSD 放置规则）仍然各带开关并登记 `DECISIONS.md` 第 4 节。
- **M0 等待原语 已落地**（`RESULTS.md` 0.6 E）。
- **M1 专家注册 + 预算骨架**：专家按"层 + 专家子对象"两级注册，预取策略照搬现状（SLRU + 2 槽环、提前 1 层），
  只换执行者；上线 `store_declare`（dense / compute / staging 报数）与 `store_split`。
  M1 期间允许一个**影子模式**（store 只记录"我会怎么放、怎么搬"，执行仍走旧路径）用来对齐两套决策，
  **M1 结束必须删除** - 留着它就是事实上的总开关。
  验收：同口径 PP/TG 与基线差 <= 1%（8176 档 PP 506.7 / TG 17.2），越权差 <= 5%，
  并给一张 `hit %` 的新旧对照表（按层 / 按专家两种口径都要出）。
- **M2 预取升级（核心 KPI，唯一以收益验收的一步）**：`use_step` 驱动的异步取替代"提前 1 层"，
  共享池搬运、可取消。验收：`store_prefetch_wasted` 比 M1 下降 >= 50%，
  **TG 提升 >= 15% 才算收益**（< 15% 只算持平，按未结项 20 的规则）。
- **M3 KV 页接入**（页固定 128 cells = 2.5 MiB）**与 SSD 冷专家**（`--store-ssd-mib` > 0）：
  验收 `test-kv-page`、10K 输出逐字、`FA 5332/5332`、`SET_ROWS 319/319`（页没变，理论上应与现有分页路径一致）；
  单步路径 SSD 同步取放行次数 = 0、`status = ok`、abort 清单不新增。
- **M4** `common/fit.cpp` 只算"非 store 占用"并申报，bank 与 KV 的分配交给 store；
  `common_moe_plan_get()` 删除或转调。

收益验收仍守 15%（未结项 20）；M1/M2 是"不回退"门，用 ±1% 判，两者的口径不能混用。

风险有两条，都要在 M1 就量出来：抽象层把已验收的显存路径拖慢（所以 M1 只换"谁算"，不换搬运实现）；
句柄间接寻址在热路径多一次查表，预期 < 1%，超了就不许进单步路径，只留给 prefill 与预取。

### 12. 加载方式与注册粒度（2026-10-27 定案）

**加载方式一个字不改**：loader 继续按现在的路径解析、映射、`lazy_read`、resident / mmap 判定；
store 只在专家数据落进 host 之后接手 - 按层分组、按专家注册，此后这些专家的驻留与搬运归 store。

- **KV 按页**：页固定 128 cells（2.5 MiB，tile 对齐），页是句柄与搬运单位，`--kv-block` 已移除；
  专家是另一个 size class（1.945 MiB），两者共用一套 slab 分配器与句柄。

- **注册粒度：层 + 专家子对象（两级）**。层是分组与记账单位（`store_group(il)`，41 个），
  专家权重是层内的子对象（每个专家张量一个子句柄，约 3 万个）。
  **搬运、预取、淘汰的单位是专家子对象**；层负责"一起判命中、一起提交预取批次、按层报 hit %"。
  索引两跳：`层 -> 专家数组下标`，两级都是 O(1)。
- **为什么不再扁平到"单个专家"**：扁平后 3 万多个对象没有父节点，`bank` 的整层驻留、一层一次提交的预取批次、
  按层统计的 `hit %` 全都要在调用方把父子关系重新拼回去 - 等于把分组关系藏在使用者那里。
  两级结构把分组显式化，同时不牺牲搬运粒度。
- **为什么不只按层**：一层专家 116-330 MiB，8 GiB 设备上放不下几层，只能提前 1 层 -
  这正是现在 `cpy_sync` 42.3 ms 的来源。搬运单位必须细到专家，否则预取没有腾挪空间。
- **代价要认**：41 个层组 + 3 万多个专家子对象，句柄表与索引必须两级 O(1)（数组下标，不是 `map`）；
  淘汰单位是专家子对象，`bank` 原来的"整层驻留"保留为**放置提示 + 批次单位 + 统计维度**，不再直接等于分配单位。
  命中统计要能同时给出"按层"和"按专家"两种口径，M1 要在同一轮里都打出来。
- **为什么其他模块不注册**：dense 权重整个 run 不换层，注册只多一次查表；compute buffer 是每图的临时品。
  两者只影响"还剩多少预算"，申报即可，不动它们的分配路径。
- **mmap 只对专家区间移交**：`store_attach_mapping` 登记含专家的区间，其余区间仍归 loader / OS，store 不碰。

判据收口：要注册的对象必须满足"run 内会换层，且换层单位就是计算单位"。
专家子对象满足（换的是单个专家）；层是它天然的分组，不单独作为搬运单位；KV 页满足（page 是搬运单位）；
dense 不满足（永不换层）。上一版讨论里"字节全归 store"的说法作废，只保留"专家与 KV 页全归 store"。

## F13 子系统划分与边界（2026-10-27 定；配套 F12）

F12 定 store 做什么，本节定**边界画在哪**：哪些该独立成模块、哪些坚决不拆、store 内部怎么切。
以后每加一个模块，先用第 1 节的判据过一遍。

### 1. 判据（四条，至少满足三条才拆）

1. **生命周期不同**：谁先创建、谁先销毁（store 必须死在 backend 之前）。
2. **失败模式不同**：坏文件是事实错误必须硬失败，容量不足只能降级，两者不能共用返回值。
3. **变化频率不同**：一周改三次的（策略、预取）不该和半年不动的（加载、内核）绑在一起。
4. **有不变量能脱离 GPU 单独测**。

反向信号也要认：拆完两边必须互相回调才能完成一件事，那不是解耦，是把一个模块拆成两个更难调的东西。

### 2. store 内部：三个单元 + 一层薄壳（不另立子系统）

| 单元 | 职责 | 独立性要求 |
|---|---|---|
| `store-slab` | 分配、句柄（槽 + 代际）、pin、size class（专家 1.945 MiB / KV 页 2.5 MiB） | 纯数据结构，**无 GPU 可单测**；use-after-free 的唯一来源，越界与代际失配必须断言 |
| `store-io` | 搬运执行、共享池、序号完成、取消、代价表 | 唯一碰 Vulkan / 文件句柄的地方；换后端只动这里 |
| `store-policy` | 放哪、预取谁、淘汰谁 | 唯一该频繁变动与 A/B 的地方；M2 的策略实验只改这里 |

- 对外只有薄壳：`store_acquire / store_prefetch / store_wait / store_release / store_declare`。
- 三者共享同一份驻留状态，**不拆成独立子系统**（拆了就要互相回调）。
- 现状反例就是 `src/llama-moe-cache.cpp`（2082 行，SLRU + 预取环 + 上传 + 私有 worker + 统计 + 图接线挤在一起）。
  store 的第一版必须避免复刻这个形状。

### 3. 该独立成子系统的四个（S1 就是第 2 节的 store 三单元）

**S0 `llama-trace`（诊断与记账）— 先做，约一天**
计数器现在散在 `hybrid_waits`、`pf_*`、`promote_*`、`reallocs`、`splits` 各处，汇总靠脚本猜，
未结项 21（PERF_LOGGER 的 GEMM/FA 行被整类丢掉）就是猜错的后果。
做统一注册表 + 一次结构化 dump；以后新增指标不改脚本。落点 `src/llama-trace.{h,cpp}`。

**S2 `ggml-sched-plan`（执行计划与可见性）— P0，store M2 的前置**
`ggml/src/ggml-backend.cpp`（2400 行）现在同时管：分片决策、图分配、拷贝调度、event / 序号可见性；
`src/llama-context.cpp`（3982 行）与 `src/llama-graph.cpp`（3638 行）还要配合"producer 必须和 lookup 同分片"
这类正确性约定（`:1910` 的根因）。抽出一个模块，独占"这一步有几次拷贝、谁等谁、什么时候可见"，
并输出可打印的 plan（`skp_splits` / `skp_copies` / `skp_cpy_sync`）。

- 与 store 的接口：store 给"这批必须在第几步之前到位"，sched-plan 返回"要等到的序号"。
  **store 不自己等整设备 fence**；反过来 sched-plan 也不知道 store 内部有几个 size class。
- 验收（目标口径）：`cpy_sync` 从 42.3 ms 降到 **< 20 ms**，输出逐字相同，`splits` 不劣化；
  TG 收益按未结项 20 的 15% 门槛判（理论上限约 26 t/s）。

**S3 `kv-seq`（KV 语义与簿记）— 与 store 分开**
`src/llama-kv-cache.cpp`（2920 行）里混了 seq↔cell 映射、shift、扩容、页表、放置。
**seq↔cell、shift、state 恢复留在 KV 层，页的物理位置与搬运去 store。**
否则 store 迟早要理解"序列"，那它就不是硬件抽象了。落点 `src/llama-kv-seq.{h,cpp}`，
对外接口（`get_n_kv`、`n_cells`）不变；验收与 M3 同批，PPL 与 10K 输出逐字不变。

**S4 `store-file`（SSD 文件层）— store-io 下的独立文件，M3 之后**
队列深度、`FILE_FLAG_NO_BUFFERING` 对齐、重叠读的机制与 H2D 没有一行可复用。
随机 1.95 MiB = 0.81 GB/s 是设备上限（4 线程 qd4 也无收益），所以这里不做无谓优化，
只保证：只服务 SSD 预算、只在预取批次与前缀 KV 复用里出现、绝不进单步路径。

### 4. 明确不拆

| 东西 | 为什么 |
|---|---|
| ggml 算子 / Vulkan kernel | 多一层间接就是纯损失，性能代码要贴着硬件写 |
| 量化 / 反量化 | 已是独立编译单元，再拆没收益 |
| CLI 解析（`common/arg.cpp` 4729 行） | 拆了只是搬家，没有不变量可守 |
| CPU 计算池与 IO 池 | **不得合并**：计算池被 GEMM 长任务占住，IO 要低延迟，优先级相反 |

### 5. 顺序与依赖

| 序 | 事项 | 前置 | 验收 |
|---|---|---|---|
| S0 | `llama-trace` | 无 | 一次 dump 覆盖 F12 第 10 节全部 `store_*` 列；`summarize_perf.py` 不再猜 key |
| S1 | store 三单元（slab / io / policy） | 与 M1 同批 | slab 与 policy 无 GPU 单测通过；policy 可热换而 slab 不动 |
| S2 | `ggml-sched-plan` | S0 | `cpy_sync` < 20 ms、输出逐字、TG 按 15% 门槛 |
| S3 | `kv-seq` | M3 | PPL / 10K 逐字不变；store 不出现任何 seq 相关字段 |
| S4 | `store-file` | M3、`--store-ssd-mib` > 0 | 单步路径 0 次 SSD；冷层命中 >= 85% |

S2 是 store M2 收益的上限：预取能重叠多少，取决于调度器肯不肯让搬运和计算同时在飞。
所以 S2 排在 M2 之前，而不是「等 store 上线再看」。
