# RESULTS - 数字的唯一来源

每条都标日期、分支/提交、电源态与配置。跨行比较前先确认口径相同（见 `README.md` 第 4 节）。
新数字往本文件加，不要新开文档。

## 0. 目标口径下的矩阵（256K ctx + 10K prompt）

2026-10-23 16:3x-17:1x，`a44e13e08` + 工作区、构建 12:47，平衡 + AC，dGPU 独占，
`bash bench/tools/run-matrix.sh`，公共参数 `-dev Vulkan1 -ngl 99 -fa on -t 5 -b 2048 -ub 2048 -ncmoe 41
-f bench/prompts/prompt10k.txt -n 64 --temp 0 --no-warmup --no-display-prompt -v`，r=1。
`KV=host` 即 `-nkvo`；`used = peak - held`，`held` 是 `GGML_VK_VRAM_LIMIT_MIB` 自己 hold back 的部分。

| tier | load | ctx | KV | bank | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | status |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 不限 | none | 262144 | host | auto (1547) | - | - | 4660 | 0 | 4660 | 25404 | 0 | **abort** `:1910` layer 1 |
| 6144 | none | 262144 | host | auto | - | - | 2623 | 1234.8 | 1388 | 25403 | 0 | **abort** `:1910` layer 0 |
| 4096 | none | 262144 | host | auto (0) | 178.5 | 6.4 | 6424 | 3282.8 | 3141 | 26555 | 3 | ok，planner 判 bank 0 |
| 不限 | none | 262144 | host | 0 | 178.5 | 6.5 | 3128 | 0 | 3128 | 26554 | 3 | ok |
| 6144 | none | 262144 | host | 0 | 178.3 | 6.4 | 4368 | 1234.8 | 3133 | 26556 | 3 | ok |
| 4096 | none | 262144 | host | 0 | 178.4 | 6.4 | 6424 | 3282.8 | 3141 | 26555 | 3 | ok |
| 不限 | none | 32768 | gpu | auto (1719) | 479.4 | **15.5** | 5644 | 0 | 5644 | 19455 | 3 | ok |
| 6144 | none | 32768 | gpu | auto (484) | 467.9 | 12.7 | 5641 | 1234.8 | 4406 | 19454 | 3 | ok |
| 4096 | none | 32768 | gpu | auto (0) | 471.1 | 13.3 | 7212 | 3282.8 | 3929 | 19454 | 2 | ok |
| 不限 | mmap | 262144 | host | 0 | 142.5 | 6.2 | 3059 | 0 | 3059 | 27464 | 3 | ok（RAM 不足档） |
| 6144 | none | 262144 | host | 3072/2048/1024 | 未测 | 未测 | - | - | - | - | - | 待跑（阶梯找可用 bank） |
| 不限 | none | 262144 | host | auto | - | - | - | - | - | - | - | **已跑，见 0.2**：291 splits 里 KV 跨总线仅约 210 MiB（全程），状态是 `conv_state_last` 1.92 GiB/ubatch |
| 不限 | none | 262144 | host | auto | 未测 | 未测 | - | - | - | - | - | **待跑** F7 实现臂：`KV=host` 但 GDN 状态驻 GPU（开关未实现） |
| 6144 | none | 262144 | host | auto | 未测 | 未测 | - | - | - | - | - | **待跑** F8 归属修正：bank 应从 311 升到约 1848 且预取开（4096 档不在预期收益内，见 0.1） |
| 4096 | none | 262144 | host | auto | - | - | - | - | - | - | - | **已跑，见 0.2 表 A**：设备 compute 578.02（16K）-> 1712.02（256K），+1134.0 MiB，与 KV 放置无关 |
| 不限 | none | 32768 | gpu | 4096 | 未测 | 未测 | - | - | - | - | - | **待跑** F8 按 query 分块：bank 从 3584 用回 4096 且不撞悬崖 |

读数：

- **256K + `-nkvo` + cache 关闭：PP 178.3-178.5、TG 6.4-6.5，三档完全一致**。
  显存档位在这个工况上不参与结果，真实足迹 `used` 3.13 GiB，4 GiB 档还剩约 0.9 GiB 余量。
- **abort 在 8176 与 6144 两档都复现**（`:1910`，一个 layer 1、一个 layer 0），4096 档因为 planner 给 bank 0 而躲过。
  即目标工况目前只有"bank = 0"这条路能跑完，三层协同形同虚设 —— 这就是 F1 的实弹证据。
- 同一 prompt 换成 32K ctx + KV 在 GPU：PP 467.9-479.4、TG 12.7-15.5。
  所以 `-nkvo` 在 256K 上的代价是 **PP 479 -> 178（x2.7）、TG 15.5 -> 6.4（x2.4）**，
  与每 eval 的 splits 从 40 涨到 291 相吻合。
- **RAM 不足档（`-lm mmap`）只损失 PP**：178.5 -> 142.5（-20%），TG 6.4 -> 6.2。
  比短上下文时代记的"mmap 腰斩"温和得多，因为这里本来就没在用 bank。
- `bank` 阶梯（3072/2048/1024）在 256K 档还没跑：需要找一个既不为 0、又不触发 `:1910` 的容量。
- 表末五行是 `THREE-TIER.md` F7/F8 的验收臂，**本轮未跑、实现未落地**，只是把口径先钉在数字来源文件里，
  不要把它们读成任何收益。F7/F8 的归因还要求四个矩阵里没有的量：`splits`、`RS buffer size`、
  `-nkvo` 下长跨距的设备 compute buffer 增量、`sched_reserve` 与 teardown 两侧的 compute 归属
  （口径见 `README.md` 第 3 节）。

### 0.1 规划器预算式的读数（`bench/raw/m-256k-auto/*_c262144_bauto_r1.err`，2026-10-23 14:41-14:43，平衡 + AC，`-nkvo` + 256K + ub2048 + `-ncmoe 41`）

口径与第 0 节表头同一批工作区（源码 `common/fit.cpp` 12:46、`src/llama-graph.cpp` 12:45；当前 `build-vulkan/bin/llama-cli.exe` 是 13:07 那次构建，复核时以它为准）。

三档共同的部分（每台机都要重算，别当常数）：设备 `self` 3014 MiB = dense 模型 1302 + context 0（`-nkvo`）+ compute **1712.02**；
`margins` 由三档日志反解 = **1024 MiB**；score 缓冲预留 **1536.0 MiB**（`n_head 16`，`common/fit.cpp:958-960`）；
`per_expert` 1.82 MiB（日志：1547.9 MiB -> 852 slots）、最大整层 498.0 MiB、双环 930.0 MiB。bank 的规则是 `surplus - min(max(surplus,0), 256)`，
再按 `bank/per_expert` 取 slots、`bank/layer_bytes` 取整层数。

| 档 | free | 日志 surplus（已扣 1536） | 若归属改对（不扣） | 那样算出的 bank | 日志实际给的 bank |
|---|---:|---:|---:|---:|---:|
| 4096 | 4094 | deficit 1480.0 | +56.0 | **0**（被 slack 吃光） | 0，预取关 |
| 6144 | 6142 | 568.0 | 2104.0 | **1848.0**（>= 930，预取开） | 311.0，预取关 |
| 8176 | 7378 | 1803.9 | 3339.9 | **3083.9**（6 个整层） | 1547.9，预取开 |

出处：`vram4096_:5771`（deficit）、`:5772`（预留）、`:5773`（"no device memory for a cache"）、`:10290`；
`vram6144_:5771`、`:5773`（recommend 311）、`:5776`（312.0 MiB，no prefetch）；
`vram8176_:5771`、`:5773`（recommend 1547）、`:5776`（1547.9 MiB，prefetch available）。
`sched_reserve` 的设备 compute 与 splits 在 `:2886`、`:2888`（三档都是 1712.02 MiB / 291 splits），breakdown 在 `:2891`。

两条要一起看的读数：
设备 compute **1712.02 MiB 与显存档无关**，32K 那批只有 772.0 MiB；但这两个读数同时变了 `n_ctx` 与 KV 放置两个变量，
"约 940 MiB 随 `n_ctx` 走"只能算假设，需要同一 `-nkvo` 放置下的 n_ctx 对照（`THREE-TIER.md` F8 第 3 条）。
4096 档收尾还有一条 warning `Vulkan_Host compute buffer size is 1446.35 MiB, does not match expectation of 1120.18`（`:10290`），
即 host 侧 compute 运行中涨了 **326.2 MiB**。这两条共同决定了 4096 档的 56 MiB 余量到底是谁占的，F8 的归因测量必须先回答它们。

跑批器伪影（同日追查，必须记住）：旧的 PowerShell 版 `run-matrix.ps1` 用 `-RedirectStandardError` 接带 `-v` 的子进程，
把 decode 拖慢约 4.5 倍，于是本表之前的 TG 全在 3.0-4.9、并由此误产过"悬崖说"和"FA 回归"两条推论
（详见 `DECISIONS.md` 判决 13）。上表是 bash 跑批器的首次真实读数；
旧跑批器的 PP 列与本表相差 1-3%，只有 `abort`/`held`/`used`/`reallocs` 这些非时间读数可直接沿用。
短深度门（`llama-bench -p 512 -n 32 -r 3`，`--moe-cache-mib 3584`）= pp512 **376.42 ± 0.07** / tg32 **25.53 ± 0.96**，
`-fa off` 376.30 / 25.87 → 与昨天一致，现有短深度基线不作废。
FA kernel 今天反而更快：PERF_LOGGER 下同形状 `dst(256,16,1,1) k(256,10496,2,1)` 是 **317-323 us/层**，昨天记的是 445。

### 0.2 F7/F8 归因测量（2026-10-23 19:0x-19:4x，`a44e13e08` + 工作区，build 13:07，平衡 + AC，dGPU 独占，r=1）

公共参数：`llama-cli -dev Vulkan1 -ngl 99 -fa on -t 5 -b 2048 -ub 2048 -lm none -ncmoe 41 -v`，
`prompt10k.txt`（10428 tok），`GGML_SCHED_TRACE=1`（拷贝构成用 `=3`）。
除表 A 外一律 `--moe-cache-mib 0`：`-nkvo` + 非零 bank 会在 `:1910` abort（见本节读数 5）。
原始日志在 `bench/raw/attr-20261023/`。

**表 A `-nkvo` 下 `sched_reserve` 随 n_ctx 的走势（小 prompt，KV 全在 host）**

| ctx | CPU KV MiB | Vulkan1 compute MiB | Vulkan_Host compute MiB |
|---:|---:|---:|---:|
| 16384 | 320.00 | 578.02 | 160.18 |
| 32768 | 640.00 | 690.02 | 224.18 |
| 65536 | 1280.00 | 738.02 | 352.18 |
| 98304 | 1920.00 | 930.02 | 480.18 |
| 131072 | 2560.00 | 1154.02 | 608.18 |
| 196608 | 3840.00 | 1328.02 | 864.18 |
| 262144 | 5120.00 | **1712.02** | **1120.18** |

host 侧 compute 精确等于 `96.18 + 0.2 x KV`（7 个点全中）。设备侧 compute 在**设备 KV 为 0** 的前提下
从 578.02 涨到 1712.02（+1134.0 MiB），即存在一笔随 `n_ctx` 走、与 KV 本身无关的设备预留。
这就是 4 GiB 档缺的那把钥匙（`DECISIONS.md` 未结项 3 的第 3 条）。

**表 B 10K prompt、bank 0、KV 放置与 n_ctx 的解耦（PP/TG 的可用值见脚注）**

| ctx | KV | bank | PP t/s | TG t/s | prefill splits | copy ms/ub | CPU ms/ub |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32768 | gpu | 0 | 472.3 | 12.9 | 121 | 3680 | 1.7 |
| 65536 | gpu | 0 | 471.5 | 13.6 | 121 | 3720 | 1.8 |
| 262144 | gpu | 0 | **239.5** | 7.5 | 121 | 7400（峰值 13126） | 1.8 |
| 32768 | host | 0 | 178.6 | 4.8* | 291 | 2778 | 6032 |
| 65536 | host | 0 | 176.0 | 6.2* | 291 | 2780 | 6174 |
| 262144 | host | 0 | 172.0 | 3.6* | 291 | 2906 | 6289 |

`*` 这几臂是 `-n 8` 的 trace 跑批，TG 有暖机偏差，只用于对照 copy/CPU 列，不要与第 0 节的 `-n 64` 读数混用
（第 0 节 256K + `-nkvo` 的 PP 178.5 / TG 6.4 仍是该档的有效值）。256K host 行的 PP 172.0 还受
`GGML_SCHED_TRACE=3` 的逐拷贝 `fprintf` 拖累（32K/64K host 行是 `=1`，PP 178.6/176.0 可直读）。
同批 32K + KV 在 GPU + bank 1719 = **PP 483.6 / TG 15.4**（`-n 64`），复现了第 0 节的 479.4 / 15.5。

**表 C prefill 单 ubatch（2048 tok，`-nkvo`，splits=291）的跨总线构成**

| 方向 | 字节 | sync | 主要项 |
|---|---:|---:|---|
| CPU->Vulkan1 | 18.53 GiB | 1.85 s | `ffn_down_exps` 6870 MiB、`ffn_up_exps` 5616 MiB、`ffn_gate_exps` 5616 MiB、`attn_pregate` 320 MiB |
| Vulkan1->CPU | 2.38 GiB | 0.92 s | `conv_state_last` **1922.6 MiB**（30 层 x 64 MiB）、`new_state` 60 MiB、`Qcur` 10 x 32 MiB |
| KV | **约 210 MiB / 整个 prefill** | - | `cache_k/v` 只在第一个 ubatch（40 MiB）与最后一个（165 MiB）出现，相对 18.1 GiB/ubatch 的专家上传可忽略 |

**表 D decode 每 token 的跨总线字节（`-nkvo`，7 evals 平均）**

| 类别 | MiB/token |
|---|---:|
| KV 视图（`cache_k/v_l*`，10 层 x k+v） | **205.7** |
| recurrent 状态（`state_predelta` + `new_state`） | **144.8** |
| attention 输入 | 0.4 |
| 其它 | 3.3 |
| 专家权重 | 0（`-ncmoe 41` 下 decode 的 MoE 在 CPU 就地算） |

读数：

1. **prefill 期间 KV 几乎不跨总线**。`-nkvo` 的税不是"KV 搬运"：整个 10K prefill 只有约 210 MiB 的 KV 字节
   （第一个 ubatch 40 MiB + 最后一个 165 MiB），而专家是 18.1 GiB/ubatch。
   `DECISIONS.md` 判决 11 原写的"瓶颈是 KV 搬运"因此作废（改为下面的 2、3）。
2. **prefill 的真实构成是"专家上传 + CPU 注意力"**：每 ubatch 18.1 GiB 专家权重（bank 0 的必然结果）
   花 1.85 s，CPU 侧注意力花 6.0-6.3 s 且随 n_kv 线性增长（单个 ubatch 内 5.3 -> 13.5 s）。
   两者合计解释了 PP 178 对 PP 483 的绝大部分差距（剩余约 2.4% 是 bank，见读数 7），且都**不随 `n_ctx` 变化**
   （表 B 三行 host 臂的 copy/CPU 两列一致）。
3. **状态搬运确实存在，但载体不是 62.81 MiB 的 RS buffer**：prefill 是每 ubatch 1.92 GiB 的 `conv_state_last`（30 层 x 64 MiB），
   decode 是 144.8 MiB/token（`state_predelta` 68.6 + `new_state` 68.6）。F7 的方向对，量级要按这两个数改。
4. **decode 反过来是 KV 在搬**：205.7 MiB/token 的 KV 视图每 token 上传一次（decode 走 FA，需要 K/V 在设备侧）。
   205.7 MiB 在 14.3 GB/s 下是 14.4 ms，占 156 ms/token 的 9%；真正贵的是 `async_get` 的等待（约 34 ms/token）。
5. **`:1910` 的触发条件是 `-nkvo` + 非零 bank，与长上下文无关**：32K + `-nkvo` + `--moe-cache-mib 1719`
   同样在 layer 1 抛同一句 abort。所以 F1 不只是 256K 档的前置项，是所有 host-KV 臂的前置项。
6. **KV 分配大小本身有代价**：表 B 里 KV 在 GPU 时，256K（239.5）比 32K（472.3）低一半，
   而两臂的 bank、n_kv（都在 10.4K 量级）、splits、CPU 时间全部相同。
   差别是每 ubatch 的专家上传从 3.7 s 劣化到 7.4 s（峰值 13.1 s），即 5 GiB 的 KV 驻留把专家 staging 挤出了快路径。
   这一条是"按工作集分配 KV"最直接的支持证据。
7. **bank 在 KV 已驻设备时对 PP 只有 2.4%**：32K + KV 在 GPU 下 bank 0 = 472.3、bank 1719 = 483.6。
   所以第 0 节"479 对 178 的差"里，bank 的贡献远小于 KV 放置（把注意力留在 GPU）的贡献。
   这与 `DECISIONS.md` 第 3 节"缩小 bank 单调变慢"不冲突，但说明那批消融的 ctx 口径与这里不同，不能互推。

### 0.3 KV 按工作集增长（A+B1）落地后的目标矩阵（2026-10-23 21:0x，工作树含本次改动，平衡 + AC，dGPU 独占，r=2）

实现：K/V 初始只分配工作集大小的 cells，越界时按 2 倍增长（device-to-device 拷贝已用区域），
上限仍是 `n_ctx`；设备分配失败时回退 host（不 abort）。开关 `-kvg/--kv-grow N`（-1 = 自动，0 = 旧行为）。
不用 `-nkvo`：256K + 10K 的工作集只有 320 MiB，KV 直接驻设备。

命令：`bash bench/tools/run-matrix.sh -T 8176,6144,4096 -K -r 2 -O bench/raw/kvgrow-20261023`
（`-K` = KV 在 GPU；其余参数与第 0 节相同，未加 `--moe-cache-mib`，由 planner 自算 bank）。

| tier | run | bank | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8176 | 1 | auto (1481) | **478.5** | **13.7** | 5082 | 0 | 5082 | **19390** | 3 | ok |
| 8176 | 2 | auto (1481) | 475.6 | 13.7 | 5082 | 0 | 5082 | 19389 | 3 | ok |
| 6144 | 1 | auto | 445.7 | 12.7 | 5079 | 1234.8 | 3844 | 19389 | 3 | ok |
| 6144 | 2 | auto | 455.8 | 12.5 | 5079 | 1234.8 | 3844 | 19389 | 3 | ok |
| 4096 | 1 | auto | 465.4 | 14.0 | 6890 | 3282.8 | 3607 | 19390 | 2 | ok |
| 4096 | 2 | auto | 464.6 | 14.1 | 6890 | 3282.8 | 3607 | 19389 | 2 | ok |

同配置在加降级标志前的一批（`bench/raw/kvgrow-20261023`）是 477.2/477.6、447.5/449.9、466.0/461.0，差在噪声内；
`reallocs`、`used`、`hit` 三列两批完全一致，所以降级标志只影响失败路径。

对照第 0 节的同 prompt 基线（256K + `-nkvo` + bank 0）：PP 178.5、TG 6.4、`used` 3128 MiB、RAM 峰值 26554 MiB。
**PP x2.67、TG x2.16，RAM 峰值下降 7.2 GiB**，且三档全部 `ok`，4G 档不再退化成"无 cache 的 host 专家路径"。

机制读数（8176 档同一臂）：

- 初始 `KV buffer size = 160.00 MiB`（8192 cells），日志 `ensure_capacity: K/V grown to 16384 cells (320.00 MiB)` 一次。
- 增长后 `sched_reserve` 从 `Vulkan1 1490.02 / Vulkan_Host 1088.09 MiB` 降到 `708.02 / 128.09 MiB`，
  即 A 自动带出了 B1 的效果（reserve 的 mask 按当前 `n_kv` 而不是 `n_ctx`）：设备省 782 MiB、host 省 960 MiB。
- planner 首次在 256K 档给出非零 bank：`recommend --moe-cache-mib 1481 -> 815 slots, 2 whole layers`，且预取打开。
- 关闭新行为（`-kvg 0`，同一 256K 档）：KV 5120 MiB、device compute 1490.02 MiB、PP **239.4** / TG 14.3。
  这与第 0.2 节表 B 的 256K KV-on-GPU 读数一致，说明收益来自分配大小而不是别的变量。

数值等价（同一二进制）：32K + `--moe-cache-mib 0`、`temp 0`、`-n 32`，`-kvg 0` 与 `-kvg -1` 的生成文本
逐字相同（只有耗时行不同）；`ctest` 32/32。

口径限制：这条只对"工作集远小于 `n_ctx`"的工况有效。真的把 256K 填满时会一路增长到 5120 MiB，
重新撞上 0.2 节表 B 的容量悬崖（那是预期行为，不是回归）。增长次数是 log 级，每次有一次图重分配（`reallocs`）。

### 0.4 长跨距 score 缓冲按 query 分块（B2）落地后的目标矩阵（2026-10-23 17:4x-18:5x，`10f651ac3` + 工作区，平衡 + AC，r=2）

实现：`src/llama-graph.cpp` 的长跨距 matmul 注意力把 query 维按预算切块，每块的 score 缓冲
`n_kv x chunk x n_head x 4B` 不超过 `-fac/--fa-chunk-mib`（默认 256）。外层 `-ub 2048` 与完整历史注意力范围不变，
K/V 读取与 softmax 范围也不变，只是把一次 1280 MiB 的 score 换成若干 256 MiB 的块。
规划器的 score 预留同步改成这个上限（`llama_fa_long_kv_reserve_bytes()` 多一个参数）。
每块少于 64 行时这个路径退回 FA（块太窄，dispatch 不划算），所以超长上下文仍走已知路径。

命令：`bash bench/tools/run-matrix.sh -T 8176,6144,4096 -K -r 2 -O bench/raw/b2-chunk256-20261023`
（`-K` = KV 在 GPU，其余与第 0 节相同；`-fac` 默认值已进二进制，矩阵不需要额外参数）。

| tier | run | bank MiB | slots/group | prefetch | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | hit % | status |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8176 | 1 | 2879.2 | 791 | 开 | **679.9** | **20.3** | 6122 | 0 | 6122 | 19481 | 2 | 57.6 | ok |
| 8176 | 2 | 2879.2 | 791 | 开 | 679.0 | 20.5 | 6122 | 0 | 6122 | 19481 | 2 | 57.6 | ok |
| 6144 | 1 | 1644.2 | 451 | **关** | 490.4 | 14.5 | 6130 | 1234.8 | 4895 | 19484 | 2 | 14.6 | ok |
| 6144 | 2 | 1644.2 | 451 | 关 | 489.9 | 14.5 | 6130 | 1234.8 | 4895 | 19485 | 2 | 14.6 | ok |
| 4096 | 1 | 0 | - | 关 | 476.4 | 14.2 | 6535 | 3282.8 | 3252 | 19484 | 1 | - | ok |
| 4096 | 2 | 0 | - | 关 | 473.6 | 14.2 | 6535 | 3282.8 | 3252 | 19484 | 1 | - | ok |

`slots/group` 是两个 group（q6_K 20 层 / q4_K 20 层）各自的 slot 数；整层预取要求每组 >= 512。

对照 0.3（同口径、同 prompt、同 `-ncmoe 41`）：8176 的 476-478 / 13.7 -> **680 / 20.4（PP +42%、TG +49%）**；
6144 的 446-456 / 12.5-12.7 -> **490 / 14.5（PP +8.6%、TG +15%）**；4096 的 465 / 14.0 -> 475 / 14.2（+2%）。

**按 cap 的消融（8176 档，同一构建、同一批窗口，r=2）**

| `-fac` | bank MiB | slots/group | prefetch | PP t/s | TG t/s | used MiB | hit % | reallocs |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0（旧行为） | 1481.2 | 406 | 关 | 456.0 / 460.5 | 13.4 / 13.3 | 5082 | 6.3 | 3 |
| 64 | 3969.2 | 1091 | 开 | 659.4 / 655.9 | 21.2 / 20.0 | 6294 | 62.1 | 1 |
| 128 | 3603.2 | 990 | 开 | 661.1 / 683.9 / 664.4 / 682.5 | 21.2 / 21.3 / 21.0 / 21.8 | 6184 | 65.3 | 1 |
| **256（默认）** | 2879.2 | 791 | 开 | **679.9 / 679.0** | 20.3 / 20.5 | 6122 | 57.6 | 2 |
| 512 | 2359.2 | 648 | 开 | 624.8 / 628.5 | 19.4 / 19.5 | 5858 | 59.2 | 2 |

读数：

1. **主收益不是"注意力更快"，而是把 1280 MiB 的 score 从设备账上换成 256 MiB**：8176 档 bank 1481 -> 2879 MiB，
   每组 slots 406 -> 791，跨过 512 的预取门槛；6144 档 bank 246.2 -> 1644.2（每组 66 -> 451 slots，仍差 61 slots）。
2. **0.3 里"预取打开"是规划器日志的误读**：plan 那句 `whole-layer prefetch available` 只比较总 bank 与 930 MiB，
   实际 cache 按 group 检查 `2 x n_expert = 512` slots，8176 档 406 slots 时是关的（`prefetch needs 512 slots per group`）。
   本轮 cap 256 之后 791 slots，预取才真的生效：`splits 114 -> 40`、hit 6.3% -> 57.6%。
   这条修正不影响 0.3 的 PP/TG 读数，但"0.3 已带预取"的说法作废。
3. **cap 不是越小越好**：64 档 bank 最大（3969 MiB、hit 62.1%）但 PP 反而比 256 低 20 t/s，
   因为 n_kv 16384 时每块只有 64 行 query（score 行宽 1 MiB），块数升到 32，dispatch 与 K 重读抵消了 bank 的收益。
   256 MiB 时 reserve 图（`n_kv` 16384）每块 256 行、8 块，10K prompt 实际（`n_kv` 10240）每块 409 行、5 块，
   是这轮实测的平衡点。
   128 与 256 在 PP 上同级（噪声内），128 的 TG 高 1 t/s、256 的 PP 更稳，默认取 256。
4. **reserve 账目变干净**：增长后设备 compute 1428.02 MiB（0.3 是 708.02），差额就是现在真的进了 reserve 图的
   256 MiB score 块与分块节点；teardown 的 `matches expectation` 两侧一致，本轮没有 `does not match` 警告。
5. **6144 档还差一步**：451 slots < 512，预取仍关，splits 114、hit 14.6%。要么让规划器为双环多留 220 MiB，
   要么放宽组的 slot 门槛（`THREE-TIER.md` F3），这是下一档最直接的 PP 来源。
   前瞻臂（`-B 1920`，r=2，同批）：PP **675.8 / 679.7**、TG 19.1 / 19.2、hit 55.2%、splits 42、
   used 5161 MiB（档位 6142，仍余约 1 GiB）、`status = ok`。也就是这一档 PP 从 490 到 678 只差那 61 slots/组，
   F3 在 6144 上的收益是 +38%，不是边际优化。
6. **4096 档仍然 bank 0**：score 只从 1536 降到 256，surplus 仍不够（planner 判 `no device memory for a cache`），
   这一档的钥匙还是设备 compute 的其它归属，不是 score（与 `DECISIONS.md` 未结项 3 的结论一致）。
7. 数值：`bash bench/tools/quality-gate.sh` 对 B2 之前的 logits 基线给出 **KLD -0.000000、maximum 0.000064、
   same top p 100.000%**，PPL 5.798313 对 5.798292。分块不改变数学，只改变切分，质量门通过。
   交付二进制（补了"每块少于 64 行退回 FA"的下限）复跑 8176 单臂 = PP 678.5 / TG 20.3、质量门仍 PASS，
   与上表的差额在噪声内；该下限只在 `n_kv` 约 64K 以上才生效，本口径不触发。
   同批短深度回归对照（`llama-bench -p 512 -n 32 -r 3 --moe-cache-mib 3584`）：pp512 374.29 ± 0.36、
   tg32 25.68 ± 0.87，对 0.1 节的 376.42 / 25.53 无回归。

## 1. 8 GiB 档 短上下文消融阶梯

2026-10-23，`a44e13e08` + 工作区，构建 10:38，「平衡」+ AC，r=2，公共参数
`-dev Vulkan1 -ngl 99 -fa on -t 5 -ncmoe 41 -b 2048 -ub 2048 -lm none`。
注意 `llama-bench` 的 `n_ctx = p + n`，所以 pp512 是 ctx 512、pp8192 是 ctx 8192。

| 配置 | 与基线的差异 | pp512 | pp8192 | tg32 | 独占显存峰值 |
|---|---|---:|---:|---:|---:|
| 全开 | 无 | 262.9 | 187.0 | 29.08 | 8143 MiB |
| 关 q6_K dense `_bk2` | `GGML_VK_Q6K_BK2=0` | **417.1** | 189.8 | 26.76 | 8143 MiB |
| 关长 KV matmul 判据 | `LLAMA_FA_LONG_KV_MIB=0` | 263.2 | **293.9** | 27.44 | 8143 MiB |
| 关整层预取 | `LLAMA_MOE_PREFETCH=0` | 253.2 | **111.2** | 28.93 | 8143 MiB |
| 关 ids 并入 split | `LLAMA_MOE_STAGE_IDS=0` | 262.3 | 186.6 | 27.84 | 8143 MiB |
| bank 4096 | `--moe-cache-mib 4096` | 384.5 | 665.5 | 24.99 | 7146 MiB |
| bank 3584 | `--moe-cache-mib 3584` | 375.2 | **732.7** | 25.65 | 6641 MiB |

- tg32 列在同一 bank 的各臂之间只是噪声（`stddev_ts` 约 1.4），判断开关要看**同一 bank 同一测试**的那对臂。
- slots 覆盖：6144 = 1690（33.0%）/ LRU 666 / 8 整层；4096 = 1126（22.0%）/ 614 / 4 层；3584 = 985（19.2%）/ 729 / 2 层。
- q6_K `_bk2` 在不溢显存的 bank 3584 上也是负的：pp2048 810.45 对 860.35（-5.8%），pp8192 729.67 对 763.90（-4.5%）。
  q6_K 与 q4_K 的真实算力差是 6.61 对 9.26 TF（1.4x），不是早先记的 1.8x。
- 2026-10-22 的另一批（混合电源态，行与行差约 x1.33，**不可与本表混用**）：
  static 373-376 / cache 405 / 418.8，pp2048 566 / 582 / 865.5（ctx 2048 口径）。

## 2. 8 GiB 档 长上下文

### 2.1 256K 与 32K（2026-10-19，`moe-gpu-cpu-hybrid` @ `b10fca3bd`，resident，`-fa on`）

| 场景 | 配置 | prompt | PP | TG@depth | 峰值 VRAM |
|---|---|---:|---:|---:|---:|
| ctx 32K 通用 | `-ncmoe 41 --moe-cache-mib 5120 -ub 2048 -t 5 -c 32768` | 10428 | **401.3** | **20.9** | 7752 MiB |
| 同上 | 同上 | 19758 | 333.9 | 20.2 | 7752 MiB |
| 真 256K | `-ncmoe 41 -c 262144 -ub 2048 -t 5`（无 cache） | 10428 | 363.4 | 14.4 | **7956 MiB** |
| 同上 | 同上 | 19758 | 309.9 | 13.7 | 7956 MiB |
| 省显存 | `-ncmoe 34 -c 32768 -ub 2048 -t 5` | 10428 | 387.1 | 15.8 | 5683 MiB |
| 同上 | 同上 | 19758 | 325.0 | 15.5 | 5683 MiB |

溢出与退化（同一批）：cache 6144 @ctx 32768 峰值 8765 MiB（PP 368）；static `-ncmoe 27` @32768 峰值 8837 MiB
（PP 78.3 / TG 9.1）；static 34 @131072 峰值 8120 MiB（157 / 15.0）；cache 5120 @40960 峰值 7946 MiB（255 / 11.0，可用但性能崩）；
cache 512 @262144 峰值 8468 MiB（不可行）。

13.2k prompt @ctx 32768 粗扫（±3%）：cache5120 394.0/19.9、cache4096 386.8/19.5、static34 380.8/15.7、
static38 368.4/14.8、static41 365.2/14.3；`-ub 4096` 与 `-t 8` 无收益；`-ctk/-ctv q8_0` static34 227.9/15.4。

KV 成本 **20 KiB/token**（f16，混合架构只有 **10** 层是全注意力：`n_layer = 40`、`full_attention_interval = 4`，
日志直接给出 5120.00 MiB / 262144 cells = 20 KiB，见 `bench/raw/m-256k-bank0/vram8176_c262144_b0_r1.err:2129`），
256K 需要 5.0 GiB。本节早前写的 "21-25 KiB/token、11 层" 是推算口径，与日志冲突，以这里的实测为准。

### 2.2 ctx 16384 + 10K prompt（2026-10-23，工作树 `trim-vulkan-cpu-qwen35moe`，「平衡」+ AC，`-lm none -fa on -t 5 -ncmoe 41 -n 64`，桌面另占约 1.4 GiB）

| bank | PP | TG | 独占显存峰值 |
|---|---:|---:|---:|
| 6144（判据默认开） | 158.0 / 161.3 | **4.1 / 4.2** | 8008 MiB，3 次 graph reallocation |
| 6144 + `LLAMA_FA_LONG_KV_MIB=0` | 253.7 | 16.5 | - |
| 判据自动（规划器 ~4.2 GiB） | 395.0 | 15.3 | 5.79 GiB |
| 4096 | 413.9 | 19.1 | 6.25 GiB |
| 5120 | 406.0 | 16.6 | 7.20 GiB |
| 3584 判据关 | 544.5 | 21.8 | 6641 MiB（消融表口径） |
| 3584 判据开 | **646.1** | **22.3** | - |
| 4096 判据开 | **646.9** | 21.7 | - |
| 4096 判据关 | 546.5 | 22.3 | - |

同一批里 bank 3584 的 A/B 是 413.9/18.9（判据关）对 454.5-514.6/18.5-18.6（判据开，PP +24%），峰值 6922 MiB；
4096 判据开的那臂不稳（411-473.8）；3072 时 TG 掉到 14-17。短上下文回归对照（cache 6144，pp256/tg256）
237.6/25.8 对改动前 240.9/26.2，约 1.5% 漂移；本 session 的显存悬崖在约 7.2 GiB。

matmul 注意力判据的 score 缓冲实测 **+1276.7 MiB**（compute buffer 1628.83 对 352.11 MiB），
10k prompt 时 1.34 GB（n_head x n_q x n_kv x 4B），默认上限 `LLAMA_FA_LONG_KV_MIB=1536` MiB。

每 eval 毫秒（FA on / off）：2048: 81/83；4096: 3203/3021；6144: 3960/3248；8192: 4651/3420；10240: 5314/3576；尾批 188: 1784/1503。
斜率：FA 0.344 ms/每 KV token（1.07 TFLOPS）对 matmul 0.0905 ms（4.08 TFLOPS），q=2048；
这里的斜率是整段图上的每 KV token 毫秒数，与层数无关，但折算到单层要用 **10** 层全注意力而不是旧口径的 11 层
（0.0344 对 0.0313 ms/层/token）；
截距 C = 1.79 s（FA）对 2.65 s（matmul）；10k prefill 里 FA 多花约 4.5 s（总约 25 s）。

FA decode 微基准（cache 3584，ctx 16384，`-n 8`，PERF_LOGGER）：`FLASH_ATTN_EXT dst(256,16,1,1) k(256,10496,2,1)`
= **445 us/层** = 48 GB/s（每层 21.5 MB KV，地板 224 GB/s）；split_k 扫（us/层）1→5210、4→1455、8→823、14→565、
21→497、**28(auto)→438**、32→740、42→764、56→667、84→OOM、224→983，固定开销约 4.5 us/WG，28 CU 上 42 WG = 1.5 waves。
端到端 TG 五臂：base 18.8/18.9、w4 15.4/18.4、w6 17.7/15.2、w8 17.1/14.5、w6+bc64 17.8/15.1。

### 2.3 ubatch 与上传（2026-10-20..21，`307d47a27`）

10k prompt：`ub2048`/5120 = 397.9-406.6（峰值 7731，上传占 21%、计算 79%）；`ub3072`/5120 = 433.1；
`ub4096`/4608 = 437.8-438.5（峰值 7598）；`ub4096`/4096 = 435.4-443.3；`ub4096`/5120 = 100.8（峰值 8110，溢出）；
20k `ub4096`/4096 = 355.6；256K `ub4096` = 89（9226 MiB，必须 `ub2048`）。`ub4096` 把上传从 71278 降到 40765 MiB（-43%）。

整层预取在 10k 上：首层 37.4 ms/ubatch（共 187 ms），每层等待 1.1-2.0 ms，约 0.43 s = 1.7%。

## 3. TG 每 token 分解

2026-10-20，`09cacdfd8`，cache 6144，n=32，干净窗口：**37.6-40.5 ms/token = 26.6 t/s**（llama-bench）/ 24.7（llama-cli）。

| 组成 | 值 |
|---|---|
| splits | 32 |
| copy 合计 | 23-30 ms = cpy_sync 12.3 ms（30 次整设备 drain）+ h2d body 11-18 ms（30 次 ids 回读，每次又一次 drain） |
| gpu_submit | 6.3-10 ms（32 次提交） |
| GPU op | 约 21.7 ms |

GPU op（PERF_LOGGER 下取的，绝对值被放大 2.6x，只能看占比）：MUL_MAT_VEC 407 x 30.7 us = 10.34 ms（47.6%）、
MUL_MAT_ID_VEC 78 x 51.9 = 4.05、MUL_MAT_ID_MUL 39 x 37.9 = 1.48、FLASH_ATTN_EXT 11 x 128.3 = 1.41、
MUL_MAT_ID 11 x 129.6 = 1.43、TOPK_MOE_EARLY_SOFTMAX_NORM 42 x 27.3 = 1.15、RMS_NORM_MUL 139 x 5.4 = 0.75。

命中与上传（`PARADIGM-D1-GATE` @ `5aeab4cae`）：hits 5628 / misses 3613 = **60.90%**，上传 6603.28 MiB / 32 token
= **206 MiB/token** = 15.5 ms @ 13.9 GB/s；每层上传 6.9 MiB = 0.50 ms，而每层 MoE 计算只有约 0.16 ms（3 倍）；
握手约 10.1 ms（0.34 ms/层：30 次 drain 0.41 ms + 回读/发起 0.37-0.6 ms）。

`L2-DECODE-LATENCY`（base `4cf3d44a9`）：splits 42→32、copy 35.3→25.0 ms、h2d body 24.4→13.7、gpu_submit 13.5→8.7，
命中率 68.4%、78 misses/token、约 140 MiB/token，tg32 26.83（26.72-26.91）对 static 18.96。

整层数量扫（r=3）：每组 6/5(default)/4/3/2/1/0 层 → LRU slots 154/410/666/922/1178/1434/1690，
pp512 406-408/402.4-402.6/390.3/387.1/384.8/375.0/372.8，tg32 21.9/26.41-26.44/27.01/27.12/26.99/27.10/26.07（TG 在约 27.1 饱和）。

多 token 并发（2026-10-22，cache 4096，ctx 32768，q8_0 KV）：np=1/2/3/4 = **17.2 / 29.1 / 39.3 / 47.2 t/s**（np4 2.75x），
每步 58/69/76/85 ms（边际 token 约 9 ms，固定开销约 85%）；ctx 94096 时 np1 17.1、np2 29.0、np4 23.6；
cache 6144 时 np4 只有 18.5-20.6。

投机解码：MTP np4 = PP 136.4 / TG 18.1-18.4（accept 152/256 ≈ 0.59）对 base np4 394.1 / 45.6-47.1；
MTP np1 = 385.4 / 11.2-11.9（accept 45/64 ≈ 0.70）对 base 422.7 / 12.0-12.1；另一轮 24.8 对 14.6。
ngram-mod：重复型文本 10.5 → 22.8 t/s（2.17x，`n_max 16`；n12 20.6、n20 22.0），默认 `n_max 64` = **0.69x**，
新颖文本 10.2 对 10.2（1.00x）。

## 4. PP 每 eval kernel 分解

2026-10-22，pp2048 ub2048，预取开，`-ncmoe 41 --moe-cache-mib 6144`，「平衡」态，eval 3.9 s、GPU busy 3.88 s：

| op | ms/eval | 占比 | 实测效率 |
|---|---:|---:|---|
| CONCAT（30 层 GDN） | 751 | 19% | 2.7 GB/s（67 MB/层） |
| MUL_MAT dense | 955 | 25% | q4_K 6.3-6.6 TF、q6_K 3.6 TF |
| MUL_MAT_ID | 899 | 23% | in-situ 3.5-3.7 TF |
| FLASH_ATTN_EXT | 359 | 9% | 36 ms/层 |
| GATED_DELTA_NET | 153 | 4% | - |
| elementwise（RMS_NORM_MUL/MUL/ADD/GLU/SSM_CONV） | 约 270 | 7% | - |

NOOP-copy 探针（跳过上传）同样 516 t/s，说明 ub2048 已是 compute bound。
CONCAT 用 `ggml_cont(transpose)` 物化后 pp2048 728.37 → **865.51**（+18.8%，约省 446 ms/eval）。

## 5. 硬件天花板

| 量 | 值 | 日期/来源 |
|---|---|---|
| H2D 单向 | **14.2-14.3 GB/s**，SDMA 与内核拷贝、3 种队列族、chunk 16 KiB..512 MiB、K=1..64 全同；同方向多队列不叠加（2 腿 13.60、3 腿 14.09） | 2026-10-21，`bench/raw/h2d-bw` |
| 链路真实能力 | 15.75 GB/s 载荷上限的 91%（Gen3x16 或 Gen4x8，MPS256/MRRS512）；Windows 报的 4/16 是虚拟化能力位 | 同上 |
| D2H | 13.55-13.94 GB/s；staging memcpy 10.8-11.2 | 同上 |
| D2D | 74.6 GB/s 拷贝（约 149 GB/s 显存读写） | 同上 |
| 全双工（transfer-only 队列族） | 23.28 / 22.18 GB/s 合计 | 同上 |
| 显存带宽地板 | 224 GB/s；dense GEMV 实测 216-217 GB/s（约 97%） | 2026-10-20 |
| 每 token 最小读 | dense 1.31 GiB + 命中专家约 0.37 GiB = **1.81 GB** → 8.1 ms → **123 t/s** | 屋顶线 |
| SSD（F: Micron 2450 1TB，`FILE_FLAG_NO_BUFFERING`） | 顺序 8 MiB x 64 = 2.31 GB/s；随机专家三元组 1.95 MiB qd1 = **2.53 ms = 0.81 GB/s**；4 线程 qd4 = 0.80 GB/s（无队列深度收益） | 2026-10-22 |
| SSD 进 decode 关键路径 | 206 MiB/token → 258 ms → 3.9 t/s；90% 命中（58 MiB）→ 72 ms → 14 t/s；全 miss 582 MiB → 728 ms | 同上 |
| CPU 读带宽 | 1T 22.1 / 2T 36.7 / 4T 39.7 / 8T 37.8 / 16T 30.3 GB/s | 2026-10-21 |
| CPU MoE 专家 arm | 8 个不同专家 @4 线程 0.868 ms/层 = **0.108 ms/专家**、16.4 GB/s；线程中位数 1/2/3/4/5/6/8 = 1.29/0.966/0.837/0.868/0.892/0.964/1.202 ms（3-4 线程顶点，8 线程比 4 慢 38%） | 2026-10-22 |
| 纯 CPU 全模型 tg32 | t=1/2/4/6/8 = 2.41 / 5.36 / 5.29 / 4.43 / 3.93 t/s | 同上 |
| bank 上限 | 本机单调变快至 6144（382.4），6400 起崩到 57.0；bank 还必须 <= 设备单次分配上限 2 GiB | 2026-10-18 / 10-22 |

## 6. 门禁与机制实测

- Gate1 每层边界（2026-10-21，`gates/`）：GPU work 0.206 ms；A alloc+record+fence 0.209、B prerecord+fence 0.194、
  C prerecord+timeline 0.191、D depth-2 pipeline **0.115**；对照调度器 **0.64 ms/split**。
- Gate2 双队列重叠：1400 MiB/200 ms → compute 90.6 ms（369.7 GB/s，被 Infinity Cache 抬高，不是带宽）+ transfer 104.8 ms
  （13.0 GB/s），串行 192.8，并行 **105.5**（隐藏 99.3%）；400 MiB/100 ms → 并行 49.1 对串行 73.4（78.0%）。
- M2 CPU/GPU 并发：CPU 4 专家 0.575 + GPU 0.330，串行 0.905，并发 **0.639**（隐藏 80.4%，需常驻线程）。
- 回读并入 split：h2d 4.2-4.6 → 2.1-2.9 ms/token，tg32 交叉 r=3 = 29.28/28.50/28.39 对 28.30/27.45/27.23（**+3.8%**）。
- 整层预取合并：pp2048 580.36 → **728.37**（+25.5%），pp512 382.45 → 406.77（+6.4%）。
- deferred 上传：`LLAMA_MOE_DEFER_UPLOAD` + `GGML_VK_DEFERRED_COPIES`，只覆盖 34% 上传字节，tg32 26.50 → 26.90（+1.5%），pp512 -0.7%。
- 目标化 wait：原语 0.028 ms/call，cpy_sync 1286 → 44 ms，但 pp512 378.9 对 382-383（-1%），叠加 ring 后 233-236。
- M1 执行器（cache 4096）：基线 LRU 21.24±1.00 → +hybrid+static 11.39 → +rolling 15.70（命中 83.7%）→ +M1b 18.26，
  pp512 335.17 对 318.37。splits 82→42、h2d 22.8→2.3 ms/token，但每层 ids 往返 0.60-0.80 ms + CPU arm 暴露 14 ms，
  净负。**tg32 单 token 串行地板 20-22 t/s。**
- 图内 `--moe-hybrid`：TG 6.1 对基线 15.9，splits 42→122，cpy_sync 72.5-102.9 ms/token。
- 回归门禁（当前全绿）：`test-backend-ops -o MUL_MAT_ID -b CPU` 945/945、`-o MUL_MAT_ID_MASK -b CPU` 4/4、
  `-o MUL_MAT_ID -b Vulkan1` 939/939、`-o MUL_MAT -b Vulkan1` 2275/2275、`-o FLASH_ATTN_EXT -b Vulkan1` 5332/5332、
  `ctest --test-dir build-vulkan` 32/32；`-o MUL_MAT_ID_MASK -b Vulkan1` = 0/0（Vulkan 无此 op）。

### 6.1 F10 M0 spike（分块 KV 的前置验证，2026-10-23 21:5x，`60feaa082` + 工作区，平衡 + AC，dGPU 独占）

三件事：确认目标工况真正走哪条 FA 路径（M0b）、块表经 BDA 在多个独立分配上是否正确（M0c）、
按块寻址对 FA 读取的代价（M0d）。

M0b 口径：`GGML_VK_FA_TRACE=1` 的目标命令（256K ctx + `prompt10k.txt` + `-n 64`，`-lm none`，未设显存档位）。
日志把每个不同的 FA 形状打一行（`bench/raw/m0-20261023/fa-trace-256k10k.err`）：

| N | KV | path | Br | Bc | D_split | row_split | sgs | wg | aligned | f32acc | flags | sparse | dequant_kv |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | scalar | 8 | 32 | 8 | 4 | 64 | 256 | 1 | 1 | 10 | 0 | 0 |
| 2048 | 2048 | scalar | 8 | 32 | 8 | 4 | 64 | 256 | 1 | 1 | 11 | 0 | **1** |
| 198 | 8448 | scalar | 8 | 32 | 8 | 4 | 64 | 256 | 1 | 1 | 11 | 0 | 0 |
| 8 | 10496 | scalar | 8 | 32 | 8 | 4 | 64 | 256 | 1 | 1 | 10 | 0 | 0 |
| 8 | 10752 | scalar | 8 | 32 | 8 | 4 | 64 | 256 | 1 | 1 | 10 | 0 | 0 |

- PP 与 TG 都是 `FA_SCALAR`（RDNA2 无 coopmat），没有独立的 decode phase（`flash_attn_decode_phase_*` 只在 Intel Xe 路径）。
  decode 的 N 是 8 而不是 1：`n_head_q/n_head_kv = 16/2` 的 gqa 合并在 `ggml-vulkan.cpp:8316-8324` 把 N 提到 8。
- `Bc = 32`，所以分块大小只需满足 `BS % 32 == 0` 与 `BS % 256 == 0`（后者来自 `get_n_kv` 的 padding），2048 两者都满足。
- `flags` 10 = 老 AMD Windows 路径 + mask；11 = 再加 `use_mask_opt`。分页沿用同一组变体，不新增 flags。
- `sparse = 0`：目标工况没有走稀疏聚集，分页不需要考虑与 gather 的交互。
- `dequant_kv = 1` 只在第一个 ubatch 出现，原因是那一批 `n_kv == kv_size` 让 `is_dense_kv_cache()` 成立；
  它会把整段 K/V 拷进 f16 scratch。分页后这个判据仍会"看起来连续"，所以 M2b 必须在分页时显式关掉它。
- 这次运行的 MoE bank 是 0（`common_moe_plan_impl: no device memory for a cache`，桌面占用导致赤字），
  所以上面这轮**不产出 PP/TG 数字**，也不与 `0.3`/`0.4` 比较。

M0c（`bench/tools/vkpage.cpp table`，两个能力已确认：`bufferDeviceAddress=1`、`shaderInt64=1`、
`storageBuffer16BitAccess=1`）：8 个逻辑块放到 3 个**独立 VkBuffer** 的 3 个槽位里，映射 `2l+1 mod 3`，
块表是 `uint64[8]` 的设备地址数组。写入经表、读回经表、按物理位置逐 cell 校验、未映射槽位必须全 0：

| 检查 | 结果 |
|---|---|
| 读经表（512 cell x 4 vec4 逐值比对） | ok |
| 未映射块的非零 half | 0 |
| 物理布局（每块每 cell 的 c/d/blk/1 编码） | ok |
| 总判定 | **PASS** |

M0d（`bench/tools/vkpage.cpp bench`，每 cell 读 2 KiB = K 512 + V 512 个 f16，一个 workgroup 处理 `chunk` 个 cell；
`chunk=256`、r=20-50）：

| cells | K+V | 连续 buffer | 经块表 | 差值 |
|---:|---:|---:|---:|---:|
| 10240 | 20.0 MiB | 0.034 ms / 616.6 GB/s | 0.035 ms / 597.3 GB/s | +3.2% |
| 32768 | 64.0 MiB | 0.317 ms / 211.7 GB/s | 0.299 ms / 224.3 GB/s | -5.6% |
| 131072 | 256.0 MiB | 1.600 ms / 167.8 GB/s | 1.614 ms / 166.4 GB/s | +0.9% |
| 262144 | 512.0 MiB | 3.189 ms / 168.3 GB/s | 3.201 ms / 167.7 GB/s | +0.4% |

- 10K 这一行带宽 616 GB/s 远高于 224 GB/s 的显存带宽：20 MiB 全在 Infinity Cache 里，是热缓存读数，不是 DRAM 读数。
- **测量陷阱**：同一程序用 `chunk=1024`（10K 只有 10 个 workgroup，1280 个线程塞不满 28 CU）时得到
  0.025 对 0.043 ms（**+69%**），换成 `chunk=256`（40 组）就变成 +3.2%。这个 +69% 是占用率造成的，不是表寻址；
  记在这里是为了防止被后人当结论引用。`chunk=64`（160 组）时 10K = -5.3%、128K = +0.3%。
- 这个微基准**每个 cell 查一次表**，是上界。真实 FA 一个 tile 查一次（`Bc=32`），查表次数少 32 倍，
  所以第 5.7 节"< 1%"的假设成立，而且富余很大。
- 128K 只有 168 GB/s 而非 224 GB/s：这个 kernel 是两个流 + f16 转 f32 的纯读，不是带宽上限探针；两臂用的是同一套代码，
  差值仍可比。

复现：
```
cd bench/tools && glslc -O vkpage_rw.comp -o vkpage_rw.spv
glslc -O vkpage_fa.comp -o vkpage_fa.spv && glslc -O -DPAGED=1 vkpage_fa.comp -o vkpage_fa_paged.spv
g++ -O2 -std=c++17 vkpage.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o vkpage.exe
vkpage.exe info | table | bench 131072 20 256
```

### 6.2 F10 M1：分块 KV 的 CPU 参考实现（2026-10-23 23:0x，工作区，`--kv-page` 默认关）

`--kv-page` / `--kv-block` 已进树（`llama_context_params` -> `llama_cparams` -> `llama_kv_cache`），
块表经 `op_params` 传给 `FLASH_ATTN_EXT` 与 `SET_ROWS`（`ggml_flash_attn_ext_set_paged` /
`ggml_set_rows_set_paged`，见 `THREE-TIER.md` 4 节）。M1 只实现 CPU 内核，设备 K/V 会被守卫挡下并打
`kv-page disabled: device K/V needs the paged Vulkan kernel (M2b)`。

单元级（`tests/test-kv-page.cpp`，已进 CTest，共 33 个用例）：

| 检查 | 结果 |
|---|---|
| paged FA 对 contiguous FA（8 块乱序映射 `(5b+3)%8`，`BS=32`） | **4096/4096 逐位相同**，max\|diff\| = 0 |
| contiguous FA 对 host 双精度参考 | max\|diff\| = 8.6e-3（f16 K/V 内核的正常量级，阈值 2e-2） |
| paged SET_ROWS：64 个乱序 cell 写到物理块 | 0/64 错位 |
| paged SET_ROWS：锚（填 NaN）被写坏的 half 数 | 0 |

模型级（CPU 后端，`-ngl 0 -c 512 --kv-block 256`，`bench/raw/m0-20261023/cpu-kvpage-{off,on}.{out,err}`）：

| 臂 | 放置 | 输出 | PP | TG |
|---|---|---|---|---|
| `off` | 连续 512 cells | "The user gave me a classic pangram" | 30.0 t/s | 4.1 t/s |
| `on` | 2 块 x 256 cells（10 MiB） | 同上，逐字相同 | 29.8 t/s | 4.1 t/s |

- 这两行是 CPU 后端读数，**不与第 0 节的 Vulkan 矩阵比较**；它只证明寻址正确。
- `init_pages` 日志：`paged K/V: 2 blocks of 256 cells (10 layers), table 320 bytes, 10.00 MiB allocated`。
- 分页模式下 `get_can_shift()` 返回 false（K-shift 逐块化在 M3），`state_write/read` 已按块拆分
  （`page_write_run` / `page_read_run`），`state_read_meta` 先 `ensure_blocks`。
- 回归：`ctest --test-dir build-vulkan` **33/33**（新增 `test-kv-page`）。

### 6.3 F10 M2b：Vulkan 侧的进度与两个新发现（2026-10-24 00:2x-02:xx，工作区，**设备路径仍未开启**）

已完成（在树里，但设备 K/V 仍被守卫挡下）：

- `flash_attn.comp` 增加分页分支：`Paged` / `PageBS` 两个 spec constant（16、17），
  K/V 绑定在分页时换成块表（`uint64` 设备地址数组），每个 f16vec4 由 `fa_load_kv()` 经
  `PageKV(blk)` 解引用。非分页路径逐字未变：`FLASH_ATTN_EXT -b Vulkan1` **5332/5332**。
- `ggml_vk_flash_attn` 侧：`paged` 时强制 `FA_SCALAR`、关闭 `use_dequant_kv` 与 `use_sparse`、
  关闭 Intel Xe 分支，并把块表绑定到 binding 1/2；`page_bs` 进 `vk_fa_pipeline_state`（参与 pipeline 缓存键）。
- `GGML_VK_FA_TRACE` 的读数确认了这条链路真的走到：`paged=1 bs=256 ... sparse=0 dequant_kv=0`。
- `SET_ROWS` 的分页变体（`copy_to_quant.comp` 的 `PAGED` 分支、`pipeline_set_rows_paged[2]`、
  表绑到 binding 2、虚拟 dst 步长走 push constant `param3`）已进树，支持 f16 与 f32 源。
- 表从"裸地址"改成 **I64 张量**（`src[5]/src[6]` / `src[3]`），调度器负责绑定，不再需要新 descriptor。

单元级（`tests/test-kv-page.cpp`，现在两种后端都跑：`test-kv-page.exe [vulkan]`）：

| 检查 | CPU | Vulkan1 |
|---|---|---|
| paged FA vs contiguous FA（8 块乱序，BS=32） | 4096/4096 逐位相同 | **4096/4096 逐位相同** |
| paged SET_ROWS 写位置（f32 源 -> f16 目标） | 0/64 错位 | 0/64 错位 |
| 锚（填 NaN）被写坏的 half 数 | 0 | 0 |

**模型级仍然不对**（这是设备路径被关掉的原因）：`-c 512` 同一文本，
`llama-perplexity --chunks 2` 得到 PPL **4.1412（off）对 6.0684（on）**，
`llama-cli -st` 两臂输出也不同。已排除的项：FA 形状与路径（trace 全部 `paged=1 bs=256 scalar`）、
表地址（`ggml_vk_buffer_write` 是同步等 fence 的）、两臂的算子级寻址（上表）。
未定位的剩余差异：多层的表/锚（测试只有一层）、`n_embd_k_gqa=512`（测试 256）、
以及分页下 `ggml_backend_tensor_get` 读到 0（当时没有先 `llama_synchronize`，回读本身不可信）。
下一步的调试入口：先在 `llama_synchronize` 之后把块内 K 与连续布局的锚逐字节对比，
再按层数（1 层 vs 10 层）二分。

两个设计层的新发现（M0 没覆盖到）：

1. **长跨距 prefill 的 matmul 注意力路径无法分页**。`build_attn_mha` 的 `fa_long_kv` 分支用
   `MUL_MAT` 直接读 K/V 矩阵，读的是视图的连续布局，分页后就是锚。因此 `fa_span` 已加 `!paged`：
   分页模式下所有跨度都留在注意力内核上。代价是 `-fac 256` 带来的 PP 680 用不上（同一档 `-fac 0` 是 458，
   `RESULTS.md` 0.4）。三条候选：(a) 接受回退；(b) 每层每 ubatch 一次 gather（`THREE-TIER.md` F11）；
   (c) 只在 KV 装不下显存时才启用分页。
   **2026-10-24 决定取 (b)**（用户确认），当天已落地并验收（`RESULTS.md` 6.4）：
   gather 让分页 prefill 的 PP 从 449 回到 530，略高于连续布局的 510。
   当天早些时候曾记过"取 (c)、gather 暂不实现"，那条作废；(c) 只剩**默认模式选择**的含义——
   `--kv-page auto` 由 `common/fit.cpp` 的预算判决（初始 KV 装得进设备 surplus 就保持连续布局），
   **这个 auto 还没实现**，不影响已落地的 gather。
2. **`SET_ROWS` 的分页变体需要一个独立的 SPIR-V 变体**（`PAGED` 是编译期宏，不是 spec constant），
   并把块表绑到 binding 2、虚拟 dst 步长走 push constant（`param3` 放 `PageBS`）。已落地（见上）。

### 6.4 F11 L0 + L1：分页内存路径打开 + prefill gather（2026-10-24，工作区，未做目标口径矩阵）

本次只验正确性，**没有 256K + 10K 三档矩阵读数**（下面最后一行的 PP 是 bank 0 口径，不能当性能结论）。

三个在实现中发现的事实，按重要性：

1. **分页 `SET_ROWS` 一直被 ROPE/RMS_NORM 的融合旁路**（`ggml-vulkan.cpp:14940` 的 `ROPE_VIEW_SET_ROWS`、
   `:14902` 的 `RMS_NORM_VIEW_SET_ROWS`）。融合把 rope 的输出直接写进 SET_ROWS 的 dst，也就是分页模式下的
   256 B 锚，块表根本没被使用：块永远是 0（实测 block0 的 131072 个 half 全 0），注意力读到全 0 的 K/V。
   `-c 512` 的输出因此与 `off` 不同，而 `-c 8192` 直接 `vk::DeviceLostError`（往 256 B 的锚里写 128 KB+）。
   **这就是 `RESULTS.md` 6.3 里"SET_ROWS 修好了但输出仍然不同"的真正原因**：6.3 的 SET_ROWS 修复是对的，
   但它没被执行。修法是给两个融合判据加 paged 检查（`ggml_set_rows_get_paged` 为真就拒绝融合）。
2. 打开设备 K/V 后，`--kv-page` 的 PPL 与连续布局**完全相同**（`-c 256 --chunks 1`，同一份 256 token 文本）：

   | 臂 | PPL |
   |---|---:|
   | `off`（连续） | 2.1676 +/- 0.35231 |
   | `on`（`--kv-page --kv-block 256`，设备 K/V） | 2.1676 +/- 0.35231 |

3. 长 prompt 逐字一致 + gather 真的触发（4535 token、`-ub 2048`、`-c 8192`、temp 0、`-n 8`）：
   `off` 与 `on` 生成文本逐字相同；`GGML_KV_PAGE_TRACE` 显示 `gather layer 3: 8192 cells, 4 blocks, workspace 8192 cells`
   （工作区 4 -> 8 -> 16 MiB 三次增长）。同一批的 PP 是 **120.86（off）对 118.55（on）**，但这一轮
   planner 判 deficit、bank = 0，注意力被专家上传压住，**不能用来评价 gather**（口径见 `README.md` 3 节）。

4. **目标工况读数（r=2，8176 档，同一 session，桌面占用未清）**。命令 = AGENTS 的统一口径
   （`-c 262144 -ub 2048 -f prompt10k.txt -ncmoe 41 -lm none`），只把 `-n` 从 64 改成 16：

   | 臂 | PP (r=2) | TG (r=2) | planner bank | 说明 |
   |---|---:|---:|---:|---|
   | `off`（连续布局，`-fac 256`） | 509.94 / 508.65 | 17.93 / 18.48 | 2879 MiB | 对照 |
   | `--kv-page`（默认 gather） | **530.48 / 528.93** | 19.03 / 19.09 | 2847 MiB | 工作区 4 -> 8 -> 16 MiB，gather 在 ubatch 2 起生效 |
   | `--kv-page --kv-gather-mib 0` | 449.20 / 449.83 | 18.97 / 18.97 | 2879 MiB | paged FA，等于 F11 之前的行为 |

   即 **gather 让分页 prefill 从 449 回到 529（+18%），并稳定高于连续布局的 509（+4%）**；TG 持平。
   注意这一轮连续布局只有 509 而不是 0.4 的 680（同一档位、同 bank），说明 session 之间有约 25% 的口径漂移，
   **所以这只作方向性证据，不能替代 L1d 的三档 x r=2 矩阵**（6144/4096 与 host 块臂仍缺）。
   产物：`bench/raw/f11-20261023/{tgt,final2}-{off,on,nogather}.{out,err}`。

5. **质量门带 `--kv-page` 通过**（`-c 4096 --chunks 2`，同一份 1.89 GiB 基线 logits）：
   `Mean PPL(Q) 5.798313` 对基线 `5.798292`，`mean KLD -0.000000`、`max KLD 0.000064`、`same top p 100.000%`
   -> **GATE PASS**。即 paged FA + gather 与连续布局在 logits 上等价（与 B2 的验收强度相同）。
   产物：`bench/raw/f11-20261023/gate-on.{out,err}`。

门禁：`ctest` 33/33；`FLASH_ATTN_EXT -b Vulkan1` **5332/5332**；`SET_ROWS -b Vulkan1` **319/319**；
`test-kv-page`（新增 gather 用例，CPU 与 Vulkan 各跑一次）两项全 PASS：
`gather K: 0/64256 halves differ, tail non-zero = 0`、`gather V: 0/64256 halves differ`。

相关产物：`bench/raw/f11-20261023/`（ppl2-{off,on}、t5-{off,on}、final-{off,on}）。

## 7. 质量与参考数字

### 7.1 质量门（F6，2026-10-23 恢复）

`tools/perplexity/` 从 `a44e13e08` 之前的树恢复（`git checkout 2fa1da4f7^ -- tools/perplexity`，
`tools/CMakeLists.txt` 重新 `add_subdirectory`），`build-vulkan/bin/llama-perplexity.exe` 可用。
一条命令：`bash bench/tools/quality-gate.sh`（`-R` 重录基线）。

口径：`prompt10k.txt`（10428 tok）、`-c 4096 --chunks 2`、`-b/-ub 2048`、`-fa on`、`-ncmoe 41`、`-lm none`，
ctx 受"至少 2 x n_ctx token"限制，所以 10K 文本最大只能到 4096；它照样覆盖长跨距 matmul 路径
（`n_kv 4096 > 1.5 x n_q 2048`）。基线 logits 存 `F:/LLM/kld/qwen35moe-10k-c4096.kld`（2.0 GB，不进 git）。

| 构建 | PPL | 对照基线 KLD | max KLD | same top p |
|---|---:|---:|---:|---:|
| B2 之前（录制基线，自比） | 5.798313 ± 0.237848 | -0.000000 ± 0.000000 | 0.000064 | 100.000% |
| B2，`-fac 256`（对上面的基线） | 5.798313 ± 0.237848 | -0.000000 ± 0.000000 | 0.000064 | 100.000% |

即 query 分块在当前构建上**数值等价**（分块不改变每行的归约范围）。这是后续 F5/F8 类改动的验收前提。

- PPL（wiki 文本，`llama-perplexity` 时代）：cache 开与关同为 **4.0688 ± 0.51413**（L6、STAGE1 各一次，文本逐字相同）。
  那是旧 arch 与旧工具时代的读数，只作历史参考，与 7.1 不可比。
- hybrid 质量扫：全 CPU 6.1161、LRU4096 6.1973、hybrid4096 6.1852、hybrid100MiB 6.3628、
  zero-residents/map-all-miss/mask-forced 三种强制臂都精确回到 6.1161；156/156 驻留行逐字节相等。
- repack（纯专家排列，理论上应严格等价）：KLD **0.01516 ± 0.00438**（1 chunk）/ 0.00920 ± 0.00265（2 chunk），
  top-1 93.548% / 95.161%，self-KLD -0.00001..0；PPL 1.8638 → 1.8958（+1.7%）；tg 18.7 → 19.0。
  原因判定：累积顺序扰动约 1e-7 相对值翻转近平局路由选择，非映射错误（router 轴 `max|R[j]-O[perm[j]]| = 0.000000`）。
- LFM2.5 24B-A2B（arch 已从本树删除，仅作硬件能力参考）：champion `-ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048`
  pp2048 **874.3**、pp512 530.3、tg128 31.35、pp8192 753.3、pp16384 618.4（q8 KV），峰值 8143/8176 MiB。
  每 token 足迹 835.7 MiB、dense 地板 2.62 ms（对照 Qwen3.6 的 582.3 MiB/层 1.820 MiB 专家、地板 6.41 ms）。
- 官方 vs fork（LFM2.5，official scoop 0.7.0-dev b11763 Clang 对 fork b11750 GCC）：
  官方最佳 pp512 326.6 / pp2048 739.9 / tg128 26.05；fork mixed 530.8 / 874.8 / 31.20（+62.5% / +18.2% / +19.8%）。
  真实生成（1203 token prompt）：官方 519.3/24.84，fork mixed 710.1/24.42。server：官方 261.4/20.9 对 fork 686.5/19.7。
  纯 CPU（`-ngl 0`）：fork pp512 347.3-349.0 对官方 187.1-193.1（+81%），但 tg128 8.54 对 18.16；
  batch=1 多线程扩展 fork 2.0x 对官方 4.0x（t1 4.21/4.49，t6 8.54/18.16）。

## 8. 归档口径（不可与上表混用）

mmap 时代（build `623c16393`，`-ncmoe 27`）：pp512 145.9-148.7、tg128 14.25-14.36、pp2048 45.5、
每 token 1.87 GiB、cache4096 tg 4.7、436 次 sync/token、全 barrier 关 tg 15.30（+21%）。
resident 早期（2026-10-18，`a3eea7166`）：pp512 336.17±1.87 / tg32 17.74±0.05，`-lm mmap` 171.31（2.06x 差），
`-fa off` 时 pp2048 ub512 163.82、ub1024 77.67（病态）、ub2048 92.80。
ubatch 拟合模型（mmap 时代）：M=128/256/512/1024/1536/2048 = 39.98/70.39/138.50/66.79/284.58/317.55，
A = 3.02 s、B = 1.666 ms/token、拐点 M* = 1814。
