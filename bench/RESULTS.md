# RESULTS - 数字的唯一来源

每条都标日期、分支/提交、电源态与配置。跨行比较前先确认口径相同（见 `README.md` 第 4 节）。
新数字往本文件加，不要新开文档。

## 0. 目标口径下的矩阵（256K ctx + 10K prompt）

2026-10-23 16:3x-17:1x，`a44e13e08` + 工作区、构建 12:47，平衡 + AC，dGPU 独占，
`bash bench/tools/run-matrix.sh`，公共参数 `-dev Vulkan1 -ngl 99 -fa on -t 5 -b 2048 -ub 2048 -ncmoe 41
-f bench/prompts/prompt10k.txt -n 64 --temp 0 --no-warmup --no-display-prompt -v`，r=1。
`KV=host` 即 `-nkvo`；`used = peak - held`，`held` 是显存档位模拟自己 hold back 的部分
（历史读数用的是旧 env `GGML_VK_VRAM_LIMIT_MIB`，2026-10-27 起并入 `--store-vram-mib`，数值不变）。

| vram | load | ctx | KV | bank | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | status |
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

| vram | run | bank | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | status |
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

| vram | run | bank MiB | slots/group | prefetch | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | hit % | status |
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

### 0.5 规划器 v2（`--fit-plan 1`，默认）+ CONCAT 快路径（2026-10-27）

代码改动（设计来源 `.research/review-20261027/`，四份文档只记读数）：

1. `margins` 从固定 1024 MiB 改成 `max(256, min(margins, free/8))`（`common/fit.cpp`）。
2. `fa_score` 只在注意力真的跑在设备上时才从设备 surplus 扣（判据 `offload_kqv`，同一文件）。
3. 预取双环按 **expert group** 判：`bank >= 2 * expert_total / n_layers_of_that_group`。
   本模型 2 组各 20 层 -> **1860 MiB**；旧式是拿总 bank 比 `2 * n_expert * per_expert` = 930 MiB（平均层字节）。
   分组键是新增的 `llama_moe_geometry::expert_layout`（`src/llama-ext.h` / `src/llama-model.cpp`）。
4. 小 bank 地板：`bank < n_expert_used * per_expert * n_layer_expert / 4`（本模型 145 MiB）时归零。
5. CONCAT dim 0 行拷贝快路径（`GGML_VK_CONCAT_FAST`，默认开）：`concat.comp` + `ggml-vulkan.cpp`。
6. `LLAMA_FA_SPAN_SQUARE`（默认**关**，已证伪，见读数 4）。

口径：AGENTS 统一口径（`-c 262144 -ub 2048 -f prompt10k.txt -n 64 -ncmoe 41 -lm none`），`-K` = KV 驻设备（不加 `-nkvo`），
「平衡」+ AC，dGPU 独占。**本轮全部结论来自同 session 配对或交替跑**，因为本 session 的 PP 在 4 次重复里从 487.7 掉到
378.8（±15%），足以吃掉一次真实的 +10%；跨 session 更不可比（见读数 6）。

**A. 6144 档，交替配对（r=1 x2，2026-10-27 10:2x）**

| 臂 | PP t/s | TG t/s | bank MiB | hit % | prefetch |
|---|---:|---:|---:|---:|---|
| `--fit-plan 0`（旧规则）r1 | 394.5 | 12.3 | 1644 | 14.6 | 关 |
| `--fit-plan 1`（v2）r1 | **510.1** | **15.7** | 1908 | 55.1 | 开 |
| `--fit-plan 0` r2 | 393.3 | 12.4 | 1644 | 14.6 | 关 |
| `--fit-plan 1` r2 | **508.8** | **15.9** | 1908 | 55.1 | 开 |

**PP +29.3%、TG +28.3%，臂内方差 < 1%**。机制：margins 1024 -> 760 让 bank 从 1644 涨到 1908，
跨过 cache 自己的 per-group 门槛（1860 MiB），预取从关变开。`THREE-TIER.md` F3 关闭。

**B. 8176 档 A/B/A（同一 session，交替 4 次）**

| 臂 | PP t/s | TG t/s |
|---|---:|---:|
| base | 487.7 | 13.7 |
| v2 | 461.9 | 13.6 |
| base | 403.7 | 13.4 |
| v2 | 378.8 | 14.6 |

同一臂的两点差 21%，臂间差在其内 -> 8176 判**中性**（v2 只多 110 MiB bank，预取两侧都已开）。

**C. 4096 档与 bank 地板**

| 臂 | bank MiB | PP t/s | TG t/s | hit % |
|---|---:|---:|---:|---:|
| base | 0 | 360.5 | 12.4 | - |
| v2（加地板之前） | 116 | 359.9 | **10.3** | 0.0 |
| v2（加地板之后） | 0 | 385.7 | 12.7 | - |
| base（同批对照） | 0 | 385.2 | 12.7 | - |

116 MiB = 64 slots 的 bank 只买到每层的 lookup split，一笔 hit 都没有 -> 地板规则成立；加地板后与 base 逐位相同。

**D. 新默认的矩阵（`--fit-plan 1` 默认 + CONCAT 快路径默认开，3 档 x r=2，2026-10-27 10:3x）**

| vram | run | bank | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | hit % | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8176 | 1 | auto (2989) | **511.7** | **16.6** | 6231 | 0 | 6231 | 19482 | 2 | 58.7 | ok |
| 8176 | 2 | auto (2989) | 511.7 | 16.5 | 6231 | 0 | 6231 | 19482 | 2 | 58.7 | ok |
| 6144 | 1 | auto (1908) | 501.2 | 15.4 | 6385 | 1234.8 | 5150 | 19480 | 2 | 55.1 | ok |
| 6144 | 2 | auto (1908) | 510.1 | 15.7 | 6385 | 1234.8 | 5150 | 19481 | 2 | 55.1 | ok |
| 4096 | 1 | 0 | 385.1 | 12.0 | 6535 | 3282.8 | 3252 | 19485 | 1 | - | ok |
| 4096 | 2 | 0 | 385.6 | 12.1 | 6535 | 3282.8 | 3252 | 19484 | 1 | - | ok |

三档 `status = ok`，档内 r1/r2 差 <= 1.8%。

读数：

1. **6 GiB 档的钥匙一直是预算归属，不是容量**：同一块显存，1024 MiB 的固定 margin 就少 6 个 slot/组，
   差 61 slots 正是 0.4 读数 5 里"前瞻臂强给 1920 才拿到 +38%"的那 61 个。现在规划器自己给出 1908。
2. **预取是否打开必须按 cache 的 per-group 判据算**。旧式的 930 MiB 是平均层字节的两倍；两组各 20 层时真实需求
   1860 MiB（0.1 的"若归属改对"表预告 1848，本轮实测落地）。
3. **4 GiB 档在 116 MiB 量级上"bank 非零"是负优化**（TG -17%、hit 0%）。`DECISIONS.md` 未结项 15 里
   "bank 非零在算术上可达"仍然成立，但要求腾出 500 MiB 以上（margins 地板 256 + 未归因 compute 464 是仅有的两块）。
4. **方阵跨度（`n_kv == n_q`，一个 prompt 的第一个 ubatch）换到 chunked matmul 更慢**：
   total 2947/3028 ms 对 2870/2888（perf logger 口径），SOFT_MAX 535 -> 572/607、MUL 396 -> 404/411、PP 也低。
   机制：这一档每层 K/V 只有 2 MB，落在 Infinity Cache 里，FA 的 256 次重读比 score 缓冲的 4 遍物化便宜。
   `LLAMA_FA_SPAN_SQUARE` 保留为开关，默认关（`DECISIONS.md` 第 3 节）。
5. **CONCAT 快路径**：perf logger 口径 CONCAT 合计 282.2/253.5 -> 226.9/224.3 ms（**-16%**），
   端到端约 -42 ms / 5 个 ubatch（-1.4% eval）。门禁：`test-backend-ops -o CONCAT -b Vulkan1` **198/198**、
   `ctest` **33/33**、质量门 **PASS**（`mean KLD -0.000000`、`same top p 100.000%`）。
   不是第 4 节那个 751 ms 的量级：那行是 2026-10-22、transpose 物化落地**之前**的读数。
6. **本 session 的绝对 PP 比 0.4 那批低 25-30%**（8176: 511.7 对 679.5；6144: 505 对 490）。
   同一臂的 r1/r2 差只有 1%，所以这不是噪声，是 session 级台阶（`0.4` 也记过 25% 的跨 session 漂移）。
   **禁止**拿本节数字与 0.4 混算收益；本节的所有收益都是同 session 配对值。
7. 本轮 profile 已复核，**"没有 GEMM 与 FA 行"是汇总脚本的误判**（2026-10-27）：按精确 key 分组时，
   GEMM/FA 的 key 都带形状后缀（`MUL_MAT_VEC q4_K m=2048 n=2 k=4096`、`FLASH_ATTN_EXT ...`），被"只取纯 op 名"
   的过滤整类丢掉。`bench/tools/summarize_perf.py <log> --agg` 现在按基名聚合，`concat/fast1-1.err`
   （目标工况，`-n 8`，即 prefill 口径，653 张表）的真实份额是：
   **MUL_MAT 39.1%、MUL_MAT_ID 31.7%**（合计 70.8%）、GATED_DELTA_NET 4.3%、FLASH_ATTN_EXT 3.9%、SOFT_MAX 3.0%、
   CONT 2.9%、RMS_NORM_MUL 2.8%、MUL 2.1%、CONCAT 1.2%、MUL_MAT_VEC 1.1%。
   最大的单一 key 是 `MUL_MAT_ID q4_K m=512 n=8 k=2048 n_expert=821 batch=2048` = **14.6%**（296 次，反推 3.58 TF，
   与 4/5 节独立测到的 in-situ 3.5-3.7 TF 一致）。所以 "2.6x 放大" 是 decode 端到端的系数，不是每个 kernel 的系数，
   大 kernel 的 logger 读数接近真值；FA 在本口径只占 3.9%，与判决 22 的方向一致。
8. 数字来源：`bench/raw/p0-20261027/{base,v2,pl2,default,concat,fasq,aba}/`，脚本 `bench/raw/p0-20261027/*.sh`。

### 0.6 红线 F1 的 `:1910` 降级 + id GEMM 的 tile 探针（2026-10-27）

改动：`-nkvo`（注意力在宿主）且不在整层预取路径上时，图上不再发 slot lookup（`src/llama-graph.cpp:2708`）；
`src/llama-moe-cache.cpp` 里"ids 与本 split 的 lookup 同侧"从 `GGML_ABORT` 改成走整层 ring 上传并计数。

**A. 崩溃配置现在能跑**（`-c 32768 -nkvo --moe-cache-mib 1719 -ncmoe 41`）

| 项 | 修复前 | 修复后 |
|---|---|---|
| 结果 | layer 1 `GGML_ABORT` | `exit=0`，输出连贯（`The user just said "hello"...`） |
| PP / TG | 无读数 | 24.0 / 7.4 t/s（5 token 的 prompt，只作"能跑"的证据，不作性能数） |
| 机制 | - | 日志 `prefetch needs 512 slots per group, group 0 has 472, disabled` -> `pf_batch` 恒假 -> 守卫生效，splits 287/278 |

**B. 目标工况无回归**（8176 档、256K + 10K、`-ncmoe 41 -lm none`，r=1）：PP **502.4** / TG **17.1**、
hit **58.74%**、bank **2986.17 MiB**、`split fallbacks = 0`，对 0.5 D 的 511.7 / 16.6 / 58.7% / 2989 MiB 同档同参。
带探针那轮 502.8 / 16.0，说明 `LLAMA_MOE_IDS_TRACE` 那次回读对 PP 基本无扰。

**C. 门禁**：`test-backend-ops -o MUL_MAT_ID -b CPU` / `-b Vulkan1`、`-o MUL_MAT_ID_MASK -b CPU`、
`-o MUL_MAT -b Vulkan1` 全 OK；`ctest` **33/33**。

**D. id GEMM 的 tile 探针（这次能定论）**

真实路由分布（prefill，`ids 16384`，`slots 821`）：`touched 252-256`、**`mean 64.0-65.0`**、`max 322-734`，
分桶 `<=8/<=16/<=32/<=64/<=128/>128` = `10/10/60/91/59/25`（layer 1）。
选择器实际取到：`n_as=821 -> n_sel=20 -> bn=32, max_tiles=1332`；同一次运行的非 bank 路径是
`n_as=256 -> n_sel=64 -> bn=64, max_tiles=511`。所以 `n_sel` 确实被 slot 数摊薄到真值的约 1/3.2。

但按上面的分桶估（桶内取中值）：换成 `bn=64` 时每层 tile 数 589 -> 364（A 矩阵少读约 133 x 590 KB = 0.6 ms），
代价是 tile 内填充 18,848 -> 23,296 行（多用 4,448 行 x 2x512x2048 = 9.5 GFLOP，约 1.6 ms）——**净负**。
这与第 3 节"id GEMM 的 tile 参数已证伪"一致，并给出原因：3.58 TF 对 6.04 渐近的缺口不在 tile 宽度上。
数字来源：`bench/raw/redline-20261027/{nkvo-fix,t1-clean,t2-trace}.{out,err}`。

**E. 等待原语（F12 的第一步，2026-10-27）**：Vulkan 侧新增按序号的搬运等待
（`ggml_backend_vk_prefetch_submit_seq` 返回批次序号、`_wait_seq(seq)` 只等到该序号；旧的 `prefetch_wait` 语义不变，
等于等最新）。命令缓冲从"只留最后一批"改成按序号排队回收，所以多批能同时在飞。
独立验证 `bench/tools/wait-seq.cpp`（先提 1 MiB 再提 1 GiB）：

| 量 | 读数 | 说明 |
|---|---:|---|
| `wait_seq(小)` | **2.64 ms** | 1 GiB 那批仍在飞，没有被它拖住 |
| `wait_seq(大)` | 72.39 ms | 1 GiB / 72.4 ms = **14.8 GB/s**，与第 5 节独立测的 H2D 14.2-14.3 GB/s 同口径 |
| `wait(最新)` 对照 | 76.51 ms | 等全部，正是原来唯一能做的事 |
| 数据校验 | 4/4 OK | 两批各自的载荷都对 |

门禁与回归：`MUL_MAT_ID` CPU / `MUL_MAT_ID_MASK` CPU / `MUL_MAT_ID` Vulkan1 / `MUL_MAT` Vulkan1 /
`FLASH_ATTN_EXT` Vulkan1 / `CONCAT` Vulkan1 **0 FAIL**；`ctest` **33/33**；
统一工况（8176 档、256K + 10K）PP **506.7** / TG **17.2**，预取侧 `190 layers, 88350 MiB, 185 host waits, 0 fallbacks`
与本节 B 的那轮逐字相同，记账改动没有影响现有调用方。
探针第一次跑出的 `small-data MISMATCH` 是探针自身缺陷（x/y 的暂存复用了载荷缓冲），修正后通过。

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

> 2026-10-27 口径变更：`--kv-block` 已移除，页固定 128 cells = 2.5 MiB。
> 本节 6.2-6.6 的读数都是旧块大小（2048 / 256 / 4096 / 8192）拍的，复现时不能再照抄参数。

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
   `--kv-page auto` 由 `common/fit.cpp` 的预算判决（扣掉 score 预留后仍有余量就保持连续布局），
   已于同日落地，实测判决见 6.5。
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
   **所以这只作方向性证据**；三档 x r=2 的正式矩阵在工作区覆盖修复之后重测，见 6.5（host 块臂在目标口径造不出来，
   见 6.5 第 5 条）。
   产物：`bench/raw/f11-20261023/{tgt,final2}-{off,on,nogather}.{out,err}`。

5. **质量门带 `--kv-page` 通过**（`-c 4096 --chunks 2`，同一份 1.89 GiB 基线 logits）：
   `Mean PPL(Q) 5.798313` 对基线 `5.798292`，`mean KLD -0.000000`、`max KLD 0.000064`、`same top p 100.000%`
   -> **GATE PASS**。即 paged FA + gather 与连续布局在 logits 上等价（与 B2 的验收强度相同）。
   产物：`bench/raw/f11-20261023/gate-on.{out,err}`。

门禁：`ctest` 33/33；`FLASH_ATTN_EXT -b Vulkan1` **5332/5332**；`SET_ROWS -b Vulkan1` **319/319**；
`test-kv-page`（新增 gather 用例，CPU 与 Vulkan 各跑一次）两项全 PASS：
`gather K: 0/64256 halves differ, tail non-zero = 0`、`gather V: 0/64256 halves differ`。

相关产物：`bench/raw/f11-20261023/`（ppl2-{off,on}、t5-{off,on}、final-{off,on}）。

### 6.5 F11 L1d：三档 x 三臂矩阵（2026-10-24，`cells.grow` 修复之后重测）

口径 = AGENTS 统一口径（`-c 262144 -ub 2048 -f prompt10k.txt -n 64 -ncmoe 41 -lm none`），`-K` 表示 KV 驻设备
（**不加 `-nkvo`**），r=2、GAP 30 s，三臂同一 session、电源「平衡」，桌面占用未清。产物 `bench/raw/l1d-20261024/`
（`contig3` / `paged3` / `nogather3`）。**修复前的 paged 数字作废**（见 6.6 第 1 条）。

| 档 | 臂（`-a`） | PP r1 / r2 | 均值 | TG r1 / r2 | bank | reallocs | gather MiB | host blk | status |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| 8176 | contig（`--no-kv-page`） | 507.7 / 424.1 | 465.9 | 17.6 / 17.6 | 2879 | 2 | 0 | 0 | ok |
| 8176 | paged（`--kv-page`） | 509.0 / 463.8 | **486.4** | 17.4 / 17.4 | 2847 | 1 | 32 | 0 | ok |
| 8176 | paged + `--kv-gather-mib 0` | 389.1 / 345.3 | 367.2 | 17.7 / 17.7 | 2879 | 4 | 0 | 0 | ok |
| 6144 | contig | 391.9 / 379.2 | 385.6 | 13.0 / 12.6 | 1644 | 2 | 0 | 0 | ok |
| 6144 | paged | 386.3 / 371.5 | **378.9** | 12.7 / 12.7 | 1612 | 1 | 32 | 0 | ok |
| 6144 | paged + `--kv-gather-mib 0` | 311.5 / 311.7 | 311.6 | 13.4 / 13.4 | 1644 | 4 | 0 | 0 | ok |
| 4096 | contig | 384.4 / 374.5 | 379.5 | 13.0 / 9.8 | 0 | 1 | 0 | 0 | ok |
| 4096 | paged | 377.7 / 343.5 | **360.6** | 12.8 / 13.0 | 0 | 0 | 32 | 0 | ok |
| 4096 | paged + `--kv-gather-mib 0` | 307.2 / 307.9 | 307.6 | 13.2 / 13.1 | 0 | 3 | 0 | 0 | ok |
| 8176 | auto（`--kv-page-auto auto`）-> 连续 | 508.6 / 420.5 | 464.6 | 17.8 / 15.2 | 2879 | 2 | 0 | 0 | ok |
| 6144 | auto -> 连续 | 390.0 / - | 390.0 | 13.1 / - | 1644 | 2 | 0 | 0 | ok |
| 4096 | auto -> 分页 | 368.1 / - | 368.1 | 12.2 / - | 0 | 0 | 32 | 0 | ok |
| 4096 | `-B 3072` + paged（host 块尝试） | 169.9 / 169.9 | 169.9 | 14.1 / 14.0 | 3072 | 1 | 16 | 0 | ok |

只有**同一档、相邻时间**的两臂能直接比（8176 档批内漂移 17%，见第 4 条）。

1. **gather 是分页 prefill 的必需件**：同一档内 paged 对 `--kv-gather-mib 0` 是 +32%（8176）、+22%（6144）、
   +17%（4096）。这两臂相邻、各自 r=2 稳定（6144/4096 的 nogather 两次差 < 1 t/s），是本表最可信的差值。
2. **paged + gather 与连续布局持平**：8176 +4.4%、6144 -1.7%、4096 -5.0%，TG 三档相同（差异都在噪声内，
   例：contig 4096 的 r2 TG 只有 9.8）。即 L1 之后分页不再牺牲 prefill，只剩 4G 档 -5% 的量级。
3. **host 块臂在目标口径造不出来**：`-B 3072` 在 4096 档把设备占满，KV 仍全在设备（`host blk = 0`），
   代价是 PP 从 379 掉到 169.9（bank 与 compute 抢空间）。10K cells 的 KV 只有 320 MiB，三档都放得下；
   host 块只在工作集本身超过设备（约 10 万 cells 以上）时出现，那是 `THREE-TIER.md` F11 3.1 的区间外，
   归 L2(b)，不在本轮目标口径内。
4. **批内漂移**：8176 档 contig 507.7 与 424.1 差 17%，同一二进制、同一 session、间隔 30 s；6144/4096 档
   同一臂的 r1/r2 差 <= 6%。所以 8176 档的臂间差（<10%）不可单独采信，判定用 6144/4096。
   这比 6.4 记的"跨 session 25% 漂移"更细一层：**同 session 内也会有一次性的台阶**。
5. **auto 的判决**（日志 `K/V cache: ... layout from the memory plan`）：8176 档余量 +1900 MiB、6144 档 +1644 MiB
   -> 连续；4096 档 -147.8 MiB -> 分页（重测 368.1，对连续 379.5）。默认值仍是 off：分页在 4096 档低约 3-5%，
   而那一档的连续布局在本机模拟限额下仍然跑通，所以 auto 换来的只是"块级降级"的安全性，不是速度。
6. 作废的 3 次读数：`bench/raw/l1d-20261024/contig/` 里 8176 档的 r1/r2 与 6144 档的 r1 是 `vram-oom`，
   原因是 `-lm none` 时机器只剩 9.5 GiB 空闲 RAM（`README.md` 第 4 节第 16 条），已重跑，脚本现在会先查 RAM。

### 6.6 F11 L1e：10K prompt 下分页与连续布局逐字相同（根因：`v_cells.resize` 清空 cell 状态）

**现象**（修复前）：10K prompt、`-c 16384`、`-n 16`、temp 0、`--moe-cache-mib 0`（固定 bank 以排除专家放置），
分页与连续布局在第 1 个生成 token 上不同；分页内部 gather 开与关（`--kv-gather-mib 0`，走 paged FA）
**彼此逐字相同**；两边各自 3 次重复稳定。所以差异不在 prefill 用 matmul 还是 FA，而在 cache 的数据/簿记。

**根因**：`llama_kv_cache::ensure_blocks()` 扩容后调 `v_cells[s].resize(cells)`（`src/llama-kv-cache.cpp:1565`），
而 `llama_kv_cells::resize()` 结尾调 `reset()`（`src/llama-kv-cells.h:69-76`），把**所有** cell 的 pos/ext/seq 清空。
连续布局的 `realloc_cache` 用的是 `grow()`（`src/llama-kv-cache.cpp:1924`，注释是 "append cells [size(), n)
without touching the existing ones"）。于是分页 cache 每长一块就丢掉已缓存 K/V 的 pos/seq：`find_slot` 从头复用
cell，mask 与块里的真实内容错位。这也解释了 6.5 第 3 条（旧稿）看到的现象：扩容后的 `used_max()` 读到 0。

**修复**：`ensure_blocks` 改用 `grow()`。验证（md5 是生成文本，去掉计时行；同一行内才可比）：

| 臂 | 修复前 | 修复后 |
|---|---|---|
| 连续（`--no-kv-page`） | b5b211df | 906b2115 |
| paged 默认（运行期扩容） | 8a940649 | **906b2115** |
| paged `-kvg 12288`（扩容 1 次） | e1f7c8b4 | **906b2115** |
| paged `-kvg 16384`（不扩容） | 28908487 = 连续 | - |
| paged `--kv-block 256` | - | **906b2115** |
| paged `--kv-block 4096` | - | e7e5c33f（仍不同） |
| paged `--kv-block 8192` + gather 关 | 28908487 = 连续 | **906b2115** |
| paged `--kv-block 8192` + gather 开 | - | 7dadd020（仍不同） |

（"修复前"的连续一行有两个值：早期那批没加 `--moe-cache-mib 0`，bank 不同，本来就不可比。）

**结论**：

1. 默认配置（`--kv-block 2048`）下分页与连续布局**逐字相同**，L1 的数值验收从 `n_kv <= 8192` 扩到 10K。
   修复前的 paged 数字偏高约 5%（6144 档 400.3 -> 378.9、4096 档 390.4 -> 360.6），因为被清空的 cell 让最后
   两个 ubatch 只看到约 2048 cell 的历史 —— 那批数字作废，产物留在 `bench/raw/l1d-20261024/paged2/`。
2. `--kv-block 256` 也逐字相同。
3. **`--kv-block > 2048` 仍有问题**：4096 / 8192 只在 gather 打开时与连续布局不同，关掉 gather 就相同，
   所以问题在 gather 对 BS > 2048 的假设上（未定位）。默认档不受影响，代码在 `kv_block > 2048` 时打 warning；
   记为 `DECISIONS.md` 第 6 节第 18 条。
4. 6.5 旧稿里的"工作区只长到 8192"是同一根因的症状，不是图构建的时差。`page_dense_need` 的兜底保留
   （reserve 阶段的图可能先于放置请求更大的范围），但不再依赖它。

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

### 0.7 目标口径的调度读数（`GGML_SCHED_TRACE=1`，2026-10-27）

统一工况（8176 档、256K + 10K、`-ncmoe 41 -lm none`、`--fit-plan 1` 默认），r=1，
同 session 的读数 **PP 507.6 / TG 17.1** -> 解码单步 = 1000/17.1 = **58.5 ms**。
71 行 `sched_trace`，按 `cpu` 分成 7 个 prefill ubatch 与 64 个解码步（解码步全部 `splits = 40`）。

| 字段（每解码 token） | ms | 计数 | 含义 |
|---|---:|---:|---|
| `copy` | 45.05 | - | 拷贝路径总耗时（含等待） |
| `cpy_sync` | **42.30** | - | 其中**纯等待**，= 单步的 **72%** |
| `n` / `mb` | 2.75 | 10 / 0.046 MiB | 非权重拷贝：字节可忽略 |
| `hw` | - | **38** | 宿主权重拷贝次数 = 每个被缓存层一次回调，每次约 **1.1 ms** |
| `h2d` | 2.73 | 48 | 拷贝本体 |
| `gpu_submit` | 10.42 | - | 提交开销 |

prefill 侧同口径：每 ubatch 墙钟 4.03 s（2048/507.6），`cpy_sync` 均值 **2697 ms**、`copy` 2733 ms，
即拷贝本体只有约 36 ms，其余全是等待；`splits` 41（首个 ubatch 115）。

**判读**：解码单步里 GPU 真正干活约 21.7 ms（本节 B 之前的 32K 锚点，待用新 profile 复核），
所以 `cpy_sync` 42.3 ms 中约 **20 ms 是纯串行气泡**。把它全部拿掉的终点是 58.5 - 20 = 38.5 ms =
**约 26 t/s**，与第 3 节整层数量扫实测到的 **27.1 t/s 平台**吻合——那条平台就是"没有气泡"的实测上界。

**为什么"只换等待粒度"不会赢**（第 3 节两条证伪的机制解释）：单队列上宿主在等 split i 时，
计算队列里只有 split 0..i-1，所以 event 等待与整设备 fence 等的是同一个点；
而"把 copy 回调挪到等待之前"能冲到 pp512 920 t/s 却输出错误，正说明缺的不是更便宜的等待，
而是**第二份目的缓冲**（`n_copies` 深度）。`cparams.pipeline_parallel` 恰好被 `moe_cache_size == 0` 挡死
（`src/llama-context.cpp:600-606`），且 post-split 回调只在 `!parallel` 注册（`:836-840`）——
这两处是同一条协议的两端，未结项 19 按此改靶。

### 0.8 q6_K dense 的缺口定位（形状探针 `bench/tools/gemm-shape.cpp`，2026-10-27）

同一 (m=8192, k=2048) 下扫类型与 token 数，min-of-7，不开 logger（所以是干净读数）：

| type | bpw | n=512 | n=1024 | n=2048 | n=4096 | TFLOP/s @2048 |
|---|---:|---:|---:|---:|---:|---:|
| q4_K | 4.5 | 5.39 | 6.06 | 10.75 | 20.89 | **6.40** |
| q5_K | 5.5 | 3.96 | 7.39 | 14.21 | 27.68 | **4.84** |
| q6_K | 6.5 | 4.24 | 7.84 | 14.94 | 29.19 | **4.60** |
| q8_0 | 8.0 | 3.13 | 5.82 | 10.97 | 21.16 | **6.26** |
| f16  | 16.0 | 3.51 | 6.11 | 11.37 | 21.88 | **6.04** |

判读（三条互相独立的证据指向同一个结论）：

1. **不是字节/带宽**：q8_0 的字节是 q4_K 的 1.78 倍、f16 是 3.6 倍，两者都跑在 6.0-6.3 TF，
   与 q4_K 同档；字节从 4.5 涨到 16 bpw 都不掉速，说明这一档不在带宽上。
2. **不是走错了 kernel**：`GGML_VK_MM_PIPELINE_TRACE=1` 显示 q4_K/q5_K/q6_K 分别落在
   `matmul_q4_k_q8_1_1` / `matmul_q5_k_q8_1_1` / `matmul_q6_k_q8_1_1`，`aligned=0`、激活统一量化成 q8_1，
   没有 dequant 回退，`cm1_k_tile`（`ggml-vulkan.cpp:1984`，只含 q3_K/q6_K/NVFP4）在 dense 侧没有换出不同形状。
3. **是内循环的解包算术**：掉速的恰好是 q5_K/q6_K 这类"高位平面 + 双级 scale"的类型（q6_K 每 16 个值要合并
   2-bit 符号平面再乘两级 scale），而整数 tensor core 本身不慢——同形状的 f16 matmul（完全不解包）只有 6.04 TF，
   比 q4_K 的 6.40 还低，说明限制不在乘加而在解包。

对目标口径的换算：in-situ `MUL_MAT q6_K m=8192 n=2048 k=2048` 每 eval 56 次、16.29 ms/次 = 912 ms，
占单 ubatch 墙钟 4.03 s 的 **22.6%**。若这些张量改成 q4_K（10.86 ms/次）省约 304 ms = **PP 约 +7.5%（同 session 口径）**，
顺带把 dense 侧字节减掉三成。q6_K 在本模型是**共享专家**那一族张量（`ffn_*_shexp`，日志里 210 MiB 一个）。

两条可走的路与各自的门：
- **重排量化**（把共享专家从 q6_K 换 q4_K）：不动 kernel，走 `tools/expert-repack`，
  但受未结项 7（repack 的 KLD != 0）阻塞，且共享专家每 token 都用，精度损失比稀疏专家更贵，必须过质量门。
- **减解包**（每 tile 解一次进 LDS，而不是每个 K 步解一次）：是 kernel 改写，不是调 tile；
  第 3 节那三条 q6_K/tile 实验之所以全负，现在有了机制解释——它们改的是几何，没碰解包指令数。

### 0.9 与官方 Scoop 构建（b11763）的目标口径对比（2026-10-27）

对照对象：`D:\Scoop\apps\llama.cpp-vulkan\current`（0.7.0-dev build **11763**，commit `a01ce9ffc`，Clang 20.1.8，
scoop 装的是同一份 upstream master 的快照）；本分支 `build-vulkan`（build **11915**，commit `a6cda676a` + 工作区，MinGW 16.2.0）。
「平衡」+ AC，dGPU 独占，`-lm none` 前逐臂检查空闲 RAM（实测 32.8-34.3 GiB）。两臂交替跑，每臂 r=2。

公共参数：`-m Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -fa on -t 5 -b 2048 -ub 2048 -c 262144
-f bench/prompts/prompt10k.txt -n 64 --temp 0 --no-warmup --no-display-prompt -lm none -ncmoe 41 -v`。
两边都吃掉 10438 tok（`slot print_timing`），PP/TG 取自同一行。

官方没有 `-kvg` / `--kv-page` / `-fac` / `--fit-plan`，只有 `-c` / `-nkvo` / `-ctk,-ctv` / `--moe-cache-mib`（上游已有 cache，
默认 0）可调，所以它的臂只能在这些旋钮里找。

**A. 目标工况（256K ctx + 10K prompt）**

| 臂 | 构建 | 配置 | PP t/s | TG t/s | GPU 独占峰值 MiB | RAM 峰值 MiB | status |
|---|---|---|---:|---:|---:|---:|---|
| 默认 | fork | auto bank 2989, KV grow 320 MiB | **469.9 / 509.3** | **16.6 / 17.0** | 6231 | 19482 | ok |
| 关 cache | fork | `--moe-cache-mib 0` | 384.8 / 386.0 | 12.1 / 12.3 | 3239 | 19484 | ok |
| 出厂 | 官方 | `-ncmoe 41`，`-ngl`/cache 用默认值 | 149.5 | 9.0 | 5892 | 22713 | ok（fit 砍到 30/42 层、KV 落宿主） |
| q8 KV | 官方 | `-ngl 99 -ctk q8_0 -ctv q8_0` | 149.4 / 147.4 | 14.5 / 14.3 | 5585 / 4753 | 20287 | ok |
| host KV | 官方 | `-ngl 99 -nkvo` | 145.3 / 149.3 | 6.9 / 6.9 | 3151 | 25457 | ok |
| q8 KV + 小 bank | 官方 | 同上 + `--moe-cache-mib 512` | 104.7 | 7.8 | 8143 | 19109 | ok（超卖） |
| f16 KV 在设备 + cache | 官方 | `-ngl 99 --moe-cache-mib 2986` | - | - | - | - | **ErrorDeviceLost** |
| host KV + cache | 官方 | `-ngl 99 -nkvo --moe-cache-mib 2986` | - | - | - | - | **abort `:577`** |

后两行是 `bench/raw/e2e-official-20261027/{off-1,off-2,off-nkvo256k}.bench.log`（同日 16:28-16:37），
其余是本轮的 `r2-*` / `r3-*` / `r4-*`。

**B. 交叉对照：32K ctx 与 llama-bench 短口径**

32K 档两边 KV 都能驻设备，旋钮可完全对齐；`llama-bench -p 2048 -n 32`（`n_ctx = 2080`）只用来隔离 kernel 与上传路径。

| 口径 | 构建 | 配置 | PP t/s | TG t/s | 备注 |
|---|---|---|---:|---:|---|
| 32K cli | fork | `--moe-cache-mib 0` | 386.0 / 386.6 | 12.3 / 12.5 | splits 121/82，mask 8192 |
| 32K cli | 官方 | 默认（bank 0） | 149.0 / 148.6 | 14.8 / 15.2 | splits 121/82，mask 32768，KV 640 MiB |
| 32K cli | fork | auto bank 2989 | 418.1 / 510.2 | 13.4 / 16.9 | hit 58.74% |
| 32K cli | 官方 | `--moe-cache-mib 2989` | 150.2 / 150.5 | 9.9 / 9.8 | hit 65.10%，GPU 5750 MiB |
| bench 2K | fork | `-ncmoe 41 --moe-cache-mib 0` | 432.1 ± 42 | 12.15 ± 1.47 | `pp2048` / `tg32` |
| bench 2K | 官方 | `-ncmoe 41` | 233.4 ± 46 | 16.60 ± 0.39 | 同上 |

读数：

1. **目标工况的默认对默认是 PP x3.3（470-509 对 148）、TG x1.17（16.8 对 14.4）**。
   但官方的 14.4 是 q8_0 KV 换来的；要同等 f16 KV，它只能 `-nkvo`（TG 6.9），此时 **TG x2.4**。
2. **最干净的对照是"同旋钮、都关 cache"**：fork 385.4 / 12.2 对官方 147.3 / 6.9，**PP x2.62、TG x1.77**。
   这一臂两边都是 f16 KV、bank 0、`-ncmoe 41`，差别只剩本分支的代码。
3. **PP 的差距主要不在 KV/ctx，而在宿主专家路径**：`llama-bench pp2048`（n_ctx 2080）已经是 432.1 对 233.4（x1.85）；
   32K 档同旋钮 386.3 对 149.2（x2.59）。官方的 PP 在 32K（149）与 256K（147）几乎一样，说明它本来就没被 ctx 卡住，
   而是被"41 层专家每 ubatch 从宿主上传"卡住。机制上最可能是本分支的延迟/异步拷贝路径
   （`ggml_backend_set_tensor_async_deferred` / `wait_deferred`）把这 18 GiB/ubatch 的重叠做掉了，
   但本轮没有逐 split 的 trace，这一条留作归因假设（0.7 的 decode 侧反而还留着 42.3 ms 的 `cpy_sync`）。
4. **本分支的 ctx 相关收益在容量与 TG 上**：`-kvg -1` 让 256K + 10K 的 KV 只有 320 MiB（16384 cells），
   mask 宽度 8192/16384 对 262144。官方 256K 的 f16 KV 是 5120 MiB，和 8 GiB 档不共存，只能降精度（q8_0 = 2720 MiB）
   或下宿主；TG 从 14.4 掉到 6.9 就是"注意力离开设备"的价。官方的 `-fit` 也不肯缩 ctx，日志是
   `projected to use 26413 MiB ... vs. 7312 MiB of free`、`context size set by user to 262144 -> no change`、
   `with only dense weights in device memory there is still a total deficit of 1670 MiB`，
   最后它自己把 42 层砍到 30 层、KV 落到宿主（PP 149.5 / TG 9.0）。
5. **上游的 cache 在目标工况帮不上忙，还会倒扣**：f16 KV 在设备 + 2986 MiB bank 直接 `ErrorDeviceLost`
   （5120 + 1302 dense + 1744 compute + 2986 早已超过 7378 MiB 的可用）；q8 KV + 512 MiB bank 的
   PP 104.7 / TG 7.8、GPU 峰值 8143 MiB，是小 bank 的典型抖动。32K 档同 bank 也印证方向相反：
   官方 TG 15.0 -> 9.85（-34%），fork 12.2 -> 16.8（+38%）。
6. **官方在 `-nkvo` + 非零 bank 上仍抛 `llama-moe-cache.cpp:577`**（"the experts of layer 0 are selected in
   the same split as their MoE cache lookup"），即上游还没有 0.6 那条降级。所以官方在目标工况要么没 cache，
   要么崩溃，第三条路不存在。
7. **方差不小，别跨 session 搬倍数**：fork 32K 同臂 PP 418/510（18%）、TG 13.4/16.9（26%），
   官方四次都在 2% 内。目标工况 fork PP 470/509（8%）、TG 16.6/17.0（2%）。
   本节所有倍数都来自同 session 交替臂，口径见 `README.md` 第 4 节。
8. 出处：`bench/raw/e2e-official-20261027/`（`r2-*` `r3-*` `r4-*` `r5-*` `r6-*` 的 `.out/.err`、
   `round2.csv` ... `round6.csv`，脚本 `run2.sh` ... `run6.sh`；官方失败臂 `off-1/off-2/off-nkvo256k`）。

### 0.10 统一预算式（Placement 账本，0.0.1 第 1a/1b/1c 步）（2026-10-27）

代码改动（`common/fit.h` + `common/fit.cpp`，设计来源 `THREE-TIER.md` 0.0 评审）：

1. `common/fit.h` 新增 `common_moe_ledger` / `common_moe_ledger_entry`（vram/ram/ssd 三列 + 命名），账本挂在
   `common_moe_plan::ledger` 上；`common_moe_plan_impl` 的一次测量产出全部条目，解算式只剩一个：
   `surplus = free_vram - dense(vram) - KV+RS(context, vram) - compute(vram) - margins`，
   预留项（`fa_score` / `kv_gather`）是账本条目，在同一个式子里减，不再"先扣再加也不回填"。
2. `--kv-page auto` 的判据、slack 规则、小 bank 地板、per-group 预取环判据**逐项保持原算术**（字节级不动的
   前提是不让 0.4/0.5 的读数漂移）；`--fit-plan 0` 仍走旧 margin/score/环形判据，只是也经账本产出。
3. 新增逐项打印：`ledger <name> <vram> <ram> <ssd> MiB` + `ledger room <free> / surplus <s>`（`-v` 下）。
4. 界定（v1 的取舍）：**RS 仍在"KV + RS (context)"一项里**（测量侧 context 类不拆；拆分归 F7 的
   `--rs-offload`，账本条目已留位）；逐层放置表（dense 按 tensor split、KV 按页）随 store 2a/3a 进来，v1 是分层汇总。

验收协议（比 1c 清单更严）：**同 session 新旧二进制 A/B**——`git stash` 两文件回退重编 = 旧臂，
三档 x r=2、全部 `-K`（KV 驻设备，目标工况）；规律读数 6 的教训，不拿本批和 0.4/0.5 跨 session 混算。
出处：`bench/raw/led-1c/{new,old}/`，脚本 `run-ab.sh`，比对器 `cmp-plan.sh`；预检臂 `bench/raw/ledger-1a-8176*/`。

**A. 规划器决策逐字相同**（`cmp-plan.sh` 去掉时间戳与新增 ledger 行后 diff 为空，r1/r2 同案）：

| tier | margins MiB | score 预留 | 预取环 | bank（recommend） | 结论 |
|---|---:|---:|---:|---:|---|
| 8176 | 914.1 | 256.0（注意力在设备） | 1860.0 | 2989.1 MiB | 新旧逐字相同 |
| 6144 | 760 | 256.0 | 1860.0 | 1908.4 MiB | 新旧逐字相同 |
| 4096 | 503.7 | 256.0 | 1860.0 | **0**（"no device memory for a cache"） | 新旧逐字相同 |

4096 档的 500 MiB 缺口现在**逐项可读**（满足 1c 的要求）：`free 4029.9 - dense 1302.9 - KV+RS 222.8
- compute 1372.0 - margins 503.7 - score 256.0 = surplus 372.4`，slack 规则减去 256 后剩 116.4 MiB（64 slots），
低于 145 MiB 小 bank 地板 -> 归 0。这就是未结项 15 的算术，不用再拿日志反解。

**B. 配对 PP/TG（三档 x r=2，`-K`，同 session，2026-10-27）**

| tier | 新规划器 PP/TG（r1，r2） | 旧规划器 PP/TG（r1，r2） | 判 |
|---|---|---|---|
| 8176 | 448.5/13.5，477.5/13.3 | 508.5/17.0，508.1/17.0 | PP 与 TG 均为 session 漂移同型（见读数 1） |
| 6144 | 506.4/16.1，506.7/16.1 | 505.7/16.1，507.1/16.2 | 持平（差 <= 0.3%） |
| 4096 | 383.3/12.6，383.3/12.6 | 383.8/12.7，383.4/12.6 | 持平（差 <= 0.2%） |

读数：

1. **两臂的机器行为相同**（8176 r1 对照：`88350.00 MiB uploaded / 185 host waits / 0 fallbacks`，
   `38 layers stream through 2 rings (1728.0 MiB)`，peak/used/held 三档逐字相同），规划器决策又逐字相同 ->
   运行路径没有第二个自由度。8176 的 PP/TG 差判 **session 漂移**：同一新二进制的 r1->r2 自己漂了 6.5%
   （448.5 -> 477.5），且新旧两组的绝对值分别落在已记录的台阶两端（0.5 B：PP 487.7 -> 378.8、TG 13.4-14.6；
   0.5 D：TG 16.5-16.6），两组数字各自的二三档配对差 <= 0.3%；不判回归。
2. **坦白记录：本批新臂先跑、旧臂后跑**，8176 的顺序效应方向恰好与漂移方向一致，若要绝对数字谁高谁低，
   应按 0.5 A 的交替 A/B/A 重排；本轮不另跑，理由：1a/1b 的命题是"决策等价 + 零行为差"，B 表只作旁证。
3. 新臂 4096 档 `hit` 空列 = bank 0 的预期空值，与 0.5 D 表一致；`status` 12 个 run 全 `ok`，`reallocs`
   8176/6144 = 2、4096 = 1，与 0.5 D 一致。
4. 预检臂（不带 `-K`，矩阵脚本默认 `-nkvo`）：KV 落 host 时 bank 自动给出 3553.5 MiB、PP 147.8/TG 5.9 ——
   与 `RESULTS.md` 0.2 的 `-nkvo` x2.4-x2.7 税同型，且 `:1910` 已不崩（0.6 A），属预期退化口径，不是缺陷。
   说明矩阵脚本默认臂与 0.3 起的目标工况（KV 驻设备）有一个 `-K` 之差，用矩阵读数时要看 `kv` 列。
5. 门禁：`ctest` **33/33**；新增打印不改变任何决策路径（A 部分为证）。质量门未跑：本改动不接触数值
   路径（规划器是分配决策），按 `bench/README.md` 第 4 节口径，质量门只给动注意力/精度/数值路径的改动。

留作下一步：账本已是 store 2a 的消费面（`--store-*` 接管时读 `common_moe_plan::ledger`）；
RS 拆分与逐层表在 F7/2a 落的时候补进同一账本。

### 0.11 llama-store 骨架（2a + 2b 的申报部分，2026-10-27）

代码改动（`src/llama-store*.h/cpp` 六个新文件 + `common/arg.cpp`、`common/common.h`、`common/common.cpp`，
设计来源 `THREE-TIER.md` F12/F13、0.0 评审；骨架 = 常开、无总开关、零行为改动）：

1. **三单元分文件编译**（DECISIONS 第 5 节"不得合回一个文件"）：`llama-store-slab`（纯分配器：class/slot/代数句柄、
   pin/in-flight/unconsumed、`busy()` 单原语）、`llama-store-policy`（默认策略：adopt 落 RAM；热插拔接口），
   `llama-store-io`（记账计数器 + 2c 的线程池占位，池未创建）。
2. **薄壳**：`llama_store_init/free/declare/adopt_expert/commit`；五入口中 `acquire/prefetch/wait/release`
   属 2c（有传输执行子才有语义），文件里未实现，调用面暂无使用者，这是骨架的界定不是偷工。
3. **两级注册**（F12 12）：专家按"层 + 专家子对象"注册（本模型 40 层 x 256 = 10240 个，层内数组 O(1)）；
   机器行为还照旧（专家物理字节仍在 dense buffer 里），adopt 现在只是 registry 记账。
4. **CLI**：`--store-vram/ram/ssd-mib`、`--store-threads`（env `LLAMA_ARG_STORE_*`）；vram 参数在
   init 最前置把值写进 `GGML_VK_VRAM_LIMIT_MIB`，hold back 走 F0 的既有路径（设备枚举之前生效）。
5. **消费面**：lctx 创建成功后用 `llama_get_memory_breakdown` 分机/主申报 dense 与 compute（`store_declare`），
   专家 adopt 后 `store_commit` 打印 `store_split`（vram/ram 各一行：declared + adopted + 预算内余量）。
   store 挂在 `common_init_result::impl` 最后一个成员位，析构先于 context（F13 生命周期）。

**A. 零行为差验证（8176 `-K` r=1，`bench/raw/store-2a-smoke*`）**

| 臂 | bank | PP/TG | hit % | reallocs | status |
|---|---|---|---:|---:|---|
| store 常开（smoke3） | auto (2989.1) | 481.8 / 17.0 | 58.7 | 2 | ok |
| 0.10 新规划器同臂（同日早） | auto (2989.1) | 477.5 / 13.3 | 58.7 | 2 | ok |

planner 决策与 0.10 相同（bank 2989.1、环 1860、4096 归 0 不变），`prefetch` 行为行（uploaded/waits/fallbacks）
与 hit 58.7% 复原；PP/TG 绝对值在 0.5/0.9 已记录的 session 台阶带内（TG 13-17），不据其判收益。

**B. store_split 读数（一眼账）**

```
store_split: vram accounted 2650.9 = dense 1302.9 compute 1348.0 staging 0 ... + adopted 0
store_split: vram free within budget 4662.0          # auto = plan ledger 的 free_vram
store_split: ram  accounted 18968.9 = dense 18872.8 compute 96.1 ... + adopted co-located 18600.0
store_split: ram  free within budget 12138.7
```

读数：

1. **adopted 与 declared 的双计已防住**：host 专家的物理字节今天在 dense buffer 里，骨架期的 split
   把 adopted 只计作 registry（`co-located` 标记，不进 accounted sum）；2c repack 把专家搬进 store 自有
   窗口后同一行自动并入，不需要回头改这里。
2. **compute 申报值（1348.0）对 plan ledger 的 reserve（1372.0）差 24 MiB**：前者是 context breakdown 的
   实际 sched buffer，后者是 reserve 图预留；reserve >= allocated 方向正常，记账以申报为准。
3. **`--store-vram-mib 4096` 的 hold back 链验证**（`bench/raw/store-2a-holdback/`，不加 `-T` 档位）：
   `VRAM limited to 4096.0 MiB, 3282.8 MiB of 7378.8 MiB held back`，随后规划器 ledger 与 4096 档臂
   **逐字相同**（room 4029.9 / surplus 372.4），bank 归 0，CSV `held 3282.8`、PP 383.4 / TG 12.6、
   `status = ok`。参数旋钮 = 硬顶 + visible tier 的语义在 CLI 上成立；旧 env 仍是迁移期机制
   （`run-matrix.sh -T` 不动）。
4. **两个骨架期缺陷在本轮抓出并修掉**（都从读数流程里抓的）：adopt 初版每层 class 只造 16 槽，
   10496 个对象里 10224 个被 "left behind"（加 `grow_class` 分段扩容，容量不足仍不 abort）；
   adopted 曾被计入 accounted sum，复现出 ram "free -6318.9" 的负数（改 co-located 计法）。
   两个修复都有单测分支。
5. 门禁：`ctest` **34/34**（新增 `test-store-slab`：busy 三守卫位、槽复用 + 代数、stale 句柄不可 evict、
   分层记账、满类拒分配后 grow 成功）；Vulkan 算子门禁未触发（不碰 kernel/scheduler）。
6. 出处：`bench/raw/store-2a-{smoke,smoke2,smoke3,holdback}/`。
7. 界定（与 2b/2c 的边界）：staging 申报、5% 越权对账（需 memory-logger 对账脚本）、五入口的传输语义
   都在 2c 收口；`--store-ram-mib` 的 auto 分母（free - 4 GiB 系统留量常数）目前只体现在 split 行，
   默认值按 M1 读数复核后再定。

### 0.12 传输执行子路由（2c 第一刀，`--store-exec` 0/1/2，2026-10-27）

代码改动（`src/llama-store*` + `src/llama-moe-cache.cpp` + arg/common 参数，2 值语义见表）：

1. store-io 新增绑定面：`llama_store_bind_backend`（进程内一次解析五个 proc 地址，
   含 0.6 E 的 `submit_seq/wait_seq`），MoE cache 的 `pf.tensor/submit/wait` 三个调用点改为
   `pf_tensor_route/pf_submit_route/pf_wait_route` 三条路由函数——**策略（SLRU/整层驻留/提前一层/双环）不动**。
2. `--store-exec 0/1/2`（env `LLAMA_ARG_STORE_EXEC`，默认 **0**）：0 = 旧路径；1 = 提交/等待**经 store-io 路由**、
   原语不变；2 = 层内等待只等本批序号（`pf_seq_last`）。promotion/回填路径（`:1867`、`:2061`、`:2141`）
   **保持 raw drain**：那些等待必须清空整条传输队列才能发布 slot map，混进序号语义会破坏正确性。
3. 老的六个 `pf*` 计数留在 cache；store-io 侧新增 `store_exec_submissions/fallbacks`（本轮非零才打印）。

**A. A/B/A 读数（8176 `-K`，同 session 交替两轮，`bench/raw/exec-2c/m{0,1,2}-p{1,2}`）**

| mode | pass1 PP/TG | pass2 PP/TG | hit % | prefetch 行为行 |
|---|---|---|---:|---|
| 0（旧路径） | 507.9 / 17.0 | 460.7 / 14.3 | 58.7 | 190 层 / 88350 MiB / 185 waits / 0 fallbacks |
| 1（路由经 store） | 507.6 / 17.0 | 508.6 / 16.9 | 58.7 | 同上，逐字相同 |
| 2（序号等待） | 508.3 / 16.9 | 507.2 / 16.9 | 58.7 | 同上，逐字相同 |

读数：

1. **mode 1/2 是"路由零行为差"的验收臂**：决策（bank auto 2989.1）、hit、内存（峰值 6231、held 0）、
   prefetch 行为行全部逐字相同；pass1 中三臂 PP 差 < 0.2%，TG 差 < 0.6%。pass2 里 mode 0 单臂
   460.7/14.3 的下沉是 session 台阶（0.10 读数 1 的同型，同批路由臂无下沉），按位移规则不判差异。
2. **mode 2 的语义界定**：双环模式下同一时刻在飞的只有一层批次，"等本批"与"等最新"当前**数学等价**；
   它的价值是序号管线（0.6 E）接进了主路径、store-io 计数器开始真实记账。真正的等待时机收益
   （等待时间后移、多批次同时在飞）属 **2g/M2 的 use_step**，本轮不评读数。
3. **输出等价**：temp 0 短跑（32 token）mode 0 与 mode 2 生成文本逐字一致（diff 只剩时序行）；
   prefill 的 ring 路径覆盖在 A 表的 10K prompt 里。
4. 门禁：`ctest` **34/34**；无 kernel/scheduler 改动，Vulkan 算子门禁不新增。
5. 出处：`bench/raw/exec-2c/{m0,m1,m2}-{p1,p2}/` 与 `/tmp/out-m{0,2}.txt`短跑。

剩余（2c 收口，不是本轮）：M2 的 use_step 驱动预取（等待时机收益 + `store_prefetch_wasted` 验收）；
promotion/回填路径入 store 计数（需要批次标签区分目的）；staging 申报与 5% 越权对账；
`--store-exec` 默认值由 M2 读数决定（本轮保持 0）。
