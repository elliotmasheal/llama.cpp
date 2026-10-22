# LFM2.5-24B-A2B (UD-Q4_K_XL) on 8 GiB Vulkan - tuning sweep plan

> Status: research output, **no benchmark was run** for this document (read-only source audit + GGUF header
> parsing + the existing evidence in `bench/moe-baseline/`). Every number below is either (a) read from the
> model file, (b) derived by arithmetic from source, or (c) explicitly marked 预测/待实测.
>
> Repo: `E:\fastLlama` @ `a9d2cc8a7`. Canonical binary: `E:\fastLlama\build-vulkan\bin\llama-bench.exe`
> (must be the freshly rebuilt one; record `llama-bench --version` in the results header).
> Do **not** use `E:\fastLlama\.worktrees\moe-hybrid\build\bin`.
>
> Machine: RX 6700S 8 GiB (`Vulkan1`, 8176 MiB total / 7378 MiB free at start, gfx1032, RDNA2 28 CU,
> no coopmat, int dot) + iGPU `Vulkan0` (UMA) + Ryzen 7 6800HS 8C/16T + 39.2 GiB RAM, Windows.
> Model: `F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`, 14,876,901,632 B = 13.855 GiB, 428 tensors.

---

## 0. 摘要（先读这一节）

1. **模型画像（实测自 GGUF header，`bench/lfm25/gguf-dump.txt`）**：40 blocks，block 0/1 是 dense（ffn 11776），
   block 2..39 是 MoE（38 层，64 专家 / 用 4，expert_ff 1536）。注意力层只有 10 层（block 2,6,...,38，
   32 heads / 8 KV heads / head_dim 64），其余 28 层是 shortconv（l_cache 3），**但它们同样有 MoE FFN**。
   专家权重 `ffn_*_exps` 合计 **13371 MiB = 13.06 GiB（占全部权重的 94.3%）**，非专家权重只有 **808.8 MiB**。
   专家张量按量化分成 **2 个 layout group**：36 层各 348.0 MiB（Q4_K gate/up + Q5_K down），
   block 7/8 各 421.5 MiB（Q5_K gate/up + Q6_K down）。
2. **`-ncmoe N` 的 N 是 block 号，不是 MoE 层号**（`common/common.h:1247-1262`）：
   对 block 0..N-1 加 override，所以 **实际进 CPU 的 MoE 层数 = max(0, N-2)**（前 2 个 block 是 dense，
   没有 `_exps` 张量）。因此 `-ncmoe 40` = 全部 38 个 MoE 层进 CPU；`-ncmoe 0/1/2` 等价（专家全在 GPU，
   这张卡装不下 13.06 GiB，会失败或严重溢出）。
3. **这张卡的显存预算**：非专家 809 MiB + Vulkan compute/reserve/staging（Qwen 实测同栈 578 MiB，
   本模型估计 500-650）→ 留给专家权重/缓存的约 **6.6-6.7 GiB**。换算成驻留层数 ≈ **18-19 层**
   （每层 348 MiB）。所以 static 最优区间是 **`-ncmoe` 21-26**，而不是 30-38：
   `-ncmoe 30` 只剩 10 层驻留，等于把 28 层 MoE 丢给 CPU。
4. **MoE cache 的上限约 6400-6656 MiB**（Qwen 是 6144，因为其非专家权重 1303 MiB 比本模型多 494 MiB）。
   按源码算术（表见 §4）：`--moe-cache-mib 6144` → 36 层组 1057 slots（5 整层 + 737 LRU，覆盖 45.9%）、
   2 层组 57 slots（覆盖 44.5%），总量 6135 MiB。6400 → 1101/60 slots。
5. **cache 的 decode 快路径只在 ubatch ≤ 32 token 时启用**（`src/llama-moe-cache.cpp:206,551-573`）。
   llama-bench 的 TG 是逐 token decode（`tools/llama-bench/llama-bench.cpp:2229-2239`，每次 1 token），
   所以 TG 一定走 slot/LRU 路径；而 `-p 512/2048` 的 prefill 永远不走 slot 路径，只享受
   "整层驻留（≤5 层/group）" 与 "LRU 池预填充切片（默认开）" 两处利息。
6. **由此产生的结构性预期（待实测）**：PP 看"host→device 拷贝层数"（static 越多驻留越好），
   TG 看"多少 MoE 层能跑在 GPU 上"（cache 全 GPU 更好）。**两个指标很可能各有自己的最优配置**，
   和 Qwen 的结论形态一致（static 抢 PP、cache 抢 TG，`AGENTS.md` 的两行基线）。
7. **TG 的上限受 H2D 带宽约束**：cache 模式每 token 需要上传 `38×4×(1-hit)×5.44 MiB` 的未命中专家
   （hit=0.55 时约 371 MiB/token），本机 H2D 实测上限 13.3-13.9 GB/s（`L8`），
   即 ~27-37 ms/token 的硬顶（≈27-37 t/s），再叠加 L9 的 "每层 ids 回读 + 全设备 drain" 串行点。
   所以 **对 TG 的中性预期是 15-30 t/s 量级，而不是 100+**；具体值只能实测。
8. **省时间的结构性建议**：同一 `-ncmoe` 下把 `--moe-cache-mib` 写成逗号列表（一次 process load 模型、
   多次建 ctx，只多付 0.5-1.5 s/配置的 bank 预填充），但 `-ncmoe`、`-t`、`-ngl` 变化会触发整模型重载
   （`llama-bench.cpp:1310-1317, 2384-2395`），所以按 `-ncmoe` 分组跑。
9. **`--moe-cache-bank` 在本树里不存在**（全仓 grep 无此 flag；`bank` 是 `llama_moe_cache` 内部对
   "整槽设备缓冲区" 的称呼，见 `src/llama-moe-cache.cpp:178,407`）。监控 bank 用量用
   `GGML_VK_MEMORY_LOGGER=1`，看 `MoE cache size = ... MiB` 日志行。
10. **`-fa on` 与 conv 层正确性无关**：shortconv 层根本不构建 attention（`src/models/lfm2.cpp:311-312`，
    `hparams.is_recr(il)` 分支），只有那 10 个 attention 层用 FA。但 `-fa auto` 会在 `sched_reserve()`
    里被静默关掉（`src/llama-context.cpp:691-730`），所以批量对比一律显式 `-fa on`。

---

## 1. 模型事实（来自 GGUF header，非估算）

由 `bench/lfm25/gguf_inspect.py` 解析（只读 header，不读张量数据），完整清单见 `bench/lfm25/gguf-dump.txt`。

| 项 | 值 |
|---|---|
| arch | `lfm2moe`（`llama-model.cpp:309` -> `llama_model_lfm2moe`） |
| type | `LLM_TYPE_24B_A2B`（`src/models/lfm2moe.cpp:16-20`，block_count==40 时命中） |
| block | 40；`leading_dense_block_count=2`（block 0,1 dense，ffn 11776 / Q8_0） |
| MoE 层 | block 2..39 = **38 层**，`expert_count=64`，`expert_used_count=4`，`expert_ff=1536` |
| 注意力层 | block 2,6,10,...,38 = **10 层**（`head_count=32`，`head_count_kv=8`，head_dim=64） |
| shortconv 层 | 其余 **28 层**（`l_cache=3`），**FFN 同样是 MoE**（`src/models/lfm2.cpp:305-325`） |
| embd / vocab | 2048 / 128000；`token_embd` Q8_0 265.625 MiB；`output` 与 token_embd 复用（`TENSOR_DUPLICATED`） |
| 专家权重 | `ffn_*_exps` 合计 **13371.0 MiB (13.06 GiB)**，占 13.855 GiB 的 94.3% |
| 非专家权重 | **808.8 MiB**（含 token_embd 265.6、shortconv 270.7、attn 106.6、dense ffn 146.6、gate_inp 19、norm 等） |
| 单层专家字节 | 36 层：gate Q4_K 108 + up Q4_K 108 + down Q5_K 132 = **348.0 MiB**；block 7,8：132+132+157.5 = **421.5 MiB** |
| KV cache | 10 层 × 8 KV head × 64 × K/V × 2 B = 20 KiB/token → ctx 2048 ≈ 41 MiB，ctx 131072 ≈ 2.6 GiB |

**代码侧要点**：lfm2moe 的专家是**三个独立张量** `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`
（`src/models/lfm2moe.cpp:44-48`），没有 fused `gate_up`。所以每层 3 次 `MUL_MAT_ID`，
而 MoE cache 的 slot_map 由三者共享（`build_moe_cache_slots(selected_experts, up, gate, down, gate_up, ...)`，
`src/llama-graph.cpp:2325`），一个 slot 里存 3 个专家切片 → **每 slot 字节 = 5.4375 MiB（A 组）/ 6.585 MiB（B 组）**。

---

## 2. 本 fork 的 flag 语义（逐条源码验证）

除非特别说明，`llama-bench` 与 `llama-cli` 共用 `common/arg.cpp` 的同一套 `common_params`。
`llama-bench` 自己解析一份子集（`tools/llama-bench/llama-bench.cpp:275-1130`），
默认值见 `llama-bench.cpp:382-427`。

### 2.1 `-ngl / --n-gpu-layers N`

- `common/arg.cpp:2809-2826`。N 是**"从末尾往前数、放到 VRAM 的层数"**，不是"前 N 层"。
- `-1`（默认）在**本 fork 里等于"全部"**：`src/llama-model.cpp:2435-2437`
  `return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;`
  对 40 block 模型 = 41（40 block + output 层）。所以 llama-bench 不写 `-ngl` 时就已经是"全上卡"。
- 分配单位是"整个 layer 的所有张量"（`src/llama-model.cpp:1686-1705` 的 `get_layer_buft_list`），
  但可以被 `-ot`/`-ncmoe` 在**张量粒度**上覆盖（见 2.4）。这就是 "layer 在 GPU、专家在 CPU" 的混合态来源。
- 显式写 `-ngl 99` 与默认等价；计划里统一写死，避免不同 build 默认语义变化。

### 2.2 `-ncmoe / --n-cpu-moe N`（**最关键的一个**）

- 实现：`common/arg.cpp:2770-2779` → `llm_add_n_cpu_ffn_overrides(N, LLM_FFN_EXPS_REGEX, overrides)`。
- 展开：`common/common.h:1255-1262` 对 `i = 0 .. N-1` 生成
  `string_format("blk\\.%d%s", i, LLM_FFN_EXPS_REGEX)`，其中
  `LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"`（`common/common.h:1243`）。
  即 **`blk.0.ffn_*_exps` ... `blk.(N-1).ffn_*_exps` 覆盖为 CPU buffer**。
- **语义 = "前 N 个 block 的专家张量"。** 对 lfm2moe，block 0/1 是 dense、没有 `_exps` 张量，
  所以 **实际进 CPU 的 MoE 层数 = max(0, N-2)**；`-ncmoe 40` = 38 层全进 CPU，`-ncmoe 42+` 无额外效果，
  `-ncmoe 0/1/2` 完全等价（= 专家全在 GPU）。
- 与 `-ngl` 的关系：**正交**。override 在 buffer 选择里先于 layer 的 buft list 生效
  （`src/llama-model-loader.cpp:1236-1268`），所以 `-ngl 99 -ncmoe 22` = 所有非专家张量在 GPU、
  block 0..21 的专家在 CPU、block 22..39 的专家在 GPU。若 `-ngl` 太小导致某 block 整体在 CPU，
  该 block 的专家自然也就在 CPU（override 与 layer 归属一致，无冲突）。
- 与 `-ot` 的关系：**同一条 override 列表，第一个匹配生效（first match wins）**
  （`llama-model-loader.cpp:1238-1259`，匹配到就 `break`）。关键是**顺序**：
  - `llama-cli/llama-server`：按命令行顺序 push，所以 `-ncmoe 20 -cmoe` 是"具体块先生效"，而
    `-cmoe -ncmoe 20` 会让 `-cmoe` 的宽 pattern 先匹配全部 → `-ncmoe` 无效。
  - `llama-bench`：把用户 `-ot` 放在前面、`-ncmoe` 生成的 pattern 追加在后
    （`tools/llama-bench/llama-bench.cpp:1279-1305`），所以**显式 `-ot` 优先于 `-ncmoe`**。
- 副作用（实测相关）：override 到 CPU 且 `use_mmap` 时会打印一次
  `"tensor overrides to CPU are used with mmap enabled - consider using --load-mode none"`
  （`llama-model-loader.cpp:1244-1249`）。Windows 默认 AUTO->resident 时不会触发（见 2.11）。
- 常见误用：把 N 当成 MoE 层号。对 24B-A2B，`-ncmoe 30` 只有 28 层进 CPU、10 层驻留，
  远没到显存上限，属于"白白多走 10 层 CPU MoE"。**正确区间见 §4.2。**

### 2.3 `-cmoe / --cpu-moe`

- `common/arg.cpp:2763-2769`：push 一条**不带 block 前缀**的 override `\.ffn_(up|down|gate|gate_up)_(ch|)exps`
  → 所有 MoE 专家的全部张量进 CPU，等价于 `-ncmoe 40`（对本模型）。与 `-ncmoe` 同列表，顺序敏感。

### 2.4 `-ot / --override-tensor "<pattern>=<buft>,..."`

- 解析：`common/arg.cpp:252-284`（`parse_tensor_buffer_overrides`）。每个 override 用 `=` 分隔
  pattern 与 buffer type；多个 override 用 `,` 分隔。
- pattern 是 **C++ `std::regex`**，匹配方式是 **`std::regex_search`（部分匹配，不是全匹配）**
  （`llama-model-loader.cpp:1239-1240`）。因此 `blk\.2\.ffn_gate_exps` 能匹配
  `blk.2.ffn_gate_exps.weight`；不需要写 `.*` 或 `$`。`.` 在正则里是任意字符，所以不转义也能工作
  （但 `blk.2` 也会匹配 `blk.2` 之外的东西——如 `blk.12` 不会误匹配，因为 pattern 里有 `\.2\.` 的分隔）。
- buffer type 名来自 `ggml_backend_buft_name`，只枚举**设备 buffer type**（`arg.cpp:255-262`），
  本机可用值为 **`CPU` / `Vulkan0` / `Vulkan1`**（Vulkan 设备 buft 名 = `GGML_VK_NAME + idx`
  = `"Vulkan"+i`，见 `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4808,5515,16389`）。
  `Vulkan1_Host` 这类 host buft **不在**枚举表里（`ggml-vulkan.cpp:13285` 只用于 host buffer），
  所以 `-ot ...=Vulkan1_Host` 会报 "unknown buffer type" 并打印可用列表。
- 顺序 = 优先级（first match wins）。**没有 "pattern 没匹配到任何张量" 的告警**——拼错 pattern 是静默无效，
  这是"配置看起来生效了但其实没生效"的第一大来源（见 §6）。
- 列出全部张量名可用 `llama-gguf`（本次未运行）；本模型名单见 `bench/lfm25/gguf-dump.txt`。

### 2.5 `--moe-cache-mib N`（专家 GPU 缓存，本 fork 核心特性）

声明：`common/arg.cpp:2780-2789`（写入 `cparams.moe_cache_size`，默认 0 = 关闭）。
实现：`src/llama-moe-cache.{h,cpp}`，构造在 `src/llama-context.cpp:623-629`。
**llama-bench 支持它**（`llama-bench.cpp:477, 741-747, 1332`）；`bench/moe-baseline/BASELINE.md:161`
说"llama-bench 不支持"是**旧 build 的记载，对当前 HEAD 不成立**。

语义拆解（全部来自源码）：

1. **只有"全部专家都在 host 内存"的层才进缓存**：构造函数里
   `if (experts.empty() || !all_of(experts, is_host_weight)) continue;`
   （`src/llama-moe-cache.cpp:274-278`），`is_host_weight` 要求 buffer usage==WEIGHTS 且 buffer 是 host
   （`:151-155`）。=> **cache 必须配合 `-ncmoe` / `-cmoe` 使用**；不用它们时日志会警告
   `"no layer has all of its experts in host memory, MoE cache is disabled"`（`:297-300`）并静默关闭。
2. **按 (device, layout) 分组**（`same_layout` 比较类型、形状、`nb[2]`，`:139-148`）。
   本模型 = **2 组**：A = 36 层（348 MiB/层），B = block 7,8（421.5 MiB/层）。
   `-ncmoe` 只覆盖前缀，所以 A/B 的实际层数取决于 N（例如 `-ncmoe 8` 时 block 7 不在其中，B 组只剩 1 层）。
3. **预算切分**（`:312-348`）：设备按 `tensor_split` 分配，若 `-ts` 全 0 则按"启动时设备空闲显存"比例分配；
   同一设备内按 `g.host_bytes / d.host_bytes` 分给各 group（即按专家字节比例，保证各组覆盖率相同）。
4. **slot 数**（`:349-352`）：`while (n_slots < layers*64 && alloc_size(n_slots+1) <= budget) n_slots++;`
   其中 `alloc_size(g,n) = sum(GGML_PAD(nb[2]*(n+1), alignment))`（`:303-310`）——
   **`+1` 是给越界读留的 padding 槽**，alignment = Vulkan 的 `minStorageBufferOffsetAlignment`
   （`ggml-vulkan.cpp:13260`），对 slot 数影响 ≤1。
5. **n_slots < n_expert_used(4) → 该组放弃**并告警 `"MoE cache budget is too small ... not cached"`
   （`:353-357`）；所有组都为 0 时构造函数直接 throw
   `"MoE cache is too small to hold the experts of one token"`（`:373-375`），
   benchmark 表现为 `failed to create context`。
6. **整层驻留（whole layer） vs LRU 池**（`:358-369`）：
   `n_full = min(5, n_slots/64, layers-1)`，并对每组最多 5 层把 64 个专家全部放进 bank
   （读表视图 `cached_static` 直接替代 host 张量，`:424-429` + `src/llama-graph.cpp:1663-1668`）；
   剩余 slots 组成该 group 其余层的 LRU 池（`slot_base = n_full*64`）。
   另有约束 "LRU 池不得少于总 slots 的 1/5"（`:363-365`），否则回退 n_full。
7. **初始化预填充**（`:629-710`）：整层 64 专家一次性上传；LRU 池再按每层等分的确定性切片预填充
   （`GGML_MOE_CACHE_PREFILL=0` 可关）。所以每个 cache 配置的 ctx 创建要额外上传 ~bank 大小
   （6144 MiB → ~0.45 s @13.9 GB/s）。
8. **哪些路径用到缓存**（关键）：
   - **ubatch ≤ 32 token 时走 slot_map / LRU（小批快路径）**：`get_slot_map()` 只有在
     `1 <= n_tokens <= max_batch(32)` 且 `min(n_tokens*n_expert_used, 64) <= lru_slots` 时返回非空
     （`:206, 551-573`）。llama-bench 的 TG 是 1 token/eval → **TG 恒走此路径**。
     slot_map 是 host 权重，调度器在此处强制开新 split，并在拷贝回调里读回专家 id、跑 LRU、
     上传未命中专家、更新 slot_map（`src/llama-graph.cpp:2601-2619`，`src/llama-moe-cache.cpp:828-959`）。
   - **整层驻留层在所有路径都直接读 bank**（包括 prefill），不产生 host 拷贝（`:424-429`,
     `llama-graph.cpp:1663-1672`）。
   - **ubatch > 32（含所有 `-p` ≥ 32 的 prefill）不走 slot 路径**；此时专家权重仍会被调度器整块拷到
     设备临时 buffer，但拷贝改由 `sched_copy_experts` 处理：**bank 里已有的专家用 D2D 拷贝代替 host 上传**
     （`src/llama-context.cpp:2854-2952`，`llama-moe-cache.cpp:792-826` `copy_experts()`）。
     因为整层层已被静态视图替换，这里实际只对"LRU 池里的预填充切片"生效。
     => **prefill 的收益只来自整层层 + 预填充切片，而且不减少"每层一次 host copy"的调用次数。**
   - **`--moe-hybrid`**（`common/arg.cpp:2790-2796`）：小批（≤32）未命中改由 CPU 计算
     （`get_miss_mask`，`:590-599`；graph 侧 `llama-graph.cpp:2311-2325`）。源码自己警告
     "experimental and about 2x slower"（`src/llama-context.cpp:617-628`），**不作为候选配置**。
9. **硬约束/崩溃点**：
   - `split_mode == TENSOR` 直接 throw（`:263-265`）——只能用 `-sm layer/none`。
   - 与 `pipeline_parallel` 互斥（`src/llama-context.cpp:583-589` 把 `moe_cache_size != 0` 当作禁用条件）。
   - 专家选择数与 slots 不匹配时 `GGML_ABORT("the MoE cache is too small for the experts selected in layer %d")`
     （`llama-moe-cache.cpp:915-917`）。本模型 4 专家/token，32 token 最多 64 个不同专家，
     只要该层 LRU slots ≥ 64 就不会触发（本机各候选配置的 LRU slots 都 ≥ 47）。
   - id 回读依赖"routing 已完成"，否则 `GGML_ABORT("...selected in the same split as their MoE cache lookup")`
     （`:866-871`）。
10. **可验证日志（需要 `-v`，因为 llama-bench 默认把 INFO 日志关掉**，`llama-bench.cpp:2304-2306`）：
    - `llama_moe_cache: N of M MoE layers have a bank: X whole layers, Y layers share the LRU pool`
    - `llama_moe_cache: group 0: 36 layers, 5 whole layers, 737 LRU slots for 31 layers`
    - `llama_moe_cache: 36 layers, Q5_K: 1057 slots (45.9%)`
    - `llama_moe_cache: Vulkan1 MoE cache size = 6134.90 MiB for 13371.00 MiB of host experts`
    - 退出时（ctx 释放）：`llama_moe_cache: ubatch <= 8: hits = ..., hit rate = ...%`
      （`:961-985`，hit rate 是判断 TG 上限的关键实测值）

### 2.6 `-fa / --flash-attn on|off|auto`

- `common/arg.cpp:1758-1772`：`on` → `LLAMA_FLASH_ATTN_TYPE_ENABLED`；`-fa on,off` 可一次跑两档
  （llama-bench 用 `string_split` 解析成列表，`llama-bench.cpp:875-899`）。
- `auto`（默认）会在 `sched_reserve()` 里被"解析"：如果 FA 融合节点被调度到了与该层不同的设备，
  就**静默关闭 FA** 并 WARN（`src/llama-context.cpp:691-730`，`cparams.auto_fa` 分支）。
  这就是 `AGENTS.md` 要求显式 `-fa on` 的原因。
- **对 conv/shortconv 层无正确性影响**：`src/models/lfm2.cpp:311-312` 按 `hparams.is_recr(il)` 二选一，
  recr 层走 `build_shortconv_block`（SSM_CONV 等），**根本不构建 attention**，`flash_attn_ext`
  只在 `build_attn` 里出现（`src/llama-graph.cpp:2971-2994`）。`-fa off` 仍然正确，只是 10 个
  attention 层走非 FA 路径（mask 变 F32，长 ctx 更费显存）。
- FA 是**量化 KV 的前提**：`-ctk/-ctv` 用非 F16 时必须 FA，否则报错/自动打开
  （`src/llama-context.cpp:633-637, 4172-4196`）。
- 本模型 head_dim=64、GQA 8 KV head，Vulkan FA 支持；`-fa off` 与 `-fa on` 的差异预计只在
  10/40 层，量级较小，仍建议跑一对做 A/B（R6）。

### 2.7 `-b / --batch-size` 与 `-ub / --ubatch-size`

- `common/arg.cpp:1671-1684`。`-b` = 逻辑 batch（llama-bench 每次 `llama_decode` 的 token 数上限），
  `-ub` = 物理 ubatch（一次图执行处理的 token 数）。
- llama-bench 默认 `-b 2048`、`-ub 512`（`llama-bench.cpp:392-393`）；`cparams.n_ctx = n_prompt+n_gen+n_depth`
  （`:1322`），所以 `-p 512 -n 32` 的 ctx 只有 544。
- `test_prompt` 按 `n_batch` 切块喂（`:2194-2211`），**ubatch 由 llama_context 再切**；所以 `-ub` 大于
  当前 batch 没有意义（`-p 512` 时 ub 2048 与 512 等价），`-p 2048 -ub 2048` 才是单个 2048 的 ubatch。
- 与 MoE cache 的耦合（重要）：**`-ub <= 32` 才会启用 slot/LRU 快路径**（§2.5 第 8 条）。
  `-ub 2048` 的 prefill 只能吃到整层驻留 + 预填充切片的利息。
- **`-ub` 是否必须配合 `-b` 提高**：Qwen 的实测结论（`PP-UB-BOUNDS.md:13-14`）是 `-ub 4096` 必须同时
  `-b 4096`，否则被逻辑 batch 卡住；本模型同样适用（`-p 2048` 时 batch=2048，ub 不能超过它）。

### 2.8 `-t / --threads` 与 `-tb / --threads-batch`

- `common/arg.cpp:1519-1538`。**llama-bench 没有 `-tb`**（`llama-bench.cpp` 里不存在该参数），
  且 `test_prompt/test_gen` 都调用 `llama_set_n_threads(ctx, n_threads, n_threads)`
  （`llama-bench.cpp:2188, 2218`）→ **bench 里 batch 与 gen 用同一个 `-t`**。
- llama-bench 默认 `-t = common_cpu_get_num_math()`（`:396`），在本机是"物理核数"量级（8）。
  `-t` 只影响 CPU 后端节点：static 配置（CPU MoE）敏感，cache 配置基本不敏感（几乎所有节点在 GPU）。
  历史基线统一用 `-t 5`（`AGENTS.md`、`L1-L9`），本计划保持一致以便横向对比。

### 2.9 `-nkvo / --no-kv-offload`

- `common/arg.cpp:2417-2424` → `cparams.offload_kqv = !no_kv_offload`。
- 关掉后 KV 张量放 CPU，attention 相关节点可能落到 CPU。本模型只有 10 层有 KV、ctx 又很小
  （`-p 512 -n 32` 时 544 tok → 10.6 MiB），**没有任何理由用它**；且源码注释指出
  `model.dev_layer()` 在 `--no-kv-offload` 下并不精确（`src/llama-context.cpp:702-703`），
  会干扰 FA 的 auto 解析与 cache 的设备归属判断。计划里保持默认（KV 在 GPU）。

### 2.10 `-ctk / -ctv`

- `common/arg.cpp:2440-2465`（默认 F16/F16）。本模型 KV 极小，`q8_0` 只能省 ~20-40 MiB，
  但要求 FA 打开。**不作为本轮扫描项**（Qwen 上已实测在噪声内，`PROGRESS.md` B 表）。

### 2.11 `-lm / --load-mode` 与 Windows resident/mmap

- `common/arg.cpp:2694-2712`：`auto` / `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`。
- `AUTO` 在 Windows 上的实际语义（`src/llama-model.cpp:1590-1616`）：如果
  `free_ram > n_bytes*1.25 + 1 GiB`，就**放弃 mmap、整模型常驻 RAM**，日志
  `"model fits in RAM (x of y GiB available), loading it resident; set LLAMA_AUTO_RESIDENT=0 to keep mmap"`。
  环境变量 `LLAMA_AUTO_RESIDENT=0` 可强制回 mmap。
- 本模型 14.88 GiB → 需要 ~19.6 GiB 空闲 RAM 才 resident。本机 39.2 GiB，通常满足。
  **必须保持 resident**：host 专家（13.06 GiB）就在这块内存里，mmap 时 Windows 会回收 mapped page，
  导致运行中随机读 SSD（Qwen 上 pp2048 因此从 144 崩到 45.5 t/s，`BASELINE.md:215-228`）。
- 若想省 RAM 或想排除 mmap 抖动，可用 `-lm none`（强制读入 RAM）；`-lm mmap+mlock` 是另一个等价选项。

### 2.12 `-dev / --device`、`-sm / --split-mode`、`-ts / --tensor-split`

- `-dev`：`common/arg.cpp:2741-2748`，逗号分隔的设备名列表，名字与 `--list-devices` 一致。
  Vulkan 设备名是 **`Vulkan<i>`**，`i` 是 **`vkEnumeratePhysicalDevices` 过滤后（仅 discrete+integrated）的序号**
  （`ggml-vulkan.cpp:5346-5400` 枚举，`:4808/5515/16389` 命名）。本机 **Vulkan0 = 核显，Vulkan1 = 6700S**
  （与 `BASELINE.md:45` 一致）。可用 `GGML_VK_VISIBLE_DEVICES` 进一步屏蔽设备
  （`ggml-vulkan.cpp:5330-5345`）。
  **不写 `-dev` = 用所有设备**，配合默认 `-sm layer` 会变成跨 iGPU/dGPU 流水 —— 这是最容易悄悄毁掉
  基准的一步，所有命令一律写 `-dev Vulkan1`（iGPU 专项除外）。
- `-sm`：`common/arg.cpp:2827-2850`，默认 `layer`。`tensor` 与 MoE cache 互斥（throw）。
- `-ts`：`common/arg.cpp:2851-2877`，各设备的层/预算比例，也是 MoE cache 预算的切分依据（§2.5 第 3 条）。
  单卡时保持不写（全 0 → 按空闲显存比例，等价于独享）。

### 2.13 `-fit / --fit` 与 `-fitt/-fitc`（容易踩的坑）

- `common_params.fit_params` 默认 **true**（`common/common.h:483`），llama-cli/llama-server 会
  "自动调整未显式设置的参数以装进显存"（`common/common.cpp:1244-1273`）。
  **这会改 `-ngl`/`-ncmoe`/ctx 等**，所以用 llama-cli 复现时务必显式给全参数，或加 `-fit off`。
- llama-bench **默认不做 fit**：只有显式给 `-fitt <MiB>` 或 `-fitc <n>` 才走 `common_fit_params`
  （`llama-bench.cpp:2348-2381`）。**本计划不使用 `-fitt`**（我们需要手工控制放置）。
- 已有证据表明 auto-fit 在本机会退化（`L8` §2/§4：按"空闲-权重-compute-256 MiB"缩小 bank，pp512 会掉 ~12 t/s）。

### 2.14 `--moe-cache-bank`

- **本树不存在此 flag**（`common/arg.cpp`、`llama-bench.cpp` 全仓 grep 无 `bank` 选项）。
  "bank" 是 `llama_moe_cache` 内部对"所有 slot 的设备存储"的命名（`llama-moe-cache.cpp:178,405-410`），
  每组每个专家张量一个 bank，`ne[2] = n_slots + 1`。
  观察手段：`GGML_VK_MEMORY_LOGGER=1`（每次分配打印 `Total device:`）+ 初始化日志
  `"MoE cache size = ... MiB for ... MiB of host experts"`。

### 2.15 llama-bench 自身行为（决定扫描怎么排）

来源：`tools/llama-bench/llama-bench.cpp`。

| 行为 | 出处 | 对计划的影响 |
|---|---|---|
| 参数是**列表**，会做笛卡尔积成多个 "instance" | `:1338-1500` | `-ncmoe 20,22,24` / `--moe-cache-mib 4096,6144` / `-fa on,off` / `-t 4,5` 一次跑完 |
| 仅当 **mparams** 变化才重载模型（`equal_mparams`：model/ngl/ncmoe/sm/ts/lm/dev/ot/...） | `:1310-1317, 2384-2395` | `-ncmoe` 变 → 重载；`--moe-cache-mib`/`-fa`/`-t`/`-ub`/`-p`/`-n` 变 → **不重载** |
| 每个 instance 都新建 ctx | `:2397` | cache 配置每个都要重新分配 bank + 预填充（~0.5-1.5 s） |
| TG 测量 = 逐 token `llama_decode`（每步 1 token） | `:2229-2239` | TG 恒走 MoE cache 小批路径 |
| PP 测量按 `-b` 切块 | `:2194-2211` | `-p 512` 实际一个 512-token batch |
| 每个 instance 先 warmup（除非 `--no-warmup`） | `:2434-2461` | 不要用 `--no-warmup`，否则首 rep 含 shader/时钟爬升 |
| 每个 rep 前清 KV（`llama_memory_clear`），**不清 MoE cache LRU** | `:2463-2464` | 第 1 个 rep 的 LRU 只被 warmup 那一 token 预热过；`-r >= 3` 才稳 |
| `-p 0` / `-n 0` 分别跳过 pp / tg 测试 | `:1372, 1411, 2505-2530` | 可以只测一侧 |
| 默认关闭 INFO 日志（除非 `-v`） | `:2304-2306` | **看 cache / offload 日志必须加 `-v`** |
| `-o md` 是默认输出格式 | `:425` | 直接贴进报告；需要机读用 `-o csv` |

---

## 3. 从 `bench/moe-baseline/` 继承的结论（可迁移到本模型的部分）

| 已测结论（Qwen3.5-35B-A3B，20.21 GiB） | 出处 | 对 LFM2.5-24B-A2B 的迁移判断 |
|---|---|---|
| H2D 实测上限 **13.3-13.9 GB/s**（PCIe Gen4 x16），与 chunk 大小无关 | `L8` §1 | 直接适用。cache 模式 TG 的未命中上传是主要成本，越大 bank 越好但受显存限 |
| Vulkan 上 **`--moe-cache-mib 6144` 是硬上限**；6400/6528 崩到 57 t/s | `L8` §2, `L6` §3 | 上限由 "总显存 - 非专家权重 - compute - staging" 决定。本模型非专家权重少 494 MiB → **上限预计 6400-6656**（需实测） |
| bank 尺寸单调收益：4096→351、5120→368、5632→373、6144→382 t/s（pp512） | `L8` §2 | 本模型预计同样单调到溢出点为止 |
| cache 的 **whole-layer 数上限 = 5/group**；LRU 池需 ≥1/5 slots | `L1` §1, `llama-moe-cache.cpp:358-365` | 直接适用（本模型 group A 36 层、group B 2 层；B 组 slot 不足 64 → n_full=0） |
| 无 cache 时 prefill 主要由 **每层 host-weight copy + 其前的全设备 drain** 支配（cpy_sync 1.28 s / 1.36 s） | `L8` §3, `L9` §2, `L5` §3 | 直接适用 → **prefill 的杠杆是"减少 host copy 的层数"（=增加驻留层），而不是减少字节**。整层驻留能完全去掉该层的 copy；LRU 只减字节不减 copy 次数 |
| cache 的 prefill 收益有限：byte 从 8187→7298 MiB（-10.8%）只换来 pp512 +3.5% | `L6` §2, §3 | 直接适用，解释同上一行（Qwen 只移除 10/41 层的 copy） |
| `pp2048 -ub 512` 会因 whole-layer 驻留+copy buffer 复用模式变化而退化（163.9 → 59 t/s） | `L1` §2, §3 | **必须同时测 `-ub 512/1024/2048`**，不能只看默认 ub |
| `-ub` 不影响 decode；`-ub 2048` 对大 prefill 有决定性影响 | `L1` §3, `PP-UB-BOUNDS.md` §1 | 直接适用；`-p 2048` 必须配 `-ub 2048` 比较 |
| decode 的硬串行点 = **每层一次 ids 回读 + `ggml_backend_synchronize`**（30 次/eval），任何流水都失效 | `L9` §2-3 | 直接适用，本模型 38 层 → **38 次/eval**，是 TG 的物理下限之一 |
| `--moe-hybrid` 约 2x 慢，源码自带警告 | `src/llama-context.cpp:617-628` | 不纳入候选 |
| 双 GPU（iGPU UMA）在 35B 上更差（OOM + SSD 换页） | `PROGRESS.md` B 表 | 35B 放不下 24 GiB UMA；**本模型专家 13.06 GiB 或许能全放进 iGPU**，值得 1-2 次低成本试验（R8） |
| `-ncmoe 27` 是 Qwen 的最优（14 层驻留 / 41 层 = 每层 466 MiB） | `BASELINE.md` 5.1 | 本模型每层 348 MiB、非专家 809 MiB → **最优驻留 18-19 层 → `-ncmoe` 21-26** |
| ubatch 512 时专家路由只覆盖 **70%** 的 expert（Qwen 256 选 8，路由偏斜） | `L5` §1 | 本模型 64 选 4、更稀疏，覆盖率需实测（影响 prefill 上传字节） |
| `-t 5` 是历史基线统一值 | `AGENTS.md`, `L1-L9` | 保持 `-t 5`，另跑 R5 线程扫描 |
| static 与 cache 的生成文本会不同（量化 kernel 数值差异，非 bug） | `L1` §4 | 不要跨配置做逐字节比对；比对只在"同放置、只换 cache 源"时成立 |
| Windows 必须 resident，否则 page fault 到 SSD | `BASELINE.md` §5.7, `src/llama-model.cpp:1590-1616` | 直接适用，见 §6 |

---

## 4. 关键算术（给扫描定区间）

### 4.1 显存预算

```
8176 MiB  Vulkan 设备总量（--list-devices / 日志 "Device memory: 8176 MB (7378 MB free)"）
 -809 MiB  非专家权重（本模型实测 808.8）
 -578 MiB  Vulkan compute buffer + reserve + staging（Qwen 同栈实测 505+63+10；本模型 vocab 更小，估 500-650）
 -120 MiB  安全余量（Windows 桌面/合成器抖动；Qwen 上 148 MiB 余量时已在临界）
= 6669 MiB 可用于"专家驻留 / MoE cache bank"（估计值，误差 ±150 MiB）
```

- static 检查：驻留层只能取 block 后缀（`-ncmoe N` 让 0..N-1 进 CPU）。每层 348 MiB（除 block 7,8 的
  421.5），故 **19 层 = 6612 MiB（临界）**、18 层 = 6264、16 层 = 5568。
  => **`-ncmoe` 20 会溢出（20 层 = 6960），21 是临界，22-24 安全，26 保守。**
- cache 检查：bank 上限 ≈ 6669 MiB → 见下表（对齐取 256 B，实际 ±1 slot）。

### 4.2 `--moe-cache-mib` → slot 表（由 `bench/lfm25/cache_slots.py` 按源码公式算出）

前提：`-ncmoe 40`（38 个 MoE 层全在 CPU），单设备 `Vulkan1`，无 `-ts`。
A 组 = 36 层（Q5_K down），B 组 = block 7,8（Q6_K down）。总 host 专家 13371 MiB。

| cache MiB | A slots | A whole | A LRU | A 覆盖 | B slots | B whole | B 覆盖 | 总覆盖 | bank 实际 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0% | 0% | 0 |
| 2048 | 351 | 4 | 95 | 15.2% | 18 | 0 | 14.1% | 15.2% | 2039 |
| 3072 | 528 | 5 | 208 | 22.9% | 28 | 0 | 21.9% | 22.9% | 3067 |
| 4096 | 704 | 5 | 384 | 30.6% | 38 | 0 | 29.7% | 30.5% | 4090 |
| 4608 | 793 | 5 | 473 | 34.4% | 43 | 0 | 33.6% | 34.4% | 4607 |
| 5120 | 881 | 5 | 561 | 38.2% | 48 | 0 | 37.5% | 38.2% | 5119 |
| 5632 | 969 | 5 | 649 | 42.1% | 52 | 0 | 40.6% | 42.0% | 5623 |
| **6144** | **1057** | **5** | **737** | **45.9%** | **57** | **0** | **44.5%** | **45.8%** | **6135** |
| 6400 | 1101 | 5 | 781 | 47.8% | 60 | 0 | 46.9% | 47.7% | 6394 |
| 6656 | 1145 | 5 | 825 | 49.7% | 62 | 0 | 48.4% | 49.6% | 6646 |
| 6912 | 1190 | 5 | 870 | 51.6% | 65 | 0 | 50.8% | 51.6% | 6911 |
| 7168 | 1234 | 5 | 914 | 53.6% | 67 | 0 | 52.3% | 53.5% | 7163 |

用法：
- 上表就是**验收基准**。跑 `-v` 时日志若与表不符（例如 slots 明显偏小），说明预算被别的设备分走
  （`-ts` 非零或存在第二个 GPU）或非专家权重比预估大。
- 每 slot 5.4375 MiB（A）/ 6.585 MiB（B）→ 想再加 1 层整层驻留 = 64 slots = 348 MiB。
- 覆盖 46%（6144）时，均匀路由下命中率 ≈ 46%；带时间局部性会更高，**实际命中率看退出日志**。

### 4.3 预期区间（**预测，必须实测**）

- static 最优：`-ncmoe 21-26`（18-19 层驻留），TG 随 N 减小而升，PP 也随 N 减小而升，**在溢出点断崖**。
- cache 最优：`--moe-cache-mib 6144-6656`（上限边界），`-ncmoe 40`。
- `-p 2048 -n 0`：static 与 cache 的差距预计比 pp512 更小（prefill 由 host copy 次数支配：
  static 有 N-2 层要拷贝，cache 有 38 层里除去 5 个整层 = 33 层要拷贝）→ **PP2048 预计 static 明显赢**。
- TG：预计 cache 赢（全 38 层 MoE 在 GPU），量级 **15-30 t/s**（H2D 未命中字节 + 每层 ids drain 限制），
  而 static 受 CPU MoE 层数限制（`-ncmoe 22` 时 20 层 CPU；每 token 约 20×4×5.44 = 435 MiB 经 RAM）。
- 这些只是量级判断，**不要写进结论**，以实测为准（§7）。

---

## 5. 扫描计划（按信息价值/时间排序）

### 0）前置检查（不做任何 GPU 工作）

- 确认 `E:\fastLlama\build-vulkan\bin\llama-bench.exe` 是 Lead 当前 build（记录
  `llama-bench.exe --version` 输出，含 build 号与 commit）。
- `llama-bench.exe --list-devices`：确认 `Vulkan0`=核显、`Vulkan1`=6700S，并且只有这两个 GPU。
- 统一变量（下称 `$B`、`$M`）：

```powershell
$B = 'E:\fastLlama\build-vulkan\bin\llama-bench.exe'
$M = 'F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf'
$COMMON = @('-m', $M, '-dev', 'Vulkan1', '-ngl', '99', '-fa', 'on', '-t', '5')
```

- 时间成本估算（每个 instance）：模型 resident 加载 5-8 s（page cache 热）/ 15-20 s（冷）；
  cache ctx 额外 0.5-1.5 s；pp512 每 rep = 512/PP 秒；tg32 每 rep = 32/TG 秒。
  所以 **每配置 r=3 约 10-25 s（含加载）**；`-ncmoe` 变化时每次另加 6-9 s 重载。
- R1/R2 之后命令里的占位符：**`<N_static>`** = R1 选出的最优 `-ncmoe`，**`<C>`** = R2 选出的最优
  `--moe-cache-mib`（例如 `22` 与 `6144`）。R1/R2 自身用字面量，不需要替换。

### R0：冒烟 + 加载可行性（r=1，约 2 分钟）

```powershell
& $B @COMMON -ncmoe 30 -p 512 -n 32 -r 1 -o md    # 有 cache 以外的常规放置
& $B @COMMON -ncmoe 40 -p 512 -n 32 -r 1 -o md    # 地板：专家全在 CPU
& $B @COMMON -ncmoe 40 -p 512 -n 32 -r 1 -v -o md # 看日志：offloaded 41/41 layers、cache 关闭
```

判定：能正常出数即通过。同时记录 wall time 反推加载时间。
（`-ncmoe 0/1/2` 必然装不下 13.06 GiB 专家权重，见 §6.1；**不要**在这轮跑，避免长时间溢出。

### R1：static `-ncmoe` 驻留扫描（r=3，一条命令，约 3 分钟）

```powershell
& $B @COMMON -ncmoe 20,21,22,24,26,28,30,40 -p 512 -n 32 -r 3 -o md
```

- 8 个 instance，每次重载模型（`-ncmoe` 变）→ ~8×(7 s + 3×6 s) ≈ 3 min。
- 预期：PP 随 N 增大而下降（20 可能是溢出点，会看到异常低值）；TG 在 N=21/22 最高，之后随 CPU 层数增加下降。
- 决策：取 **N_static = TG 最高且未溢出**的 N（并列取小 N，PP 更好）；若 N=20/21 出现断崖
  （TG 掉到 1/5 或 pp 掉到 1/5），记录溢出边界，退回 22/24。

### R2：MoE cache 尺寸扫描（r=3，一条命令，约 3 分钟）

```powershell
& $B @COMMON -ncmoe 40 --moe-cache-mib 0,2048,4096,5120,5632,6144,6400,6656 -p 512 -n 32 -r 3 -o md
```

- 模型只加载一次（mparams 未变）；8 个 ctx，每个建 bank + 预填充 0-1 s。
- `--moe-cache-mib 0` 就是 R1 的 `-ncmoe 40`，作为同 session A/B 锚点。
- 预期：TG 单调升到 ~6144-6656；若 6656 已溢出，会看到 TG 掉到个位数（Qwen 在 6400 就崩）。
- 决策：取 **cache_MiB = TG 最高且未溢出**；记录断崖位置（下一步的余量判据）。
- 若 log 里 `MoE cache size` 明显小于请求值（例如请求 6656 却只拿到 6135），说明分配被别处吃掉。

### R3：头部对头 A/B（r=5 交叉，约 3 分钟）

四组，**交叉两轮**（先 A1 A2 B1 B2，隔 1 分钟再 A1 A2）以抵消时钟漂移：

```powershell
# TG/短 PP（canonical）
& $B @COMMON -ncmoe <N_static>              -p 512 -n 32 -r 5 -o md
& $B @COMMON -ncmoe 40 --moe-cache-mib <C>  -p 512 -n 32 -r 5 -o md

# 长 prefill（ub 2048）
& $B @COMMON -ncmoe <N_static>              -p 2048 -n 0 -ub 2048 -r 3 -o md
& $B @COMMON -ncmoe 40 --moe-cache-mib <C>  -p 2048 -n 0 -ub 2048 -r 3 -o md
```

- r=5 的 32-token TG 在 20 t/s 下约 1.6 s/rep，代价可接受。
- 输出：**两条候选命令**（很可能就是最终结论）：PP 最优 vs TG 最优。

### R4：`-ub` / `-b` 边界（r=2，约 4 分钟）

```powershell
# static 与 cache 各三档 ub
& $B @COMMON -ncmoe <N_static>             -p 2048 -n 0 -ub 512,1024,2048 -r 2 -o md
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 2048 -n 0 -ub 512,1024,2048 -r 2 -o md
# 更大 ub（需要 ctx 4096；本模型 KV 小，可能装得下）
& $B @COMMON -ncmoe <N_static>             -p 4096 -n 0 -b 4096 -ub 4096 -r 2 -o md
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 4096 -n 0 -b 4096 -ub 4096 -r 2 -o md
```

- 依据 `L1`/`PP-UB-BOUNDS`：`-ub 512` 在长 prefill 上可能严重退化（Qwen 163.9 vs 576），所以必须扫。
- 4096 一档若溢出（PP 掉到 ~1/3）说明 compute buffer 顶到显存，记录边界后回到 2048。
- `-p 512` 下 ub 512/2048 等价，不用扫。

### R5：线程（r=3，约 3 分钟）

```powershell
& $B -m $M -dev Vulkan1 -ngl 99 -fa on -ncmoe <N_static>             -p 512 -n 32 -t 4,5,6,8 -r 3 -o md
& $B -m $M -dev Vulkan1 -ngl 99 -fa on -ncmoe 40 --moe-cache-mib <C> -p 512 -n 32 -t 4,5,8   -r 3 -o md
```

- `-t` 不在 `equal_mparams` 里 → 模型不重载。
- 预期：static 对 `-t` 敏感（CPU MoE），cache 基本不敏感（可用来验证"cache 模式确实没有 CPU MoE"）。
- 决策：static 取最优 t；同时保持 `-t 5` 作为跨报告可比值。

### R6：`-fa` A/B（r=3，约 1.5 分钟）

```powershell
& $B -m $M -dev Vulkan1 -ngl 99 -ncmoe <N_static>             -p 512 -n 32 -t 5 -fa on,off -r 3 -o md
& $B -m $M -dev Vulkan1 -ngl 99 -ncmoe 40 --moe-cache-mib <C> -p 512 -n 32 -t 5 -fa on,off -r 3 -o md
```

- `-fa on,off` 是列表（`llama-bench.cpp:875-899`），一次跑两档；`-fa` 不进 `equal_mparams`，不重载。
- 预期差异小（只有 10/40 层用 FA）。若 `off` 不慢反快，说明这 10 层 attention 在小 ctx 下不是瓶颈，
  记录下来（与 Qwen 的 `-fa on` 结论并不矛盾：那边 attention 层占比高得多）。

### R7：混合/边界（可选，r=2，约 2 分钟）

```powershell
# 2 层驻留（block 38/39）+ 剩余 36 层走 cache
& $B @COMMON -ncmoe 38 --moe-cache-mib 5632 -p 512 -n 32 -r 2 -o md
# deferred upload（只影响小批未命中路径）
$env:LLAMA_MOE_DEFER_UPLOAD='1'
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 512 -n 32 -r 2 -o md
Remove-Item Env:LLAMA_MOE_DEFER_UPLOAD
# hybrid 反例（预计 ~2x 慢，仅用于钉死"不要用"）
& $B @COMMON -ncmoe 40 --moe-cache-mib 4096 --moe-hybrid -p 512 -n 32 -r 1 -o md
```

- R7-1 的依据：纯 cache 用满显存，留 2 层驻留会挤掉 bank，预计负收益；跑一次确认即可（`L6` §3 有同类负例）。
- R7-2 的 env 名见 `src/llama-moe-cache.cpp:920-923`。
- **注意 `--moe-hybrid` 需要显式 `-v` 才能看到它的命中率行**；且它是"实验性 2x 慢"，不当候选。

### R8：iGPU（`Vulkan0`）——只在 R0-R3 完成后按需执行（约 4-6 分钟）

本模型专家 13.06 GiB + 非专家 0.82 GiB = 13.9 GiB，理论上能塞进 24 GiB UMA 共享内存，
所以"全权重常驻 iGPU、零 H2D、零 CPU MoE"第一次变得**可能**（35B 时代不可能，故 `PROGRESS.md` 已排除）。
风险：Windows/AMD 驱动给 iGPU 的 device-local heap 往往远小于 24 GiB，或 Vulkan 报告的总量是
共享堆；不保证能分配。先做 a)，只有 a) 能跑且数字可观时才做 b)。

```powershell
# a) iGPU 单卡，全部放进去（先 r=1 试探，10 秒内没开始出数就认为在溢出、Ctrl-C）
& $B -m $M -dev Vulkan0 -ngl 99 -fa on -ncmoe 0 -p 512 -n 32 -r 1 -o md
# b) iGPU 单卡 + 每 token 少读专家（如果 a) 溢出）
& $B -m $M -dev Vulkan0 -ngl 99 -fa on -ncmoe 20,40 -p 512 -n 32 -r 2 -o md
# c) 双设备分层（只有 a) 的 TG 达到 dGPU 最优的一半以上才值得做）
& $B -m $M -dev Vulkan1,Vulkan0 -sm layer -ts 0.6,0.4 -ngl 99 -fa on -ncmoe 0 -p 512 -n 32 -r 2 -o md
# d) 双设备 + cache（预算按 -ts 切：0.6 给 dGPU、0.4 给 iGPU 共享内存）
& $B -m $M -dev Vulkan1,Vulkan0 -sm layer -ts 0.6,0.4 -ngl 99 -fa on -ncmoe 40 --moe-cache-mib 8192 -p 512 -n 32 -r 2 -o md
```

- 判定 c)/d)：只有在 a) 的 TG 很接近（≥50%）dGPU 最优，或 c)/d) 明显超过 R3 最优时才继续深挖。
- 注意：`-dev Vulkan1,Vulkan0` 时 `pipeline_parallel` 会在"无 override 且无 cache"时自动开启
  （`src/llama-context.cpp:583-589`），c) 与 d) 的执行模式不同，不能互相当基线。
- d) 的 `--moe-cache-mib 8192` 是"总预算"，按 `-ts` 比例切给两个设备的 bank；iGPU 侧能否分配是未知数。

### R9：诊断（r=1，不计入性能对比，约 3 分钟）

```powershell
# 1) 显存峰值 / bank 实际大小
$env:GGML_VK_MEMORY_LOGGER='1'
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 512 -n 0 -r 1 -v -o md
# 2) prefill 的 host vs bank 字节（判断整层/预填充切片是否真的生效）
$env:GGML_MOE_COPY_TRACE='1'
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 512 -n 0 -r 1 -o md
# 3) split 数与每段耗时（判断 CPU/GPU 往返）
$env:GGML_SCHED_TRACE='1'
& $B @COMMON -ncmoe <N_static> -p 512 -n 32 -r 1 -o md
# 4) 命中率（退出日志）
& $B @COMMON -ncmoe 40 --moe-cache-mib <C> -p 512 -n 32 -r 1 -v -o md
```

验收点（对照 §4.2 表）：

- `MoE cache size = 6134.90 MiB for 13371.00 MiB of host experts`（6144 配置）
- `36 layers, Q5_K: 1057 slots (45.9%)` / `2 layers, Q6_K: 57 slots (44.5%)`
- `group 0: 36 layers, 5 whole layers, 737 LRU slots for 31 layers`
- `Total device:` 峰值应 < 8176 MiB（接近时就要警惕溢出）
- 退出时 `ubatch <= 8: hit rate = xx%`：这是 TG 上限的直接解释量

### 5.1 总成本

| Round | instance 数 | r | 估计 GPU 时间 |
|---|---:|---:|---:|
| R0 | 3 | 1 | ~1.5-2 min |
| R1 | 8 | 3 | ~3 min |
| R2 | 8 | 3 | ~3 min |
| R3 | 8 (4 命令 x2) | 3-5 | ~4 min |
| R4 | 10 | 2 | ~4-5 min |
| R5 | 7 | 3 | ~2.5 min |
| R6 | 4 | 3 | ~2 min |
| R7（可选） | 3 | 1-2 | ~2 min |
| R8（条件） | 4 | 1-2 | ~5 min |
| R9（诊断） | 4 | 1 | ~3 min |
| **合计** | | | **~30-35 min**（不含 Lead 的交叉/复测开销） |

优先级：**R0 → R1 → R2 → R3 → R9 → R4 → R5 → R6 →（R7）→（R8）**。
前 5 轮（约 13 分钟）就能拿到"两条候选命令 + 证据"。

### 5.2 回退计划（cache 装不下 / cache 不划算时）

1. **纯 static**：`-ncmoe <N_static>`（R1 的结果），不碰 `--moe-cache-mib`。
   这是零风险配置：显存只放 18-19 层专家 + 非专家权重。
2. **保守 static**：`-ncmoe 26`（16 层驻留，余量 ~1.2 GiB），适合长 ctx（ctx 32K 时 KV 才 ~650 MiB，
   不像 Qwen 那么吃紧，但仍要留 compute buffer 余量）。
3. **cache 缩小保命**：`--moe-cache-mib 4096-5120`（覆盖 30-38%，TG 收益比 6144 小但不会溢出）。
4. **混合**：`-ncmoe 38 + --moe-cache-mib 5632`（R7-1），若测量显示"少 2 层 CPU MoE"的收益
   大于 bank 缩小 512 MiB 的损失，则它可能是 TG 的次优拐点。
5. **PP 专用**：若 cache 配置的 PP 明显低于 static，就明确写成两条命令（PP 用 static、TG 用 cache），
   和 `AGENTS.md` 对 Qwen 的记录一致。

### 5.3 长上下文提示（本轮不做，但要写进结论边界）

`-p 2048` 的 ctx 只有 2048，KV 才 41 MiB；真实 32K/128K ctx 下：
KV = 20 KiB/token → 32K ≈ 655 MiB、128K ≈ 2.6 GiB（10 个 KV 层，F16）。
这会从同一个 6669 MiB 预算里抢走 0.65-2.6 GiB，**cache 上限要相应下调**
（32K：~6000；128K：~4000）。若 Lead 后续要长上下文基线，需要单独一轮（不在本计划内）。

---

## 6. 失败模式与陷阱（按"最容易浪费一轮 benchmark"排序）

### 6.1 OOM / 显存溢出的 4 种形态

1. **权重放不下**（例如 `-ncmoe 0/1/2`，13.06 GiB 专家要进 8 GiB）：模型加载阶段
   `ggml_backend_alloc_ctx_tensors` 失败，日志 `failed to allocate ... buffer of size N`
   （`ggml/src/ggml-backend.cpp:102`），`llama_model_load_from_file` 返回 NULL，
   llama-bench 打印 `error: failed to load model` 并 exit 1。**快**，不浪费时间。
2. **MoE cache bank 放不下**：`ggml_backend_vk_buffer_type_alloc_buffer` 失败返回 nullptr
   （`ggml-vulkan.cpp:13242-13255`）→ `llama_moe_cache` 构造函数 throw
   `"failed to allocate the MoE cache buffers"`（`llama-moe-cache.cpp:492-500`）→
   llama-bench `error: failed to create context` 并 exit 1。**快**。
   （另一种：bank 太小而不是"放不下"，见 §2.5 第 5 条。）
3. **compute buffer 放不下**：模型和 bank 都成功，但第一次 decode 时
   `failed to allocate buffer of size N`（`ggml-backend.cpp:2754`）→
   llama-bench `failed to run prompt/gen` 并 exit 1（`llama-bench.cpp:2512-2529`）。
4. **静默溢出到共享内存（最危险）**：Windows/AMD 驱动允许 device-local 分配落到系统内存，
   **不报错、只是慢 5-10 倍**。Qwen 上 `--moe-cache-mib 6400` → pp512 从 383 掉到 57 t/s
   （`L8` §2）。识别：吞吐出现 5x 断崖 + `GGML_VK_MEMORY_LOGGER` 的 `Total device` 逼近/超过 8176 MiB。
   **只要看到断崖就当溢出处理，不要当作"调参噪声"。**
5. **宿主 RAM 压力**（不是 VRAM）：resident 模式下 RSS ≈ 14.9 GiB（含 host 专家），39.2 GiB 够用；
   但若 `LLAMA_AUTO_RESIDENT=0`（mmap）且 Windows 回收 page cache，专家读盘会让 PP/TG 断崖
   （`BASELINE.md` §5.7：p2048 从 144 → 45.5）。**看 `LLAMA_LOG` 里的 load mode 行 + 任务管理器磁盘读。**

### 6.2 设备枚举顺序

- `Vulkan0` = 核显、`Vulkan1` = 6700S（本机实测，与 `BASELINE.md:45` 一致）；
  顺序来自 `vkEnumeratePhysicalDevices` 过滤 discrete+integrated 后的序号
  （`ggml-vulkan.cpp:5346-5400`），**不是按性能排序**。
- **不写 `-dev` = 所有设备** → 默认 `-sm layer` 会跨 iGPU+dGPU 分层，数字完全不可比。
  所有命令都写 `-dev Vulkan1`；iGPU 专项写 `-dev Vulkan0`；双卡写 `-dev Vulkan1,Vulkan0`。
- 想彻底排除干扰可 `GGML_VK_VISIBLE_DEVICES=1`（只留 6700S，`ggml-vulkan.cpp:5330-5345`）。
- 每个进程启动都要重建 Vulkan pipeline（无盘级 VkPipelineCache，`BASELINE.md:46`）→
  进程数越多固定开销越多（这也是"`-ncmoe` 不变时把 cache 值写成列表"的价值）。

### 6.3 `-fa on` 是否必需（conv 层正确性）

- **不用于 conv 层正确性**：28 个 shortconv 层根本不走 attention（`src/models/lfm2.cpp:311-312`）。
- 必需性只出现在：（a）量化 KV（`-ctk/-ctv` 非 F16），`llama-context.cpp:633-637,4172-4196`；
  （b）`-sm tensor`（本计划不用）。
- **`-fa auto` 会被静默改写**：`llama-context.cpp:691-730` 在 `sched_reserve()` 里检查 FA 融合节点
  是否落在该层设备上，不是就关闭并只打一条 WARN。benchmark 用显式 `-fa on`，并用 R6 验证。

### 6.4 如何识别"配置静默退化/没生效"

| 现象 | 检测手段 | 对应源码 |
|---|---|---|
| cache 被静默关闭（没有 CPU 专家层） | `-v` 看 `"no layer has all of its experts in host memory, MoE cache is disabled"` | `llama-moe-cache.cpp:297-300` |
| cache 预算太小、某组被放弃 | `-v` 看 `"MoE cache budget is too small for N layers, they are not cached"` | `:353-357` |
| cache 实际大小与请求不符 | `-v` 看 `"MoE cache size = ... MiB"`，对照 §4.2 表 | `:528-536` |
| `-ot` pattern 拼错（静默无效） | 无告警；改用 `-ncmoe`，或用 `-v` 看 `tensor ... buffer type overridden to ...`（DEBUG 级，需 `-v`） | `llama-model-loader.cpp:1236-1261` |
| `-fa auto` 被关掉 | `-v` 看 `"flash_attn = ..."` 与 `"... not supported, set to disabled"` | `llama-context.cpp:467,718-724` |
| 层没按预期上卡（`-ngl` 太小/被 fit 改） | `-v` 看 `"offloaded N/M layers to GPU"`；确认没给 `-fitt`；llama-cli 确认 `-fit off` | `llama-model.cpp:2020`，`llama-bench.cpp:2348-2381` |
| KV 被放到 CPU（`-nkvo`） | `-v` 看 offload_kqv / sched 分配；本计划不使用 | `arg.cpp:2417-2424` |
| 走成了跨设备流水 | 日志 `"pipeline parallelism enabled"`；或检查是否忘了 `-dev` | `llama-context.cpp:613-615` |
| 溢出到共享内存 | `GGML_VK_MEMORY_LOGGER=1` 的 `Total device` 峰值 + 吞吐断崖 | `ggml-vulkan-debug.cpp:26` |

### 6.5 计时卫生

- **串行跑**（Lead 已在做）：任何并发负载都会污染 8 GiB 卡的占用率与 H2D 带宽。
- 每个 instance 的 warmup 会消掉 shader 编译/时钟爬升，但**模型加载后的第一个测量仍有 LRU 冷启动**；
  `-r >= 3`，最终结论用 `-r 5` + 交叉重复。
- 移动 GPU 会热降频：R3 的交叉 A/B 与"隔 1 分钟复测"是必要的。
- 不要用 `--no-warmup`（除了显式的诊断 run）。
- 不要在测量窗口里跑磁盘密集任务（模型 14.88 GiB 在 F:，resident 加载会抢 I/O）。
- 一次进程内多个 instance 时，前面 instance 释放的显存可能被驱动延迟回收 → 溢出边界附近建议
  把可疑配置单独起进程复测一次。

---

## 7. 只有实测能回答的问题（不要猜）

1. **static 与 cache 谁赢、赢多少**：本模型 4/64 专家（比 Qwen 的 8/256 稀疏得多）会不会让 LRU 命中率
   明显高于覆盖率？只有 `ubatch <= 8: hit rate` 日志能回答。
2. **cache 的显存上限到底是多少**（预测 6400-6656，取决于驱动/桌面占用的 ~120-150 MiB 余量）：
   需要 R2 的断崖位置 + `Total device` 峰值。
3. **TG 到底被什么卡住**：H2D 未命中字节 vs 每层 ids 回读 drain（38 次/eval）vs GPU kernel 串行。
   需要 R9 的 `GGML_SCHED_TRACE` + `GGML_MOE_COPY_TRACE` + hit rate 三者一起归因。
4. **prefill 的杠杆是"host copy 层数"还是"字节"**：本模型的 cache 只整层驻留 5/38 层，
   但 LRU 池可覆盖 46% 的字节 —— 两种机制都变，必须靠 `GGML_MOE_COPY_TRACE` 的 host/bank 字节
   + 实测 PP 判断（Qwen 的经验是"层数"为主，但那是 256 专家的情形）。
5. **`-ub` 的最优值与 `-ub 512` 是否如 Qwen 一样崩**：本模型 KV 极小、MUL_MAT_ID 形状不同
   （每专家 token 数 = ub*4/64，即 ub=2048 时每专家 ~128 token，Qwen 是 2048*8/256=64），
   GEMM 形状更友好，可能呈现出不同的 ub 曲线。
6. **`-t` 的最优值**：8C/16T 上 CPU MoE 是 SMT 友好还是超订，只有 R5 知道。
7. **`-fa on/off` 的实际差异**（只有 10 层 attention，预计小，但小 ctx 下 FA 的 mask 优势本来也小）。
8. **iGPU 能否装载 13.9 GiB**：取决于驱动给出的 device-local heap，实测前无法判断。
9. **双设备/分层流水在本模型上是否有意义**（Qwen 的负结论不能直接外推：本模型更小、KV 更小）。
10. **长上下文（32K/128K）下的最优 cache 尺寸与放置**：本计划只覆盖 ctx ≤ 4K，长 ctx 需要单独一轮。
11. **`LLAMA_MOE_DEFER_UPLOAD` / `GGML_MOE_CACHE_PREFILL` 在本模型上的净收益**（代码在，未针对本模型测过）。
12. **static 与 cache 的生成质量差异**（量化 kernel 数值差，`L1` §4）；本轮不测 PPL。

---

## 8. 附录

### 8.1 环境变量（本树已验证存在）

| 变量 | 作用 | 出处 |
|---|---|---|
| `LLAMA_AUTO_RESIDENT=0` | 关掉 Windows 自动 resident，保留 mmap | `src/llama-model.cpp:1594-1597` |
| `GGML_VK_VISIBLE_DEVICES` | 只暴露指定 Vulkan 设备 | `ggml-vulkan.cpp:5330-5345` |
| `GGML_VK_MEMORY_LOGGER=1` | 每次分配打印 `Total device:` peak | `ggml-vulkan.cpp:5313`，`ggml-vulkan-debug.cpp:26` |
| `GGML_VK_PERF_LOGGER=1` | 逐 op GPU 耗时（prefill 不可靠，见 `PP-UB-BOUNDS.md` §1） | `ggml-vulkan.cpp:5310` |
| `GGML_SCHED_TRACE=1`（3 = 逐拷贝） | split 数 / copy / cpu / gpu_submit 分解 | `ggml-backend.cpp:1990` |
| `GGML_MOE_COPY_TRACE=1` | 每次 prefill 拷贝的 host/bank 字节 | `src/llama-context.cpp:2946-2949` |
| `GGML_MOE_CACHE_PREFILL=0` | 关闭 LRU 池预填充（A/B 用，默认开） | `src/llama-moe-cache.cpp:660-661` |
| `LLAMA_MOE_DEFER_UPLOAD=1` | 最后一个专家张量走延迟批上传 | `src/llama-moe-cache.cpp:922-923` |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 小批量算子是否上 GPU 的阈值（默认 32） | `ggml-vulkan.cpp:16383` |
| `GGML_VK_ALLOW_SYSMEM_FALLBACK` | 允许 Vulkan 分配回落到系统内存（**不要开**，会掩盖溢出） | `ggml-vulkan.cpp:4072-4073` |

### 8.2 本目录的分析产物（只读工具，可删）

- `bench/lfm25/gguf_inspect.py`：解析 GGUF header + 张量清单（不读张量数据）。
- `bench/lfm25/gguf-dump.txt`：完整张量/元数据清单（14.88 GiB 模型的 ground truth）。
- `bench/lfm25/cache_slots.py`：按 `llama-moe-cache.cpp` 的预算公式复算 slot 表（§4.2）。

### 8.3 关键源码位置索引

| 主题 | 位置 |
|---|---|
| `-ncmoe` 展开为 `blk.i.ffn_*_exps` | `common/common.h:1243-1262`，`common/arg.cpp:2770-2779` |
| override 匹配（regex_search / first match / CPU+mmap 警告） | `src/llama-model-loader.cpp:1236-1261` |
| 层设备分配 / `-ngl` 自动=全部 | `src/llama-model.cpp:1686-1705, 2435-2437` |
| AUTO resident | `src/llama-model.cpp:1590-1616` |
| cache 构造 / group / slot / n_full / LRU | `src/llama-moe-cache.cpp:249-372` |
| cache 预填充（整层 + LRU 切片） | `src/llama-moe-cache.cpp:629-710` |
| cache 小批路径判定（ubatch ≤ 32） | `src/llama-moe-cache.cpp:206,551-573` |
| cache 拷贝回调（ids 回读 + drain + 上传） | `src/llama-moe-cache.cpp:828-959`，`src/llama-context.cpp:2854-2952` |
| cache 大批评分（bank D2D） | `src/llama-moe-cache.cpp:792-826` |
| graph 接线（slot_map 强制 split / 整层静态视图） | `src/llama-graph.cpp:1663-1672, 2311-2325, 2575-2620` |
| `-fa auto` 解析 / 量化 KV 要求 FA | `src/llama-context.cpp:691-730, 633-637` |
| lfm2moe 张量与图（dense 前缀、conv/attn 二选一） | `src/models/lfm2moe.cpp:23-84`，`src/models/lfm2.cpp:302-332` |
| llama-bench 默认值 / 列表解析 / 重载规则 / TG 逐 token | `tools/llama-bench/llama-bench.cpp:382-427, 1338-1500, 1310-1317, 2229-2239` |
| Vulkan 设备名与枚举顺序 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4808, 5330-5400, 5515, 16389` |
