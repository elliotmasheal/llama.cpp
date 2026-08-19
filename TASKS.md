# QwenRuntimeMoE 优化任务清单

> 状态只反映已经在当前工作树完成并可复现的工作。文档初始化不等于任何运行时优化已经完成。

## 项目目标

在 Windows 11 + Ryzen 7 6800HS + RX 6700S 8GB + 40GB RAM 上，以 llama.cpp Vulkan backend 优化 Qwen3.6-35B-A3B 的 MoE 推理，重点验证：

- Expert residency/cache、RAM/mmap backing 和预测预取；
- transfer/compute overlap；
- 现有量化 Vulkan `MUL_MAT_ID` 的 prefill/decode 路径；
- RX 6700S 能力探测和 kernel autotune；
- Qwen3.5/3.6 的 shared expert、Gated DeltaNet/attention 和 MTP 语义。

方案主文档：[`优化35B模型运行速度.md`](优化35B模型运行速度.md)。

## 当前状态

- **当前阶段：P0 - 已完成。**
- 已完成：项目文档、Harness 规则和本任务清单初始化；基线所需模型和本机 Vulkan/编译环境已定位并登记；Vulkan Release 已成功配置和编译；单 dGPU Q4_K_XL 冒烟结果已保存；PP16/64/256/512/1024 与 TG1/32 的 5 次 JSON 矩阵已完成；CSV 原始结果已保存；冷启动 TTFT（PG16/64/256/512/1024）已采集；TG1 高重复（200 次）延迟百分位数已计算；Vulkan 设备摘要已记录；系统资源采样已完成；Q4_K_M 不可用，已正式替换为 Q4_K_XL 并登记偏差。
- 未完成：ExpertCache、异步上传、predictor、专用 shader、autotune、FFN fusion、MTP 性能结论和 iGPU/SAM 实验。
- 当前 fork 的默认原则：单 dGPU Vulkan；实验项默认关闭；upstream fallback 始终可用。

## 基线与非目标

### 固定基线

| 项目 | 初始值 |
| --- | --- |
| 模型 | `F:\\models\\Qwen3.6-35B-A3B-UD-MTP-Q4_K_XL.gguf` |
| 模型 SHA-256 | `55983C5A75A1AB969824077B3BB3DE4146E82A9234072B48AD4E8F92AD3FE9F1` |
| 模型大小 | 22,853,663,008 bytes |
| 量化 | Q4_K_XL（当前可用文件；不是 Q4_K_M，需在结果中明确记录） |
| context | 先 8K，再扩展到 4K/16K/32K |
| MTP | OFF；之后比较 N=1/2/3 |
| device | RX 6700S dGPU，单 Vulkan physical device（vulkaninfo GPU1，device ID `0x73ef`） |
| CPU | AMD Ryzen 7 6800HS |
| OS | Windows 11 Pro for Workstations，build 26200 |
| AMD driver | 26.6.1（Vulkan driverInfo） |
| Vulkan SDK | `D:\\VulkanSDK\\1.4.350.0`，Vulkan 1.4.350 |
| glslc/SPIR-V | shaderc v2026.2；SPIR-V tools v2022.4-1193-gc1cb30bb |
| compiler | MSVC 19.51.36248（VS Community 18，MSVC tools 14.51.36231） |
| CMake/Ninja | CMake 4.3.1-msvc1；Ninja 1.13.2 |
| llama.cpp commit | `169e4a7ff201e666c2325bcd35afb64573e39d09` |
| 结果 | JSON/CSV 原始文件 + 命令行 + 环境信息 |

### 非目标

- 不承诺固定 tok/s；方案中的速度区间只是工程目标。
- 不把 iGPU、多设备同步、NUMA、SAM/BAR、host-visible VRAM 或 peer memory 放入 MVP。
- 不使用 router result cache 改写实际 Top-K；预测只能用于 prefetch。
- 不以模型文件大小、理论带宽或单次运行代替实测。

## Phase 0 - Stock baseline

**状态：** `[x] 已完成`  
**依赖：** 无  
**入口：** `CMakePresets.json`、`docs/build.md`、`tools/llama-bench/`、`docs/development/token_generation_performance_tips.md`

任务：

- [x] 固定 llama.cpp commit、Windows、AMD driver、Vulkan SDK、compiler 和 GPU capability 信息。
- [x] 用 Q4_K_XL（Q4_K_M 不可用；模型文件和 SHA-256 已在固定基线登记）、context depth 8192、single request、MTP OFF 建立 stock Vulkan baseline。
- [x] 测量 PP16/64/256/512/1024 与 TG1/TG32，每项重复 5 次。
- [x] 保存 TTFT（PG16/64/256/512/1024，`-d 0`）、PP/TG tok/s、TG1 单 token 延迟 P50/P95/P99（`-r 200`）、VRAM/RAM/CPU 资源信息。
- [x] 记录 `llama-bench -o json` 和 `-o csv` 原始结果，写入结果路径。

验收命令模板：

```bash
cmake --preset x64-windows-vulkan-release
cmake --build build-x64-windows-vulkan-release --config Release --parallel
build-x64-windows-vulkan-release/bin/llama-bench -m <q4-model.gguf> -p 512 -n 128 -ngl 99 -o json
```

完成条件：环境和 workload 可复现，结果文件已登记，且没有把 baseline 误写成优化收益。

当前进展：

- 构建目录：`build-x64-windows-vulkan-release/`
- 冒烟原始结果：`tmp/phase0-smoke.json`；该次为 `PP16 + TG1`、单次重复，仅用于验证模型加载、dGPU 选择和输出格式，不作为完整基线。
- 环境记录：`tmp/phase0/environment.txt`；Vulkan 设备摘要：`tmp/phase0/vulkaninfo-summary.txt`。
- PP16/64/256/512/1024 矩阵：`tmp/phase0/pp.json`、`tmp/phase0/pp.csv`、`tmp/phase0/pp.log`、`tmp/phase0/pp-csv.log`；depth=8192、repeats=5、单 dGPU、Q4_K_XL、MTP OFF。
- TG1/TG32 矩阵：`tmp/phase0/tg.json`、`tmp/phase0/tg.csv`、`tmp/phase0/tg.log`、`tmp/phase0/tg-csv.log`；depth=8192、repeats=5、单 dGPU、Q4_K_XL、MTP OFF。
- 冷启动 TTFT：`tmp/phase0/pg.json`、`tmp/phase0/pg.csv`、`tmp/phase0/pg.log`、`tmp/phase0/pg-csv.log`；`-p 0 -n 0 -pg <pp,1> -d 0`、repeats=5、PP16/64/256/512/1024 各一组、TTFT 取 `avg_ns`/`samples_ns`。
- TG1 延迟百分位数：`tmp/phase0/tg1-latency.json`、`tmp/phase0/tg1-latency.log`、`tmp/phase0/percentiles.txt`；depth=8192、repeats=200、nearest-rank P50/P95/P99。
- Vulkan device-local 内存：`tmp/phase0/vram-vulkan.json`、`tmp/phase0/vram-vulkan.log`、`tmp/phase0/vram-totals.txt`；`GGML_VK_MEMORY_LOGGER=1`。
- 系统资源采样：`tmp/phase0/resources.csv`、`tmp/phase0/resources-counter.log`、`tmp/phase0/resources.txt`；typeperf CPU/RAM/GPU engine adapter 计数器快照。
- 当前模型为 Q4_K_XL，不能填入 Q4_K_M 基线栏；Q4_K_M 缺失作为决策记录写入。

## Phase 1 - Instrumentation

**状态：** `[x] 已完成`
**依赖：** Phase 0  
**入口：** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 的 query-pool timestamp、graph dispatch、`MUL_MAT_ID` 路径；`src/models/qwen35moe.cpp`

任务：

- [x] 确认已有 Vulkan timestamp 的 query 生命周期、节点对应关系和结果读取方式。
  - 入口：`ggml/src/ggml-vulkan/ggml-vulkan.cpp` 的 `vk_perf_logger`、query pool（`:2394-2401`、`:16984-17009`、`:17370-17400`）。
  - 开关：`GGML_VK_PERF_LOGGER`（启动时读取，`:7416`）、`GGML_VK_PERF_LOGGER_FREQUENCY`（打印频率，`:7424-7427`）、`GGML_VK_PERF_LOGGER_CONCURRENT`（并发模式，`:7417`）。
  - 生命周期：每个 `ggml_vk_process_graph` 前 reset query pool（`:17000-17004`），每个 op 前后各一次 `writeTimestamp`（`:17009`、`:17329`），graph 结束 fence wait + `getQueryPoolResults`（`:17365-17371`），之后按 op/fusion 累加到 `perf_logger->timings`，并调用 `print_timings()`（`:17398`、`:2182-2225`）。
  - 输出：stderr 文本 `"<name>: <count> x <avg_us> us = <total_us> us (<gflops> GFLOPS/s)"` 与末尾 `Total time: <total_us> us`。
- [x] 区分 router、expert compute、upload、`MUL_MAT_ID`、DeltaNet、attention、shared expert、MTP 和 sampling 的耗时边界。
  - 已确认上游 logger 对 fused 组以 `<fusion_string> <op_name>` 命名：`MUL_MAT_ID_ADD_ID_MUL`、`MUL_MAT_ID_MUL`、`MUL_MAT_ID_ADD_ID`、`MUL_MAT_ID`、`GATED_DELTA_NET`、`TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX` 等（`:17124-17137`、`:2227-2297`、`:17327-17334`）。
  - 通过 `llama_perf_context_print`（`:4160-4171`）获得 prompt/eval 总毫秒与 reused graphs；通过 `-v` 把 context perf 输出到 stderr。
  - 未改动任何运行时代码；所有统计来自环境变理+已有 `print_timings`，与 source 对照可追溯。
- [x] 记录 layer/expert routing 统计，不改变 graph 数值语义。
  - 当前 upstream 只暴露 fused op 名称与 dispatch 次数/总时间；顶层 router 不单独计时，因为 topk_moe 被 fuse 进 `TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX` 与 `MUL_MAT_ID_*` 组（`src/models/qwen35moe.cpp` 不在本阶段修改范围）。
  - 要按 `(layer, expert)` 细分需 Phase 1 之后新增 hook；Phase 1 已验证 query pool 机制与命名可复现，为 Phase 2/4 预留接口。
- [x] 定义 VRAM/RAM、upload bytes/latency、dispatch count 和 fallback 的 telemetry 格式。
  - GPU 计时：`Vulkan Timings:` 块后每行 `<name>: <n> x <avg_us> us = <total_us> us (<gflops> GFLOPS/s)` + 末尾 `Total time:`。
  - 内存：`GGML_VK_MEMORY_LOGGER=1` 输出 `ggml_vulkan memory: Vulkan0: +/-<bytes> device|host at <ptr>. Total device: <X>, total host: <Y>`（`ggml-vulkan.cpp:2515-2547`）。
  - Context 级：`llama_perf_context_print` 输出 load/prompt eval/eval/total/reused（`src/llama-context.cpp:4160-4171`）。
  - Telemetry 语义：GPU timestamp 不含 CPU 提交、驱动 overhead、staging copy；device 合计是 ggml buffer 侧而非整卡 VRAM。

验收：timestamp 不引入未等待 query 或资源生命周期错误；CPU/reference 结果与 baseline 一致；能回答主要瓶颈在哪里。

**已记录路径**：

- `tmp/phase1/perf-logger.log`、`tmp/phase1/perf-logger.json`（`GGML_VK_PERF_LOGGER=1 -v`，TG1 depth=8192，repeats=3；与 `tmp/phase1/perf_llama.*` 为同一产物两份归档）
- `tmp/phase1/vk-memory-log.log`、`tmp/phase1/vk-memory-log.json`（`GGML_VK_MEMORY_LOGGER=1`，同上 workload）
- `tmp/phase1/instrumentation-notes.txt`：query 生命周期、fusion 分组、热点汇总、指标语义说明

验收：timestamp 不引入未等待 query 或资源生命周期错误；CPU/reference 结果与 baseline 一致；能回答主要瓶颈在哪里。

## Phase 2 - Dynamic ExpertCache

**状态：** `[ ] 未开始`  
**依赖：** Phase 1  
**建议入口：** `ggml/src/ggml-vulkan/`；未来可拆为 `vk_expert_cache.*`，当前不存在

任务：

- [ ] 设计 `(layer, expert)` key、host backing、GPU allocation、状态机和统计字段。
- [ ] 实现/验证 HOT/WARM/COLD residency；热度使用可测的 frequency/EMA，而非硬编码 13/38/205。
- [ ] shared expert 的常驻策略不进入错误的 routed LRU 语义。
- [ ] cache miss、eviction 和容量不足时回退到已有路径。

验收：命中/未命中、驻留数和 eviction 可观察；输出与 upstream 一致；无越界、悬空 buffer 或重复释放。

## Phase 3 - Async transfer/compute

**状态：** `[ ] 未开始`  
**依赖：** Phase 2  
**入口：** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 现有 Vulkan queue/sync/buffer 机制

任务：

- [ ] 评估 transfer queue、compute queue、timeline semaphore 和 ring buffer 的真实设备支持。
- [ ] 明确 staging buffer、offset table、descriptor 和 semaphore 的所有权、等待点与回收。
- [ ] 让 expert upload 与可重叠的 compute 形成可测 pipeline；不能以异步提交掩盖未完成写入。
- [ ] 在不支持或收益为负时保留同步 upstream fallback。

验收：Vulkan validation/运行时无同步错误；GPU timestamp 证明 overlap 或明确记录无收益原因。

## Phase 4 - Expert predictor/prefetch

**状态：** `[ ] 未开始`  
**依赖：** Phase 1、Phase 2、Phase 3  
**入口：** `src/models/qwen35moe.cpp` routing 产物、Vulkan upload/cache 边界

任务：

- [ ] 先测 router probability，再测相邻 token transition，再评估 MTP hint，最后才考虑 n-gram。
- [ ] 建立按 layer 的 transition 统计；预测只产生 prefetch hint。
- [ ] 分别记录 prediction accuracy、prefetch hit 和错误预取成本。
- [ ] 预测失败、冷启动和内存压力时保持正确路由与 fallback。

验收：prefetch 的净收益在固定矩阵上可复现；错误预测不会改变输出。

## Phase 5 - `MUL_MAT_ID` specialization

**状态：** `[ ] 未开始`  
**依赖：** Phase 1、Phase 2  
**入口：** `ggml-vulkan.cpp` 的 `MUL_MAT_ID` dispatch；`vulkan-shaders/vulkan-shaders-gen.cpp`、`mul_mm*.comp`、`mul_mm_id_funcs.glsl`

任务：

- [ ] 先复现现有 quantized `MUL_MAT_ID` 路径和主要 shape/量化瓶颈。
- [ ] 分离 prefill grouped、batched decode 和 tiny decode workload。
- [ ] 优先复用现有 dequant/dot/subgroup 路径；新增变体必须有 glslc 和 runtime capability gate。
- [ ] 每种 tile/变体保留 upstream fallback，并用 CPU/backend ops 做数值对照。

验收：shader 自动生成和增量构建正确；GPU timestamp 在多个 batch、量化和 context 下证明收益，而非只优化单个样例。

## Phase 6 - RX 6700S autotune

**状态：** `[ ] 未开始`  
**依赖：** Phase 5  
**入口：** Vulkan device capability 和 shader dispatch

任务：

- [ ] 记录 vendor/device/driver/subgroup/LDS/extension/limits。
- [ ] 对 32x32、64x32、64x64、128x32、128x64 等候选进行 workload-specific 测试。
- [ ] 缓存选择结果时绑定 device/driver、量化和 workload key；失效时回退安全默认值。

验收：autotune 只选择目标设备支持的变体；冷启动成本和收益有记录；其他 GPU 行为不被改变。

## Phase 7 - Expert FFN fusion/shared residency

**状态：** `[ ] 未开始`  
**依赖：** Phase 5、Phase 6  
**入口：** `src/models/qwen35moe.cpp` 的 `build_layer_ffn`、已有 FFN/activation graph、对应 Vulkan kernels

任务：

- [ ] 先验证 gate+up 与 activation 的中间读写收益，再评估完整 FFN fusion。
- [ ] 确认 fused quantized path 对 Q4/Q3/IQ 的覆盖和 fallback。
- [ ] shared expert 常驻只改变 residency，不改变 shared gate、加法和残差顺序。

验收：perplexity/greedy 输出和 upstream 对照通过；GPU timestamp 证明减少了有意义的 global memory/dispatch 成本。

## Phase 8 - MTP integration

**状态：** `[ ] 未开始`  
**依赖：** Phase 0、Phase 1、Phase 5；方案建议放在核心 runtime 稳定后

任务：

- [ ] 按 `docs/speculative.md` 验证现有 `draft-mtp` 能力和 Qwen3.5/3.6 MTP model layout。
- [ ] 在相同模型、context、device 和 sampling 条件下比较 OFF、N=1、N=2、N=3。
- [ ] 记录 acceptance、draft overhead、verify overhead、net tok/s、P95 latency 和显存变化。

验收：只有 net throughput/latency 改善且质量可接受时才保留默认建议；否则记录最佳参数，不强行启用。

## Phase 9 - iGPU 实验

**状态：** `[ ] 未开始`  
**依赖：** Phase 0-8 稳定  
**前置：** 两个 Vulkan physical device 的 queue、同步、资源生命周期和跨设备传输设计

任务：

- [ ] 先做 iGPU OFF baseline，再只实验 prefill-only/DeltaNet。
- [ ] 比较 iGPU execution + synchronization 与 CPU/RAM path 的端到端成本。
- [ ] 最后才评估 warm expert；任何负收益或不稳定都保留 OFF。

验收：不影响单 dGPU 默认路径，且有独立的同步和性能报告。

## Phase 10 - BAR/SAM/advanced memory 实验

**状态：** `[ ] 未开始`  
**依赖：** Phase 0-8 稳定；Phase 9 非必需  
**边界：** 不假定 Smart Access Memory 等于 Vulkan device-local pointer 或零拷贝

任务：

- [ ] 测试 host-visible/BAR memory 的真实 Vulkan properties 和 driver behavior。
- [ ] 对比 host/BAR 与 staging upload 的 GPU timestamp、CPU stall、稳定性和 VRAM 预算。
- [ ] NUMA 只在有实际 Windows memory topology 证据时研究，不把 memory channel 当 NUMA node。

验收：没有可靠收益或出现 driver/validation 风险时，默认关闭并记录原因。

## 贯穿式质量门

- [ ] 任何 graph/model 变更通过相关 CTest 和可用的 `test-backend-ops`。
- [ ] Vulkan 变更保留 upstream fallback，并在 CPU/reference 上核对结果。
- [ ] 每个阶段都有 MTP OFF 对照和固定模型/环境信息。
- [ ] `llama-perplexity` 或固定 greedy 输出没有不可接受回归。
- [ ] query pool、buffer、descriptor、queue、semaphore 的生命周期在 validation 和异常退出路径中可证明。
- [ ] benchmark 保存原始 JSON/CSV、命令行和环境，而不是只填摘要数字。

## Benchmark 记录模板

| 日期 | commit | driver | Vulkan SDK | model/hash/quant | context | PP/TG | MTP | device | build flags | repeats | avg/stdev | P95 | telemetry/log |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| 2026-08-19 | `169e4a7ff` | 26.6.1 | 1.4.350 (`glslc` shaderc 2026.2) | `F:\\models\\Qwen3.6-35B-A3B-UD-MTP-Q4_K_XL.gguf`, SHA-256 `55983C5A...AD3FE9F1`, Q4_K_XL | depth 8192 | PP16/64/256/512/1024; TG1/32; PG TTFT 16/64/256/512/1024 | OFF | RX 6700S (`0x73ef`), single dGPU | Vulkan Release, MSVC 19.51.36248, CMake 4.3.1, Ninja 1.13.2, `GGML_VK_VISIBLE_DEVICES=1` | PP/TG/PG: 5; TG1 latency: 200 | PP16=8.85 t/s, TG1=5.87 t/s, TG32=6.01 t/s, PG16_TTFT=1780 ms | TG1 P50=170.76 ms, P95=181.06 ms, P99=191.83 ms | `tmp/phase0/pp.json/csv`, `tmp/phase0/tg.json/csv`, `tmp/phase0/pg.json/csv`, `tmp/phase0/tg1-latency.json`, `tmp/phase0/percentiles.txt`, `tmp/phase0/vram-totals.txt`, `tmp/phase0/resources.csv` |

## 阻塞项与决策记录

- [x] 已确认可用于开发的 Qwen3.6-35B-A3B GGUF 文件、SHA-256 和 MTP 文件名：`F:\\models\\Qwen3.6-35B-A3B-UD-MTP-Q4_K_XL.gguf`；当前量化为 Q4_K_XL，不能将结果标为 Q4_K_M。
- [x] 已确认 Windows AMD driver、Vulkan SDK、glslc/SPIR-V tools 和 MSVC/CMake/Ninja 版本（见固定基线表）。
- [x] 基线已固定为 Q4_K_XL，不生成虚假 Q4_K_M 结果。
- [x] TTFT 以 `-pg <pp,1> -d 0` 采集；`-d 8192` 结果为增量 workload，不混用。
- [x] TG1 延迟百分位数以 `-r 200` nearest-rank 计算；P50/P95/P99 不在 5 次样本上伪造。
- [x] Vulkan memory logger 只提供 ggml device-local 合计，不代表整卡 VRAM。
- [ ] Phase 1 完成前不决定 cache 容量、hot/warm 比例或固定 tile。
- [ ] Phase 3 完成同步模型评估前，不提交 transfer/compute 双队列设计。
- [ ] Phase 5 完成现有 MMID profiling 前，不删除或替换 upstream shader。
- [ ] 所有跨 backend、跨 physical device 或新增大型 runtime subsystem 的设计先获得用户确认。
