# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> [!NOTE]
> 本仓库是 `elliotmasheal/llama.cpp` fork，基线是上游 master；`origin` 指向本 fork，不配置 upstream。
> **本项目不考虑"能否上游化"，追求小 VRAM 设备下指定模型的极致性能。**
> 做难而正确的事；异常现象不猜，靠实际数据 / 代码说话。
> 每次行动前必须有充足的行动依据，而不是先猜再做最后回滚，浪费时间和精力。
> 工具调用尽可能只接收有关的数据，不要一股脑全量接收工具输出，实在没办法也应该将工具输出指向临时文件然后去文件中搜索关键字。

## 回复风格（每轮最后）

先一句话结论，再一句话"这意味着什么"，最后才给数字；数字只留关键几个，其余指向 `bench/`。
术语第一次出现必须用生活类比（bank = 专家在显存里的停车位；miss = 车位不够回内存取；档位 = 显存的限高杆）。
一段话说清"改了什么 / 快了还是慢了 / 为什么 / 下一步"。口径和坏消息（`abort`、`vram-oom`、`status` 非 `ok`）
照实说且先说，不得用"应该没问题"盖过去。

## 唯一目标

`qwen35moe`（40 层 x 256 专家，top-8；Vulkan）上，以 **256K ctx + 10K prompt** 为统一工况，
追求 **PP 与 TG 最大化**，实现 **VRAM + RAM + SSD 三层协同**，且显存与内存都不足时 SSD 能当临时缓冲。

当前可用口径 best：PP 703.9 / TG 20.9（IQ4_XS，8176 档，per-group 环）。物理墙：TG 123 t/s、PP1000 需 2.05 s/eval。
距离与成本账见 [bench/RESULTS.md](bench/RESULTS.md) 第 1-2 节；三层分工见 [bench/THREE-TIER.md](bench/THREE-TIER.md) 第 1 节。

## 每轮三步

1. 改一个机制，在目标口径跑 A/B 矩阵：`bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2`。
2. 把 ΔPP / ΔTG / 生效条件记进 `bench/RESULTS.md` 2.1 账本；`status` 非 `ok` 记进未结项。只更新受影响的那份文档。
3. 动 Vulkan / 调度 / GEMM 跑回归；动数值路径附质量门读数。A/B 开关只活在这一轮。

## 开关纪律

- 开关只用于 A/B，不是产品旋钮：赢了折叠进自适应（规划器 / 代价表 / 预算式），输了删除。
- 生产必须自适应：默认值由档位、bank、命中率、预算自动决定，不允许"默认关"长期挂着。
- 用户可见只有三类：档位与预算（`--store-*-mib`）、容量（`-c` / `-ncmoe` / `-lm`）、标注 experimental 的开关。
- 每个机制必须留下可量化的 ΔPP / ΔTG 与代价条件（多吃的显存 / 带宽 / 命中率），否则不予验收。
- 加权规则：TG 权重 2、PP 权重 1；同权重先看 r=2 可复现性，再看是否吃容量。

## 工程红线

- 容量不足与运行期失败必须退化，不得 abort。`src/llama-moe-cache.cpp` 已无 `GGML_ABORT`，用计数验收退化。
- SSD 只做容量层，绝不进 decode 单步（0.81 GB/s，206 MiB/token = 258 ms）。
- 复用现有件，不新增平行子系统；跨层搬运只走 `llama-store` 两个入口。
- 数字必须带口径（日期、分支、bank、显存档、电源态）；跨口径比较视为错误。
- 结论只写进 `bench/` 四份文档，不新增第五份 md。

## 回归门禁（`build-vulkan/bin/`）

`MUL_MAT_ID -b CPU` 945/945、`MUL_MAT_ID_MASK -b CPU` 4/4、`MUL_MAT_ID -b Vulkan1` 939/939、
`MUL_MAT -b Vulkan1` 2275/2275、`FLASH_ATTN_EXT -b Vulkan1` 5332/5332、`ctest` 34/35
（`test-model-resolution` 环境性失败）。动注意力数值（含 `-fac`）加跑 `bash bench/tools/quality-gate.sh`。

## 机器事实

RX 6700S 8 GiB（`-dev Vulkan1`）、Ryzen 7 6800HS、RAM 40 GiB、电源「平衡」+ AC。
设备总量 8176 MiB，桌面占 0.5-1.4 GiB；resident 门槛 `n_bytes x 1.25 + 1 GiB`。
8 GiB 档 resident 基准是 `IQ4_XS`（16.96 GiB）；`Q4_K_XL`（21.28 GiB）过不了 27.6 GiB 门槛，`-lm none` 会退 mmap。

## 仓库裁剪范围

backend 只有 CPU 与 Vulkan；模型只有 qwen35moe；tools 只有 cli / server / llama-bench / mtmd / perplexity。
已删：其他 backend / arch、quantize / imatrix 等工具、`examples/`、`pocs/`。CI 只有 Vulkan + 代码风格。

## 现成基础设施（不要重复造）

- MoE cache `src/llama-moe-cache.{h,cpp}`；预算 `common/fit.cpp` 的规划器 + 账本。
- 算子 `GGML_OP_MUL_MAT_ID` + 本 fork 的 `MUL_MAT_ID_MASK`（只有 CPU）。
- 调度钩子 `ggml_backend_sched_set_*_split_callback`；Vulkan 原语 `ggml_backend_vk_prefetch_*_seq`。
- store `src/llama-store*`（常开，无总开关），边界见 [bench/THREE-TIER.md](bench/THREE-TIER.md) 第 6 节。

## 文件索引

```
AGENTS.md              本文件：目标、流程、开关纪律、红线、门禁
bench/README.md        口径、档位、必报列、测量陷阱
bench/RESULTS.md       数字、机制收益账本、硬件天花板
bench/DECISIONS.md     判决、已证伪、实验开关、硬约束、未结项
bench/THREE-TIER.md    三层、缺口 F0-F13、唯一执行顺序、store 边界
bench/prompts/         prompt10k（10428 tok）prompt20k（19758 tok）
bench/env/             power.txt  machine.txt
bench/tools/           run-matrix.sh  quality-gate.sh  探针
skills/code-review/    写代码 / 加注释前读
```

## 代码地图

- `src/`：加载 `llama-model*`、图 `llama-graph.cpp`、KV `llama-memory*` / `llama-kv-cache*`、
  MoE `llama-moe-cache.cpp` / `llama-moe-exec.cpp`、C API（`include/llama.h`，扩展 `src/llama-ext.h`）。
- `ggml/`：张量库 + CPU / Vulkan backend；调度器 `ggml/src/ggml-backend.cpp`。
- `common/`：参数 `arg.cpp`、规划 `fit.cpp`、chat / jinja、`speculative.cpp`；`tools/` 在 `app/` 下有统一入口。

## 构建与测试

```
cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
cmake --build build-vulkan -j 8
```

Git Bash 跑 `build-vulkan/bin/*.exe` 前 `export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"`。性能一律 Release。

## 代码与提交规范

只用 ASCII 标点：破折号、箭头、乘号、省略号写成 `-`、`->`、`x`、`...`。注释 1-2 行，只写不显然的约定。
提交小而可回退；改 Vulkan / 调度 / GEMM 必须写明跑了哪些门禁。

## 硬性禁止

- 不确定时少动手；大改动或引入新模式前先读相关文件，暂停并征求确认。
- 不得用 `GGML_ABORT` 处理容量不足、内存不足、路由过散这类运行期条件。
- 不得把只在一个档位 / 一个 bank / 一个电源态下成立的结论写进文档。
- 不得为让数字好看跳过 `-nkvo`、跳过小显存档或缩短 prompt。
- 不得为每个功能新增永久开关；不得给 store 加总开关。
- 不得在调用方另写第三套跨层搬运；不得把路由预测塞进 store。
