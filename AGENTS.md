# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> [!NOTE]
> 本仓库是 `elliotmasheal/llama.cpp` fork，基线是 ggml-org/llama.cpp 的上游 master。
> `origin` 指向本 fork，未配置 `upstream` remote。**本项目不考虑"能否上游化"，追求小VRAM设备下的指定模型的极致性能。**

## 回复风格（每轮最后的回复）

回复是讲给普通用户听的，不是论文摘要。术语、日志、推导留在代码与 `bench/` 文档里：

- 先一句话结论，再一句话"这意味着什么"，最后才给数字；数字只留关键的几个，其余指到文档与日志路径。
- 术语第一次出现必须用一个生活类比带过，例如：bank = 专家在显存里占的"停车位"，
  miss = 车位不够，只能回内存重新取；档位（vram）= 显存被卡住的"限高杆"；held back = 先扣下不给用的那部分。
  不解释就直接甩"调度器分片启发式""带宽放大""非一致访存"这类词，属于不合格回复。
- 一段话说清"改了什么 / 快了还是慢了 / 为什么 / 下一步"，长度以非本行的人能读完为准。
- 通俗不等于含糊：口径（日期、分支、bank、显存档、电源态）和坏消息（`abort`、`vram-oom`、`status` 非 `ok`）
  照实说，且先说坏消息；不得用"应该没问题""总体向好"盖过去。
- 表格只用来并排比数，不用来解释概念；需要长推导就写进 `bench/` 那四份文档，回复里给链接式路径。

## 目标（唯一）

在 qwen35moe（测试模型 `F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf`，20.2 GiB，Vulkan backend）上，
以 **256K ctx + 10K prompt** 为统一工况，追求 **PP 与 TG 最大化**，实现 **VRAM + RAM + SSD 三层协同**，
其中 **显存与内存都不足时 SSD 必须能当临时缓冲**。

三层分工与硬约束写在 `bench/THREE-TIER.md`；工程红线写在下面，每次改动都要遵守。

## 期望约束

### 1. 每轮优化都在目标工况下测

一轮 = 一个改动 + 一次矩阵跑 + 四份文档的更新。不允许"先在短上下文验证，之后再外推"。

```
# 统一工况（长上下文口径，必须用 llama-cli，llama-bench 会把 n_ctx 设成 p+n）
# K/V 默认按工作集增长（-kvg -1），256K + 10K 只用 320 MiB 并驻设备，所以不再加 -nkvo
llama-cli -m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -fa on `
  -t 5 -b 2048 -ub 2048 -c 262144 -f bench\prompts\prompt10k.txt -n 64 `
  --temp 0 --no-warmup --no-display-prompt -lm none -v

# 一条命令跑完整矩阵（bash，子进程用普通重定向）
bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2
```

- 显存档位用 `GGML_VK_VRAM_LIMIT_MIB=<MiB>` 实现：设备初始化时 hold back 差额，
  于是 `llama-fit-params`、MoE bank 规划器和每次分配看到的都是缩小后的预算。集显不生效。
- RAM 条件用 `-lm none`（resident，模型全入内存）与 `-lm mmap`（内存不足档）两臂；
  不加 `--moe-cache-mib` 时由 `common_moe_plan_get()` 自己算 bank，这就是"自适应"。
- 电源：「平衡」方案 + AC 供电 + dGPU 上无其它推理进程（GUID 与机器事实见 `bench/env/`）。
- 必报列：`tier | load | bank | ncmoe | -nkvo | PP | TG | VRAM peak | held | VRAM used | RAM peak | reallocs | hit % | status`。
  `VRAM peak` 取**每进程独占**（`\GPU Process Memory(pid_<pid>*)\Dedicated Usage` 峰值），不要看设备总量；
  真实足迹看 `used = peak - held`（`held` 是显存档自己 hold back 的部分，它也计入进程峰值）。
- `status` 不是 `ok`（`abort` / `vram-oom`）算缺陷，要进 `bench/DECISIONS.md` 的未结项，不能当成"配置不对"忽略。

### 2. 工程红线

- **容量不足必须退化，不得 abort**。显存/内存/bank 不够时回退到慢但正确的路径并计数告警。
  现在还有 5 处 `GGML_ABORT`（`src/llama-moe-cache.cpp` 的 :128、:1910、:2013、:2045、:1740），
  其中 :1910 把正确性绑在调度器分片启发式上。这是 4G/6G 档的前置修复项（`THREE-TIER.md` F1）。
- **SSD 只做容量层，绝不进 decode 单步关键路径**：随机 1.95 MiB 实测 0.81 GB/s，
  206 MiB/token 要 258 ms。SSD 只允许出现在预取批次与 prefill，且服务命中率 >= 85% 的冷层。
- 新行为一律带开关，默认值由**目标工况**的实测决定；开关同时登记进 `bench/DECISIONS.md` 的开关表。
  只靠 env 而不上 CLI 的行为不得成为默认路径。
- 复用现有基础设施（MoE cache、fit、graph、MUL_MAT_ID），避免新增平行子系统。
- 数字必须带口径（日期、分支/提交、电源态、bank、显存档）；跨口径比较一律视为错误，
  换算陷阱清单在 `bench/README.md` 第 4 节。
- 结论只写进 `bench/` 的那四份文档，**不新增第五份 md**；原始产物写进 `bench/raw/`（不进 git）。
- 质量门已恢复（`THREE-TIER.md` F6）：`bash bench/tools/quality-gate.sh` 给出 PPL 与 logit KL，
  基线 logits 在 `F:/LLM/kld/`（2.0 GB，不进 git）。任何降精度、混合精度、CPU 分支或注意力数值路径的改动，
  都要贴这条命令的读数（PPL | mean KLD | same top p），否则不予验收。

### 3. 回归门禁（改 Vulkan / 调度 / GEMM 后必跑，命令在 `build-vulkan/bin/` 下）

`test-backend-ops -o MUL_MAT_ID -b CPU` = 945/945、`-o MUL_MAT_ID_MASK -b CPU` = 4/4、
`-o MUL_MAT_ID -b Vulkan1` = 939/939、`-o MUL_MAT -b Vulkan1` = 2275/2275、
`-o FLASH_ATTN_EXT -b Vulkan1` = 5332/5332（2026-10-23 已补齐）、
`ctest --test-dir build-vulkan` = 32/32。`-o MUL_MAT_ID_MASK -b Vulkan1` = 0/0，因为 Vulkan 侧还没有这个 op。
动注意力数值（含 `-fac` 分块）再加一条：`bash bench/tools/quality-gate.sh` 必须 `GATE PASS`。

## 仓库裁剪范围

本 fork 已物理裁剪（`trim-vulkan-cpu-qwen35moe` 分支），不要去找不存在的东西：

- backend 只有 **CPU 与 Vulkan**；CUDA/Metal/SYCL/OpenCL/MUSA/CANN/... 的源码、CI、文档、fixture 全部删除。
- 模型只支持 **qwen35moe**（`src/models/qwen35moe.cpp` + 依赖的 `delta-net-base.cpp`）；
  `llama_model_mapping()` 只有这一个 case，其他 arch 直接抛 `unsupported model architecture`。
- tools 只有 **cli、server、llama-bench、mtmd、perplexity**（外加 server 依赖的 `tools/ui`）；
  `perplexity` 是 2026-10-23 为质量门从 git 历史恢复的（`THREE-TIER.md` F6）；
  `quantize`、`completion`、`batched-bench`、`fit-params`、`gguf-split`、`imatrix`、
  `tokenize`、`tts`、`cvector-generator`、`export-lora`、`results`、`connect`、`examples/`、`pocs/` 已删除。
  统一入口 `llama` 只分发 `serve`/`cli`/`bench`/`download`，`llama-perplexity` 是独立二进制。
- tests 只剩 32 个可跑的用例；`test-tokenizer-0` 与 `models/ggml-vocab-*.gguf` 已删除
  （加载 vocab 也要走模型工厂，其他 arch 加载不了）。`models/templates/` 保留。
- CI 只留 `build-vulkan.yml` + `code-style.yml`；`.devops` 只留 `vulkan.Dockerfile` + `tools.sh`。
- 仍保留的 Python 侧：`convert_hf_to_gguf.py` + `conversion/` + `gguf-py/` + `requirements/`。

## 机器事实

RX 6700S 8 GiB（`-dev Vulkan1`），Ryzen 7 6800HS（以 `llama-bench` 的 `cpu_info` 为准），
RAM 40172 MiB（采集值见 `bench/env/machine.txt`），电源「平衡」（`bench/env/power.txt`）。
桌面已占显存 0.5-1.4 GiB；设备总量 8176 MiB。resident 门槛是 `n_bytes x 1.25 + 1 GiB`，
20.2 GiB 模型要约 26.3 GiB 空闲内存；不足时 AUTO 退回 mmap，而 mmap 下 bank 与预取被守卫强制为 0，
pp512 直接腰斩，所以基准测试要显式 `-lm none`。

## 现成基础设施（不要重复造）

- MoE 专家 GPU cache：`src/llama-moe-cache.{h,cpp}`；`--moe-cache-mib N`（`cparams.moe_cache_size`）；
  图接线 `build_moe_cache_slots()` 与 `src/llama-graph.{h,cpp}` 的 `moe_cache` 字段；
  SLRU + 整层静态驻留 + 2 槽预取环（`LLAMA_MOE_PREFETCH*`）。
- 预算与放置：`common/fit.cpp` 的 `common_moe_plan_get()`（未指定 `--moe-cache-mib` 时给 CLI/server 默认值）、
  `--cpu-moe`、`--n-cpu-moe`、`-ot`；`llama-fit-params` 工具已删除，只用库函数。
- MoE 算子：`GGML_OP_MUL_MAT_ID`，加上本 fork 的 `GGML_OP_MUL_MAT_ID_MASK`（**只有 CPU 实现**，
  要 "GPU 算命中 / CPU 算 miss" 必须先补 Vulkan，见 `THREE-TIER.md` F4）。
- 调度器扩展点：`ggml_backend_sched_set_pre_split_callback()` / `_post_split_callback()`、
  `ggml_backend_set_tensor_async_deferred()` / `wait_deferred()`、`ggml_backend_buft_get_max_alloc_size()`。
- Vulkan 侧 API：`ggml_backend_vk_prefetch_tensor/submit/wait`（`ggml/include/ggml-vulkan.h`）。
- MTP / 投机解码：`--spec-type draft-mtp`，`LLM_TENSOR_NEXTN_*`，draft 循环在 `common/speculative.cpp`。

## 文件索引

```
AGENTS.md                     本文件：目标、期望约束、门禁、索引
bench/README.md               测试口径、显存/内存档位实现、必报列、测量陷阱、目录说明
bench/RESULTS.md              数字的唯一来源（含尚未开测的目标矩阵）
bench/DECISIONS.md            可达性判决、已证伪清单、开关表、代码级硬约束、未结项
bench/THREE-TIER.md           三层协同的现状判定、缺口 F0-F10 与顺序
bench/prompts/                prompt10k.txt(10428 tok) prompt20k.txt(19758 tok) + d3 的两份
bench/env/                    power.txt（电源方案 GUID）、machine.txt（采集的硬件事实）
bench/tools/                  run-matrix.sh（矩阵，bash） quality-gate.sh（PPL+KLD，bash）
                              bench10k.ps1 matrix.ps1 run-case.ps1 monitor.ps1
                              ssd_probe.py membw_probe.py gguf_dims.py read_trace.py summarize_perf.py
                              vkgate.cpp deferred-copy.cpp vklimit-probe.cpp（显存档位自检）
bench/raw/                    历史原始日志与 CSV，不进 git，只用于复核数字出处
tools/expert-repack/          GGUF 专家重排脚本（质量门已就位，但 repack 的 KLD != 0 未修）
skills/code-review/SKILL.md   写代码/加注释前读
```

## 代码地图

- `src/` 核心库 libllama：加载 `llama-model.cpp`/`llama-model-loader.cpp`/`llama-mmap.cpp`；图 `llama-graph.cpp`；
  KV/内存 `llama-memory*.cpp`、`llama-kv-cache*.cpp`；MoE cache `llama-moe-cache.cpp`、执行器 `llama-moe-exec.cpp`；
  采样 `llama-sampler.cpp`；C API `include/llama.h`，私有扩展 `src/llama-ext.h`。
- `ggml/` 张量库 + backend（只有 `ggml/src/ggml-cpu` 与 `ggml/src/ggml-vulkan`）；调度器在 `ggml/src/ggml-backend.cpp`。
- `common/` 参数/CLI（`arg.cpp`、`common.cpp`）、放置与规划（`fit.cpp`、`fit.h`）、chat/解析（`chat.cpp`、`jinja/`、`parsers/`）、`speculative.cpp`。
- `tools/` `cli`、`server`、`llama-bench`、`mtmd`、`perplexity`、`ui`；`app/` 统一 `llama` 入口，经 `LLAMA_APP_CMD` 分发。
- `tests/` CTest（32 个用例）+ `test-backend-ops.cpp`（不进 CTest，要手跑）。

## 构建与测试

- 本机实际在用的是 MinGW + Ninja（不是 MSVC preset）：

  ```
  cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe \
    -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
  cmake --build build-vulkan -j 8
  ```

- Git Bash 里跑 `build-vulkan/bin/*.exe` 前要 `export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"`，
  否则找不到 libstdc++/libgcc DLL，直接 exit 127。
- Linux/WSL：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON`，`cmake --build build -j`。
  注意 auto-resident 是 `_WIN32` 专属路径，跨平台的性能结论不可直接搬。
- 性能相关一律用 Release/RelWithDebInfo；Debug 推理慢到不可用。
- 测试：`ctest --test-dir build-vulkan --output-on-failure -C Release`；改算子/kernel 先跑
  `build-vulkan/bin/test-backend-ops -o MUL_MAT_ID -b <backend>`。

## 代码与提交规范

- 只用 ASCII：不要 `—`、`→`、`x` 写成 `×`、`...` 写成 `…`；改用 `-`、`->`、`x`、`...`。
- 注释 1-2 行，只写不显然的约定；不复述代码；不按固定列宽硬换行；不要从句子中间断行。
- 先写代码后写注释，只在确有必要处加；先写注释会堆出冗余注释。
- llama.cpp 不用 Minja，其 Jinja 引擎在 `common/jinja`。
- 提交要小而可回退；改 Vulkan/调度/GEMM 的提交必须写明跑了哪些门禁。

## 硬性禁止

- 不确定时尽量少动手；大改动或引入新模式前先读相关文件，**暂停并征求用户确认**。
- 不得用 `GGML_ABORT` 处理容量不足、内存不足、路由过散这类运行期条件。
- 不得把只在一个显存档、一个 bank、一个电源态下成立的结论写进文档。
- 不得为了让数字好看而跳过 `-nkvo`、跳过小显存档或缩短 prompt。

## 参考

- 新增模型：`docs/development/HOWTO-add-model.md`。
- 构建：`docs/build.md`；server：`tools/server/README.md`、`tools/server/README-dev.md`。
- 解析/模板：`docs/development/parsing.md`、`docs/autoparser.md`、`common/jinja/README.md`。
- 可复用工作流：`skills/` 目录。
