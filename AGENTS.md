# Instructions for llama.cpp

**除关键术语等之外，尽可能用中文回复**

> [!NOTE]
> 本仓库是 `elliotmasheal/llama.cpp` fork，基于 ggml-org/llama.cpp 的上游 master。
> `origin` 指向本 fork，未配置 `upstream` remote。

## 本 fork 目标

构建一个本地运行大型 MoE 模型的 runtime：GPU + RAM + SSD 三者协同（专家预取/缓存）、算子优化、MTP。变更要小而可上游化，复用现有基础设施，避免新增子系统。

以Qwen3.6 35B A3B Q4量化为测试模型(F盘LLM目录下：Ornith-1.5-35B-Q4_K_M.gguf)，VULKAN Backend下在8G RX 6700S显卡硬件基础上实现PP1000+/TG200+的第一阶段目标(这是一个肯定可以且必须达到的目标，但是需要你综合多方面知识并不断深入技术底层，因此需要你不畏艰难、不断探索，基于实际PP和TG过程在不同阶段的时间消耗现状，逐项优化甚至是打破传统，另辟蹊径，最终完成第一阶段的优化目标)。

## 现成基础设施（不要重复造）

- MoE 专家 GPU cache：`src/llama-moe-cache.{h,cpp}`；开关 `--moe-cache-mib N`（`cparams.moe_cache_size`）。图接线：`build_moe_cache_slots()` 与 `src/llama-graph.{h,cpp}` 里的 `moe_cache` 字段。
- 专家分层/放置：`--cpu-moe`、`--n-cpu-moe`、`-ot`/`--override-tensor`（`common/arg.cpp`）；自动放置见 `common/fit.cpp`（`llama-fit-params`）。
- MoE 算子：`GGML_OP_MUL_MAT_ID`，kernel 在各 backend `ggml/src/ggml-*`。
- MTP / 投机解码：`--spec-type draft-mtp`；MTP "nextn" 头为 `LLM_TENSOR_NEXTN_*`（`src/llama-arch.cpp`）；draft 循环在 `common/speculative.cpp`；见 `docs/speculative.md`。

## 代码地图

- `src/` 核心库 libllama：加载 `llama-model.cpp`/`llama-model-loader.cpp`/`llama-mmap.cpp`；图 `llama-graph.cpp`；KV/内存 `llama-memory*.cpp`、`llama-kv-cache*.cpp`；采样 `llama-sampler.cpp`；C API `include/llama.h`。
- `ggml/` 张量库 + backend（`ggml/src/ggml-cpu`、`-cuda`、`-vulkan`、`-sycl` 等）。
- `common/` 参数/CLI（`arg.cpp`、`common.cpp`）、chat/解析（`chat.cpp`、`jinja/`、`parsers/`）、`speculative.cpp`。
- `tools/` 命令行工具（`cli`、`server`、`quantize`、`llama-bench` 等）；`app/` 统一 `llama` 入口，经 `LLAMA_APP_CMD` 分发 `serve`/`cli`/`download` 等。
- `tests/` CTest；`conversion/` + `convert_hf_to_gguf.py` GGUF 转换；`pocs/` 微基准。

## 构建与测试

- Windows MSVC/Ninja：`cmake --preset x64-windows-msvc-release`，然后 `cmake --build build-x64-windows-msvc-release`。
- Linux/WSL：`cmake -B build -DCMAKE_BUILD_TYPE=Release`（CUDA 加 `-DGGML_CUDA=ON`），`cmake --build build --config Release -j`。
- 性能相关一律用 Release/RelWithDebInfo；Debug 推理慢到不可用。
- 测试：`ctest --test-dir <build-dir> --output-on-failure -C Release`。
- 改 ggml 算子/kernel 先跑：`build/bin/test-backend-ops -o MUL_MAT_ID`（`-b <backend>` 过滤 backend）。

## 代码与提交规范

- 只用 ASCII：不要 `—`、`→`、`×`、`…`；改用 `-`、`->`、`x`、`...`。
- 注释 1-2 行，只写不显然的约定；不复述代码；不按固定列宽硬换行；用 ASD-STE100 简明英语。
- 先写代码后写注释，只在确有必要处加。
- 不要从句子中间断行，也不要为凑列宽拆行。
- 复用现有基础设施，避免新增子系统或破坏现有行为。
- 大改动或引入新模式前先读相关文件，**暂停并征求用户确认**。
- 写代码/实现新功能前先读 `skills/code-review/SKILL.md`。
- 常见错误：先写注释再写代码会堆出冗余注释；llama.cpp 不用 Minja，其 Jinja 引擎在 `common/jinja`。

## 硬性禁止

- 不确定时尽量少动手。

## 参考

- `CONTRIBUTING.md`；上游 issue/PR 先搜（`gh search issues`、`gh search prs`）。
- 新增模型：`docs/development/HOWTO-add-model.md`。
- 构建：`docs/build.md`；server：`tools/server/README.md`、`tools/server/README-dev.md`。
- 解析/模板：`docs/development/parsing.md`、`docs/autoparser.md`、`common/jinja/README.md`。
- 可复用工作流：`skills/` 目录。
