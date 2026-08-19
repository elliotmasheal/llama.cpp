# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## 项目定位

这是基于 llama.cpp 和 ggml 的 QwenRuntimeMoE fork，目标是优化以提高特定平台下特定模型的运行速度。根目录的 `优化35B模型运行速度.md` 是本项目的方案基线，目标硬件为 Windows 11、Ryzen 7 6800HS、RX 6700S 8GB 和 40GB RAM，目标模型为 Qwen3.6-35B-A3B，主要后端为 Vulkan。

方案当前是待验证的优化路线，不代表 Expert Cache、异步传输、predictor、autotune 或 fused FFN 已经实现。第一阶段只以单 dGPU Vulkan 为 MVP；iGPU、多 physical device、SAM/BAR、NUMA 和 peer memory 都是后续实验。不要把方案中的预计 tok/s 写成实测结果。

## 开始工作前

按顺序阅读：

1. `优化35B模型运行速度.md`：目标硬件、取舍、Phase 0-10 和指标。
2. `TASKS.md`：当前阶段、依赖、验收门和 benchmark 记录模板，无明确任务时从最近未完成的任务开始。
3. `.claude/rules/*.md`：本 fork 的 Harness 约束，按职责拆分。
4. `CONTRIBUTING.md`、`SECURITY.md`、`.editorconfig`、`.clang-format`、`.clang-tidy`。

规则入口：

- `.claude/rules/00-repository-safety.md`
- `.claude/rules/10-cpp-ggml-llama.md`
- `.claude/rules/20-qwen-moe-model.md`
- `.claude/rules/30-vulkan-backend.md`
- `.claude/rules/40-vulkan-shaders.md`
- `.claude/rules/50-performance-validation.md`
- `.claude/rules/60-task-workflow.md`

每完成一个任务记得更新TASKS.md任务状态，并总结本轮流程中可以积累的知识和经验，固定到项目辅助文件中，为后续任务提供参考。

## 代码架构

- `CMakeLists.txt`：顶层构建，聚合 ggml、llama library、common、tools、examples、tests 和 app。
- `CMakePresets.json`：Ninja 预设；Windows Vulkan 入口是 `x64-windows-vulkan-debug` 和 `x64-windows-vulkan-release`。
- `src/`：llama runtime、公共 API、模型加载、graph 构建、KV/memory 和调度。
- `src/models/qwen35moe.cpp`：Qwen3.5/3.6 MoE 模型入口，包含 Gated DeltaNet/attention 混合 block、routed expert、shared expert 和 MTP graph。改动这里必须保持 routing 和数值语义。
- `ggml/`：tensor library 和各硬件 backend；不要把 Vulkan 特化无条件带入其他 backend。
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`：Vulkan device 初始化、能力探测、buffer/graph dispatch、已有 `MUL_MAT_ID` 路径和 query-pool timestamp 路径。新增 runtime 机制前先查找可复用的生命周期、同步和 fallback。
- `ggml/src/ggml-vulkan/vulkan-shaders/`：GLSL compute shader、dequant 实现、feature tests 和 `vulkan-shaders-gen.cpp` 生成链。源码、generator、CMake 依赖和生成的 SPIR-V/header 必须保持一致。
- `tools/llama-bench/`：PP/TG/PG 性能工具，支持 Markdown、CSV、JSON、JSONL 和 SQL 输出。
- `tools/perplexity/`：模型质量和回归护栏。
- `tests/`：CTest 测试；`test-backend-ops` 用于跨 backend 算子结果检查。
- `ci/run.sh` 和 `ci/README.md`：完整 CI 的本地入口，依赖 Linux/bash 环境，不能在 Windows PowerShell 中直接假设可用。

## 常用命令

以下命令从仓库根目录执行。Windows 多配置生成器通常需要同时传 `--config Release`；`<build>` 表示实际配置目录。

### CPU 构建

```bash
cmake -S . -B build
cmake --build build --config Release --parallel
```

### Windows Vulkan 构建

先安装并进入已配置 Vulkan SDK 的开发终端，确保 `glslc`、Vulkan loader 和 SPIR-V headers 可用：

```bash
cmake --preset x64-windows-vulkan-release
cmake --build build-x64-windows-vulkan-release --config Release --parallel
```

Debug 或需要调试符号时：

```bash
cmake --preset x64-windows-vulkan-debug
cmake --build build-x64-windows-vulkan-debug --config Debug --parallel
```

手工配置也可以使用：

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-vulkan --config RelWithDebInfo --parallel
```

Vulkan backend 的 CMake 会探测 glslc 对 cooperative matrix、integer dot、bfloat16 等扩展的编译支持。编译器支持不等于目标 GPU 运行时支持，运行时仍须检查 device capability。

### CTest 和单个测试

```bash
ctest --test-dir <build> --config Release --output-on-failure
ctest --test-dir <build> --config Release --output-on-failure -L main
ctest --test-dir <build> --config Release --output-on-failure -L model
ctest --test-dir <build> --config Release --output-on-failure -R 'test-backend-ops'
```

涉及模型下载、生成模型或特定 backend 的测试要先确认模型、设备和依赖已经准备好。构建完成后，`<build>/bin`（多配置生成器可能是 `<build>/bin/Release`）中的 `test-backend-ops` 可用于 CPU/Vulkan backend 对照；具体参数以该程序的 `--help` 为准。

### 性能和质量验证

```bash
<build>/bin/llama-bench -m <model.gguf> -p 512 -n 128 -ngl 99 -o json
<build>/bin/llama-bench -m <model.gguf> -p 512 -n 0 -b 128,256,512 -o csv
<build>/bin/llama-bench -m <model.gguf> -p 0 -n 128 -t 1,2,4,8 -o json
<build>/bin/llama-perplexity -m <model.gguf> -f <text-file>
```

`llama-bench` 的 `-p` 是 prompt processing，`-n` 是 text generation，`-pg` 可组合两者；`-b/-ub` 控制 batch/ubatch，`-ngl` 控制 GPU layers，`-ncmoe` 控制 CPU MoE 层数，`-ctk/-ctv` 控制 KV cache 类型。优化报告必须保存原始 JSON/CSV，并记录 commit、driver、SDK、模型量化、context、设备、build flags、重复次数和回退情况。

Phase 0 冒烟测试的原始结果保存在 `tmp/phase0-smoke.json`。它只覆盖单 dGPU Vulkan、Q4_K_XL、MTP OFF、PP16 + TG1、单次重复，用于验证模型加载、设备选择和输出格式，不代表完整性能基线，也不能作为优化收益。完整 Phase 0 原始结果应保存在 `tmp/phase0/`，环境记录在 `tmp/phase0/environment.txt`，Vulkan 设备摘要在 `tmp/phase0/vulkaninfo-summary.txt`。

MTP 使用现有 `docs/speculative.md` 中的接口，例如 `--spec-type draft-mtp` 和 `--spec-draft-n-max`。任何 MTP 结果都要和 MTP OFF 在相同 workload 下对比；acceptance rate 不能单独作为吞吐结论。

### 格式和只读检查

```bash
git diff --check
```

C/C++ 使用 `.clang-format` 和 `.clang-tidy`；Python 检查遵守 `.flake8`、`pyrightconfig.json`、`mypy.ini` 和 `ty.toml`。不要为了运行检查而恢复当前工作树中已经被用户删除的 CI 或配置文件。

## 工作边界

- 先完成 `TASKS.md` 中的 baseline 和 instrumentation，再改变调度、内存或 shader。
- Expert cache 的 key 至少包含 `(layer, expert)`；prediction 只能用作 prefetch hint，不能跳过 router。
- 复用现有 quantized Vulkan kernel 和 `MUL_MAT_ID` 生成链；不要把量化权重当作 FP16 数组。
- 每个性能优化都必须有 GPU timestamp、正确性/perplexity 对照和 upstream fallback。
- 先运行 `git status --short --untracked-files=all`，只改当前任务声明的文件。保留既有删除和未跟踪 worktree，不使用 `git restore` 清理它们。
- 不自动 commit、push、创建 PR、发布 issue 或代写 PR/commit/reviewer 文案；遵守 `CONTRIBUTING.md` 的 AI 披露要求。

## 外部 agent 配置

当前工作树中的 `.gemini/settings.json` 也处于用户删除状态。不要读取或自行重写外部 agent 配置。若需要导入可导入项，请由用户使用 `/import` 扫描后，再使用扫描结果给出的 `/import --yes=<digest>`；若当前界面不支持 `/import`，请用户在终端运行 `claude import`。
