# C++、ggml 和 llama 约束

- 先阅读相关现有实现，再新增抽象；优先复用 llama graph、ggml backend buffer、scheduler、同步对象和 CTest 基础设施。
- C/C++ 遵守 `CONTRIBUTING.md`、`.editorconfig`、`.clang-format` 和 `.clang-tidy`：4 空格、仓库既有指针/引用风格、snake_case 和小写短横线文件名。
- 尊重 ggml 的 row-major 数据布局、dimension 约定，以及 `ggml_mul_mat` 的转置语义。不要只凭常见 BLAS 直觉重排 tensor。
- 公共 API 使用合适的定宽整数类型；内部接口要清楚区分 token、layer、expert、字节偏移和设备索引。
- 模型 graph、ggml 核心和 backend 特化的正确性影响要分别说明。Vulkan 性能路径不得无条件改变 CPU、CUDA、Metal 或其他 backend。
- 不为一个优化目标引入平行的推理 runtime；新增模块必须说明生命周期、线程模型、fallback 和测试入口。
