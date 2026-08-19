# Vulkan shader 与生成链约束

- `.comp`/`.glsl` 源码、`vulkan-shaders-gen.cpp`、`ggml/src/ggml-vulkan/CMakeLists.txt`、SPIR-V 和 generated header 是一条构建链。修改任一环节后，确认 CMake 依赖会触发正确重生成。
- 优先扩展现有 `mul_mm*.comp`、`mul_mm_id_funcs.glsl`、`mul_mat_vec*.comp`、dequant 和 feature-test 机制，不另起一套量化格式或 shader loader。
- `MUL_MAT_ID` 要分别考虑 prefill、batched decode 和 tiny decode；tile、subgroup、coopmat 选择必须由 workload 和设备实测支持，而不是照搬其他型号的结果。
- Q4、Q3、IQ 等权重保持 quantized 表示，不能在 shader 中假定为连续 FP16 数组。Grouped GEMM 应复用已有 dequant/dot 路径。
- 新 extension 同时需要 glslc 编译门槛、运行时 capability 检查和 fallback。shader 结果用 backend ops、CPU reference 或等价结果检查验证。
- 每次 shader benchmark 记录量化类型、tensor shape、PP/TG、batch、设备、driver、shader 变体和 GPU timestamp；不要只报告一个 tok/s。
