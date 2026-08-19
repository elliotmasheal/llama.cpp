# Vulkan backend 约束

- 先阅读 `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 中现有 device 初始化、capability、buffer allocation、graph dispatch、query pool、fence/semaphore 和 `MUL_MAT_ID` 路径，再设计新代码。
- GPU 能力必须运行时探测，包括 vendor/device、driver、subgroup、integer dot、cooperative matrix 和资源 limits。不能把其他 GPU 的 tile 或 extension 假定套到 RX 6700S/RDNA2。
- glslc 编译支持和目标 GPU 运行时支持是两件事。能力不足时必须有清晰的 upstream kernel fallback。
- Transfer/compute 并行、timeline semaphore、ring buffer、descriptor/offset table 等方案必须写清资源所有权、生命周期、等待点、异常路径和退出时的回收。
- GPU 阶段耗时使用已有 Vulkan timestamp/query-pool 机制；CPU wall-clock 只能说明端到端时间，不能替代 GPU kernel 证据。
- MVP 默认只使用单 dGPU。iGPU 多 physical device、SAM/BAR/host-visible VRAM、NUMA 绑定和 peer memory 只能在独立实验和 benchmark 后启用。
- Vulkan 特化不能破坏其他 ggml backend；新增开关应有明确默认值、设备条件和关闭后的行为。
