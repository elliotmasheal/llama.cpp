# 构建与测试

MinGW + Ninja，不是 MSVC preset。构建目录按 `docs/agent/resources.md` 第 3 节每 agent 一份。

    cmake -B build-<id> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe \
      -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
    cmake --build build-<id> -j 8

- 性能一律 Release / RelWithDebInfo。
- 测试：`ctest --test-dir build-<id> --output-on-failure -C Release`；runtime 的 CPU 契约用
  `ctest --test-dir build-<id> -L qrt`（`qrt_checks` 构建目标含 `qrt` 与三个契约测试程序）。
- Git Bash 里跑 `build-*/bin/*.exe` 前必须 `export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"`，否则找不到 libstdc++ / libgcc DLL，直接 exit 127。
- 编译属于 `machine-quiet`：所需资源不能由其他任务持有；本人当前任务持锁可用，同一任务的计时与编译也必须串行。
- 记录源码版本/未提交 diff、CMake 源目录/配置、构建目标、实际二进制路径；构建退出 0 后运行该产物，不能只凭时间戳证明包含改动。
- 先跑 `qrt --selftest`、受影响检查与真实消费者链；编译不等于验证。失败保留 stdout/stderr/退出码，自行定位环境或代码问题，不拿旧二进制顶替。
- resident 门槛约 `21.28 x 1.25 + 1 = 27.6 GiB` 空闲内存；不足时 mmap，性能不可比。

## 机器事实

- RX 6700S 8 GiB（`-dev Vulkan1`），设备总量 8176 MiB，桌面常占 0.5-1.4 GiB。
- Ryzen 7 6800HS，RAM 40172 MiB（`bench/env/machine.txt`），电源「平衡」（`bench/env/power.txt`，AC）。
- 硬件极限（不可越）：GDDR6 224 GB/s（本机实测读上限 156.5）、PCIe 4.0 x8 15.75 GB/s、DDR5-4800 双通道 76.8 GB/s、RDNA2 FP32 7.9 TFLOPS、SSD 随机读 0.81 GB/s。
