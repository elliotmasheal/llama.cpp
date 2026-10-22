# H2D 链路天花板复测（2026-10-21）

> 目的：确认本机 host->device 上传的真实上限，判定 L8 里 "13.9 GB/s = 96% 上限" 这个结论是否成立，
> 以及 TG 观察到的约 10 GB/s 到底是链路限制还是调度限制。
> 工具：`bench/h2d-bw/`（纯 Vulkan 探针，不依赖 ggml/llama，可独立复现）。
> 口径：以下 GB/s 一律为十进制 1e9 B/s；每行至少 512 MiB 传输量。

## 1. 结论

1. **单方向 H2D 上限 = 14.2-14.3 GB/s**。换任何软件手段都突破不了：SDMA 与 GPU 内核拷贝、
   3 种队列族、chunk 16 KiB x ... x 512 MiB、在途深度 K=1..64、WC/cached、是否保持映射，
   全部落在同一个 14.1-14.3 的平台上。
2. **Windows 报的 "PCIe Gen4 x16" 不是协商结果**。`DEVPKEY_PciDevice_CurrentLinkSpeed/Width = 4/16`
   是 AMD 驱动虚拟化的能力位；实测 14.3 GB/s 是 15.75 GB/s 载荷上限（Gen3 x16 或 Gen4 x8）的 91%，
   与 MPS 256 / MRRS 512 的效率一致。同方向多队列并发不叠加（见第 4 节），排除了"单个 copy engine
   受限、链路还有余量"这一种解释。
3. **L8 的 13.9 数字成立，但含义要改**：prefill 实测边际速率 13.3 GB/s 是上限的 93%，不是 96%；
   剩余空间约 7%，不值得再投人到上传通路上。
4. **全双工是唯一的真实余量**：H2D 与 D2H 同时跑，只要 D2H 落在 *transfer-only* 队列族
   （本机 fam2，flags=0xc），合计 22.2-23.3 GB/s；若 D2H 落在 gfx/compute 队列族，合计塌回 13.7 GB/s。
   也就是说 decode 每层的 ids D2H 回读，在与专家上传共用队列时会直接吃掉上传带宽。
5. **TG 的约 10 GB/s 不是链路，是 chunk 小 + 串行**：4 MiB 逐次 submit+wait = 10.2，
   1.5 MiB 逐次 = 6.8-7.0；同样 chunk 只要把在途深度提到 8，就是 13.0 / 11.4。
   可回收 30-65%。

## 2. 天花板扫描（`h2d-bw.exe`，512 MiB/行）

| 路径 | queue | chunk | 方式 | K | GB/s |
|---|---|---:|---|---:|---:|
| H2D wc/map-freed | gfx | full | bulk | 1 | 14.06 |
| H2D wc/map-kept | gfx | full | bulk | 1 | 14.19 |
| H2D cached/map-kept | gfx | full | bulk | 1 | 14.20 |
| H2D cached/map-freed | gfx | full | bulk | 1 | 14.18 |
| H2D wc/map-kept | xfer | full | bulk | 1 | 14.17 |
| H2D wc/map-kept | gfx | 64 MiB | pipe | 8 | 14.14 |
| H2D wc/map-kept | gfx | 16 MiB | pipe | 8 | 13.88 |
| H2D wc/map-kept | gfx | 16 MiB | pipe | 1 | 12.78 |
| H2D wc/map-kept | gfx | 16 MiB | serial | 1 | 12.74 |
| H2D wc/map-kept | gfx | 4 MiB | pipe | 8 | 12.99 |
| H2D wc/map-kept | gfx | 4 MiB | serial | 1 | 10.19 |
| H2D wc/map-kept | gfx | 1.5 MiB | pipe | 8 | 11.42 |
| H2D wc/map-kept | gfx | 1.5 MiB | serial | 1 | 7.01 |

在途深度扫描（H2D，wc/keep，gfx 队列）：

| chunk | K=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 MiB | 10.19 | 11.48 | 12.67 | 13.10 | 13.40 | 13.46 | 13.37 |
| 16 MiB | 12.72 | 13.30 | 13.57 | 13.83 | 13.98 | 14.04 | 14.02 |

其他通路：D2H 16 MiB pipe K=8 = 13.55，D2H full bulk = 13.94，
D2D full bulk = 74.6 GB/s 拷贝量（约 149 GB/s 显存读写，对上 RX 6700S 的 GDDR6 规格）。

## 3. GPU 内核拷贝能否绕过 SDMA（`kernel-copy.exe`）

compute shader 跨 PCIe 读 host 内存写显存，每个 invocation 取 4 个相距 `stride` 的字，
保证上千条独立读流在途。workgroup 从 16 扫到 4096：

| 通路 | GB/s |
|---|---:|
| H2D，wc->dev，wg=16/64/256/1024/4096 | 14.29 / 14.28 / 14.28 / 14.30 / 14.24 |
| H2D，cached->dev，wg=256/1024/4096 | 14.29 / 14.28 / 14.27 |
| D2H，dev->wc，wg=256/1024/4096 | 13.97 / 13.95 / 13.99 |
| D2D，内核，wg=256/1024/4096 | 72.6 / 71.7 / 74.3 |
| SDMA 对照（同 buffer，bulk） | 14.15 |

**GPU 驱动拷贝与 SDMA 撞在同一堵墙上**（甚至略高 1%）。这意味着"host-direct 权重访问"
这一类范式改造并不能提高带宽，它省掉的是 copy 的调度往返和显存占用，不是字节。

## 4. 并发判定（链路还是引擎）

| 组合 | 合计 GB/s | 各腿 |
|---|---:|---|
| 1 腿 H2D（gfx） | 12.30-13.90 | 单腿 |
| 2 腿 H2D（gfx + fam1 cmp/xfer） | 13.60 | 13.36 + 7.29 |
| 3 腿 H2D（gfx + fam1 + fam2 xfer-only） | 14.09 | 8.89 + 4.76 + 5.07 |
| 2 腿 H2D（fam1 + fam2，不走 gfx） | 13.95 | 9.14 + 7.15 |
| kernel H2D + SDMA H2D（不同队列） | 13.79 | 9.68 + 7.15 |
| H2D(gfx) + D2H(fam1 cmp/xfer) | 13.68 | 13.19 + 6.96 |
| **H2D(gfx) + D2H(fam2 xfer-only)** | **23.28** | 13.52 + 12.14 |
| **kernel H2D(gfx) + SDMA D2H(xfer-only)** | **22.18** | 13.97 + 11.52 |

同方向永远不叠加（13.6-14.1），反方向分到一个真正 transfer-only 的队列族才叠加到 22-23。
本机队列族：fam0 flags=0xf（gfx/cmp/xfer，8 条），fam1 0xe（cmp/xfer，4 条），
fam2 0xc（xfer-only，1 条），fam3 0x40、fam4 0x20 是视频引擎。

## 5. host 侧 memcpy 这一跳的代价（`h2d-bw.exe` 末段）

pageable RAM -> pinned staging -> device，模拟 `sync_staging` 路径
（`ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp:486-507` 的 deferred_memcpy + 一次 vkCmdCopyBuffer）：

| chunk | 深度 K=1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 4 MiB | 4.88 | 10.58 | 10.84 | 9.71 |
| 16 MiB | 5.63 | 11.04 | 10.89 | 10.84 |

单线程 host memcpy 13.1-14.4 GB/s（读+写两倍的 RAM 流量），8 线程 19.0 GB/s。
纯 DMA 在 8 个 memcpy 线程满载时只从 13.84 掉到 12.57 GB/s，说明 DMA 本身不需要 CPU 搬运
（不是驱动内部二次拷贝），但 memcpy 与 DMA 会争 RAM 带宽。

结论：走 staging 的上传实际落在 **10.8-11.2 GB/s**，正好对上运行时观察到的约 10 GB/s 量级；
而 `ggml_vk_host_get()`（`ggml-vulkan-buffers.cpp:368`）的 pinned/import 路径可以整跳跳过 memcpy，
直达 14.2。

## 6. 对本项目的含义

- **PP**：10 GiB/eval 量级的上传已经贴着链路（13.3/14.3 = 93%）。PP1000 的 <=13.9 MB/token
  门槛是硬的，只能减字节或换访问范式，不能指望带宽。
- **TG**：按 14.3 GB/s 算，每 token 若真上传 1.4 GB 活跃专家，下限就是 98 ms/token = 10.2 t/s。
  上传带宽不可能是 TG200 的杠杆；杠杆只有减少每 token 字节（cache 命中、更小专家量化）、
  每次上传换多个 token（投机解码）、或不上传（host 侧算）。
- **可立即吃到的两点**：
  1. decode 的 ids D2H 回读改到 transfer-only 队列族，别和专家上传抢同一预算（第 4 节：
     同队列合计 13.7，分队列合计 23.3）。
  2. 小 chunk 上传把在途深度从 1 提到 8-32（第 2 节：4 MiB 10.19 -> 13.46，1.5 MiB 7.01 -> 11.4）。
     这与 L9 记录的"每次 host-weight copy 前全设备 drain"是同一件事的两面。

## 7. 复现

```
cd bench/h2d-bw
g++ -O2 -std=c++17 -I D:/Scoop/apps/vulkan/current/Include h2d-bw.cpp \
    -o h2d-bw.exe -L D:/Scoop/apps/vulkan/current/Lib -l vulkan-1
./h2d-bw.exe                                   # 512 MiB/行

/d/Scoop/apps/vulkan/current/Bin/glslangValidator.exe -V copy.comp -o copy.spv
g++ -g -O0 -std=c++17 -I D:/Scoop/apps/vulkan/current/Include kernel-copy.cpp \
    -o kc-dbg.exe -L D:/Scoop/apps/vulkan/current/Lib -l vulkan-1
./kc-dbg.exe 536870912 copy.spv                # -O2 版会崩，先用 -O0

powershell -NoProfile -ExecutionPolicy Bypass -File tuner/pcie-link.ps1
```

原始输出：`T1-h2d-ceiling.out`（h2d-bw），`T2-kernel-copy.out`（kernel copy）。
注意 `kernel-copy.cpp` 的 -O2 构建在打印完结果后崩在退出路径上，数值来自 -O0 构建；
-O0 只影响 CPU 侧提交开销，不影响带宽结论（同一 buffer 的 SDMA bulk 在两版一致）。
