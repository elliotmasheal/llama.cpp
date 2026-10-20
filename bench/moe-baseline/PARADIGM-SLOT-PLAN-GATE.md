# T4 门禁：GPU 侧 slot 规划 —— "miss 由谁上传" 答案是否定的，STOP（2026-10-20，task-26）

> 分支 `moe-gpu-cpu-hybrid` @ `425166f25`。**门禁结论：GPU 侧 slot 规划单独无法过 35 t/s 线，且不存在
> race-free 的机制去掉逐层 host 往返，按任务要求 STOP，未写 shader。**
> 只有"层内上传重叠"（不需要 GPU 规划）还有 ~10 ms/token 的空间，见第 5 节。

## 1. 确定性计数（与机器负载无关，有效）

`-ncmoe 41 --moe-cache-mib 6144 -ub 2048`，`-p "Hello" -n 32`，cache 统计行：
`ubatch <= 8: hits = 5628, misses = 3613, hit rate = 60.90%, uploaded = 6603.28 MiB`

| 量 | 值 |
|---|---:|
| 上传字节 / token | **~206 MiB = 216 MB**（6603 MiB / 32 decode token） |
| 上传时间 / token（13.9 GB/s，完全串行） | **~15.5 ms** |
| 命中率 | 60.9%（30 个 partial 层里每 token 约 113/240 次 miss） |
| 部分驻留层 | 30（10 个整层 bank 无回调） |

## 2. 依赖与时间线（谁上传 layer L 的 miss）

```
token t:
  for L in 0..39:
    GPU: attn(L) -> routing(L) -> ids(L)        # ids 依赖上一层 MoE 输出
    ---- ids(L) 只能在这一刻知道 ----
    判定 miss 需要: ids(L) + bank 常驻表
    host: 读 miss 列表, 发 PCIe 上传 (216MB/32 = ~6.7MB/层)
    GPU: MoE gate(L) 必须等上传完成 -> up(L) -> down(L)
    GPU: routing(L+1) 依赖 MoE(L) 输出
```

关键结论：**L 层的 miss 不能提前判定**（ids(L) 依赖 MoE(L-1) 的输出），**上传源在 host**（GPU 不能
自己发起 PCIe 读，见 Option C），所以 "host 读一个小 miss 列表 + 发上传" 这一步在依赖链上无法消除。

## 3. 各选项的判决

| 选项 | 机制 | 判决 | 依据 |
|---|---|---|---|
| A | GPU 规划，host 每层只读回 tiny miss list（32-64 B）再发上传 | **往返仍在**；相比现状省下的是 "ids 128B→32B + CPU LRU 规划 + 两次 drain 变一次" | 现状每层 copy 相关 ~0.8-1.0 ms（trace: 30 层 23-30 ms），Option A 约 0.6-0.7 ms/层 -> 省 6-9 ms/token |
| B | 一次规划 K 层（4 层一握手） | **不合法** | layer L+1 的 ids 需要 MoE(L) 输出，而 MoE(L) 需要 upload(L) -> 无合法 lookahead；预测方案已被 task-25 否决（recall 38-56%） |
| C | GPU 自己发起 PCIe 读（host 镜像常驻显存可读） | **本机不可能（已确认）** | `ggml_vk_host_buffer_import_usable()` 对 AMD 返回 false（驱动每次提交重缺页所有导入页）；把 host buft 接入 `supports_buft` 的实验直接崩溃（0xC0000005，无输出） |
| D1 | **层内重叠**：gate split 时就把 up/down 权重的上传发给 ring（L9 已验证 ring 安全），让 down 的上传叠在 gate/up 计算上 | **合法、有空间** | 层内 ids 已就绪；上传 216 MB/token 中约 2/3 可被 compute 覆盖 |
| D2 | 减少上传字节 | 不可行 | bank 6144 是 VRAM 上限（L8）；命中率受容量限制 |

## 4. token 时间模型与 TG

用干净窗口的实测值（task-23）：token 37.6 ms（26.6 t/s），其中 GPU 数学 ~12 ms、上传字节等价 ~15.5 ms、
其余 ~10.1 ms 是 30 层的握手/submit（0.34 ms/层，trace 里 30 次 drain 0.41 ms + 回读/发起 0.37-0.6 ms）。

| 方案 | token ms | TG | 过 35 线？ |
|---|---:|---:|---|
| 现状 | 37.6 | 26.6 | - |
| **A（GPU 规划 + tiny miss list）** | 28.6-31.6 | **32-35** | **否（临界/不足）** |
| A + D1（层内重叠 2/3 上传） | 19-23 | 43-52 | 是（乐观），现实 ~25 ms -> 40 |
| 仅 D1（现状路径 + 层内重叠） | 28-30 | 33-36 | 临界 |
| 理论极限（C 可行时：上传全重叠） | 17-18 | 55-58 | 但 C 被驱动堵死 |

门禁判定：**Option A 单独 32-35 t/s，达不到 35 t/s 的停止线，也没有 race-free 机制去掉那一层往返 ->
STOP，不实现 slot-plan kernel。**

## 5. 下一步建议（唯一有 >35 空间的合法机制）

**D1：层内上传重叠**（不需要 GPU 规划、不需要预测）：
1. 在 gate split 的 copy 阶段，一次把该层 gate/up/down 三个权重的 miss 全部发往 ring 槽
   （L9 的 ring + 批规则已证明 race-free；down 的槽在 gate/up 计算期间被填充）；
2. 后续 up/down split 直接使用 ring 槽，不再有自己的 host 回调与 drain；
3. 配合 L10 的 targeted wait 去掉 gate 处多余的一次全设备 drain。
预期：token 37.6 -> 28-30 ms（TG 33-36），叠加握手最小化可到 ~25 ms（TG 40）。
这需要 scheduler 在 gate split 为后续 split 的目标槽提交 uploads（跨 split 目的地预填充），是
L4/L9 的 ring 与 L10 的 wait 原语的组合，风险中等，必须在干净窗口测量。

## 6. 环境状态（重要）

16:13 与 16:35 两次 canonical 校验（cache6144）：
- 16:13: `--list-devices` 显示 dGPU 8176/7378 MiB free，pp512 404.7 / tg32 26.36（干净）。
- 16:35 起：pp512 **139-184** / tg32 **14.2-14.6**（干净值应是 401 / 26.4），持续到本报告写完。
  排除项：dGPU 显存仍报 7378 MiB free；CPU 总量 3.5%；两个 iGPU `llama-server`（PID 800/20452）
  20 秒内 CPU 时间不增长（空闲）；GPU engine 只有 msedgewebview2 1.6%；电源为 AC + 平衡方案。
  未能定位污染源（怀疑其它会话/驱动状态/热降频）。
因此本报告**所有时间数字都来自 12:39-12:50 的干净窗口**（task-23）以及确定性字节计数（第 1 节），
没有用污染窗口的计时。任何时间类验收需在干净窗口重跑。
