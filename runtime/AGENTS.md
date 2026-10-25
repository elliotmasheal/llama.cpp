# runtime/ 内约定（作用域层）

进入本子树就适用。这里只放"写 `runtime/` 代码时一直需要"的约定；跑数纪律在 `bench/README.md`，门禁总表在 `docs/agent/gates.md`。

## 1. 门

- 改数值路径、内核、精度：**L-KERN** —— 对 CPU 双精度参考 `rel_L1 <= 1e-6` 且逐位可复现，才允许合入。
- 整模型行为，或要让某开关成为默认值：**L-MODEL**（KLD <= 1e-4 + 逐 token）；未过就默认关闭。
- `runtime/` 不走 G-REG。从 fork 拷代码时，先在 G-REG 上留一次源基线，之后只对拷进来的版本跑 L-KERN / X-4。
- 每次提交前跑 `qrt --selftest` 与受影响的 `--*-check`。**分段全绿不等于串联全绿，串联本身是一个测试。**

## 2. check 清单

| 开关 | 覆盖 |
|---|---|
| `--selftest` | 开关解析、默认值、坏值回退（design 第 7 节的开关表） |
| `--gguf-index` | 张量表与 hparam 与 `gguf_dims.py` 一致；畸形文件返回错误不崩溃 |
| `--dense-gemm-check` | 稠密 Q8_0 GEMM 对 CPU 双精度；含"两个 dispatch 写不同输出"用例 |
| `--dn-weights-verify` | 六个 DeltaNet 权重缓冲加载时回读逐字节比对 |
| `--layer-check` | 一层的逐阶段对 CPU 双精度（norm / 投影 / 注意力 / 残差 / post norm / router / 共享专家 / 路由 MoE / 层输出） |
| `--prefill-check` | 专家腿与层优先调度；预热 pass 之后才计时 |

## 3. 已踩过的坑（写代码前读一遍，别再复现）

1. **descriptor set 不能复用**：命令缓冲只引用 set 对象，改写已记录 dispatch 用过的 set 会改掉所有这些 dispatch。一次 dispatch 一个 set（`RESULTS` 71 / `DECISIONS` 66）。
2. **`packHalf2x16` 在本驱动是向零取整**，不是 RNE。需要正确舍入时显式做 RNE（`DECISIONS` 64.1）。
3. **权重上传要回读校验**：上传曾读到被后续循环覆盖的临时向量（512 B 堆块读 128 KiB）；加载时逐字节回读才抓得到（`DECISIONS` 65）。
4. **预热 pass 不可省**：首次触页会造假加速比，两臂对比前各跑一次预热（`DECISIONS` 35）。
5. **分段测试会蒙住真缺陷**：检查里自己重传一遍权重，把"上传读了脏缓冲"掩盖了；串联才暴露（`DECISIONS` 63.4）。
6. **检查器本身也会错**：一层 1e-3 的"缺陷"实际是检查自己的 f16 次正规数解码错了 2 倍（`DECISIONS` 67）。

## 4. 红线

- 不用 `GGML_ABORT` 处理运行期条件（容量不足、内存不足、路由过散）：降级到慢但正确的路径并计数告警。
- SSD 只做容量层，绝不进 decode 单步关键路径。
- 新行为一律带 CLI 开关 + `DECISIONS.md` 开关表；只靠 env 而不上 CLI 的行为不得成为默认路径。
- 不允许第二套调度器、缓存或预算器；模块必须在 `design` 5.1 登记。
- 代码只用 ASCII（`-` `->` `x` `...`）；注释 1-2 行，只写不显然的约定。
- 数字必须带口径；结论只写进三份 bench 文档，原始产物落 `bench/raw/`。

## 5. 提交前自检

1. `qrt --selftest` 通过；受影响的 `--*-check` 全过；`rel_L1` 与上轮逐位相同或更好。
2. `python bench/tools/state_check.py` 退出 0。
3. 新增代码无全角字符（`git diff` 搜 `[^\x00-\x7F]`）。
4. 提交信息写清：改了什么、跑了哪些门禁、读数出处（slug）。
