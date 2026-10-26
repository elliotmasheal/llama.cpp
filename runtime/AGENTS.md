# runtime/ 内约定

写本子树前读 `skills/code-review/SKILL.md`；执行闭环见 `docs/agent/cycle.md`，跑数见 `bench/README.md`，判门见 `docs/agent/gates.md`。

## 1. 完成定义

- 改数值路径/内核/精度：L-KERN，对独立 CPU 双精度参考 `rel_L1 <= 1e-6` 且逐位可复现，才允许合入。
- 整模型行为/默认开关：L-MODEL，KLD <= 1e-4 + 逐 token；豁免只按 gates，不自行扩展。
- 内核正确、消费者串联正确、任务验收完成分别举证；编译、`--selftest`、格式校验不能证明模型正确。`--prefill-check` 只计时，不能当输出校验。
- `runtime/` 不走 G-REG；从 fork 拷代码先留源 G-REG 基线，拷入后跑 L-KERN/X-4。检查器 PASS 若阈值更宽或覆盖不全，不得标硬门 pass。

## 2. 实现前与验证后

1. 从真实入口追到算子及参考；核对维序/stride、量化符号/缩放、head 映射、归一化、causal mask、跨 chunk 状态和同步。设计草案不能代替算子语义。
2. 用非对称输入、量化边界、不同 token/head 固定契约。参考从原始输入独立计算；复制被测索引公式或用 GPU 中间量串出的参考，只能诊断，不能验收。
3. 修复配先红后绿反例；测试调用生产入口，不重新上传正确权重绕过加载器，不只检查自己组装的参数。
4. 局部检查后跑消费者串联。状态改动覆盖跨 chunk（含 2048 边界）、连续调用、重置/回滚、修改后缀不影响前缀；适用的检查缺失就补，不延后到无人负责的任务。
5. 模型任务走实际 GGUF -> embedding -> 层链 -> final norm/head -> logits；核实 CLI 参数实际生效。格式通过、跨引擎近似或同引擎 A/B 均不能掩盖两臂共有错误。
6. 所有 `done_when` 条目与适用门禁通过才交付；未测清楚标未验证，不把短输入诊断升级成正式 10.4K/长上下文验收。

## 3. 现有检查入口

下表是定位索引，不是自动覆盖证明；命令/选项先核对实际代码，补充测试沿用现有体系。

| 开关 | 覆盖 |
|---|---|
| `--selftest` | CLI 解析、默认值、坏值回退 |
| `--gguf-index` | 张量表/hparam 对 `gguf_dims.py`；坏文件报错 |
| `--dense-gemm-check` | Q8_0 GEMM 对 CPU；连续 dispatch 独立输出 |
| `--norm-check` / `--deltanet-check` / `--fullattn-check` | norm+Q8_1、DeltaNet 递推、full-attn 与 KV 行精度 |
| `--layer-check` | 一层 norm/投影/注意力/残差/router/专家/输出；含 60,1,2,1 分块/reset/前缀因果回归 |
| `--moe-check` | 专家 gate/down，单槽与层批生产入口 |
| `--logits-dump` / `--model-check` / `--chunk-check` | 40 层整模型 QRTLOG1 出口、CPU 双精度参考、跨 chunk 状态 |
| `--cache-check` | 按层槽位/SLRU/路由 trace 回放与容量退化；不做真实 H2D |
| `--moe-batch-check` | routed-only k=1/2/4/5/8 对 k=1 串行与 CPU 参考 |
| `--prefill-check` | 专家腿和层优先调度计时；不校验输出 |
| CPU 回归（`qrt_checks`） | `qrt_selftest` 与 model/prefill/cache/cli/workflow 契约 |

## 4. 已知陷阱与红线

- 每次 dispatch 使用独立 descriptor set；重写已被命令缓冲引用的 set 会影响之前的 dispatch（DECISIONS 66）。
- `packHalf2x16` 本驱动向零舍入；需要 RNE 就显式实现并测边界（DECISIONS 64.1）。
- 上传源必须活到传输完成，加载后回读；测试重传会掩盖脏缓冲。参考也曾错解 f16 次正规数；双方同错不算证明（DECISIONS 65/67）。
- 实验/优化行为带 CLI 开关，默认资格按门禁；正确性修复不加保留错误语义的开关。
- 运行期容量不足不用 `GGML_ABORT`；走既定慢而正确路径并计数。SSD 不进 decode 单步；不建第二套调度器/缓存/预算器，模块依 design 5.1。
- 代码 ASCII，注释只写非显然原因。提交前跑 `qrt --selftest`、受影响检查与串联回归、`state_check.py`；有提交授权才提交，证据关联实际构建。
