# 门禁

## 1. 六道门

| id | 触发/命令 | 通过标准 | 失败动作 |
|---|---|---|---|
| G-CONS | 每个检查点：`python bench/tools/state_check.py` + diff 自审 | 退出 0，本增量边界清楚；代码 ASCII | 修一致性，不以绿灯证明代码可用 |
| G-EVID | 正式跑数：`bench/tools/` 对应脚本 | 必报列齐、status=ok、r >= 3、相对极差 <= 15%、口径齐 | 批次作废，排除扰动后同口径重跑 |
| G-REG | 改 `src/ggml/common/tools`：第 3 节 | 整组 100% 通过 | 修复并重跑，不带失败合入 |
| L-KERN | runtime 数值路径/内核/精度：受影响 `--*-check` | 独立 CPU 双精度参考 rel_L1 <= 1e-6，逐位可复现 | 不合入，继续定位 |
| L-MODEL | 整模型数值/默认开关：X-1 `bench/tools/quality/` | 按 4.2 三层门 A/B/C 同时满足 | 不成为默认，继续修复 |
| G-DEC | 里程碑：design 第 8/9 节 | 阈值全过，证据可复核 | 按 design 停问，不擅自跨门 |

## 2. 覆盖与完成

- `src/`、`ggml/`、`common/`、`tools/` 改动跑 G-REG 整组；`runtime/` 跑适用 L-KERN/L-MODEL，内核回归为 X-4。从 fork 拷入 runtime 前留一次源 G-REG 基线。
- `bench/` 改脚本也要有反例/回归测试；G-CONS 不测试脚本行为。`docs/` 校验预算、引用和行为冲突；`design/` 只由用户改。
- 每项记录**输入/命令/退出码/实测值/权威阈值/覆盖范围**。检查器 PASS、产物格式合法、局部改善不是硬门 pass；缺测试或没运行写未验证，不是 n-a。
- `done_when` 与适用门禁必须分别满足。局部 L-KERN 过可合入，不代表整模型完成；调用者串联、真实入口与受影响状态边界仍须验证。
- task/check 使用的较宽阈值不能冒充硬门通过。参考必须独立：不能复制被测公式，也不能用同错的两臂互证。性能验收不能替正确性。

## 3. G-REG 固化基线

以下为历史基线，不是本次通过证据。使用本 agent 构建目录执行；backend-ops 为 2026-10-23 / 44bd79305 的记录。

- `test-backend-ops -o MUL_MAT_ID -b CPU` = 945/945
- `test-backend-ops -o MUL_MAT_ID_MASK -b CPU` = 4/4；`-b Vulkan1` = 0/0（无此 op）
- `test-backend-ops -o MUL_MAT_ID -b Vulkan1` = 939/939
- `test-backend-ops -o MUL_MAT -b Vulkan1` = 2275/2275
- `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` = 5332/5332
- `ctest --test-dir build-vulkan` = 34/34（M2-0：33 原有 + `qrt_selftest`）

## 4. 默认值与长上下文

L-KERN 通过才允许数值实现合入；L-MODEL 通过才允许成为默认值。未过的实验能力默认关闭并在 `DECISIONS.md` 开关表标注；修复错误语义不保留坏路径。KV 默认档 q8_0 K + q4_0 V 的 U-11 豁免见 `DECISIONS.md#kv-exempt-q8k-q4v`；L-KERN 不豁免。

### 4.1 G-QUAL-LONG

- 触发：KV 精度/窗口/分页改动、M2-7 收口。`qrt --logits-dump` + `qgate.py compare`。
- A 侧为同引擎 f16 KV；跨引擎不判此门（fork 地板 1.5e-2，`RESULTS.md#q-kv1-fork-floor`）。这不能替代基础模型的独立正确性验证。
- 输入 `prompt256k.txt`，fp32 只取尾 512 行；256K f16 A 侧需分页。
- 10K / 128K / 256K 每档 tail mean KLD <= 1e-4、top-1 >= 99%、分叉 margin < 0.05、needle@0.9N 两侧都对；报随长度斜率，只过 10K 不算过。
- U-12：默认 KV 档只报不判；1e-4 判据落在 f16/f16 A 侧与非默认档，不扩展豁免。

### 4.2 三层数值门（U-13，2026-10-26）

L-MODEL 由三层组成，必须同时满足；A/B 管默认数值路径，C 是外部质量锚。

- **A 量化器逐位一致**：Q8_1 与 KV 行格式在边界、次正规、ties、±0、block max 上，CPU 参考
  与 shader 逐位一致；`--norm-check` 的 `qs_bad=0`。A 不允许共享打包决策。
- **B 算术链可证伪**：完整 40 层链、`reseed off`，参考从原始输入独立计算；在共享同一批
  量化字节的前提下 logits rel_L1 <= 1e-4、top-1 100%。逐层隔离读数不算 B；若使用设备平面，
  必须显式记录平面来源与输入 norm 的处理，并在 RESULTS 标注。
- **C 模型质量外部锚**：runtime f16/f16 对 fork f16 的 KLD <= **1.5e-2**（fork 自身路径地板）、
  top-1 >= **85%**；同一固定 prompt/行集，不用 16 行小样本冒充。另报一个无 Q8_1 激活的
  高精度短切片读数。C 的 top-1 margin 只报不判（U-7 是推测接受口径；fork 自身 on/off 也有
  margin >=0.05 的分叉）；当前短切片只覆盖 dense projection，全模型 f32 锚标未验证。
- f16 KV 的 append 用 RNE；`packHalf2x16` 的向零取整会使 A 侧系统性偏低。
- 失败动作：不成为默认，继续修复；不得用共享打包、fork 地板或短样本放行 B。

## 5. 证据纪律

阈值不因失败而改变；不缩正式 prompt、不跳档位、不改 seed。失败留复现命令与排除项。正式性能 r >= 3，极差 > 15% 作废，报中位/极差，不报单次最优。诊断可用短输入和单次实验，但不能宣称通过正式门禁。跨日期/提交/电源/显存/缓存/k/词表口径不可直接比较。
