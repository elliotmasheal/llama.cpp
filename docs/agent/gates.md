# 门禁

## 1. 六道门

| id | 触发条件 | 命令 | 通过标准 | 不通过怎么办 |
|---|---|---|---|---|
| **G-CONS** | 每轮 | `python bench/tools/state_check.py`；`git diff` 自查 ASCII 与注释规范 | 退出码 0，diff 只含本轮改动 | 不提交 |
| **G-EVID** | 每轮有读数 | `bench/tools/` 里的统一脚本 | 必报列齐全、`status = ok`、r >= 3 且相对极差 <= 15%、口径头齐全 | 该批作废，重建口径后重跑 |
| **G-REG** | 改 `src/` `ggml/` `common/` `tools/`（含从 fork vendoring 进这些目录） | 见第 3 节六条 | 全部 100% 通过 | 回退改动，记未结项 |
| **L-KERN** | 改 `runtime/` 的数值路径、内核、精度 | 该子树的 `--*-check`（见 `runtime/AGENTS.md`） | 对 CPU 双精度参考 `rel_L1 <= 1e-6` 且逐位可复现 | 不许合入 |
| **L-MODEL** | 改整模型数值路径，或要让某开关成为默认值 | X-1 质量门（`bench/tools/quality/`） | 整模型 KLD <= 1e-4 + 逐 token 一致 | 不许成为默认值 |
| **G-DEC** | 里程碑 | design 第 8/9 节 | 阈值全过且证据可复核 | 停，问用户 |

## 2. 目录归属（决定跑哪道门）

| 目录 | 门 |
|---|---|
| `src/`、`ggml/`、`common/`、`tools/` | G-REG（改了就要整组重跑） |
| `runtime/` | L-KERN；要成为默认值另需 L-MODEL；内核回归是 X-4 |
| `bench/` | 无门，只有 G-CONS 的 lint 与 G-EVID 的口径 |
| `design/`、`docs/` | 无门；`design/` 只由用户改 |

从 fork 拷代码时：先在 G-REG 上留一次源基线，之后只对拷进 `runtime/` 的版本跑 L-KERN / X-4。

## 3. G-REG 的固化数字

`build-vulkan/bin/` 下；backend-ops 是 2026-10-23 / 44bd79305 的记录。动 ggml 或 backend 前必须整组重跑；只改 `common/` 时至少重跑 ctest。

- `test-backend-ops -o MUL_MAT_ID -b CPU` = 945/945
- `test-backend-ops -o MUL_MAT_ID_MASK -b CPU` = 4/4；`-b Vulkan1` = 0/0（Vulkan 侧没有这个 op）
- `test-backend-ops -o MUL_MAT_ID -b Vulkan1` = 939/939
- `test-backend-ops -o MUL_MAT -b Vulkan1` = 2275/2275
- `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` = 5332/5332
- `ctest --test-dir build-vulkan` = 34/34（M2-0 实测；33 原有 + `qrt_selftest`）

## 4. 为什么质量门是两级

旧条文把改数值路径一律判不许合入，而工作全在 `runtime/` 内核里。所以拆开：

- **L-KERN**：内核正确性。过了就**允许合入**。
- **L-MODEL**：整模型保真。过了才**允许成为默认值**。

未过 L-MODEL 的能力默认关闭并在 `DECISIONS.md` 开关表标注。**豁免只有一份**：`U-11`
（KV 默认档 q8_0 K + q4_0 V，`DECISIONS.md#kv-exempt-q8k-q4v`）；**L-KERN 不豁免**。

## 4.1 长上下文质量门（G-QUAL-LONG，2026-10-25 用户要求）

- 触发：改 KV 精度 / 窗口 / 分页；M2-7 收口。命令 `qrt --logits-dump` + `qgate.py compare`。
- A 侧必须是**同引擎 f16 KV**；跨引擎禁止（fork 地板 1.5e-2，`RESULTS.md#q-kv1-fork-floor`）。
- 数据 `prompt256k.txt`；rt fp32 只取尾部 512 行（全量 65 GB 不可行）。
- 判据：10K / 128K / 256K 三档，每档 tail mean KLD <= 1e-4、top-1 >= 99%、分叉 margin < 0.05、
  needle@0.9N 两侧都对。**只过 10K 不算过**，报随长度的斜率。
- **默认 KV 档只报不判（U-12）**：读数照报，1e-4 判据落在 A 侧（f16/f16）与非默认档上。
- 256K 的 f16 A 侧装不下，必须分页（M2-6a 已实现，窗口按预算推导）。

## 5. 门禁纪律

- 门禁数字由 design 或本文件规定，不得为过门改判据、缩 prompt、跳档位、改 seed。
- 失败必须留复现命令与已排除项；"配置不对"不是结论。
- 方差：同口径 r >= 3，相对极差 > 15% 该批作废；报中位与极差，禁报单次最优。
- 跨口径比较视为错误（口径含：日期、提交、电源态、显存档位、专家缓存 MiB、k、草稿词表）。
