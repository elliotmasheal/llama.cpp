# quality: 质量门工具（X-1）

`G-QUAL` 要的三件事在这里一次做完：**PPL**、**mean KLD**、**same top-1**，最后给一行 `PASS/FAIL`。
不跑模型，只读两份 logits 转储；这样同一个参考基线可以对任意多次运行时改动复用。

```
python bench/tools/quality/qgate.py compare A.bin B.bin     # A 是基线，B 是被测
```

## 1. 口径

| 项 | 定义 |
|---|---|
| 行（row） | 一个"预测位置"：给出该位置的整词表分布，并附带它要预测的 token |
| token 流 | 两份转储必须来自**同一个** token 流；token 不一致直接 FAIL，不做 KLD |
| PPL | `exp(mean(-log p[target]))`，只用转储里有的行，不做 chunk 拼接口径；`log p` 先在**保留词表**上重新归一 |
| mean KLD | 基线分布对被测分布：`sum p_A (log p_A - log p_B)`，先按 **16 nat 地板**截断，**两侧都在保留词表上重新归一**（否则量化噪声会让 KLD 变成 -1e-7 这种负值） |
| 16 nat 地板 | 转储只保留比本行最大值低 16 nat 以内的词；更靠后的尾被夹到地板值。`clamp mass` 报的就是被夹掉后**凭空多出来**的质量，`kept mass` 是保留部分的真实质量 |
| same top-1 | 每个位置逐行 `argmax` 一致的比例 |
| 分叉 | 第一次 top-1 不同；同时报该位置的 `margin = logp_A[top_A] - logp_A[top_B]` |
| PASS | token 流一致 **且** `mean KLD <= 1e-4` **且** 每次 top-1 分叉的 margin `< 0.05 nat`（U-7） |

数字只对同一份转储口径有意义：换 `n_ctx`、换 prompt、换 chunk 数都会改变行的集合，跨口径不可比。
一个位置上的分布是 fp32/量化后的近似值，所以 `mean KLD` 的分辨率下限约 1e-6（见第 6 节自测）。

**为什么 PPL 和 KLD 都要归一**：fork 把 16 nat 以下的词夹到地板值，这在很"尖"的行上会凭空多出
最多 `n_vocab * exp(-16)` 的质量（248320 词表 = 2.8%）。不对齐分母的话，同一份数据转成 rt 格式
再读回来，PPL 会差一个跟夹值质量同量级的常数；归一之后差 0（自测第 5 项）。
代价是**尾部的真实差异被排除在外**：地板以外的词两边都看不见。

## 2. 参考转储格式（fork，`--save-all-logits`）

由 `tools/perplexity/perplexity.cpp` 写出，`--kl-divergence-base` 是同一个开关。

```
0    char[8]  "_logits_"            （8 字节，无 NUL）
8    u32      n_ctx                 写入时的 -c（llama-perplexity 默认 512）
12   i32      n_vocab
16   i32      n_chunk               = min(--chunks, tokens / n_ctx)
20   i32      tokens[n_chunk*n_ctx]
...  行数据，共 n_chunk x (n_ctx - 1 - n_ctx/2) 行，每行 nv 个 u16：
       u16[0:4]  = f32 scale, f32 min_log_prob   （原始字节，小端 float）
       u16[4:..] = 量化词表，nv = 2*((n_vocab+1)/2) + 4
     logp[i] = scale * q[i] + min_log_prob
```

一个 chunk 只写**后半窗口**（位置 `n_ctx/2 .. n_ctx-2`），所以整份转储的行数固定是
`n_chunk * (n_ctx/2 - 1)`；`qgate.py verify` 会用这个式子核对文件大小，对不上直接拒绝。

## 3. 运行时转储格式（rt，留给 M2-7 的 qrt）

```
0    char[8]  "QRTLOG1\0"
8    u32      version = 1
12   u32      n_vocab
16   u32      n_pos                 行数
20   u32      start_pos             第一行对应的绝对 token 下标
24   i32      tokens[n_pos + 1]     覆盖区间的 token，多一个给最后一行当预测目标
...  f32      rows[n_pos][n_vocab]  逐行原始 logits（也可以是 log-prob，读入后统一归一化）
```

`rows[j]` 是"读完 `tokens[0..j]` 之后"的分布，预测 `tokens[j+1]`。
行可以只覆盖一段区间（`start_pos` 不为 0），因为 10K prompt 的整份 fp32 转储是 10 GB 量级；
只需要覆盖与参考转储相同的行，`--align` 会取两边的交集。
`qgate.py convert` 可以把一份 fork 转储转成 rt 格式（只用于工具自测和兼容期）。

## 4. 命令

```
# 单份转储的 PPL（不带基线也能用）
python bench/tools/quality/qgate.py ppl dump.bin

# 对比，输出固定字段 + PASS/FAIL；退出码 0=PASS 3=FAIL 2=格式错误
python bench/tools/quality/qgate.py compare base.bin test.bin
python bench/tools/quality/qgate.py compare base.bin test.bin --json report.json
python bench/tools/quality/qgate.py compare base.bin test.bin --align      # 只比交集
python bench/tools/quality/qgate.py compare base.bin test.bin --kld-max 1e-4 --margin-max 0.05

# 格式自检：文件大小、每行归一化质量、行最大值范围
python bench/tools/quality/qgate.py verify dump.bin

# 合成数据自测（格式 + 指标，应 9/9）
python bench/tools/quality/selftest.py
```

## 5. 生成参考基线的命令

```
powershell -File bench/tools/quality/reference.ps1 -Out bench/raw/x-1/ref-t5.bin -Threads 5
```

脚本只是 `llama-perplexity` 的一层壳：`-dev Vulkan1 -ngl 99 -fa on -ncmoe 41 -c 256 -b 256
--chunks 2 -f bench/prompts/prompt10k.txt --save-all-logits <Out>`。
`-c 256 -b 256` 让 `n_parallel = n_batch/n_ctx = 1`（多序列会写多份行，转储翻倍）。

## 6. 已自测的边界（`selftest.py`，9 项全过）

- fork 往返：量化误差 <= 半步长，地板以下的值被夹到 `<= -16 nat` 且行宽不变。
- rt 往返：归一化误差 < 2e-6；`convert` 后再比，KLD ~1e-9（log-softmax 幂等）。
- 指标对独立 float64 参考：KLD 相对误差 < 0.01%（0.35 logit 噪声下 4.06e-2）。
- 1e-3 logit 噪声 -> KLD 3.4e-7、same top 100%、PASS；0.35 噪声 -> FAIL。
- 坏 magic / 截断 / 负 n_vocab / 篡改 n_vocab / token 流不一致：全部拒绝，不崩。
- 同样的两份转储：KLD 恒等于 0。

## 7. 已知限制

- fork 转储的 KLD 分辨率受 16 nat 地板与 u16 量化限制，`< 1e-6` 的差异读不出来；要更细就得改成 fp32 转储（rt 格式已经是 fp32）。
- 地板把尾部抹掉：两份转储在地板以下都只看到夹值，所以"被测实现把某些低频词的排序打乱"这类错误不会被 KLD 抓到。
- PPL 只覆盖被转储的行（每个 chunk 的后半窗口），不是标准 wikitext PPL；跨口径不可比。
- rt 转储还没实现，阻塞在 qrt 有 logits 出口（M2-7）。工具侧不需要再改。
