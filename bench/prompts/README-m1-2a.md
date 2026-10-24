# M1-2a 非技术语料：对话 / 代码 / 多语言 10K prompt

口径：2026-10-24，分支 `data/m1-2a`，模型 `F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
（Q4_K_M，21.28 GiB），CLI 构建 `b11904-887bd32e7`，`-ngl 0`（纯 CPU，不占 GPU）。
三份文本均为原创，未复制或改写 `prompt10k.txt` / `prompt20k.txt`。

用途：M1-2b 局部性复测（`design` 8.2 / R2：现有语料只有技术类，需按对话 / 代码 / 多语言分类复测）。
本文件只记录语料本身与 token/重复度读数；路由与命中率不属于本任务。

## 1. 文件

| 文件 | 内容 | 行数 | 字节 | 目标 |
|---|---|---:|---:|---|
| `prompt-chat10k.txt` | 两人多轮日常对话（搬家、骑行、书友会、火车、狗、市场、婚礼、烘焙），无代码/公式/手册 | 651 | 41692 | 9500-11000 tok |
| `prompt-code10k.txt` | 一个自洽的 Python 项目 `pantry`（单位换算、配料解析、SQLite 存储、购物清单合并、CLI、pytest 测试），含注释与 docstring | 1153 | 38894 | 同上 |
| `prompt-multi10k.txt` | 中文原创散文 + 西班牙语原创散文（街坊、市场、旧书店、海边、园艺、雪、信件） | 189 | 42789 | 同上 |

三份都是 UTF-8 无 BOM、LF 行尾、文件末尾带换行。`prompt-multi10k.txt` 含非 ASCII 是设计如此。

## 2. token 实测（必须用 `-f`，不是 `-p`）

本 build 的 `-p` 把参数当**字面 prompt**：`-p <文件路径>` 只会 eval 那个路径字符串（实测 34.7 t/s，无 timings 行），
所以这里用 `-f <file>`；`-st` 单轮后退出，`-v` 让 `slot print_timing` 打印 prompt eval 行，
`-n 1` 只为触发 timings，`-ngl 0` 保证不占 GPU，`-t 4` 与 bench 口径一致。

模板偏移实测（chat template 会包一层 user/assistant 标记）：

| 探针文件 | prompt_n | 结论 |
|---|---:|---|
| 内容 `a` | 11 | |
| 内容 `aa` | 11 | `a` / `aa` 各 1 token |
| `the` x 50（空格分隔） | 60 | 50 + 10 |

所以固定模板开销 = 10 token，文件 token = prompt_n - 10。

| 文件 | prompt_n | 文件 token | prompt eval | r |
|---|---:|---:|---:|---:|
| `prompt-chat10k.txt` | 10012 | **10002** | 45636.79 ms / 219.38 t/s | 2 |
| `prompt-code10k.txt` | 10159 | **10149** | 51018.88 ms / 199.12 t/s | 1 |
| `prompt-multi10k.txt` | 9646 | **9636** | 42643.74 ms / 226.20 t/s | 1 |

三次全在 9500-11000 区间。chat 的 r=2 是同一文件改稿前后各一次（9011 -> 10002），
只有 10002 这一版是当前文件；code/multi 各只测了最终版一次（r=1）。
这里的 ms / t/s 是 CPU prefill 读数，只用于证明 token 数，**不是性能数字**，不得写进 `RESULTS.md` 的性能列。

复现（PowerShell；Git Bash 里先把 `/d/Scoop/apps/mingw/current/bin` 放进 PATH）：

```powershell
$env:PATH="D:\Scoop\apps\mingw\current\bin;"+$env:PATH
& E:\fastLlama\build-vulkan\bin\llama-cli.exe `
  -m F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf `
  -f <repo>\bench\prompts\prompt-chat10k.txt `
  -n 1 -c 16384 -lm none -ngl 0 -t 4 -st --simple-io --perf -v *> chat.log
Select-String -Path chat.log -Pattern "prompt eval time" | Select-Object -Last 1
```

其余两份只换 `-f` 的文件名。`prompt_n` 取 `prompt eval time = ... / N tokens` 的 N。

## 3. 唯一性 / 重复度

度量脚本（对每份文件跑一遍；`words` 按空白切分）：

```python
p = 'bench/prompts/prompt-chat10k.txt'
t = open(p, encoding='utf-8').read()
w = t.split()
sh = [' '.join(w[i:i+8]) for i in range(len(w) - 7)]
ch = [c for c in t if not c.isspace()]
sh8 = [''.join(ch[i:i+8]) for i in range(len(ch) - 7)]
print(len(w), round(len(set(w)) / len(w) * 100, 2))
print(len(sh), round(len(set(sh)) / len(sh) * 100, 2))
print(len(sh8), round(len(set(sh8)) / len(sh8) * 100, 2))
```

| 文件 | 空白词数 | 唯一词 % | 8-word shingle 数 | 唯一 8-word shingle % | 唯一字符 8-gram % |
|---|---:|---:|---:|---:|---:|
| `prompt-chat10k.txt` | 8051 | 25.72 | 8044 | 100.00 | 88.45 |
| `prompt-code10k.txt` | 3898 | 40.61 | 3891 | 97.69 | 64.60 |
| `prompt-multi10k.txt` | 4345 | 36.55 | 4338 | 100.00 | 95.06 |

读法：

- 8-word shingle 没有一条重复（chat/multi 100%，code 97.69%），说明三份文本都不循环、不套模板，
  不会把"文本自身重复"混进 M1-2b 的专家命中率。
- chat 的唯一词比 25.72% 明显低于技术类 prompt 的约 40%，这是自然对话的常态
  （the / a / I 这类功能词高频重复），不是质量缺陷；判循环看 shingle，不看唯一词比。
- code 的 97.69% 里那 90 个重复 shingle 来自 `def` / `return` / `self` / `assert` 这类语法骨架，
  属于代码的正常形态。
- multi 的空白切词对中文没有意义（整句被当成一个词），所以额外给了"唯一字符 8-gram = 95.06%"作为
  同口径的重复度指标；按空白算的 36.55% 主要由西班牙语段贡献。

## 4. 局限

1. 模板 10 token 是当前 chat template 下的实测常数；模板一改就要重测，不能沿用。
2. 只测 token 数与重复度。三类文本的路由分布、命中率、E 属于 M1-2b，不在本文件。
3. `prompt-code10k.txt` 是仓库内自洽的示例项目，不是某个真实上游仓库的快照；目的是"连贯的工程代码 + 注释"，
   不是版权素材。
4. 三份文本是单作者原创，风格上比真实语料更整齐；若 M1-2b 的读数与真实流量差很多，先怀疑这一点。
5. 仓库 `core.autocrlf=true`，新检出会把这三份文件转成 CRLF（字节数 41692 -> 42343）。已实测这不影响 token 数：
   `prompt-chat10k.txt` 的 CRLF 版本 prompt_n 仍是 10012，与 LF 版一致（Qwen 把 `\r\n` 当同一个 token）。
   若将来换 tokenizer，需要重测这一个点。
