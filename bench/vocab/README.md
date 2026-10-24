# bench/vocab: restricted draft vocabulary (M1-3a)

Draft-side candidate set for the MTP draft (M1-3c) and the acceptance measurement (M1-3b).
Size 16384 matches `QRT_DRAFT_VOCAB` in design section 7; the draft output head then reads
about 33 MB per draft step instead of ~0.5 GB for the full 248320-token vocab.
The draft vocab only affects speed: verification always runs on the full vocab, so accepted
tokens stay identical to plain greedy (design section 6.1).

## 1. Files

| file | content |
|---|---|
| `draft-16k.txt` | 16384 lines, one decimal int32 token id per line, ascending, no comments |
| `draft-16k.bin` | the same 16384 ids as little-endian int32 (65536 bytes), same order, no header |
| `coverage.json` | counts, coverage fractions, leave-one-prompt-out folds, sha256 of both vocab files |
| `bin/`, `raw/` | local build and per-prompt token streams, git-ignored |

Runtime contract: the set is an unordered membership set. `draft-16k.bin` loads with

```cpp
std::vector<int32_t> ids(16384);
FILE * f = fopen("draft-16k.bin", "rb");
fread(ids.data(), sizeof(int32_t), ids.size(), f);
```

`draft-16k.txt` is for review and diffing; both files carry the same ids in the same order.

## 2. Model and hardware condition

- model: `F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (arch `qwen35moe`, n_vocab 248320)
- load: `LLAMA_LOAD_MODE_MMAP`, `n_gpu_layers = 0`, CPU only, 8 threads, 4 threads for nothing
  (there is no GPU work in this task)
- power: balanced + AC (same as `bench/env/power.txt`)

## 3. Method

Validation stream (two parts, kept separate for reporting):

1. **prompt stream**: the four 10K prompts tokenized by the model tokenizer alone
   (`prompt10k`, `prompt-chat10k`, `prompt-code10k`, `prompt-multi10k`), `add_special = 0`
   because the GGUF sets `tokenizer.ggml.add_bos_token = false`.
2. **generated stream**: 128 greedy tokens per prompt decoded on CPU from the same prompts.
   Generation stops early on EOG; per-prompt lengths are in `coverage.json`.

Selection (`bench/tools/build_draft_vocab.py`):

1. union of ids seen in both streams = observed set
2. always keep every special token: 27 control + 6 user-defined = 33 ids
   (`<|im_end|>` 248046 is EOS, `<|endoftext|>` 248044 is BOS, plus `<tool_call>`,
   `<think>`, `</think>`, FIM and audio markers)
3. never keep the 243 unused `[PAD...]` ids (GGUF token type 5)
4. include every observed id (7632 distinct, well under 16384)
5. fill the remaining slots with normal tokens in ascending id order

Fill order: this GGUF has **no** `tokenizer.ggml.scores` (Qwen BPE stores a merge table, not
unigram scores), so there is no score to sort by. Ascending token id is the merge-priority
proxy: in BPE the earliest merges (lowest ids) are the most frequent subwords. The fill is
therefore a prior, not a likelihood estimate; M1-3b measures the acceptance that matters.

Result: 33 special + 7659 observed + 8692 score-filled = 16384 ids. Two observed ids were excluded
(one unused `[PAD]` and one duplicate); `observed_distinct_total` is 7661.

## 4. Coverage

| stream | tokens | covered | fraction |
|---|---:|---:|---:|
| prompt stream | 40225 | 40225 | 100.00% |
| generated stream | 143 | 143 | 100.00% |

The first row is 100% by construction: the delivered set contains every observed prompt token.
The generated row is also 100%, but it is **weak evidence**: the greedy continuation stops early
on EOG for chat (1 token) and multi (1), and only 13 for code; prompt10k generates the full 128.
Per-prompt numbers are in `coverage.json` (`gen_coverage_per_prompt`). The real acceptance test
is M1-3b's E, not this coverage proxy.

Caveat: coverage is a token-id coverage proxy, not the MTP acceptance E. E depends on the
draft distribution as well, and is measured by M1-3b.

## 5. Reproduce

Full run (builds the tool, tokenizes, generates, selects, writes coverage):

```
python bench/tools/build_draft_vocab.py \
  --build-dir E:/fastLlama/build-vulkan \
  --vulkan-sdk D:/Scoop/apps/vulkan/1.4.363.0
```

Fast re-analysis from the cached streams under `bench/vocab/raw/` (no model load):

```
python bench/tools/build_draft_vocab.py --reuse-raw \
  --build-dir E:/fastLlama/build-vulkan \
  --vulkan-sdk D:/Scoop/apps/vulkan/1.4.363.0
```

Prompt-only variant (generated coverage marked as not measured):

```
python bench/tools/build_draft_vocab.py --skip-gen \
  --build-dir E:/fastLlama/build-vulkan \
  --vulkan-sdk D:/Scoop/apps/vulkan/1.4.363.0
```

The build command the script runs (static libs from the main tree, MinGW):

```
g++ -O2 -std=c++17 -fopenmp bench/tools/vocab_dump.cpp -Iinclude -Iggml/include \
  -Wl,--start-group build-vulkan/src/libllama.a \
  build-vulkan/ggml/src/ggml.a build-vulkan/ggml/src/ggml-cpu.a \
  build-vulkan/ggml/src/ggml-base.a \
  build-vulkan/ggml/src/ggml-vulkan/ggml-vulkan.a -Wl,--end-group \
  -L<vulkan-sdk>/Lib -lvulkan-1 -o bench/vocab/bin/vocab_dump.exe
```

`vocab_dump` has two modes: `tokens FILE` prints the ids of a UTF-8 file, `gen FILE N` prints
N greedy ids. Both load with mmap and keep `n_gpu_layers = 0`.

Tokenizer metadata (token strings, token types, special ids) is read with gguf-py's reader
module directly from the GGUF; `gguf-py/gguf/__init__.py` needs PyYAML, which is not installed
here, so the script imports `gguf_reader.py` by path. No tensor data is read.

## 6. Limitations

- The generated stream comes from a CPU-only, 8-thread greedy decode. Thread count and machine
  change floating-point reduction order and can flip near-tie argmax tokens, so a re-run
  elsewhere may produce a slightly different generated stream (and therefore a slightly
  different fill, though the observed set is stable). The committed ids and hashes are the
  reference.
- Generation stops at EOG; a prompt that stops early contributes fewer than 128 tokens
  (`gen_notes` in `coverage.json`).
- The prompt token counts here are measured on the raw file bytes with `add_special = 0`.
  They are one token above the "file token" numbers in `bench/prompts/README-m1-2a.md`, which
  derived them by subtracting a fixed 10-token chat template from a `llama-cli` prompt count.
  The difference is the template accounting, not a different tokenizer.
- The generated sample is only 512 tokens from the same four domains as the prompts. Long
  code, JSON, or other tool/agent output can miss more; a corpus-based resample is the M4
  tuning knob, not an M1 deliverable.
