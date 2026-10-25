# qrt - the qwen36 MoE runtime

This directory holds the runtime that replaces the fork's decode path (design section 3.3).
It links neither ggml nor llama: the runtime parses the GGUF itself and drives Vulkan directly.

## Status: M2-0 scaffold, M2-1 GGUF index, M2-2 Vulkan device and budget

What exists now:

- `qrt_config.*` - the ten switches of design section 7 plus two runtime extensions
  (`--telemetry-path`, `--vram-tier-mib`), their design defaults, strict validation, and
  non-fatal notes for legal-but-costly values.
- `qrt_telemetry.*` - a CSV sink with one row per decode pass, holding the columns
  `bench/README.md` section 2 asks for.
- `gguf_index.*` - read-only GGUF metadata, tensor table and absolute byte offsets, plus
  the byte account `bench/tools/gguf_dims.py` prints. No abort path: malformed files come
  back as an error string.
- `vk_dev.*` - Vulkan device, compute and transfer queues, memory type selection, the
  `VK_EXT_memory_budget` query, the design 5.5 budget planner with a tier cap, and the
  staging buffer with the H2D/D2H copy path.
- `weights.*` - the repacked expert store: `.idx` page table, per-expert byte ranges, reads.
- `qrt_main.cpp` - the `qrt` binary: `--help`, `--print-config`, `--selftest`,
  `--gguf-index`, `--require-tensor`, `--vk-info`, `--vk-alloc-probe`, `--vk-io-check`.
- `CMakeLists.txt` - the `qrt_core` library and the `qrt` executable, plus the
  `qrt_selftest` ctest case.

What does not exist yet: the dense residency and the expert cache (M2-3, in progress),
kernels (M2-4), prefill (M2-5), decode (M3). `runtime/proto/` and `runtime/cpu_moe*` predate
the scaffold and keep their own targets.

## Build

The target is on by default (`LLAMA_BUILD_QRT=ON`); it needs no Vulkan headers.

```
cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan --target qrt -j 8
```

## Run

```
build-vulkan/bin/qrt.exe --help            # switch table with the design defaults
build-vulkan/bin/qrt.exe --print-config    # effective values, one key=value per line
build-vulkan/bin/qrt.exe --selftest        # switches, error paths, telemetry round trip
```

`--selftest` clears the `QRT_*` variables it tests, so it does not depend on the caller's
environment. It is also registered as the `qrt_selftest` ctest case.

Exit codes: `0` success, `2` bad usage or an unusable telemetry sink (nothing has been
allocated or opened, so the caller can fix the flags and rerun), `1` selftest failure.
No code path aborts on a bad value.

Precedence: CLI flag > `QRT_*` environment variable > design default. A flag takes its
value either as `--flag value` or `--flag=value`; the last occurrence wins.

Notes are not errors. Values that are legal but change the design's cost model (for
example `--prefill-chunk 512`, `--kv-precision f16`, `--draft-vocab 0`) print a `note:`
line on stderr and keep running.

## GGUF index

```
build-vulkan/bin/qrt.exe --gguf-index <model.gguf>            # report on stdout
build-vulkan/bin/qrt.exe --gguf-index <model.gguf> --require-tensor blk.0.ffn_gate_exps.weight
```

The report is `#` comment lines and then a CSV table:

```
# gguf version=3 tensors=753 alignment=32 data_offset=10991392 file_size=22853663008
# meta arch=qwen35moe block_count=41 nextn=1 expert_count=256 ...
# account expert_bytes=... expert_layers=41 per_expert_bytes=... dense_bytes=... ...
name,n_dims,ne0,ne1,ne2,ne3,type,type_name,offset,nbytes
```

- `offset` is absolute, `nbytes` is the block-aligned size from the type table.
- The account uses the same definitions as `bench/tools/gguf_dims.py`: names containing
  `_exps` are expert bytes, `token_embd.weight` is separate, everything else is dense.
- Opening reads the header only (about 11 MB for the target model, 107 ms); weights are
  never mapped or read.
- `--require-tensor` returns exit 2 with a message when the tensor is missing, so callers
  get an error instead of a null pointer.

Cross-check against gguf-py, the reader `gguf_dims.py` uses:

```
python bench/tools/gguf_index_check.py [build-vulkan/bin/qrt.exe] [model.gguf]
```

It compares the 753 tensor rows (name, dims, type id, absolute offset, bytes), the 11
hparams, and the account, then feeds nine crafted files (bad magic, truncated table,
out-of-bounds tensor, unknown type, duplicate names, dim overflow, nine dims, oversized
string, nested array) and requires exit 2 with a message for each.

## Vulkan device and VRAM budget

```
build-vulkan/bin/qrt.exe --vk-info
build-vulkan/bin/qrt.exe --vk-info --vram-tier-mib 6144
build-vulkan/bin/qrt.exe --vk-info --vk-alloc-probe 64
```

`--vk-info` opens the device and prints:

- `device`, `heap_mib` (largest device-local heap), `budget_mib` (usable now),
  `usage_mib` (from `VK_EXT_memory_budget`), `tier_mib`, and whether the budget extension
  is present.
- the compute and transfer queue family indices and the compute family's timestamp bits.
- allocation counters: `alloc_count`, `alloc_mib`, `alloc_failures`, `cache_reductions`.
- `plan`: the five buckets of design 5.5, their total, the expert-cache coverage, and how
  many 256 MiB steps the planner had to give up.

Budget rules:

- With `VK_EXT_memory_budget`, `heapBudget` is used as is; it already accounts for other
  processes, so the desktop reserve is not subtracted twice. Without it, the planner
  subtracts a 1000 MiB desktop reserve from the heap.
- `--vram-tier-mib N` (or `QRT_VRAM_TIER_MIB`) caps the budget, which is how the 4G/6G
  tiers of `bench/README.md` are simulated. 0 means no cap.
- Fixed buckets: dense 2079 MiB (RESULTS 25), KV `kv_page_tokens x 6336 B` for q4_0 or
  `x 22528 B` for f16 (RESULTS 25), compute 700 MiB (design 5.5 lower bound), draft 100 MiB.
- If the fixed buckets do not fit, the plan fails with a message (exit 2). Otherwise the
  expert cache takes `min(requested, what is left)`, rounded down to whole MiB, and every
  256 MiB it had to give up is counted in `reductions`.
- `qrt_vk_alloc` refuses an allocation larger than the budget and counts every failure in
  `alloc_failures`; it never aborts, so a caller can shrink the cache and retry.
- On this driver (WDDM) `heapBudget` does not move after a 64 MiB allocation, so the
  planner cannot use budget deltas to measure driver overhead. That needs a real
  allocation of the whole plan (M2-3).

The device opens with Vulkan 1.2 and a timeline semaphore, which is what the single-submit
pass model of design 4.6 needs later.

## Host to device transfers

```
build-vulkan/bin/qrt.exe --vk-io-check
build-vulkan/bin/qrt.exe --vk-io-check --repack F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL
```

`--vk-io-check` proves the staging path before anything depends on it (RESULTS 35):

- a 4 KiB, 64 KiB, 1.9 MB and 64 MiB pattern round trip through one staging buffer, 7
  samples each after a warm-up, `match=1` required;
- an 8 slot pipelined upload of 1.9 MB blocks, which is the shape M2-5 and M3-0 need;
- with `--repack <prefix>`, three experts per layer (123 in total) are read from the
  repacked store straight into the staging mapping, uploaded, read back and compared byte
  for byte.

Rules the implementation follows:

- staging memory must be `HOST_VISIBLE`; `HOST_CACHED|HOST_COHERENT` is preferred because
  the CPU then reads the mapping at PCIe speed instead of 0.21 GB/s (RESULTS 35).
  `pick_host_type` never falls back to a type that cannot be mapped, which is what made the
  first attempt fault (DECISIONS 29/30).
- device buffers need `QRT_VK_USAGE_TRANSFER_DST` to be an upload target and
  `QRT_VK_USAGE_TRANSFER_SRC` to be a download source.
- one copy is one `vkCmdCopyBuffer` plus one submit and one fence wait; that costs a fixed
  0.065 ms, so a single 1.9 MB copy serializes at only 4.7 GB/s. Use
  `qrt_vk_upload_submit` on several staging buffers and wait once for the pipelined rate.
- failures return a message and bump `io_failures`; nothing aborts (design 9).

## Telemetry

`--telemetry 1 --telemetry-path <file>` writes a CSV with this header:

```
pass,pos,k,accepted,hit_experts,miss_experts,miss_cpu,miss_h2d,h2d_bytes,draft_ms,gpu_ms,cpu_leg_ms,pass_ms
```

- `pass` is the decode-pass index, `pos` the position of the first verified token.
- `miss_cpu` and `miss_h2d` split `miss_experts` by the arm that served it, so the
  adaptive split (design 4.7) can be checked against the three-leg cycle.
- The file is line buffered and every row is flushed, so a crash keeps the measured rows.
- A pass with telemetry off calls `qrt_telemetry_pass(NULL, ...)`, which returns after one
  branch; the sink is never opened.
- `--telemetry-path` is a runtime extension, not a design switch: it only says where the
  sink goes. `bench/raw/` is the place for measurement runs.

## Files

| file | role |
|---|---|
| `qrt_config.h/.cpp` | switch table, defaults, parsing, validation, notes |
| `qrt_telemetry.h/.cpp` | per-pass CSV sink |
| `gguf_index.h/.cpp` | GGUF metadata, tensor table, absolute offsets, byte account |
| `vk_dev.h/.cpp` | device, queues, memory types, budget planner, allocator, staging transfers |
| `weights.h/.cpp` | repacked expert store: index page table, expert ranges, reads |
| `qrt_main.cpp` | CLI entry, `--selftest`, `--gguf-index`, `--vk-info`, `--vk-io-check` |
| `CMakeLists.txt` | `qrt_core`, `qrt`, `qrt_selftest` |
