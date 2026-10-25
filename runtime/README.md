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
  back as an error string. Tensor bytes are streamed later through the same handle.
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
  0.065 ms, so a single 1.9 MB copy serializes at a few GB/s. Use
  `qrt_vk_upload_submit` on several staging buffers and wait once for the pipelined rate.
- failures return a message and bump `io_failures`; nothing aborts (design 9).

## Dense residency

```
build-vulkan/bin/qrt.exe --dense-check F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

- `qrt_dense_open` puts every non-expert tensor of the GGUF into one device buffer, 256 B
  aligned, streamed in 8 MiB chunks: 629 tensors, 2079.2 MiB, about 0.9 s warm (RESULTS 36).
- `token_embd.weight` (515 MiB) and the `_exps` tensors are excluded. Experts come from the
  repacked store; the design 5.5 dense bucket is 2079 MiB and does not include the embedding,
  so where it lives is still an M2-5 decision.
- `qrt_dense_locate` gives the device offset of a tensor and `qrt_dense_buffer` the buffer,
  which is what M2-5 binds. `qrt_dense_verify` compares evenly spaced probes plus the tail of
  the last tensor against the file and returns the bytes it compared.

## Prefill scheduler

```
build-vulkan/bin/qrt.exe --prefill-check F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL
build-vulkan/bin/qrt.exe --prefill-check F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL --prefill-slots 4 --prefill-layers 8
```

- `qrt_prefill_run` walks layers first and keeps `slots` staging buffers in flight: block `i`
  is uploaded, and by the time block `i - slots` comes back for compute its fence has already
  signalled. Waiting collapsed from 352 to 2 ms per 1024 blocks (RESULTS 37).
- The math is a callback. `--prefill-compute moe` (default) plugs in the int8 GEMM leg in
  `kernels/moe_expert.*`: gate, up, swiglu and down for the block that just landed, all in
  one command buffer. `--prefill-compute probe` plugs in the M1-4a read kernel instead, which
  isolates the scheduler from the math.
- `--prefill-mode layer` (default) uploads a whole layer of experts into one layer buffer and
  runs the layer as a single submit: `qrt_moe_compute_batch` records gate, up, a barrier,
  swiglu, a barrier and down for every expert of the layer into one command buffer. That took
  the compute leg from 2.0 to 3.7 TFLOPS (RESULTS 41). `--prefill-mode perexpert` is the
  earlier one-submit-per-expert shape, kept for comparison.
- `--prefill-group-kib 16384` (default) coalesces consecutive experts into one ~16 MiB copy: 1024
  copies become 128 and a block costs 0.179 ms instead of 0.191 (RESULTS 43). `0` keeps one
  copy per expert for comparison.
- `--prefill-readers N` splits the copy of a group across host threads. Measured at 2, 3, 4 and
  6 readers: no change at all (the read leg stays at 13.6 GB/s), so the default is 1. The read
  and the concurrent H2D copy run at the same rate, which points at shared bandwidth on the
  coherent staging buffer rather than at CPU issue rate (DECISIONS 40).
- `--prefill-chunks C` is how many 2048-token chunks the prompt has. The chunk loop lives inside
  the layer loop, so a layer is uploaded once and every chunk's tokens go through it, which is
  what design 4.5 asks for ("each layer's experts are read once"). At C=5 the 10K prompt expert
  leg takes 2976-2980 ms, 3436 tok/s at 7.09-7.10 TFLOPS, and `prefill_prompt` prints it
  (RESULTS 44). Chunk-major (re-reading every expert per chunk) was about 1190 tok/s.
- `--prefill-pipeline 1` (default) gives the scheduler two layer buffers and two command rings: layer
  L+1 is read and uploaded while layer L is still computing, which took 0.272 to 0.190 ms per
  block (RESULTS 42). `--prefill-pipeline 0` keeps the single buffer for comparison.
- The two layer buffers (about 952 MiB) come out of the decode expert-cache region during
  prefill (U-9): the cache is a decode structure and the two phases never run together.
- In the layer batch every expert gets its own slice of the hidden planes, because a shared
  plane would be overwritten by the next expert's swiglu before this expert's down reads it.
  Expert outputs rotate over eight slices since accumulation into the token rows is not
  implemented yet.
- The Q4_K x Q8_1 GEMM is `kernels/q4k_gemm.comp` (the M2-4 kernel plus base offsets) and
  reaches only ~1.5 TFLOPS at 64 tokens per expert because each expert is a 16 workgroup
  dispatch behind its own submit (RESULTS 39). One submit per layer is the fix that is in.
- The store mixes quant types: gate/up are Q4_K on most layers, down is Q5_K, layers 34/38
  have Q6_K down and layer 39 Q5_K gate/up, so a matrix of another type is skipped and counted
  (`down_skipped`) instead of being read as garbage.
- `--moe-check <prefix>` compares the gate output of one expert against a double-precision
  dequant reference (rel L1 1.6e-7) and exits non-zero above 1e-4.
- A failed slot halves the ring and bumps `slot_reductions`; a read, locate or upload failure
  counts `blocks_failed` and moves on (design 9, no aborts).
- `qrt_prefill_unique_experts` gives the unique experts a chunk of `chunk` tokens touches, so
  the byte volume follows `--prefill-chunk`: at 2048 every one of the 256 experts is hit.
- The shader is compiled by CMake from `proto/qrt-read.comp` and copied next to `qrt`; pass
  `--prefill-shader` to point somewhere else.
- The int8 kernels are `kernels/q4k_gemm.comp` (Q4_K x Q8_1, built with the dot-product
  extension) and `kernels/swiglu_q8.comp`; `vk_dev` enables
  `VK_KHR_shader_integer_dot_product` when the device has it, and `--prefill-compute moe`
  refuses to run when it does not.

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
| `weights.h/.cpp` | repacked expert store (index page table, expert reads) and dense residency |
| `prefill.h/.cpp` | layer-first prefill scheduler, staging ring, read-probe compute leg |
| `kernels/q4k_gemm.*` | Q4_K x Q8_1 int8 GEMM and swiglu+Q8_1 kernels, one submit per call |
| `kernels/moe_expert.*` | one expert's gate/up/swiglu/down as a prefill compute callback |
| `qrt_main.cpp` | CLI entry, `--selftest`, `--gguf-index`, `--vk-info`, `--vk-io-check`, `--dense-check`, `--prefill-check`, `--moe-check` |
| `CMakeLists.txt` | `qrt_core`, `qrt`, `qrt_selftest` |
