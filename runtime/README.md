# qrt - the qwen36 MoE runtime

This directory holds the runtime that replaces the fork's decode path (design section 3.3).
It links neither ggml nor llama: the runtime parses the GGUF itself and drives Vulkan directly.

## Status: M2 prefill, model diagnostics, a k=1 decode slice and the speculation cycle

What exists now:

- `qrt_config.*` - the ten switches of design section 7 plus two runtime extensions
  (`--telemetry-path`, `--vram-tier-mib`), their design defaults, strict validation, and
  non-fatal notes for legal-but-costly values. Parsing is atomic: a rejected value leaves
  the configuration unchanged, an explicit CLI value overrides even an invalid environment
  value, and the command line is validated against the same bounds as the structure.
- `qrt_cli.*` - the command table and dispatch options. Every flag is strict (`strtoll`,
  range checked), commands are mutually exclusive, `--flag=value` works for all flags, and
  an unknown or stray argument is rejected before any allocation.
- `qrt_telemetry.*` - a CSV sink with one row per decode pass, holding the columns
  `bench/README.md` section 2 asks for.
- `gguf_index.*` - read-only GGUF metadata, tensor table and absolute byte offsets, plus
  the byte account `bench/tools/gguf_dims.py` prints. No abort path: malformed files come
  back as an error string. Tensor bytes are streamed later through the same handle.
- `vk_dev.*` - Vulkan device, compute and transfer queues, memory type selection, the
  `VK_EXT_memory_budget` query, the design 5.5 budget planner with a tier cap, the
  staging buffer with the H2D/D2H copy path, and the pass command-buffer recorder
  (`qrt_vk_cmd`) the layer chain submits once per half.
- `weights.*` - the repacked expert store: `.idx` page table, per-expert byte ranges, reads.
- `layer.*`, `ffn.*`, `model.*` - the layer chain, FFN half and whole-model logits chain.
- `expert_cache.*` - per-layer slot plan, SLRU/LRU/static, trace replay and the H2D callback
  contract (M3-0); `moe.*` - routed-only k=1..8 batch over the same kernels (M3-1).
- `spec.*` - the speculation decision (longest accepted prefix, the verified token at the
  first mismatch) and the rollback of the layers that carry it (M3-3).
- `qrt_main.cpp` - the `qrt` binary: checks, model logits, cache plan and MoE batch.
- `CMakeLists.txt` - `qrt_core`, `qrt`, `qrt_checks` and the CPU contract tests.

What does not exist yet: the MTP draft (the `spec` module verifies and rolls back, it does not
draft yet), the pre-recorded per-k command buffers, GPU-side route/cache lookup and indirect
expert dispatch, and the GPU head/argmax. The layer chain, the prefill scheduler and the
persistent decode session all run; prefill at 2048-token chunks across 40 layers still needs
the shared scratch arena, so `--generate` caps the resident chunk at 64.

## Routed MoE batch (M3-1)

```
build-vulkan/bin/qrt.exe --moe-batch-check <model.gguf> --layer-store <prefix> [--moe-layer 0]
```

- Drives one layer's router -> top-8 -> gather -> per-expert gate/up/swiglu/down -> weighted
  accumulate through the existing kernels for k=1, 2, 4, 5 and 8. `routed_only` skips the
  shared expert and the final residual, so the token count no longer has to be a multiple
  of 32; the full layer/prefill path keeps them and is unchanged.
- Each k is checked against k=1 serial runs of the same tokens (ids and MoE output) and
  against one CPU double reference (`qrt_ffn_ref_cpu`); the union expert count is compared
  with the union of the serial top-k ids.
- The expert row blocks are padded to 8 rows per used expert (the K-quant token tile), so a
  k<=8 decode batch is weight-read bound instead of the old 32-row compute-bound shape. This
  is the correctness path; M3-2 still needs grouped/indirect dispatch and a persistent model.
  The full prefill path keeps its 32-token layer chunks and 64-row expert blocks.

## Expert cache (M3-0)

```
build-vulkan/bin/qrt.exe --cache-check <store-prefix> [--cache-trace <trace.bin>] \
    [--cache-decode 128] [--expert-cache-mib 4198] [--cache-policy slru|lru|static] \
    [--cache-protected 20]
```

- Slot bytes follow the layer: the maximum expert block in that layer (RESULTS 35).
  The single expert-cache budget keeps one floor slot per layer and splits the rest in
  proportion to the trace's cold-cache miss traffic divided by slot bytes. Missing or
  short trace layers get the floor only; `mixed_layers` counts layers whose experts are
  not all the same size.
- SLRU is per layer (probation/protected, default protected segment 20%; measured on
  the two M1 traces). `lru` and `static` are comparison arms; static fills slots and then
  reports no-slot misses instead of evicting.
- `--cache-check` replays the last `--cache-decode` rows of each routing trace with no
  IO callbacks, so `mode=policy_only`, `h2d_uploads=0`, `miss_mib` is the H2D volume the
  plan would request, and `no_slot` is the CPU-leg remainder. The upload/wait scheduling
  contract is covered by `qrt_cache_contract` (success, upload failure, wait failure,
  per-layer pending guard, cross-layer overlap).
- M3-2 supplies the real callbacks: read the expert into staging, `qrt_vk_upload_submit`
  to `slot_offset`, `qrt_vk_staging_wait` before use, and allocates one device buffer of
  `plan.bytes`. The cache never allocates VRAM and never reads the store.
- A budget below one slot per layer degrades: zero-slot layers report every lookup as
  `no_slot`; no path aborts. `--expert-cache-mib 0` is the zero-cache counterexample.

## Diagnostic workflow

```
python -B bench/tools/quality/qgate.py run --qrt build-vulkan/bin/qrt.exe \
    --build-dir build-vulkan --model <model.gguf> --store <repack-prefix> \
    --tokens bench/raw/m2-7a/tokens10k.txt --output-dir bench/raw/<new-dir> \
    --rows 16 [--baseline <earlier same-token dump> --align]
```

The workflow refuses to overwrite an existing output directory, runs build, the `qrt`
CTest label, `--selftest`, `--print-config`, `--gguf-index`, the 40-layer logits dump and
`qgate.py verify` in order, stops at the first non-zero exit, and records every command,
exit code and log path in `workflow.json`. With `--baseline` it adds `qgate.py compare`;
its result is a diagnostic comparison, not a model or performance acceptance. The same
exit code (3) is preserved for a quality failure. `--chunk` defaults to 2048, so a prompt
whose length is not a multiple of 32 is processed as a short final chunk instead of being
truncated.

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
example `--prefill-chunk 512`, `--kv-k f16`, `--draft-vocab 0`) print a `note:`
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
- Fixed buckets: dense 2079 MiB (RESULTS 25), compute 700 MiB (design 5.5 lower bound), draft
  100 MiB, and the KV window. Since M2-6a the window is derived, not configured: it is
  `min(ctx 262144, whole pages of (budget - the other buckets - the expert cache target))`, where
  one page is `QRT_KV_PAGE_TOKENS x 11 x 2 x (k_row + v_row)` (row bytes q8_0 272, q4_0 144,
  f16 512 at head_dim 256; the default pair is 9152 B/token). A bigger card gets a bigger window
  up to the context; the expert cache keeps its requested size, the window takes the headroom.
- If the fixed buckets do not fit, the plan fails with a message (exit 2). Otherwise the
  expert cache takes `min(requested, what is left)`, rounded down to whole MiB, and every
  256 MiB it had to give up is counted in `reductions`. When not even one KV page fits, the
  cache gives one up instead of failing.
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
- `--moe-check <prefix>` compares one expert against a double-precision dequant reference in
  the token-major layout the kernels write: the gate matrix (rel L1 1.6e-7), the down matrix,
  and the whole gate -> silu*up -> Q8_1 -> down chain through both `qrt_moe_compute` and
  `qrt_moe_compute_batch` (5.9e-08..1.5e-05, all four gates 1e-4, RESULTS 74). The chain legs
  call the production entry points, so a missing `token_major` turns the check red instead of
  passing silently in `--prefill-check`, which only times that leg.
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

## DeltaNet recurrence (M2-6c)

```
build-vulkan/bin/qrt.exe --deltanet-check F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --deltanet-layer 0 --deltanet-tokens 256
```

- Two variants, chosen with `qrt_dn_set_recur()`; both keep the token loop inside the
  shader, so a batched chunk and a token-by-token decode are the same operations in the
  same order (bitwise identical, now checked for both, RESULTS 63).
  - `cols` (the default): a workgroup owns 32 columns of one head, a thread owns 32 rows
    of one column, and the four slices of a column meet in shared memory (the same idiom as
    the gate kernel). 4.75 -> 1.23 us per token.
  - `serial` (kept for comparison): one workgroup owns two state columns of one v-head,
    which means 64 workgroups per head and two subgroup reductions per token.
- The kernel takes q (already L2 normalised), k, v, decay (already exp'd) and beta; the prep
  chain (conv1d(4)+silu, the alpha/beta/dt projections) and the gated RMS norm with `ssm_out`
  are the next step, so the check feeds random unit q/k and N(0,1) v with the real
  `ssm_a`/`ssm_dt` for the decay.
- Speed: 1.23-1.26 us per token for the recurrence, i.e. 0.38 s for the 30 linear layers of a
  10K prompt against 1.35 s for the whole attention budget (RESULTS 63, DECISIONS 61).
- `qrt_dn_weights_load()` reads every weight buffer back and compares it with the host copy
  before returning; a mismatch names the buffer and the byte count. The conv matrix needed
  this: it was uploaded from a scratch vector the gate weights had already overwritten, a
  128 KiB read out of a 512-byte allocation (RESULTS 70, DECISIONS 65).
- **The lane group must not be hard coded to 32.** This device reports subgroup size 64;
  assuming 32 reduces across 64 lanes, which is silently wrong (rel_L1 1.036) rather than a
  crash. `qrt_vk_dev` now carries `subgroup_size` and `has_subgroup_arithmetic`, the value is
  passed as a specialization constant, and the kernel refuses shapes it cannot tile. The
  `cols` variant is only built for `head_dim % 32 == 0`; otherwise the serial one is used.

## RMS norm and Q8_1 pack (M2-6, first piece of the layer chain)

```
build-vulkan/bin/qrt.exe --norm-check F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
build-vulkan/bin/qrt.exe --norm-check <gguf> --norm-tensor blk.0.attn_norm.weight --dense-tokens 512
```

- `--norm-check` runs the three steps the layer chain will run in this order and gates all
  three: the normalised row against a double reference, the Q8_1 planes against the same
  quantiser, and the dense projection computed from the device planes
  (`--dense-tensor`, default `blk.3.attn_q.weight`; `--norm-tensor`, default
  `blk.3.attn_norm.weight`; `--dense-tokens`, default 10240).
- Output layout is the one `kernels/gemm.h` documents: `qs` is 32 int8 per 32-element block
  in uvec4 pairs, `ds` is one word per block with the f16 delta low and `sum(q) + 32768`
  high, so the GEMM reads the planes without a repack.
- **Never use `packHalf2x16` for a delta here.** This driver lowers it to `v_cvt_pkrtz`,
  which rounds toward zero: half the deltas came out one half ULP below ggml's
  `quantize_row_q8_1` before the shader started converting with an explicit
  round-to-nearest-even sequence (RESULTS 69, DECISIONS 64.1).
- Speed at 10K tokens: 1.37-1.72 ms per call (0.134-0.168 us per token, 111-140 GB/s moved),
  512 tokens is launch bound at 0.19 ms. Two norm sites per layer cost about 0.12 s of a
  10K prompt, i.e. under 1% of the PP budget.
- The caller owns every buffer, so the chain passes the same planes to the GEMM; the module
  only owns its pipeline and two command rings.

## Layer chain (M2-6)

```
build-vulkan/bin/qrt.exe --layer-check <gguf> --layer 3 --layer-tokens 256
build-vulkan/bin/qrt.exe --layer-check <gguf> --layer 0 --layer-tokens 10240
build-vulkan/bin/qrt.exe --layer-check <gguf> --layer 3 --layer-tokens 512 --layer-store <prefix>
```

- `runtime/layer.*` runs one transformer block in the order the model runs it: input RMS norm,
  the q8_0 projections, the attention path (`qrt_dn_layer_run` for a DeltaNet layer,
  `qrt_full_qk` + `qrt_full_attn` otherwise), the Q8_1 packing of the attention output, the
  output projection, the residual add, the post-attention norm, and with `--layer-store` the
  FFN half (`runtime/ffn.*`: router, top-k, gated shared expert, routed experts from the
  repacked store) plus the final residual add. The projections write token-major output because
  the attention modules read [token][channel] and a transpose would cost 0.3-0.7 GB per layer
  at 10K.
- `--layer-pass on` (default) records the layer into a shared command buffer (`qrt_vk_cmd` in
  `vk_dev.*`): one submit for the attention half and the router, one for the FFN. The router's
  top-k has to reach the host before the expert GEMM grids can be recorded, so the FFN costs a
  second submit (DECISIONS 68). `--layer-pass off` keeps the module-by-module path as the A/B.
- The FFN slot holds the whole layer and persists across the chunk loop: `qrt_ffn_group` skips
  experts already resident and uploads a contiguous run of missing experts as one H2D copy.
  `model_upload` prints the cumulative MiB per layer; on a 10.4K prompt the first chunk of each
  layer loads the layer once and the following chunks add 0 unless a new expert is routed. The
  counter is the evidence for the layer-first read-once rule; it is not a performance gate.
- `--layer-ffn on|off` (default on when `--layer-store` is given), `--dn-recur serial|cols`
  (default cols, the 32-column recurrence from RESULTS 63).
- `--layer-check` compares each stage against one CPU double chain. The reference for a stage
  is fed the *device's* previous stage output, so the check validates the wiring; each kernel
  keeps its own check. For a full-attention layer the reference reads the KV cache (device
  planes in direct mode, the host rows when paged) and dequantises it with the same f16
  rounding the attention kernel stages, so the row precision is shared instead of landing in
  the gate (M2-6a). The
  FFN reference covers the router and the shared expert over the first 64 tokens and the routed
  experts over their pairs, and reuses the K-quant dequant order of `--moe-check`.
- KV cache (M2-6a): `QRT_KV_K` / `QRT_KV_V` pick the row format (f16, q8_0, q4_0; default
  q8_0 + q4_0), `QRT_KV_PAGE_TOKENS` is the swap granularity, and the resident VRAM window is
  derived from the budget (`qrt_vk_kv_window_tokens`, capped at the context). When the window
  is smaller than the context the cache pages: the authoritative rows live in host memory and
  the attention runs one dispatch per page, carrying the running softmax state. `--vk-info`
  prints `kv_tokens` and `alloc_peak_mib`; `--fullattn-check --fullattn-resident R
  --fullattn-page P` exercises the paged path (0 = direct) and prints `full_paging`.
- Cost at 10K (two submits, the expert store read counted apart): 95 ms attention + 93 ms FFN
  for a DeltaNet layer, 328 + 94 ms for a full-attention layer. The 464 MiB expert read per
  layer is host work the prefill scheduler overlaps (M2-5).
- QRT_LAYER_STOP_PROJ=1 stops after the projections; it bisects the projection readings of
  RESULTS 72 and changes nothing by default.
- Still missing for M2-6: the router tile is far from its roofline (the activation tile is
  re-read once per row block), and GPU-side grouping plus indirect dispatch is what a true
  one-submit layer needs.

## Whole model run and the reference chain (M2-7a)

```
build-vulkan/bin/qrt.exe --logits-dump <gguf> --logits-tokens <ids> --logits-out <bin>
                         --logits-layers 40 --logits-rows 16 --layer-store <prefix>
build-vulkan/bin/qrt.exe --model-check <gguf> --logits-tokens <ids> --logits-rows 16
                         --logits-out <bin> --model-check-ref <bin> --layer-store <prefix>
                         --kv-k f16 --kv-v f16 [--model-check-reseed on|off]
```

- `runtime/model.*` runs the real chain: `token_embd` rows on the host (q8_0, one row per
  token), then the 40 blocks in order - one layer open / set_input / zero_state / run / read
  back / close at a time, so only one layer's weights are resident - then `output_norm` and
  `output.weight` on the host in double. It writes QRTLOG1 v1
  (`bench/tools/quality/README.md` section 3): 24 B header, `tokens[n_rows + 1]`, f32 rows.
- `--model-check` runs the same device chain beside one CPU double chain
  (`qrt_layer_ref_cpu` in `layer.*`), which recomputes the Q8_1 planes, the KV rows and the
  expert legs from the host copies. Unlike `--layer-check` no stage reads a device buffer
  back, so the two arms are independent and the residual stream can be carried across layers.
  Only the f16 KV arm is referenceable: a q8_0/q4_0 row would need the block-max search of the
  append kernel, which `--fullattn-check` gates (the reference fails instead of quietly
  comparing another precision).
- It prints per layer `model_stage` (xnorm / src / outproj / res / post / moe / shexp, device
  against reference, for localisation) and `model_ref` (the whole hidden row), plus the first
  layer of each type re-run against the chain hidden (`replay_rel_l1`, which is the
  "adjacent layer hidden equals the single-layer chain" item), and ends with `model_check`
  carrying the worst hidden, the logit rel_L1 of the two arms and the verdict against 1e-4.
  `--model-check-reseed on` starts every reference layer from the device hidden, which
  isolates one layer instead of accumulating the chain.
- The activation quantiser is a step function, so it dominates any model-level distance
  between two arms: sharing only its two packing decisions (input-norm planes excepted, they
  are overwritten before the readback) moves the same reference chain from 1.2e-3 to 2.1e-6 on
  the layer output - a factor 590 - while the arithmetic stages stay at 1e-7. `qgate.py
  compare` of `--logits-out` against `--model-check-ref` is the reading that behaves like a
  distance (RESULTS `m2-7a-model-ref-chain`).

## Telemetry## Telemetry

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
| `attn_deltanet.h/.cpp`, `attn_deltanet_check.cpp` | DeltaNet delta-rule recurrence (M2-6c) and its CPU reference |
| `attn_full.h/.cpp`, `attn_full_check.cpp` | full attention: per-head norm + IMRoPE + flash causal + gate (M2-6d), KV row formats and host paging (M2-6a) |
| `kernels/norm.h/.cpp` | RMS norm + Q8_1 activation pack (M2-6), the dense GEMM's activation leg |
| `kernels/q4k_gemm.*` | Q4_K x Q8_1 int8 GEMM and swiglu+Q8_1 kernels, one submit per call |
| `kernels/moe_expert.*` | one expert's gate/up/swiglu/down as a prefill compute callback |
| `layer.h/.cpp` | one layer chain: norm, projections, attention path, output projection, residual, post norm, FFN half, one pass submit (M2-6) |
| `ffn.h/.cpp` | FFN half: router, top-k, gated shared expert, routed experts, weighted accumulate (M2-6); `routed_only` skips shared/residual for k=1..8 |
| `moe.h/.cpp` | routed-only batch over the FFN kernels (M3-1) |
| `expert_cache.h/.cpp` | per-layer cache plan, SLRU/LRU/static, trace parser/replay, H2D upload/wait contract (M3-0); masked acquire for the CPU leg and region uploads (M3-2a) |
| `miss_split.h/.cpp` | H2D/CPU miss split: greedy pass-level balance under the measured leg rates, fixed 0..100 control arms (M3-2a) |
| `cpu_moe.h/.cpp` | CPU expert core: Q4_K/Q5_K/Q6_K x Q8_1 GEMV, typed expert pass, subnormal fp16 deltas (M3-1a/M3-2a) |
| `model_generate.cpp` | the persistent decode session shared by `--generate` (k=1, teacher-forced `--generate-batch`) and `--spec-check` (M3-2/M3-3) |
| `model.h/.cpp` | whole-model 40-layer logits chain and the QRTLOG1 dump; the per-layer expert cache replaces the prefill slot after prefill (M3-2a) |
| `spec.h/.cpp` | the speculation decision: accepted-prefix rule, bonus token at the first mismatch, layer rollback, the restricted draft vocabulary (M3-3) |
| `qrt_main.cpp` | CLI entry: every check plus `--logits-dump`, `--model-check`, `--chunk-check`, `--cache-check`, `--moe-batch-check`, `--generate`, `--spec-check` |
| `CMakeLists.txt` | `qrt_core`, `qrt`, `qrt_checks` and the CPU contract tests |

## Decode expert cache and miss split (M3-2a)

After prefill, `--generate` frees the shared FFN slot and builds one slot array per layer
from `QRT_EXPERT_CACHE_MIB` (default 4198 MiB, 21.9%) through `qrt_cache_plan`. Each decode
call probes the layer's unique experts, asks `qrt_miss_split_plan` which misses go to H2D and
which to the CPU, uploads the H2D part with one region copy per staging load, and computes the
CPU part from the same Q8_1 planes the device produced. The CPU sum is added to the layer row
after the download, so the GPU pass records only the hit + H2D experts. An expert's host result
is staged per `(expert, row)` (`cpu_res[cpu_pos][run_tokens][n_embd]`): one miss expert serves
every row that routed to it, and a single per-expert vector would hand each row the summed
result of all of them (M3-2c). `--generate-batch K`
feeds K prompt rows per pass: a teacher-forced diagnostic for the k=5 legs, not an acceptance
path. `--miss-split` takes auto or the fixed 0..100 control arms.

## Speculation cycle (M3-3)

`qrt --spec-check <gguf> --logits-tokens <ids> --generate-new 12 --layer-store <prefix>
--logits-layers 4 --spec-k 4 [--spec-rollback off] [--telemetry 1 --telemetry-path f.csv]`

Design 5.4 runs as two arms over the same prompt inside one command. Arm A decodes serially,
one row per pass, and keeps every logits row. Arm B reopens the session and runs cycles of up
to `spec_k` rows whose drafts are arm A's true continuation with one position broken per
cycle, so the accept count walks 0..k-1; `qrt_spec_emit` decides what the cycle emits (the
accepted drafts plus `verified[accepted]`), `qrt_spec_rollback` puts every layer back at the
state after the first `kept` rows, and the caller moves the length pointer. The two arms must
emit the same ids, and every verified row that was fed a true token must match arm A's logits
for the same predicted index. `--spec-rollback off` keeps all rows instead: that arm must
part ways with arm A, otherwise the comparison could not see a lost state.

Rollback needs the state after each row, so the DeltaNet recurrence keeps a ring of
`dn_snap_rows + 1` states per layer (two without speculation) and reads slot `cur`, writing
`(cur + 1 + t) % slots`. That replaced the 2 MiB per layer per chunk `state_out -> state_in`
copy with a cursor move. The conv taps need no extra memory: after accepting `kept` rows the
same carry stages them from the rows that stay, which is why the carry has to be recorded as
the first command of the next pass (the qkv projection of that pass overwrites exactly the
rows the carry reads).
