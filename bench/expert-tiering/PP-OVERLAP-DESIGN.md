# PP overlap: whole-layer expert prefetch over the transfer queue

Date: 2026-10-21. Direction confirmed by the user: PP overlap first, then repack KLD=0 and hot prefix.
Target: pp2048 (ub2048) from 578-590 t/s to >= 780 t/s, tg32 not worse than -3%.

## 1. Why the previous overlap attempts failed

Measurements (L5, L10) showed prefill as a serial sum of upload and compute:
pp2048 ub2048 = 3.54 s/eval, of which about 1.08-1.34 s is host->device expert bytes and about
2.48 s is GPU compute. Every pipelined variant was slower. The mechanism is now clear:

- The MoE cache uploads the experts of layer L inside the scheduler copy callback, which runs only
  when the split of layer L is prepared. `ggml_backend_sched_copy_input` first calls
  `ggml_backend_sched_wait_backend`, a full `ggml_backend_synchronize` on this build because
  `n_copies == 1` (single GPU, no pipeline parallel, so no events are created). The host therefore
  waits for all previously submitted compute before the copy, and the copy itself is drained by the
  compute of the same layer (transfer semaphore). No overlap is possible on this path.
- The expert selection of layer L is only known after the MoE output of layer L-1, so the
  gather-upload cannot be issued earlier (L10). The only information-free way to start earlier is to
  upload the whole expert layer, which needs no routing at all.

## 2. Design

For large ubatches (default ubatch tokens >= 1024, `LLAMA_MOE_PREFETCH_MIN_TOKENS`) the cache maps
every expert of every non-fully-resident MoE layer to two ring buffers of 256 slots each, taken from
the LRU pool of the bank. Layers stream through the rings in execution order:

    copy callback of layer at position pos:
        [scheduler already drained all previous compute]
        if pos > 0: host-wait for prefetch(pos)           # issued one layer earlier, hidden
        write identity slot map: map[e] = ring(pos%2) + e
        set_async the small slot map copy (1 KiB)          # the compute of this layer waits on this
        issue prefetch(pos+1): 3 x copyBuffer of 466 MiB total from pinned host weights
        submit: slot map copies first, then the prefetch batch

    first layer: issue prefetch(0), submit, host-wait, then continue with pos=1

Properties:

- The drain before the copy is not a problem: it paces the host at one layer of compute, and
  prefetch(pos+1) runs on the transfer queue while the compute of layer pos runs. Upload per layer
  is 466 MiB, which at 13.9 GB/s takes 33.5 ms; compute per layer is about 62 ms, so the upload is
  hidden after the first layer (33.5 ms exposed once per eval).
- Ring reuse is safe because the host drain before the copy of layer pos guarantees that the compute
  of layer pos-1 is done; ring((pos+1)%2) was last read by layer pos-1.
- The prefetch batch is signaled by its own timeline semaphore, so the compute queue never waits for
  it: the compute of layer pos waits only for the 1 KiB slot map copy. The host waits for the
  prefetch of layer pos exactly at the copy of layer pos.
- Ring slots alias the LRU pool, so decode keeps the full bank (tg32, measured after a prefill, was
  already cold in the baseline: the large-batch path does not populate the LRU). The LRU residency
  map is invalidated when a prefetch pass starts and rebuilt before the next small batch.
- Fully resident layers (n_full, 5 layers with `--moe-cache-mib 6144`) keep their static views and
  are skipped by the prefetch order.

## 3. Implementation

| Part | File | Change |
|---|---|---|
| prefetch API | `ggml/src/ggml-vulkan/ggml-vulkan.{cpp,types.h}` | new transfer context + timeline semaphore; `prefetch_tensor` (pinned source only), `prefetch_submit`, `prefetch_wait`; exported through `ggml_backend_vk_reg_get_proc_address` |
| orchestration | `src/llama-moe-cache.cpp` | ring region in the bank, prefetch order, identity slot maps, `copy_prefetch`, LRU invalidate |
| slot map buffer | `src/llama-moe-cache.cpp` | slot maps move to the device host buffer type (pinned), so the 1 KiB copy is a direct transfer |
| switch | env `LLAMA_MOE_PREFETCH` (default on, `=0` disables), `LLAMA_MOE_PREFETCH_MIN_TOKENS` (default 512), `LLAMA_MOE_PREFETCH_MAX_FULL` (default 4) | default path unchanged when the prefetch is not usable |

No public ggml API and no graph structure change: the graph keeps the slot map host weight of every
layer, which is also what gives the per-layer split boundary and the copy callback.

## 4. Measured (2026-10-21, same binary, same session, cross r=3 unless noted)

Command (both configs): `llama-bench -m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99
-ncmoe 41 --moe-cache-mib 6144 -fa on -t 5`, prefetch on = `LLAMA_MOE_PREFETCH=1` (defaults:
min tokens 512, max static whole layers 4 per group).

| Test | prefetch off | prefetch on | delta |
|---|---:|---:|---:|
| pp2048 (-ub 2048) | 440.14 +- 0.94 | 523.41 +- 0.40 (521.61 r2) | **+18.9%** |
| pp512 (-ub 2048) | 334.91 +- 1.55 | 393.95 +- 0.16 | **+17.6%** |
| pp256 (-ub 2048, r2) | 242.13 +- 2.72 | 236.75 +- 1.70 | -2.2% (below the 512 threshold, only the residency split changed) |
| tg32 (llama-bench) | 22.54 +- 0.32 | 23.14 +- 0.34 | +2.7% |
| real prefill, 711-token prompt | 80.28 t/s | 116.38 t/s | **+45%** |
| real generation, 127 tokens after that prompt | 13.56 t/s | 14.14 t/s | +4.3% |

Session power state is about 25% below the AGENTS numbers (pp512 335 vs 383-387), so compare within
this table only. Scaled to the AGENTS state the pp2048 gain projects to about 696 t/s.

Quality: the generated text is byte-identical with the switch on and off for a 78-token prompt and for
a 711-token prompt (temperature 0, same seed), which is expected because the bytes and the kernels are
the same, only the device slot of each expert changes. `test-backend-ops -o MUL_MAT_ID -b Vulkan1`
passes (3/3 backends OK).

Profile of the prefetch run (pp2048, r1, `GGML_SCHED_TRACE=1 -v`):

- splits = 42 (was 90), no per-expert gather copies, 40 host weight copies (the 1 KiB slot maps)
- copy wait = 3753 ms of a 3920 ms eval, cpy_sync = 3705 ms: the host waits for the compute of the
  previous layer (the scheduler drains before every host weight copy), the prefetch of the next layer
  runs in that window
- 80 layers uploaded per 2 evals = 37200 MiB, i.e. 18.6 GiB per eval at the 13.9-14.2 GB/s large-copy
  rate = about 1.33 s, hidden under the about 3.7 s of compute
- one exposed prefetch of 33 ms at the start of each pass (first layer), about 0.8% of the eval

Conclusion: the upload/compute overlap now works. The eval is compute bound: pp2048 ceiling at this
compute time is 2048/3.7 s = 553 t/s, and 780 t/s needs the GPU compute per eval to drop to about
2.6 s (-30%). That is MoE GEMM / kernel-gap work (see `bench/moe-baseline/L5-PREFILL-UPLOAD-FLOOR.md`
section 5 and `P8-ID-GEMM-GATE.md`), not more overlap. Raising the PP target further is a compute item.

Side effect worth keeping: the slot maps now live in the pinned host buffer type instead of the CPU
buffer, so their per-split copy is a direct transfer instead of a staging copy followed by a device
drain. Same binary, prefetch off: tg32 20.08 -> 22.54 t/s and pp512 332.25 -> 334.91 t/s.

Default-on check (user decision 2026-10-22): the switch is now on by default
(`LLAMA_MOE_PREFETCH=0` disables). Qwen default build: pp2048 523.41 (r3), pp512 393.95 (r3),
tg32 23.14. LFM2.5 with its canonical `-ncmoe 36 --moe-cache-mib 5120`: the second group has only 53
LRU slots (rings need 128), so the prefetch disables itself before the residency split and the layout
is the plain cache: pp2048 654.5 vs 650.1, pp512 430.2 vs 422.4, tg128 27.65 vs 27.96 (all within
noise of the `LLAMA_MOE_PREFETCH=0` control).

## 5. Risks

- Upload bandwidth lower than 13.9 GB/s for one 466 MiB batch: measured upload is hidden at ub2048 and
  ub512 because the whole-layer bytes stay below the compute time; at 256 tokens the threshold keeps
  the old path.
- Ring/aliasing correctness: validated by identical outputs with the switch on and off, plus
  `test-backend-ops -o MUL_MAT_ID -b Vulkan1` (939/939).
- Pinned host weights: the prefetch is disabled with a warning if the experts are not allocated in
  the device host buffer type.
- The switch changes the residency split (4 static whole layers per group instead of 5, two whole
  layers of slots are used by the rings). With the default 6144 MiB cache this leaves 666 LRU slots
  per group and decode is slightly faster (tg32 +2.7%); smaller caches may not fit the rings and the
  prefetch then disables itself.
