# L2: decode latency of the cache config - what is left

Date: 2026-10-18. Branch `moe-gpu-cpu-hybrid`, base 4cf3d44a9 (L1). RX 6700S 8 GB, `-dev Vulkan1 -ngl 99 -fa on -t 5`.
Canonical config: `-ncmoe 41 --moe-cache-mib 6144 -ub 2048`. Static reference: `-ncmoe 27`.

## Change kept: whole layers read the bank directly

A whole layer keeps its experts at bank slots `j*n_expert + e`, so the bank view of the layer is
exactly its expert tensor and the selected expert ids index it without a slot map. The graph uses
that view (`get_experts_static`) and the layer needs no slot map, no host copy and no copy callback.
This removes one split boundary per whole layer from every graph.

| per decode token (GGML_SCHED_TRACE) | before | after |
|---|---:|---:|
| splits | 42 | 32 |
| copy total | 35.3 ms | 25.0 ms |
| h2d body | 24.4 ms | 13.7 ms |
| cpy_sync (waits) | 10.8 ms | 11.2 ms |
| gpu_submit | 13.5 ms | 8.7 ms |

Callback breakdown (instrumented, before): ids readback + device drain 3.0 ms/token, LRU plan
0.03 ms, miss-expert upload issue 0.2 ms. Hit rate 68.4% (10 whole layers always hit), 78 misses
per token, about 140 MiB uploaded per token.

## Acceptance numbers (llama-bench, r=3, interleaved A/B, 2 rounds)

| test | static -ncmoe 27 | cache -ncmoe 41 --moe-cache-mib 6144 | delta |
|---|---:|---:|---:|
| tg32 | 18.96 (18.86/19.06) | 26.83 (26.72/26.79/26.86/26.91) | +41% vs static |
| pp512 | 346.1 | 356.4 | +3.0% |
| pp2048 ub2048 | 576.5 | 578.3 | +0.3% |

TG is +8.1% over L1 (24.79 -> 26.83). The target of 30 t/s is not reached.

## Text identity

cache vs `-ncmoe 41` (same placement) with the decode MoE forced onto the GPU in both runs
(`GGML_OP_OFFLOAD_MIN_BATCH=1`), 514 token prompt, `-n 32 --temp 0`: byte-identical, including
after this change. All copies still carry identical bytes, so no numeric change.

## Tried and reverted: run the copy callback before the scheduler wait

Moving `callback_copy` above `ggml_backend_sched_wait_backend` in `ggml/src/ggml-backend.cpp` made
the prefill copies pipeline: pp512 346 -> 920 t/s, tg generation 15 -> 36 t/s in llama-cli. But the
generated text became garbage for both the static and the cache config. The wait is not redundant:
a copy destination can reuse compute buffer memory that the previous split still reads, so removing
it is a data race. Reverted, no ggml change is kept.

## Where the remaining decode time goes

About 37 ms per token: 32 split boundaries x (about 0.37 ms device wait + 0.27 ms submit) plus the
3 ms ids readback and the GPU compute. Each of the 30 partially resident layers must send its
selected experts to the host to plan and upload them, so a host round trip per layer is structural
for a host managed LRU. Reaching 30 t/s needs the residency plan on the device (slots computed by
the GPU from a device side map), which is a larger change than L2.
