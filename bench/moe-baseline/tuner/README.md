# Tuning service

A dedicated tuner that measures a (hardware, model) pair once, fits a model of prefill
throughput as a function of the ubatch, and persists the best launch config so it can be
queried cheaply and refined as the machine is used.

## Why

A fixed `-ub 2048` is wrong in general. The right ubatch depends on the GPU (bandwidth,
compute), the model (weight bytes, active params) and the VRAM budget. This service derives
it from measurements instead of hardcoding it.

## The model it fits

For a single prefill pass of `M` tokens (prompt = ubatch = M), prefill throughput is:

```
T(M) = M / (A + B*M)          [tokens/s]
1/T  = B + A*(1/M)            -> linear in 1/M, least-squares fit
```

- `A = weight_bytes / BW_eff`  : time to read all weights once (independent of M)
- `B = flops_per_token / FLOPS`: per-token compute time
- knee `M* = A/B`              : past it the pass is compute-bound
- from the fit: `BW_eff = weight_bytes / A`, `FLOPS_eff = flops_per_token / B`

The recommended ubatch is the smallest grid value `>= M*` that actually ran (i.e. on the
compute plateau); it is bounded by VRAM and by attention cost at large prompts.

## Files

| file | role |
|---|---|
| `tune.ps1` | measure + fit + persist one (hardware, model) entry |
| `best.ps1` | print the stored best args for a model (or all) |
| `service.ps1` | background loop: tune every new model, refresh periodically |
| `fit-ubatch.ps1` | offline: fit A/B from existing `results/P3fit_M*` logs |
| `store/<gpu>__<model>.json` | persisted profiles (identity, fit, best, history) |

## Use

```powershell
.\tune.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
.\best.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
.\service.ps1            # leave running; tunes new models as they appear
```

`tune.ps1` measures, in order:
1. residency (`-ncmoe`) sweep at a small ubatch -> best ncmoe by whole-turn wall time
2. PP(ubatch) single-pass sweep at that ncmoe -> fit A, B, knee, BW_eff -> best ubatch
3. residency refine at the chosen ubatch (ubatch and ncmoe interact)
4. final measure of the chosen (ubatch, ncmoe)

and writes everything to the store.

## Interaction to know about

ubatch and `-ncmoe` (residency) are not independent: the best residency measured at ub 512 can differ
from the best at ub 2048. That is why the service re-sweeps residency after picking the ubatch (step 3).
Some ubatch values also land on pathological code paths (a non-monotonic point was measured at M=1024 with
one residency setting), so the fit should be eye-checked against the raw `pp_points` in the store.

## TG is not tunable here

The tuner optimises PP (ubatch) and the residency split; it does not move TG. Measured at the best config
(`-ncmoe 27 -fa on`): TG ~13-14 t/s, GPU engine utilisation 36-42% but GPU kernel execution only ~16% of the
token time, aggregate weight bandwidth ~11% of peak (individual kernels already hit peak). KV quantisation
(`-ctk/-ctv q8_0`), `-fa off` and extra residency were all within +-0.7 t/s noise (see `../AB-TUNING.md` batch 7).
So TG is latency/serialisation bound (per-kernel latency plus the CPU MoE / GPU alternation at batch=1), and no
launch-config knob raises it. TG work has to go to prefetch/streaming (`../PREFETCH-DESIGN.md`), not the tuner.

## Prefetch/streaming has a low TG ceiling here too

The end state of prefetch/streaming is "all experts resident on a GPU tier". Measured on this box (dGPU + iGPU/UMA),
that tops out at ~12-16 t/s, i.e. no better than the single-GPU `-ncmoe 27` baseline of 14.2 (the 2-GPU path also
pulls experts from SSD and OOMs under a large ubatch). So expert placement is not the TG lever either; the ceiling
is the Vulkan backend's per-kernel latency. See `../AB-TUNING.md` batch 8.
