# qrt_proto - M1-4a Vulkan pass prototype

Measures one decode "pass" of the planned qwen36 MoE runtime at k = 1, 2, 4, 5, 8, without the
model and without any math. It answers one question for M1-4b: what do a pre-recorded pass, its
single submit and its single timeline wait cost on this GPU.

## What a pass is here

- Synthetic device-local buffers hold the bytes the real pass would read: 2.02 GiB dense
  (design 2.1, bucketed as 40 equal layer slices) and one expert buffer sized for the largest
  union in the arm table. Buffer contents are undefined; nothing is read back for value.
- A pass is 40 layers. Each layer does one dense dispatch over its slice, one dispatch over
  `U_k` experts (1.828 MiB each) and then a compute-to-compute memory barrier. That is 80
  dispatches and 40 barriers in ONE command buffer, with one `vkQueueSubmit` and one timeline
  semaphore wait per pass (design 4.6).
- `U_k`, the unique experts per layer for k tokens, comes from the M1 trace:
  k=1 -> 8.0, k=2 -> 12.7, k=4 -> 19.9, k=5 -> 23.5, k=8 -> 31.6. Expert bytes per pass are
  therefore `40 * U_k * 1.828 MiB` (k=1: 585 MiB, k=8: 2311 MiB).
- Reads go through a compute shader (`qrt-read.comp`, four independent vec4 chains per thread,
  grid-stride), never through `vkCmdCopyBuffer`, so the time is a read and not a copy.
- The command buffer for every k is recorded once at startup and then submitted `reps` times.
  Recording cost is outside every measurement.

## Build

Option A, CMake (guarded by `GGML_VULKAN`, opt-in because it does not link ggml/llama):

```
cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_QRT_PROTO=ON
cmake --build build-vulkan --target qrt_proto -j 8
```

The target compiles `qrt-read.comp` with `glslc` into `build-vulkan/runtime/proto/qrt-read.spv`
and puts `qrt_proto.exe` next to it.

Option B, one shader command plus one compiler command (MinGW, Vulkan SDK):

```
glslc -O runtime/proto/qrt-read.comp -o qrt-read.spv
g++ -O2 -std=c++17 runtime/proto/qrt-proto.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o qrt_proto.exe
```

## Run

```
cd build-vulkan/runtime/proto
qrt_proto.exe [reps] [warmup] [groups] [shader.spv]
```

Defaults: `reps=20`, `warmup=3`, `groups=1024`, `shader=qrt-read.spv` (relative to the working
directory, so run it from the directory that holds the .spv or pass the path).

Memory: one device-local allocation of about 4380 MiB (2068 MiB dense + 2311 MiB largest expert
union + alignment) plus a 16 KiB output buffer, so it needs the dGPU idle. Dense and expert
slices live in the same allocation because two separate multi-GiB blocks can fail to fit on an
8 GiB card even when their sum does. If allocation fails the tool prints the size it needed and
exits with code 2.

## Output

Comment lines start with `#`. The rest is CSV, one header line and one row per k:

```
k,u_experts,expert_mib,read_mib,compute_mib,reps,submit_ms,sync_ms,pass_ms,gpu_ms,gbs
```

| column | meaning |
|---|---|
| `k` | verification width of the arm |
| `u_experts` | unique experts per layer assumed for that k (M1 trace) |
| `expert_mib` | expert bytes read in one pass, `40 * U_k * 1.828 MiB` |
| `read_mib` | total bytes read in one pass, dense + expert |
| `compute_mib` | device-local compute buffers allocated for the arm (same for every k: dense plus the largest expert union plus the output slice) |
| `reps` | timed submissions behind the medians |
| `submit_ms` | median wall time spent inside `vkQueueSubmit` (CPU cost of one pass submit) |
| `sync_ms` | median wall time spent inside `vkWaitSemaphores` (wait for the pass, GPU time plus wake-up) |
| `pass_ms` | median wall time from just before `vkQueueSubmit` to just after the wait returned |
| `gpu_ms` | median GPU timestamp delta over the 80 dispatches and 40 barriers; `-1.000` when the compute family has no timestamps |
| `gbs` | effective read rate, `read_mib` in GB (1e9) divided by `pass_ms` |

Read `submit_ms + sync_ms - pass_ms` as the CPU overhead outside both calls; it should be small.
`pass_ms` is the number to compare against the 2.17 GB + expert-union roofline for the same arm
(M1-4b), and `submit_ms + sync_ms` against the 1 ms per-pass budget.

## Limits

- No math, no routing, no H2D, no miss split: the prototype measures the GPU read legs only.
- Expert slices are contiguous and identical in size, so this is a byte-volume model of the
  union, not the real page layout.
- Timestamps need `timestampValidBits != 0` on the compute family; otherwise the `gpu_ms` column
  is `-1.000` and only the CPU-side numbers are usable.
