# q4k-gemm - R1c int8 GEMM probe (Q4_K x Q8_1)

Measures the effective throughput of a Q4_K (weights) x Q8_1 (activations) int8 dot product
GEMM at prefill shape, without a model: one expert gate matrix, N = 512 rows, K = 2048,
M = 2048 tokens. This is the compute leg that `bench/RESULTS.md` section 11 says prefill needs
(6.00 TFLOPS effective for PP 1000 at chunk >= 1420, 7.06 TFLOPS for 0.85 x roofline).

Nothing here links ggml or llama: the block layouts are mirrored byte for byte from
`ggml/src/ggml-common.h` (`block_q4_K` line 337 with `K_SCALE_SIZE` = 12, `block_q8_1` line
268), the 6-bit scale/min unpacking comes from `get_scale_min_k4`
(`ggml/src/ggml-quants.c` line 880), and the dequant order from `dequantize_row_q4_K`
(`ggml/src/ggml-quants.c` line 1530).

## Files

| file | what |
|---|---|
| `q4k-gemm.comp` | the kernel; one Q4_K x Q8_1 int8 dot per output element |
| `q4k-gemm.cpp` | host: generate, upload, dispatch, time, CPU cross-check |
| `q4k-tiles.h` | tile geometry shared by both (single source for the grid) |
| `build.sh` | glslc + g++, two shader variants, no CMake |

## Layout and algebra

Weights: row-major `N x K`, `K/256` superblocks of 144 B per row (d, dmin, 12 scale bytes,
128 `qs` bytes). Blocks are addressed as 36 32-bit words: word 0 = (d, dmin) as half2,
words 1-3 = the 12 scale bytes, words 4-35 = `qs`.

Activations: row-major `M x K`, `K/32` blocks of 36 B per token (d, s, 32 int8). Word 0 =
(d, s) as half2, words 1-8 = `qs`.

For a 32-element subblock `j` with 6-bit `sc`/`m` and a Q8_1 block with delta `aq` and
`s = aq * sum(qs)`, the contribution to the output element is

```
sum_i (d*sc*q_i - dmin*m) * (aq*a_i)
  = (d*sc)*aq * sum_i(q_i*a_i)  -  (dmin*m) * s
```

so the kernel only needs the integer product `sum q_i*a_i` and never dequantizes to float:
one `dotPacked4x8EXT` per 4 elements, `(d*sc)*aq` and `(dmin*m)` folded in once per subblock.
This is why `block_q8_1` carries `s`, and it is what makes the fp16 `s` the only lossy step
(see the check tolerance below).

Subblock element order (same as `dequantize_row_q4_K`): a 64-element group `g` covers `qs`
bytes `32g..32g+31`; subblock `2g` takes the low nibbles of those bytes (elements 0-31 of the
group) and subblock `2g+1` their high nibbles (elements 32-63). Byte `32g+e` maps to Q8_1 block
`8*sb + 2g + half`, element `e`.

## Tile map

128 threads (4 subgroups of 32) per workgroup by default. Two knobs select the register tile
(`Q4K_ROWS` rows and `Q4K_TOKENS` tokens per lane, defaults 2 x 8):

- lane `l` of subgroup `s` covers rows `TILE_N*gx + 32*r + l` for `r` in 0..ROWS-1 and tokens
  `TILE_M*gy + 8*s + t` for `t` in 0..TOKENS-1, with `TILE_N = 32*ROWS`, `TILE_M = 4*TOKENS`.
  That is `ROWS x TOKENS` float accumulators; one activation word feeds `ROWS` dots and one
  weight word feeds `TOKENS` dots.
- per group of 64 elements the 8 weight words of every row are masked to 8 low-nibble and 8
  high-nibble `uint`s and reused across the tokens.
- grid is `(N/TILE_N, M/TILE_M)`: 8 x 64 = 512 workgroups at the default shape, so every row
  block is read 64 times (once per token tile).

## M2-4 sweep (2026-10-24)

The tile knobs were added to close the 6.0 TFLOPS gap of design 6.4. Measured on the same
shape, `--check` passing for every arm, raw output `bench/raw/m2/q4k-sweep.out`:

| rows x tokens / threads | TFLOPS | note |
|---|---:|---|
| R1c kernel before this sweep | 3.61-3.64 | baseline (RESULTS 14: 3.66) |
| 2 x 8 / 128 | **4.45-4.49** | new default, +23% |
| 2 x 8 / 256 | 4.40-4.42 | no gain |
| 4 x 4 / 128 | 3.94-3.97 | more rows, fewer loads, slower |
| 4 x 8 / 128 | 3.90-3.96 | dots double, loads +11%, still slower |

Static instruction counts per superblock (`spirv-dis`): 2x8 has 1024 `OpSDot` / 655 `OpLoad`
/ 671 `OpAccessChain`; 4x8 has 2048 / 727 / 759. The 4-row arms are limited by registers
(4 x 16 weight masks + 32 accumulators) and code size, not by load count. The remaining gap
to 6.0 is the 16 activation loads and the scale fold per token and group; the order of the
next steps is in `bench/DECISIONS.md` section 20.

Build a swept arm with the same -D on both the shader and the host, for example:

```
glslc -O --target-env=vulkan1.1 -DQ4K_DOT4 -DQ4K_ROWS=4 -DQ4K_TOKENS=4 \
    -Iruntime/kernels/bench runtime/kernels/bench/q4k-gemm.comp -o out/dot-4x4.spv
g++ -O2 -std=c++17 -DQ4K_ROWS=4 -DQ4K_TOKENS=4 runtime/kernels/bench/q4k-gemm.cpp \
    -Iruntime/kernels/bench -I"$VULKAN_SDK/Include" -L"$VULKAN_SDK/Lib" -lvulkan-1 -o out/gemm-4x4.exe
```


Estimated instruction mix per 32-element subblock and per output: 8 `dotPacked4x8EXT`, 5
float ops for the fold, 4 activation word loads, 1 weight load, 2 masks, about 3.4 FLOP per
issued instruction. The dot4 issue ceiling on a 28 CU RDNA2 part at 1.8 GHz is roughly
25 TFLOPS if every instruction were a dot, so 6-7 TFLOPS implies 25-30% of that ceiling. If the
measured number is far below that, the order to look at is: activation loads (8/T, halved by
raising rows per lane), the fold (5 instructions per subblock), then occupancy from the
16-accumulator + 32-mask register set.

## Build

```
sh runtime/kernels/bench/build.sh
```

which is exactly:

```
glslc -O --target-env=vulkan1.1 -DQ4K_DOT4 -Iruntime/kernels/bench \
    runtime/kernels/bench/q4k-gemm.comp -o runtime/kernels/bench/out/q4k-gemm-dot.spv
glslc -O --target-env=vulkan1.0 -Iruntime/kernels/bench \
    runtime/kernels/bench/q4k-gemm.comp -o runtime/kernels/bench/out/q4k-gemm-fallback.spv
g++ -O2 -std=c++17 -Wall runtime/kernels/bench/q4k-gemm.cpp -Iruntime/kernels/bench \
    -I"$VULKAN_SDK/Include" -L"$VULKAN_SDK/Lib" -lvulkan-1 \
    -o runtime/kernels/bench/out/q4k-gemm.exe
```

`VULKAN_SDK` defaults to `D:/Scoop/apps/vulkan/current` in `build.sh`. The MinGW `bin`
directory must be on `PATH` when running the exe (libstdc++/libgcc), as with the other tools
in this repo.

## Run

```
cd runtime/kernels/bench/out
./q4k-gemm.exe                 # default shape, dot pipeline if the device allows it
./q4k-gemm.exe --check         # CPU cross-check, one dispatch, exit 1 on failure
./q4k-gemm.exe --reps 64 --runs 5
./q4k-gemm.exe --fallback      # force the portable kernel
./q4k-gemm.exe --list          # devices only
./q4k-gemm.exe --dev 1         # pick a device explicitly
```

Options: `--dev`, `--m`, `--n`, `--k`, `--reps`, `--warmup`, `--runs`, `--seed`, `--spv`,
`--spv-fallback`, `--fallback`, `--check`, `--list`. Overrides need `n % 64 == 0`,
`m % 32 == 0`, `k % 256 == 0`.

Startup prints the device list, then the chosen device and the integer dot capabilities
(`shaderIntegerDotProduct`, `integerDotProduct4x8BitPackedSignedAccelerated`), then the
pipeline actually used. `#` lines are comments, everything else is CSV.

### Timing CSV

```
shape,reps,runs,ms,tflops,w_gbs
2048x512x2048,32,3,0.78,5.512,0.76
```

| column | meaning |
|---|---|
| `shape` | `MxNxK` (tokens x rows x reduction) |
| `reps` | dispatches recorded into one command buffer, submitted once |
| `runs` | timed submissions; `ms` is the median of the per-run `ms/reps` |
| `ms` | median wall time per dispatch, from just before `vkQueueSubmit` to just after the timeline wait returned |
| `tflops` | `2 * M * N * K / ms`, 2 FLOP per MAC |
| `w_gbs` | weight bytes read per dispatch / `ms`, GB = 1e9 |

`ms` is a steady-state throughput number: one submit and one wait are amortized over `reps`
dispatches, so submit/sync cost is not in it (that is `M1-4b`, not this probe). Scripts that
want a latency number should run `--reps 1`.

`w_gbs` is a cache rate, not a DRAM rate: the 576 KiB matrix is read `M/32` = 64 times per
dispatch (37.7 MB requested) and stays in L2, so this probe cannot say anything about GDDR6
bandwidth. Use `bench/tools/vkgate.cpp` (`read8`) for that.

### Check CSV

```
check,pipeline,samples,max_abs,max_rel_l1,tol,result
check,dot,529,5.1e-03,1.9e-06,1.0e-04,pass
```

The reference is a double-precision direct dequant of a sample of rows x tokens (tile and lane
boundaries plus `N/2`, `N-2`, `N-1` and `M/2`, `M-2`, `M-1`: 23 x 23 = 529 samples at the
default shape), plus a second reference that repeats the shader's own folded algebra so a
failure can be blamed on the kernel or on the fp16 `s` field.

`max_rel_l1 = |gpu - ref| / sum|terms|`. That normalisation matters: a per-element relative
error explodes when a random dot product nearly cancels. With this data the fp16 `s` alone
leaves about 7e-6, and fp32 accumulation adds about 1e-6, so `tol = 1e-4` has an order of
magnitude of margin; an index mapping bug puts O(0.1) on the same scale. The tool exits 1 when
the tolerance is exceeded.

## Integer dot path and portability

The GLSL extension is `GL_EXT_integer_dot_product` (not `GL_EXT_shader_integer_dot_product`,
which glslc rejects) and its entry points are `dotPacked4x8EXT(uint, uint)` /
`dotPacked4x8AccSatEXT(...)` (not `dot4add_i8packedEXT`, which does not exist in GLSL). With
`-DQ4K_DOT4` the shader declares `#extension GL_EXT_integer_dot_product : require` and calls
`dotPacked4x8EXT`, which glslc lowers to `OpSDot` with
`SPV_KHR_integer_dot_product` / `DotProductInput4x8BitPacked` (checked with `spirv-dis`).

Without `-DQ4K_DOT4` the same source compiles the portable path: unpack the four bytes, sign
extend, multiply and add. It needs no extension and is correct on any Vulkan device, but its
numbers are not comparable (about 15 extra instructions per 4 MACs).

The host enables `shaderIntegerDotProduct` through `VkPhysicalDeviceVulkan13Features` when both
the loader and the device are 1.3, otherwise through `VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR`
plus the device extension. If the feature is missing, the dot module is missing, or the driver
rejects the module, it prints why and runs the fallback pipeline instead of failing. Vulkan 1.2
is required (timeline semaphores). Nothing is hard-coded for the RX 6700S; the workload shape is
the only thing that is specific.

## Limits

- No softmax/router/dense residual: one expert gate matrix, no memory-bound phase, no H2D.
  It answers "how many int8 dots per second can this shape sustain", not "what is PP".
- No shared memory staging: weights come straight from L2/L1. A production kernel would stage
  the activations of a token tile in workgroup memory; that is a later step (M2).
- The fully unrolled inner body is about 1024 `OpSDot` plus roughly as many other instructions
  (24 KB of code per superblock iteration); the SPIR-V is 120 KB before the driver. If the
  measured rate is low, check the ISA size before blaming the math.
- Only the `int8 x int8 -> int32` packed-signed form is exercised. Q5_K/Q6_K paths (which need
  the mixed sign form or a different unpack) are out of scope for R1c.
