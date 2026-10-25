// Tile geometry shared by the host and the three quant kernels. Copied from the M2-4
// probe (kernels/bench/) so the runtime build does not depend on bench/.
// Q4K_ROWS and Q4K_TOKENS are the sweep knobs that closed the 6.0 TFLOPS gate.
#ifndef QRT_TILES_H
#define QRT_TILES_H

#define Q4K_QK 256              // superblock length of every K-quant
#define Q4K_SUBGROUP 32         // lanes per row block; lane covers Q4K_ROWS rows 32 apart
#ifndef Q4K_WG_INVOCATIONS
#define Q4K_WG_INVOCATIONS 128  // subgroups x 32; each subgroup owns Q4K_TOKENS tokens
#endif
#define Q4K_SUBGROUPS (Q4K_WG_INVOCATIONS / Q4K_SUBGROUP)
#define Q4K_GROUPS 4            // 64-element groups per superblock

#ifndef Q4K_ROWS
#define Q4K_ROWS 2
#endif
#ifndef Q4K_TOKENS
#define Q4K_TOKENS 8
#endif

#define Q4K_TILE_N (Q4K_ROWS * Q4K_SUBGROUP)
#define Q4K_TILE_M (Q4K_TOKENS * Q4K_SUBGROUPS)

// Superblock bytes of the quant types the store mixes (ggml-common.h).
#define QRT_BLOCK_Q4K 144
#define QRT_BLOCK_Q5K 176
#define QRT_BLOCK_Q6K 210

#endif
