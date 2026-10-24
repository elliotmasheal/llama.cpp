// Tile geometry shared by the host and the shader (R1c Q4_K x Q8_1 GEMM probe).
// Kept in one place so the grid in q4k-gemm.cpp cannot drift from the index math.
// Q4K_ROWS and Q4K_TOKENS are the M2-4 sweep knobs: rows and tokens per lane.
#ifndef Q4K_TILES_H
#define Q4K_TILES_H

#define Q4K_QK 256              // superblock length of Q4_K
#define Q4K_W_WORDS 36          // 144 B superblock: d, dmin, 12 scale bytes, 128 qs bytes
#define Q4K_A_WORDS 9           // 36 B q8_1 block: d, s, 32 qs bytes
#define Q4K_SUBGROUP 32         // lanes per row block; lane covers Q4K_ROWS rows 32 apart
#ifndef Q4K_WG_INVOCATIONS
#define Q4K_WG_INVOCATIONS 128  // subgroups x 32; each subgroup owns Q4K_TOKENS tokens
#endif
#define Q4K_SUBGROUPS (Q4K_WG_INVOCATIONS / Q4K_SUBGROUP)
#define Q4K_GROUPS 4            // 64-element groups per superblock

#ifndef Q4K_ROWS
#define Q4K_ROWS 2              // rows per lane; R1c baseline is 2
#endif
#ifndef Q4K_TOKENS
#define Q4K_TOKENS 8            // tokens per lane; R1c baseline is 8
#endif

#define Q4K_TILE_N (Q4K_ROWS * Q4K_SUBGROUP)
#define Q4K_TILE_M (Q4K_TOKENS * Q4K_SUBGROUPS)

#endif
