// Tile geometry shared by the host and the shader (R1c Q4_K x Q8_1 GEMM probe).
// Kept in one place so the grid in q4k-gemm.cpp cannot drift from the index math.
#ifndef Q4K_TILES_H
#define Q4K_TILES_H

#define Q4K_QK 256              // superblock length of Q4_K
#define Q4K_W_WORDS 36          // 144 B superblock: d, dmin, 12 scale bytes, 128 qs bytes
#define Q4K_A_WORDS 9           // 36 B q8_1 block: d, s, 32 qs bytes
#define Q4K_SUBGROUP 32         // lanes per row block; lane covers two rows 32 apart
#define Q4K_TILE_N 64           // rows per workgroup
#define Q4K_TILE_M 32           // tokens per workgroup
#define Q4K_WG_INVOCATIONS 128  // four subgroups, each on its own group of tokens
#define Q4K_TOKENS 8            // tokens per lane
#define Q4K_GROUPS 4            // 64-element groups per superblock

#endif
