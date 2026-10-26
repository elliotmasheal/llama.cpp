#pragma once

#include <stddef.h>
#include <stdint.h>

// Miss split (design 5.1 `miss_split`, M3-2a): choose which missed experts go to the
// H2D leg and which are computed by the CPU leg so the two legs finish together.
// The module is host-only policy: the caller supplies each miss with its byte size
// and the number of tokens it serves (the multiplicity of RESULTS 22), plus the two
// leg rates it currently observes. Auto is a greedy longest-processing-time
// assignment: the largest miss goes to the currently shorter leg, which keeps
// max(h2d, cpu) close to the optimum for the handful of misses one layer sees.
//
// A fixed percentage exists only as a control arm: 0 sends every miss to the CPU,
// 100 sends every miss to H2D.

enum { QRT_SPLIT_MAX_TOKENS = 16 };

struct qrt_split_miss {
    uint32_t expert;
    uint64_t bytes;
    uint32_t tokens;    // 1..QRT_SPLIT_MAX_TOKENS; 0 is treated as 1
};

struct qrt_split_rates {
    double h2d_bytes_per_s;              // contended H2D bandwidth
    double cpu_ms[QRT_SPLIT_MAX_TOKENS + 1]; // 1..N tokens per expert
};

struct qrt_split_stats {
    uint32_t misses;
    uint32_t h2d;
    uint32_t cpu;
    uint64_t h2d_bytes;
    double   h2d_ms;
    double   cpu_ms;
};

// Time each leg has already spent in this pass. One layer holds too few misses for
// a <=10% split on its own (two granularity steps already move a leg by 20%), so auto
// balances the pass: the layer's assignment starts from the cumulative deficit and
// the rounding error of one layer is paid back by the next.
struct qrt_split_carry {
    double h2d_ms;
    double cpu_ms;
};

// Fills out_stats and (when h2d_mask is not NULL) one byte per miss, 1 = H2D leg,
// 0 = CPU leg. percent is QRT_MISS_SPLIT_AUTO or 0..100. carry may be NULL (zero).
// The order of misses is preserved; duplicates are the caller's business. Returns -1
// only for bad arguments (NULL outputs with a count, percent outside auto/0..100,
// non-positive rates); a zero miss count is valid and writes zeros.
int qrt_miss_split_plan(const qrt_split_miss * misses, uint32_t count, int percent,
                        const qrt_split_rates * rates, const qrt_split_carry * carry,
                        uint8_t * h2d_mask, qrt_split_stats * out_stats,
                        char * err, size_t err_size);

// Rate model defaults from RESULTS 9/20/22: contended H2D 9.9 GB/s and the 8-thread
// CPU expert curve 0.079/0.106/0.160/0.199 ms at t=1/2/4/5. Values between the
// measured points are linear; above t=5 the t=5 slope continues (conservative).
void qrt_miss_split_default_rates(qrt_split_rates * rates);

// Cost of one CPU expert with this token multiplicity, in ms.
double qrt_miss_split_cpu_ms(const qrt_split_rates * rates, uint32_t tokens);

// Folds one measured layer into the rates: h2d_ms is the wall time of the uploads
// and cpu_ms the CPU leg's compute time. A zero or negative observation keeps the
// previous value; otherwise the new sample replaces 3/4 of the old estimate, which
// follows contention changes without letting one outlier dominate.
void qrt_miss_split_update(qrt_split_rates * rates, uint32_t h2d_experts,
                           uint64_t h2d_bytes, double h2d_ms, uint32_t cpu_experts,
                           uint32_t cpu_tokens, double cpu_ms);
