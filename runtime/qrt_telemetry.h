#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// One decode pass. These are the columns bench/README.md section 2 asks for
// (accepted tokens, hit rate, both miss legs, H2D bytes) plus the pass wall time.
struct qrt_pass_stats {
    uint32_t k;
    uint32_t pos;
    uint32_t accepted;
    uint32_t hit_experts;
    uint32_t miss_experts;
    uint32_t miss_cpu;
    uint32_t miss_h2d;
    uint64_t h2d_bytes;
    double   draft_ms;
    double   gpu_ms;
    double   cpu_leg_ms;
    double   pass_ms;
};

struct qrt_telemetry {
    FILE *   f;
    uint64_t rows;
    int      failed;
};

// Writes the header and returns the sink. NULL with a message in err means the
// caller keeps running without telemetry instead of aborting (design 9).
qrt_telemetry * qrt_telemetry_open(const char * path, char * err, size_t err_size);
void qrt_telemetry_close(qrt_telemetry * t);

// NULL-safe: a disabled run pays one call and one branch per pass.
void qrt_telemetry_pass(qrt_telemetry * t, uint64_t pass_index, const qrt_pass_stats * stats);
int  qrt_telemetry_failed(const qrt_telemetry * t);
