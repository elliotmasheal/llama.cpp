#include "qrt_telemetry.h"

#include <inttypes.h>
#include <stdlib.h>

static const char CSV_HEADER[] =
    "pass,pos,k,accepted,hit_experts,miss_experts,miss_cpu,miss_h2d,h2d_bytes,"
    "draft_ms,gpu_ms,cpu_leg_ms,pass_ms\n";

qrt_telemetry * qrt_telemetry_open(const char * path, char * err, size_t err_size) {
    if (path == NULL || *path == '\0') {
        snprintf(err, err_size, "telemetry sink path is empty");
        return NULL;
    }
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        snprintf(err, err_size, "cannot open telemetry sink '%s'", path);
        return NULL;
    }
    // Line buffered: a crash keeps every pass already measured.
    setvbuf(f, NULL, _IOLBF, 0);
    if (fputs(CSV_HEADER, f) == EOF || fflush(f) != 0) {
        snprintf(err, err_size, "cannot write the header to '%s'", path);
        fclose(f);
        return NULL;
    }
    qrt_telemetry * t = new qrt_telemetry();
    t->f      = f;
    t->rows   = 0;
    t->failed = 0;
    return t;
}

void qrt_telemetry_close(qrt_telemetry * t) {
    if (t == NULL) {
        return;
    }
    if (t->f != NULL) {
        fclose(t->f);
    }
    delete t;
}

void qrt_telemetry_pass(qrt_telemetry * t, uint64_t pass_index, const qrt_pass_stats * s) {
    if (t == NULL || t->failed || s == NULL) {
        return;
    }
    const int n = fprintf(t->f,
                          "%" PRIu64 ",%u,%u,%u,%u,%u,%u,%u,%" PRIu64 ",%.3f,%.3f,%.3f,%.3f\n",
                          pass_index, s->pos, s->k, s->accepted, s->hit_experts, s->miss_experts,
                          s->miss_cpu, s->miss_h2d, s->h2d_bytes,
                          s->draft_ms, s->gpu_ms, s->cpu_leg_ms, s->pass_ms);
    if (n < 0 || fflush(t->f) != 0) {
        t->failed = 1;
        return;
    }
    ++t->rows;
}

int qrt_telemetry_failed(const qrt_telemetry * t) {
    return t != NULL ? t->failed : 0;
}
