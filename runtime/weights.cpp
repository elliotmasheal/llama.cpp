#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qrt_weights {
    FILE *        bin;
    uint64_t      payload_bytes;
    uint32_t      layers;
    uint32_t      experts;
    uint32_t      records;
    uint64_t      align;
    char          arch[64];
    char          payload_sha[65];
    qrt_expert_loc * loc;   // layers * experts, row major
};

#ifdef _WIN32
#define WSEEK _fseeki64
#define WTELL _ftelli64
#else
#define WSEEK fseeko
#define WTELL ftello
#endif

int qrt_weights_open(const char * prefix, qrt_weights ** out, char * err, size_t err_size) {
    *out = NULL;
    if (prefix == NULL || *prefix == '\0') {
        snprintf(err, err_size, "no repack prefix given");
        return -1;
    }
    char idx_path[1024];
    char bin_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s.qwen36rt.idx", prefix);
    snprintf(bin_path, sizeof(bin_path), "%s.qwen36rt.bin", prefix);

    FILE * fi = fopen(idx_path, "r");
    if (fi == NULL) {
        snprintf(err, err_size, "cannot open %s", idx_path);
        return -1;
    }
    FILE * fb = fopen(bin_path, "rb");
    if (fb == NULL) {
        snprintf(err, err_size, "cannot open %s", bin_path);
        fclose(fi);
        return -1;
    }

    qrt_weights * w = (qrt_weights *) calloc(1, sizeof(qrt_weights));
    if (w == NULL) {
        snprintf(err, err_size, "out of memory for the weights table");
        fclose(fi);
        fclose(fb);
        return -1;
    }
    w->bin = fb;
    char line[1024];
    int have_layers = 0, have_experts = 0;
    while (fgets(line, sizeof(line), fi) != NULL) {
        if (line[0] == '#') {
            // Header lines may carry several key=value tokens separated by spaces.
            char * p = line;
            while (*p != '\0') {
                while (*p == '#' || *p == ' ') {
                    ++p;
                }
                if (*p == '\0') {
                    break;
                }
                char key[64] = {0};
                char val[128] = {0};
                if (sscanf(p, "%63[^=]=%127s", key, val) == 2) {
                    if (strcmp(key, "layers") == 0) {
                        w->layers = (uint32_t) atoi(val);
                        have_layers = 1;
                    } else if (strcmp(key, "experts") == 0) {
                        w->experts = (uint32_t) atoi(val);
                        have_experts = 1;
                    } else if (strcmp(key, "align") == 0) {
                        w->align = (uint64_t) strtoull(val, NULL, 10);
                    } else if (strcmp(key, "arch") == 0) {
                        snprintf(w->arch, sizeof(w->arch), "%s", val);
                    } else if (strcmp(key, "payload_sha256") == 0) {
                        snprintf(w->payload_sha, sizeof(w->payload_sha), "%s", val);
                    }
                }
                char * sp = strchr(p, ' ');
                if (sp == NULL) {
                    break;
                }
                p = sp + 1;
            }
            continue;
        }
        if (!have_layers || !have_experts) {
            snprintf(err, err_size, "index header is missing layers/experts");
            qrt_weights_close(w);
            fclose(fi);
            return -1;
        }
        unsigned L = 0, E = 0;
        unsigned long long off = 0, bytes = 0, g = 0, u = 0, d = 0;
        char sha[65] = {0};
        if (sscanf(line, "%u,%u,%llu,%llu,%llu,%llu,%llu,%64s", &L, &E, &off, &bytes, &g, &u, &d, sha) != 8) {
            continue;
        }
        if (w->loc == NULL) {
            w->loc = (qrt_expert_loc *) calloc((size_t) w->layers * w->experts, sizeof(qrt_expert_loc));
            if (w->loc == NULL) {
                snprintf(err, err_size, "out of memory for %u x %u expert slots", w->layers, w->experts);
                qrt_weights_close(w);
                fclose(fi);
                return -1;
            }
        }
        if (L >= w->layers || E >= w->experts) {
            snprintf(err, err_size, "index record %u,%u is outside %u x %u", L, E, w->layers, w->experts);
            qrt_weights_close(w);
            fclose(fi);
            return -1;
        }
        if (g + u + d != bytes) {
            snprintf(err, err_size, "expert %u,%u: gate+up+down does not match bytes", L, E);
            qrt_weights_close(w);
            fclose(fi);
            return -1;
        }
        qrt_expert_loc * l = &w->loc[(size_t) L * w->experts + E];
        l->offset = off;
        l->bytes  = bytes;
        l->gate   = g;
        l->up     = u;
        l->down   = d;
        if (off + bytes > w->payload_bytes) {
            w->payload_bytes = off + bytes;
        }
        ++w->records;
    }
    fclose(fi);

    const uint64_t need = (uint64_t) w->layers * w->experts;
    if (w->loc == NULL || w->records != need) {
        snprintf(err, err_size, "index has %u records, expected %llu", w->records,
                 (unsigned long long) need);
        qrt_weights_close(w);
        return -1;
    }
    // The payload on disk must be at least as long as the table claims.
    if (WSEEK(fb, 0, SEEK_END) != 0) {
        snprintf(err, err_size, "cannot size %s", bin_path);
        qrt_weights_close(w);
        return -1;
    }
    const long long file_bytes = WTELL(fb);
    if (file_bytes < 0 || (uint64_t) file_bytes < 64 + w->payload_bytes) {
        snprintf(err, err_size, "table claims %llu payload bytes, %s holds %lld",
                 (unsigned long long) w->payload_bytes, bin_path, file_bytes);
        qrt_weights_close(w);
        return -1;
    }
    *out = w;
    return 0;
}

void qrt_weights_close(qrt_weights * w) {
    if (w == NULL) {
        return;
    }
    if (w->bin != NULL) {
        fclose(w->bin);
    }
    free(w->loc);
    free(w);
}

uint32_t qrt_weights_layers(const qrt_weights * w) { return w->layers; }
uint32_t qrt_weights_experts(const qrt_weights * w) { return w->experts; }
uint64_t qrt_weights_payload_bytes(const qrt_weights * w) { return w->payload_bytes; }
uint32_t qrt_weights_records(const qrt_weights * w) { return w->records; }
const char * qrt_weights_arch(const qrt_weights * w) { return w->arch; }

int qrt_weights_locate(const qrt_weights * w, uint32_t layer, uint32_t expert,
                       qrt_expert_loc * out, char * err, size_t err_size) {
    if (layer >= w->layers || expert >= w->experts) {
        snprintf(err, err_size, "expert %u,%u is outside %u x %u", layer, expert, w->layers, w->experts);
        return -1;
    }
    *out = w->loc[(size_t) layer * w->experts + expert];
    if (out->bytes == 0) {
        snprintf(err, err_size, "expert %u,%u has no record", layer, expert);
        return -1;
    }
    return 0;
}

int qrt_weights_read_expert(const qrt_weights * w, uint32_t layer, uint32_t expert,
                            void * dst, size_t cap, char * err, size_t err_size) {
    qrt_expert_loc loc;
    if (qrt_weights_locate(w, layer, expert, &loc, err, err_size) != 0) {
        return -1;
    }
    if (loc.bytes > cap) {
        snprintf(err, err_size, "expert %u,%u needs %llu bytes, buffer holds %llu",
                 layer, expert, (unsigned long long) loc.bytes, (unsigned long long) cap);
        return -1;
    }
    const long long pos = (long long) (64 + loc.offset);
    if (WSEEK(w->bin, pos, SEEK_SET) != 0 ||
        fread(dst, 1, (size_t) loc.bytes, w->bin) != (size_t) loc.bytes) {
        snprintf(err, err_size, "short read for expert %u,%u at %llu", layer, expert,
                 (unsigned long long) pos);
        return -1;
    }
    return 0;
}
