#include "weights.h"

#include "gguf_index.h"

#include <stdarg.h>
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
    uint64_t      min_expert;
    uint64_t      max_expert;
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
        if (w->min_expert == 0 || bytes < w->min_expert) {
            w->min_expert = bytes;
        }
        if (bytes > w->max_expert) {
            w->max_expert = bytes;
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
uint64_t qrt_weights_max_expert_bytes(const qrt_weights * w) { return w->max_expert; }
uint64_t qrt_weights_min_expert_bytes(const qrt_weights * w) { return w->min_expert; }

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

namespace {

const uint64_t DENSE_ALIGN = 256;   // keeps every tensor on a storage-buffer offset

struct qrt_dense_tensor {
    char     name[96];
    uint64_t offset;       // inside the device buffer
    uint64_t bytes;
    uint64_t file_offset;
};

} // namespace

struct qrt_dense {
    qrt_gguf *        gguf;
    VkBuffer          buf;
    VkDeviceMemory    mem;
    uint64_t          bytes;
    qrt_dense_tensor * t;
    uint32_t          n;
};

int qrt_dense_open(qrt_vk_dev * dev, const char * gguf_path, uint64_t staging_bytes,
                   qrt_dense ** out, char * err, size_t err_size) {
    *out = NULL;
    if (staging_bytes == 0) {
        snprintf(err, err_size, "dense residency needs a non-zero staging size");
        return -1;
    }
    qrt_dense * d = (qrt_dense *) calloc(1, sizeof(qrt_dense));
    if (d == NULL) {
        snprintf(err, err_size, "out of memory for the dense table");
        return -1;
    }
    if (qrt_gguf_open(gguf_path, &d->gguf, err, err_size) != 0) {
        free(d);
        return -1;
    }

    const uint32_t n_all = qrt_gguf_tensor_count(d->gguf);
    d->t = (qrt_dense_tensor *) calloc(n_all, sizeof(qrt_dense_tensor));
    if (d->t == NULL) {
        snprintf(err, err_size, "out of memory for %u dense slots", n_all);
        qrt_dense_close(dev, d);
        return -1;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_all; ++i) {
        const qrt_gguf_tensor * t = qrt_gguf_tensor_at(d->gguf, i);
        // Experts go through the repacked store, token_embd is not in the dense
        // bucket of design 5.5 (it would need 515 MiB more).
        if (strstr(t->name, "_exps") != NULL || strcmp(t->name, "token_embd.weight") == 0) {
            continue;
        }
        qrt_dense_tensor * s = &d->t[d->n];
        snprintf(s->name, sizeof(s->name), "%s", t->name);
        s->bytes       = t->nbytes;
        s->file_offset = t->offset;
        s->offset      = total;
        total = (total + t->nbytes + DENSE_ALIGN - 1) / DENSE_ALIGN * DENSE_ALIGN;
        ++d->n;
    }
    if (d->n == 0 || total == 0) {
        snprintf(err, err_size, "%s has no dense tensors", gguf_path);
        qrt_dense_close(dev, d);
        return -1;
    }
    d->bytes = total;
    if (qrt_vk_alloc(dev, total, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &d->buf, &d->mem, err, err_size) != 0) {
        qrt_dense_close(dev, d);
        return -1;
    }
    qrt_vk_staging * stg = NULL;
    if (qrt_vk_staging_create(dev, staging_bytes, &stg, err, err_size) != 0) {
        qrt_dense_close(dev, d);
        return -1;
    }
    unsigned char * map = (unsigned char *) qrt_vk_staging_map(stg);
    int rc = 0;
    for (uint32_t i = 0; i < d->n && rc == 0; ++i) {
        const qrt_dense_tensor * s = &d->t[i];
        for (uint64_t done = 0; done < s->bytes; ) {
            const uint64_t chunk = (s->bytes - done) < staging_bytes
                                   ? s->bytes - done : staging_bytes;
            if (qrt_gguf_read_range(d->gguf, s->file_offset + done, map, (size_t) chunk,
                                    err, err_size) != 0 ||
                qrt_vk_upload_mapped(dev, stg, d->buf, s->offset + done, chunk,
                                     err, err_size) != 0) {
                rc = -1;
                break;
            }
            done += chunk;
        }
    }
    qrt_vk_staging_destroy(dev, stg);
    if (rc != 0) {
        qrt_dense_close(dev, d);
        return -1;
    }
    *out = d;
    return 0;
}

void qrt_dense_close(qrt_vk_dev * dev, qrt_dense * d) {
    if (d == NULL) {
        return;
    }
    if (dev != NULL) {
        if (d->mem != VK_NULL_HANDLE) {
            qrt_vk_free(dev, d->buf, d->mem);
        }
    }
    if (d->gguf != NULL) {
        qrt_gguf_close(d->gguf);
    }
    free(d->t);
    free(d);
}

uint64_t qrt_dense_bytes(const qrt_dense * d) { return d != NULL ? d->bytes : 0; }
uint32_t qrt_dense_tensor_count(const qrt_dense * d) { return d != NULL ? d->n : 0; }

const char * qrt_dense_tensor_name(const qrt_dense * d, uint32_t index) {
    if (d == NULL || index >= d->n) {
        return "";
    }
    return d->t[index].name;
}

int qrt_dense_locate(const qrt_dense * d, const char * name, uint64_t * offset, uint64_t * bytes) {
    if (d == NULL || name == NULL) {
        return -1;
    }
    for (uint32_t i = 0; i < d->n; ++i) {
        if (strcmp(d->t[i].name, name) == 0) {
            if (offset != NULL) {
                *offset = d->t[i].offset;
            }
            if (bytes != NULL) {
                *bytes = d->t[i].bytes;
            }
            return 0;
        }
    }
    return -1;
}

VkBuffer qrt_dense_buffer(const qrt_dense * d) { return d != NULL ? d->buf : VK_NULL_HANDLE; }

uint64_t qrt_dense_verify(qrt_vk_dev * dev, qrt_dense * d, qrt_vk_staging * stg,
                          uint64_t max_bytes, char * err, size_t err_size) {
    if (d == NULL || stg == NULL) {
        snprintf(err, err_size, "dense verify needs the residency and a staging buffer");
        return 0;
    }
    const uint64_t chunk = qrt_vk_staging_bytes(stg);
    if (chunk == 0) {
        snprintf(err, err_size, "staging buffer is empty");
        return 0;
    }
    unsigned char * map = (unsigned char *) qrt_vk_staging_map(stg);
    unsigned char * host = (unsigned char *) malloc((size_t) chunk);
    if (host == NULL || map == NULL) {
        free(host);
        snprintf(err, err_size, "out of memory for a %llu byte compare buffer",
                 (unsigned long long) chunk);
        return 0;
    }
    // Evenly spaced probes over the tensor table plus the tail of the last tensor:
    // a stop-on-first-error stream would leave the end of the buffer unchecked.
    uint64_t checked = 0;
    const uint32_t stride = d->n > 16 ? d->n / 16 : 1;
    for (uint32_t i = 0; i < d->n && checked < max_bytes; i += stride) {
        const qrt_dense_tensor * s = &d->t[i];
        const uint64_t want = s->bytes < chunk ? s->bytes : chunk;
        if (qrt_gguf_read_range(d->gguf, s->file_offset, host, (size_t) want, err, err_size) != 0 ||
            qrt_vk_download_mapped(dev, stg, d->buf, s->offset, want, err, err_size) != 0) {
            free(host);
            return checked;
        }
        if (memcmp(host, map, (size_t) want) != 0) {
            snprintf(err, err_size, "dense tensor %s differs in its first %llu bytes",
                     s->name, (unsigned long long) want);
            free(host);
            return checked;
        }
        checked += want;
    }
    {
        const qrt_dense_tensor * s = &d->t[d->n - 1];
        const uint64_t want = s->bytes < chunk ? s->bytes : chunk;
        const uint64_t at = s->bytes - want;
        if (qrt_gguf_read_range(d->gguf, s->file_offset + at, host, (size_t) want,
                                err, err_size) != 0 ||
            qrt_vk_download_mapped(dev, stg, d->buf, s->offset + at, want, err, err_size) != 0) {
            free(host);
            return checked;
        }
        if (memcmp(host, map, (size_t) want) != 0) {
            snprintf(err, err_size, "dense tensor %s differs in its last %llu bytes",
                     s->name, (unsigned long long) want);
            free(host);
            return checked;
        }
        checked += want;
    }
    free(host);
    return checked;
}
