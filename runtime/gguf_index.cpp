#include "gguf_index.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define QRT_FSEEK _fseeki64
#define QRT_FTELL _ftelli64
#else
#define QRT_FSEEK fseeko
#define QRT_FTELL ftello
#endif

struct qrt_gguf {
    FILE *            f;
    char              path[1024];
    uint64_t          file_size;
    uint32_t          version;
    uint32_t          alignment;
    uint64_t          data_offset;
    uint32_t          n_tensors;
    qrt_gguf_tensor * tensors;
    qrt_gguf_hparams  hp;
};

namespace {

struct qrt_reader {
    FILE *   f;
    uint64_t size;
    uint64_t pos;
    char *   err;
    size_t   err_size;
    int      failed;
};

struct qrt_type_info {
    const char * name;
    uint32_t     block;   // elements per block
    uint32_t     size;    // bytes per block
};

// GGML_TYPE_* ids as of this fork; 0 marks ids that no longer exist. Sizes are
// the ones gguf-py uses, so the account matches gguf_dims.py byte for byte.
const qrt_type_info TYPE_INFO[43] = {
    { "f32",       1,   4 }, { "f16",       1,   2 }, { "q4_0",     32,  18 },
    { "q4_1",     32,  20 }, { "?",         0,   0 }, { "?",         0,   0 },
    { "q5_0",     32,  22 }, { "q5_1",     32,  24 }, { "q8_0",     32,  34 },
    { "q8_1",     32,  36 }, { "q2_K",    256,  84 }, { "q3_K",    256, 110 },
    { "q4_K",    256, 144 }, { "q5_K",    256, 176 }, { "q6_K",    256, 210 },
    { "q8_K",    256, 292 }, { "iq2_xxs", 256,  66 }, { "iq2_xs",  256,  74 },
    { "iq3_xxs", 256,  98 }, { "iq1_s",   256,  50 }, { "iq4_nl",   32,  18 },
    { "iq3_s",   256, 110 }, { "iq2_s",   256,  82 }, { "iq4_xs",  256, 136 },
    { "i8",        1,   1 }, { "i16",       1,   2 }, { "i32",       1,   4 },
    { "i64",       1,   8 }, { "f64",       1,   8 }, { "iq1_m",   256,  56 },
    { "bf16",      1,   2 }, { "?",         0,   0 }, { "?",         0,   0 },
    { "?",         0,   0 }, { "tq1_0",   256,  54 }, { "tq2_0",   256,  66 },
    { "?",         0,   0 }, { "?",         0,   0 }, { "?",         0,   0 },
    { "mxfp4",    32,  17 }, { "nvfp4",    64,  36 }, { "q1_0",    128,  18 },
    { "q2_0",     64,  18 },
};

const uint32_t TYPE_COUNT = (uint32_t) (sizeof(TYPE_INFO) / sizeof(TYPE_INFO[0]));

const uint64_t MAX_STRING = 1ull << 20;
const uint64_t MAX_ARRAY  = 1ull << 26;
const uint32_t MAX_TENSORS = 1u << 20;
const uint32_t MAX_KV      = 1u << 20;

int r_fail(qrt_reader * r, const char * fmt, ...) {
    if (!r->failed) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(r->err, r->err_size, fmt, ap);
        va_end(ap);
        r->failed = 1;
    }
    return -1;
}

int r_read(qrt_reader * r, void * dst, size_t n) {
    if (r->failed) {
        return -1;
    }
    if (n > r->size - r->pos) {
        return r_fail(r, "header is truncated: wanted %llu bytes at %llu of %llu",
                      (unsigned long long) n, (unsigned long long) r->pos,
                      (unsigned long long) r->size);
    }
    if (fread(dst, 1, n, r->f) != n) {
        return r_fail(r, "read failed at offset %llu", (unsigned long long) r->pos);
    }
    r->pos += n;
    return 0;
}

int r_skip(qrt_reader * r, uint64_t n) {
    if (r->failed) {
        return -1;
    }
    if (n > r->size - r->pos) {
        return r_fail(r, "value of %llu bytes at %llu runs past the file (%llu bytes)",
                      (unsigned long long) n, (unsigned long long) r->pos,
                      (unsigned long long) r->size);
    }
    if (QRT_FSEEK(r->f, (long long) (r->pos + n), SEEK_SET) != 0) {
        return r_fail(r, "seek to %llu failed", (unsigned long long) (r->pos + n));
    }
    r->pos += n;
    return 0;
}

// GGUF is little-endian; assemble the value so the reader does not depend on the host.
int r_le(qrt_reader * r, uint64_t * out, int n) {
    uint8_t b[8];
    if (r_read(r, b, (size_t) n) != 0) {
        return -1;
    }
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) {
        v = (v << 8) | b[i];
    }
    *out = v;
    return 0;
}

int r_u32(qrt_reader * r, uint32_t * out) {
    uint64_t v = 0;
    if (r_le(r, &v, 4) != 0) {
        return -1;
    }
    *out = (uint32_t) v;
    return 0;
}

int r_u64(qrt_reader * r, uint64_t * out) {
    return r_le(r, out, 8);
}

int r_str(qrt_reader * r, char ** out) {
    uint64_t len = 0;
    if (r_u64(r, &len) != 0) {
        return -1;
    }
    if (len > MAX_STRING) {
        return r_fail(r, "string of %llu bytes is above the 1 MiB limit",
                      (unsigned long long) len);
    }
    if (len > r->size - r->pos) {
        return r_fail(r, "string of %llu bytes at %llu runs past the file",
                      (unsigned long long) len, (unsigned long long) r->pos);
    }
    char * s = (char *) malloc((size_t) len + 1);
    if (s == NULL) {
        return r_fail(r, "out of memory for a %llu-byte string", (unsigned long long) len);
    }
    if (len > 0 && fread(s, 1, (size_t) len, r->f) != (size_t) len) {
        free(s);
        return r_fail(r, "read of a %llu-byte string failed", (unsigned long long) len);
    }
    s[len] = '\0';
    r->pos += len;
    *out = s;
    return 0;
}

uint32_t elem_size(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;   // u8, i8, bool
        case 2: case 3:         return 2;   // u16, i16
        case 4: case 5: case 6: return 4;   // u32, i32, f32
        case 10: case 11: case 12: return 8; // u64, i64, f64
    }
    return 0;
}

int ends_with(const char * s, const char * suffix) {
    const size_t ns = strlen(s);
    const size_t nf = strlen(suffix);
    return ns >= nf && strcmp(s + ns - nf, suffix) == 0;
}

int apply_hparam(qrt_gguf * g, const char * key, int64_t v) {
    struct entry {
        const char * suffix;
        int32_t      offset;
        uint32_t     bit;
    };
    // Offsets are into qrt_gguf_hparams, which is a flat POD.
    static const entry ENTRIES[] = {
        { ".block_count",                (int32_t) offsetof(qrt_gguf_hparams, block_count),        QRT_HP_BLOCK_COUNT },
        { ".expert_count",               (int32_t) offsetof(qrt_gguf_hparams, expert_count),       QRT_HP_EXPERT_COUNT },
        { ".expert_used_count",          (int32_t) offsetof(qrt_gguf_hparams, expert_used_count),  QRT_HP_EXPERT_USED_COUNT },
        { ".embedding_length",           (int32_t) offsetof(qrt_gguf_hparams, n_embd),             QRT_HP_N_EMBD },
        { ".expert_feed_forward_length", (int32_t) offsetof(qrt_gguf_hparams, expert_ff),          QRT_HP_EXPERT_FF },
        { ".nextn_predict_layers",       (int32_t) offsetof(qrt_gguf_hparams, nextn),              QRT_HP_NEXTN },
        { ".attention.head_count_kv",    (int32_t) offsetof(qrt_gguf_hparams, head_count_kv),      QRT_HP_HEAD_COUNT_KV },
        { ".attention.key_length",       (int32_t) offsetof(qrt_gguf_hparams, key_length),         QRT_HP_KEY_LENGTH },
        { ".attention.value_length",     (int32_t) offsetof(qrt_gguf_hparams, value_length),       QRT_HP_VALUE_LENGTH },
        { ".context_length",             (int32_t) offsetof(qrt_gguf_hparams, context_length),     QRT_HP_CONTEXT_LENGTH },
        { ".full_attention_interval",    (int32_t) offsetof(qrt_gguf_hparams, full_attn_interval), QRT_HP_FULL_ATTN_INTERVAL },
    };
    for (size_t i = 0; i < sizeof(ENTRIES) / sizeof(ENTRIES[0]); ++i) {
        if (!ends_with(key, ENTRIES[i].suffix)) {
            continue;
        }
        if (v < 0 || v > 0x7fffffff) {
            return -1;
        }
        *(int32_t *) ((char *) &g->hp + ENTRIES[i].offset) = (int32_t) v;
        g->hp.present |= ENTRIES[i].bit;
        return 0;
    }
    return 0;
}

int read_int(qrt_reader * r, uint32_t type, int64_t * out) {
    uint64_t v = 0;
    switch (type) {
        case 0: case 7:                                  // u8, bool
            if (r_le(r, &v, 1) != 0) return -1;
            *out = (int64_t) v;
            return 0;
        case 1: {                                         // i8
            if (r_le(r, &v, 1) != 0) return -1;
            *out = (int64_t) (int8_t) (uint8_t) v;
            return 0;
        }
        case 2:                                           // u16
            if (r_le(r, &v, 2) != 0) return -1;
            *out = (int64_t) v;
            return 0;
        case 3: {                                         // i16
            if (r_le(r, &v, 2) != 0) return -1;
            *out = (int64_t) (int16_t) (uint16_t) v;
            return 0;
        }
        case 4:                                           // u32
            if (r_le(r, &v, 4) != 0) return -1;
            *out = (int64_t) v;
            return 0;
        case 5: {                                         // i32
            if (r_le(r, &v, 4) != 0) return -1;
            *out = (int64_t) (int32_t) (uint32_t) v;
            return 0;
        }
        case 10:                                          // u64
            if (r_le(r, &v, 8) != 0) return -1;
            *out = (int64_t) v;
            return 0;
        case 11:                                          // i64
            if (r_le(r, &v, 8) != 0) return -1;
            *out = (int64_t) v;
            return 0;
    }
    return r_fail(r, "metadata type %u is not an integer", type);
}

int read_value(qrt_reader * r, qrt_gguf * g, const char * key, uint32_t type) {
    switch (type) {
        case 0: case 1: case 2: case 3: case 4: case 5: case 7: case 10: case 11: {
            int64_t v = 0;
            if (read_int(r, type, &v) != 0) {
                return -1;
            }
            if (key != NULL && strcmp(key, "general.alignment") == 0) {
                if (v < 1 || v > 0x7fffffff) {
                    return r_fail(r, "general.alignment %lld is out of range", (long long) v);
                }
                g->alignment = (uint32_t) v;
                return 0;
            }
            if (key != NULL && apply_hparam(g, key, v) != 0) {
                return r_fail(r, "value %lld for '%s' does not fit an int32",
                              (long long) v, key);
            }
            return 0;
        }
        case 6:
            return r_skip(r, 4);
        case 12:
            return r_skip(r, 8);
        case 8: {
            char * s = NULL;
            if (r_str(r, &s) != 0) {
                return -1;
            }
            if (key != NULL && strcmp(key, "general.architecture") == 0) {
                snprintf(g->hp.arch, sizeof(g->hp.arch), "%s", s);
            }
            free(s);
            return 0;
        }
        case 9: {
            uint64_t elem = 0;
            uint64_t count = 0;
            if (r_le(r, &elem, 4) != 0 || r_u64(r, &count) != 0) {
                return -1;
            }
            if (elem > 12) {
                return r_fail(r, "array element type %llu is unknown", (unsigned long long) elem);
            }
            if (elem == 9) {
                return r_fail(r, "nested arrays are not part of GGUF");
            }
            if (count > MAX_ARRAY) {
                return r_fail(r, "array of %llu elements is above the 2^26 limit",
                              (unsigned long long) count);
            }
            if (elem == 8) {
                for (uint64_t i = 0; i < count; ++i) {
                    char * s = NULL;
                    if (r_str(r, &s) != 0) {
                        return -1;
                    }
                    free(s);
                }
                return 0;
            }
            const uint32_t esize = elem_size((uint32_t) elem);
            if (esize == 0) {
                return r_fail(r, "array element type %llu has no fixed size",
                              (unsigned long long) elem);
            }
            if (count > (r->size - r->pos) / esize) {
                return r_fail(r, "array of %llu x %u bytes runs past the file",
                              (unsigned long long) count, esize);
            }
            return r_skip(r, count * esize);
        }
    }
    return r_fail(r, "metadata type %u is unknown", type);
}

// ne0*ne1*ne2*ne3 with an explicit overflow check, then the block-aligned size.
int tensor_nbytes(const uint64_t ne[4], uint32_t n_dims, uint32_t type, uint64_t * out) {
    uint64_t total = 1;
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (ne[i] == 0) {
            return -1;
        }
        if (total > UINT64_MAX / ne[i]) {
            return -1;
        }
        total *= ne[i];
    }
    const uint32_t block = TYPE_INFO[type].block;
    if (total % block != 0) {
        return -1;
    }
    if (total / block > UINT64_MAX / TYPE_INFO[type].size) {
        return -1;
    }
    *out = total / block * TYPE_INFO[type].size;
    return 0;
}

int align_up(uint64_t v, uint64_t a, uint64_t * out) {
    const uint64_t rem = v % a;
    if (rem == 0) {
        *out = v;
        return 0;
    }
    if (v > UINT64_MAX - (a - rem)) {
        return -1;
    }
    *out = v + (a - rem);
    return 0;
}

// Returns the layer of a "blk.<N>." name, or -1 when the name does not have one.
int tensor_layer(const char * name, int32_t * layer, const char ** rest) {
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    char * end = NULL;
    const long v = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || v < 0 || v > 4096) {
        return -1;
    }
    *layer = (int32_t) v;
    if (rest != NULL) {
        *rest = end + 1;
    }
    return 0;
}

} // namespace

int qrt_gguf_open(const char * path, qrt_gguf ** out, char * err, size_t err_size) {
    *out = NULL;
    if (path == NULL || *path == '\0') {
        snprintf(err, err_size, "no GGUF path given");
        return -1;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(err, err_size, "cannot open '%s'", path);
        return -1;
    }
    if (QRT_FSEEK(f, 0, SEEK_END) != 0) {
        snprintf(err, err_size, "cannot seek '%s'", path);
        fclose(f);
        return -1;
    }
    const long long end = QRT_FTELL(f);
    if (end <= 0 || QRT_FSEEK(f, 0, SEEK_SET) != 0) {
        snprintf(err, err_size, "cannot size '%s'", path);
        fclose(f);
        return -1;
    }

    qrt_gguf * g = (qrt_gguf *) calloc(1, sizeof(qrt_gguf));
    if (g == NULL) {
        snprintf(err, err_size, "out of memory for the GGUF index");
        fclose(f);
        return -1;
    }
    g->f          = f;
    g->file_size  = (uint64_t) end;
    g->alignment  = 32;

    qrt_reader rd;
    rd.f        = f;
    rd.size     = g->file_size;
    rd.pos      = 0;
    rd.err      = err;
    rd.err_size = err_size;
    rd.failed   = 0;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    uint64_t data_bytes = 0;
    if (r_u32(&rd, &magic) != 0 || r_u32(&rd, &version) != 0 ||
        r_u64(&rd, &n_tensors) != 0 || r_u64(&rd, &n_kv) != 0) {
        goto fail;
    }
    if (magic != 0x46554747u) {
        r_fail(&rd, "'%s' is not a GGUF file (magic 0x%08x)", path, magic);
        goto fail;
    }
    if (version != 2 && version != 3) {
        r_fail(&rd, "GGUF version %u is not supported (2 or 3)", version);
        goto fail;
    }
    if (n_tensors == 0 || n_tensors > MAX_TENSORS) {
        r_fail(&rd, "tensor count %llu is outside 1..%u",
               (unsigned long long) n_tensors, MAX_TENSORS);
        goto fail;
    }
    if (n_kv > MAX_KV) {
        r_fail(&rd, "metadata count %llu is above %u", (unsigned long long) n_kv, MAX_KV);
        goto fail;
    }
    g->version   = version;
    g->n_tensors = (uint32_t) n_tensors;

    for (uint64_t i = 0; i < n_kv; ++i) {
        char * key = NULL;
        uint32_t type = 0;
        if (r_str(&rd, &key) != 0) {
            goto fail;
        }
        if (r_u32(&rd, &type) != 0) {
            free(key);
            goto fail;
        }
        const int rc = read_value(&rd, g, key, type);
        free(key);
        if (rc != 0) {
            goto fail;
        }
    }

    if (g->alignment == 0 || (g->alignment & (g->alignment - 1)) != 0 || g->alignment > (1u << 20)) {
        r_fail(&rd, "alignment %u is not a power of two below 2^20", g->alignment);
        goto fail;
    }

    g->tensors = (qrt_gguf_tensor *) calloc(g->n_tensors, sizeof(qrt_gguf_tensor));
    if (g->tensors == NULL) {
        r_fail(&rd, "out of memory for %u tensor descriptors", g->n_tensors);
        goto fail;
    }
    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        qrt_gguf_tensor * t = &g->tensors[i];
        uint64_t n_dims = 0;
        uint64_t type = 0;
        if (r_str(&rd, &t->name) != 0 || r_le(&rd, &n_dims, 4) != 0) {
            goto fail;
        }
        if (n_dims < 1 || n_dims > 4) {
            r_fail(&rd, "tensor '%s' has %llu dims, GGUF allows 1..4",
                   t->name, (unsigned long long) n_dims);
            goto fail;
        }
        t->n_dims = (uint32_t) n_dims;
        for (uint32_t d = 0; d < 4; ++d) {
            t->ne[d] = 1;
        }
        for (uint32_t d = 0; d < t->n_dims; ++d) {
            if (r_u64(&rd, &t->ne[d]) != 0) {
                goto fail;
            }
        }
        if (r_le(&rd, &type, 4) != 0 || r_u64(&rd, &t->offset) != 0) {
            goto fail;
        }
        if (type >= TYPE_COUNT || TYPE_INFO[type].size == 0) {
            r_fail(&rd, "tensor '%s' has type %llu, which this build does not know",
                   t->name, (unsigned long long) type);
            goto fail;
        }
        t->type = (uint32_t) type;
        if (tensor_nbytes(t->ne, t->n_dims, t->type, &t->nbytes) != 0) {
            r_fail(&rd, "tensor '%s' has dims that do not fit its type %s",
                   t->name, TYPE_INFO[t->type].name);
            goto fail;
        }
    }

    if (align_up(rd.pos, g->alignment, &g->data_offset) != 0 || g->data_offset > g->file_size) {
        r_fail(&rd, "tensor table ends at %llu, past the %llu-byte file",
               (unsigned long long) rd.pos, (unsigned long long) g->file_size);
        goto fail;
    }
    data_bytes = g->file_size - g->data_offset;
    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        qrt_gguf_tensor * t = &g->tensors[i];
        if (t->offset > data_bytes || t->nbytes > data_bytes - t->offset) {
            r_fail(&rd, "tensor '%s' needs bytes %llu..%llu of %llu data bytes",
                   t->name, (unsigned long long) t->offset,
                   (unsigned long long) (t->offset + t->nbytes),
                   (unsigned long long) data_bytes);
            goto fail;
        }
        t->offset += g->data_offset;
    }
    // Duplicate names would make qrt_gguf_find ambiguous; 753 tensors is cheap to scan.
    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        for (uint32_t j = i + 1; j < g->n_tensors; ++j) {
            if (strcmp(g->tensors[i].name, g->tensors[j].name) == 0) {
                r_fail(&rd, "tensor '%s' appears twice", g->tensors[i].name);
                goto fail;
            }
        }
    }

    // The index is metadata only; a later read reopens the file once (read_range).
    fclose(f);
    g->f = NULL;
    snprintf(g->path, sizeof(g->path), "%s", path);
    *out = g;
    return 0;

fail:
    qrt_gguf_close(g);
    return -1;
}

void qrt_gguf_close(qrt_gguf * g) {
    if (g == NULL) {
        return;
    }
    if (g->tensors != NULL) {
        for (uint32_t i = 0; i < g->n_tensors; ++i) {
            free(g->tensors[i].name);
        }
        free(g->tensors);
    }
    if (g->f != NULL) {
        fclose(g->f);
    }
    free(g);
}

uint32_t qrt_gguf_version(const qrt_gguf * g) {
    return g->version;
}

uint32_t qrt_gguf_tensor_count(const qrt_gguf * g) {
    return g->n_tensors;
}

uint64_t qrt_gguf_data_offset(const qrt_gguf * g) {
    return g->data_offset;
}

uint64_t qrt_gguf_file_size(const qrt_gguf * g) {
    return g->file_size;
}

const qrt_gguf_tensor * qrt_gguf_tensor_at(const qrt_gguf * g, uint32_t index) {
    return index < g->n_tensors ? &g->tensors[index] : NULL;
}

const qrt_gguf_tensor * qrt_gguf_find(const qrt_gguf * g, const char * name) {
    if (g == NULL || name == NULL) {
        return NULL;
    }
    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) {
            return &g->tensors[i];
        }
    }
    return NULL;
}

const qrt_gguf_hparams * qrt_gguf_hparams_of(const qrt_gguf * g) {
    return &g->hp;
}

const char * qrt_gguf_type_name(uint32_t type) {
    return type < TYPE_COUNT ? TYPE_INFO[type].name : "?";
}

int qrt_gguf_require(const qrt_gguf * g, const char * name, const qrt_gguf_tensor ** out,
                     char * err, size_t err_size) {
    const qrt_gguf_tensor * t = qrt_gguf_find(g, name);
    if (t == NULL) {
        snprintf(err, err_size, "tensor '%s' is not in the file", name != NULL ? name : "");
        return -1;
    }
    if (out != NULL) {
        *out = t;
    }
    return 0;
}

int qrt_gguf_read_range(qrt_gguf * g, uint64_t file_offset, void * dst, size_t bytes,
                        char * err, size_t err_size) {
    if (bytes > 0 && dst == NULL) {
        snprintf(err, err_size, "read of %llu bytes needs a destination",
                 (unsigned long long) bytes);
        return -1;
    }
    if (file_offset > g->file_size || bytes > g->file_size - file_offset) {
        snprintf(err, err_size, "read of %llu bytes at %llu runs past the %llu byte file",
                 (unsigned long long) bytes, (unsigned long long) file_offset,
                 (unsigned long long) g->file_size);
        return -1;
    }
    if (g->f == NULL) {
        g->f = fopen(g->path, "rb");
        if (g->f == NULL) {
            snprintf(err, err_size, "cannot reopen %s for reading", g->path);
            return -1;
        }
    }
    if (QRT_FSEEK(g->f, (long long) file_offset, SEEK_SET) != 0) {
        snprintf(err, err_size, "cannot seek to %llu", (unsigned long long) file_offset);
        return -1;
    }
    if (bytes > 0 && fread(dst, 1, bytes, g->f) != bytes) {
        snprintf(err, err_size, "short read of %llu bytes at %llu",
                 (unsigned long long) bytes, (unsigned long long) file_offset);
        return -1;
    }
    return 0;
}

int qrt_gguf_read_tensor(qrt_gguf * g, const qrt_gguf_tensor * t, void * dst,
                         size_t cap, char * err, size_t err_size) {
    if (t == NULL) {
        snprintf(err, err_size, "no tensor given");
        return -1;
    }
    if (t->nbytes > cap) {
        snprintf(err, err_size, "tensor %s needs %llu bytes, the buffer holds %llu", t->name,
                 (unsigned long long) t->nbytes, (unsigned long long) cap);
        return -1;
    }
    return qrt_gguf_read_range(g, t->offset, dst, (size_t) t->nbytes, err, err_size);
}

int qrt_gguf_account_compute(const qrt_gguf * g, qrt_gguf_account * out, char * err, size_t err_size) {
    memset(out, 0, sizeof(*out));
    const uint32_t need = QRT_HP_BLOCK_COUNT | QRT_HP_EXPERT_COUNT;
    if ((g->hp.present & need) != need) {
        snprintf(err, err_size, "metadata lacks block_count or expert_count");
        return -1;
    }
    if (g->hp.block_count <= 0 || g->hp.expert_count <= 0) {
        snprintf(err, err_size, "block_count %d or expert_count %d is not positive",
                 g->hp.block_count, g->hp.expert_count);
        return -1;
    }
    const uint32_t n_layers = (uint32_t) g->hp.block_count;
    uint8_t * expert_seen = (uint8_t *) calloc(n_layers, 1);
    uint8_t * attn_seen   = (uint8_t *) calloc(n_layers, 1);
    if (expert_seen == NULL || attn_seen == NULL) {
        free(expert_seen);
        free(attn_seen);
        snprintf(err, err_size, "out of memory for %u layer flags", n_layers);
        return -1;
    }

    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        const qrt_gguf_tensor * t = &g->tensors[i];
        int32_t layer = -1;
        const char * rest = NULL;
        if (strstr(t->name, "_exps") != NULL) {
            out->expert_bytes += t->nbytes;
            if (tensor_layer(t->name, &layer, &rest) != 0 || (uint32_t) layer >= n_layers) {
                snprintf(err, err_size, "expert tensor '%s' has no usable blk.N prefix", t->name);
                free(expert_seen);
                free(attn_seen);
                return -1;
            }
            expert_seen[layer] = 1;
        } else if (strcmp(t->name, "token_embd.weight") == 0) {
            out->token_embd_bytes += t->nbytes;
        } else {
            out->dense_bytes += t->nbytes;
        }
        if (tensor_layer(t->name, &layer, &rest) == 0 && (uint32_t) layer < n_layers &&
            strncmp(rest, "attn_k", 6) == 0 && (rest[6] == '.' || rest[6] == '\0')) {
            attn_seen[layer] = 1;
        }
    }

    for (uint32_t l = 0; l < n_layers; ++l) {
        out->expert_layers += expert_seen[l] ? 1 : 0;
    }
    free(expert_seen);

    out->main_layers = (uint32_t) g->hp.block_count;
    if (g->hp.present & QRT_HP_NEXTN) {
        if (g->hp.nextn < 0 || g->hp.nextn > g->hp.block_count) {
            free(attn_seen);
            snprintf(err, err_size, "nextn_predict_layers %d is outside 0..%d",
                     g->hp.nextn, g->hp.block_count);
            return -1;
        }
        out->main_layers = (uint32_t) (g->hp.block_count - g->hp.nextn);
    }
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (!attn_seen[l]) {
            continue;
        }
        if (l < out->main_layers) {
            ++out->full_attn_main;
        } else {
            ++out->full_attn_nextn;
        }
    }
    free(attn_seen);

    if (out->expert_layers == 0) {
        snprintf(err, err_size, "no tensor name contains _exps");
        return -1;
    }
    out->per_expert_bytes = (double) out->expert_bytes /
                            ((double) out->expert_layers * (double) g->hp.expert_count);
    return 0;
}

void qrt_gguf_print_report(const qrt_gguf * g, FILE * f) {
    const qrt_gguf_hparams * hp = &g->hp;
    fprintf(f, "# gguf version=%u tensors=%u alignment=%u data_offset=%llu file_size=%llu\n",
            g->version, g->n_tensors, g->alignment,
            (unsigned long long) g->data_offset, (unsigned long long) g->file_size);
    fprintf(f, "# meta arch=%s", hp->arch[0] != '\0' ? hp->arch : "?");
    if (hp->present & QRT_HP_BLOCK_COUNT)        fprintf(f, " block_count=%d", hp->block_count);
    if (hp->present & QRT_HP_NEXTN)              fprintf(f, " nextn=%d", hp->nextn);
    if (hp->present & QRT_HP_EXPERT_COUNT)       fprintf(f, " expert_count=%d", hp->expert_count);
    if (hp->present & QRT_HP_EXPERT_USED_COUNT)  fprintf(f, " expert_used_count=%d", hp->expert_used_count);
    if (hp->present & QRT_HP_N_EMBD)             fprintf(f, " n_embd=%d", hp->n_embd);
    if (hp->present & QRT_HP_EXPERT_FF)          fprintf(f, " expert_ff=%d", hp->expert_ff);
    if (hp->present & QRT_HP_HEAD_COUNT_KV)      fprintf(f, " head_count_kv=%d", hp->head_count_kv);
    if (hp->present & QRT_HP_KEY_LENGTH)         fprintf(f, " key_length=%d", hp->key_length);
    if (hp->present & QRT_HP_VALUE_LENGTH)       fprintf(f, " value_length=%d", hp->value_length);
    if (hp->present & QRT_HP_CONTEXT_LENGTH)     fprintf(f, " context_length=%d", hp->context_length);
    if (hp->present & QRT_HP_FULL_ATTN_INTERVAL) fprintf(f, " full_attention_interval=%d", hp->full_attn_interval);
    fprintf(f, "\n");

    qrt_gguf_account a;
    char err[256] = {0};
    if (qrt_gguf_account_compute(g, &a, err, sizeof(err)) == 0) {
        fprintf(f, "# account expert_bytes=%llu expert_layers=%u per_expert_bytes=%.1f "
                   "dense_bytes=%llu token_embd_bytes=%llu main_layers=%u "
                   "full_attn_main=%u full_attn_nextn=%u\n",
                (unsigned long long) a.expert_bytes, a.expert_layers, a.per_expert_bytes,
                (unsigned long long) a.dense_bytes, (unsigned long long) a.token_embd_bytes,
                a.main_layers, a.full_attn_main, a.full_attn_nextn);
    } else {
        fprintf(f, "# account unavailable: %s\n", err);
    }

    fprintf(f, "name,n_dims,ne0,ne1,ne2,ne3,type,type_name,offset,nbytes\n");
    for (uint32_t i = 0; i < g->n_tensors; ++i) {
        const qrt_gguf_tensor * t = &g->tensors[i];
        fprintf(f, "%s,%u,%llu,%llu,%llu,%llu,%u,%s,%llu,%llu\n",
                t->name, t->n_dims,
                (unsigned long long) t->ne[0], (unsigned long long) t->ne[1],
                (unsigned long long) t->ne[2], (unsigned long long) t->ne[3],
                t->type, qrt_gguf_type_name(t->type),
                (unsigned long long) t->offset, (unsigned long long) t->nbytes);
    }
}
