#include "spec.h"

#include "layer.h"
#include "qrt_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

// A verified pass carries rows = drafts + 1: the token the cycle starts from, then one row
// per draft. verified[i] is the argmax of row i, which predicts the token of row i+1, so it
// lines up with draft[i]. The bonus is verified[accepted], always inside the array.
uint32_t qrt_spec_accept(const uint32_t * draft, const uint32_t * verified, uint32_t n) {
    uint32_t a = 0;
    while (a < n && draft[a] == verified[a]) {
        ++a;
    }
    return a;
}

int qrt_spec_emit(const uint32_t * draft, const uint32_t * verified, uint32_t n, uint32_t * out,
                  uint32_t * n_out) {
    if (out == NULL || n_out == NULL || verified == NULL || draft == NULL || n == 0) {
        return -1;
    }
    const uint32_t a = qrt_spec_accept(draft, verified, n);
    for (uint32_t i = 0; i < a; ++i) {
        out[i] = draft[i];
    }
    out[a] = verified[a];
    *n_out = a + 1u;
    return 0;
}

#ifdef QRT_HAVE_VULKAN
// The rollback itself is the layers' state ring, so it only exists where the layers do; the
// accept decision above is the part that can be pinned down without a device.
int qrt_spec_rollback(qrt_layer ** layers, uint32_t n_layers, uint32_t kept, char * err,
                      size_t err_size) {
    if (layers == NULL) {
        return fail(err, err_size, "rollback needs the layer list");
    }
    if (kept == 0) {
        return fail(err, err_size, "rollback to 0 rows leaves nothing to continue from");
    }
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (layers[l] != NULL && qrt_layer_rollback(layers[l], kept, err, err_size) != 0) {
            return fail(err, err_size, "layer %u: %s", l, err);
        }
    }
    return 0;
}
#endif

struct qrt_spec_vocab {
    std::vector<uint32_t> ids;
};

int qrt_spec_vocab_open(const char * path, uint32_t n_allowed, qrt_spec_vocab ** out,
                        char * err, size_t err_size) {
    if (out == NULL) {
        return -1;
    }
    *out = NULL;
    if (path == NULL || n_allowed == 0) {
        return fail(err, err_size, "a restricted draft vocabulary needs a path and a size");
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open the draft vocabulary '%s'", path);
    }
    std::vector<uint32_t> raw;
    int ok = 1;
    for (uint32_t i = 0; ok && i < n_allowed; ++i) {
        uint32_t id = 0;
        // the file is little-endian int32 ids in ascending order (M1-3a), which is native
        // for every host this runtime builds on
        ok = fread(&id, sizeof(id), 1, f) == 1 ? 1 : 0;
        raw.push_back(id);
    }
    fclose(f);
    if (!ok) {
        return fail(err, err_size, "'%s' holds fewer than %u ids", path, n_allowed);
    }
    for (uint32_t i = 1; i < n_allowed; ++i) {
        if (raw[i] <= raw[i - 1]) {
            return fail(err, err_size, "draft vocabulary '%s' is not ascending at %u (%u <= %u)",
                        path, i, raw[i], raw[i - 1]);
        }
    }
    qrt_spec_vocab * v = new qrt_spec_vocab();
    v->ids.swap(raw);
    *out = v;
    return 0;
}

void qrt_spec_vocab_free(qrt_spec_vocab * v) {
    delete v;
}

uint32_t qrt_spec_vocab_count(const qrt_spec_vocab * v) {
    return v == NULL ? 0 : (uint32_t) v->ids.size();
}

const uint32_t * qrt_spec_vocab_ids(const qrt_spec_vocab * v) {
    return v == NULL ? NULL : v->ids.data();
}
