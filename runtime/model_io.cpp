#include "model_io.h"

#include "model.h"
#include "model_file.h"
#include "qrt_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

namespace {

// One handle per scope: an early return still closes the file, and close() releases it when
// the caller has to report a failed close itself.
struct file_scope {
    qrt_model_file * f = nullptr;

    file_scope(const char * path, const char * mode) : f(qrt_model_file_open(path, mode)) {}
    file_scope(const file_scope &) = delete;
    file_scope & operator=(const file_scope &) = delete;
    ~file_scope() {
        if (f != nullptr) {
            qrt_model_file_close(f);
        }
    }
    int close() {
        const int rc = qrt_model_file_close(f);
        f = nullptr;
        return rc;
    }
    explicit operator bool() const { return f != nullptr; }
};

} // namespace

int qrt_model_write_dump(const char * path, uint32_t V, uint32_t n_rows, uint32_t start,
                         const uint32_t * ids, const float * logits, size_t row_stride,
                         char * err, size_t err_size) {
    file_scope out(path, "wb");
    if (!out) {
        return fail(err, err_size, "cannot write '%s'", path);
    }
    const char magic[8] = { 'Q', 'R', 'T', 'L', 'O', 'G', '1', '\0' };
    const uint32_t version = 1;
    const uint32_t hdr[4] = { version, V, n_rows, start };
    bool short_write = qrt_model_file_write(out.f, magic, 1, 8) != 8 ||
                       qrt_model_file_write(out.f, hdr, sizeof(uint32_t), 4) != 4 ||
                       qrt_model_file_write(out.f, ids + start, sizeof(uint32_t),
                                            n_rows + 1u) != n_rows + 1u;
    for (uint32_t r = 0; r < n_rows && !short_write; ++r) {
        short_write = qrt_model_file_write(out.f, logits + (size_t) r * row_stride,
                                          sizeof(float), V) != V;
    }
    const int close_rc = out.close();
    if (short_write || close_rc != 0) {
        return fail(err, err_size, "cannot write '%s': %s%s", path,
                    short_write ? "short write" : "", close_rc != 0 ? " close failed" : "");
    }
    return 0;
}

int qrt_model_write_ids(const char * path, const uint32_t * ids, size_t count,
                        char * err, size_t err_size) {
    if (path == NULL || *path == '\0' || (ids == NULL && count != 0)) {
        return fail(err, err_size, "the id output needs a path and ids");
    }
    file_scope out(path, "wb");
    if (!out) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    std::string text;
    char line[16];
    for (size_t i = 0; i < count; ++i) {
        const int n = snprintf(line, sizeof(line), "%u\n", ids[i]);
        if (n <= 0) {
            return fail(err, err_size, "cannot format token id %zu", i);
        }
        text.append(line, (size_t) n);
    }
    const size_t written = text.empty() ? 0 : qrt_model_file_write(out.f, text.data(), 1,
                                                                   text.size());
    const int close_rc = out.close();
    if (written != text.size()) {
        return fail(err, err_size, "cannot write '%s': short write", path);
    }
    if (close_rc != 0) {
        return fail(err, err_size, "cannot write '%s': close failed", path);
    }
    return 0;
}

int qrt_model_read_ids(const char * path, uint32_t ** ids, uint32_t * count,
                       char * err, size_t err_size) {
    if (ids != NULL) *ids = NULL;
    if (count != NULL) *count = 0;
    if (path == NULL || *path == '\0' || ids == NULL || count == NULL) {
        return fail(err, err_size, "token input needs a path and output pointers");
    }
    file_scope in(path, "rb");
    if (!in) {
        return fail(err, err_size, "cannot open '%s'", path);
    }
    std::vector<uint32_t> v;
    uint32_t value = 0;
    bool in_token = false;
    for (;;) {
        const int c = qrt_model_file_getc(in.f);
        if (c >= '0' && c <= '9') {
            const uint32_t digit = (uint32_t) (c - '0');
            if (value > ((uint32_t) INT32_MAX - digit) / 10u) {
                return fail(err, err_size, "'%s': token exceeds INT32_MAX", path);
            }
            value = value * 10u + digit;
            in_token = true;
            continue;
        }
        if (c != EOF && c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') {
            return fail(err, err_size, "'%s': token ids must be complete unsigned decimal integers", path);
        }
        if (c == EOF && qrt_model_file_error(in.f)) {
            return fail(err, err_size, "cannot read '%s'", path);
        }
        if (in_token) {
            if (v.size() >= UINT32_MAX || v.size() >= SIZE_MAX / sizeof(uint32_t)) {
                return fail(err, err_size, "'%s': too many token ids", path);
            }
            v.push_back(value);
            value = 0;
            in_token = false;
        }
        if (c == EOF) break;
    }
    if (in.close() != 0) {
        return fail(err, err_size, "cannot close token input '%s'", path);
    }
    if (v.empty()) {
        return fail(err, err_size, "'%s' holds no token ids", path);
    }
    uint32_t * result = (uint32_t *) malloc(v.size() * sizeof(uint32_t));
    if (result == NULL) {
        return fail(err, err_size, "out of memory for %zu token ids", v.size());
    }
    memcpy(result, v.data(), v.size() * sizeof(uint32_t));
    *ids = result;
    *count = (uint32_t) v.size();
    return 0;
}
