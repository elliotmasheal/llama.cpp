// CPU contract for the speculation decision (M3-3). The rollback it drives is device state,
// so it is gated by --spec-check against a serial decode; this file pins the accept rule
// itself, including where an adversarial draft first disagrees.

#include "spec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); \
} } while (0)

namespace {

// One cycle: n drafts, n+1 verified rows. The emitted stream and the rows that stay are the
// two things the caller has to get right, so both come back from emit.
void cycle(const std::vector<uint32_t> & draft, const std::vector<uint32_t> & verified,
           uint32_t * accepted, std::vector<uint32_t> * out) {
    const uint32_t n = (uint32_t) draft.size();
    CHECK(verified.size() >= n + 1);
    const uint32_t a = qrt_spec_accept(draft.data(), verified.data(), n);
    uint32_t n_out = 0;
    out->assign(n + 1u, 0xffffffffu);
    CHECK(qrt_spec_emit(draft.data(), verified.data(), n, out->data(), &n_out) == 0);
    out->resize(n_out);
    *accepted = a;
}

int fails = 0;

void require(int cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "spec contract: FAIL %s\n", what);
        ++fails;
    }
}

// The draft is the verified tail shifted by one, which is what a perfect MTP head would
// produce: every row but the last one agrees.
void perfect_draft(uint32_t n) {
    std::vector<uint32_t> verified(n + 1u);
    for (uint32_t i = 0; i <= n; ++i) {
        verified[i] = 100u + i;
    }
    std::vector<uint32_t> draft(verified.begin(), verified.end() - 1);
    uint32_t a = 0;
    std::vector<uint32_t> out;
    cycle(draft, verified, &a, &out);
    require(a == n, "a perfect draft accepts every row");
    require(out.size() == n + 1u, "a perfect draft emits the bonus too");
    for (uint32_t i = 0; i < out.size(); ++i) {
        require(out[i] == verified[i], "emitted ids are the verified rows");
    }
}

// Exactly one disagreement at position i: everything before it is kept, the token at i is
// the verified one, and the rows after i are dropped.
void one_miss(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<uint32_t> verified(n + 1u);
        for (uint32_t t = 0; t <= n; ++t) {
            verified[t] = 7u * t + 1u;
        }
        std::vector<uint32_t> draft(verified.begin(), verified.end() - 1);
        draft[i] = verified[i] + 1000u;
        uint32_t a = 0;
        std::vector<uint32_t> out;
        cycle(draft, verified, &a, &out);
        require(a == i, "the prefix stops at the first disagreement");
        require(out.size() == i + 1u, "the cycle emits accepted + 1 tokens");
        for (uint32_t t = 0; t < i; ++t) {
            require(out[t] == draft[t], "accepted tokens are the drafts");
        }
        require(out[i] == verified[i], "the token at the break is the verified one");
    }
}

// Every draft wrong: nothing is accepted, the pass still yields one token.
void all_wrong(uint32_t n) {
    std::vector<uint32_t> verified(n + 1u), draft(n);
    for (uint32_t t = 0; t <= n; ++t) {
        verified[t] = t;
    }
    for (uint32_t t = 0; t < n; ++t) {
        draft[t] = 500u + t;
    }
    uint32_t a = 0;
    std::vector<uint32_t> out;
    cycle(draft, verified, &a, &out);
    require(a == 0, "an all-wrong draft accepts nothing");
    require(out.size() == 1u && out[0] == verified[0], "the bonus still comes from row 0");
}

void bad_args(void) {
    uint32_t draft[2] = { 1, 2 };
    uint32_t verified[3] = { 1, 2, 3 };
    uint32_t out[3];
    uint32_t n_out = 0;
    require(qrt_spec_emit(NULL, verified, 2, out, &n_out) == -1, "a null draft is refused");
    require(qrt_spec_emit(draft, NULL, 2, out, &n_out) == -1, "null verified rows are refused");
    require(qrt_spec_emit(draft, verified, 2, NULL, &n_out) == -1, "a null output is refused");
    require(qrt_spec_emit(draft, verified, 0, out, &n_out) == -1, "zero drafts are refused");
    require(qrt_spec_accept(NULL, NULL, 0) == 0, "an empty comparison accepts nothing");
}

void vocab_file(void) {
    const uint32_t n = 6;
    uint32_t ids[n] = { 3, 17, 42, 1000, 65000, 151000 };
    char path[512];
    snprintf(path, sizeof(path), "qrt_spec_vocab.tmp");
    FILE * f = fopen(path, "wb");
    require(f != NULL, "the temp vocabulary file is writable");
    if (f != NULL) {
        fwrite(ids, sizeof(ids[0]), n, f);
        fclose(f);
    }
    qrt_spec_vocab * v = NULL;
    char err[256] = {0};
    require(qrt_spec_vocab_open(path, n, &v, err, sizeof(err)) == 0 && v != NULL,
            "an ascending list loads");
    if (v != NULL) {
        require(qrt_spec_vocab_count(v) == n, "the count is what was asked for");
        const uint32_t * got = qrt_spec_vocab_ids(v);
        for (uint32_t i = 0; i < n; ++i) {
            require(got[i] == ids[i], "the ids come back unchanged");
        }
    }
    qrt_spec_vocab_free(v);
    v = NULL;
    err[0] = '\0';
    require(qrt_spec_vocab_open(path, n + 1u, &v, err, sizeof(err)) == -1 && v == NULL,
            "a short file is refused");
    err[0] = '\0';
    require(qrt_spec_vocab_open(NULL, n, &v, err, sizeof(err)) == -1, "no path is refused");
    err[0] = '\0';
    require(qrt_spec_vocab_open(path, 0, &v, err, sizeof(err)) == -1, "zero size is refused");
    // an unsorted list would break the binary search the draft step relies on
    f = fopen(path, "wb");
    if (f != NULL) {
        ids[2] = 1;
        fwrite(ids, sizeof(ids[0]), n, f);
        fclose(f);
    }
    err[0] = '\0';
    require(qrt_spec_vocab_open(path, n, &v, err, sizeof(err)) == -1 && v == NULL,
            "a list that is not ascending is refused");
    remove(path);
}

} // namespace

int main(void) {
    for (uint32_t n = 1; n <= 8; ++n) {
        perfect_draft(n);
        one_miss(n);
        all_wrong(n);
    }
    bad_args();
    vocab_file();
    if (fails != 0) {
        std::fprintf(stderr, "spec contract: %d failed\n", fails);
        return 1;
    }
    std::printf("spec contract: 0 failed\n");
    return 0;
}
