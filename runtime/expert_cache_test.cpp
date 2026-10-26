#include "expert_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); \
} } while (0)

namespace {

qrt_cache_plan manual_plan(int32_t policy, const uint32_t * slots, uint32_t layers,
                           const uint64_t * slot_bytes, uint32_t protected_percent) {
    qrt_cache_plan p{};
    p.layers = layers;
    p.experts = 8;
    p.policy = policy;
    p.protected_percent = protected_percent;
    for (uint32_t l = 0; l < layers; ++l) {
        p.slot_bytes[l] = slot_bytes[l];
        p.slots_per_layer[l] = slots[l];
    }
    uint64_t off = 0;
    uint64_t total = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        p.layer_offset[l] = off;
        off += (uint64_t) slots[l] * slot_bytes[l];
        total += slots[l];
    }
    p.bytes = off;
    p.budget_bytes = off;
    p.slots = total;
    return p;
}

struct fake_io {
    std::vector<unsigned char> slab;
    int uploads = 0;
    int waits = 0;
    int fail_upload = -1;
    int fail_wait = -1;
    std::vector<uint32_t> uploaded_experts;
};

int upload(void * user, uint32_t layer, uint32_t expert, uint64_t dst, uint64_t bytes,
           char * err, size_t err_size) {
    fake_io * io = (fake_io *) user;
    if (io->uploads++ == io->fail_upload) {
        std::snprintf(err, err_size, "injected upload failure");
        return -1;
    }
    CHECK(dst + bytes <= io->slab.size());
    for (uint64_t i = 0; i < bytes; ++i) {
        io->slab[(size_t) (dst + i)] = (unsigned char) (expert + i + layer);
    }
    io->uploaded_experts.push_back(expert);
    return 0;
}

int wait_io(void * user, uint32_t layer, uint32_t expert, uint64_t dst, uint64_t bytes,
            char * err, size_t err_size) {
    (void) layer;
    (void) expert;
    (void) dst;
    (void) bytes;
    fake_io * io = (fake_io *) user;
    if (io->waits++ == io->fail_wait) {
        std::snprintf(err, err_size, "injected wait failure");
        return -1;
    }
    return 0;
}

void write_trace(const char * path, uint32_t n_tok, const std::vector<std::vector<int32_t>> & layers,
                 uint32_t top_k) {
    FILE * f = std::fopen(path, "wb");
    CHECK(f != nullptr);
    const int32_t header[3] = { (int32_t) n_tok, (int32_t) layers.size(), (int32_t) top_k };
    CHECK(std::fwrite(header, sizeof(header), 1, f) == 1);
    for (const auto & ids : layers) {
        const int32_t n = (int32_t) ids.size();
        CHECK(std::fwrite(&n, sizeof(n), 1, f) == 1);
        if (!ids.empty()) {
            CHECK(std::fwrite(ids.data(), sizeof(int32_t), ids.size(), f) == ids.size());
        }
    }
    CHECK(std::fclose(f) == 0);
}

void test_plan() {
    const uint64_t slots[3] = { 100, 200, 500 };
    const uint64_t weights[3] = { 1, 1, 1 };
    qrt_cache_plan p{};
    char err[256] = {};
    CHECK(qrt_cache_plan_build(3, 4, slots, weights, 1200, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) == 0);
    CHECK(p.bytes <= 1200 && p.bytes == 1200);
    CHECK(p.slots == 6 && p.slots_per_layer[0] == 3 && p.slots_per_layer[1] == 2 &&
          p.slots_per_layer[2] == 1);
    CHECK(p.layer_offset[0] == 0 && p.layer_offset[1] == 300 && p.layer_offset[2] == 700);
    CHECK(p.degraded_layers == 0);
    CHECK(qrt_cache_plan_build(3, 4, slots, weights, 0, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) == 0);
    CHECK(p.bytes == 0 && p.slots == 0 && p.degraded_layers == 3);
    CHECK(qrt_cache_plan_build(3, 4, slots, weights, 250, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) == 0);
    CHECK(p.bytes == 100 && p.slots == 1 && p.degraded_layers == 2);
    CHECK(p.layer_offset[0] == 0 && p.layer_offset[1] == 100 && p.layer_offset[2] == 100);
    const uint64_t one[2] = { 1, 1 };
    CHECK(qrt_cache_plan_build(2, 4, one, nullptr, 1000000, QRT_CACHE_LRU, 80, &p, err,
                               sizeof(err)) == 0);
    CHECK(p.slots_per_layer[0] == 4 && p.slots_per_layer[1] == 4 && p.bytes == 8);
    CHECK(qrt_cache_plan_build(0, 4, slots, weights, 1000, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) != 0);
    CHECK(qrt_cache_plan_build(3, 0, slots, weights, 1000, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) != 0);
    CHECK(qrt_cache_plan_build(3, 4, slots, weights, 1000, 9, 80, &p, err, sizeof(err)) != 0);
    CHECK(qrt_cache_plan_build(3, 4, slots, weights, 1000, QRT_CACHE_SLRU, 101, &p, err,
                               sizeof(err)) != 0);
    const uint64_t zero[3] = { 0, 1, 1 };
    CHECK(qrt_cache_plan_build(3, 4, zero, weights, 1000, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) != 0);
    const uint64_t huge[2] = { UINT64_MAX, 1 };
    CHECK(qrt_cache_plan_build(2, 4, huge, nullptr, 1000, QRT_CACHE_SLRU, 80, &p, err,
                               sizeof(err)) != 0);
}

void test_hits_and_evictions() {
    const uint32_t slots[2] = { 3, 2 };
    const uint64_t slot_bytes[2] = { 4, 8 };
    for (int32_t policy : { QRT_CACHE_SLRU, QRT_CACHE_LRU }) {
        qrt_cache_plan p = manual_plan(policy, slots, 2, slot_bytes, 66);
        qrt_cache * c = nullptr;
        char err[256] = {};
        CHECK(qrt_cache_open(&p, nullptr, &c, err, sizeof(err)) == 0);
        qrt_cache_lookup look[3];
        CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0, 1, 2 }, 3, look, err, sizeof(err)) == 0);
        CHECK(look[0].slot == 0 && look[1].slot == 1 && look[2].slot == 2);
        CHECK(!look[0].hit && look[0].ready && look[0].slot >= 0);
        CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0, 1, 2 }, 3, look, err, sizeof(err)) == 0);
        CHECK(look[0].hit && look[1].hit && look[2].hit);
        CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 3, 4, 5 }, 3, look, err, sizeof(err)) == 0);
        CHECK(!look[0].hit && !look[1].hit && !look[2].hit);
        qrt_cache_stats st;
        qrt_cache_get_stats(c, &st);
        CHECK(st.lookups == 9 && st.hits == 3 && st.misses == 6 && st.evictions == 3);
        CHECK(st.pending == 0 && st.h2d_uploads == 0 && st.miss_bytes == 24);
        CHECK(qrt_cache_slot_offset(c, 0, 2) == 8 && qrt_cache_slot_offset(c, 1, 1) == 20);
        qrt_cache_close(c);
    }

    qrt_cache_plan p = manual_plan(QRT_CACHE_SLRU, slots, 2, slot_bytes, 66);
    qrt_cache * c = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_open(&p, nullptr, &c, err, sizeof(err)) == 0);
    qrt_cache_lookup look[1];
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 1 }, 1, look, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 2 }, 1, look, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 3 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].slot == 1);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit && look[0].slot == 0);
    qrt_cache_stats st;
    qrt_cache_get_stats(c, &st);
    CHECK(st.promotions >= 1 && st.evictions == 1);
    qrt_cache_close(c);
}

void test_static_and_no_slot() {
    const uint32_t slots[2] = { 1, 1 };
    const uint64_t slot_bytes[2] = { 4, 8 };
    qrt_cache_plan p = manual_plan(QRT_CACHE_STATIC, slots, 2, slot_bytes, 80);
    qrt_cache * c = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_open(&p, nullptr, &c, err, sizeof(err)) == 0);
    qrt_cache_lookup look[2];
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 1 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].failed && look[0].slot == -1);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit);
    qrt_cache_stats st;
    qrt_cache_get_stats(c, &st);
    CHECK(st.no_slot_misses == 1 && st.evictions == 0);
    qrt_cache_close(c);

    qrt_cache_plan zero = manual_plan(QRT_CACHE_SLRU, (const uint32_t[]) { 0, 0 }, 2,
                                      slot_bytes, 80);
    CHECK(qrt_cache_open(&zero, nullptr, &c, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0, 1 }, 2, look, err, sizeof(err)) == 0);
    CHECK(look[0].failed && look[1].failed);
    qrt_cache_get_stats(c, &st);
    CHECK(st.no_slot_misses == 2 && st.pending == 0);
    qrt_cache_close(c);
}

void test_io_scheduling() {
    const uint32_t slots[2] = { 1, 1 };
    const uint64_t slot_bytes[2] = { 4, 8 };
    qrt_cache_plan p = manual_plan(QRT_CACHE_SLRU, slots, 2, slot_bytes, 80);
    fake_io io;
    io.slab.assign((size_t) p.bytes, 0);
    qrt_cache_io hooks = { upload, wait_io, &io };
    qrt_cache * c = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_open(&p, &hooks, &c, err, sizeof(err)) == 0);
    qrt_cache_lookup look[2];
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].pending && !look[0].ready && io.uploads == 1);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 1 }, 1, look, err, sizeof(err)) != 0);
    CHECK(qrt_cache_acquire(c, 1, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].pending && io.uploads == 2);
    CHECK(qrt_cache_wait_expert(c, 0, 0, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit && look[0].ready && !look[0].pending);
    CHECK(qrt_cache_wait_all(c, err, sizeof(err)) == 0);
    qrt_cache_stats st;
    qrt_cache_get_stats(c, &st);
    CHECK(st.pending == 0 && st.h2d_uploads == 2 && st.h2d_bytes == 12);
    CHECK(st.lookups == 3 && st.hits == 1 && st.misses == 2);
    CHECK(io.slab[0] == 0 && io.slab[3] == 3);
    qrt_cache_close(c);

    io = fake_io{};
    io.slab.assign((size_t) p.bytes, 0);
    io.fail_upload = 0;
    CHECK(qrt_cache_open(&p, &hooks, &c, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].failed && look[0].slot == -1 && !look[0].pending);
    io.fail_upload = -1;
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(look[0].slot == 0);
    CHECK(qrt_cache_wait_expert(c, 0, 0, err, sizeof(err)) == 0);
    qrt_cache_get_stats(c, &st);
    CHECK(st.upload_failures == 1 && st.h2d_uploads == 1);
    qrt_cache_close(c);

    io = fake_io{};
    io.slab.assign((size_t) p.bytes, 0);
    io.fail_wait = 0;
    CHECK(qrt_cache_open(&p, &hooks, &c, err, sizeof(err)) == 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(qrt_cache_wait_expert(c, 0, 0, err, sizeof(err)) != 0);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    CHECK(!look[0].hit && look[0].slot >= 0);
    CHECK(qrt_cache_wait_expert(c, 0, 0, err, sizeof(err)) == 0);
    qrt_cache_get_stats(c, &st);
    CHECK(st.wait_failures == 1 && st.pending == 0);
    qrt_cache_close(c);
}

void test_lookup_edges() {
    const uint32_t slots[2] = { 1, 1 };
    const uint64_t slot_bytes[2] = { 4, 8 };
    qrt_cache_plan p = manual_plan(QRT_CACHE_SLRU, slots, 2, slot_bytes, 80);
    qrt_cache * c = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_open(&p, nullptr, &c, err, sizeof(err)) == 0);
    qrt_cache_lookup look[4];
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0, 0 }, 2, look, err, sizeof(err)) == 0);
    CHECK(look[0].slot == look[1].slot && look[0].expert == look[1].expert);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0, 1 }, 2, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit && look[1].failed && look[1].slot == -1);
    qrt_cache_stats st;
    qrt_cache_get_stats(c, &st);
    CHECK(st.lookups == 3 && st.hits == 1 && st.misses == 2 && st.no_slot_misses == 1);
    qrt_cache_reset_stats(c);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) == 0);
    qrt_cache_get_stats(c, &st);
    CHECK(st.lookups == 1 && st.hits == 1);
    CHECK(qrt_cache_acquire(c, 0, (const uint32_t[]) { 8 }, 1, look, err, sizeof(err)) != 0);
    CHECK(qrt_cache_acquire(c, 2, (const uint32_t[]) { 0 }, 1, look, err, sizeof(err)) != 0);
    qrt_cache_close(c);
}

void test_split_legs() {
    const uint32_t slots[1] = { 3 };
    const uint64_t slot_bytes[1] = { 4 };
    qrt_cache_plan p = manual_plan(QRT_CACHE_SLRU, slots, 1, slot_bytes, 80);
    qrt_cache * c = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_open(&p, nullptr, &c, err, sizeof(err)) == 0);
    qrt_cache_lookup look[3];
    const uint32_t ids[3] = { 0, 1, 2 };
    const uint8_t mask[3] = { 1, 0, 1 };
    CHECK(qrt_cache_acquire_mask(c, 0, ids, 3, mask, look, err, sizeof(err)) == 0);
    CHECK(look[0].slot >= 0 && !look[0].failed);
    CHECK(look[1].slot == -1 && look[1].failed && !look[1].ready);
    CHECK(look[2].slot >= 0 && !look[2].failed);
    CHECK(qrt_cache_resident(c, 0, 0) == 1);
    CHECK(qrt_cache_resident(c, 0, 1) == 0);
    CHECK(qrt_cache_resident(c, 0, 8) == 0);
    qrt_cache_stats st;
    qrt_cache_get_stats(c, &st);
    CHECK(st.misses == 3 && st.cpu_leg_misses == 1 && st.no_slot_misses == 0);
    CHECK(st.h2d_uploads == 0 && st.evictions == 0);
    // A later acquire with no mask puts the CPU-leg expert into the cache like any miss.
    CHECK(qrt_cache_acquire(c, 0, ids, 3, look, err, sizeof(err)) == 0);
    CHECK(look[0].hit && look[1].slot >= 0 && look[2].hit);
    CHECK(qrt_cache_resident(c, 0, 1) == 1);
    qrt_cache_note_upload(c, 0, 4);
    qrt_cache_get_stats(c, &st);
    CHECK(st.h2d_uploads == 1 && st.h2d_bytes == 4);
    CHECK(qrt_cache_layer_base(c, 0) == 0);
    qrt_cache_close(c);

    // A split that hands every miss to the CPU needs no slot at all: the acquire must
    // not touch the policy, and the layer can run with zero slots.
    qrt_cache_plan zero = manual_plan(QRT_CACHE_SLRU, (const uint32_t[]) { 0 }, 1,
                                      slot_bytes, 80);
    CHECK(qrt_cache_open(&zero, nullptr, &c, err, sizeof(err)) == 0);
    const uint8_t none[3] = { 0, 0, 0 };
    CHECK(qrt_cache_acquire_mask(c, 0, ids, 3, none, look, err, sizeof(err)) == 0);
    CHECK(look[0].failed && look[1].failed && look[2].failed);
    qrt_cache_get_stats(c, &st);
    CHECK(st.cpu_leg_misses == 3 && st.no_slot_misses == 0 && st.evictions == 0);
    qrt_cache_close(c);
}

void test_trace_and_simulation() {
    const char * path = "expert-cache-test-trace.bin";
    std::vector<std::vector<int32_t>> routes(3);
    for (int r = 0; r < 6; ++r) {
        routes[0].push_back(r % 8);
        routes[0].push_back((r + 1) % 8);
        routes[1].push_back((r + 2) % 8);
        routes[1].push_back((r + 3) % 8);
    }
    routes[2].push_back(0);
    routes[2].push_back(1);
    routes[2].push_back(2);
    routes[2].push_back(3);
    write_trace(path, 6, routes, 2);
    qrt_cache_trace * t = nullptr;
    char err[256] = {};
    CHECK(qrt_cache_trace_open(path, 8, &t, err, sizeof(err)) == 0);
    CHECK(qrt_cache_trace_tokens(t) == 6 && qrt_cache_trace_layers(t) == 3);
    CHECK(qrt_cache_trace_top_k(t) == 2 && qrt_cache_trace_rows(t, 2) == 2);
    CHECK(qrt_cache_trace_route(t, 0, 0)[0] == 0 && qrt_cache_trace_route(t, 0, 0)[1] == 1);
    uint64_t weights[4] = {};
    uint32_t observed = 0;
    CHECK(qrt_cache_trace_weights(t, 4, 3, weights, &observed, err, sizeof(err)) == 0);
    CHECK(weights[0] == 8 && weights[1] == 8 && weights[2] == 0 && observed == 2);

    const uint32_t slots[3] = { 1, 1, 1 };
    const uint64_t slot_bytes[3] = { 4, 4, 4 };
    for (int32_t policy : { QRT_CACHE_SLRU, QRT_CACHE_LRU, QRT_CACHE_STATIC }) {
        qrt_cache_plan p = manual_plan(policy, slots, 3, slot_bytes, 80);
        qrt_cache_stats st{};
        uint32_t layers = 0;
        uint32_t rows = 0;
        CHECK(qrt_cache_simulate(&p, t, 4, &st, &layers, &rows, err, sizeof(err)) == 0);
        CHECK(layers == 2 && rows == 8);
        CHECK(st.lookups == 16 && st.hits + st.misses == st.lookups);
        CHECK(st.pending == 0 && st.h2d_uploads == 0);
        if (policy == QRT_CACHE_STATIC) {
            CHECK(st.no_slot_misses > 0);
        }
    }
    qrt_cache_trace_close(t);

    CHECK(qrt_cache_trace_open(path, 4, &t, err, sizeof(err)) != 0);
    routes[0][0] = 9;
    write_trace(path, 6, routes, 2);
    CHECK(qrt_cache_trace_open(path, 8, &t, err, sizeof(err)) != 0);
    std::remove(path);
    CHECK(qrt_cache_trace_open(path, 8, &t, err, sizeof(err)) != 0);
}

} // namespace

int main() {
    test_plan();
    test_hits_and_evictions();
    test_static_and_no_slot();
    test_io_scheduling();
    test_lookup_edges();
    test_split_legs();
    test_trace_and_simulation();
    std::puts("expert_cache_test: PASS");
    return 0;
}
