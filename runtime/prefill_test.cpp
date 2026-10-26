#include "prefill.h"
#include "weights.h"
#include "vk_transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#ifdef QRT_HAVE_VULKAN
#error prefill_test must link prefill.cpp without QRT_HAVE_VULKAN
#endif

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); \
} } while (0)

struct VkBuffer_T {
    std::vector<unsigned char> data;
    uint32_t uploads = 0;
    uint32_t borrowers = 0;
};

struct VkDeviceMemory_T {
    VkBuffer buf;
};

struct qrt_vk_staging {
    std::vector<unsigned char> map;
    std::vector<unsigned char> submitted;
    VkBuffer dst = nullptr;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    bool pending = false;
};

struct fake_device : qrt_vk_dev {
    std::vector<qrt_vk_staging *> staging;
    int live_buffers = 0;
    int alloc_calls = 0;
    int uploads = 0;
    int waits = 0;
    int fail_upload = -1;
    int fail_wait = -1;
    size_t staging_limit = 64;
};

struct qrt_weights {
    std::vector<std::vector<std::vector<unsigned char>>> payload;
    uint64_t max_bytes = 8;
    uint32_t layers = 4;
    bool mapped = true;
    int locate_calls = 0;
    int fail_locate = -1;
    int fail_read_layer = -1;
    int fail_read_expert = -1;
    int large_layer = -1;
    int large_expert = -1;
    uint64_t large_bytes = 0;
    bool fail_pair = false;
    std::atomic<int> read_calls{0};
    std::atomic<int> active_reads{0};
    std::mutex mu;
    std::condition_variable cv;
    int pair_entered = 0;
    std::set<std::thread::id> read_threads;
};

namespace {

int error(char * err, size_t size, const char * text) {
    std::snprintf(err, size, "%s", text);
    return -1;
}

const unsigned char known[12][8] = {
    {0x01, 0x93, 0x25, 0xd7, 0x49, 0xfb, 0x6d, 0x0f},
    {0xe2, 0x14, 0xa6, 0x38, 0xca, 0x5c, 0xee, 0x70},
    {0x83, 0x35, 0xc7, 0x59, 0xeb, 0x7d, 0x0f, 0xa1},
    {0x44, 0xf6, 0x68, 0x1a, 0x8c, 0x3e, 0xb0, 0x62},
    {0x65, 0x17, 0x89, 0x3b, 0xad, 0x5f, 0xd1, 0x83},
    {0xa6, 0x58, 0xca, 0x7c, 0xee, 0xa0, 0x12, 0xc4},
    {0xc7, 0x79, 0xeb, 0x9d, 0x0f, 0xc1, 0x33, 0xe5},
    {0x28, 0xda, 0x4c, 0xfe, 0x70, 0x22, 0x94, 0x46},
    {0x59, 0x0b, 0x7d, 0x2f, 0xa1, 0x53, 0xc5, 0x77},
    {0xfa, 0xac, 0x1e, 0xd0, 0x42, 0xf4, 0x66, 0x18},
    {0x9b, 0x4d, 0xbf, 0x71, 0xe3, 0x95, 0x07, 0xb9},
    {0x3c, 0xee, 0x60, 0x12, 0x84, 0x36, 0xa8, 0x5a},
};
const size_t lengths[12] = {4, 7, 8, 5, 8, 6, 4, 8, 7, 6, 8, 5};

struct ring_work {
    VkBuffer buf = nullptr;
    std::vector<unsigned char> image;
};

struct fixture {
    fake_device dev{};
    qrt_weights weights;
    qrt_prefill * p = nullptr;
    qrt_prefill_backend backend{};
    qrt_prefill_stats stats{};
    char err[256] = {};
    uint32_t ids[4][12] = {};
    uint32_t counts[4] = {};
    ring_work rings[2];
    std::vector<std::pair<uint32_t, uint32_t>> computed;
    std::vector<uint32_t> batches;
    std::vector<uint32_t> waited;
    int fail_compute = -1;
    int fail_batch = -1;
    int fail_backend_wait = -1;
    uint32_t fail_wait_mask = 0;
    bool partial_batch = true;

    fixture();
    ~fixture();
    void open(uint32_t slots = 8, uint32_t readers = 0, uint32_t pipeline = 1,
              uint32_t group = 0, uint32_t slot_bytes = 64);
    void reset();
    int run(bool batch);
    void drained();
    void verify_complete();
};

void verify_payload(fixture * f, uint32_t layer, uint32_t expert, VkBuffer buf,
                    uint64_t offset, uint64_t bytes) {
    CHECK(layer < f->weights.payload.size());
    CHECK(expert < f->weights.payload[layer].size());
    const auto & expected = f->weights.payload[layer][expert];
    CHECK(bytes == expected.size());
    CHECK(offset <= buf->data.size());
    CHECK(bytes <= buf->data.size() - offset);
    CHECK(std::memcmp(buf->data.data() + offset, expected.data(), expected.size()) == 0);
}

int compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
            uint64_t offset, uint64_t bytes, char * err, size_t size) {
    auto * f = static_cast<fixture *>(user);
    verify_payload(f, layer, expert, buf, offset, bytes);
    const int call = (int) f->computed.size();
    f->computed.emplace_back(layer, expert);
    if (call == f->fail_compute) {
        return error(err, size, "compute failure");
    }
    return 0;
}

int batch(void * user, uint32_t layer, uint32_t ring, VkBuffer buf,
          const uint32_t * experts, const uint64_t * offsets, const uint64_t * bytes,
          uint32_t count, char * err, size_t size) {
    auto * f = static_cast<fixture *>(user);
    CHECK(ring < 2);
    CHECK(f->rings[ring].buf == nullptr);
    CHECK(buf->uploads == 0);
    CHECK(f->weights.active_reads.load() == 0);
    CHECK(count == f->counts[layer]);
    const int call = (int) f->batches.size();
    f->batches.push_back(layer);
    for (uint32_t e = 0; e < count; ++e) {
        CHECK(experts[e] == f->ids[layer][e]);
        CHECK(offsets[e] % 64 == 0);
        if (e != 0) {
            CHECK(offsets[e] >= offsets[e - 1] + bytes[e - 1]);
        }
        verify_payload(f, layer, experts[e], buf, offsets[e], bytes[e]);
        f->computed.emplace_back(layer, experts[e]);
    }
    if (call != f->fail_batch || f->partial_batch) {
        // Only the device buffer remains borrowed; callback arrays are not retained.
        f->rings[ring].buf = buf;
        f->rings[ring].image = buf->data;
        ++buf->borrowers;
    }
    if (call == f->fail_batch) {
        return error(err, size, "batch failure");
    }
    return 0;
}

int wait(void * user, uint32_t ring, char * err, size_t size) {
    auto * f = static_cast<fixture *>(user);
    CHECK(ring < 2);
    const int call = (int) f->waited.size();
    f->waited.push_back(ring);
    ring_work & work = f->rings[ring];
    if (work.buf != nullptr) {
        CHECK(work.buf->data == work.image);
        CHECK(work.buf->borrowers == 1);
        --work.buf->borrowers;
        work.buf = nullptr;
    }
    if (call == f->fail_backend_wait || (f->fail_wait_mask & (1u << ring)) != 0) {
        std::snprintf(err, size, "wait ring %u failure", ring);
        return -1;
    }
    std::snprintf(err, size, "successful wait scratch");
    return 0;
}

fixture::fixture() {
    weights.payload.resize(4);
    for (uint32_t l = 0; l < 4; ++l) {
        for (uint32_t e = 0; e < 12; ++e) {
            std::vector<unsigned char> data(known[e], known[e] + lengths[e]);
            data[0] ^= (unsigned char) (l * 41);
            weights.payload[l].push_back(data);
            ids[l][e] = e;
        }
    }
    backend = {compute, batch, wait, this};
}

fixture::~fixture() {
    drained();
    qrt_prefill_close(&dev, p);
    CHECK(dev.live_buffers == 0);
    CHECK(dev.staging.empty());
}

void fixture::open(uint32_t slots, uint32_t readers, uint32_t pipeline,
                   uint32_t group, uint32_t slot_bytes) {
    qrt_prefill_cfg cfg{};
    cfg.slots = slots;
    cfg.readers = readers;
    cfg.pipeline = pipeline;
    cfg.group_bytes = group;
    cfg.slot_bytes = slot_bytes;
    CHECK(qrt_prefill_open(&dev, &weights, &cfg, &p, err, sizeof(err)) == 0);
    CHECK(p != nullptr);
    CHECK(qrt_prefill_layer_buffers(p) == (pipeline ? 2u : 1u));
}

void fixture::drained() {
    CHECK(weights.active_reads.load() == 0);
    for (const auto * stg : dev.staging) {
        CHECK(!stg->pending);
    }
    for (const auto & ring : rings) {
        CHECK(ring.buf == nullptr);
    }
}

void fixture::reset() {
    drained();
    dev.uploads = dev.waits = 0;
    dev.fail_upload = dev.fail_wait = -1;
    weights.locate_calls = 0;
    weights.fail_locate = -1;
    weights.fail_read_layer = weights.fail_read_expert = -1;
    weights.large_layer = weights.large_expert = -1;
    weights.fail_pair = false;
    weights.read_calls.store(0);
    weights.pair_entered = 0;
    weights.read_threads.clear();
    fail_compute = fail_batch = fail_backend_wait = -1;
    fail_wait_mask = 0;
    partial_batch = true;
    computed.clear();
    batches.clear();
    waited.clear();
    err[0] = '\0';
}

int fixture::run(bool use_batch) {
    const int rc = use_batch
        ? qrt_prefill_run_layers(p, &ids[0][0], 12, counts, &backend, &stats, err, sizeof(err))
        : qrt_prefill_run(p, &ids[0][0], 12, counts, &backend, &stats, err, sizeof(err));
    drained();
    return rc;
}

void fixture::verify_complete() {
    std::vector<std::pair<uint32_t, uint32_t>> expected;
    for (uint32_t l = 0; l < 4; ++l) {
        for (uint32_t e = 0; e < counts[l]; ++e) {
            expected.emplace_back(l, ids[l][e]);
        }
    }
    CHECK(computed == expected);
    CHECK(stats.blocks == expected.size());
    CHECK(stats.blocks_done == expected.size());
    CHECK(stats.blocks_failed == 0);
    CHECK((size_t) weights.read_calls.load() == expected.size());
    CHECK(dev.uploads == dev.waits);
}

} // namespace

uint32_t qrt_weights_layers(const qrt_weights * w) { return w->layers; }
uint64_t qrt_weights_max_expert_bytes(const qrt_weights * w) { return w->max_bytes; }
int qrt_weights_mapped(const qrt_weights * w) { return w->mapped ? 1 : 0; }

int qrt_weights_locate(const qrt_weights * weights, uint32_t layer, uint32_t expert,
                       qrt_expert_loc * out, char * err, size_t size) {
    auto * w = const_cast<qrt_weights *>(weights);
    const int call = w->locate_calls++;
    if (call == w->fail_locate || layer >= w->layers || expert >= 12) {
        return error(err, size, "locate failure");
    }
    CHECK(layer < w->payload.size());
    *out = {};
    out->bytes = w->payload[layer][expert].size();
    if ((int) layer == w->large_layer &&
        (w->large_expert == -1 || (int) expert == w->large_expert)) {
        out->bytes = w->large_bytes;
    }
    return 0;
}

int qrt_weights_read_expert(const qrt_weights * weights, uint32_t layer, uint32_t expert,
                            void * dst, size_t cap, char * err, size_t size) {
    auto * w = const_cast<qrt_weights *>(weights);
    ++w->active_reads;
    ++w->read_calls;
    {
        std::unique_lock<std::mutex> lk(w->mu);
        w->read_threads.insert(std::this_thread::get_id());
        if (w->fail_pair) {
            ++w->pair_entered;
            w->cv.notify_all();
            CHECK(w->cv.wait_for(lk, std::chrono::seconds(2), [&] { return w->pair_entered == 2; }));
        }
    }
    const auto & data = w->payload[layer][expert];
    CHECK(cap >= data.size());
    std::memcpy(dst, data.data(), data.size());
    const bool failed = w->fail_pair ||
        ((int) layer == w->fail_read_layer && (int) expert == w->fail_read_expert);
    if (failed) {
        std::memset(dst, 0xba, data.size());
        std::snprintf(err, size, "read %u,%u failure", layer, expert);
    }
    --w->active_reads;
    return failed ? -1 : 0;
}

int qrt_vk_alloc(qrt_vk_dev * dev, uint64_t bytes, VkBufferUsageFlags,
                 VkBuffer * buf, VkDeviceMemory * mem, char * err, size_t size) {
    auto * d = static_cast<fake_device *>(dev);
    ++d->alloc_calls;
    *buf = nullptr;
    *mem = nullptr;
    if (bytes == 0 || bytes > 1024 * 1024) {
        return error(err, size, "fake allocation capacity exceeded");
    }
    *buf = new VkBuffer_T;
    (*buf)->data.resize((size_t) bytes, 0xcc);
    *mem = new VkDeviceMemory_T{*buf};
    ++d->live_buffers;
    return 0;
}

void qrt_vk_free(qrt_vk_dev * dev, VkBuffer buf, VkDeviceMemory mem) {
    CHECK(buf != nullptr && mem != nullptr && mem->buf == buf);
    CHECK(buf->uploads == 0);
    CHECK(buf->borrowers == 0);
    --static_cast<fake_device *>(dev)->live_buffers;
    delete mem;
    delete buf;
}

int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t size) {
    auto * d = static_cast<fake_device *>(dev);
    *out = nullptr;
    if (d->staging.size() >= d->staging_limit) {
        return error(err, size, "staging allocation failure");
    }
    CHECK(bytes > 0 && bytes <= 1024 * 1024);
    *out = new qrt_vk_staging;
    (*out)->map.resize((size_t) bytes, 0xdd);
    d->staging.push_back(*out);
    return 0;
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    CHECK(stg != nullptr);
    CHECK(!stg->pending);
    auto & all = static_cast<fake_device *>(dev)->staging;
    const auto it = std::find(all.begin(), all.end(), stg);
    CHECK(it != all.end());
    all.erase(it);
    delete stg;
}

void * qrt_vk_staging_map(qrt_vk_staging * stg) {
    CHECK(!stg->pending);
    return stg->map.data();
}

int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t offset, uint64_t bytes, char * err, size_t size) {
    auto * d = static_cast<fake_device *>(dev);
    CHECK(!stg->pending);
    CHECK(dst->borrowers == 0);
    CHECK(bytes != 0 && bytes <= stg->map.size());
    CHECK(offset <= dst->data.size() && bytes <= dst->data.size() - offset);
    const int call = d->uploads++;
    if (call == d->fail_upload) {
        return error(err, size, "upload failure");
    }
    stg->dst = dst;
    stg->offset = offset;
    stg->bytes = bytes;
    stg->submitted = stg->map;
    stg->pending = true;
    ++dst->uploads;
    return 0;
}

int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t size) {
    if (!stg->pending) {
        return 0;
    }
    auto * d = static_cast<fake_device *>(dev);
    const int call = d->waits++;
    CHECK(stg->submitted == stg->map);
    CHECK(stg->dst->uploads > 0);
    --stg->dst->uploads;
    stg->pending = false;
    if (call == d->fail_wait) {
        return error(err, size, "staging wait failure");
    }
    std::memcpy(stg->dst->data.data() + stg->offset, stg->map.data(), (size_t) stg->bytes);
    std::snprintf(err, size, "successful staging wait scratch");
    return 0;
}

namespace {

void test_open_limits() {
    for (int mode = 0; mode < 5; ++mode) {
        fixture f;
        qrt_prefill_cfg cfg{};
        cfg.slots = 8;
        if (mode == 0) cfg.slots = 0;
        if (mode == 1) cfg.slots = 65;
        if (mode == 2) cfg.readers = 17;
        if (mode == 3) f.weights.max_bytes = UINT64_MAX;
        if (mode == 4) {
            cfg.slot_bytes = 1;
            cfg.group_bytes = 128;
            f.weights.max_bytes = UINT64_MAX;
        }
        CHECK(qrt_prefill_open(&f.dev, &f.weights, &cfg, &f.p, f.err, sizeof(f.err)) != 0);
        CHECK(f.p == nullptr);
        CHECK(f.dev.alloc_calls == 0);
    }
    {
        fixture f;
        f.open(64, 16);
    }
    {
        fixture f;
        f.dev.staging_limit = 3;
        f.open(8);
        CHECK(f.dev.staging.size() == 2);
        f.counts[0] = 5;
        CHECK(f.run(false) == 0);
        f.verify_complete();
        CHECK(f.stats.slot_reductions == 2);
    }
    {
        fixture f;
        f.dev.staging_limit = 0;
        qrt_prefill_cfg cfg{};
        cfg.slots = 8;
        CHECK(qrt_prefill_open(&f.dev, &f.weights, &cfg, &f.p, f.err, sizeof(f.err)) != 0);
        CHECK(f.p == nullptr);
        CHECK(f.dev.live_buffers == 0);
    }
}

void test_perexpert_boundaries() {
    const uint32_t order[12] = {7, 2, 11, 0, 9, 4, 1, 10, 6, 3, 8, 5};
    for (uint32_t slots : {1u, 8u}) {
        fixture f;
        f.open(slots);
        for (uint32_t count : {1u, slots > 1 ? slots - 1 : 1, slots, slots + 1}) {
            for (uint32_t pass = 0; pass < 2; ++pass) {
                f.reset();
                std::fill(f.counts, f.counts + 4, 0u);
                f.counts[pass + 1] = count;
                std::copy(order, order + 12, f.ids[pass + 1]);
                CHECK(f.run(false) == 0);
                f.verify_complete();
                CHECK(f.stats.groups == count);
            }
        }
    }
}

void test_perexpert_failures() {
    for (int mode = 0; mode < 7; ++mode) {
        fixture f;
        f.open(4);
        f.counts[0] = 5;
        if (mode == 0) f.weights.fail_locate = 2;
        if (mode == 1 || mode == 6) {
            f.weights.fail_read_layer = 0;
            f.weights.fail_read_expert = 2;
        }
        if (mode == 2) f.dev.fail_upload = 2;
        if (mode == 3) f.fail_compute = 0;
        if (mode == 4 || mode == 6) f.dev.fail_wait = 0;
        if (mode == 5) {
            f.weights.large_layer = 0;
            f.weights.large_expert = 2;
            f.weights.large_bytes = 65;
        }
        CHECK(f.run(false) != 0);
        CHECK(f.stats.blocks_failed == (mode == 6 ? 2u : 1u));
        CHECK(f.stats.blocks_done == 0);
        CHECK(f.dev.waits == ((mode == 3 || mode == 4) ? 4 : 2));
        CHECK(f.computed.size() == (mode == 3 ? 1u : 0u));
        if (mode == 0) CHECK(std::strcmp(f.err, "locate failure") == 0);
        if (mode == 1 || mode == 6) CHECK(std::strcmp(f.err, "read 0,2 failure") == 0);
        if (mode == 2) CHECK(std::strcmp(f.err, "upload failure") == 0);
        if (mode == 3) CHECK(std::strcmp(f.err, "compute failure") == 0);
        if (mode == 4) CHECK(std::strcmp(f.err, "staging wait failure") == 0);
        f.reset();
        CHECK(f.run(false) == 0);
        f.verify_complete();
    }
}

void test_capabilities_and_routes() {
    for (bool use_batch : {false, true}) {
        fixture f;
        f.open();
        f.counts[0] = 1;
        const int allocs = f.dev.alloc_calls;
        CHECK((use_batch
            ? qrt_prefill_run_layers(f.p, &f.ids[0][0], 12, f.counts, nullptr, &f.stats, f.err, sizeof(f.err))
            : qrt_prefill_run(f.p, &f.ids[0][0], 12, f.counts, nullptr, &f.stats, f.err, sizeof(f.err))) != 0);
        if (use_batch) f.backend.wait = nullptr;
        else f.backend.compute = nullptr;
        CHECK(f.run(use_batch) != 0);
        f.backend = {compute, batch, wait, &f};
        if (use_batch) {
            f.backend.batch = nullptr;
            CHECK(f.run(true) != 0);
            f.backend.batch = batch;
        }
        f.counts[0] = 13;
        CHECK(f.run(use_batch) != 0);
        f.counts[0] = 0;
        CHECK(f.run(use_batch) != 0);
        CHECK(f.dev.alloc_calls == allocs);
        CHECK(f.dev.uploads == 0 && f.weights.read_calls.load() == 0);
        CHECK(f.weights.locate_calls == 0);
        CHECK(f.computed.empty() && f.batches.empty());
        f.counts[0] = 1;
        f.ids[0][0] = UINT32_MAX;
        CHECK(f.run(use_batch) != 0);
        CHECK(f.stats.blocks_failed == 1);
        CHECK(std::strcmp(f.err, "locate failure") == 0);
        CHECK(f.dev.alloc_calls == allocs);
    }
    for (uint64_t bytes : {UINT64_MAX - 31, UINT64_MAX / 2 + 1}) {
        fixture f;
        f.open();
        f.counts[0] = 2;
        f.weights.large_layer = 0;
        f.weights.large_bytes = bytes;
        const int allocs = f.dev.alloc_calls;
        CHECK(f.run(true) != 0);
        CHECK(f.dev.alloc_calls == allocs);
        CHECK(f.dev.uploads == 0);
    }
}

void test_batch_success() {
    for (uint32_t pipeline : {0u, 1u}) {
        for (uint32_t group : {0u, 256u}) {
            for (uint32_t readers : {0u, 1u, 2u}) {
                fixture f;
                f.open(8, readers, pipeline, group);
                f.counts[0] = 3;
                f.counts[2] = 4;
                f.counts[3] = 2;
                for (int pass = 0; pass < 3; ++pass) {
                    f.reset();
                    f.weights.mapped = pass != 1;
                    std::reverse(f.ids[2], f.ids[2] + 4);
                    CHECK(f.run(true) == 0);
                    f.verify_complete();
                    CHECK(f.batches == std::vector<uint32_t>({0, 2, 3}));
                    CHECK(f.waited == (pipeline ? std::vector<uint32_t>({0, 0, 1})
                                                : std::vector<uint32_t>({0, 0, 0})));
                    CHECK(f.stats.layers == 3);
                    CHECK(f.stats.groups == (group ? 3u : 9u));
                    if (readers == 2 && group != 0) {
                        CHECK(f.weights.read_threads.size() == 2);
                    }
                }
            }
        }
    }
}

void test_batch_upload_failures() {
    for (int mode = 0; mode < 4; ++mode) {
        fixture f;
        f.open(4);
        f.counts[0] = f.counts[1] = f.counts[2] = 2;
        if (mode == 0) f.weights.fail_locate = 8;
        if (mode == 1) {
            f.weights.fail_read_layer = 1;
            f.weights.fail_read_expert = 1;
        }
        if (mode == 2) f.dev.fail_upload = 3;
        if (mode == 3) f.dev.fail_wait = 2;
        CHECK(f.run(true) != 0);
        CHECK(f.stats.blocks_failed == 1);
        CHECK(f.stats.blocks_done == 2);
        CHECK(f.batches == std::vector<uint32_t>({0}));
        CHECK(f.waited == std::vector<uint32_t>({0}));
        CHECK(f.dev.waits == (mode == 0 ? 2 : mode == 3 ? 4 : 3));
        if (mode == 0) CHECK(std::strcmp(f.err, "locate failure") == 0);
        if (mode == 1) CHECK(std::strcmp(f.err, "read 1,1 failure") == 0);
        if (mode == 2) CHECK(std::strcmp(f.err, "upload failure") == 0);
        if (mode == 3) CHECK(std::strcmp(f.err, "staging wait failure") == 0);
        f.reset();
        CHECK(f.run(true) == 0);
        f.verify_complete();
    }
}

void test_batch_drains() {
    for (int mode = 0; mode < 5; ++mode) {
        fixture f;
        f.open();
        f.counts[0] = f.counts[1] = 2;
        if (mode < 3) {
            f.fail_batch = 1;
            f.partial_batch = mode != 1;
            if (mode == 2) f.fail_wait_mask = 3;
        }
        if (mode == 3) f.fail_wait_mask = 3;
        if (mode == 4) {
            f.counts[2] = 2;
            f.fail_backend_wait = 0;
        }
        CHECK(f.run(true) != 0);
        CHECK(f.batches == std::vector<uint32_t>({0, 1}));
        CHECK(f.waited == (mode == 4 ? std::vector<uint32_t>({0, 1})
                                    : std::vector<uint32_t>({1, 0})));
        CHECK(f.stats.blocks_failed == ((mode == 2 || mode == 3) ? 4u : 2u));
        CHECK(f.stats.blocks_done == ((mode == 2 || mode == 3) ? 0u : 2u));
        if (mode < 3) CHECK(std::strcmp(f.err, "batch failure") == 0);
        if (mode == 3) CHECK(std::strcmp(f.err, "wait ring 1 failure") == 0);
        if (mode == 4) CHECK(std::strcmp(f.err, "wait ring 0 failure") == 0);
        f.reset();
        f.counts[3] = 4;
        CHECK(f.run(true) == 0);
        f.verify_complete();
    }
}

void test_reader_failures() {
    for (uint32_t readers : {1u, 2u}) {
        fixture f;
        f.open(8, readers, 1, 256);
        f.counts[0] = 2;
        f.weights.fail_read_layer = 0;
        f.weights.fail_read_expert = 1;
        CHECK(f.run(true) != 0);
        CHECK(f.dev.uploads == 0);
        CHECK(f.batches.empty());
        CHECK(f.stats.blocks_failed == 1);
        CHECK(std::strcmp(f.err, "read 0,1 failure") == 0);
        if (readers == 2) {
            CHECK(std::any_of(f.weights.read_threads.begin(), f.weights.read_threads.end(),
                             [](std::thread::id id) { return id != std::this_thread::get_id(); }));
        }
        f.reset();
        CHECK(f.run(true) == 0);
        f.verify_complete();
    }
    {
        fixture f;
        f.open(8, 2, 1, 256);
        f.counts[0] = 2;
        f.weights.fail_pair = true;
        CHECK(f.run(true) != 0);
        CHECK(f.weights.pair_entered == 2);
        CHECK(f.stats.blocks_failed == 2);
        CHECK(f.dev.uploads == 0);
        CHECK(std::strcmp(f.err, "read 0,0 failure") == 0 ||
              std::strcmp(f.err, "read 0,1 failure") == 0);
        f.reset();
        CHECK(f.run(true) == 0);
        f.verify_complete();
    }
}

void test_tail_failures_and_mixed_reuse() {
    fixture f;
    f.open(8, 2, 1, 256);
    f.counts[0] = 2;
    for (int mode = 0; mode < 2; ++mode) {
        f.reset();
        if (mode == 0) f.fail_compute = 0;
        else f.dev.fail_wait = 0;
        CHECK(f.run(false) != 0);
        CHECK(f.dev.waits == 2);
        CHECK(f.stats.blocks_failed == 1);
        CHECK(f.stats.blocks_done == 0);
        CHECK(f.computed.size() == (mode == 0 ? 1u : 0u));
        f.reset();
        CHECK(f.run(true) == 0);
        f.verify_complete();
        f.reset();
        CHECK(f.run(false) == 0);
        f.verify_complete();
    }
}

} // namespace

int main() {
    test_open_limits();
    test_perexpert_boundaries();
    test_perexpert_failures();
    test_capabilities_and_routes();
    test_batch_success();
    test_batch_upload_failures();
    test_batch_drains();
    test_reader_failures();
    test_tail_failures_and_mixed_reuse();
    std::puts("prefill_test: PASS");
    return 0;
}
