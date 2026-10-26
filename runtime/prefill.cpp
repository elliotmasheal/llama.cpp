#include "prefill.h"
#include "vk_transfer.h"

#include "qrt_util.h"
#include "weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

namespace {

// Copies one group of experts into the staging mapping across the workers. The job is a
// plain function pointer so no allocation or std::function dispatch happens per group.
struct par_job {
    const qrt_weights * w;
    uint32_t layer;
    const uint32_t * experts;
    const uint64_t * offsets;
    const uint64_t * sizes;
    uint32_t first;
    uint32_t last;
    uint64_t span_off;
    unsigned char * map;
    bool mapped;
    std::atomic<int> failed;
    char err[256];
};

struct thread_pool {
    std::vector<std::thread> workers;
    std::mutex mu;
    std::mutex read_mu;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    par_job * job = nullptr;
    uint64_t epoch = 0;
    int done = 0;
    int readers = 1;
    bool stop = false;
};

void pool_read(thread_pool * pool, par_job * job, uint32_t id) {
    for (uint64_t i = (uint64_t) job->first + id; i < job->last; i += (uint32_t) pool->readers) {
        if (job->failed.load(std::memory_order_relaxed) != 0) {
            break;
        }
        char err[256] = {};
        // The unmapped store shares a seek/fread cursor between readers.
        std::unique_lock<std::mutex> lk(pool->read_mu, std::defer_lock);
        if (!job->mapped) {
            lk.lock();
            if (job->failed.load(std::memory_order_relaxed) != 0) {
                break;
            }
        }
        if (qrt_weights_read_expert(job->w, job->layer, job->experts[i],
                                    job->map + (job->offsets[i] - job->span_off),
                                    (size_t) job->sizes[i], err, sizeof(err)) != 0) {
            if (job->failed.fetch_add(1) == 0) {
                snprintf(job->err, sizeof(job->err), "%s", err[0] ? err : "group read failed");
            }
        }
    }
}

void pool_worker(thread_pool * pool, int id) {
    uint64_t seen = 0;
    for (;;) {
        par_job * job = nullptr;
        {
            std::unique_lock<std::mutex> lk(pool->mu);
            pool->cv_start.wait(lk, [&] { return pool->stop || pool->epoch != seen; });
            if (pool->stop) {
                return;
            }
            seen = pool->epoch;
            job = pool->job;
        }
        pool_read(pool, job, (uint32_t) id);
        {
            std::lock_guard<std::mutex> lk(pool->mu);
            ++pool->done;
        }
        pool->cv_done.notify_one();
    }
}

} // namespace

struct qrt_prefill {
    qrt_vk_dev *    dev;
    qrt_weights *   w;
    qrt_prefill_cfg cfg;
    uint64_t        slot_bytes;
    uint32_t        slots;
    thread_pool     pool;
    par_job         job;
    VkBuffer        buf;      // slots x slot_bytes of scratch for the compute leg
    VkDeviceMemory  mem;
    VkBuffer        layer_buf[2];   // whole layer of experts, for the batch path
    VkDeviceMemory  layer_mem[2];
    uint64_t        layer_bytes[2];
    qrt_vk_staging ** stg;    // one staging buffer per slot, all slot_bytes long
    uint32_t *      pend_layer;
    uint32_t *      pend_expert;
    uint64_t *      pend_bytes;
};

namespace {

void record_failure(int & rc, char * err, size_t err_size, const char * detail) {
    if (rc == 0) {
        rc = fail(err, err_size, "%s", detail[0] ? detail : "prefill operation failed");
    }
}

bool aligned_bytes(uint64_t bytes, uint64_t * aligned) {
    if (bytes == 0 || bytes > UINT64_MAX - 63) {
        return false;
    }
    *aligned = (bytes + 63) / 64 * 64;
    return true;
}

int validate_routes(uint32_t layers, uint32_t stride, const uint32_t * counts,
                    uint64_t * total, char * err, size_t err_size) {
    if ((uint64_t) layers * stride > SIZE_MAX / sizeof(uint32_t) ||
        layers > SIZE_MAX / sizeof(uint32_t)) {
        return fail(err, err_size, "router table size overflows");
    }
    *total = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        if (counts[l] > stride) {
            return fail(err, err_size, "layer %u count %u exceeds stride %u", l, counts[l], stride);
        }
        *total += counts[l];
    }
    if (*total == 0 || *total > UINT32_MAX || *total > SIZE_MAX / sizeof(uint32_t)) {
        return fail(err, err_size, "router block count is empty or overflows");
    }
    return 0;
}

} // namespace

uint32_t qrt_prefill_unique_experts(uint32_t chunk, uint32_t top_k, uint32_t experts) {
    if (experts == 0 || top_k == 0 || chunk == 0) {
        return 0;
    }
    const double picks = (double) chunk * (double) top_k;
    const double hidden = 1.0 - 1.0 / (double) experts;
    uint32_t n = (uint32_t) ((double) experts * (1.0 - pow(hidden, picks)) + 0.5);
    if (n == 0) {
        n = 1;
    }
    return n > experts ? experts : n;
}

int qrt_prefill_open(qrt_vk_dev * dev, qrt_weights * w, const qrt_prefill_cfg * cfg,
                     qrt_prefill ** out, char * err, size_t err_size) {
    if (out == NULL) {
        return fail(err, err_size, "prefill needs an output pointer");
    }
    *out = NULL;
    if (dev == NULL || w == NULL || cfg == NULL) {
        return fail(err, err_size, "prefill needs a device, a store and a config");
    }
    if (cfg->slots < 1 || cfg->slots > 64 || cfg->readers > 16) {
        return fail(err, err_size, "prefill needs 1..64 slots and 0..16 readers");
    }
    const uint64_t max_expert = qrt_weights_max_expert_bytes(w);
    uint64_t slot_bytes = cfg->slot_bytes != 0 ? cfg->slot_bytes : max_expert;
    if (cfg->group_bytes > slot_bytes) {
        const uint64_t group_bytes = ((uint64_t) cfg->group_bytes / 64 + 1) * 64;
        if (max_expert > UINT64_MAX - group_bytes) {
            return fail(err, err_size, "prefill group capacity overflows");
        }
        slot_bytes = group_bytes + max_expert;
    }
    const uint32_t want = cfg->slots;
    if (slot_bytes == 0 || slot_bytes > UINT64_MAX / want || slot_bytes > SIZE_MAX / want) {
        return fail(err, err_size, "prefill slot capacity is empty or overflows");
    }
    qrt_prefill * p = new (std::nothrow) qrt_prefill{};
    if (p == NULL) {
        return fail(err, err_size, "out of memory for the prefill scheduler");
    }
    p->dev = dev;
    p->w   = w;
    p->cfg = *cfg;
    p->cfg.readers = cfg->readers != 0 ? cfg->readers : 1;
    p->cfg.slot_reductions = 0;
    p->slot_bytes = slot_bytes;
    p->pool.readers = (int) p->cfg.readers;

    // Slots are one allocation: two multi-GiB blocks can fail to fit where the sum
    // does (the same reason runtime/proto keeps its dense and expert slices joined).
    if (qrt_vk_alloc(dev, p->slot_bytes * want,
                     QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &p->buf, &p->mem, err, err_size) != 0) {
        qrt_prefill_close(dev, p);
        return -1;
    }
    p->stg = (qrt_vk_staging **) calloc(want, sizeof(*p->stg));
    if (p->stg == NULL) {
        qrt_prefill_close(dev, p);
        return fail(err, err_size, "out of memory for %u staging slots", want);
    }
    // A failed slot takes half the ring with it, so the count drops by halves and
    // every step is counted; at one slot the scheduler is the serial baseline.
    for (uint32_t attempt = want; attempt >= 1; attempt /= 2) {
        p->slots = 0;
        while (p->slots < attempt) {
            if (qrt_vk_staging_create(dev, p->slot_bytes, &p->stg[p->slots],
                                      err, err_size) != 0) {
                break;
            }
            ++p->slots;
        }
        if (p->slots == attempt) {
            if (err != NULL && err_size != 0) {
                err[0] = '\0';
            }
            break;
        }
        for (uint32_t i = 0; i < p->slots; ++i) {
            qrt_vk_staging_destroy(dev, p->stg[i]);
            p->stg[i] = NULL;
        }
        ++p->cfg.slot_reductions;
        if (attempt == 1) {
            break;
        }
    }
    if (p->slots == 0) {
        qrt_prefill_close(dev, p);
        return -1;
    }
    p->pend_layer  = (uint32_t *) calloc(want, sizeof(uint32_t));
    p->pend_expert = (uint32_t *) calloc(want, sizeof(uint32_t));
    p->pend_bytes  = (uint64_t *) calloc(want, sizeof(uint64_t));
    if (p->pend_layer == NULL || p->pend_expert == NULL || p->pend_bytes == NULL) {
        qrt_prefill_close(dev, p);
        return fail(err, err_size, "out of memory for the slot bookkeeping");
    }
    try {
        p->pool.workers.reserve(p->pool.readers - 1);
        for (int id = 1; id < p->pool.readers; ++id) {
            p->pool.workers.emplace_back(pool_worker, &p->pool, id);
        }
    } catch (...) {
        qrt_prefill_close(dev, p);
        return fail(err, err_size, "cannot start the prefill readers");
    }
    *out = p;
    return 0;
}

void qrt_prefill_close(qrt_vk_dev * dev, qrt_prefill * p) {
    if (p == NULL) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(p->pool.mu);
        p->pool.stop = true;
    }
    p->pool.cv_start.notify_all();
    for (std::thread & t : p->pool.workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (p->stg != NULL) {
        for (uint32_t i = 0; i < p->slots; ++i) {
            qrt_vk_staging_destroy(dev, p->stg[i]);
        }
        free(p->stg);
    }
    if (dev != NULL && p->mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, p->buf, p->mem);
    }
    for (int i = 0; i < 2; ++i) {
        if (dev != NULL && p->layer_mem[i] != VK_NULL_HANDLE) {
            qrt_vk_free(dev, p->layer_buf[i], p->layer_mem[i]);
        }
    }
    free(p->pend_layer);
    free(p->pend_expert);
    free(p->pend_bytes);
    delete p;
}

int qrt_prefill_run(qrt_prefill * p, const uint32_t * ids, uint32_t stride,
                    const uint32_t * counts, const qrt_prefill_backend * backend,
                    qrt_prefill_stats * out, char * err, size_t err_size) {
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (p == NULL || ids == NULL || counts == NULL || out == NULL ||
        backend == NULL || backend->compute == NULL) {
        return fail(err, err_size, "prefill run needs ids, counts, stats and a compute backend");
    }
    qrt_vk_dev * dev = p->dev;
    out->slot_reductions = p->cfg.slot_reductions;
    const uint32_t layers = qrt_weights_layers(p->w);
    uint64_t total = 0;
    if (validate_routes(layers, stride, counts, &total, err, err_size) != 0) {
        return -1;
    }

    // Flatten (layer, expert) once so the ring has a single index to work with.
    uint32_t * flat_layer = (uint32_t *) malloc((size_t) total * sizeof(uint32_t));
    uint32_t * flat_expert = (uint32_t *) malloc((size_t) total * sizeof(uint32_t));
    if (flat_layer == NULL || flat_expert == NULL) {
        free(flat_layer);
        free(flat_expert);
        return fail(err, err_size, "out of memory for %llu blocks", (unsigned long long) total);
    }
    uint64_t n = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t e = 0; e < counts[l]; ++e) {
            flat_layer[n] = l;
            flat_expert[n] = ids[(size_t) l * stride + e];
            ++n;
        }
    }
    out->layers = layers;
    out->blocks = (uint32_t) total;

    const double t_start = now_ms();
    int rc = 0;
    auto finish_slot = [&](uint32_t slot) {
        if (p->pend_bytes[slot] == 0) {
            return;
        }
        char detail[256] = {};
        const double w0 = now_ms();
        if (qrt_vk_staging_wait(dev, p->stg[slot], detail, sizeof(detail)) != 0) {
            ++out->blocks_failed;
            record_failure(rc, err, err_size, detail);
        }
        out->wait_ms += now_ms() - w0;
        if (rc == 0) {
            detail[0] = '\0';
            const double c0 = now_ms();
            if (backend->compute(backend->user, p->pend_layer[slot], p->pend_expert[slot], p->buf,
                                 (uint64_t) slot * p->slot_bytes, p->pend_bytes[slot],
                                 detail, sizeof(detail)) != 0) {
                ++out->blocks_failed;
                record_failure(rc, err, err_size, detail);
            } else {
                ++out->blocks_done;
            }
            out->compute_ms += now_ms() - c0;
        }
        p->pend_bytes[slot] = 0;
    };
    uint64_t submitted = 0;
    for (uint64_t i = 0; i < total && rc == 0; ++i) {
        const uint32_t slot = (uint32_t) (i % p->slots);
        if (i >= p->slots) {
            finish_slot(slot);
            if (rc != 0) {
                break;
            }
        }
        char detail[256] = {};
        qrt_expert_loc loc;
        if (qrt_weights_locate(p->w, flat_layer[i], flat_expert[i], &loc,
                               detail, sizeof(detail)) != 0) {
            ++out->blocks_failed;
            record_failure(rc, err, err_size, detail);
            break;
        }
        if (loc.bytes == 0 || loc.bytes > p->slot_bytes) {
            ++out->blocks_failed;
            rc = fail(err, err_size, "expert %u,%u needs %llu bytes, the slot holds %llu",
                      flat_layer[i], flat_expert[i], (unsigned long long) loc.bytes,
                      (unsigned long long) p->slot_bytes);
            break;
        }
        unsigned char * map = (unsigned char *) qrt_vk_staging_map(p->stg[slot]);
        detail[0] = '\0';
        const double r0 = now_ms();
        const int read = qrt_weights_read_expert(p->w, flat_layer[i], flat_expert[i],
                                                map, (size_t) p->slot_bytes, detail, sizeof(detail));
        out->read_ms += now_ms() - r0;
        if (read != 0) {
            ++out->blocks_failed;
            record_failure(rc, err, err_size, detail);
            break;
        }
        detail[0] = '\0';
        const double u0 = now_ms();
        const int upload = qrt_vk_upload_submit(dev, p->stg[slot], p->buf,
                                                (uint64_t) slot * p->slot_bytes, loc.bytes,
                                                detail, sizeof(detail));
        out->upload_ms += now_ms() - u0;
        if (upload != 0) {
            ++out->blocks_failed;
            record_failure(rc, err, err_size, detail);
            break;
        }
        p->pend_layer[slot]  = flat_layer[i];
        p->pend_expert[slot] = flat_expert[i];
        p->pend_bytes[slot]  = loc.bytes;
        out->bytes += loc.bytes;
        ++out->groups;
        ++submitted;
    }

    const uint64_t pending = submitted < p->slots ? submitted : p->slots;
    for (uint64_t i = submitted - pending; i < submitted; ++i) {
        finish_slot((uint32_t) (i % p->slots));
    }
    out->total_ms = now_ms() - t_start;
    free(flat_layer);
    free(flat_expert);
    return rc;
}

VkBuffer qrt_prefill_buffer(const qrt_prefill * p) { return p != NULL ? p->buf : VK_NULL_HANDLE; }
uint64_t qrt_prefill_slot_bytes(const qrt_prefill * p) { return p != NULL ? p->slot_bytes : 0; }

uint32_t qrt_prefill_layer_buffers(const qrt_prefill * p) {
    return (p != NULL && p->cfg.pipeline) ? 2u : 1u;
}

namespace {

// Makes sure buffer `slot` holds at least `need` bytes, growing or shrinking it.
int ensure_layer_buffer(qrt_prefill * p, uint32_t slot, uint64_t need, char * err,
                        size_t err_size) {
    if (p->layer_bytes[slot] >= need) {
        return 0;
    }
    if (p->layer_mem[slot] != VK_NULL_HANDLE) {
        qrt_vk_free(p->dev, p->layer_buf[slot], p->layer_mem[slot]);
        p->layer_buf[slot] = VK_NULL_HANDLE;
        p->layer_mem[slot] = VK_NULL_HANDLE;
        p->layer_bytes[slot] = 0;
    }
    if (qrt_vk_alloc(p->dev, need, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                     &p->layer_buf[slot], &p->layer_mem[slot], err, err_size) != 0) {
        return -1;
    }
    p->layer_bytes[slot] = need;
    return 0;
}

// Streams one layer's experts into the given buffer through the staging ring. Consecutive
// experts share one copy when group_bytes is set, because a 1.8 MiB vkCmdCopyBuffer only
// reaches 11.6 GB/s while an 8-16 MiB one reaches 13.8-14.0 (RESULTS 35/43).
int upload_layer(qrt_prefill * p, uint32_t layer, uint32_t slot_index, const uint32_t * ids,
                 uint32_t stride, uint32_t count, uint32_t * experts, uint64_t * offsets,
                 uint64_t * sizes, qrt_prefill_stats * out, char * err, size_t err_size) {
    qrt_vk_dev * dev = p->dev;
    const VkBuffer dst = p->layer_buf[slot_index];
    const uint64_t group = p->cfg.group_bytes;

    // The destination layout has to be known before a group can be assembled.
    uint64_t off = 0;
    for (uint32_t e = 0; e < count; ++e) {
        qrt_expert_loc loc;
        if (qrt_weights_locate(p->w, layer, ids[(size_t) layer * stride + e], &loc,
                               err, err_size) != 0) {
            ++out->blocks_failed;
            return -1;
        }
        uint64_t aligned = 0;
        if (!aligned_bytes(loc.bytes, &aligned) || off > p->layer_bytes[slot_index] ||
            aligned > p->layer_bytes[slot_index] - off) {
            ++out->blocks_failed;
            return fail(err, err_size, "expert layout exceeds the layer buffer");
        }
        experts[e] = ids[(size_t) layer * stride + e];
        sizes[e] = loc.bytes;
        offsets[e] = off;
        off += aligned;
    }

    int rc = 0;
    uint32_t pending[64] = {};
    auto finish_slot = [&](uint32_t slot) {
        if (pending[slot] == 0) {
            return;
        }
        char detail[256] = {};
        const double w0 = now_ms();
        if (qrt_vk_staging_wait(dev, p->stg[slot], detail, sizeof(detail)) != 0) {
            out->blocks_failed += pending[slot];
            record_failure(rc, err, err_size, detail);
        }
        out->wait_ms += now_ms() - w0;
        pending[slot] = 0;
    };
    uint32_t e = 0;
    while (e < count && rc == 0) {
        uint32_t last = e;
        if (group == 0) {
            ++last;
        } else {
            const uint64_t start = offsets[e];
            while (last < count && offsets[last] + sizes[last] - start < group) {
                ++last;
            }
            if (last == e) {
                ++last;
            }
        }
        const uint64_t span_off = offsets[e];
        const uint64_t span_bytes = offsets[last - 1] + sizes[last - 1] - span_off;

        const uint32_t slot = (uint32_t) (out->groups % p->slots);
        finish_slot(slot);
        if (rc != 0) {
            break;
        }
        unsigned char * map = (unsigned char *) qrt_vk_staging_map(p->stg[slot]);
        if (span_bytes > p->slot_bytes) {
            out->blocks_failed += last - e;
            rc = fail(err, err_size, "group of %llu bytes exceeds the %llu byte staging slot",
                      (unsigned long long) span_bytes, (unsigned long long) p->slot_bytes);
            break;
        }
        const double r0 = now_ms();
        if (p->pool.readers <= 1) {
            for (uint32_t i = e; i < last; ++i) {
                char detail[256] = {};
                if (qrt_weights_read_expert(p->w, layer, experts[i],
                                            map + (offsets[i] - span_off),
                                            (size_t) sizes[i], detail, sizeof(detail)) != 0) {
                    ++out->blocks_failed;
                    record_failure(rc, err, err_size, detail);
                    break;
                }
            }
        } else {
            par_job * job = &p->job;
            job->w = p->w;
            job->layer = layer;
            job->experts = experts;
            job->offsets = offsets;
            job->sizes = sizes;
            job->first = e;
            job->last = last;
            job->span_off = span_off;
            job->map = map;
            job->mapped = qrt_weights_mapped(p->w) != 0;
            job->failed.store(0);
            job->err[0] = '\0';
            {
                std::lock_guard<std::mutex> lk(p->pool.mu);
                p->pool.job = job;
                p->pool.done = 0;
                ++p->pool.epoch;
            }
            p->pool.cv_start.notify_all();
            pool_read(&p->pool, job, 0);
            const int want = p->pool.readers - 1;
            {
                std::unique_lock<std::mutex> lk(p->pool.mu);
                p->pool.cv_done.wait(lk, [&] { return p->pool.done == want; });
                p->pool.job = nullptr;
            }
            if (job->failed.load() != 0) {
                out->blocks_failed += (uint32_t) job->failed.load();
                record_failure(rc, err, err_size, job->err);
            }
        }
        out->read_ms += now_ms() - r0;
        if (rc != 0) {
            break;
        }
        char detail[256] = {};
        const double u0 = now_ms();
        const int upload = qrt_vk_upload_submit(dev, p->stg[slot], dst, span_off, span_bytes,
                                                detail, sizeof(detail));
        out->upload_ms += now_ms() - u0;
        if (upload != 0) {
            out->blocks_failed += last - e;
            record_failure(rc, err, err_size, detail);
            break;
        }
        pending[slot] = last - e;
        for (uint32_t i = e; i < last; ++i) {
            out->bytes += sizes[i];
        }
        ++out->groups;
        e = last;
    }
    for (uint32_t k = 0; k < p->slots; ++k) {
        finish_slot(k);
    }
    return rc;
}

} // namespace

int qrt_prefill_run_layers(qrt_prefill * p, const uint32_t * ids, uint32_t stride,
                           const uint32_t * counts, const qrt_prefill_backend * backend,
                           qrt_prefill_stats * out, char * err, size_t err_size) {
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (p == NULL || ids == NULL || counts == NULL || out == NULL || backend == NULL ||
        backend->batch == NULL || backend->wait == NULL) {
        return fail(err, err_size, "the layer batch needs ids, counts, stats and batch/wait backend");
    }
    out->slot_reductions = p->cfg.slot_reductions;
    out->pipeline = p->cfg.pipeline ? 1u : 0u;
    const uint32_t layers = qrt_weights_layers(p->w);
    uint64_t total = 0;
    if (validate_routes(layers, stride, counts, &total, err, err_size) != 0) {
        return -1;
    }
    out->blocks = (uint32_t) total;

    // One buffer has to hold the largest layer, so size it from the table, not a guess.
    uint64_t need = 0;
    uint32_t n_active = 0;
    uint32_t max_count = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        if (counts[l] == 0) {
            continue;
        }
        ++n_active;
        if (counts[l] > max_count) {
            max_count = counts[l];
        }
        uint64_t layer = 0;
        for (uint32_t e = 0; e < counts[l]; ++e) {
            qrt_expert_loc loc;
            if (qrt_weights_locate(p->w, l, ids[(size_t) l * stride + e], &loc,
                                   err, err_size) != 0) {
                ++out->blocks_failed;
                return -1;
            }
            uint64_t aligned = 0;
            if (!aligned_bytes(loc.bytes, &aligned) || aligned > SIZE_MAX ||
                layer > SIZE_MAX - aligned) {
                ++out->blocks_failed;
                return fail(err, err_size, "layer %u byte layout overflows", l);
            }
            layer += aligned;
        }
        if (layer > need) {
            need = layer;
        }
    }
    if (max_count > SIZE_MAX / sizeof(uint64_t) || n_active > SIZE_MAX / sizeof(uint32_t)) {
        return fail(err, err_size, "layer bookkeeping size overflows");
    }
    const uint32_t buffers = qrt_prefill_layer_buffers(p);
    for (uint32_t i = 0; i < buffers; ++i) {
        if (ensure_layer_buffer(p, i, need, err, err_size) != 0) {
            return -1;
        }
    }

    uint32_t * active = (uint32_t *) malloc((size_t) n_active * sizeof(uint32_t));
    uint64_t * offsets = (uint64_t *) malloc((size_t) max_count * sizeof(uint64_t));
    uint64_t * sizes = (uint64_t *) malloc((size_t) max_count * sizeof(uint64_t));
    uint32_t * experts = (uint32_t *) malloc((size_t) max_count * sizeof(uint32_t));
    if (active == NULL || offsets == NULL || sizes == NULL || experts == NULL) {
        free(active); free(offsets); free(sizes); free(experts);
        return fail(err, err_size, "out of memory for %u slot offsets", max_count);
    }
    uint32_t w = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        if (counts[l] != 0) {
            active[w++] = l;
        }
    }

    const double t_start = now_ms();
    int rc = 0;
    bool inflight[2] = {};
    bool batch_ok[2] = {};
    uint32_t ring_count[2] = {};
    uint32_t last_ring = 0;
    auto finish_ring = [&](uint32_t ring) {
        if (!inflight[ring]) {
            return;
        }
        char detail[256] = {};
        const double w0 = now_ms();
        if (backend->wait(backend->user, ring, detail, sizeof(detail)) != 0) {
            if (batch_ok[ring]) {
                out->blocks_failed += ring_count[ring];
            }
            record_failure(rc, err, err_size, detail);
        } else if (batch_ok[ring]) {
            out->blocks_done += ring_count[ring];
        }
        out->wait_ms += now_ms() - w0;
        inflight[ring] = false;
    };
    for (uint32_t a = 0; a < n_active && rc == 0; ++a) {
        const uint32_t layer = active[a];
        const uint32_t slot = buffers == 2 ? (a & 1u) : 0u;
        finish_ring(slot);
        if (rc != 0) {
            break;
        }
        if (upload_layer(p, layer, slot, ids, stride, counts[layer], experts,
                         offsets, sizes, out, err, err_size) != 0) {
            rc = -1;
            break;
        }
        // A failed callback can still have submitted part of this ring.
        inflight[slot] = true;
        ring_count[slot] = counts[layer];
        last_ring = slot;
        char detail[256] = {};
        const double c0 = now_ms();
        batch_ok[slot] = backend->batch(backend->user, layer, slot, p->layer_buf[slot],
                                        experts, offsets, sizes, counts[layer],
                                        detail, sizeof(detail)) == 0;
        out->compute_ms += now_ms() - c0;
        if (!batch_ok[slot]) {
            out->blocks_failed += counts[layer];
            record_failure(rc, err, err_size, detail);
            break;
        }
        ++out->layers;
    }
    for (uint32_t k = 0; k < buffers; ++k) {
        finish_ring((last_ring + buffers - k) % buffers);
    }
    out->total_ms = now_ms() - t_start;
    free(active);
    free(offsets);
    free(sizes);
    free(experts);
    return rc;
}
