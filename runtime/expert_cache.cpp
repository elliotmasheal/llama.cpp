#include "expert_cache.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <list>
#include <new>
#include <vector>

namespace {

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (err != NULL && err_size != 0) {
        vsnprintf(err, err_size, fmt, ap);
    }
    va_end(ap);
    return -1;
}

bool add_bytes(uint64_t * sum, uint64_t bytes) {
    if (bytes > UINT64_MAX - *sum) {
        return false;
    }
    *sum += bytes;
    return true;
}

bool multiply_bytes(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

enum slot_state { SLOT_FREE = 0, SLOT_PROBATION = 1, SLOT_PROTECTED = 2 };

struct slot_entry {
    uint32_t expert;
    uint8_t  state;
    bool     pending;
};

struct layer_cache {
    std::vector<slot_entry> entries;
    std::vector<int32_t> resident;
    std::vector<uint32_t> free_slots;
    std::list<uint32_t> probation;
    std::list<uint32_t> protected_;
    uint32_t protected_cap;
    uint32_t pending_count;
};

} // namespace

struct qrt_cache {
    qrt_cache_plan plan;
    qrt_cache_io io;
    std::vector<layer_cache> layers;
    qrt_cache_stats stats;
};

namespace {

void release_slot(qrt_cache * c, uint32_t layer, uint32_t slot) {
    layer_cache & lc = c->layers[layer];
    slot_entry & e = lc.entries[slot];
    if (e.state == SLOT_FREE) {
        return;
    }
    if (e.expert < lc.resident.size() && lc.resident[e.expert] == (int32_t) slot) {
        lc.resident[e.expert] = -1;
    }
    lc.probation.remove(slot);
    lc.protected_.remove(slot);
    if (e.pending) {
        e.pending = false;
        if (lc.pending_count > 0) {
            --lc.pending_count;
        }
        if (c->stats.pending > 0) {
            --c->stats.pending;
        }
    }
    e.state = SLOT_FREE;
    e.expert = 0;
    lc.free_slots.push_back(slot);
}

void evict_slot(qrt_cache * c, uint32_t layer, uint32_t slot) {
    layer_cache & lc = c->layers[layer];
    slot_entry & e = lc.entries[slot];
    if (e.state == SLOT_FREE) {
        return;
    }
    if (e.expert < lc.resident.size() && lc.resident[e.expert] == (int32_t) slot) {
        lc.resident[e.expert] = -1;
    }
    lc.probation.remove(slot);
    lc.protected_.remove(slot);
    e.state = SLOT_FREE;
    e.expert = 0;
    ++c->stats.evictions;
    ++c->stats.layer_evictions[layer];
}

int32_t claim_slot(qrt_cache * c, uint32_t layer, const std::vector<uint8_t> & claimed) {
    layer_cache & lc = c->layers[layer];
    if (!lc.free_slots.empty()) {
        const uint32_t slot = lc.free_slots.back();
        lc.free_slots.pop_back();
        return (int32_t) slot;
    }
    if (c->plan.policy == QRT_CACHE_STATIC) {
        return -1;
    }
    for (uint32_t slot : lc.probation) {
        if (!claimed[slot]) {
            return (int32_t) slot;
        }
    }
    if (c->plan.policy == QRT_CACHE_SLRU) {
        for (uint32_t slot : lc.protected_) {
            if (!claimed[slot]) {
                return (int32_t) slot;
            }
        }
    }
    return -1;
}

void insert_probation(layer_cache & lc, uint32_t slot, uint32_t expert) {
    slot_entry & e = lc.entries[slot];
    e.expert = expert;
    e.state = SLOT_PROBATION;
    e.pending = false;
    lc.resident[expert] = (int32_t) slot;
    lc.probation.push_back(slot);
}

void touch_lru(layer_cache & lc, uint32_t slot) {
    lc.probation.remove(slot);
    lc.probation.push_back(slot);
}

void promote_slru(qrt_cache * c, uint32_t layer, uint32_t slot) {
    layer_cache & lc = c->layers[layer];
    slot_entry & e = lc.entries[slot];
    lc.probation.remove(slot);
    if (lc.protected_cap == 0) {
        lc.probation.push_back(slot);
        return;
    }
    if (lc.protected_.size() < lc.protected_cap) {
        e.state = SLOT_PROTECTED;
        lc.protected_.push_back(slot);
        ++c->stats.promotions;
        return;
    }
    const uint32_t demoted = lc.protected_.front();
    lc.protected_.pop_front();
    lc.entries[demoted].state = SLOT_PROBATION;
    lc.probation.push_back(demoted);
    ++c->stats.demotions;
    e.state = SLOT_PROTECTED;
    lc.protected_.push_back(slot);
    ++c->stats.promotions;
}

} // namespace

int qrt_cache_plan_build(uint32_t layers, uint32_t experts, const uint64_t * slot_bytes,
                         const uint64_t * weights, uint64_t budget_bytes, int32_t policy,
                         uint32_t protected_percent, qrt_cache_plan * out,
                         char * err, size_t err_size) {
    if (out == NULL || slot_bytes == NULL || layers == 0 || layers > QRT_CACHE_MAX_LAYERS ||
        experts == 0 || experts > QRT_CACHE_MAX_EXPERTS) {
        return fail(err, err_size, "cache plan needs 1..%u layers, 1..%u experts and a slot table",
                    QRT_CACHE_MAX_LAYERS, QRT_CACHE_MAX_EXPERTS);
    }
    if (policy < QRT_CACHE_SLRU || policy > QRT_CACHE_STATIC) {
        return fail(err, err_size, "cache policy %d is not slru, lru or static", policy);
    }
    if (protected_percent > 100) {
        return fail(err, err_size, "protected percent %u is above 100", protected_percent);
    }
    *out = {};
    out->layers = layers;
    out->experts = experts;
    out->policy = policy;
    out->protected_percent = protected_percent;
    out->budget_bytes = budget_bytes;

    std::vector<uint64_t> uniform;
    if (weights == NULL) {
        uniform.assign(layers, 1);
        weights = uniform.data();
    }
    uint64_t floor_bytes = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        if (slot_bytes[l] == 0) {
            return fail(err, err_size, "layer %u has a zero-byte expert slot", l);
        }
        if (!add_bytes(&floor_bytes, slot_bytes[l])) {
            return fail(err, err_size, "slot table overflows");
        }
        out->slot_bytes[l] = slot_bytes[l];
    }

    if (budget_bytes >= floor_bytes) {
        for (uint32_t l = 0; l < layers; ++l) {
            out->slots_per_layer[l] = 1;
        }
        uint64_t bytes = floor_bytes;
        const uint64_t remaining = budget_bytes - floor_bytes;
        if (remaining != 0) {
            long double total_weight = 0;
            for (uint32_t l = 0; l < layers; ++l) {
                total_weight += (long double) weights[l];
            }
            if (total_weight > 0) {
                std::vector<long double> fractional(layers, 0);
                for (uint32_t l = 0; l < layers; ++l) {
                    const long double ideal = (long double) remaining * (long double) weights[l] /
                                              total_weight;
                    uint64_t extra = (uint64_t) floorl(ideal / (long double) slot_bytes[l]);
                    if (extra > (uint64_t) experts - 1u) {
                        extra = (uint64_t) experts - 1u;
                    }
                    out->slots_per_layer[l] += (uint32_t) extra;
                    uint64_t added = 0;
                    if (!multiply_bytes(extra, slot_bytes[l], &added) ||
                        !add_bytes(&bytes, added)) {
                        return fail(err, err_size, "cache plan bytes overflow");
                    }
                    fractional[l] = ideal - (long double) added;
                }
                uint64_t leftover = budget_bytes - bytes;
                while (leftover != 0) {
                    int32_t best = -1;
                    long double best_fraction = -1;
                    for (uint32_t l = 0; l < layers; ++l) {
                        if (out->slots_per_layer[l] < experts &&
                            slot_bytes[l] <= leftover && fractional[l] > best_fraction) {
                            best = (int32_t) l;
                            best_fraction = fractional[l];
                        }
                    }
                    if (best < 0) {
                        break;
                    }
                    ++out->slots_per_layer[best];
                    bytes += slot_bytes[best];
                    leftover -= slot_bytes[best];
                    fractional[best] -= (long double) slot_bytes[best];
                }
            }
        }
        out->bytes = bytes;
    } else {
        std::vector<uint32_t> order(layers);
        for (uint32_t l = 0; l < layers; ++l) {
            order[l] = l;
        }
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (weights[a] != weights[b]) {
                return weights[a] > weights[b];
            }
            return a < b;
        });
        uint64_t used = 0;
        for (uint32_t l : order) {
            if (slot_bytes[l] <= budget_bytes - used) {
                out->slots_per_layer[l] = 1;
                used += slot_bytes[l];
            }
        }
        out->bytes = used;
    }

    uint64_t offset = 0;
    uint64_t total_slots = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        out->layer_offset[l] = offset;
        total_slots += out->slots_per_layer[l];
        uint64_t added = 0;
        if (!multiply_bytes(out->slots_per_layer[l], slot_bytes[l], &added) ||
            !add_bytes(&offset, added)) {
            return fail(err, err_size, "cache layer offsets overflow");
        }
        if (out->slots_per_layer[l] == 0) {
            ++out->degraded_layers;
        }
    }
    if (offset != out->bytes) {
        return fail(err, err_size, "cache plan byte account disagrees");
    }
    if (offset > budget_bytes) {
        return fail(err, err_size, "cache plan %llu bytes above the %llu byte budget",
                    (unsigned long long) offset, (unsigned long long) budget_bytes);
    }
    out->slots = total_slots;
    return 0;
}

int qrt_cache_open(const qrt_cache_plan * plan, const qrt_cache_io * io, qrt_cache ** out,
                   char * err, size_t err_size) {
    if (out == NULL) {
        return fail(err, err_size, "cache needs an output pointer");
    }
    *out = NULL;
    if (plan == NULL || plan->layers == 0 || plan->layers > QRT_CACHE_MAX_LAYERS ||
        plan->experts == 0 || plan->experts > QRT_CACHE_MAX_EXPERTS ||
        plan->policy < QRT_CACHE_SLRU || plan->policy > QRT_CACHE_STATIC) {
        return fail(err, err_size, "cache needs a valid plan");
    }
    uint64_t accounted = 0;
    for (uint32_t l = 0; l < plan->layers; ++l) {
        uint64_t added = 0;
        if (plan->slots_per_layer[l] > plan->experts ||
            !multiply_bytes(plan->slots_per_layer[l], plan->slot_bytes[l], &added) ||
            !add_bytes(&accounted, added)) {
            return fail(err, err_size, "cache plan layer %u is inconsistent", l);
        }
        if (plan->layer_offset[l] != accounted - added) {
            return fail(err, err_size, "cache plan layer %u offset is inconsistent", l);
        }
    }
    if (accounted != plan->bytes || accounted > plan->budget_bytes) {
        return fail(err, err_size, "cache plan does not account bytes within budget");
    }
    qrt_cache * c = new (std::nothrow) qrt_cache{};
    if (c == NULL) {
        return fail(err, err_size, "out of memory for the expert cache");
    }
    c->plan = *plan;
    if (io != NULL) {
        c->io = *io;
    }
    c->layers.resize(plan->layers);
    for (uint32_t l = 0; l < plan->layers; ++l) {
        layer_cache & lc = c->layers[l];
        const uint32_t slots = plan->slots_per_layer[l];
        lc.entries.assign(slots, slot_entry{});
        lc.resident.assign(plan->experts, -1);
        lc.protected_cap = (uint32_t) ((uint64_t) slots * plan->protected_percent / 100u);
        if (slots > 1 && lc.protected_cap >= slots) {
            lc.protected_cap = slots - 1;
        }
        if (slots == 1) {
            lc.protected_cap = 0;
        }
        for (uint32_t s = slots; s > 0; --s) {
            lc.free_slots.push_back(s - 1);
        }
    }
    *out = c;
    return 0;
}

void qrt_cache_close(qrt_cache * c) {
    delete c;
}

int qrt_cache_acquire_mask(qrt_cache * c, uint32_t layer, const uint32_t * experts,
                           uint32_t count, const uint8_t * h2d_miss_mask,
                           qrt_cache_lookup * lookups, char * err, size_t err_size) {
    if (c == NULL || experts == NULL || lookups == NULL || layer >= c->plan.layers ||
        count > c->plan.experts) {
        return fail(err, err_size, "expert cache acquire needs a layer, ids and result rows");
    }
    if (count == 0) {
        return 0;
    }
    layer_cache & lc = c->layers[layer];
    if (lc.pending_count != 0) {
        return fail(err, err_size, "layer %u still has %u uploads in flight", layer,
                    lc.pending_count);
    }
    std::vector<uint8_t> claimed(lc.entries.size(), 0);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t expert = experts[i];
        if (expert >= c->plan.experts) {
            return fail(err, err_size, "expert %u is outside the %u expert table", expert,
                        c->plan.experts);
        }
        qrt_cache_lookup & r = lookups[i];
        r = {};
        r.expert = expert;
        r.slot = -1;
        r.ready = 1;
        bool duplicate = false;
        for (uint32_t j = 0; j < i; ++j) {
            if (lookups[j].expert == expert) {
                r = lookups[j];
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        ++c->stats.lookups;
        const int32_t resident = lc.resident[expert];
        if (resident >= 0) {
            slot_entry & e = lc.entries[resident];
            r.slot = resident;
            r.hit = 1;
            r.pending = e.pending ? 1 : 0;
            r.ready = e.pending ? 0 : 1;
            claimed[resident] = 1;
            ++c->stats.hits;
            ++c->stats.layer_hits[layer];
            if (c->plan.policy == QRT_CACHE_SLRU && e.state == SLOT_PROBATION) {
                promote_slru(c, layer, (uint32_t) resident);
            } else if (c->plan.policy == QRT_CACHE_LRU) {
                touch_lru(lc, (uint32_t) resident);
            }
            continue;
        }
        ++c->stats.misses;
        ++c->stats.layer_misses[layer];
        if (h2d_miss_mask != NULL && h2d_miss_mask[i] == 0) {
            // CPU-leg miss: no victim, no eviction, no upload.
            ++c->stats.cpu_leg_misses;
            r.failed = 1;
            r.ready = 0;
            continue;
        }
        const int32_t victim = claim_slot(c, layer, claimed);
        if (victim < 0) {
            ++c->stats.no_slot_misses;
            r.failed = 1;
            r.ready = 0;
            continue;
        }
        const uint32_t slot = (uint32_t) victim;
        if (lc.entries[slot].state != SLOT_FREE) {
            evict_slot(c, layer, slot);
        }
        insert_probation(lc, slot, expert);
        claimed[slot] = 1;
        r.slot = victim;
        c->stats.miss_bytes += c->plan.slot_bytes[layer];
        if (c->io.upload == NULL) {
            continue;
        }
        const uint64_t dst = c->plan.layer_offset[layer] +
                             (uint64_t) slot * c->plan.slot_bytes[layer];
        char detail[256] = {};
        if (c->io.upload(c->io.user, layer, expert, dst, c->plan.slot_bytes[layer], detail,
                         sizeof(detail)) != 0) {
            ++c->stats.upload_failures;
            release_slot(c, layer, slot);
            r.slot = -1;
            r.pending = 0;
            r.ready = 0;
            r.failed = 1;
            if (detail[0] != '\0') {
                snprintf(err, err_size, "%s", detail);
            }
            continue;
        }
        slot_entry & e = lc.entries[slot];
        e.pending = true;
        ++lc.pending_count;
        ++c->stats.pending;
        ++c->stats.h2d_uploads;
        c->stats.h2d_bytes += c->plan.slot_bytes[layer];
        r.pending = 1;
        r.ready = 0;
    }
    return 0;
}

int qrt_cache_acquire(qrt_cache * c, uint32_t layer, const uint32_t * experts, uint32_t count,
                      qrt_cache_lookup * lookups, char * err, size_t err_size) {
    return qrt_cache_acquire_mask(c, layer, experts, count, NULL, lookups, err, err_size);
}

int qrt_cache_resident(const qrt_cache * c, uint32_t layer, uint32_t expert) {
    if (c == NULL || layer >= c->plan.layers || expert >= c->plan.experts) {
        return 0;
    }
    return c->layers[layer].resident[expert] >= 0 ? 1 : 0;
}

void qrt_cache_note_upload(qrt_cache * c, uint32_t layer, uint64_t bytes) {
    (void) layer;
    if (c == NULL) {
        return;
    }
    ++c->stats.h2d_uploads;
    c->stats.h2d_bytes += bytes;
}

uint64_t qrt_cache_layer_base(const qrt_cache * c, uint32_t layer) {
    if (c == NULL || layer >= c->plan.layers) {
        return 0;
    }
    return c->plan.layer_offset[layer];
}

int qrt_cache_wait_expert(qrt_cache * c, uint32_t layer, uint32_t expert,
                          char * err, size_t err_size) {
    if (c == NULL || layer >= c->plan.layers || expert >= c->plan.experts) {
        return fail(err, err_size, "cache wait needs a valid layer and expert");
    }
    layer_cache & lc = c->layers[layer];
    const int32_t slot = lc.resident[expert];
    if (slot < 0 || !lc.entries[slot].pending) {
        return 0;
    }
    slot_entry & e = lc.entries[slot];
    char detail[256] = {};
    if (c->io.wait != NULL) {
        const uint64_t dst = c->plan.layer_offset[layer] +
                             (uint64_t) slot * c->plan.slot_bytes[layer];
        if (c->io.wait(c->io.user, layer, expert, dst, c->plan.slot_bytes[layer], detail,
                       sizeof(detail)) != 0) {
            ++c->stats.wait_failures;
            release_slot(c, layer, (uint32_t) slot);
            return fail(err, err_size, "%s", detail[0] ? detail : "cache wait failed");
        }
    }
    e.pending = false;
    if (lc.pending_count > 0) {
        --lc.pending_count;
    }
    if (c->stats.pending > 0) {
        --c->stats.pending;
    }
    return 0;
}

int qrt_cache_wait_all(qrt_cache * c, char * err, size_t err_size) {
    if (c == NULL) {
        return fail(err, err_size, "no cache to wait for");
    }
    int rc = 0;
    for (uint32_t l = 0; l < c->plan.layers; ++l) {
        layer_cache & lc = c->layers[l];
        for (uint32_t s = 0; s < lc.entries.size(); ++s) {
            slot_entry & e = lc.entries[s];
            if (!e.pending) {
                continue;
            }
            char detail[256] = {};
            if (c->io.wait != NULL) {
                const uint64_t dst = c->plan.layer_offset[l] +
                                     (uint64_t) s * c->plan.slot_bytes[l];
                if (c->io.wait(c->io.user, l, e.expert, dst, c->plan.slot_bytes[l], detail,
                               sizeof(detail)) != 0) {
                    ++c->stats.wait_failures;
                    if (rc == 0) {
                        fail(err, err_size, "%s", detail[0] ? detail : "cache wait failed");
                        rc = -1;
                    }
                    release_slot(c, l, s);
                    continue;
                }
            }
            e.pending = false;
            if (lc.pending_count > 0) {
                --lc.pending_count;
            }
            if (c->stats.pending > 0) {
                --c->stats.pending;
            }
        }
    }
    return rc;
}

void qrt_cache_get_stats(const qrt_cache * c, qrt_cache_stats * out) {
    if (out == NULL) {
        return;
    }
    *out = c != NULL ? c->stats : qrt_cache_stats{};
}

void qrt_cache_reset_stats(qrt_cache * c) {
    if (c == NULL) {
        return;
    }
    const uint32_t pending = c->stats.pending;
    c->stats = qrt_cache_stats{};
    c->stats.pending = pending;
}

uint64_t qrt_cache_bytes(const qrt_cache * c) {
    return c != NULL ? c->plan.bytes : 0;
}

uint32_t qrt_cache_layers(const qrt_cache * c) {
    return c != NULL ? c->plan.layers : 0;
}

uint32_t qrt_cache_slots(const qrt_cache * c, uint32_t layer) {
    return c != NULL && layer < c->plan.layers ? c->plan.slots_per_layer[layer] : 0;
}

uint64_t qrt_cache_slot_offset(const qrt_cache * c, uint32_t layer, uint32_t slot) {
    if (c == NULL || layer >= c->plan.layers || slot >= c->plan.slots_per_layer[layer]) {
        return UINT64_MAX;
    }
    return c->plan.layer_offset[layer] + (uint64_t) slot * c->plan.slot_bytes[layer];
}

struct qrt_cache_trace {
    uint32_t n_tokens;
    uint32_t n_layers;
    uint32_t top_k;
    uint32_t experts;
    std::vector<std::vector<uint32_t>> routes;
};

int qrt_cache_trace_open(const char * path, uint32_t experts, qrt_cache_trace ** out,
                         char * err, size_t err_size) {
    if (out == NULL) {
        return fail(err, err_size, "trace needs an output pointer");
    }
    *out = NULL;
    if (path == NULL || *path == '\0' || experts == 0 || experts > QRT_CACHE_MAX_EXPERTS) {
        return fail(err, err_size, "trace needs a path and a valid expert count");
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return fail(err, err_size, "cannot open trace '%s'", path);
    }
    int32_t header[3] = {};
    if (fread(header, sizeof(header), 1, f) != 1 || header[0] <= 0 || header[1] <= 0 ||
        header[1] > QRT_CACHE_MAX_LAYERS || header[2] <= 0 || header[2] > QRT_CACHE_MAX_EXPERTS) {
        fclose(f);
        return fail(err, err_size, "'%s': invalid routing trace header", path);
    }
    qrt_cache_trace * t = new (std::nothrow) qrt_cache_trace{};
    if (t == NULL) {
        fclose(f);
        return fail(err, err_size, "out of memory for the trace");
    }
    t->n_tokens = (uint32_t) header[0];
    t->n_layers = (uint32_t) header[1];
    t->top_k = (uint32_t) header[2];
    t->experts = experts;
    t->routes.resize(t->n_layers);
    for (uint32_t l = 0; l < t->n_layers; ++l) {
        int32_t n_ints = 0;
        if (fread(&n_ints, sizeof(n_ints), 1, f) != 1 || n_ints < 0 ||
            (uint64_t) n_ints > (uint64_t) t->n_tokens * t->top_k ||
            (uint32_t) n_ints % t->top_k != 0) {
            qrt_cache_trace_close(t);
            fclose(f);
            return fail(err, err_size, "'%s': layer %u record is inconsistent", path, l);
        }
        std::vector<int32_t> raw((size_t) n_ints);
        if (n_ints != 0 && fread(raw.data(), sizeof(int32_t), (size_t) n_ints, f) !=
                          (size_t) n_ints) {
            qrt_cache_trace_close(t);
            fclose(f);
            return fail(err, err_size, "'%s': layer %u record is truncated", path, l);
        }
        t->routes[l].resize((size_t) n_ints);
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] < 0 || (uint32_t) raw[i] >= experts) {
                qrt_cache_trace_close(t);
                fclose(f);
                return fail(err, err_size, "'%s': layer %u expert %d is outside 0..%u", path,
                            l, raw[i], experts - 1u);
            }
            t->routes[l][i] = (uint32_t) raw[i];
        }
    }
    const int trailing = fgetc(f);
    const int read_error = ferror(f);
    const int close_error = fclose(f);
    if (trailing != EOF || read_error || close_error != 0) {
        qrt_cache_trace_close(t);
        return fail(err, err_size, "'%s': trailing data or a read error", path);
    }
    *out = t;
    return 0;
}

void qrt_cache_trace_close(qrt_cache_trace * t) {
    delete t;
}

uint32_t qrt_cache_trace_tokens(const qrt_cache_trace * t) {
    return t != NULL ? t->n_tokens : 0;
}

uint32_t qrt_cache_trace_layers(const qrt_cache_trace * t) {
    return t != NULL ? t->n_layers : 0;
}

uint32_t qrt_cache_trace_top_k(const qrt_cache_trace * t) {
    return t != NULL ? t->top_k : 0;
}

uint32_t qrt_cache_trace_rows(const qrt_cache_trace * t, uint32_t layer) {
    if (t == NULL || layer >= t->n_layers) {
        return 0;
    }
    return (uint32_t) (t->routes[layer].size() / t->top_k);
}

const uint32_t * qrt_cache_trace_route(const qrt_cache_trace * t, uint32_t layer, uint32_t row) {
    if (t == NULL || layer >= t->n_layers || row >= qrt_cache_trace_rows(t, layer)) {
        return NULL;
    }
    return t->routes[layer].data() + (size_t) row * t->top_k;
}

int qrt_cache_trace_weights(const qrt_cache_trace * t, uint32_t decode_tokens, uint32_t layers,
                            uint64_t * weights, uint32_t * observed_layers,
                            char * err, size_t err_size) {
    if (t == NULL || weights == NULL || layers == 0 || decode_tokens == 0) {
        return fail(err, err_size, "trace weights need a trace, layer count and decode length");
    }
    memset(weights, 0, (size_t) layers * sizeof(*weights));
    uint32_t observed = 0;
    const uint32_t used = layers < t->n_layers ? layers : t->n_layers;
    for (uint32_t l = 0; l < used; ++l) {
        const uint32_t rows = qrt_cache_trace_rows(t, l);
        if (rows < decode_tokens) {
            continue;
        }
        weights[l] = (uint64_t) decode_tokens * t->top_k;
        ++observed;
    }
    if (observed_layers != NULL) {
        *observed_layers = observed;
    }
    return 0;
}

int qrt_cache_simulate(const qrt_cache_plan * plan, const qrt_cache_trace * t,
                       uint32_t decode_tokens, qrt_cache_stats * out,
                       uint32_t * observed_layers, uint32_t * decode_rows,
                       char * err, size_t err_size) {
    if (plan == NULL || t == NULL || out == NULL || decode_tokens == 0 ||
        plan->experts != t->experts) {
        return fail(err, err_size, "cache simulation needs a matching plan and trace");
    }
    qrt_cache * cache = NULL;
    if (qrt_cache_open(plan, NULL, &cache, err, err_size) != 0) {
        return -1;
    }
    std::vector<qrt_cache_lookup> lookups(t->top_k);
    uint32_t observed = 0;
    uint32_t rows_done = 0;
    const uint32_t used = plan->layers < t->n_layers ? plan->layers : t->n_layers;
    for (uint32_t l = 0; l < used; ++l) {
        const uint32_t rows = qrt_cache_trace_rows(t, l);
        if (rows < decode_tokens) {
            continue;
        }
        const uint32_t warm = rows - decode_tokens;
        for (uint32_t r = 0; r < warm; ++r) {
            if (qrt_cache_acquire(cache, l, qrt_cache_trace_route(t, l, r), t->top_k,
                                  lookups.data(), err, err_size) != 0) {
                qrt_cache_close(cache);
                return -1;
            }
        }
        ++observed;
        rows_done += decode_tokens;
    }
    if (observed == 0) {
        qrt_cache_close(cache);
        return fail(err, err_size, "the trace has no layer with %u decode rows", decode_tokens);
    }
    qrt_cache_reset_stats(cache);
    uint64_t failed_acquires = 0;
    for (uint32_t l = 0; l < used; ++l) {
        const uint32_t rows = qrt_cache_trace_rows(t, l);
        if (rows < decode_tokens) {
            continue;
        }
        const uint32_t start = rows - decode_tokens;
        for (uint32_t r = 0; r < decode_tokens; ++r) {
            if (qrt_cache_acquire(cache, l, qrt_cache_trace_route(t, l, start + r), t->top_k,
                                  lookups.data(), err, err_size) != 0) {
                ++failed_acquires;
                break;
            }
        }
        if (failed_acquires != 0) {
            break;
        }
    }
    qrt_cache_get_stats(cache, out);
    if (observed_layers != NULL) {
        *observed_layers = observed;
    }
    if (decode_rows != NULL) {
        *decode_rows = rows_done;
    }
    qrt_cache_close(cache);
    if (failed_acquires != 0) {
        return fail(err, err_size, "cache simulation acquire failed");
    }
    return 0;
}
