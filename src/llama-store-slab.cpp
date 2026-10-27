#include "llama-store-slab.h"

#include <cassert>

int32_t llama_store_slab::add_class(int64_t bytes, int32_t slots) {
    assert(bytes > 0 && slots > 0);
    size_cls & c = m_classes.emplace_back();
    c.bytes        = bytes;
    c.slots.resize(slots);
    return (int32_t) m_classes.size() - 1;
}

void llama_store_slab::grow_class(int32_t cls, int32_t extra_slots) {
    assert(cls >= 0 && cls < (int32_t) m_classes.size());
    assert(extra_slots > 0);
    m_classes[cls].slots.resize(m_classes[cls].slots.size() + extra_slots);
}

llama_store_slab::slot_state & llama_store_slab::state(const handle & h) {
    assert(h.cls >= 0 && h.cls < (int32_t) m_classes.size());
    assert(h.slot >= 0 && h.slot < (int32_t) m_classes[h.cls].slots.size());
    return m_classes[h.cls].slots[h.slot];
}

const llama_store_slab::slot_state & llama_store_slab::state(const handle & h) const {
    assert(h.cls >= 0 && h.cls < (int32_t) m_classes.size());
    assert(h.slot >= 0 && h.slot < (int32_t) m_classes[h.cls].slots.size());
    return m_classes[h.cls].slots[h.slot];
}

bool llama_store_slab::alloc(int32_t cls, uint8_t tier, handle * out) {
    assert(cls >= 0 && cls < (int32_t) m_classes.size());
    assert(out != nullptr);
    size_cls & c = m_classes[cls];
    const int32_t n = (int32_t) c.slots.size();
    for (int32_t i = 0; i < n; i++) {
        slot_state & s = c.slots[i];
        if (s.used) {
            continue;
        }
        s.used  = true;
        s.tier  = tier;
        s.pins  = 0;
        s.unconsumed = 0;
        s.in_flight  = false;
        s.gen++; // every (re)allocation carries a fresh generation
        handle h;
        h.cls  = cls;
        h.slot = i;
        h.gen  = s.gen;
        *out   = h;
        c.used++;
        return true;
    }
    return false;
}

void llama_store_slab::set_tier(const handle & h, uint8_t tier) {
    state(h).tier = tier;
}

void llama_store_slab::pin(const handle & h) {
    assert(state(h).gen == h.gen);
    state(h).pins++;
}

void llama_store_slab::unpin(const handle & h) {
    slot_state & s = state(h);
    assert(s.gen == h.gen);
    assert(s.pins > 0);
    s.pins--;
}

void llama_store_slab::begin_transfer(const handle & h) {
    slot_state & s = state(h);
    assert(s.gen == h.gen);
    assert(!s.in_flight);
    s.in_flight = true;
}

void llama_store_slab::end_transfer(const handle & h) {
    slot_state & s = state(h);
    assert(s.gen == h.gen && s.in_flight);
    s.in_flight = false;
}

void llama_store_slab::hand_out(const handle & h) {
    assert(state(h).gen == h.gen);
    state(h).unconsumed++;
}

void llama_store_slab::consume(const handle & h) {
    slot_state & s = state(h);
    assert(s.gen == h.gen);
    assert(s.unconsumed > 0);
    s.unconsumed--;
}

uint32_t llama_store_slab::busy(const handle & h) const {
    const slot_state & s = state(h);
    if (s.gen != h.gen || !s.used) {
        return 0xffffffffu; // stale slot holds nothing: nothing to evict either
    }
    uint32_t b = 0;
    if (s.pins > 0) {
        b |= BUSY_PIN;
    }
    if (s.in_flight) {
        b |= BUSY_INFLIGHT;
    }
    if (s.unconsumed != 0u) {
        b |= BUSY_UNCONSUMED;
    }
    return b;
}

bool llama_store_slab::evict(const handle & h) {
    if (h.cls < 0 || h.cls >= (int32_t) m_classes.size()) {
        return false;
    }
    const slot_state & s = m_classes[h.cls].slots[h.slot];
    if (s.gen != h.gen || !s.used) {
        return false; // stale handle: the slot has already been reused
    }
    if (busy(h) != 0) {
        return false; // the single captioned primitive is the eviction guard
    }
    slot_state & s2 = m_classes[h.cls].slots[h.slot];
    s2.used = false;
    s2.pins = 0;
    s2.unconsumed = 0;
    s2.in_flight = false;
    s2.tier = 0;
    s2.gen++; // stale handles died here, not in slot reuse
    m_classes[h.cls].used--;
    return true;
}

int64_t llama_store_slab::bytes(uint8_t tier) const {
    int64_t sum = 0;
    for (const size_cls & c : m_classes) {
        for (const slot_state & s : c.slots) {
            if (s.used && s.tier == tier) {
                sum += c.bytes;
            }
        }
    }
    return sum;
}

int32_t llama_store_slab::used_slots(uint8_t tier) const {
    int32_t n = 0;
    for (const size_cls & c : m_classes) {
        for (const slot_state & s : c.slots) {
            n += s.used && s.tier == tier ? 1 : 0;
        }
    }
    return n;
}

int64_t llama_store_slab::class_bytes(int32_t cls) const {
    assert(cls >= 0 && cls < (int32_t) m_classes.size());
    return m_classes[cls].bytes;
}
