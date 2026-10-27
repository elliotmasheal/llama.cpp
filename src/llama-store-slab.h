#pragma once

#include <cstdint>
#include <vector>

// store-slab (bench/THREE-TIER.md F13): pure allocator of the store, the single owner of handles =
// class + slot + generation. No backend, no threads, no policy: io and policy sit on top of it.
//
// When this file is changed, the three eviction guards live in exactly one place: busy(). Eviction
// and handle resolution both read the same function; a second checkpoint encoded elsewhere is a
// defect (bench/DECISIONS.md hard constraint).
class llama_store_slab {
public:
    struct handle {
        int32_t cls   = -1; // size class index (0..)
        int32_t slot  = -1; // slot inside the class
        uint64_t gen  = 0;  // generation for stale-handle detection after evict
    };

    static constexpr uint32_t BUSY_PIN        = 1u << 0; // slot pinned for a pending transfer
    static constexpr uint32_t BUSY_INFLIGHT   = 1u << 1; // a copy to/from this slot is running
    static constexpr uint32_t BUSY_UNCONSUMED = 1u << 2; // a handed-out handle was not consumed yet

    int32_t add_class(int64_t bytes, int32_t slots);
    void    grow_class(int32_t cls, int32_t extra_slots); // shells grow lazily, slot indexes stay stable
    bool alloc(int32_t cls, uint8_t tier, handle * out); // fresh slot + fresh generation
    void set_tier(const handle & h, uint8_t tier);

    void pin(const handle & h);
    void unpin(const handle & h);
    void begin_transfer(const handle & h);
    void end_transfer(const handle & h);
    void hand_out(const handle & h);
    void consume(const handle & h);

    // the one busy primitive: bits = which guards currently block eviction; 0 = evictable
    uint32_t busy(const handle & h) const;
    bool evict(const handle & h); // free the slot, bump the generation; stale handles now miss

    int64_t bytes(uint8_t tier) const;            // bytes occupied per tier, for the split log
    int32_t used_slots(uint8_t tier) const;       // occupied slots per tier
    int64_t class_bytes(int32_t cls) const;

private:
    struct slot_state {
        uint64_t gen      = 1;
        uint8_t  tier     = 0; // tier where the data of this slot lives
        int32_t  pins     = 0;
        uint32_t unconsumed = 0;
        bool     in_flight = false;
        bool     used     = false;
    };

    struct size_cls {
        int64_t bytes = 0;
        int32_t used  = 0; // occupied slots of the class
        std::vector<slot_state> slots;
    };

    slot_state & state(const handle & h);
    const slot_state & state(const handle & h) const;

    std::vector<size_cls> m_classes;
};
