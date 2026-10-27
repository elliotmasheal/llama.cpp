// unit test of store-slab (bench/THREE-TIER.md F13 S1): pure allocator, no GPU required.
// The single busy primitive is the only eviction guard; pinning this contract to one function is
// what the DECISIONS.md hard constraint "one captioned primitive" means, so the test checks it.
#include "../src/llama-store-slab.h"

int main() {
    llama_store_slab slab;

    // two size classes: the expert class (1.945 MiB) and the KV page class (2.5 MiB), F10/F12
    const int32_t cls_e = slab.add_class(2039808, 4);
    const int32_t cls_p = slab.add_class(2621440, 2);

    llama_store_slab::handle ea, eb, pa;
    if (!slab.alloc(cls_e, 1 /* ram */, &ea)) return 1;
    if (!slab.alloc(cls_e, 1 /* ram */, &eb)) return 1;
    if (!slab.alloc(cls_p, 0 /* vram */, &pa)) return 1;
    if (!(ea.cls == cls_e && eb.cls == cls_e && pa.cls == cls_p && pa.slot == 0)) return 2;
    if (ea.slot == eb.slot) return 3;

    // a fresh slot is evictable: no pins, no in-flight copy, no unconsumed handles
    if (slab.busy(ea) != 0) return 4;

    // each guard alone blocks eviction, and the three bits stay distinguishable
    slab.pin(ea);
    if (slab.busy(ea) != llama_store_slab::BUSY_PIN) return 5;
    slab.unpin(ea);
    if (slab.busy(ea) != 0) return 6;

    slab.begin_transfer(ea);
    if (slab.busy(ea) != llama_store_slab::BUSY_INFLIGHT) return 7;
    slab.end_transfer(ea);
    if (slab.busy(ea) != 0) return 8;

    slab.hand_out(ea);
    if (slab.busy(ea) != llama_store_slab::BUSY_UNCONSUMED) return 9;
    slab.consume(ea);
    if (slab.busy(ea) != 0) return 10;

    // eviction frees the slot and bumps the generation, so stale handles can never alias
    if (!slab.evict(ea)) return 11;
    llama_store_slab::handle ec;
    if (!slab.alloc(cls_e, 1, &ec)) return 12;
    if (ec.slot != ea.slot) return 13;   // the freed slot is reused
    if (ec.gen != ea.gen + 2) return 14; // alloc and evict carry a fresh generation each

    // stale handle holds the old generation: resolve sees a mismatch, evict must be a no-op
    if (slab.evict(ea)) return 15;
    (void) slab.busy(ea);
    if (slab.busy(ec) != 0) return 16;

    // tier accounting of the split log: two experts on ram, one page on vram
    if (slab.bytes(1) != 2*2039808) return 17;
    if (slab.bytes(0) != 2621440) return 18;
    if (slab.used_slots(1) != 2 || slab.used_slots(0) != 1) return 19;

    // a full class refuses allocation (the shell degrades, never aborts)
    llama_store_slab::handle e2, e3, e4;
    if (!slab.alloc(cls_e, 1, &e2)) return 20;
    if (!slab.alloc(cls_e, 1, &e3)) return 21;
    if (slab.alloc(cls_e, 1, &e4)) return 22; // class is out of slots now
    (void) e4;

    // the shell-side remedy: grow the class and the next allocation lands in the new slots
    slab.grow_class(cls_e, 2);
    if (!slab.alloc(cls_e, 1, &e4)) return 23;
    if (slab.bytes(1) != 5*2039808) return 24; // eb + reused ec + e2 + e3 + e4
    return 0;
}
