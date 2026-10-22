# Mirror of src/llama-moe-cache.cpp budget/slot math for LFM2.5-24B-A2B-UD-Q4_K_XL.
# Pure arithmetic from the source; no model load, no GPU.
import math

MiB = 1024 * 1024

# expert tensor nb[2] (bytes per expert) from gguf-dump.txt; per layer = gate+up+down
# group A: blk.2..39 except 7,8 (36 layers): gate Q4_K 108 MiB, up Q4_K 108 MiB, down Q5_K 132 MiB
gA = [108 * MiB, 108 * MiB, 132 * MiB]
# group B: blk.7, blk.8 (2 layers): gate Q5_K 132, up Q5_K 132, down Q6_K 157.5
gB = [132 * MiB, 132 * MiB, 157.5 * MiB]

groups = [
    ("A (Q4_K up/gate, Q5_K down)", 36, [x // 64 for x in gA], gA),
    ("B (Q5_K up/gate, Q6_K down)", 2, [x // 64 for x in gB], gB),
]

host_bytes = [n * sum(per_layer) for _, n, _, per_layer in groups]
host_total = sum(host_bytes)
print("host expert bytes total = %.1f MiB  (A %.1f + B %.1f)" %
      (host_total / MiB, host_bytes[0] / MiB, host_bytes[1] / MiB))

ALIGN = 256  # Vulkan minStorageBufferOffsetAlignment upper bound; 16/64/256 change <=1 slot


def alloc_size(per_slot, n_slots):
    return sum(((b * (n_slots + 1)) + ALIGN - 1) // ALIGN * ALIGN for b in per_slot)


print()
hdr = "cache  | gA slots nf lru cov%%  | gB slots nf lru cov%% | total cov%% | banks MiB"
print(hdr)
print("-" * len(hdr))
for size_mib in (0, 1024, 2048, 3072, 4096, 4608, 5120, 5632, 6144, 6400, 6656, 6912, 7168):
    size = size_mib * MiB
    row = []
    total_slots = 0
    total_max = 0
    banks = 0
    for gi, (name, nlay, per_slot, per_layer) in enumerate(groups):
        budget = size * host_bytes[gi] // host_total if host_total else 0
        n_slots = 0
        while n_slots < nlay * 64 and alloc_size(per_slot, n_slots + 1) <= budget:
            n_slots += 1
        n_full = 0
        if n_slots < 4:
            n_slots = 0
        else:
            n_full = min(5, n_slots // 64, nlay - 1)
            while n_full > 0 and n_slots - n_full * 64 < n_slots // 5:
                n_full -= 1
        lru = n_slots - n_full * 64
        cov = 100.0 * n_slots / (nlay * 64)
        total_slots += n_slots
        total_max += nlay * 64
        banks += alloc_size(per_slot, n_slots) if n_slots else 0
        row.append("%5d %2d %4d %5.1f" % (n_slots, n_full, lru, cov))
    print("%6d | %s | %s | %8.1f | %8.1f" %
          (size_mib, row[0], row[1], 100.0 * total_slots / total_max, banks / MiB))

print()
print("weights on GPU when all expert tensors are on CPU: %.1f MiB" % (816.7))
