# Mixed static+cache slot math (mirrors src/llama-moe-cache.cpp) + measured CSV digest.
# Read-only. No GPU.
import csv, glob, os, math

MiB = 1024 * 1024

# per-expert bytes per layer (nb[2]) from GGUF
A_PER = [108 * MiB // 64, 108 * MiB // 64, 132 * MiB // 64]   # gate Q4_K, up Q4_K, down Q5_K
B_PER = [132 * MiB // 64, 132 * MiB // 64, 157.5 * MiB // 64] # gate Q5_K, up Q5_K, down Q6_K
ALIGN = 256
N_EXP = 64
N_USED = 4


def alloc(per, n):
    return sum(((b * (n + 1)) + ALIGN - 1) // ALIGN * ALIGN for b in per)


def group_stats(n_layers, per, budget):
    n = 0
    while n < n_layers * N_EXP and alloc(per, n + 1) <= budget:
        n += 1
    if n < N_USED:
        return 0, 0, 0, 0.0
    n_full = min(5, n // N_EXP, n_layers - 1)
    while n_full > 0 and n - n_full * N_EXP < n // 5:
        n_full -= 1
    lru = n - n_full * N_EXP
    return n, n_full, lru, 100.0 * n / (n_layers * N_EXP)


def evaluate(cpu_blocks, cache_mib):
    # cpu_blocks = N of -ncmoe: blocks 0..N-1 keep experts in host RAM
    host_layers_A = host_layers_B = 0
    for il in range(2, 40):
        if il >= cpu_blocks:
            continue  # expert weights stay on the GPU: not cacheable
        if il in (7, 8):
            host_layers_B += 1
        else:
            host_layers_A += 1
    host_A = host_layers_A * sum(A_PER)
    host_B = host_layers_B * sum(B_PER)
    host_tot = host_A + host_B
    size = cache_mib * MiB
    bud_A = size * host_A // host_tot if host_tot else 0
    bud_B = size * host_B // host_tot if host_tot else 0
    sA, fA, lA, cA = group_stats(host_layers_A, A_PER, bud_A)
    sB, fB, lB, cB = group_stats(host_layers_B, B_PER, bud_B)
    bank = (alloc(A_PER, sA) if sA else 0) + (alloc(B_PER, sB) if sB else 0)
    n_res = 38 - (host_layers_A + host_layers_B)   # MoE layers with GPU-resident experts
    # expected hits per decode token (4 of 64 per layer; hit prob = per-layer slot coverage)
    hits = n_res * N_USED
    if sA:
        hits += fA * N_USED
        if host_layers_A > fA:
            hits += (host_layers_A - fA) * N_USED * (lA / (host_layers_A - fA) / N_EXP)
    if sB:
        hits += fB * N_USED
        if host_layers_B > fB:
            hits += (host_layers_B - fB) * N_USED * (lB / (host_layers_B - fB) / N_EXP)
    ids = 38 * N_USED
    miss = ids - hits
    miss_MiB = miss * sum(A_PER) / MiB     # ~5.44 MiB/expert (A layout)
    # H2D-bound ceiling at 13.3-13.9 GB/s (L8 measured 13.3-13.9); report the 13.9 upper bound
    ceiling = 1000.0 / (miss_MiB * MiB / 13.9e9 * 1000.0)
    ceiling_lo = 1000.0 / (miss_MiB * MiB / 13.3e9 * 1000.0)
    return dict(cpu=cpu_blocks, cache=cache_mib, lA=host_layers_A, lB=host_layers_B,
                sA=sA, fA=fA, lruA=lA, cA=cA, sB=sB, lBf=fB, lruB=lB, cB=cB,
                bank=bank / MiB, nres=n_res, hits=hits, ids=ids, miss=miss,
                miss_MiB=miss_MiB, ceiling=ceiling,
                vram=(bank + 809 + n_res * 348) / MiB)


print("ncmoe cache  hostA hostB  sA  whole LRU  covA   sB  LRU  covB  res | bank   VRAM  hits/152 miss  MiB  H2Dceil")
for cpu, cache in [(40, 5120), (40, 5632), (40, 6144), (38, 5120), (36, 5120), (36, 5376),
                   (36, 5632), (35, 5120), (35, 5376), (34, 5120), (34, 4864), (34, 4608),
                   (32, 3072), (30, 2048)]:
    r = evaluate(cpu, cache)
    print("%5d %5d  %5d %5d %4d %5d %5d %5.1f%% %4d %4d %5.1f%% %4d | %6.0f %6.0f  %5.1f %4d %5.0f %7.1f" % (
        r["cpu"], r["cache"], r["lA"], r["lB"], r["sA"], r["fA"], r["lruA"], r["cA"],
        r["sB"], r["lruB"], r["cB"], r["nres"], r["bank"], r["vram"], r["hits"], r["miss"],
        r["miss_MiB"], r["ceiling"]))

print()
print("measured (CSV):")
for path in sorted(glob.glob(os.path.join("results", "*.csv"))):
    tag = os.path.splitext(os.path.basename(path))[0]
    try:
        rows = list(csv.DictReader(open(path, newline="")))
    except Exception as e:
        print("%-26s parse-fail %s" % (tag, e))
        continue
    out = []
    for row in rows:
        try:
            pp = int(row.get("n_prompt") or 0)
            gen = int(row.get("n_gen") or 0)
            ts = float(row.get("avg_ts") or 0)
            sd = float(row.get("stddev_ts") or 0)
        except Exception:
            continue
        ub = row.get("n_ubatch") or "-"
        thr = row.get("n_threads") or "-"
        fa = row.get("flash_attn") or "-"
        out.append("%s=%.2f(sd %.1f)ub%s t%s fa%s" % ("pp" + str(pp) if gen == 0 else "tg" + str(gen), ts, sd, ub, thr, fa))
    if out:
        print("%-26s %s" % (tag, " | ".join(out)))
