"""Roofline model: what the hardware allows, what the current software gets.

Usage: python roofline.py
Constants come from bench/RESULTS.md sections 7-10 and the RX 6700S spec.
Prints per-leg ceilings, TG(cache, k), PP(chunk, TFLOPS), the VRAM capacity
curve, and the 0.85 x roofline target next to the 130 / 1000 hard floors.
"""
import sys

MIB = 1024 * 1024
GIB = 1024 * MIB
GB = 1e9

# hardware ceilings
VRAM_SPEC = 224.0 * GB          # RX 6700S, 128-bit x 14 Gbps
PCIE_SPEC = 15.754 * GB         # PCIe 4.0 x8, 128b/130b
DDR_SPEC = 76.8 * GB            # DDR5-4800, 2 x 64-bit
FP32_TFLOPS = 7.9               # design 6.4
VRAM_TOTAL_MIB = 8176
DESKTOP_MIB = 1000

# measured, RESULTS sections 7-10
VRAM_NOW = 156.5 * GB
H2D_SOLO = 14.17 * GB
H2D_NOW = 9.9 * GB              # under CPU read, lower bound of 8.9-11.2
CPU_EXPERT_NOW = 0.137          # ms per expert, 4 threads
EXPERT_BYTES = 1.828 * MIB      # gguf_dims.py, 41 layers x 256 experts
DENSE_BYTES = 2.030 * GIB
KV_MAIN = 20480                 # bytes per token, 10 full-attn main layers
KV_MTP = 22528                  # including the nextn layer
GLOP_PER_TOKEN = 6.0
HIT = {20: 0.773, 25: 0.816, 30: 0.854}   # RESULTS section 7
U = {4: 19.9, 5: 23.5, 8: 31.6}           # unique experts per layer, section 7
E = {4: 2.84, 5: 3.24, 8: 3.82}           # RESULTS section 3, full vocab
DRAFT_MIB = 33.0                           # 16K restricted output head per step

# R1 targets and the 0.85 rule
VRAM_TARGET = 190.0 * GB
H2D_TARGET = 12.5 * GB
CPU_EXPERT_TARGET = 0.070
CPU_EXPERT_SPEC = EXPERT_BYTES / DDR_SPEC * 1e3
ROOFLINE_FRACTION = 0.85
TG_MIN = 130.0
PP_MIN = 1000.0

SCEN = (
    ("now", VRAM_NOW, H2D_NOW, CPU_EXPERT_NOW),
    ("R1", VRAM_TARGET, H2D_TARGET, CPU_EXPERT_TARGET),
    ("spec", VRAM_SPEC, PCIE_SPEC, CPU_EXPERT_SPEC),
)


def legs():
    rows = [
        ("vram read GB/s", VRAM_SPEC / GB, VRAM_NOW / GB, VRAM_TARGET / GB, "software"),
        ("h2d solo GB/s", PCIE_SPEC / GB, H2D_SOLO / GB, H2D_SOLO / GB, "link"),
        ("h2d conc GB/s", PCIE_SPEC / GB, H2D_NOW / GB, H2D_TARGET / GB, "software"),
        ("cpu read GB/s", DDR_SPEC / GB, 40.0, 70.0, "software"),
        ("cpu ms/expert", CPU_EXPERT_SPEC, CPU_EXPERT_NOW, CPU_EXPERT_TARGET, "software"),
    ]
    print("leg               spec      now   target   now/spec  binds")
    for name, spec, now, target, kind in rows:
        ratio = 100.0 * now / spec
        if name == "cpu ms/expert":
            ratio = 100.0 * spec / now   # lower is better, so report spec/now
        print("%-16s %8.3f %8.3f %8.3f %9.0f%%  %s"
              % (name, spec, now, target, ratio, kind))


def cycle_ms(k, cache, vram, h2d, cpu_ms):
    miss = 40 * 8 * k * (1.0 - HIT[cache])
    miss_bytes = miss * EXPERT_BYTES
    slots = U[k] * 40
    gpu_bytes = DENSE_BYTES + max(0.0, slots - miss) * EXPERT_BYTES
    gpu = gpu_bytes / vram * 1e3
    h2d_ms = miss_bytes / h2d * 1e3
    cpu = miss * cpu_ms
    miss_ms = h2d_ms * cpu / (h2d_ms + cpu) if h2d_ms + cpu else 0.0
    draft = (k - 1) * DRAFT_MIB * MIB / vram * 1e3
    return max(gpu, miss_ms) + draft, gpu, miss_ms, miss, miss_bytes


def best_tg(vram, h2d, cpu):
    best = (0.0, 0, 0)
    for cache in (20, 25):
        for k in (4, 5):
            t = cycle_ms(k, cache, vram, h2d, cpu)[0]
            tg = E[k] * 1e3 / t
            if tg > best[0]:
                best = (tg, cache, k)
    return best


def tg_table():
    print("")
    print("TG: miss split by equal leg time, draft = (k-1) x 33 MiB, E = full vocab")
    print("cache  k     E   " + "  ".join("%-20s" % c[0] for c in SCEN))
    for cache in (20, 25, 30):
        for k in (4, 5):
            cells = []
            for _name, vram, h2d, cpu in SCEN:
                t, gpu, miss, mreq, mbytes = cycle_ms(k, cache, vram, h2d, cpu)
                cells.append("%5.1f ms %3.0f tok/s" % (t, E[k] * 1e3 / t))
            print("%4d%%  %d  %5.2f  " % (cache, k, E[k]) + "  ".join(cells))


def e_min_table():
    print("")
    print("E_min = 130 x cycle: the observed E must clear this, it is not a constant")
    print("cache  k     now     R1   spec")
    for cache in (20, 25):
        for k in (4, 5):
            vals = [TG_MIN * cycle_ms(k, cache, *s[1:])[0] / 1e3 for s in SCEN]
            print("%4d%%  %d  %6.2f  %6.2f  %6.2f" % (cache, k, *vals))


def pp_table():
    total = EXPERT_BYTES * 41 * 256
    per_layer = total / 40.0
    print("")
    print("PP: %.1f MiB experts per layer, every expert read once per chunk" % (per_layer / MIB))
    print("chunk  xfer14  xfer16   tf6     tf7     tf8   (xfer = transfer only)")
    for chunk in (512, 1024, 1420, 2048, 4096):
        cells = []
        for h2d in (H2D_SOLO, PCIE_SPEC):
            cells.append(chunk / (40.0 * per_layer / h2d))
        for tf in (6.0, 7.0, 8.0):
            t = GLOP_PER_TOKEN * 1e9 * chunk / (tf * 1e12) / 40.0
            cells.append(chunk / (40.0 * max(per_layer / H2D_SOLO, t)))
        print("%5d  " % chunk + "  ".join("%6.0f" % v for v in cells))
    need_tf = PP_MIN * GLOP_PER_TOKEN / 1e3
    chunk_min = (per_layer / H2D_SOLO) * (need_tf * 1e12) / (GLOP_PER_TOKEN * 1e9 / 40.0)
    print("  PP %.0f needs %.2f TFLOPS effective; at %.2f x roofline, %.2f"
          % (PP_MIN, need_tf, ROOFLINE_FRACTION, need_tf / ROOFLINE_FRACTION))
    print("  H2D limits PP below %.0f tokens per chunk at 14.17 GB/s (6 TFLOPS)" % chunk_min)


def capacity_table():
    print("")
    print("VRAM: total %d MiB, usable %d MiB, dense %.0f MiB, draft 100 MiB"
          % (VRAM_TOTAL_MIB, VRAM_TOTAL_MIB - DESKTOP_MIB, DENSE_BYTES / MIB))
    print("KV per token: %d B main, %d B with nextn (f16)" % (KV_MAIN, KV_MTP))
    print("ctx     kv MiB   cache@700   cache%700   cache%1100")
    for ctx in (10240, 16384, 32768, 65536, 262144):
        kv = KV_MTP * ctx / MIB
        base = VRAM_TOTAL_MIB - DESKTOP_MIB - DENSE_BYTES / MIB - 100 - kv
        print("%6d  %8.1f  %9.1f  %9.1f%%  %9.1f%%"
              % (ctx, kv, base - 700,
                 100.0 * (base - 700) * MIB / (EXPERT_BYTES * 41 * 256),
                 100.0 * (base - 1100) * MIB / (EXPERT_BYTES * 41 * 256)))


def main(_argv):
    legs()
    tg_table()
    e_min_table()
    pp_table()
    capacity_table()
    hw_tg, hw_cache, hw_k = best_tg(VRAM_SPEC, PCIE_SPEC, CPU_EXPERT_SPEC)
    r1_tg, r1_cache, r1_k = best_tg(VRAM_TARGET, H2D_TARGET, CPU_EXPERT_TARGET)
    print("")
    print("hardware roofline best: %.0f tok/s (cache %d%%, k=%d)" % (hw_tg, hw_cache, hw_k))
    print("0.85 x hardware roofline: TG >= %.0f (hard floor %.0f)"
          % (ROOFLINE_FRACTION * hw_tg, TG_MIN))
    print("R1-ready best: %.0f tok/s (cache %d%%, k=%d), 0.85 x it = %.0f"
          % (r1_tg, r1_cache, r1_k, ROOFLINE_FRACTION * r1_tg))
    t = cycle_ms(r1_k, r1_cache, VRAM_TARGET, H2D_TARGET, CPU_EXPERT_TARGET)[0]
    for e in (2.85, 3.24):
        tg = e * 1e3 / t
        print("  restricted-vocab sensitivity E=%.2f -> TG %.0f, 0.85 x it = %.0f"
              % (e, tg, ROOFLINE_FRACTION * tg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
