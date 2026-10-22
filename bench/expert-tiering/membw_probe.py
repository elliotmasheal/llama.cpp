"""CPU memory bandwidth probe for the H-mode ceiling (cold experts computed on the host).

Usage: python membw_probe.py [size_mib]
Prints read-only and copy bandwidth for 1..16 threads. numpy releases the GIL in
copy and reduction loops, so threads scale the load. Best of 3 reps per point.
"""
import sys
import threading
import time

import numpy as np


def run(n_threads, size_mib, reps=3):
    n = size_mib * 1024 * 1024 // 4
    per = n // n_threads
    src = [np.ones(per, dtype=np.float32) for _ in range(n_threads)]
    dst = [np.empty(per, dtype=np.float32) for _ in range(n_threads)]
    sink = [0.0] * n_threads

    def copy_work(i):
        np.copyto(dst[i], src[i])

    def read_work(i):
        sink[i] = float(np.max(src[i]))

    res = {}
    for name, fn, traffic in (("read", read_work, 1), ("copy", copy_work, 2)):
        best = float("inf")
        for _ in range(reps):
            ths = [threading.Thread(target=fn, args=(i,)) for i in range(n_threads)]
            t0 = time.perf_counter()
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            best = min(best, time.perf_counter() - t0)
        res[name] = per * n_threads * 4 * traffic / best / 1e9
    return res


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    print(f"buffer per run = {size} MiB (float32), best of 3")
    print("threads  read_GB/s  copy_GB/s(read+write)")
    for th in (1, 2, 4, 8, 16):
        r = run(th, size)
        print(f"{th:7d}  {r['read']:9.1f}  {r['copy']:9.1f}")
