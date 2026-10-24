#!/usr/bin/env python3
"""
Feasibility probe for the SSD tier of the MoE runtime.

It reads the expert tensors of a GGUF with FILE_FLAG_NO_BUFFERING, so the page
cache does not hide the device, and reports what an expert-granularity miss
would cost.

Usage:
    python ssd_probe.py F:\\LLM\\llm\\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [--seq-mib 1024]

Only the Python standard library is used (no numpy / gguf-py).
"""

import argparse
import ctypes
import os
import random
import statistics
import struct
import sys
import threading
import time
from ctypes import wintypes

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000

# ggml type -> (block bytes, block elements)
GGML_TYPES = {
    0:  (4, 1),      # F32
    1:  (2, 1),      # F16
    2:  (18, 32),    # Q4_0
    3:  (20, 32),    # Q4_1
    6:  (22, 32),    # Q5_0
    7:  (24, 32),    # Q5_1
    8:  (34, 32),    # Q8_0
    10: (84, 256),   # Q2_K
    11: (110, 256),  # Q3_K
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
    15: (292, 256),  # Q8_K
}


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_str(f):
    n = read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def skip_value(f, t):
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t in (10, 11, 12):
        f.read(8)
    elif t == 8:
        read_str(f)
    elif t == 9:
        at = read_u32(f)
        n = read_u64(f)
        for _ in range(n):
            skip_value(f, at)
    else:
        raise ValueError("unknown gguf value type %d" % t)


def read_value(f, t):
    if t == 0:
        return struct.unpack("<B", f.read(1))[0]
    if t == 1:
        return struct.unpack("<b", f.read(1))[0]
    if t == 2:
        return struct.unpack("<H", f.read(2))[0]
    if t == 3:
        return struct.unpack("<h", f.read(2))[0]
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return struct.unpack("<B", f.read(1))[0] != 0
    if t == 8:
        return read_str(f)
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError("unsupported gguf value type %d" % t)


def parse_gguf(path):
    """returns (data_start, alignment, [tensor dicts])"""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a gguf file")
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)

        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            t = read_u32(f)
            if key == "general.alignment":
                alignment = read_value(f, t)
            else:
                skip_value(f, t)

        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            ttype = read_u32(f)
            offset = read_u64(f)
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
        return data_start, alignment, tensors, version


def tensor_nbytes(t):
    if t["type"] not in GGML_TYPES:
        return None
    block_bytes, block_elems = GGML_TYPES[t["type"]]
    ne = 1
    for d in t["dims"]:
        ne *= d
    return ne // block_elems * block_bytes


def tensor_expert_bytes(t):
    """bytes of one expert slice (nb[2]) and the number of experts"""
    if t["type"] not in GGML_TYPES or len(t["dims"]) != 3:
        return None, 0
    block_bytes, block_elems = GGML_TYPES[t["type"]]
    row = t["dims"][0] * t["dims"][1]
    return row // block_elems * block_bytes, t["dims"][2]


class RawFile:
    def __init__(self, path):
        self.k32 = ctypes.windll.kernel32
        self.k32.CreateFileW.restype = wintypes.HANDLE
        self.k32.CreateFileW.argtypes = [ctypes.c_wchar_p, wintypes.DWORD, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self.k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
        self.k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.k32.VirtualAlloc.restype = ctypes.c_void_p
        self.k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        self.k32.CloseHandle.argtypes = [wintypes.HANDLE]

        self.h = self.k32.CreateFileW(
            path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING, None)
        if self.h == wintypes.HANDLE(-1).value:
            raise OSError("CreateFileW failed: %d" % ctypes.get_last_error())
        self.sector = 4096
        # page aligned buffer, kept for the lifetime of the object
        self.buf_size = 64 * 1024 * 1024 + self.sector
        self.buf = self.k32.VirtualAlloc(None, self.buf_size, 0x1000 | 0x2000, 0x04)
        if not self.buf:
            raise OSError("VirtualAlloc failed")

    def read(self, offset, size):
        """reads [offset, offset+size), both rounded to the sector size"""
        start = offset // self.sector * self.sector
        end = (offset + size + self.sector - 1) // self.sector * self.sector
        n = end - start
        if n > self.buf_size:
            raise ValueError("read too large")
        if not self.k32.SetFilePointerEx(self.h, ctypes.c_longlong(start), None, 0):
            raise OSError("SetFilePointerEx failed")
        read = wintypes.DWORD(0)
        t0 = time.perf_counter()
        ok = self.k32.ReadFile(self.h, self.buf, n, ctypes.byref(read), None)
        dt = time.perf_counter() - t0
        if not ok or read.value != n:
            raise OSError("ReadFile failed: %d/%d" % (read.value, n))
        return dt

    def close(self):
        self.k32.CloseHandle(self.h)


def stat_line(name, secs, nbytes):
    if not secs:
        print("%-28s no data" % name)
        return
    med = statistics.median(secs)
    gbs = nbytes / med / 1e9
    print("%-28s n=%3d  median %8.3f ms  min %8.3f  max %8.3f  %6.2f GB/s"
          % (name, len(secs), med * 1e3, min(secs) * 1e3, max(secs) * 1e3, gbs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seq-mib", type=int, default=1024)
    ap.add_argument("--random", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    data_start, alignment, tensors, version = parse_gguf(args.model)
    print("gguf v%d, alignment %d, %d tensors, data at %d" % (version, alignment, len(tensors), data_start))

    experts = []
    for t in tensors:
        if "_exps" not in t["name"]:
            continue
        per_expert, n_expert = tensor_expert_bytes(t)
        if per_expert is None:
            continue
        experts.append({
            "name": t["name"],
            "abs": data_start + t["offset"],
            "per_expert": per_expert,
            "n_expert": n_expert,
            "nbytes": tensor_nbytes(t),
        })

    if not experts:
        print("no expert tensors found")
        return 1

    total = sum(e["nbytes"] for e in experts)
    by_layer = {}
    for e in experts:
        layer = e["name"].split("blk.")[1].split(".")[0]
        by_layer.setdefault(layer, []).append(e)
    trips = [v for v in by_layer.values() if len(v) >= 3]
    per_trip = [sum(t["per_expert"] for t in v) for v in trips]
    print("%d expert tensors, %.2f GiB, median %.2f MiB per expert triple" % (
        len(experts), total / 2**30, statistics.median(per_trip) / 2**20))

    f = RawFile(args.model)
    try:
        # 1. sequential, 8 MiB chunks from the start of the expert region
        chunk = 8 * 2**20
        n_chunks = max(1, args.seq_mib * 2**20 // chunk)
        base = experts[0]["abs"]
        secs = []
        for i in range(n_chunks):
            secs.append(f.read(base + i * chunk, chunk))
        stat_line("sequential 8 MiB", secs, chunk)

        # 2. one expert triple (gate, up, down) at a random index, queue depth 1
        layers = trips
        rnd = random.Random(1234)
        triples = []
        for _ in range(args.random):
            trip = rnd.choice(layers)
            idx = rnd.randrange(min(e["n_expert"] for e in trip))
            triples.append([(e["abs"] + idx * e["per_expert"], e["per_expert"]) for e in trip])

        secs = []
        for trip in triples:
            t0 = time.perf_counter()
            for off, size in trip:
                f.read(off, size)
            secs.append(time.perf_counter() - t0)
        nbytes = sum(size for _, size in triples[0])
        stat_line("random expert triple qd1", secs, nbytes)

        # 3. the same, queue depth = --threads
        def worker(sub, out, k):
            fh = RawFile(args.model)
            try:
                s = []
                for trip in sub:
                    t0 = time.perf_counter()
                    for off, size in trip:
                        fh.read(off, size)
                    s.append(time.perf_counter() - t0)
                out[k] = s
            finally:
                fh.close()

        per = max(1, len(triples) // args.threads)
        parts = [triples[i * per:(i + 1) * per] for i in range(args.threads)]
        out = [[] for _ in parts]
        ths = [threading.Thread(target=worker, args=(p, out, i)) for i, p in enumerate(parts)]
        t0 = time.perf_counter()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        wall = time.perf_counter() - t0
        done = sum(len(s) for s in out)
        stats = [s for part in out for s in part]
        print("%-28s n=%3d  wall %8.3f ms  %6.2f triple/s  %6.2f GB/s" % (
            "random expert triple qd%d" % args.threads, done, wall * 1e3, done / wall, done * nbytes / wall / 1e9))
        if stats:
            print("%-28s median %8.3f ms per triple (single thread)" % ("", statistics.median(stats) * 1e3))
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
