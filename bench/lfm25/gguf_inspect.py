# Read-only GGUF header/tensor-info inspector. Reads metadata + tensor infos only,
# never the tensor data. Used to size the MoE cache budget for LFM2.5-24B-A2B.
import struct
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else r"F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"

# ggml type -> (block_elems, block_bytes); only the types this file plausibly uses
TYPE = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}


def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))


def rd_str(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    if t in (0, 1, 7, 4, 5, 6):  # u8 i8 bool u32 i32 f32 (f32=6? handled below)
        f.read({0: 1, 1: 1, 7: 1, 4: 4, 5: 4, 6: 4}[t])
    elif t in (2, 3):  # u16 i16
        f.read(2)
    elif t == 8:  # string
        rd_str(f)
    elif t == 9:  # array
        at, = rd(f, "<I")
        (n,) = rd(f, "<Q")
        for _ in range(n):
            skip_val(f, at)
    elif t in (10, 11, 12):  # u64 i64 f64
        f.read(8)
    else:
        raise ValueError("bad kv type %d" % t)


def rd_val(f, t):
    if t == 8:
        return rd_str(f)
    if t in (0, 1, 7):
        return rd(f, "<B")[0]
    if t in (2, 3):
        return rd(f, "<H")[0]
    if t in (4, 5):
        return rd(f, "<I")[0] if t == 4 else rd(f, "<i")[0]
    if t == 6:
        return rd(f, "<f")[0]
    if t in (10, 11):
        return rd(f, "<Q")[0] if t == 10 else rd(f, "<q")[0]
    if t == 12:
        return rd(f, "<d")[0]
    if t == 9:
        at, = rd(f, "<I")
        (n,) = rd(f, "<Q")
        vals = [rd_val(f, at) for _ in range(n)]
        return vals[:16] + (["...%d" % n] if n > 16 else [])
    raise ValueError("bad kv type %d" % t)


with open(PATH, "rb") as f:
    magic = f.read(4)
    assert magic == b"GGUF", magic
    version, = rd(f, "<I")
    n_tensors, = rd(f, "<Q")
    n_kv, = rd(f, "<Q")
    print("gguf version %d, tensors %d, kv %d" % (version, n_tensors, n_kv))
    kv = {}
    for _ in range(n_kv):
        k = rd_str(f)
        t, = rd(f, "<I")
        kv[k] = rd_val(f, t)
    for k in sorted(kv):
        if k.startswith("tokenizer.") and isinstance(kv[k], list):
            continue
        v = kv[k]
        if isinstance(v, list) and len(v) > 16:
            v = "%s ... (%d)" % (v[:8], len(v))
        print("KV %-45s = %s" % (k, v))

    tensors = []
    for _ in range(n_tensors):
        name = rd_str(f)
        nd, = rd(f, "<I")
        dims = rd(f, "<%dQ" % nd)
        tt, = rd(f, "<I")
        off, = rd(f, "<Q")
        tname, be, bb = TYPE.get(tt, ("TYPE%d" % tt, 1, 0))
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = nelem // be * bb
        tensors.append((name, dims, tname, nelem, nbytes))

# aggregate
tot = sum(t[4] for t in tensors)
by_type = defaultdict(int)
by_class = defaultdict(int)
by_layer_exps = defaultdict(int)
print("\nfile headers end; total tensor bytes = %.2f GiB" % (tot / 2**30))
print("tensor data region starts at offset 0")
for name, dims, tname, nelem, nbytes in tensors:
    by_type[tname] += nbytes
    if "_exps" in name:
        by_class["ffn_*_exps"] += nbytes
        parts = name.split(".")
        blk = parts[1]
        by_layer_exps[blk] += nbytes
    elif "token_embd" in name:
        by_class["token_embd"] += nbytes
    elif name.startswith("output"):
        by_class["output"] += nbytes
    elif "shortconv" in name:
        by_class["shortconv"] += nbytes
    elif "attn" in name:
        by_class["attn"] += nbytes
    else:
        by_class[name] += nbytes

print("\n--- bytes by type ---")
for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
    print("%-8s %10.2f MiB" % (k, v / 2**20))
print("\n--- bytes by class ---")
for k, v in sorted(by_class.items(), key=lambda x: -x[1]):
    print("%-24s %10.2f MiB" % (k, v / 2**20))
print("\n--- expert bytes per blk (MiB) ---")
for k in sorted(by_layer_exps, key=lambda s: int(s)):
    print("blk.%-3s %8.2f MiB" % (k, by_layer_exps[k] / 2**20))

print("\n--- per-tensor list ---")
for name, dims, tname, nelem, nbytes in tensors:
    print("%-52s %-6s %8.3f MiB  ne=%s" % (name, tname, nbytes / 2**20, list(dims)))
