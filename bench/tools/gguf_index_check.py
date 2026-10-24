"""Cross-checks the runtime GGUF index against gguf-py (the reader gguf_dims.py uses).

Usage: python bench/tools/gguf_index_check.py [qrt.exe] [model.gguf]

Compares the tensor table row by row, the metadata the runtime reads, and the byte
account. Then feeds crafted files to qrt and requires exit code 2 with a message,
never a crash. Prints PASS/FAIL per group; exit code 1 when anything mismatches.
"""
import os
import struct
import subprocess
import sys
import tempfile
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
# gguf-py imports yaml for model-card frontmatter, which this tool never reads.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
from gguf import GGUFReader  # noqa: E402

DEFAULT_EXE = os.path.join(ROOT, "build-vulkan", "bin", "qrt.exe")
DEFAULT_MODEL = r"F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

FAILED = []
TOTAL = 0


def check(ok, what):
    global TOTAL
    TOTAL += 1
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAILED.append(what)


def run(exe, args):
    p = subprocess.run([exe] + args, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_report(text):
    header = {}
    meta = {}
    account = {}
    rows = []
    for line in text.splitlines():
        if line.startswith("# gguf "):
            for tok in line[len("# gguf "):].split():
                k, _, v = tok.partition("=")
                header[k] = v
        elif line.startswith("# meta "):
            for tok in line[len("# meta "):].split():
                k, _, v = tok.partition("=")
                meta[k] = v
        elif line.startswith("# account "):
            for tok in line[len("# account "):].split():
                k, _, v = tok.partition("=")
                account[k] = v
        elif line and not line.startswith("#") and not line.startswith("name,"):
            f = line.split(",")
            rows.append({
                "name": f[0],
                "n_dims": int(f[1]),
                "ne": [int(x) for x in f[2:6]],
                "type": int(f[6]),
                "offset": int(f[8]),
                "nbytes": int(f[9]),
            })
    return header, meta, account, rows


def field(r, key, default=None):
    f = r.fields.get(key)
    return default if f is None else f.contents()


def gguf_file(tensors, kv=b"", n_kv=0, version=3, magic=b"GGUF", align=32, data=b""):
    out = bytearray()
    out += magic
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", n_kv)
    out += kv
    for name, shape, type_id, offset in tensors:
        nb = name.encode()
        out += struct.pack("<Q", len(nb)) + nb
        out += struct.pack("<I", len(shape))
        for d in shape:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", type_id)
        out += struct.pack("<Q", offset)
    out += b"\x00" * ((-len(out)) % align)
    out += data
    return bytes(out)


def crafted_cases():
    # (label, bytes) - every one of these must come back as exit 2 with a message.
    cases = [
        ("bad magic", gguf_file([("t", [8], 0, 0)], magic=b"GGUX", data=b"\x00" * 32)),
        ("truncated tensor table", gguf_file([("t", [8], 0, 0)])[:24]),
        ("tensor past the data section", gguf_file([("t", [8], 0, 0)])),
        ("unknown tensor type", gguf_file([("t", [1], 4, 0)], data=b"\x00" * 4)),
        ("duplicate tensor names", gguf_file([("t", [8], 0, 0), ("t", [8], 0, 32)],
                                            data=b"\x00" * 64)),
        ("dim overflow", gguf_file([("t", [1 << 62, 4], 0, 0)], data=b"\x00" * 4)),
        ("nine dims", gguf_file([("t", [1] * 9, 0, 0)], data=b"\x00" * 4)),
    ]
    # A string length above the 1 MiB limit must be refused before it is allocated.
    big = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 1))
    big += struct.pack("<Q", 2 << 20)
    cases.append(("oversized metadata string", bytes(big)))
    # Array of arrays is not GGUF; the reader must reject it instead of recursing.
    kv = bytearray()
    key = b"x"
    kv += struct.pack("<Q", len(key)) + key + struct.pack("<I", 9)
    kv += struct.pack("<I", 9) + struct.pack("<Q", 1)
    cases.append(("nested metadata array",
                  gguf_file([("t", [8], 0, 0)], kv=bytes(kv), n_kv=1, data=b"\x00" * 32)))
    return cases


def main(argv):
    exe = argv[1] if len(argv) > 1 else DEFAULT_EXE
    model = argv[2] if len(argv) > 2 else DEFAULT_MODEL
    if not os.path.isfile(exe):
        print("FAIL qrt executable not found: " + exe)
        return 1
    if not os.path.isfile(model):
        print("FAIL model not found: " + model)
        return 1

    rc, out, err = run(exe, ["--gguf-index", model])
    check(rc == 0, "qrt --gguf-index exits 0 (rc=%d %s)" % (rc, err.strip()[:200]))
    header, meta, account, rows = parse_report(out)

    r = GGUFReader(model, "r")
    arch = field(r, "general.architecture")
    with open(model, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
    check(magic == b"GGUF" and header.get("version") == str(version), "GGUF version matches the file")
    check(header.get("tensors") == str(len(r.tensors)), "tensor count matches gguf-py")
    check(header.get("alignment") == str(r.alignment), "alignment matches gguf-py")
    check(header.get("data_offset") == str(r.data_offset), "data section offset matches gguf-py")
    check(header.get("file_size") == str(os.path.getsize(model)), "file size matches the file")
    check(meta.get("arch") == arch, "architecture matches gguf-py")

    hp_keys = [
        ("block_count", "block_count"),
        ("nextn", "nextn_predict_layers"),
        ("expert_count", "expert_count"),
        ("expert_used_count", "expert_used_count"),
        ("n_embd", "embedding_length"),
        ("expert_ff", "expert_feed_forward_length"),
        ("head_count_kv", "attention.head_count_kv"),
        ("key_length", "attention.key_length"),
        ("value_length", "attention.value_length"),
        ("context_length", "context_length"),
        ("full_attention_interval", "full_attention_interval"),
    ]
    hp_bad = []
    for got, key in hp_keys:
        want = field(r, arch + "." + key)
        if str(want) != meta.get(got):
            hp_bad.append("%s: got %s want %s" % (got, meta.get(got), want))
    check(not hp_bad, "the 11 hparams match gguf-py" + ("" if not hp_bad else ": " + "; ".join(hp_bad)))

    # Tensor table: name, dims, type id, absolute offset, byte size, in file order.
    want_rows = []
    for t in r.tensors:
        shape = [int(x) for x in t.shape] + [1] * (4 - len(t.shape))
        want_rows.append({
            "name": t.name,
            "n_dims": len(t.shape),
            "ne": shape,
            "type": int(t.tensor_type),
            "offset": int(t.data_offset),
            "nbytes": int(t.n_bytes),
        })
    check(len(rows) == len(want_rows), "the table has %d rows" % len(want_rows))
    diffs = []
    for got, want in zip(rows, want_rows):
        if got != want:
            diffs.append("%s: got %r want %r" % (want["name"], got, want))
    check(not diffs, "every tensor row matches gguf-py" +
          ("" if not diffs else " (%d differ, first: %s)" % (len(diffs), diffs[0])))

    # The account gguf_dims.py prints.
    exp_bytes = 0
    exp_layers = set()
    dense_bytes = 0
    emb_bytes = 0
    for t in r.tensors:
        nb = int(t.n_bytes)
        parts = t.name.split(".")
        if "_exps" in t.name:
            exp_bytes += nb
            exp_layers.add(parts[1])
        elif t.name == "token_embd.weight":
            emb_bytes += nb
        else:
            dense_bytes += nb
    n_moe = len(exp_layers)
    per_expert = exp_bytes / (n_moe * field(r, arch + ".expert_count"))
    check(int(account.get("expert_bytes", -1)) == exp_bytes, "expert bytes match gguf_dims.py")
    check(int(account.get("dense_bytes", -1)) == dense_bytes, "dense bytes match gguf_dims.py")
    check(int(account.get("token_embd_bytes", -1)) == emb_bytes, "token_embd bytes match gguf_dims.py")
    check(int(account.get("expert_layers", -1)) == n_moe, "expert layer count matches gguf_dims.py")
    check(abs(float(account.get("per_expert_bytes", -1)) - per_expert) < 0.1,
          "per-expert bytes match gguf_dims.py")
    full_layers = sorted(int(t.name.split(".")[1]) for t in r.tensors
                         if t.name.startswith("blk.") and len(t.name.split(".")) >= 3
                         and t.name.split(".")[2] == "attn_k")
    n_main = field(r, arch + ".block_count") - field(r, arch + ".nextn_predict_layers", 0)
    check(int(account.get("full_attn_main", -1)) == len([l for l in full_layers if l < n_main]),
          "full-attention main layer count matches gguf_dims.py")
    check(int(account.get("full_attn_nextn", -1)) == len([l for l in full_layers if l >= n_main]),
          "full-attention nextn layer count matches gguf_dims.py")

    # A present tensor resolves, a missing one is an error and not a crash.
    rc, out, err = run(exe, ["--gguf-index", model, "--require-tensor", "blk.0.ffn_gate_exps.weight"])
    check(rc == 0 and "tensor=blk.0.ffn_gate_exps.weight" in out, "a present tensor resolves")
    rc, out, err = run(exe, ["--gguf-index", model, "--require-tensor", "no.such.tensor"])
    check(rc == 2 and "not in the file" in err, "a missing tensor exits 2 with a message")

    with tempfile.TemporaryDirectory() as tmp:
        for label, blob in crafted_cases():
            path = os.path.join(tmp, "case.gguf")
            with open(path, "wb") as f:
                f.write(blob)
            rc, out, err = run(exe, ["--gguf-index", path])
            check(rc == 2 and err.strip() != "", "crafted file refused, not crashed: " + label)
        good = gguf_file([("t", [8], 0, 0)], data=b"\x00" * 32)
        path = os.path.join(tmp, "good.gguf")
        with open(path, "wb") as f:
            f.write(good)
        rc, out, err = run(exe, ["--gguf-index", path])
        check(rc == 0 and len(parse_report(out)[3]) == 1, "a minimal well-formed file is accepted")

    print("\n%d checks, %d failed" % (TOTAL, len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
