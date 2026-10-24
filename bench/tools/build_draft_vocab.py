#!/usr/bin/env python3
"""M1-3a: build the restricted draft vocabulary (<= 16384 token ids) from the validation streams.

Steps (all reproducible, CPU only, model is mmap-ed):
  1. compile bench/tools/vocab_dump.cpp against the static llama/ggml archives
  2. tokenize the four 10K prompts (prompt stream)
  3. greedy-generate N tokens per prompt on CPU (generated stream)
  4. read tokenizer token strings + token types from the GGUF metadata (gguf-py reader)
  5. keep special + observed tokens, fill the rest, write the vocab and coverage

Outputs:
  bench/vocab/draft-16k.txt    one int32 token id per line, ascending
  bench/vocab/draft-16k.bin    16384 little-endian int32 ids, same ascending order
  bench/vocab/coverage.json    counts, coverage fractions, held-out folds and file hashes

The delivered vocab contains every observed token, so its own prompt/generated coverage is
100% by construction; coverage.json also reports leave-one-prompt-out folds (select from three
prompts, score the fourth generated stream) as the real unseen-text estimate.

Note: this GGUF has no tokenizer.ggml.scores (Qwen BPE stores merge order, not unigram
scores), so the score fill uses token id ascending as the merge-priority proxy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(r"F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
PROMPTS = [
    "prompt10k.txt",
    "prompt-chat10k.txt",
    "prompt-code10k.txt",
    "prompt-multi10k.txt",
]
SPECIAL_TYPES = (3, 4)  # control, user_defined
UNUSED_TYPE = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def find_gxx(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("g++")
    if found:
        return found
    for cand in (r"D:\Scoop\apps\mingw\current\bin\g++.exe", r"C:\msys64\mingw64\bin\g++.exe"):
        if Path(cand).exists():
            return cand
    raise SystemExit("error: g++ not found, pass --gxx")


def find_vulkan_lib(explicit: str | None) -> Path:
    if explicit:
        lib = Path(explicit) / "Lib" / "vulkan-1.lib"
        if not lib.exists():
            raise SystemExit(f"error: {lib} not found")
        return lib.parent
    env = os.environ.get("VULKAN_SDK")
    cands: list[Path] = []
    if env:
        cands.append(Path(env))
    for root in (Path("C:/VulkanSDK"), Path("D:/VulkanSDK"), Path("D:/Scoop/apps/vulkan")):
        if root.exists():
            cands.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    for cand in reversed(cands):
        if (cand / "Lib" / "vulkan-1.lib").exists():
            return cand / "Lib"
    raise SystemExit("error: vulkan-1.lib not found, pass --vulkan-sdk")


def build_dump(gxx: str, build_dir: Path, vulkan_lib: Path, out_exe: Path, rebuild: bool) -> None:
    src = REPO / "bench" / "tools" / "vocab_dump.cpp"
    src_mtime = src.stat().st_mtime
    if out_exe.exists() and not rebuild and out_exe.stat().st_mtime >= src_mtime:
        log(f"[1/5] vocab_dump already built: {out_exe}")
        return
    libs = [
        build_dir / "src" / "libllama.a",
        build_dir / "ggml" / "src" / "ggml.a",
        build_dir / "ggml" / "src" / "ggml-cpu.a",
        build_dir / "ggml" / "src" / "ggml-base.a",
        build_dir / "ggml" / "src" / "ggml-vulkan" / "ggml-vulkan.a",
    ]
    missing = [str(p) for p in libs if not p.exists()]
    if missing:
        raise SystemExit("error: static libs missing:\n  " + "\n  ".join(missing))
    cmd = [
        gxx, "-O2", "-std=c++17", "-fopenmp", str(src),
        "-I", str(REPO / "include"), "-I", str(REPO / "ggml" / "include"),
        "-Wl,--start-group", *map(str, libs), "-Wl,--end-group",
        "-L", str(vulkan_lib), "-lvulkan-1", "-o", str(out_exe),
    ]
    log("[1/5] build: " + " ".join(cmd))
    out_exe.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def run(cmd: list[str], stderr_path: Path) -> str:
    t0 = time.time()
    with stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"error: {' '.join(cmd)} failed ({proc.returncode}), see {stderr_path}")
    log(f"      {Path(stderr_path).name}: {time.time() - t0:.1f}s")
    return proc.stdout


def read_ids(text: str) -> list[int]:
    return [int(line) for line in text.splitlines() if line.strip()]


def load_gguf_reader(model: Path):
    """Import gguf-py's reader module only: the package __init__ needs PyYAML, which may be absent."""
    pkg_dir = REPO / "gguf-py" / "gguf"
    if not pkg_dir.exists():
        raise SystemExit(f"error: {pkg_dir} missing (gguf-py is part of this repo)")
    if "gguf" not in sys.modules:
        pkg = types.ModuleType("gguf")
        pkg.__path__ = [str(pkg_dir)]
        sys.modules["gguf"] = pkg
    spec = importlib.util.spec_from_file_location("gguf.gguf_reader", pkg_dir / "gguf_reader.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf.gguf_reader"] = mod
    spec.loader.exec_module(mod)
    return mod.GGUFReader


def read_vocab_meta(model: Path) -> tuple[list[str], list[int], dict]:
    GGUFReader = load_gguf_reader(model)
    reader = GGUFReader(model)
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    types = [int(t) for t in reader.fields["tokenizer.ggml.token_type"].contents()]
    meta = {
        "n_vocab": len(tokens),
        "model": reader.fields["tokenizer.ggml.model"].contents(),
        "pre": reader.fields["tokenizer.ggml.pre"].contents() if "tokenizer.ggml.pre" in reader.fields else None,
        "has_scores": "tokenizer.ggml.scores" in reader.fields,
        "eos_token_id": int(reader.fields["tokenizer.ggml.eos_token_id"].contents()),
        "bos_token_id": int(reader.fields["tokenizer.ggml.bos_token_id"].contents()),
        "add_bos_token": bool(reader.fields["tokenizer.ggml.add_bos_token"].contents()),
        "type_counts": {str(k): v for k, v in sorted(collections.Counter(types).items())},
    }
    return tokens, types, meta


def coverage(ids: list[int], keep: set[int]) -> dict:
    total = len(ids)
    hit = sum(1 for t in ids if t in keep)
    return {"tokens": total, "covered": hit, "fraction": (hit / total) if total else 0.0}


def select_set(observed: set[int], counts: collections.Counter, types: list[int], target: int) -> tuple[set[int], bool]:
    """Specials always win, then observed tokens (frequency), then id-ascending normal tokens."""
    specials = {i for i, t in enumerate(types) if t in SPECIAL_TYPES}
    unused = {i for i, t in enumerate(types) if t == UNUSED_TYPE}
    normal = [i for i, t in enumerate(types) if t == 1]
    truncated = False
    keep = specials | (observed - unused)
    if len(keep) > target:
        ranked = sorted(observed - unused, key=lambda i: (-counts[i], i))
        keep = specials | set(ranked[: max(0, target - len(specials))])
        truncated = True
    if len(keep) < target:
        # no scores in this GGUF: ascending id == earlier BPE merge == more frequent subword
        for tok in normal:
            if tok not in keep:
                keep.add(tok)
                if len(keep) == target:
                    break
    return keep, truncated


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="build the M1-3a restricted draft vocabulary")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--build-dir", type=Path, default=REPO / "build-vulkan",
                    help="directory holding the static libllama/ggml archives")
    ap.add_argument("--vulkan-sdk", default=None, help="Vulkan SDK root or its Lib directory")
    ap.add_argument("--gxx", default=None)
    ap.add_argument("--prompts", type=Path, default=REPO / "bench" / "prompts")
    ap.add_argument("--out-dir", type=Path, default=REPO / "bench" / "vocab")
    ap.add_argument("--target", type=int, default=16384)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-gen", action="store_true", help="prompt stream only (generated coverage not measured)")
    ap.add_argument("--reuse-raw", action="store_true", help="reuse token/gen id files under OUT/raw instead of running the model")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"error: model not found: {args.model}")
    prompts = [args.prompts / name for name in PROMPTS]
    for p in prompts:
        if not p.exists():
            raise SystemExit(f"error: prompt missing: {p}")
    if args.target > 16384:
        raise SystemExit("error: design caps QRT_DRAFT_VOCAB at 16384")

    out_dir: Path = args.out_dir
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "bin" / "vocab_dump.exe"

    gxx = find_gxx(args.gxx)
    vulkan_lib = find_vulkan_lib(args.vulkan_sdk)
    build_dump(gxx, args.build_dir, vulkan_lib, exe, args.rebuild)

    # 2. prompt stream: one load per prompt, tokenizer only
    prompt_streams: dict[str, list[int]] = {}
    reuse_tokens = args.reuse_raw and all((raw_dir / f"{p.stem}.tokens").exists() for p in prompts)
    for p in prompts:
        if reuse_tokens:
            prompt_streams[p.name] = read_ids((raw_dir / f"{p.stem}.tokens").read_text(encoding="ascii"))
            continue
        log(f"[2/5] tokenize {p.name}")
        out = run([str(exe), str(args.model), "tokens", str(p)], raw_dir / f"{p.stem}.tokens.err")
        prompt_streams[p.name] = read_ids(out)
        (raw_dir / f"{p.stem}.tokens").write_text(out, encoding="ascii")
    if reuse_tokens:
        log("[2/5] reused raw prompt token ids (--reuse-raw)")

    # 3. generated stream: greedy decode on CPU (n_gpu_layers=0, mmap load)
    gen_streams: dict[str, list[int]] = {}
    gen_notes: dict[str, dict] = {}
    reuse_gen = (not args.skip_gen and args.reuse_raw
                 and all((raw_dir / f"{p.stem}.gen").exists() for p in prompts))
    if args.skip_gen:
        log("[3/5] skipped (--skip-gen): generated coverage NOT measured")
    elif reuse_gen:
        for p in prompts:
            gen_streams[p.name] = read_ids((raw_dir / f"{p.stem}.gen").read_text(encoding="ascii"))
            note = {"generated": len(gen_streams[p.name])}
            err = raw_dir / f"{p.stem}.gen.err"
            if err.exists():
                for line in err.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("# generated="):
                        for kv in line[2:].split()[1:]:
                            k, v = kv.split("=", 1)
                            note[k] = int(v)
            gen_notes[p.name] = note
        log("[3/5] reused raw generated ids (--reuse-raw)")
    else:
        for p in prompts:
            log(f"[3/5] greedy {args.gen_tokens} tokens: {p.name}")
            out = run([str(exe), str(args.model), "gen", str(p), str(args.gen_tokens),
                       "--threads", str(args.threads), "--add-special", "0"],
                      raw_dir / f"{p.stem}.gen.err")
            gen_streams[p.name] = read_ids(out)
            (raw_dir / f"{p.stem}.gen").write_text(out, encoding="ascii")
            note = {"generated": len(gen_streams[p.name])}
            for line in (raw_dir / f"{p.stem}.gen.err").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("# generated="):
                    parts = line[2:].split()
                    for kv in parts[1:]:
                        k, v = kv.split("=", 1)
                        note[k] = int(v)
            gen_notes[p.name] = note

    # 4. tokenizer metadata (strings and types; this GGUF has no scores)
    log("[4/5] read tokenizer metadata with gguf-py")
    tokens, types, meta = read_vocab_meta(args.model)

    # 5. selection
    log("[5/5] select the restricted vocabulary")
    prompt_counts: collections.Counter[int] = collections.Counter()
    for ids in prompt_streams.values():
        prompt_counts.update(ids)
    gen_counts: collections.Counter[int] = collections.Counter()
    for ids in gen_streams.values():
        gen_counts.update(ids)
    total_counts = prompt_counts + gen_counts
    observed = set(total_counts)

    specials = {i for i, t in enumerate(types) if t in SPECIAL_TYPES}
    keep, truncated_observed = select_set(observed, total_counts, types, args.target)
    assert len(keep) <= args.target, len(keep)
    assert len(keep) == len(set(keep))

    ids_sorted = sorted(keep)
    txt_path = out_dir / "draft-16k.txt"
    bin_path = out_dir / "draft-16k.bin"
    txt_path.write_text("".join(f"{t}\n" for t in ids_sorted), encoding="ascii")
    bin_path.write_bytes(struct.pack(f"<{len(ids_sorted)}i", *ids_sorted))

    special_kept = len(keep & specials)
    observed_kept = len((keep & observed) - specials)
    filled = len(keep) - special_kept - observed_kept

    prompt_cov = {p.name: coverage(prompt_streams[p.name], keep) for p in prompts}
    prompt_total = {
        "tokens": sum(c["tokens"] for c in prompt_cov.values()),
        "covered": sum(c["covered"] for c in prompt_cov.values()),
    }
    prompt_total["fraction"] = prompt_total["covered"] / prompt_total["tokens"] if prompt_total["tokens"] else 0.0
    if gen_streams:
        gen_cov = {p.name: coverage(gen_streams[p.name], keep) for p in prompts}
        gen_total = {
            "tokens": sum(c["tokens"] for c in gen_cov.values()),
            "covered": sum(c["covered"] for c in gen_cov.values()),
        }
        gen_total["fraction"] = gen_total["covered"] / gen_total["tokens"] if gen_total["tokens"] else 0.0
    else:
        gen_cov, gen_total = {}, None

    # Generalisation checks. The delivered vocab contains every observed token, so its own
    # prompt/gen coverage is 100% by construction; these folds estimate unseen-text misses.
    heldout = None
    prompt_only = None
    if gen_streams:
        folds = {}
        for h in [p.name for p in prompts]:
            train: set[int] = set()
            train_counts: collections.Counter[int] = collections.Counter()
            for name in prompt_streams:
                if name == h:
                    continue
                train.update(prompt_streams[name])
                train.update(gen_streams[name])
                train_counts.update(prompt_streams[name])
                train_counts.update(gen_streams[name])
            fold_keep, _ = select_set(train, train_counts, types, args.target)
            folds[h] = coverage(gen_streams[h], fold_keep)
            folds[h]["unseen_in_train"] = folds[h]["tokens"] - folds[h]["covered"]
        heldout = {
            "mode": "leave-one-prompt-out: select from the other three prompts (prompt+gen), score the held-out gen stream",
            "folds": folds,
            "mean_fraction": sum(f["fraction"] for f in folds.values()) / len(folds),
            "tokens": sum(f["tokens"] for f in folds.values()),
            "covered": sum(f["covered"] for f in folds.values()),
        }
        heldout["fraction"] = heldout["covered"] / heldout["tokens"] if heldout["tokens"] else 0.0

        prompt_obs: set[int] = set()
        prompt_counts_all: collections.Counter[int] = collections.Counter()
        for ids in prompt_streams.values():
            prompt_obs.update(ids)
            prompt_counts_all.update(ids)
        prompt_keep, _ = select_set(prompt_obs, prompt_counts_all, types, args.target)
        prompt_only = {
            "mode": "selection from the prompt streams only, scored on all generated tokens",
            "vocab_distinct_observed": len(prompt_obs),
            **coverage([t for ids in gen_streams.values() for t in ids], prompt_keep),
        }

    report = {
        "task": "M1-3a",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(args.model),
        "target": args.target,
        "vocab_size": len(ids_sorted),
        "selection": {
            "special": special_kept,
            "observed": observed_kept,
            "score_filled": filled,
            "observed_distinct_total": len(observed),
            "truncated_observed": truncated_observed,
            "fill_order": "token id ascending (BPE merge-priority proxy; no tokenizer.ggml.scores in this GGUF)",
        },
        "tokenizer": meta,
        "gen_tokens_requested": 0 if args.skip_gen else args.gen_tokens,
        "gen_notes": gen_notes,
        "prompt_coverage": prompt_total,
        "prompt_coverage_per_prompt": prompt_cov,
        "gen_coverage": gen_total,
        "gen_coverage_per_prompt": gen_cov,
        "heldout_gen_coverage": heldout,
        "prompt_only_selection_gen_coverage": prompt_only,
        "files": {
            "draft-16k.txt": {"sha256": sha256(txt_path)},
            "draft-16k.bin": {"sha256": sha256(bin_path)},
        },
    }
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    log("")
    log(f"vocab size      : {len(ids_sorted)} / {args.target}")
    log(f"  special       : {special_kept}")
    log(f"  observed      : {observed_kept} (distinct observed overall: {len(observed)})")
    log(f"  score filled  : {filled}")
    log(f"prompt coverage : {prompt_total['covered']}/{prompt_total['tokens']} = {prompt_total['fraction']*100:.2f}%")
    if gen_total:
        log(f"gen coverage    : {gen_total['covered']}/{gen_total['tokens']} = {gen_total['fraction']*100:.2f}%")
        for name, note in gen_notes.items():
            log(f"  {name}: generated={note.get('generated')} eog={note.get('eog')}")
    else:
        log("gen coverage    : not measured (--skip-gen)")
    for name, c in prompt_cov.items():
        log(f"  {name}: {c['covered']}/{c['tokens']} = {c['fraction']*100:.2f}%")
    if heldout:
        log(f"held-out gen    : {heldout['covered']}/{heldout['tokens']} = {heldout['fraction']*100:.2f}% "
            f"(leave-one-prompt-out selection)")
        for name, c in heldout["folds"].items():
            log(f"  fold {name}: {c['covered']}/{c['tokens']} = {c['fraction']*100:.2f}%")
    if prompt_only:
        log(f"prompt-only sel : {prompt_only['covered']}/{prompt_only['tokens']} = "
            f"{prompt_only['fraction']*100:.2f}% of generated tokens")
    log(f"wrote {txt_path}")
    log(f"wrote {bin_path}")
    log(f"wrote {out_dir / 'coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
