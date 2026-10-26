import json
from pathlib import Path
import subprocess
import sys
import time


def add_parser(subparsers):
    parser = subparsers.add_parser("run", help="build/check/run the runtime diagnostic workflow")
    parser.add_argument("--qrt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, help="optionally build an already configured CMake tree")
    parser.add_argument("--baseline", type=Path, help="compare against an existing same-token logits dump")
    parser.add_argument("--align", action="store_true", help="explicitly compare the intersection of dump rows")
    parser.add_argument("--layers", type=int, choices=range(1, 41), default=40, metavar="1..40")
    parser.add_argument("--rows", type=int, choices=range(1, 513), default=16, metavar="1..512")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--kv-k", choices=("f16", "q8_0", "q4_0"), default="q8_0")
    parser.add_argument("--kv-v", choices=("f16", "q8_0", "q4_0"), default="q4_0")
    parser.add_argument("--kv-page-tokens", type=int, default=16384)
    parser.add_argument("--vram-tier-mib", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600, help="per-stage timeout in seconds (1..7200)")
    parser.set_defaults(fn=run)


def run(args):
    if not 1 <= args.timeout <= 7200:
        print("ERROR: --timeout must be 1..7200 seconds", file=sys.stderr)
        return 2
    if not 32 <= args.chunk <= 16384 or args.chunk % 32:
        print("ERROR: --chunk must be a multiple of 32 in 32..16384", file=sys.stderr)
        return 2
    required = [args.model, args.tokens, Path(str(args.store) + ".qwen36rt.idx"),
                Path(str(args.store) + ".qwen36rt.bin")]
    if not args.build_dir:
        required.append(args.qrt)
    else:
        required.append(args.build_dir / "CMakeCache.txt")
    if args.baseline:
        required.append(args.baseline)
    for path in required:
        if not path.is_file():
            print("ERROR: required file does not exist: %s" % path, file=sys.stderr)
            return 2
    output = args.output_dir.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print("ERROR: use a new output directory: %s" % exc, file=sys.stderr)
        return 2
    binary = str(args.qrt.resolve())
    model = str(args.model.resolve())
    store = str(args.store.resolve())
    tokens = str(args.tokens.resolve())
    gate = str(Path(__file__).with_name("qgate.py"))
    config = ["--prefill-chunk", str(args.chunk), "--kv-k", args.kv_k,
              "--kv-v", args.kv_v, "--kv-page-tokens", str(args.kv_page_tokens),
              "--vram-tier-mib", str(args.vram_tier_mib)]
    dump = str(output / "model.bin")
    stages = []
    if args.build_dir:
        build = str(args.build_dir.resolve())
        stages.extend([
            ("build", ["cmake", "--build", build, "--target", "qrt_checks", "-j", "8"]),
            ("ctest", ["ctest", "--test-dir", build, "-L", "qrt", "--output-on-failure"]),
        ])
    stages.extend([
        ("selftest", [binary, "--selftest"]),
        ("config", [binary, "--print-config", *config]),
        ("index", [binary, "--gguf-index", model]),
        ("model", [binary, "--logits-dump", model, "--layer-store", store,
                   "--logits-tokens", tokens, "--logits-layers", str(args.layers),
                   "--logits-rows", str(args.rows), "--logits-out", dump, *config]),
        ("verify", [sys.executable, "-B", gate, "verify", dump]),
    ])
    if args.baseline:
        command = [sys.executable, "-B", gate, "compare", str(args.baseline.resolve()), dump,
                   "--json", str(output / "quality.json")]
        if args.align:
            command.append("--align")
        stages.append(("compare", command))
    report = {
        "kind": "diagnostic_workflow",
        "performance_acceptance": False,
        "model_quality_acceptance": False,
        "status": "running",
        "binary": binary,
        "model": model,
        "store": store,
        "tokens": tokens,
        "layers": args.layers,
        "rows": args.rows,
        "chunk": args.chunk,
        "kv": [args.kv_k, args.kv_v],
        "stages": [],
    }
    manifest = output / "workflow.json"
    for name, command in stages:
        started = time.monotonic()
        stdout = output / (name + ".out")
        stderr = output / (name + ".err")
        error = None
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                result = subprocess.run(command, stdout=out, stderr=err, timeout=args.timeout,
                                        check=False)
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = 2
            error = str(exc)
        stage = {"name": name, "command": command, "exit_code": code,
                 "elapsed_seconds": time.monotonic() - started,
                 "stdout": str(stdout), "stderr": str(stderr)}
        if error:
            stage["error"] = error
        report["stages"].append(stage)
        report["status"] = "failed" if code else "running"
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("%s: exit %d" % (name, code), flush=True)
        if code:
            print("Stopped; evidence: %s" % manifest, file=sys.stderr)
            return 3 if code == 3 else 2
    report["status"] = "comparison_passed" if args.baseline else "format_verified"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("%s: %s (diagnostic only, not a model/performance acceptance)" % (report["status"], manifest))
    return 0
