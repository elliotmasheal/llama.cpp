import csv, glob, os, sys

rows = []
for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "*.csv"))):
    tag = os.path.splitext(os.path.basename(path))[0]
    with open(path, newline="") as f:
        try:
            data = list(csv.DictReader(f))
        except Exception:
            continue
    for r in data:
        try:
            pp = int(r.get("n_prompt") or 0)
            gen = int(r.get("n_gen") or 0)
        except ValueError:
            continue
        test = f"pp{pp}" if gen == 0 else f"tg{gen}"
        rows.append({
            "tag": tag,
            "test": test,
            "pp": pp,
            "gen": gen,
            "ts": float(r.get("avg_ts") or 0),
            "sd": float(r.get("stddev_ts") or 0),
            "ngl": r.get("n_gpu_layers"),
            "ncmoe": r.get("n_cpu_moe"),
            "ub": r.get("n_ubatch"),
            "batch": r.get("n_batch"),
            "thr": r.get("n_threads"),
            "fa": r.get("flash_attn"),
            "lm": r.get("load_mode"),
            "tk": r.get("type_k"),
        })

def show(tagfilter):
    sel = [r for r in rows if tagfilter(r)]
    if not sel:
        print("(no rows)")
        return
    print(f"{'tag':30s} {'test':7s} {'ts':>8s} {'sd':>6s} {'ub':>5s} {'b':>5s} {'thr':>4s} {'fa':>3s} {'lm':>5s} {'kv':>5s}")
    for r in sel:
        print(f"{r['tag']:30s} {r['test']:7s} {r['ts']:8.2f} {r['sd']:6.2f} {r['ub']:>5s} {r['batch']:>5s} {r['thr']:>4s} {r['fa']:>3s} {r['lm']:>5s} {r['tk']:>5s}")

if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    show(lambda r: filt in r["tag"])
