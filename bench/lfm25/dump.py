import parse

rows = [r for r in parse.rows
        if 'vram' not in r['tag'] and 'sanity' not in r['tag'] and 'env' not in r['tag']]
rows.sort(key=lambda r: (r['tag'], r['test']))
for r in rows:
    print(f"{r['tag']:32s} {r['test']:8s} {r['ts']:8.2f} sd={r['sd']:5.2f} ub={r['ub']:>4s} t={r['thr']:>2s} fa={r['fa']} ncmoe={r['ncmoe']:>2s} lm={r['lm']}")
