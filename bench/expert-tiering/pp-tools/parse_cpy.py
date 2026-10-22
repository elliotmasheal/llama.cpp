import re
rows = []
for line in open(r"bench\expert-tiering\pp-cpy.err", encoding="utf-8", errors="ignore"):
    m = re.match(r"cpy: split=(\d+) .*?bytes=(\d+) us=(\d+) sync_us=(\d+) .*?name=moe_cache\.slot_map-(\d+)", line)
    if m:
        rows.append((int(m.group(1)), int(m.group(5)), int(m.group(3)), int(m.group(4))))
print("rows", len(rows))
for split, layer, us, sync in rows:
    print(f"split {split:2d} layer {layer:2d} copy {us/1000:7.1f} ms sync {sync/1000:7.1f} ms")
