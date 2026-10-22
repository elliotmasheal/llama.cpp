import re
lines = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
# find first "Total time: 95.0" occurrence and print preceding 25 lines
for i, l in enumerate(lines):
    if l.startswith("Total time: 95.0"):
        for x in lines[max(0,i-30):i+1]:
            print(x)
        break
