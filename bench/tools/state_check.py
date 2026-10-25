#!/usr/bin/env python3
"""Validate the task graph, the shared resources and the doc layout.

The script is the machine-checkable half of the agent contract: it keeps
bench/state/state.json, the STATE block in AGENTS.md and the L0/L1/L3 doc
layout in agreement, so "按需读取" cannot silently degrade into "读了旧的".

  python bench/tools/state_check.py                briefing + all checks
  python bench/tools/state_check.py --board        full board, grouped by lane
  python bench/tools/state_check.py --lane impl    one lane
  python bench/tools/state_check.py --brief M2-6   one task: reads, deps, done_when
  python bench/tools/state_check.py --lint         docs only
  python bench/tools/state_check.py --selftest     pure unit tests, no repo reads

Locking has exactly one source now: a task that is `doing` and lists a
resource holds it. The old bench/state/quiet-* files are deprecated; finding
one is a warning, not an error.
"""
import fnmatch
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "bench" / "state" / "state.json"
AGENTS = ROOT / "AGENTS.md"
DOCS = ROOT / "docs" / "agent"
INDEX = DOCS / "INDEX.md"

PHASES = ["M0", "M1", "M2", "M3", "M4", "M5"]
PHASE_STATUS = {"not_started", "in_progress", "blocked", "done"}
TASK_STATUS = {"todo", "doing", "partial", "done", "blocked"}
SCHEMA = 3
REQUIRED_TOP = [
    "schema", "track", "updated", "phase", "phase_status", "decision_gate",
    "resources", "lanes", "queues", "tasks", "last_gate", "blockers",
    "escalations",
]
REQUIRED_TASK = ["id", "milestone", "lane", "status", "title", "deps",
                 "resources", "paths", "done_when"]
CLAIM_TTL_HOURS = 4
DATE_BULLET = re.compile(r"^\s*-\s.*\b20\d\d-\d\d-\d\d\b")
TASK_ROW = re.compile(r"^\|\s*\*{0,2}M\d")
SLUG_DEF = re.compile(r"<!--\s*slug:\s*([A-Za-z0-9_.-]+)\s*-->")
SLUG_REF = re.compile(r"(RESULTS|DECISIONS)\.md#([A-Za-z0-9_.-]+)")
MD_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.md)`")


# ---------------------------------------------------------------- paths

def split_path(p):
    return [c for c in p.replace("\\", "/").split("/") if c]


def _has_wild(seg):
    return "*" in seg or "?" in seg or "[" in seg


def _seg_match(a, b):
    return fnmatch.fnmatchcase(a, b) or fnmatch.fnmatchcase(b, a)


def paths_overlap(a, b):
    """Segment-wise glob overlap; a wildcard segment consumes the rest.

    Windows is case-insensitive, so segments compare lowercased. A path that
    runs out first is a directory prefix of the other and therefore overlaps.
    """
    sa, sb = split_path(a), split_path(b)
    i = 0
    while i < len(sa) and i < len(sb):
        x, y = sa[i].lower(), sb[i].lower()
        if x == "**" or y == "**":
            return True
        if _has_wild(x) or _has_wild(y):
            if not _seg_match(x, y):
                return False
        elif x != y:
            return False
        i += 1
    return True


PATH_CASES = [
    (("runtime/attn.*", "runtime/attn.cpp"), True),
    (("runtime/attn.*", "runtime/attn.h"), True),
    (("runtime/layer.cpp", "runtime/attn.cpp"), False),
    (("runtime/*", "runtime/kernels/gemm.cpp"), True),
    (("bench/raw/**", "bench/raw/a.bin"), True),
    (("runtime/kernels/gemm.cpp", "runtime/kernels/gemm_q8k.comp"), False),
    (("runtime", "runtime/layer.cpp"), True),
    (("bench/state/state.json", "bench/tools/state_check.py"), False),
]


# ---------------------------------------------------------------- renders

def canonical_queues(st):
    return " ".join("%s=%s" % (q["lane"], q.get("head") or "-")
                    for q in st["queues"])


def render_last_gate(st):
    g = st.get("last_gate") or {}
    if not g.get("id"):
        return "none"
    return "%s %s (%s)" % (g.get("id"), g.get("result", "unknown"),
                           g.get("date", "?"))


def render_blockers(st):
    bl = st.get("blockers") or []
    return ", ".join(bl) if bl else "none"


def parse_state_block(text):
    m = re.search(r"<!--\s*STATE:BEGIN\s*-->(.*?)<!--\s*STATE:END\s*-->", text, re.S)
    if not m:
        raise ValueError("AGENTS.md has no STATE:BEGIN/STATE:END block")
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("STATE line without ':' -> " + line)
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def task_reads(st, task):
    """Task-level reads override the lane default."""
    if task.get("reads"):
        return list(task["reads"])
    lane = st["lanes"].get(task.get("lane"), {})
    return list(lane.get("reads") or [])


def find_task(st, tid):
    for t in st["tasks"]:
        if t.get("id") == tid:
            return t
    return None


# ---------------------------------------------------------------- validate

def validate(st):
    errors, warnings = [], []
    for key in REQUIRED_TOP:
        if key not in st:
            errors.append("state.json missing key: " + key)
    if errors:
        return errors, warnings

    if st["schema"] != SCHEMA:
        (warnings if st["schema"] in (1, 2) else errors).append(
            "schema is %r, expected %d" % (st["schema"], SCHEMA))
    if st["phase"] not in PHASES:
        errors.append("unknown phase: %s" % st["phase"])
    if st["phase_status"] not in PHASE_STATUS:
        errors.append("unknown phase_status: %s" % st["phase_status"])

    resources = st["resources"]
    for rid, res in resources.items():
        if res.get("cap") != 1:
            errors.append("resource %s: only cap 1 is supported" % rid)

    lanes = st["lanes"]
    for lid, lane in lanes.items():
        for rid in lane.get("resources", []):
            if rid not in resources:
                errors.append("lane %s uses unknown resource %s" % (lid, rid))
        for rel in lane.get("reads", []) or []:
            if not (ROOT / rel).exists():
                errors.append("lane %s reads a missing file: %s" % (lid, rel))

    ids = [t.get("id") for t in st["tasks"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task id")
    by_id = {t.get("id"): t for t in st["tasks"]}

    for t in st["tasks"]:
        tid = t.get("id", "?")
        for key in REQUIRED_TASK:
            if key not in t:
                errors.append("task %s missing %s" % (tid, key))
        if t.get("status") not in TASK_STATUS:
            errors.append("task %s has unknown status %r" % (tid, t.get("status")))
        if t.get("lane") not in lanes:
            errors.append("task %s uses unknown lane %r" % (tid, t.get("lane")))
        for rid in t.get("resources", []):
            if rid not in resources:
                errors.append("task %s uses unknown resource %s" % (tid, rid))
        for dep in t.get("deps", []):
            if dep not in by_id:
                errors.append("task %s depends on unknown task %s" % (tid, dep))
        for rel in t.get("reads", []) or []:
            if not (ROOT / rel).exists():
                errors.append("task %s reads a missing file: %s" % (tid, rel))
        if t.get("status") in ("doing", "done"):
            open_deps = [d for d in t.get("deps", [])
                         if by_id.get(d, {}).get("status") != "done"]
            if open_deps:
                errors.append("task %s is %s but deps are open: %s"
                              % (tid, t["status"], ", ".join(open_deps)))
        if t.get("evidence") and not isinstance(t["evidence"], dict):
            errors.append("task %s evidence must be an object" % tid)

    doing = [t for t in st["tasks"] if t.get("status") == "doing"]
    now = time.time()
    for t in doing:
        ts = t.get("claimed_at")
        if not ts:
            errors.append("task %s is doing but has no claimed_at" % t["id"])
            continue
        try:
            stamp = datetime.fromisoformat(ts)
        except ValueError:
            errors.append("task %s has a bad claimed_at: %r" % (t["id"], ts))
            continue
        if stamp.tzinfo is None:
            warnings.append("task %s claimed_at has no timezone; write %s+08:00"
                            % (t["id"], ts))
        age_h = (now - stamp.timestamp()) / 3600.0
        if age_h > CLAIM_TTL_HOURS:
            errors.append("task %s has been doing for %.1f h; re-claim or release it"
                          % (t["id"], age_h))
        if not t.get("owner"):
            warnings.append("task %s is doing without owner; write owner and worktree"
                            % t["id"])

    holders = {}
    for t in doing:
        for rid in t.get("resources", []):
            holders.setdefault(rid, []).append(t["id"])
    for rid, tids in holders.items():
        if len(tids) > 1:
            errors.append("resource %s is claimed by %s at the same time"
                          % (rid, ", ".join(tids)))

    legacy = sorted((ROOT / "bench" / "state").glob("quiet-*"))
    if legacy:
        warnings.append("deprecated lock files present (use doing+resources): "
                        + ", ".join(q.name for q in legacy))
    machine_doing = [t["id"] for t in doing if t.get("lane") in ("bench", "io")]
    if legacy and machine_doing:
        errors.append("deprecated lock %s is held while %s owns the machine"
                      % (legacy[0].name, machine_doing[0]))

    for i, a in enumerate(doing):
        for b in doing[i + 1:]:
            if a["lane"] == b["lane"] == "bench":
                shared = set(a.get("resources", [])) & set(b.get("resources", []))
                if shared:
                    errors.append("two bench tasks share %s: %s and %s"
                                  % (", ".join(sorted(shared)), a["id"], b["id"]))
            for pa in a.get("paths", []):
                for pb in b.get("paths", []):
                    if paths_overlap(pa, pb):
                        errors.append("paths overlap between running tasks %s (%s) and %s (%s)"
                                      % (a["id"], pa, b["id"], pb))

    heads = {}
    for q in st["queues"]:
        lid = q.get("lane")
        if lid not in lanes:
            errors.append("queue for unknown lane: %s" % lid)
            continue
        if lid in heads:
            errors.append("lane %s has more than one queue" % lid)
        heads[lid] = q
        head = q.get("head")
        if head is None:
            continue
        if head not in by_id:
            errors.append("lane %s head %s does not exist" % (lid, head))
            continue
        t = by_id[head]
        if t.get("status") not in ("todo", "partial"):
            errors.append("lane %s head %s has status %s"
                          % (lid, head, t.get("status")))
        if t.get("lane") != lid:
            errors.append("lane %s head %s belongs to lane %s"
                          % (lid, head, t.get("lane")))
        if lid != "user":
            open_deps = [d for d in t.get("deps", [])
                         if by_id.get(d, {}).get("status") != "done"]
            if open_deps:
                errors.append("lane %s head %s is not runnable, open deps: %s"
                              % (lid, head, ", ".join(open_deps)))

    gate = st["decision_gate"]
    if gate.get("id") not in PHASES:
        errors.append("decision_gate.id is not a phase: %s" % gate.get("id"))
    if gate.get("open") and gate.get("id") in PHASES:
        if PHASES.index(st["phase"]) > PHASES.index(gate["id"]):
            errors.append("phase %s advanced past open decision gate %s"
                          % (st["phase"], gate["id"]))
    if gate.get("result") not in (None, "pass", "fail"):
        errors.append("decision_gate.result must be null/pass/fail")

    if st["last_gate"].get("result") not in ("pass", "fail", "unknown"):
        errors.append("last_gate.result must be pass/fail/unknown")
    for i, esc in enumerate(st.get("escalations", [])):
        for key in ("id", "question", "needs_by"):
            if key not in esc:
                errors.append("escalation[%d] missing %s" % (i, key))

    errors.extend(state_block_check(st))
    return errors, warnings


def state_block_check(st):
    """Every STATE key that AGENTS.md mirrors must match state.json exactly."""
    errors = []
    try:
        block = parse_state_block(AGENTS.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [str(exc)]
    gate = st["decision_gate"]
    want = {
        "phase": st["phase"],
        "phase_status": st["phase_status"],
        "decision_gate": "%s %s" % (gate["id"], "open" if gate.get("open") else "closed"),
        "queues": canonical_queues(st),
        "last_gate": render_last_gate(st),
        "blockers": render_blockers(st),
    }
    for key, value in want.items():
        if block.get(key) != value:
            errors.append("AGENTS %s=%r but state.json renders %r"
                          % (key, block.get(key), value))
    return errors


# ---------------------------------------------------------------- doc lint

def _resolvable(hit):
    """Only slash paths are unambiguous; bare names and overlays may be absent."""
    return ("/" in hit) and not hit.endswith(".local.md")


def lint_docs(st):
    """The four rules that keep L0 slim and the index honest."""
    errors, warnings = [], []
    budget = st.get("doc_budget") or {}
    max_bytes = budget.get("agents_md_bytes", 8192)
    max_lines = budget.get("agents_md_lines", 140)

    if not AGENTS.exists():
        return ["AGENTS.md is missing"], warnings
    raw = AGENTS.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    if len(raw) > max_bytes:
        errors.append("AGENTS.md is %d B, budget is %d B" % (len(raw), max_bytes))
    if len(lines) > max_lines:
        errors.append("AGENTS.md is %d lines, budget is %d" % (len(lines), max_lines))
    for n, line in enumerate(lines, 1):
        if DATE_BULLET.match(line):
            errors.append("AGENTS.md:%d dated bullet: history belongs in RESULTS/DECISIONS"
                          % n)
        if TASK_ROW.match(line):
            errors.append("AGENTS.md:%d task table row; use --board" % n)

    if not DOCS.is_dir():
        errors.append("docs/agent/ is missing")
        return errors, warnings
    for rel, cap in sorted((budget.get("docs") or {}).items()):
        p = ROOT / rel
        if not p.exists():
            errors.append("doc budget lists a missing file: " + rel)
            continue
        size = p.stat().st_size
        if size > cap:
            errors.append("%s is %d B, budget is %d B" % (rel, size, cap))
    idx = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
    if not idx:
        errors.append("docs/agent/INDEX.md is missing")
    else:
        for p in sorted(DOCS.glob("*.md")):
            if p.name == "INDEX.md":
                continue
            if p.name not in idx:
                errors.append("docs/agent/%s is not listed in INDEX.md" % p.name)
        for hit in sorted(set(MD_PATH.findall(idx))):
            if "*" in hit or not _resolvable(hit):
                continue
            if not (ROOT / hit).exists():
                errors.append("INDEX.md points at a missing file: %s" % hit)

    for path in (ROOT / "AGENTS.md", INDEX, DOCS / "cycle.md"):
        if not path.exists():
            continue
        for hit in sorted(set(MD_PATH.findall(path.read_text(encoding="utf-8")))):
            if "*" in hit or not _resolvable(hit) or hit.startswith("build-"):
                continue
            if not (ROOT / hit).exists():
                warnings.append("%s points at a missing file: %s" % (path.name, hit))

    slugs = {}
    for name in ("RESULTS.md", "DECISIONS.md"):
        p = ROOT / "bench" / name
        if not p.exists():
            continue
        found = SLUG_DEF.findall(p.read_text(encoding="utf-8"))
        dupes = sorted({s for s in found if found.count(s) > 1})
        if dupes:
            errors.append("%s has duplicate slugs: %s" % (name, ", ".join(dupes)))
        slugs[name[:-3]] = set(found)

    for p in list(DOCS.glob("*.md")) + [AGENTS]:
        if not p.exists():
            continue
        for doc, slug in SLUG_REF.findall(p.read_text(encoding="utf-8")):
            if doc in slugs and slug not in slugs[doc]:
                warnings.append("%s references %s.md#%s which does not exist"
                                % (p.name, doc, slug))

    for esc in st.get("escalations", []):
        if esc.get("resolved") and esc.get("state") != "closed":
            warnings.append("escalation %s has resolved but state is not closed"
                            % esc.get("id"))
    return errors, warnings


# ---------------------------------------------------------------- output

def print_board(st, only_lane=None):
    for lid in st["lanes"]:
        if only_lane and lid != only_lane:
            continue
        lane = st["lanes"][lid]
        q = next((x for x in st["queues"] if x["lane"] == lid), {})
        print("")
        print("[%s] resources=%s head=%s" % (
            lid, ",".join(lane.get("resources", [])) or "-", q.get("head") or "-"))
        for t in st["tasks"]:
            if t["lane"] != lid:
                continue
            deps = ",".join(t["deps"]) or "-"
            print("  %-8s %-7s %-5s deps=%-28s %s"
                  % (t["id"], t["status"], t["milestone"], deps, t["title"]))
        if q.get("note"):
            print("  note: " + q["note"])


def print_briefing(st):
    gate = st["decision_gate"]
    print("track  : %s (schema %s, %s tasks)" % (st["track"], st["schema"], len(st["tasks"])))
    print("phase  : %s / %s" % (st["phase"], st["phase_status"]))
    print("gate   : %s %s result=%s" % (
        gate["id"], "open" if gate.get("open") else "closed", gate.get("result")))
    for q in st["queues"]:
        head = q.get("head")
        if head:
            t = find_task(st, head) or {}
            owner = (" owner=%s" % t["owner"]) if t.get("owner") else ""
            print("queue  : %-5s %-8s %s%s" % (q["lane"], head, t.get("title", "?"), owner))
        else:
            print("queue  : %-5s -        (waiting)" % q["lane"])
            if q.get("note"):
                print("         " + q["note"])
    doing = [t for t in st["tasks"] if t.get("status") == "doing"]
    print("running: %s" % (", ".join("%s(%s)" % (
        t["id"], ",".join(t.get("resources", [])) or "no-res") for t in doing) or "none"))
    held = {}
    for t in doing:
        for rid in t.get("resources", []):
            held[rid] = t["id"]
    print("lock   : %s" % (", ".join("%s=%s" % (k, v) for k, v in sorted(held.items()))
                           or "free"))
    print("gates  : %s" % render_last_gate(st))
    print("block  : %s" % render_blockers(st))
    for esc in st.get("escalations", []):
        if esc.get("state") == "closed":
            continue
        print("escal  : %s needs user by %s - %s"
              % (esc.get("id"), esc.get("needs_by"), esc.get("question")))


def print_brief(st, tid):
    t = find_task(st, tid)
    if not t:
        print("no such task: " + tid)
        return 1
    by_id = {x["id"]: x for x in st["tasks"]}
    deps = ["%s(%s)" % (d, by_id.get(d, {}).get("status", "?")) for d in t.get("deps", [])]
    print("task     : %s [%s / %s] %s" % (t["id"], t["lane"], t["status"], t["title"]))
    if t.get("owner"):
        print("owner    : %s  claimed_at=%s" % (t["owner"], t.get("claimed_at")))
    print("deps     : %s" % (", ".join(deps) or "-"))
    print("resources: %s" % (", ".join(t.get("resources", [])) or "-"))
    print("paths    : %s" % (", ".join(t.get("paths", [])) or "-"))
    print("")
    print("READ THESE FIRST, in this order:")
    for rel in task_reads(st, t):
        p = ROOT / rel
        print("  [%s] %s%s" % ("ok" if p.exists() else "MISSING", rel,
                              ("  (%d B)" % p.stat().st_size) if p.exists() else ""))
    print("")
    print("done_when: %s" % t.get("done_when", "-"))
    if t.get("evidence"):
        print("evidence : %s" % json.dumps(t["evidence"], ensure_ascii=False))
    if t.get("note"):
        print("note     : %s" % t["note"])
    return 0


def run_selftest():
    bad = 0
    for (a, b), want in PATH_CASES:
        got = paths_overlap(a, b)
        if got != want:
            bad += 1
            print("FAIL paths_overlap(%r, %r) = %r, want %r" % (a, b, got, want))
    st = {"queues": [{"lane": "bench", "head": None}, {"lane": "impl", "head": "M2-6"}],
          "last_gate": {"id": "G-REG", "result": "unknown", "date": "2026-10-23"},
          "blockers": []}
    if canonical_queues(st) != "bench=- impl=M2-6":
        bad += 1
        print("FAIL canonical_queues")
    if render_last_gate(st) != "G-REG unknown (2026-10-23)":
        bad += 1
        print("FAIL render_last_gate")
    if render_blockers(st) != "none":
        bad += 1
        print("FAIL render_blockers empty")
    if render_blockers({"blockers": ["a", "b"]}) != "a, b":
        bad += 1
        print("FAIL render_blockers list")
    sample = "<!-- STATE:BEGIN -->\nphase: M2\nqueues: bench=- impl=M2-6\n<!-- STATE:END -->"
    if parse_state_block(sample) != {"phase": "M2", "queues": "bench=- impl=M2-6"}:
        bad += 1
        print("FAIL parse_state_block")
    if not DATE_BULLET.match("- **M2-5 done（2026-10-25）**"):
        bad += 1
        print("FAIL DATE_BULLET should match a dated bullet")
    if DATE_BULLET.match("last_gate: G-REG unknown (2026-10-23)"):
        bad += 1
        print("FAIL DATE_BULLET should not match a key line")
    if not TASK_ROW.match("| M2-6 | impl | x |"):
        bad += 1
        print("FAIL TASK_ROW should match a task table row")
    if TASK_ROW.match("| **G-CONS** | every round |"):
        bad += 1
        print("FAIL TASK_ROW should not match a gate row")
    total = len(PATH_CASES) + 9
    print("%d/%d selftest cases passed" % (total - bad, total))
    return 0 if bad == 0 else 1


def main(argv):
    if "--selftest" in argv:
        return run_selftest()
    if not STATE.exists() or not AGENTS.exists():
        print("missing bench/state/state.json or AGENTS.md")
        return 2
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print("state.json is not valid JSON: %s" % exc)
        return 1

    if "--brief" in argv:
        i = argv.index("--brief")
        if i + 1 >= len(argv):
            print("--brief needs a task id")
            return 2
        return print_brief(st, argv[i + 1])

    only_lane = None
    if "--lane" in argv:
        j = argv.index("--lane")
        if j + 1 < len(argv):
            only_lane = argv[j + 1]

    errors, warnings = validate(st)
    if "--lint" not in argv:
        lerr, lwarn = lint_docs(st)
        errors.extend(lerr)
        warnings.extend(lwarn)

    if "--board" in argv or only_lane:
        print_board(st, only_lane)
    else:
        print_briefing(st)

    if warnings:
        print("")
        for w in warnings:
            print("WARN   : " + w)
    if errors:
        print("")
        for e in errors:
            print("ERROR  : " + e)
        return 1
    print("")
    print("OK: graph, resources, queues, AGENTS.md and the doc layout agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
