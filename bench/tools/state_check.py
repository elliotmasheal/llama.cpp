#!/usr/bin/env python3
"""Validate bench/state/state.json and its mirror block in AGENTS.md.

Checks the task graph, the resource claims and the per-lane queues, so that
several agents can work in parallel without taking the same exclusive resource
or writing the same file.

  python bench/tools/state_check.py            briefing + validation
  python bench/tools/state_check.py --board    full task board, grouped by lane
  python bench/tools/state_check.py --lane impl

A quiet lock is a file bench/state/quiet-<task>. impl and io agents create it
before building or running anything on the machine and remove it afterwards, so
that a build never overlaps a timed bench run.
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "bench" / "state" / "state.json"
AGENTS = ROOT / "AGENTS.md"

PHASES = ["M0", "M1", "M2", "M3", "M4", "M5"]
PHASE_STATUS = {"not_started", "in_progress", "blocked", "done"}
TASK_STATUS = {"todo", "doing", "partial", "done", "blocked"}
REQUIRED_TOP = [
    "schema", "track", "updated", "phase", "phase_status", "decision_gate",
    "resources", "lanes", "queues", "tasks", "last_gate", "blockers", "escalations",
]
REQUIRED_TASK = ["id", "milestone", "lane", "status", "title", "deps", "resources", "paths", "done_when"]


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


def split_path(p):
    return [c for c in p.replace("\\", "/").split("/") if c]


def paths_overlap(a, b):
    """Conservative glob overlap: component-wise, wildcards match anything."""
    for x, y in zip(split_path(a), split_path(b)):
        if x == "**" or y == "**" or x.startswith("*") or y.startswith("*"):
            return True
        if x != y:
            return False
    return True  # one path is a prefix (directory) of the other


def canonical_queues(st):
    parts = []
    for q in st["queues"]:
        parts.append("%s=%s" % (q["lane"], q["head"] if q.get("head") else "-"))
    return " ".join(parts)


def validate(st):
    errors = []
    for key in REQUIRED_TOP:
        if key not in st:
            errors.append("state.json missing key: " + key)
    if errors:
        return errors

    if st["schema"] != 2:
        errors.append("schema must be 2, got %r" % st["schema"])
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
        if t.get("status") in ("doing", "done"):
            open_deps = [d for d in t.get("deps", []) if by_id.get(d, {}).get("status") != "done"]
            if open_deps:
                errors.append("task %s is %s but deps are open: %s" % (tid, t["status"], ", ".join(open_deps)))

    doing = [t for t in st["tasks"] if t.get("status") == "doing"]
    holders = {}
    for t in doing:
        for rid in t.get("resources", []):
            holders.setdefault(rid, []).append(t["id"])
    for rid, tids in holders.items():
        if len(tids) > 1:
            errors.append("resource %s is claimed by %s at the same time" % (rid, ", ".join(tids)))

    quiet = sorted((ROOT / "bench" / "state").glob("quiet-*"))
    if len(quiet) > 1:
        errors.append("more than one quiet lock: " + ", ".join(q.name for q in quiet))
    machine_doing = [t["id"] for t in st["tasks"] if t.get("status") == "doing" and t.get("lane") in ("bench", "io")]
    if quiet and machine_doing:
        errors.append("quiet lock %s is held while %s owns the machine (bench/io)" % (quiet[0].name, machine_doing[0]))
    if quiet:
        age_min = (time.time() - quiet[0].stat().st_mtime) / 60.0
        if age_min > 120:
            errors.append("quiet lock %s is stale (%.0f min); remove it if the holder died" % (quiet[0].name, age_min))
    for i, a in enumerate(doing):
        for b in doing[i + 1:]:
            if a["lane"] == b["lane"] == "bench" and a.get("resources") == b.get("resources"):
                errors.append("two bench tasks are running: %s and %s" % (a["id"], b["id"]))
            for pa in a.get("paths", []):
                for pb in b.get("paths", []):
                    if paths_overlap(pa, pb):
                        errors.append("paths overlap between running tasks %s (%s) and %s (%s)" % (a["id"], pa, b["id"], pb))

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
            errors.append("lane %s head %s has status %s" % (lid, head, t.get("status")))
        if t.get("lane") != lid:
            errors.append("lane %s head %s belongs to lane %s" % (lid, head, t.get("lane")))
        if lid != "user":
            open_deps = [d for d in t.get("deps", []) if by_id.get(d, {}).get("status") != "done"]
            if open_deps:
                errors.append("lane %s head %s is not runnable, open deps: %s" % (lid, head, ", ".join(open_deps)))

    gate = st["decision_gate"]
    if gate.get("id") not in PHASES:
        errors.append("decision_gate.id is not a phase: %s" % gate.get("id"))
    if gate.get("open") and gate.get("id") in PHASES:
        if PHASES.index(st["phase"]) > PHASES.index(gate["id"]):
            errors.append("phase %s advanced past open decision gate %s" % (st["phase"], gate["id"]))
    if gate.get("result") not in (None, "pass", "fail"):
        errors.append("decision_gate.result must be null/pass/fail")

    if st["last_gate"].get("result") not in ("pass", "fail", "unknown"):
        errors.append("last_gate.result must be pass/fail/unknown")
    for i, esc in enumerate(st.get("escalations", [])):
        for key in ("id", "question", "needs_by"):
            if key not in esc:
                errors.append("escalation[%d] missing %s" % (i, key))

    try:
        block = parse_state_block(AGENTS.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
        block = {}
    if block:
        for key, want in (
            ("phase", st["phase"]),
            ("phase_status", st["phase_status"]),
            ("decision_gate", "%s %s" % (gate["id"], "open" if gate.get("open") else "closed")),
            ("queues", canonical_queues(st)),
        ):
            if block.get(key) != want:
                errors.append("AGENTS %s=%r but state.json expects %r" % (key, block.get(key), want))

    return errors


def print_board(st, only_lane=None):
    order = list(st["lanes"].keys())
    for lid in order:
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
            print("  %-8s %-7s %-5s deps=%-28s %s" % (t["id"], t["status"], t["milestone"], deps, t["title"]))
        if q.get("note"):
            print("  note: " + q["note"])


def print_briefing(st):
    by_id = {t["id"]: t for t in st["tasks"]}
    gate = st["decision_gate"]
    print("track  : %s (schema %s, %s tasks)" % (st["track"], st["schema"], len(st["tasks"])))
    print("phase  : %s / %s" % (st["phase"], st["phase_status"]))
    print("gate   : %s %s result=%s" % (gate["id"], "open" if gate.get("open") else "closed", gate.get("result")))
    for q in st["queues"]:
        head = q.get("head")
        if head:
            t = by_id.get(head, {})
            print("queue  : %-5s %-8s %s" % (q["lane"], head, t.get("title", "?")))
        else:
            print("queue  : %-5s -        (waiting)" % q["lane"])
            if q.get("note"):
                print("         " + q["note"])
    doing = [t for t in st["tasks"] if t["status"] == "doing"]
    print("running: %s" % (", ".join("%s(%s)" % (t["id"], ",".join(t["resources"]) or "no-res") for t in doing) or "none"))
    quiet = sorted((ROOT / "bench" / "state").glob("quiet-*"))
    print("quiet  : %s" % (", ".join(q.name for q in quiet) if quiet else "free"))
    print("gates  : %s %s (%s)" % (st["last_gate"].get("id"), st["last_gate"].get("result"), st["last_gate"].get("date")))
    print("block  : %s" % (", ".join(st["blockers"]) if st["blockers"] else "none"))
    for esc in st.get("escalations", []):
        print("escal  : %s needs user by %s - %s" % (esc.get("id"), esc.get("needs_by"), esc.get("question")))


def main(argv):
    if not STATE.exists() or not AGENTS.exists():
        print("missing bench/state/state.json or AGENTS.md")
        return 2
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print("state.json is not valid JSON: %s" % exc)
        return 1

    only_lane = None
    if "--lane" in argv:
        i = argv.index("--lane")
        if i + 1 < len(argv):
            only_lane = argv[i + 1]

    errors = validate(st)
    if "--board" in argv or only_lane:
        print_board(st, only_lane)
    else:
        print_briefing(st)
    if errors:
        print("")
        for e in errors:
            print("ERROR  : " + e)
        return 1
    print("")
    print("OK: graph, resources, queues and AGENTS.md agree; every queue head is runnable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
