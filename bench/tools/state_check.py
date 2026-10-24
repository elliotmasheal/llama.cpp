#!/usr/bin/env python3
"""Validate bench/state/state.json and the STATE block mirrored in AGENTS.md.

Exit code 0 means the two are consistent and the next task is executable.
Exit code 1 prints every problem found. Exit code 2 means a file is missing.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "bench" / "state" / "state.json"
AGENTS = ROOT / "AGENTS.md"

PHASES = ["M0", "M1", "M2", "M3", "M4", "M5"]
PHASE_STATUS = {"not_started", "in_progress", "blocked", "done"}
TASK_STATUS = {"todo", "doing", "partial", "done", "blocked"}
REQUIRED_TOP = [
    "schema", "track", "updated", "phase", "phase_status", "decision_gate",
    "tasks", "next", "last_gate", "blockers", "escalations",
]
REQUIRED_TASK = ["id", "title", "status", "blocked_by", "done_when"]


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


def main():
    missing = [str(p) for p in (STATE, AGENTS) if not p.exists()]
    if missing:
        print("missing file: " + ", ".join(missing))
        return 2

    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print("state.json is not valid JSON: %s" % exc)
        return 1

    errors = []
    for key in REQUIRED_TOP:
        if key not in st:
            errors.append("state.json missing key: " + key)
    if errors:
        print("\n".join(errors))
        return 1

    if st["phase"] not in PHASES:
        errors.append("unknown phase: %s" % st["phase"])
    if st["phase_status"] not in PHASE_STATUS:
        errors.append("unknown phase_status: %s" % st["phase_status"])

    ids = []
    all_ids = [t.get("id") for t in st["tasks"]]
    doing = []
    for i, task in enumerate(st["tasks"]):
        for key in REQUIRED_TASK:
            if key not in task:
                errors.append("task[%d] missing %s" % (i, key))
        tid = task.get("id", "?")
        if tid in ids:
            errors.append("duplicate task id: " + tid)
        ids.append(tid)
        if task.get("status") not in TASK_STATUS:
            errors.append("task %s has unknown status: %s" % (tid, task.get("status")))
        if task.get("status") == "doing":
            doing.append(tid)
        unknown = [d for d in task.get("blocked_by", []) if d not in all_ids]
        if unknown:
            errors.append("task %s blocked_by unknown task: %s" % (tid, unknown))
    if len(doing) > 1:
        errors.append("more than one task is doing (one round, one change): " + ", ".join(doing))

    nxt = st["next"].get("task_id")
    by_id = {t.get("id"): t for t in st["tasks"]}
    if nxt not in by_id:
        errors.append("next.task_id does not exist: %s" % nxt)
    elif by_id[nxt].get("status") == "done":
        errors.append("next.task_id is already done: " + nxt)
    elif by_id[nxt].get("status") == "blocked":
        errors.append("next.task_id is blocked: %s, pick the unblocking task first" % nxt)

    gate = st["decision_gate"]
    gate_id = gate.get("id")
    if gate_id not in PHASES:
        errors.append("decision_gate.id is not a phase: %s" % gate_id)
    if gate.get("open") and gate_id in PHASES:
        if PHASES.index(st["phase"]) > PHASES.index(gate_id):
            errors.append("phase %s advanced past open decision gate %s" % (st["phase"], gate_id))
    if gate.get("result") not in (None, "pass", "fail"):
        errors.append("decision_gate.result must be null/pass/fail")

    last = st["last_gate"]
    if last.get("result") not in ("pass", "fail", "unknown"):
        errors.append("last_gate.result must be pass/fail/unknown")

    if not isinstance(st["escalations"], list):
        errors.append("escalations must be a list")
    else:
        for i, esc in enumerate(st["escalations"]):
            for key in ("id", "question", "needs_by"):
                if key not in esc:
                    errors.append("escalation[%d] missing %s" % (i, key))

    try:
        block = parse_state_block(AGENTS.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
        block = {}
    if block:
        if block.get("phase") != st["phase"]:
            errors.append("AGENTS phase=%r but state.json phase=%r" % (block.get("phase"), st["phase"]))
        if block.get("phase_status") != st["phase_status"]:
            errors.append("AGENTS phase_status=%r but state.json=%r" % (block.get("phase_status"), st["phase_status"]))
        if block.get("next") != nxt:
            errors.append("AGENTS next=%r but state.json next=%r" % (block.get("next"), nxt))
        want_gate = "%s %s" % (gate_id, "open" if gate.get("open") else "closed")
        if block.get("decision_gate") != want_gate:
            errors.append("AGENTS decision_gate=%r but expected %r" % (block.get("decision_gate"), want_gate))

    print("track  : %s (schema %s)" % (st["track"], st["schema"]))
    print("phase  : %s / %s" % (st["phase"], st["phase_status"]))
    print("gate   : %s %s result=%s" % (gate_id, "open" if gate.get("open") else "closed", gate.get("result")))
    print("next   : %s - %s" % (nxt, by_id.get(nxt, {}).get("title", "?")))
    print("why    : %s" % st["next"].get("why", ""))
    print("step   : %s" % st["next"].get("first_step", ""))
    print("done   : %s" % by_id.get(nxt, {}).get("done_when", ""))
    print("gates  : %s %s (%s)" % (last.get("id"), last.get("result"), last.get("date")))
    print("block  : %s" % (", ".join(st["blockers"]) if st["blockers"] else "none"))
    if st.get("escalations"):
        for esc in st["escalations"]:
            print("escal  : %s needs user by %s - %s" % (esc.get("id"), esc.get("needs_by"), esc.get("question")))

    if errors:
        print("")
        for e in errors:
            print("ERROR  : " + e)
        return 1
    print("")
    print("OK: state.json and AGENTS.md agree, next task is executable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
