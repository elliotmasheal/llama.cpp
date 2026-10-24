#!/usr/bin/env python3
"""Planned decode compute workspace per k (M1-4b).

Derived from the model shapes, not driver-measured: the runtime that owns these
buffers only exists after M2-0/M2-2, so M2-2 must confirm the number. The point
is to size the decode-time tensor set; prefill workspaces (chunk 2048) and the
Vulkan driver transients are separate and not included.

    python bench/tools/workspace_plan.py
"""
MIB = 1024 * 1024

N_EMBD = 2048
N_FF = 512
N_VOCAB = 248320
DRAFT_VOCAB = 16384
N_LAYER = 40
TOP_K = 8
N_ROUTER = 256

DESKTOP = 1000
DENSE = 2079
KV_WINDOW = 99
DRAFT_BLOCK = 100
VRAM = 8176
EXPERT_TOTAL = 40 * 256 * 1.84   # MiB, RESULTS 2.1/8


def workspace(k):
    x = k * N_EMBD * 4
    q8x = k * (N_EMBD + 2 * (N_EMBD // 32) * 4)
    gate_up = k * N_FF * 4 * 2
    act = k * (N_FF * 4 + N_FF + 2 * (N_FF // 32) * 4)
    out = k * N_EMBD * 4
    router = k * N_ROUTER * 4
    route_ids = k * N_LAYER * TOP_K * 4
    logits = k * N_VOCAB * 4
    draft = (k - 1) * DRAFT_VOCAB * 4
    cmd = 1 * MIB
    scratch = 4 * MIB
    return {
        "x+q8": x + q8x,
        "gate_up+act": gate_up + act,
        "out": out,
        "router+ids": router + route_ids,
        "verify_logits": logits,
        "draft_logits": draft,
        "cmd+scratch": cmd + scratch,
    }


def main():
    print("k,workspace_mib,verify_logits_mib,draft_mib,cache_room_mib,cache_pct_of_experts")
    for k in (1, 2, 4, 5, 8):
        parts = workspace(k)
        total = sum(parts.values()) / MIB
        room = VRAM - DESKTOP - DENSE - KV_WINDOW - DRAFT_BLOCK - total
        print("%d,%.1f,%.1f,%.1f,%.0f,%.1f" % (
            k, total, parts["verify_logits"] / MIB, parts["draft_logits"] / MIB,
            room, room * 100.0 / EXPERT_TOTAL))
    print("")
    print("cache_room_mib = 8176 - desktop(1000) - dense(2079) - KV window(99) - draft(100) - workspace.")
    print("design 5.5 keeps 4198 MiB (21.9%) and reserves 700-1100 MiB for driver/prefill")
    print("transients that this table does not include; M2-2 must measure them before")
    print("the cache is raised. Hit rates (RESULTS 7): 20%% 77.3, 25%% 81.6, 30%% 85.4.")


if __name__ == "__main__":
    main()
