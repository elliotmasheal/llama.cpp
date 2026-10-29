#!/bin/bash
# quality-gate.sh - PPL and logit KL of the current build against a recorded logits baseline.
#
# The gate is a numeric check, not a target-config run: it needs a text long enough for 2 x -C tokens
# (prompt10k has 10428, so -C 4096 is the largest context it can score). It still exercises the long
# cross-chunk attention path, which is what a graph or attention change can break.
#
# Usage:
#   bench/tools/quality-gate.sh              # PPL + KLD against the stored baseline
#   bench/tools/quality-gate.sh -R           # (re)record the baseline logits, prints PPL only
#   bench/tools/quality-gate.sh -C 8192 -f bench/prompts/prompt20k.txt
#
# The baseline file is ~2.0 GB at -C 4096 (2 chunks), keep it off the repo (`.gate/` is ignored).
# It lives under the repo root because a sandboxed child can only create files in directories that
# inherit the workspace write ACE; `bench/` and other pre-existing subdirectories do not.
set -u

cd "$(dirname "$0")/../.." || exit 1
export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"

MODEL="F:/LLM/llm/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
EXE="build-vulkan/bin/llama-perplexity.exe"
PROMPT="bench/prompts/prompt10k.txt"
CTX=4096
CHUNKS=2
BASE=".gate/qwen35moe-10k-c4096.kld"
RECORD=0
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
    -R) RECORD=1; shift ;;
    -C) CTX=$2; shift 2 ;;
    -f) PROMPT=$2; shift 2 ;;
    -b) BASE=$2; shift 2 ;;
    -m) MODEL=$2; shift 2 ;;
    -O) OUT=$2; shift 2 ;;
    -h) sed -n '1,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -x "$EXE" ] || { echo "build first: $EXE not found" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "prompt not found: $PROMPT" >&2; exit 1; }
[ -n "$OUT" ] || OUT="bench/raw/gate-$(date +%Y-%m-%d-%H%M)"
mkdir -p "$OUT" || exit 1

# the same shape as the target口径 minus the context length: resident, experts on the host, FA on
COMMON=(-m "$MODEL" -dev Vulkan1 -ngl 99 -ncmoe 40 -fa on -t 5 -b 2048 -ub 2048
        -c "$CTX" -lm none --store-vram-mib 8176 --chunks "$CHUNKS" -f "$PROMPT" -v)

if [ "$RECORD" -eq 1 ]; then
    mkdir -p "$(dirname "$BASE")" || exit 1
    echo "recording the baseline logits to $BASE"
    "$EXE" "${COMMON[@]}" --kl-divergence-base "$BASE" > "$OUT/record.out" 2> "$OUT/record.err"
    rc=$?
    # the estimate line goes through the logger (stderr), the per-chunk progress stays on stdout
    ppl=$(grep -hE "Final estimate: PPL = " "$OUT/record.out" "$OUT/record.err" | tail -1 | awk '{print $(NF-2)}')
    echo "PPL = ${ppl:-?} (rc=$rc), baseline -> $BASE ($(du -h "$BASE" 2>/dev/null | cut -f1))"
    tail -1 "$OUT/record.out"
    exit $rc
fi

[ -f "$BASE" ] || { echo "no baseline: $BASE (run -R first)" >&2; exit 1; }

"$EXE" "${COMMON[@]}" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/check.out" 2> "$OUT/check.err"
rc=$?

# the KLD path prints the statistics table instead of a final estimate
ppl=$(grep -E "^Mean PPL\(Q\) +:" "$OUT/check.out" | awk '{print $4}')
[ -n "$ppl" ] || ppl=$(grep -hE "Final estimate: PPL = " "$OUT/check.out" "$OUT/check.err" | tail -1 | awk '{print $(NF-2)}')
kld=$(grep -E "^Mean    KLD:" "$OUT/check.out" | awk '{print $3}')
topp=$(grep -E "^Same top p:" "$OUT/check.out" | awk '{print $4}')

echo "PPL = ${ppl:-?}  mean KLD = ${kld:-?}  same top p = ${topp:-?}%  (rc=$rc)"
grep -E "^(Mean    KLD|Maximum KLD|Same top p|Mean PPL\(Q\)|Mean PPL\(base\))" "$OUT/check.out"

[ "$rc" -eq 0 ] || { echo "GATE FAIL: perplexity run failed"; exit 1; }
[ -n "$kld" ] || { echo "GATE FAIL: no KLD in the output"; exit 1; }

# a change that is numerically neutral stays under 1e-4; chunking and tiling can move the last digits
awk -v k="$kld" -v t="${topp:-0}" 'BEGIN {
    if (k < 0) k = -k;
    if (k > 1e-4 || t < 99.5) { printf("GATE FAIL: KLD %.6f, top-p %.3f%%\n", k, t); exit 1; }
    printf("GATE PASS: KLD %.6f, top-p %.3f%%\n", k, t);
}'
