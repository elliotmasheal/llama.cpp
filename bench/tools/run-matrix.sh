#!/bin/bash
# run-matrix.sh - unified target matrix: 256K ctx + 10K prompt on emulated small-VRAM tiers.
#
# The child runs with plain shell redirection. Do not "improve" this into PowerShell
# Start-Process -RedirectStandardError: pumping a -v child through it measured 4.5x slower decode.
#
# Usage:
#   bench/tools/run-matrix.sh                          # 8176,6144,4096 at ctx 262144 with -nkvo
#   bench/tools/run-matrix.sh -T 8176 -C 16384 -B 3584 -K   # one arm, KV on GPU
#   bench/tools/run-matrix.sh -L mmap                  # the RAM-short tier
set -u

cd "$(dirname "$0")/../.." || exit 1
export PATH="/d/Scoop/apps/mingw/current/bin:$PATH"

MODEL="F:/LLM/llm/Ornith-1.5-35B-Q4_K_M.gguf"
EXE="build-vulkan/bin/llama-cli.exe"
PROMPT="bench/prompts/prompt10k.txt"
TIERS="8176,6144,4096"
CTX=262144
BANK=auto
NCMOE=41
KV_ON_GPU=0
LOAD=none
REPEAT=1
GEN=64
UB=2048
THREADS=5
GAP=30
OUT=""
EXTRA_ENV=""

while [ $# -gt 0 ]; do
    case "$1" in
    -T) TIERS=$2; shift 2 ;;
    -C) CTX=$2; shift 2 ;;
    -B) BANK=$2; shift 2 ;;
    -n) NCMOE=$2; shift 2 ;;
    -K) KV_ON_GPU=1; shift ;;
    -L) LOAD=$2; shift 2 ;;
    -r) REPEAT=$2; shift 2 ;;
    -g) GEN=$2; shift 2 ;;
    -G) GAP=$2; shift 2 ;;
    -E) EXTRA_ENV=$2; shift 2 ;;
    -O) OUT=$2; shift 2 ;;
    -h) sed -n '1,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -x "$EXE" ] || { echo "build first: $EXE not found" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "prompt not found: $PROMPT" >&2; exit 1; }
[ -n "$OUT" ] || OUT="bench/raw/matrix-$(date +%Y-%m-%d-%H%M)"
mkdir -p "$OUT" || exit 1
CSV="$OUT/matrix.csv"

# --- preflight: the口径 requires the balanced scheme and an idle dGPU ------------------------------
scheme=$(powershell -NoProfile -Command "powercfg /getactivescheme" 2>/dev/null | tr -d '\r')
case "$scheme" in
    *381b4222*) ;;
    *) echo "WARNING: power scheme is not Balanced (381b4222), numbers are not comparable" >&2 ;;
esac
busy=$(tasklist 2>/dev/null | grep -icE "^llama-")
[ "$busy" -eq 0 ] || { echo "ERROR: $busy inference process(es) hold the dGPU, stop them first" >&2; exit 1; }

printf 'tier,run,load,ctx,kv,bank,ncmoe,pp_tps,tg_tps,vram_peak_mib,held_mib,vram_used_mib,ram_peak_mib,reallocs,hit_pct,status\n' > "$CSV"

sample_gpu() { # $1 = windows pid
    powershell -NoProfile -Command \
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | \
         Where-Object InstanceName -like '*pid_$1_*' | ForEach-Object { \$_.CookedValue / 1MB }" 2>/dev/null | \
        tr -d '\r' | sort -g | tail -1
}

run_arm() { # $1 = tier, $2 = run index
    tier=$1; run=$2
    arm="vram${tier}_c${CTX}_b${BANK}_r${run}"
    out="$OUT/$arm.out"; err="$OUT/$arm.err"

    set -- -m "$MODEL" -dev Vulkan1 -ngl 99 -fa on -t "$THREADS" -b "$UB" -ub "$UB" \
           -c "$CTX" -lm "$LOAD" -f "$PROMPT" -n "$GEN" \
           --temp 0 --no-warmup --no-display-prompt -v
    [ "$BANK" = auto ] || set -- "$@" --moe-cache-mib "$BANK"
    [ "$NCMOE" -ge 0 ] && set -- "$@" -ncmoe "$NCMOE"
    [ "$KV_ON_GPU" -eq 1 ] || set -- "$@" -nkvo

    GGML_VK_VRAM_LIMIT_MIB=$tier env $EXTRA_ENV "$EXE" "$@" > "$out" 2> "$err" &
    job=$!
    # \$! is an MSYS pid, the GPU counter needs the Windows one; preflight guarantees a single child
    winpid=""
    while [ -z "$winpid" ] && kill -0 "$job" 2>/dev/null; do
        winpid=$(powershell -NoProfile -Command \
            "(Get-Process -Name llama-cli -ErrorAction SilentlyContinue | Select-Object -First 1).Id" 2>/dev/null | tr -d '\r')
        sleep 2
    done

    peak=0
    peakram=0
    while kill -0 "$job" 2>/dev/null; do
        v=$(sample_gpu "$winpid")
        [ -n "$v" ] && peak=$(echo "$v $peak" | awk '{if ($1>$2) print $1; else print $2}')
        r=$(powershell -NoProfile -Command \
            "(Get-Process -Id $winpid -ErrorAction SilentlyContinue).PeakWorkingSet64 / 1MB" 2>/dev/null | tr -d '\r' | head -1)
        [ -n "$r" ] && peakram=$(echo "$r $peakram" | awk '{if ($1>$2) print $1; else print $2}')
        sleep 2
    done
    wait "$job"; rc=$?

    held=$(grep -oE "VRAM limited to [0-9.]+ MiB, [0-9.]+ MiB of" "$err" | head -1 | \
           sed -n 's/.*, \([0-9.]*\) MiB of/\1/p')
    reallocs=$(grep -c "graph reallocation" "$err")
    line=$(grep -E "Prompt: " "$out" | tail -1)
    pp=$(echo "$line" | sed -n 's/.*Prompt: \([0-9.]*\) t.*/\1/p')
    tg=$(echo "$line" | sed -n 's/.*Generation: \([0-9.]*\) t.*/\1/p')
    hit=$(grep -oE "hit rate = [0-9.]+%" "$err" | sed -n 's/hit rate = //; s/%//p' | \
          awk '{s+=$1; n++} END {if (n>0) printf "%.1f", s/n}')
    if [ -n "$pp" ]; then status=ok
    elif grep -q "ErrorOutOfDeviceMemory" "$err"; then status=vram-oom
    elif grep -qE "GGML_ABORT|the experts of layer .* are selected|\.cpp:[0-9]+: " "$err"; then status=abort
    elif grep -q "0xC0000005\|EXCEPTION" "$err"; then status=crash
    else status="no-result(rc=$rc)"
    fi
    peak_i=$(echo "${peak:-0}" | awk '{printf "%.0f", $1}')
    ram_i=$(echo "${peakram:-0}" | awk '{printf "%.0f", $1}')
    used=$(echo "$peak_i ${held:-0}" | awk '{printf "%.0f", $1-$2}')
    kv=$([ "$KV_ON_GPU" = 1 ] && echo gpu || echo host)

    echo "$tier,$run,$LOAD,$CTX,$kv,$BANK,$NCMOE,${pp:-},${tg:-},$peak_i,${held:-0},$used,$ram_i,$reallocs,${hit:-},$status" >> "$CSV"
    echo "arm $arm: PP=${pp:-?} TG=${tg:-?} peak=$peak_i held=${held:-0} used=$used ram=$ram_i reallocs=$reallocs status=$status"
    [ "$run" -lt "$REPEAT" ] && sleep "$GAP"
    return 0
}

IFS=, read -r -d '' -a tier_list < <(printf '%s\0' "$TIERS") || true
for tier in "${tier_list[@]}"; do
    [ -n "$tier" ] || continue
    run=1
    while [ "$run" -le "$REPEAT" ]; do
        run_arm "$tier" "$run"
        run=$((run + 1))
    done
done

echo ""
echo "| tier | run | load | ctx | KV | bank | ncmoe | PP t/s | TG t/s | peak MiB | held MiB | used MiB | RAM peak MiB | reallocs | hit % | status |"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
tail -n +2 "$CSV" | awk -F, '{printf "|"; for (i = 1; i <= 16; i++) printf " %s |", $i; print ""}'
