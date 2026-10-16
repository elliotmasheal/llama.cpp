# fit-ubatch.ps1 - fit the single-pass PP model from llama-bench results.
#
# Model (single pass, prompt = ubatch = M):
#   T(M) = M / (A + B*M)          T in tokens/s, M in tokens
#   1/T  = B + A*(1/M)            -> linear in 1/M
# where:
#   A = weight_bytes / BW_eff     per-pass weight-load time (independent of M)
#   B = flops_per_token / FLOPS   per-token compute time
#   knee M* = A/B                 beyond it the pass is compute-bound
#
# Usage: .\fit-ubatch.ps1 -ResultsDir ..\results -Pattern "P3fit_M*" -WeightGiB 20.21 -ActiveParamsB 2.4
param(
    [string]$ResultsDir = "$PSScriptRoot\..\results",
    [string]$Pattern   = "P3fit_M*",
    [double]$WeightGiB = 20.21,   # bytes read per pass (all experts + dense)
    [double]$ActiveParamsB = 2.4, # active params per token (billions)
    [double]$PeakBW_GBs = 224.0,  # GPU peak bandwidth, for reporting efficiency
    [double]$PeakFLOPS_TF = 16.0, # GPU peak (fp16 packed) TFLOP/s, for reporting
    [int[]]$Skip = @()            # M values to exclude (e.g. anomalous ubatches)
)

$pts = @()
foreach ($log in Get-ChildItem $ResultsDir -Filter "$Pattern.bench.log") {
    if ($log.Name -notmatch 'M(\d+)') { continue }
    $M = [int]$Matches[1]
    # the PP row looks like: | ... | pp<M> |   321.10 +- 7.32 |
    $line = Select-String -Path $log.FullName -Pattern "\|\s+pp\d+\s+\|\s+([0-9.]+)" | Select-Object -First 1
    if (-not $line) {
        # llama-bench table row: find the numeric after the test column
        $line = Select-String -Path $log.FullName -Pattern "\|\s+pp\d+\s+\|" | Select-Object -First 1
    }
    $rx = [regex]::Match((Get-Content $log.FullName -Raw), "\|\s+pp\d+\s+\|\s+([0-9]+(?:\.[0-9]+)?)")
    if ($rx.Success) {
        $pts += [pscustomobject]@{ M = $M; T = [double]$rx.Groups[1].Value }
    }
}

if ($pts.Count -lt 2) { Write-Error "need >= 2 points, got $($pts.Count)"; exit 1 }

$pts = $pts | Where-Object { $Skip -notcontains $_.M } | Sort-Object M
Write-Host "points:"
$pts | ForEach-Object { "  M={0,5}  T={1,8:N2} t/s" -f $_.M, $_.T | Write-Host }

# least squares on y = B + A*x  with x = 1/M, y = 1/T
$xs = $pts | ForEach-Object { 1.0 / $_.M }
$ys = $pts | ForEach-Object { 1.0 / $_.T }
$n  = $xs.Count
$sx = ($xs | Measure-Object -Sum).Sum
$sy = ($ys | Measure-Object -Sum).Sum
$sxx = 0.0; $sxy = 0.0
for ($i = 0; $i -lt $n; $i++) { $sxx += $xs[$i]*$xs[$i]; $sxy += $xs[$i]*$ys[$i] }
$A = ($n*$sxy - $sx*$sy) / ($n*$sxx - $sx*$sx)   # slope = A
$B = ($sy - $A*$sx) / $n                          # intercept = B

$weightBytes = $WeightGiB * 1024 * 1024 * 1024
$flopsPerTok = 2.0 * $ActiveParamsB * 1e9

$BW_eff    = $weightBytes / $A / 1e9                  # GB/s
$FLOPS_eff = $flopsPerTok / $B / 1e12                 # TFLOP/s
$knee      = ($A / $B)

"`nfit:  1/T = B + A*(1/M)"
"  A (per-pass weight-load) = {0:N4} s  = {1:N1} ms" -f $A, ($A*1000)
"  B (per-token compute)    = {0:N6} s  = {1:N3} ms/token" -f $B, ($B*1000)
"  knee M* = A/B            = {0:N0} tokens" -f $knee
"  => BW_eff    = {0:N1} GB/s  ({1:N1}% of {2} GB/s peak)" -f $BW_eff, (100*$BW_eff/$PeakBW_GBs), $PeakBW_GBs
"  => FLOPS_eff = {0:N2} TFLOP/s ({1:N1}% of {2} TFLOP/s peak)" -f $FLOPS_eff, (100*$FLOPS_eff/$PeakFLOPS_TF), $PeakFLOPS_TF
""
"predicted PP at M:"
foreach ($m in 256,512,1024,2048,4096) {
    $t = $m / ($A + $B*$m)
    "  M={0,5} -> {1,8:N1} t/s" -f $m, $t
}
