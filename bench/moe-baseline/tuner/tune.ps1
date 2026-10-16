# tune.ps1 - measure and fit the best launch config for a (hardware, model) pair, and persist it.
#
# Order:
#   1. residency sweep (-ncmoe) at a small ubatch -> best ncmoe by whole-turn wall time
#   2. PP(ubatch) single-pass sweep at that ncmoe -> fit T(M)=M/(A+B*M) -> knee M* -> best ubatch
#   3. re-measure the chosen config
# The result is stored as JSON keyed by (gpu, model); re-running refines the entry.
#
# Usage:
#   .\tune.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$RepoRoot = "E:\fastLlama",
    [string]$Backend  = "Vulkan1",
    [object]$UbatchGrid = "256,512,1024,1536,2048",
    [object]$NcmoeGrid  = "41,33,29,27,25",
    [int]$Reps = 2,
    [string]$StoreDir = "",
    [int]$ResidencyPrompt = 2048,
    [int]$ResidencyGen    = 128
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $StoreDir) { $StoreDir = Join-Path $ScriptDir "store" }
function Parse-Grid($v) {
    @(@($v) | ForEach-Object { "$_" } | ForEach-Object { [regex]::Split($_, '[, ]+') } |
      Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
}
$UbatchGrid = Parse-Grid $UbatchGrid
$NcmoeGrid  = Parse-Grid $NcmoeGrid

$bin = Join-Path $RepoRoot "build-vulkan\bin\llama-bench.exe"
if (-not (Test-Path $bin)) { throw "llama-bench not found: $bin" }
New-Item -ItemType Directory -Force -Path $StoreDir | Out-Null
$ErrorActionPreference = "Continue"   # native tools write progress to stderr; do not abort on that

function Invoke-Bench([int]$pp, [int]$tg, [int]$ub, [int]$ncmoe) {
    $a = @("-m", $Model, "-dev", $Backend, "-ngl", "99", "-fa", "on", "-o", "json",
           "-r", "$Reps", "--offline", "-p", "$pp", "-n", "$tg")
    if ($ub -gt 0)     { $a += @("-ub", "$ub") }
    if ($ncmoe -ge 0)  { $a += @("-ncmoe", "$ncmoe") }
    $out = & $bin @a 2>$null
    $txt = ($out -join "`n")
    $s = $txt.IndexOf('['); $e = $txt.LastIndexOf(']')
    if ($s -lt 0 -or $e -lt 0) { throw "no json from llama-bench" }
    return ($txt.Substring($s, $e - $s + 1) | ConvertFrom-Json -ErrorAction Stop)
}

# ---- identity --------------------------------------------------------------
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "6700S" } | Select-Object -First 1)
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$mi  = Get-Item $Model
$modelId   = "{0}__{1}" -f (($gpu.Name -replace '[^A-Za-z0-9]+','_')), (($mi.BaseName -replace '[^A-Za-z0-9]+','_'))
$storePath = Join-Path $StoreDir "$modelId.json"

Write-Host "ubatch grid ($($UbatchGrid.Count)): $($UbatchGrid -join ' ')"
Write-Host "ncmoe  grid ($($NcmoeGrid.Count)): $($NcmoeGrid -join ' ')"
Write-Host "=== tune $($mi.BaseName) on $($gpu.Name) ==="

# ---- 1. residency sweep at a small ubatch ---------------------------------
Write-Host "`n-- residency sweep (ub 512) --"
$res = @()
foreach ($nc in ($NcmoeGrid | Sort-Object -Descending)) {
    try {
        $r  = Invoke-Bench -pp $ResidencyPrompt -tg $ResidencyGen -ub 512 -ncmoe $nc
        $pp = ($r | Where-Object { $_.n_prompt -eq $ResidencyPrompt } | Select-Object -First 1)
        $tg = ($r | Where-Object { $_.n_gen    -eq $ResidencyGen    } | Select-Object -First 1)
        if ($pp -and $tg) {
            $res += [pscustomobject]@{ ncmoe = $nc; pp = [double]$pp.avg_ts; tg = [double]$tg.avg_ts }
            Write-Host ("  ncmoe {0,3} -> PP {1,7:N1}  TG {2,6:N2}" -f $nc, [double]$pp.avg_ts, [double]$tg.avg_ts)
        }
    } catch { Write-Host "  ncmoe $nc failed: $_" }
}
$bestNcmoe = 0
if ($res.Count -gt 0) {
    $b = ($res | Sort-Object { $ResidencyPrompt/$_.pp + $ResidencyGen/$_.tg } | Select-Object -First 1)
    $bestNcmoe = $b.ncmoe
    Write-Host ("  -> best ncmoe = {0}" -f $bestNcmoe)
}

# ---- 2. ubatch single-pass sweep at the best ncmoe, then fit ---------------
Write-Host "`n-- ubatch sweep (ncmoe $bestNcmoe) --"
$pts = @()
foreach ($M in ($UbatchGrid | Sort-Object)) {
    try {
        $r   = Invoke-Bench -pp $M -tg 0 -ub $M -ncmoe $bestNcmoe
        $row = ($r | Where-Object { $_.n_prompt -eq $M } | Select-Object -First 1)
        if ($row) {
            $pts += [pscustomobject]@{ M = $M; T = [double]$row.avg_ts }
            Write-Host ("  ubatch {0,5} -> PP {1,8:N1} t/s" -f $M, [double]$row.avg_ts)
        }
    } catch { Write-Host "  ubatch $M failed: $_" }
}

$fit = $null; $ubBest = $null
if ($pts.Count -ge 2) {
    $xs = $pts | ForEach-Object { 1.0 / $_.M }
    $ys = $pts | ForEach-Object { 1.0 / $_.T }
    $n = $xs.Count
    $sx = ($xs | Measure-Object -Sum).Sum; $sy = ($ys | Measure-Object -Sum).Sum
    $sxx = 0.0; $sxy = 0.0
    for ($i = 0; $i -lt $n; $i++) { $sxx += $xs[$i]*$xs[$i]; $sxy += $xs[$i]*$ys[$i] }
    $den = ($n*$sxx - $sx*$sx)
    if ([math]::Abs($den) -gt 1e-12) {
        $A = ($n*$sxy - $sx*$sy) / $den
        $B = ($sy - $A*$sx) / $n
        $weightBytes = $mi.Length
        $knee = if ($B -gt 0 -and $A -gt 0) { $A / $B } else { 0 }
        $fit = [pscustomobject]@{
            A_s = [math]::Round($A, 4); B_s_per_tok = [math]::Round($B, 6)
            knee_tokens = [math]::Round($knee); BW_eff_gbs = if ($A -gt 0) { [math]::Round($weightBytes/$A/1e9, 1) } else { $null }
        }
        Write-Host ("  fit: A={0:N4}s  B={1:N6}s/tok  knee={2:N0} tok  BW_eff={3} GB/s" -f $A, $B, $knee, $fit.BW_eff_gbs)
        # recommended ubatch: the smallest grid value >= knee; else the largest that ran
        $ran = $pts | Sort-Object M
        $cand = $ran | Where-Object { $knee -gt 0 -and $_.M -ge $knee } | Sort-Object M
        if ($cand.Count -eq 0) { $cand = $ran }
        $ubBest = ($cand | Select-Object -Last 1).M
        Write-Host "  -> recommended ubatch = $ubBest"
    }
}
if (-not $ubBest -and $pts.Count -ge 1) { $ubBest = ($pts | Sort-Object T | Select-Object -Last 1).M }

# ---- 2b. refine residency at the chosen ubatch (ubatch and ncmoe interact) --
if ($ubBest -and $NcmoeGrid.Count -gt 1) {
    Write-Host "`n-- residency refine (ub $ubBest) --"
    $res2 = @()
    foreach ($nc in ($NcmoeGrid | Sort-Object -Descending)) {
        try {
            $r  = Invoke-Bench -pp $ResidencyPrompt -tg $ResidencyGen -ub $ubBest -ncmoe $nc
            $pp = ($r | Where-Object { $_.n_prompt -eq $ResidencyPrompt } | Select-Object -First 1)
            $tg = ($r | Where-Object { $_.n_gen    -eq $ResidencyGen    } | Select-Object -First 1)
            if ($pp -and $tg) {
                $res2 += [pscustomobject]@{ ncmoe = $nc; pp = [double]$pp.avg_ts; tg = [double]$tg.avg_ts }
                Write-Host ("  ncmoe {0,3} -> PP {1,7:N1}  TG {2,6:N2}" -f $nc, [double]$pp.avg_ts, [double]$tg.avg_ts)
            }
        } catch { Write-Host "  ncmoe $nc failed: $_" }
    }
    if ($res2.Count -gt 0) {
        $b = ($res2 | Sort-Object { $ResidencyPrompt/$_.pp + $ResidencyGen/$_.tg } | Select-Object -First 1)
        $bestNcmoe = $b.ncmoe
        $res = $res2
        Write-Host ("  -> refined ncmoe = {0}" -f $bestNcmoe)
    }
}

# ---- 3. final measure of the chosen config --------------------------------
$bestPP = $null; $bestTG = $null
if ($ubBest) {
    try {
        $r  = Invoke-Bench -pp $ResidencyPrompt -tg $ResidencyGen -ub $ubBest -ncmoe $bestNcmoe
        $bestPP = [double](($r | Where-Object { $_.n_prompt -eq $ResidencyPrompt } | Select-Object -First 1).avg_ts)
        $bestTG = [double](($r | Where-Object { $_.n_gen    -eq $ResidencyGen    } | Select-Object -First 1).avg_ts)
        Write-Host ("`n  final: PP {0:N1}  TG {1:N2}" -f $bestPP, $bestTG)
    } catch { Write-Host "  final measure failed: $_" }
}

# ---- persist ---------------------------------------------------------------
$entry = [ordered]@{
    hardware  = [ordered]@{ gpu = $gpu.Name; backend = $Backend; cpu = $cpu.Name; ram_gb = $ram; updated = (Get-Date -Format o) }
    model     = [ordered]@{ path = $Model; name = $mi.BaseName; size_gib = [math]::Round($mi.Length/1GB, 2) }
    pp_fit    = $fit
    pp_points = $pts
    best      = [ordered]@{ ubatch = $ubBest; ncmoe = $bestNcmoe; pp = $bestPP; tg = $bestTG }
    residency = $res
}
$entry | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $storePath
Write-Host "`nsaved: $storePath"

$argline = "-dev $Backend -ngl 99 -fa on"
if ($ubBest) { $argline += " -ub $ubBest" }
$argline += " -ncmoe $bestNcmoe"
Write-Host "recommended: $argline"
