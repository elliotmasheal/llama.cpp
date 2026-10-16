# service.ps1 - keep the per-(hardware, model) tuning store up to date.
#
# Scans a model directory, tunes every (backend, model) pair that is not yet in the store, and
# re-tunes the stored ones on a slower cadence so the config improves as the machine/model are used.
# Run it in the background; it is safe to stop (all state is on disk in the store).
#
# Usage:
#   .\service.ps1                       # watch F:\LLM\llm, refresh every 24h
#   .\service.ps1 -ModelsDir D:\models -IntervalMin 60
param(
    [string]$ModelsDir = "F:\LLM\llm",
    [int]$IntervalMin  = 1440,
    [string]$RepoRoot  = "E:\fastLlama",
    [string]$StoreDir  = "$PSScriptRoot\store",
    [string]$Backend   = "Vulkan1",
    [switch]$Once
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $StoreDir | Out-Null
$log = Join-Path $StoreDir "service.log"

function Write-Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Encoding utf8 $log $line
}

function Model-Key([string]$p) {
    $gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "6700S" } | Select-Object -First 1).Name
    $g = $gpu -replace '[^A-Za-z0-9]+','_'
    $n = (Get-Item $p).BaseName -replace '[^A-Za-z0-9]+','_'
    "$g`__$n"
}

function Is-Tuned([string]$p) {
    Test-Path (Join-Path $StoreDir ((Model-Key $p) + ".json"))
}

Write-Log "service start: models='$ModelsDir' interval=${IntervalMin}m backend=$Backend"

while ($true) {
    $models = Get-ChildItem $ModelsDir -Filter *.gguf -ErrorAction SilentlyContinue
    foreach ($m in $models) {
        $key = Model-Key $m.FullName
        if (-not (Is-Tuned $m.FullName)) {
            Write-Log "tuning new model: $($m.Name)"
            try {
                & "$PSScriptRoot\tune.ps1" -Model $m.FullName -RepoRoot $RepoRoot -Backend $Backend -StoreDir $StoreDir |
                    ForEach-Object { Write-Log $_ }
            } catch { Write-Log "tune failed for $($m.Name): $_" }
        } else {
            Write-Log "already tuned: $($m.Name)"
        }
    }
    if ($Once) { break }
    Write-Log "sleep ${IntervalMin}m"
    Start-Sleep -Minutes $IntervalMin
}
