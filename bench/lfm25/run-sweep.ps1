# run-sweep.ps1 - run a list of llama-bench configs serially, one CSV per case.
# case file lines:  <tag> | <extra args>
param(
    [Parameter(Mandatory = $true)][string]$CaseFile,
    [string]$Bin       = "E:\fastLlama\build-vulkan\bin\llama-bench.exe",
    [string]$Model     = "F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf",
    [string]$OutDir    = "E:\fastLlama\bench\lfm25\results",
    [string]$Common    = "-dev Vulkan1",
    [int]$Reps         = 2,
    [int]$WarmupMin    = 0
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "sweep-progress.log"
$cases = Get-Content $CaseFile | Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*#' }
$n = 0
foreach ($line in $cases) {
    $n++
    $parts = $line -split '\|', 2
    $tag = $parts[0].Trim()
    $extra = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
    $csv = Join-Path $OutDir "$tag.csv"
    $err = Join-Path $OutDir "$tag.err"
    $args = "-m `"$Model`" $Common $extra -r $Reps -o csv"
    Write-Host "[$n/$($cases.Count)] $tag"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Bin -ArgumentList $args -RedirectStandardOutput $csv -RedirectStandardError $err -NoNewWindow -PassThru -Wait
    $sw.Stop()
    $ok = ($p.ExitCode -eq 0) -and (Test-Path $csv) -and ((Get-Item $csv).Length -gt 20)
    "$tag | exit=$($p.ExitCode) | wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s | ok=$ok | args=$args" | Add-Content -Path $log
    if (-not $ok) {
        $tail = (Get-Content $err -Tail 5) -join " / "
        Write-Host "  FAILED: $tail"
    }
}
Write-Host "done: $n cases"
