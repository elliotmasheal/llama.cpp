# Reference logits dump via llama-perplexity (fork side of the quality gate).
# usage: powershell -File bench/tools/quality/reference.ps1 -Out bench/raw/x-1/ref.bin [-Threads 5]
param(
    [string]$Model = "F:\LLM\llm\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    [string]$Prompt = "bench\prompts\prompt10k.txt",
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$Chunks = 2,
    [int]$Threads = 5,
    [int]$Ncmoe = 41,
    [string]$Fa = "on",
    [string]$Out = "bench\raw\x-1\ref.bin",
    [string]$Log = "",
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Continue"
$env:PATH = "/d/Scoop/apps/mingw/current/bin;$env:PATH"
$exe = "build-vulkan\bin\llama-perplexity.exe"
if (-not (Test-Path $exe)) { throw "missing $exe" }
if (-not $Log) { $Log = [System.IO.Path]::ChangeExtension($Out, ".log") }
$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

$args = @("-m", $Model, "-dev", "Vulkan1", "-ngl", "99", "-fa", $Fa,
          "-t", $Threads, "-ncmoe", $Ncmoe, "-c", $Ctx, "-b", $Batch,
          "--chunks", $Chunks, "-f", $Prompt, "--save-all-logits", $Out,
          "-lm", "none") + $Extra

Write-Host ("run: {0} {1}" -f $exe, ($args -join " "))
$t0 = Get-Date
# stderr must go to the log with the stdout lines, so go through cmd: PS 5.1
# turns native stderr into ErrorRecords and can abort the script
$cmdline = ('"{0}" {1} > "{2}" 2>&1' -f $exe, ($args -join " "), $Log)
& cmd /c $cmdline
$rc = $LASTEXITCODE
$sec = ((Get-Date) - $t0).TotalSeconds
$size = if (Test-Path $Out) { (Get-Item $Out).Length } else { 0 }
Write-Host ("exit {0}, {1:N1} s, dump {2:N1} MiB -> {3}" -f $rc, $sec, ($size / 1MB), $Out)
Get-Content $Log | Select-String "Final estimate|PPL|calculating perplexity" | ForEach-Object { $_.Line }
if ($rc -ne 0) { exit $rc }
