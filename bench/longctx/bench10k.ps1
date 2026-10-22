# 10k prompt harness: prefill + N-token decode at depth, interleaved repeats.
# usage: .\bench10k.ps1 -Runs 2 -N 256 -CacheMib 6144 -Ctx 16384 [-Tag base]
param(
    [string]$Prompt = "bench\longctx\prompt10k.txt",
    [int]$Ncmoe = 41,
    [int]$CacheMib = 6144,
    [int]$Ub = 2048,
    [int]$T = 5,
    [int]$Ctx = 16384,
    [int]$N = 256,
    [int]$Runs = 2,
    [int]$GapSec = 40,
    [string]$Tag = "run"
)

$env:PATH = "/d/Scoop/apps/mingw/current/bin;$env:PATH"
$model = "F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf"
$exe = "build-vulkan\bin\llama-cli.exe"
$log = "bench\longctx\10k-$Tag.log"
if (Test-Path $log) { Remove-Item $log }

for ($i = 1; $i -le $Runs; $i++) {
    $a = @("-m", $model, "-dev", "Vulkan1", "-ngl", "99", "-fa", "on", "-t", $T,
           "-ncmoe", $Ncmoe, "-c", $Ctx, "-f", $Prompt, "-n", $N,
           "--temp", "0", "--no-warmup", "--no-display-prompt", "-st", "-lm", "none")
    if ($CacheMib -gt 0) { $a += @("--moe-cache-mib", $CacheMib) }
    if ($Ub -gt 0) { $a += @("-ub", $Ub) }

    $out = "bench\longctx\tmp-$Tag.out"
    $err = "bench\longctx\tmp-$Tag.err"
    & $exe @a 1> $out 2> $err
    $line = (Get-Content $out | Select-String "Prompt:").Line
    Add-Content $log ("run{0} -> {1}" -f $i, $line)
    Write-Host ("run{0} -> {1}" -f $i, $line)
    if ($i -lt $Runs) { Start-Sleep -Seconds $GapSec }
}
