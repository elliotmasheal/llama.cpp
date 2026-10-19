# long-context runner: prompt-eval and decode t/s at depth, with peak device VRAM
# usage: .\run.ps1 -Prompt bench\longctx\prompt10k.txt -Ncmoe 41 -CacheMib 5120 -Ub 2048 -T 5 -Ctx 32768 -N 48
param(
    [string]$Prompt = "bench\longctx\prompt10k.txt",
    [int]$Ncmoe = 41,
    [int]$CacheMib = 5120,
    [int]$Ub = 2048,
    [int]$T = 5,
    [int]$Ctx = 32768,
    [int]$N = 48,
    [switch]$Vram
)

$model = "F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf"
$exe = "build\bin\llama-cli.exe"

$a = @("-m", $model, "-dev", "Vulkan1", "-ngl", "99", "-fa", "on", "-t", $T, "-ncmoe", $Ncmoe,
       "-c", $Ctx, "-f", $Prompt, "-n", $N, "--temp", "0", "--no-warmup", "--no-display-prompt", "-st")
if ($CacheMib -gt 0) { $a += @("--moe-cache-mib", $CacheMib) }
if ($Ub -gt 0) { $a += @("-ub", $Ub) }

$out = "bench\longctx\run-$([System.IO.Path]::GetFileNameWithoutExtension($Prompt))-$Ncmoe-$CacheMib-$Ctx.out"
$err = $out -replace '\.out$', '.err'
if ($Vram) { $env:GGML_VK_MEMORY_LOGGER = "1" }
& $exe @a 1> $out 2> $err
$env:GGML_VK_MEMORY_LOGGER = $null

$line = (Get-Content $out | Select-String "Prompt:").Line
$peak = 0
foreach ($l in Get-Content $err) {
    if ($l -match "Total device: ([\d.]+) (KiB|MiB|GiB)") {
        $v = [double]$matches[1]; $u = $matches[2]
        if ($u -eq "KiB") { $v /= 1024 } elseif ($u -eq "GiB") { $v *= 1024 }
        if ($v -gt $peak) { $peak = $v }
    }
}
"{0} ncmoe={1} cache={2} ctx={3} ub={4} t={5} -> {6} peak={7} MiB" -f `
    [System.IO.Path]::GetFileName($Prompt), $Ncmoe, $CacheMib, $Ctx, $Ub, $T, $line, [math]::Round($peak, 0)
