# 10k prompt config matrix: one line per config, with peak device VRAM.
# usage: .\matrix.ps1 -Spec "5120/32768/2048;4096/16384/2048" [-EnvPairs "LLAMA_MOE_PREFETCH=0"]
# spec entry = cacheMib/ctx/ub
param(
    [string]$Prompt = "bench\prompts\prompt10k.txt",
    [int]$Ncmoe = 41,
    [int]$N = 32,
    [int]$T = 5,
    [string]$Spec = "5120/32768/2048",
    [string]$EnvPairs = "",
    [string]$Tag = "matrix"
)

$env:PATH = "/d/Scoop/apps/mingw/current/bin;$env:PATH"
$model = "F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf"
$exe = "build-vulkan\bin\llama-cli.exe"
$log = "bench\raw\$Tag.log"
if (Test-Path $log) { Remove-Item $log }

$envs = @{}
if ($EnvPairs -ne "") {
    foreach ($kv in $EnvPairs.Split(';')) {
        if ($kv -eq "") { continue }
        $p = $kv.Split('=')
        $envs[$p[0]] = $p[1]
    }
}

foreach ($cfg in $Spec.Split(';')) {
    if ($cfg -eq "") { continue }
    $f = $cfg.Split('/')
    $cache = [int]$f[0]; $ctx = [int]$f[1]; $ub = [int]$f[2]

    foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
    $env:GGML_VK_MEMORY_LOGGER = "1"
    $a = @("-m", $model, "-dev", "Vulkan1", "-ngl", "99", "-fa", "on", "-t", $T,
           "-ncmoe", $Ncmoe, "-c", $ctx, "-f", $Prompt, "-n", $N,
           "--temp", "0", "--no-warmup", "--no-display-prompt", "-st", "-lm", "none")
    if ($cache -gt 0) { $a += @("--moe-cache-mib", $cache) }
    if ($ub -gt 0) { $a += @("-ub", $ub) }

    $out = "bench\raw\tmp-$Tag.out"
    $err = "bench\raw\tmp-$Tag.err"
    & $exe @a 1> $out 2> $err
    $env:GGML_VK_MEMORY_LOGGER = $null
    foreach ($k in $envs.Keys) { Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue }

    $line = (Get-Content $out | Select-String "Prompt:").Line
    $line = $line -replace '^.*\[ Prompt:', 'pp/gen -> ['
    $peak = 0
    foreach ($l in Get-Content $err) {
        if ($l -match "Total device: ([\d.]+) (KiB|MiB|GiB)") {
            $v = [double]$matches[1]; $u = $matches[2]
            if ($u -eq "KiB") { $v /= 1024 } elseif ($u -eq "GiB") { $v *= 1024 }
            if ($v -gt $peak) { $peak = $v }
        }
    }
    $msg = "cache=$cache ctx=$ctx ub=$ub env=[$EnvPairs] -> $line peak=$([math]::Round($peak,0))MiB"
    Add-Content $log $msg
    Write-Host $msg
    Start-Sleep -Seconds 30
}
