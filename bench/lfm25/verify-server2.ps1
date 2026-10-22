# verify-server2.ps1 - same as verify-server.ps1 but with extra server args as parameter.
param([string]$Extra = "", [int]$Port = 8124, [int]$Cache = 5120, [int]$Ncmoe = 36)
$ErrorActionPreference = "Continue"
$root = "E:\fastLlama"
$bin  = Join-Path $root "build-vulkan\bin\llama-server.exe"
$model = "F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
$out  = Join-Path $root "bench\lfm25\results"
$base = "-m $model -dev Vulkan1 -ngl 99 -ncmoe $Ncmoe --moe-cache-mib $Cache -fa on -t 6 -ub 2048 -b 2048 --port $Port --no-webui $Extra"
$srv = Start-Process -FilePath $bin -ArgumentList $base -NoNewWindow -PassThru `
        -RedirectStandardOutput "$out\server2.log" -RedirectStandardError "$out\server2.err"
$ok = $false; $deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq "ok") { $ok = $true; break } }
    catch { if ($srv.HasExited) { break } }
    Start-Sleep -Milliseconds 700
}
Write-Host "extra='$Extra' ready=$ok exited=$($srv.HasExited)"
if ($ok) {
    $prompt = (Get-Content (Join-Path $root "bench\lfm25\prompt512.txt") -Raw)
    $body = @{ prompt = $prompt; n_predict = 64; temperature = 0; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    Write-Host ("prompt {0} tok, {1:N1} t/s; predicted {2:N1} t/s ({3} tok)" -f $r.timings.prompt_n, $r.timings.prompt_per_second, $r.timings.predicted_per_second, $r.timings.predicted_n)
}
if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
Start-Sleep -Milliseconds 800
