# verify-server-official.ps1 - end-to-end llama-server check for the official scoop build.
param([int]$Port = 8130, [string]$Bin = "D:\Scoop\apps\llama.cpp-vulkan\current\llama-server.exe")
$ErrorActionPreference = "Continue"
$root = "E:\fastLlama"
$model = "F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
$out  = Join-Path $root "bench\lfm25\results"
$args = "-m $model -dev Vulkan1 -ngl 99 -ncmoe 22 -fa on -t 6 -ub 2048 -b 2048 -c 4096 --parallel 1 --no-webui -lm none --port $Port"
$srv = Start-Process -FilePath $Bin -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput "$out\server-official.log" -RedirectStandardError "$out\server-official.err"
$ok = $false; $deadline = (Get-Date).AddSeconds(150)
while ((Get-Date) -lt $deadline) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq "ok") { $ok = $true; break } }
    catch { if ($srv.HasExited) { break } }
    Start-Sleep -Milliseconds 700
}
Write-Host "official server ready=$ok exited=$($srv.HasExited)"
if ($ok) {
    $prompt = (Get-Content (Join-Path $root "bench\lfm25\prompt512.txt") -Raw)
    $body = @{ prompt = $prompt; n_predict = 64; temperature = 0; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    Write-Host ("prompt {0} tok, {1:N1} t/s; predicted {2:N1} t/s ({3} tok)" -f $r.timings.prompt_n, $r.timings.prompt_per_second, $r.timings.predicted_per_second, $r.timings.predicted_n)
}
if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
Start-Sleep -Milliseconds 800
