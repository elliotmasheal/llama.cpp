# verify-server.ps1 - start llama-server with the recommended config, send one request, print timings.
$ErrorActionPreference = "Continue"
$root = "E:\fastLlama"
$bin  = Join-Path $root "build-vulkan\bin\llama-server.exe"
$model = "F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
$out  = Join-Path $root "bench\lfm25\results"
$port = 8123
$args = "-m $model -dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6 -ub 2048 -b 2048 -c 8192 --port $port --no-webui"
$srv = Start-Process -FilePath $bin -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput "$out\server.log" -RedirectStandardError "$out\server.err"
Write-Host "server pid $($srv.Id), waiting for readiness..."
$ok = $false
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {
        if ($srv.HasExited) { break }
    }
    Start-Sleep -Milliseconds 700
}
Write-Host "ready=$ok exited=$($srv.HasExited)"
if ($ok) {
    $prompt = (Get-Content (Join-Path $root "bench\lfm25\prompt512.txt") -Raw)
    $body = @{ prompt = $prompt; n_predict = 32; temperature = 0; cache_prompt = $true } | ConvertTo-Json
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    $sw.Stop()
    Write-Host "wall=$([math]::Round($sw.Elapsed.TotalSeconds,2))s"
    Write-Host ("prompt tokens {0}, prompt {1:N2} t/s, predicted {2:N2} t/s" -f `
        $r.timings.prompt_n, $r.timings.prompt_per_second, $r.timings.predicted_per_second)
    Write-Host ("sample: " + ($r.content.Substring(0, [Math]::Min(160, $r.content.Length))))
}
if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
Write-Host "--- server.err tail ---"
Get-Content "$out\server.err" -Tail 8
