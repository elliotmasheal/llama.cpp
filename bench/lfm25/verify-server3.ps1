# verify-server3.ps1 - generic llama-server end-to-end check.
param(
    [string]$Bin = "E:\fastLlama\build-vulkan\bin\llama-server.exe",
    [string]$Args = "-dev Vulkan1 -ngl 99 -ncmoe 36 --moe-cache-mib 5120 -fa on -t 6",
    [int]$Port = 8140,
    [int]$Npredict = 128,
    [string]$Tag = "server"
)
$ErrorActionPreference = "Continue"
$root = "E:\fastLlama"
$model = "F:\LLM\llm\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
$out = Join-Path $root "bench\lfm25\results"
$all = "-m $model $Args -ub 2048 -b 2048 -c 4096 --parallel 1 --no-webui --port $Port"
$srv = Start-Process -FilePath $Bin -ArgumentList $all -NoNewWindow -PassThru `
        -RedirectStandardOutput "$out\$Tag.log" -RedirectStandardError "$out\$Tag.err"
$ok = $false; $deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq "ok") { $ok = $true; break } }
    catch { if ($srv.HasExited) { break } }
    Start-Sleep -Milliseconds 700
}
Write-Host "tag=$Tag ready=$ok exited=$($srv.HasExited)"
if ($ok) {
    $prompt = (Get-Content (Join-Path $root "bench\lfm25\prompt512.txt") -Raw)
    $body = @{ prompt = $prompt; n_predict = $Npredict; temperature = 0; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    $line = ("tag={0} prompt {1} tok, {2:N1} t/s; predicted {3:N1} t/s ({4} tok)" -f $Tag, $r.timings.prompt_n, $r.timings.prompt_per_second, $r.timings.predicted_per_second, $r.timings.predicted_n)
    Write-Host $line
    Add-Content -Path "$out\server-compare.txt" -Value $line
}
if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
Start-Sleep -Milliseconds 800
