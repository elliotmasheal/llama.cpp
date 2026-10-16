# best.ps1 - print the stored best launch args for a model (or all stored models).
#
# Usage:
#   .\best.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
#   .\best.ps1                 # list every tuned model
param(
    [string]$Model = "",
    [string]$StoreDir = "$PSScriptRoot\store"
)

if (-not (Test-Path $StoreDir)) { Write-Host "no store yet ($StoreDir)"; return }

$entries = Get-ChildItem $StoreDir -Filter *.json | ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json }
if ($Model) { $entries = $entries | Where-Object { $_.model.path -eq $Model -or $_.model.name -eq (Get-Item $Model).BaseName } }

foreach ($e in $entries) {
    $a = "-dev $($e.hardware.backend) -ngl 99 -fa on"
    if ($e.best.ubatch) { $a += " -ub $($e.best.ubatch)" }
    if ($e.best.ncmoe)  { $a += " -ncmoe $($e.best.ncmoe)" }
    "{0}  [{1}]  PP~{2} TG~{3}" -f $a, $e.model.name, $e.best.pp, $e.best.tg
    if ($e.pp_fit) { "    fit: knee={0} tok  BW_eff={1} GB/s" -f $e.pp_fit.knee_tokens, $e.pp_fit.BW_eff_gbs }
}
