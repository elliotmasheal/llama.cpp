# Runs one llama-bench invocation, serialized on the GPU.
# Usage: run_pp.ps1 -Tag <name> -Args "<flags>" [-Env "K=V;K2=V2"]
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][string]$Args,
    [string]$Env = ""
)

$dir  = "E:\fastLlama\bench\expert-tiering\pp-baseline"
$exe  = "E:\fastLlama\build-vulkan\bin\llama-bench.exe"
$out  = Join-Path $dir "$Tag.log"
$err  = Join-Path $dir "$Tag.log.err"

New-Item -ItemType Directory -Force -Path $dir | Out-Null

# GPU exclusivity: never start a second llama run
$busy = Get-Process -Name "llama*" -ErrorAction SilentlyContinue
if ($busy) {
    "ABORT: llama process already running: " + (($busy | ForEach-Object { $_.Id }) -join ",") | Out-File -FilePath (Join-Path $dir "$Tag.abort.txt") -Encoding ascii
    exit 2
}

# set env vars for this run only
$names = @()
if ($Env -ne "") {
    foreach ($pair in $Env.Split(';')) {
        if ($pair.Trim() -eq "") { continue }
        $kv = $pair.Split('=', 2)
        Set-Item -Path ("Env:" + $kv[0]) -Value $kv[1]
        $names += $kv[0]
    }
}

"# command: llama-bench.exe $Args" | Out-File -FilePath $out -Encoding ascii
if ($Env -ne "") { "# env: $Env" | Out-File -FilePath $out -Append -Encoding ascii }
"# started: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $out -Append -Encoding ascii

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$argList = $Args -split ' '
& $exe @argList 1>> $out 2>> $err
$code = $LASTEXITCODE
$sw.Stop()

"# exit: $code  wall: {0:N1}s" -f ($sw.Elapsed.TotalSeconds) | Out-File -FilePath $out -Append -Encoding ascii

# restore env
foreach ($n in $names) { Remove-Item -Path ("Env:" + $n) -ErrorAction SilentlyContinue }
exit $code
