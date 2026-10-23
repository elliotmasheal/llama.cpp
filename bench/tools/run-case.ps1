# run-case.ps1 - run one llama.cpp benchmark under resource monitoring.
#
# Example:
#   .\run-case.ps1 -Tag pp512_tg128_resident -Bin llama-bench `
#       -Args '-m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 29 -p 512 -n 128 -fa on -r 3 -o md'
#
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$Bin,      # llama-bench | llama-cli | test-backend-ops
    [Parameter(Mandatory = $true)][string]$Args,     # argument string (paths without spaces)
    [string]$RepoRoot   = "E:\fastLlama",
    [string]$ResultsDir = "",
    [string]$Env        = "",
    [int]$IntervalMs    = 1000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ResultsDir) { $ResultsDir = Join-Path $ScriptDir "results" }
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$csv = Join-Path $ResultsDir "$Tag.metrics.csv"
$log = Join-Path $ResultsDir "$Tag.bench.log"
$sum = Join-Path $ResultsDir "$Tag.summary.txt"
if (Test-Path "$csv.stop") { Remove-Item "$csv.stop" -Force }

# --- machine context -------------------------------------------------------
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "6700S" }) |
        Select-Object -First 1
$os  = Get-CimInstance Win32_OperatingSystem
$cs  = Get-CimInstance Win32_ComputerSystem
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
$disk = Get-PhysicalDisk | Where-Object { $_.DeviceID -ne $null } | Select-Object FriendlyName, MediaType, Size

$ctx = @(
    "tag            : $Tag",
    "bin            : $Bin",
    "args           : $Args",
    "gpu            : $($gpu.Name)  drv $($gpu.DriverVersion)",
    "cpu            : $($cpu.Name)  cores $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)",
    "ram            : $([math]::Round($cs.TotalPhysicalMemory/1GB,1)) GB",
    "os             : $($os.Caption) $($os.Version)",
    "date           : $(Get-Date -Format o)",
    "--- disks ---",
    ($disk | ForEach-Object { "  $($_.FriendlyName) $($_.MediaType) $([math]::Round($_.Size/1GB,0))GB" }) -join "`n"
) -join "`n"

Write-Host "=== $Tag ==="
Write-Host $ctx

# --- start monitor ---------------------------------------------------------
$job = Start-Job -FilePath "$ScriptDir\monitor.ps1" -ArgumentList $csv, $IntervalMs, "llama"
Start-Sleep -Milliseconds 800

# --- run benchmark ---------------------------------------------------------
$binPath = Join-Path $RepoRoot "build-vulkan\bin\$Bin.exe"
Write-Host "> $binPath $Args"
$ErrorActionPreference = "Continue"
if ($Env) {
    foreach ($kv in $Env -split ';') {
        $p = $kv -split '=', 2
        Set-Item -Path "Env:$($p[0])" -Value $p[1]
    }
}
$sw = [Diagnostics.Stopwatch]::StartNew()
cmd /c "`"$binPath`" $Args > `"$log`" 2>&1"
$exitCode = $LASTEXITCODE
$sw.Stop()

# --- stop monitor ----------------------------------------------------------
New-Item -ItemType File -Path "$csv.stop" -Force | Out-Null
Wait-Job $job -Timeout 30 | Out-Null
Remove-Job $job -Force
Get-Content $log -Tail 30
Write-Host "exit code: $exitCode"

# --- summarize -------------------------------------------------------------
$m = Import-Csv $csv
$dur = [math]::Round($sw.Elapsed.TotalSeconds, 1)
if ($m.Count -gt 0) {
    $stat = [ordered]@{
        "wall_s"           = $dur
        "samples"          = $m.Count
        "cpu_pct_avg"      = [math]::Round(($m | Measure-Object cpu_pct -Average).Average, 1)
        "cpu_pct_max"      = [math]::Round(($m | Measure-Object cpu_pct -Maximum).Maximum, 1)
        "mem_used_gb_peak" = [math]::Round(($m | Measure-Object mem_used_mb -Maximum).Maximum / 1024, 2)
        "disk_read_mbs_avg"= [math]::Round(($m | Measure-Object disk_read_mbs -Average).Average, 1)
        "disk_read_mbs_max"= [math]::Round(($m | Measure-Object disk_read_mbs -Maximum).Maximum, 1)
        "gpu_util_pct_avg" = [math]::Round(($m | Measure-Object gpu_util_pct -Average).Average, 1)
        "gpu_util_pct_max" = [math]::Round(($m | Measure-Object gpu_util_pct -Maximum).Maximum, 1)
        "gpu_ded_mb_peak"  = [math]::Round(($m | Measure-Object gpu_ded_mb -Maximum).Maximum, 0)
        "proc_ws_gb_peak"  = [math]::Round(($m | Measure-Object proc_ws_mb -Maximum).Maximum / 1024, 2)
    }
} else {
    $stat = @{ "wall_s" = $dur; "note" = "no samples" }
}

$report = $ctx + "`n--- resource summary ---`n" + (($stat.GetEnumerator() | ForEach-Object { "  {0,-20} {1}" -f $_.Key, $_.Value }) -join "`n")
$report | Set-Content -Encoding utf8 $sum
Write-Host $report
Write-Host "wrote: $csv`n       $log`n       $sum"
