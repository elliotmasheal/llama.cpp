# monitor.ps1 - sample Windows resource counters while a benchmark runs.
# Writes one CSV row per interval until a "<Out>.stop" file appears.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$IntervalMs = 1000,
    [string]$ProcName = "llama"
)

$ErrorActionPreference = "Stop"

$counters = @(
    "\Processor Information(_Total)\% Processor Time",
    "\Memory\Available MBytes",
    "\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    "\PhysicalDisk(_Total)\Disk Write Bytes/sec",
    "\GPU Engine(*)\Utilization Percentage",
    "\GPU Adapter Memory(*)\Dedicated Usage",
    "\GPU Adapter Memory(*)\Shared Usage"
)

$totalMb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB, 0)
$nproc   = [Environment]::ProcessorCount

"t_ms,cpu_pct,mem_avail_mb,mem_used_mb,disk_read_mbs,disk_write_mbs,gpu_util_pct,gpu_ded_mb,gpu_shared_mb,proc_cpu_pct,proc_ws_mb" |
    Set-Content -Encoding ascii $Out

$stop = "$Out.stop"
if (Test-Path $stop) { Remove-Item $stop -Force }

function Sum-ByPath($samples, $pattern) {
    $v = ($samples | Where-Object { $_.Path -like $pattern }).CookedValue
    if ($null -eq $v) { return 0.0 }
    return [double](($v | Measure-Object -Sum).Sum)
}

function One-ByPath($samples, $pattern) {
    $s = $samples | Where-Object { $_.Path -like $pattern } | Select-Object -First 1
    if ($null -eq $s) { return 0.0 }
    return [double]$s.CookedValue
}

$sw = [Diagnostics.Stopwatch]::StartNew()
$prevCpu = 0.0
$prevT   = 0.0
$prevWall = 0.0

while (-not (Test-Path $stop)) {
    try {
        $s = Get-Counter -Counter $counters -MaxSamples 1 -ErrorAction Stop
        $cs = $s.CounterSamples

        $cpu   = One-ByPath $cs "*processor information(_total)\% processor time"
        $avail = One-ByPath $cs "*\memory\available mbytes"
        $dr    = One-ByPath $cs "*physicaldisk(_total)\disk read bytes/sec"
        $dw    = One-ByPath $cs "*physicaldisk(_total)\disk write bytes/sec"
        $gpu   = Sum-ByPath $cs "*gpu engine*utilization percentage"
        $gded  = Sum-ByPath $cs "*gpu adapter memory*\dedicated usage"
        $gshr  = Sum-ByPath $cs "*gpu adapter memory*\shared usage"

        $procCpu = 0.0; $procWs = 0.0
        $p = Get-Process | Where-Object { $_.ProcessName -like "$ProcName*" } |
                Sort-Object WorkingSet64 -Descending | Select-Object -First 1
        if ($p) {
            $procWs = [math]::Round($p.WorkingSet64 / 1MB, 1)
            $now = $sw.Elapsed.TotalSeconds
            $procCpu = [math]::Round((($p.CPU - $prevCpu) / ($now - $prevWall)) * 100.0 / $nproc, 1)
            $prevCpu = $p.CPU
            $prevWall = $now
        }

        $t = $sw.ElapsedMilliseconds
        $row = [string]::Format([Globalization.CultureInfo]::InvariantCulture,
            "{0},{1:0.0},{2:0},{3:0},{4:0.0},{5:0.0},{6:0.0},{7:0},{8:0},{9:0.0},{10:0.0}",
            $t, $cpu, $avail, ($totalMb - $avail), ($dr / 1MB), ($dw / 1MB), $gpu, ($gded / 1MB), ($gshr / 1MB), $procCpu, $procWs)
        Add-Content -Encoding ascii $Out $row
    } catch {
        # counters can briefly fail while the GPU is busy; skip the sample
    }
    Start-Sleep -Milliseconds $IntervalMs
}
