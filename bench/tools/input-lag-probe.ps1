# input-lag-probe.ps1 - sample the counters that explain desktop input lag while a run is active.
# One CSV row per second; stops after -Seconds, or when "<Out>.stop" appears (0 = run until stop file).
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$IntervalMs = 1000,
    [int]$Seconds = 0,
    [string]$ProcName = "llama"
)

$ErrorActionPreference = "Stop"

$counters = @(
    "\Processor(_Total)\% Processor Time",
    "\Processor(_Total)\% DPC Time",
    "\Processor(_Total)\% Interrupt Time",
    "\Processor(_Total)\Interrupts/sec",
    "\Processor(_Total)\DPCs Queued/sec",
    "\Processor Information(_Total)\% Processor Performance",
    "\Processor Information(_Total)\Processor Frequency",
    "\Memory\Available MBytes",
    "\Memory\Committed Bytes",
    "\Memory\Pages/sec",
    "\Memory\Page Reads/sec",
    "\PhysicalDisk(_Total)\Current Disk Queue Length",
    "\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    "\PhysicalDisk(_Total)\Disk Write Bytes/sec",
    "\GPU Engine(*)\Utilization Percentage",
    "\GPU Adapter Memory(*)\Dedicated Usage",
    "\GPU Adapter Memory(*)\Shared Usage"
)

$header = "t,cpu_pct,cpu_perf_pct,cpu_mhz,dpc_pct,irq_pct,irq_sec,dpcq_sec,avail_mb,committed_mb,pages_sec,page_reads_sec,disk_q,disk_rb_mbs,disk_wb_mbs,gpu_compute_pct,gpu_copy_pct,gpu_ded_mb,gpu_shared_mb,proc_n,proc_cpu_s"
$header | Set-Content -Encoding ascii $Out

$stop = "$Out.stop"
if (Test-Path $stop) { Remove-Item $stop -Force }

$script:cs = @()
function Sample([string]$pattern) {
    $x = $script:cs | Where-Object { $_.Path -like $pattern } | Select-Object -First 1
    if ($null -eq $x) { return 0.0 }
    return [double]$x.CookedValue
}
function SampleSum([string]$pattern) {
    $x = $script:cs | Where-Object { $_.Path -like $pattern } | Measure-Object CookedValue -Sum
    if ($null -eq $x.Sum) { return 0.0 }
    return [double]$x.Sum
}

$deadline = if ($Seconds -gt 0) { (Get-Date).AddSeconds($Seconds) } else { [datetime]::MaxValue }
$dpc_max = 0.0
$rows = 0

while ((Get-Date) -lt $deadline -and -not (Test-Path $stop)) {
    try {
        $script:cs = (Get-Counter -Counter $counters -MaxSamples 1 -ErrorAction Stop).CounterSamples

        $procs = @(Get-Process -Name ($ProcName + "*") -ErrorAction SilentlyContinue)
        $proc_cpu = 0.0
        if ($procs.Count -gt 0) { $proc_cpu = ($procs | Measure-Object -Property CPU -Sum).Sum }

        $dpc = [math]::Round((Sample "*processor(_total)\% dpc time"), 2)
        if ($dpc -gt $dpc_max) { $dpc_max = $dpc }

        $row = @(
            (Get-Date).ToString("HH:mm:ss"),
            [math]::Round((Sample "*processor(_total)\% processor time"), 1),
            [math]::Round((Sample "*processor information(_total)\% processor performance"), 1),
            [int](Sample "*processor information(_total)\processor frequency"),
            $dpc,
            [math]::Round((Sample "*processor(_total)\% interrupt time"), 2),
            [int](Sample "*processor(_total)\interrupts/sec"),
            [int](Sample "*processor(_total)\dpcs queued/sec"),
            [int](Sample "*\memory\available mbytes"),
            [int]((Sample "*\memory\committed bytes") / 1MB),
            [int](Sample "*\memory\pages/sec"),
            [int](Sample "*\memory\page reads/sec"),
            [math]::Round((Sample "*physicaldisk(_total)\current disk queue length"), 1),
            [math]::Round((Sample "*physicaldisk(_total)\disk read bytes/sec") / 1MB, 1),
            [math]::Round((Sample "*physicaldisk(_total)\disk write bytes/sec") / 1MB, 1),
            [math]::Round((SampleSum "*gpu engine*engtype_compute*utilization percentage"), 1),
            [math]::Round((SampleSum "*gpu engine*engtype_copy*utilization percentage"), 1),
            [int]((SampleSum "*gpu adapter memory*dedicated usage") / 1MB),
            [int]((SampleSum "*gpu adapter memory*shared usage") / 1MB),
            $procs.Count,
            [math]::Round($proc_cpu, 0)
        ) -join ","
        Add-Content -Path $Out -Value $row -Encoding ascii
        $rows++
    } catch {
        Add-Content -Path $Out -Value ((Get-Date).ToString("HH:mm:ss") + ",ERR") -Encoding ascii
    }
    Start-Sleep -Milliseconds $IntervalMs
}

# tail summary: input lag shows up as high DPC% and a low processor performance percentage
"rows = $rows, max dpc_pct = $dpc_max"
