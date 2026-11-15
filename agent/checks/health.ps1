# The machine's standing: what is compared from run to run first, then,
# after the `---` line, the figures that move every time.
$ErrorActionPreference = 'SilentlyContinue'

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

Write-Output "Machine: $($cs.Name), $($os.Caption) build $($os.BuildNumber)"
Write-Output "Processor: $($cpu.Name.Trim()), $($cs.NumberOfLogicalProcessors) logical processors"
Write-Output ("Memory fitted: {0:N0} GB" -f ($cs.TotalPhysicalMemory / 1GB))

$pending = @()
if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') { $pending += 'Windows Update' }
if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') { $pending += 'component servicing' }
if ($pending.Count -gt 0) {
    Write-Output "Restart pending: yes ($($pending -join ', '))"
} else {
    Write-Output 'Restart pending: no'
}

Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Sort-Object DeviceID | ForEach-Object {
    $free = [math]::Floor($_.FreeSpace / 10GB) * 10
    Write-Output ("Volume {0} {1:N0} GB in all, about {2:N0} GB free (to the nearest 10 below)" -f $_.DeviceID, ($_.Size / 1GB), $free)
}

Write-Output '---'

$up = (Get-Date) - $os.LastBootUpTime
Write-Output ("Up for {0} days {1} hours, since {2:yyyy-MM-dd HH:mm}" -f $up.Days, $up.Hours, $os.LastBootUpTime)
$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
Write-Output ("Processor load: {0:N0}%" -f $load)
$usedGb = ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB
Write-Output ("Memory in use: {0:N1} GB of {1:N1} GB" -f $usedGb, ($os.TotalVisibleMemorySize / 1MB))
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Sort-Object DeviceID | ForEach-Object {
    Write-Output ("Volume {0} free: {1:N1} GB" -f $_.DeviceID, ($_.FreeSpace / 1GB))
}
