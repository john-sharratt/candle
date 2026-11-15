# The machine's own defences. Compared from run to run: what is on and what is
# off. After `---`: ages, which grow every day.
$ErrorActionPreference = 'SilentlyContinue'

$moving = @()

foreach ($p in (Get-NetFirewallProfile | Sort-Object Name)) {
    $state = if ($p.Enabled) { 'on' } else { 'OFF' }
    Write-Output "Firewall, $($p.Name) networks: $state"
}

$av = Get-MpComputerStatus
if ($av) {
    $on = { param($b) if ($b) { 'on' } else { 'OFF' } }
    Write-Output "Antivirus: $(& $on $av.AntivirusEnabled)"
    Write-Output "Real-time protection: $(& $on $av.RealTimeProtectionEnabled)"
    Write-Output "Tamper protection: $(& $on $av.IsTamperProtected)"
    $moving += "Antivirus definitions: $($av.AntivirusSignatureAge) days old (version $($av.AntivirusSignatureVersion))"
    if ($av.QuickScanEndTime) { $moving += ("Last quick scan: {0:yyyy-MM-dd HH:mm}" -f $av.QuickScanEndTime) }
} else {
    Write-Output 'Antivirus: Windows Defender does not answer here (another product, or none).'
}

$threats = @(Get-MpThreatDetection | Where-Object { $_.InitialDetectionTime -gt (Get-Date).AddDays(-7) })
if ($threats.Count -gt 0) {
    foreach ($t in ($threats | Select-Object -First 10)) {
        Write-Output ("Threat detected {0:yyyy-MM-dd HH:mm}: id {1}, {2}" -f $t.InitialDetectionTime, $t.ThreatID, ($t.Resources -join '; '))
    }
} else {
    Write-Output 'Threats detected in the last week: none'
}

$rdp = (Get-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server').fDenyTSConnections
Write-Output ("Remote desktop: {0}" -f $(if ($rdp -eq 0) { 'allowed' } else { 'not allowed' }))

Write-Output '---'
$moving | ForEach-Object { Write-Output $_ }
$fix = Get-HotFix | Where-Object { $_.InstalledOn } | Sort-Object InstalledOn -Descending | Select-Object -First 1
if ($fix) {
    $age = ((Get-Date) - $fix.InstalledOn).Days
    Write-Output ("Last update installed: {0} on {1:yyyy-MM-dd}, {2} days ago" -f $fix.HotFixID, $fix.InstalledOn, $age)
}
