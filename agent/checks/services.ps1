# The daemons this machine keeps running. Compared from run to run: which are
# up, and the moment each started (so a restart shows). After `---`: memory.
$ErrorActionPreference = 'SilentlyContinue'

$watched = 'keep', 'motherd', 'web', 'tokera-agent', 'mcp'
$moving = @()
foreach ($name in $watched) {
    $procs = @(Get-Process -Name $name)
    if ($procs.Count -eq 0) {
        Write-Output "${name}: not running"
        continue
    }
    foreach ($p in $procs) {
        $since = if ($p.StartTime) { $p.StartTime.ToString('yyyy-MM-dd HH:mm') } else { 'a time this account cannot read' }
        Write-Output "${name}: running since $since"
        $moving += ("{0} (pid {1}): {2:N0} MB in memory" -f $name, $p.Id, ($p.WorkingSet64 / 1MB))
    }
}

$svc = Get-Service | Where-Object { $_.Name -match '^(keep|tokera|motherd)' }
foreach ($s in $svc) {
    Write-Output "Windows service $($s.Name): $($s.Status), starts $($s.StartType)"
}

Write-Output '---'
$moving | ForEach-Object { Write-Output $_ }
