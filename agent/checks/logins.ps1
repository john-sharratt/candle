# Sign-ins. Compared from run to run: who is signed in, and which accounts
# signed in or failed to from where over the last day. After `---`: counts and
# the newest of each.
$ErrorActionPreference = 'SilentlyContinue'

$sessions = & quser 2>$null
if ($sessions) {
    foreach ($line in ($sessions | Select-Object -Skip 1)) {
        $f = ($line.Trim() -replace '^>', '') -split '\s{2,}'
        $remote = if ($line -match 'rdp') { ' (remote desktop)' } else { '' }
        Write-Output "Signed in now: $($f[0])$remote"
    }
} else {
    Write-Output 'Signed in now: nobody this account can see'
}

$since = (Get-Date).AddHours(-24)
$events = Get-WinEvent -FilterHashtable @{ LogName = 'Security'; Id = 4624, 4625; StartTime = $since } -MaxEvents 3000 -ErrorVariable failed
if ($failed) {
    Write-Output 'Sign-in history: the Security log is not readable by the account the agent runs as.'
    exit 0
}

# Interactive (2), network (3), unlock (7), remote desktop (10), cached (11).
$kinds = @{ '2' = 'at the machine'; '3' = 'over the network'; '7' = 'unlocking it'; '10' = 'by remote desktop'; '11' = 'at the machine (cached)' }
$rows = foreach ($e in $events) {
    $x = [xml]$e.ToXml()
    $d = @{}
    foreach ($n in $x.Event.EventData.Data) { $d[$n.Name] = $n.'#text' }
    $user = $d['TargetUserName']
    if (-not $user -or $user -match '\$$' -or $user -in 'SYSTEM', 'ANONYMOUS LOGON', 'LOCAL SERVICE', 'NETWORK SERVICE' -or $user -match '^(DWM|UMFD)-') { continue }
    $kind = $kinds[[string]$d['LogonType']]
    if (-not $kind) { continue }
    $from = $d['IpAddress']
    if (-not $from -or $from -eq '-') { $from = 'this machine' }
    [pscustomobject]@{
        What = if ($e.Id -eq 4625) { 'Failed sign-in' } else { 'Sign-in' }
        Key  = "$user $kind from $from"
        When = $e.TimeCreated
    }
}
$rows = @($rows)
$groups = $rows | Group-Object What, Key | Sort-Object Name
foreach ($g in $groups) {
    Write-Output "$($g.Group[0].What): $($g.Group[0].Key)"
}
if ($rows.Count -eq 0) { Write-Output 'No sign-ins or failed attempts by a person in the last day.' }

Write-Output '---'
foreach ($g in ($groups | Sort-Object Count -Descending)) {
    $newest = ($g.Group | Sort-Object When -Descending | Select-Object -First 1).When
    Write-Output ("{0} x {1}: {2} - newest {3:MM-dd HH:mm}" -f $g.Count, $g.Group[0].What, $g.Group[0].Key, $newest)
}
