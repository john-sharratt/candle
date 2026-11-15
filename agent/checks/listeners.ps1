# The listening ports, each with the program behind it. All of it is compared
# from run to run: a port that was not open is the thing to see. Ports the
# system hands out for a moment (49152 and up) are counted, not listed.
$ErrorActionPreference = 'SilentlyContinue'

$names = @{}
Get-Process | ForEach-Object { $names[[int]$_.Id] = $_.ProcessName }

$tcp = Get-NetTCPConnection -State Listen
$rows = foreach ($c in $tcp) {
    $reach = if ($c.LocalAddress -in '127.0.0.1', '::1') { 'this machine only' } else { 'the network' }
    [pscustomobject]@{ Port = [int]$c.LocalPort; Reach = $reach; Program = $names[[int]$c.OwningProcess] }
}
$rows = @($rows | Sort-Object Port, Reach, Program -Unique)
foreach ($r in ($rows | Where-Object { $_.Port -lt 49152 })) {
    $program = if ($r.Program) { $r.Program } else { 'a program this account cannot name' }
    Write-Output ("TCP {0} open to {1}: {2}" -f $r.Port, $r.Reach, $program)
}

Write-Output '---'
$passing = @($rows | Where-Object { $_.Port -ge 49152 })
Write-Output "$($passing.Count) more ports from 49152 up, handed out by the system as programs ask."
