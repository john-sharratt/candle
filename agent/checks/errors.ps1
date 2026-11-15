# The last day's errors and critical events from the System and Application
# logs. Compared from run to run: which sources raised which event ids. After
# `---`: how many of each, and the newest message of each.
$ErrorActionPreference = 'SilentlyContinue'

$since = (Get-Date).AddHours(-24)
$events = @()
foreach ($log in 'System', 'Application') {
    $events += Get-WinEvent -FilterHashtable @{ LogName = $log; Level = 1, 2; StartTime = $since } -MaxEvents 2000
}
if ($events.Count -eq 0) {
    Write-Output 'No errors or critical events in the last day.'
    exit 0
}

$groups = $events | Group-Object { "$($_.LogName) / $($_.ProviderName) / event $($_.Id)" } | Sort-Object Name
foreach ($g in $groups) {
    Write-Output "Raised: $($g.Name)"
}

Write-Output '---'
Write-Output "$($events.Count) errors and critical events in the last day:"
foreach ($g in ($groups | Sort-Object Count -Descending | Select-Object -First 25)) {
    $newest = $g.Group | Sort-Object TimeCreated -Descending | Select-Object -First 1
    $text = ($newest.Message -split "`n")[0]
    if ($text.Length -gt 200) { $text = $text.Substring(0, 200) + '...' }
    Write-Output ("{0} x {1} - newest {2:MM-dd HH:mm}: {3}" -f $g.Count, $g.Name, $newest.TimeCreated, $text)
}
