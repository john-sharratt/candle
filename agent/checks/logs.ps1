# One event log, searched. The whole report moves — a search is asked for what
# it finds now — so everything follows the `---` line and nothing is compared.
param(
    [Parameter(Mandatory = $true)][ValidateSet('System', 'Application', 'Security', 'Setup')][string]$Log,
    [ValidateSet('1h', '6h', '24h', '7d')][string]$Since = '24h',
    [string]$Grep = ''
)
$ErrorActionPreference = 'SilentlyContinue'

$hours = @{ '1h' = 1; '6h' = 6; '24h' = 24; '7d' = 168 }[$Since]
$from = (Get-Date).AddHours(-$hours)

Write-Output '---'
$events = Get-WinEvent -FilterHashtable @{ LogName = $Log; StartTime = $from } -MaxEvents 5000 -ErrorVariable failed
# Finding nothing is an answer, not a failure to read the log.
if ($failed -and $failed[0].FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') {
    Write-Output "The $Log log could not be read: $($failed[0].Exception.Message)"
    exit 0
}
if ($Grep -ne '') {
    # A plain substring, never a pattern: the word given is matched as written.
    $events = $events | Where-Object { $_.Message -and $_.Message.IndexOf($Grep, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 }
}
$events = @($events)
$what = if ($Grep -ne '') { " holding `"$Grep`"" } else { '' }
Write-Output "$($events.Count) events in the $Log log over the last $Since$what. Newest first:"
foreach ($e in ($events | Select-Object -First 40)) {
    $text = ($e.Message -split "`n")[0]
    if ($text.Length -gt 240) { $text = $text.Substring(0, 240) + '...' }
    Write-Output ("{0:MM-dd HH:mm:ss} {1} {2} event {3}: {4}" -f $e.TimeCreated, $e.LevelDisplayName, $e.ProviderName, $e.Id, $text)
}
if ($events.Count -gt 40) { Write-Output "... and $($events.Count - 40) older." }
