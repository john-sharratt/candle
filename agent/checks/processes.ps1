# What is running. Compared from run to run: each program by name and where it
# runs from, so one that was not there before shows. After `---`: the heaviest.
$ErrorActionPreference = 'SilentlyContinue'

$procs = Get-Process
$seen = foreach ($p in $procs) {
    $path = $p.Path
    if (-not $path) { $path = 'a place this account cannot read' }
    "Running: $($p.ProcessName) from $path"
}
$seen | Sort-Object -Unique | ForEach-Object { Write-Output $_ }

Write-Output '---'
Write-Output "$(@($procs).Count) processes in all."
Write-Output 'Heaviest by memory:'
foreach ($p in ($procs | Sort-Object WorkingSet64 -Descending | Select-Object -First 8)) {
    Write-Output ("  {0} (pid {1}): {2:N0} MB" -f $p.ProcessName, $p.Id, ($p.WorkingSet64 / 1MB))
}
Write-Output 'Heaviest by processor time since each started:'
foreach ($p in ($procs | Where-Object { $_.CPU } | Sort-Object CPU -Descending | Select-Object -First 8)) {
    Write-Output ("  {0} (pid {1}): {2:N0} s" -f $p.ProcessName, $p.Id, $p.CPU)
}
