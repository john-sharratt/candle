# The folders named in `storage.paths`, beside this script: one path to a
# line, lines beginning `#` ignored. Compared from run to run: each folder's
# size to the nearest gigabyte below, so growth shows and a few megabytes do
# not. After `---`: the exact sizes and the newest file of each.
$ErrorActionPreference = 'SilentlyContinue'

$list = Join-Path $PSScriptRoot 'storage.paths'
$paths = @()
if (Test-Path $list) {
    $paths = Get-Content $list | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }
}
if (@($paths).Count -eq 0) {
    Write-Output "No folders are named to watch: add one path to a line in $list."
    exit 0
}

$moving = @()
foreach ($path in $paths) {
    if (-not (Test-Path $path)) {
        Write-Output "${path}: not there"
        continue
    }
    $files = Get-ChildItem -LiteralPath $path -Recurse -File -Force
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if (-not $bytes) { $bytes = 0 }
    Write-Output ("{0}: about {1:N0} GB (to the nearest below)" -f $path, [math]::Floor($bytes / 1GB))
    $newest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $age = if ($newest) { "newest file written {0:yyyy-MM-dd HH:mm}" -f $newest.LastWriteTime } else { 'holds no files' }
    $moving += ("{0}: {1:N2} GB in {2:N0} files, {3}" -f $path, ($bytes / 1GB), @($files).Count, $age)
}

Write-Output '---'
$moving | ForEach-Object { Write-Output $_ }
