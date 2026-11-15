# The graphics cards and what holds them. Compared from run to run: which
# cards there are and which programs hold them. After `---`: the figures.
$ErrorActionPreference = 'SilentlyContinue'

$smi = Get-Command nvidia-smi
if (-not $smi) {
    Write-Output 'No NVIDIA card answers here: nvidia-smi is not on this machine.'
    exit 0
}

$cards = & nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits
foreach ($c in $cards) {
    $f = $c -split ',\s*'
    Write-Output "Card $($f[0]): $($f[1]), driver $($f[2]), $($f[3]) MiB"
}

$apps = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
$names = @()
foreach ($a in $apps) {
    $f = $a -split ',\s*'
    if ($f.Count -ge 2) { $names += (Split-Path $f[1] -Leaf) }
}
foreach ($n in ($names | Sort-Object -Unique)) {
    Write-Output "Held by: $n"
}
if ($names.Count -eq 0) { Write-Output 'Held by: nothing that computes on it' }

Write-Output '---'

$now = & nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
foreach ($n in $now) {
    $f = $n -split ',\s*'
    Write-Output "Card $($f[0]) now: $($f[1]) of $($f[2]) MiB used, $($f[3])% busy, $($f[4]) C, $($f[5]) W"
}
$used = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
foreach ($u in $used) {
    $f = $u -split ',\s*'
    if ($f.Count -ge 3) { Write-Output "  $(Split-Path $f[1] -Leaf) (pid $($f[0])): $($f[2])" }
}
