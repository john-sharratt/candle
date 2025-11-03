# Fix GPU Power Management for RTX 4090
# Run this as Administrator

Write-Host "Configuring Windows Power Settings for GPU Performance..." -ForegroundColor Cyan

# Set High Performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable USB Selective Suspend
powercfg /setacvalueindex scheme_current 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
powercfg /setdcvalueindex scheme_current 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0

# Disable PCI Express Link State Power Management
powercfg /setacvalueindex scheme_current 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0
powercfg /setdcvalueindex scheme_current 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0

# Apply settings
powercfg /setactive scheme_current

Write-Host "Done! Now configure NVIDIA Control Panel:" -ForegroundColor Green
Write-Host "1. Open NVIDIA Control Panel" -ForegroundColor Yellow
Write-Host "2. Manage 3D Settings > Global Settings" -ForegroundColor Yellow
Write-Host "3. Power management mode: 'Prefer maximum performance'" -ForegroundColor Yellow
Write-Host "4. Apply and restart your application" -ForegroundColor Yellow

Write-Host "`nMonitor GPU with: nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,power.draw --format=csv -l 1" -ForegroundColor Cyan
