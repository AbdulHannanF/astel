# AURIGA GPU box setup — run ONCE as Administrator on each GPU machine.
# Right-click PowerShell -> "Run as administrator", then:
#   powershell -ExecutionPolicy Bypass -File setup-gpu-box.ps1
# Reboot once when it finishes. Everything after this is done remotely by the agent.

$ErrorActionPreference = 'Stop'
Write-Host "=== AURIGA GPU box setup ===" -ForegroundColor Cyan

# 1. OpenSSH Server (remote access for the agent)
Write-Host "[1/4] Enabling OpenSSH Server..."
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
# Make PowerShell the default SSH shell (agent expects it)
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null
# Firewall: allow SSH on the Tailscale interface (port 22)
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}
Write-Host "      SSH ready." -ForegroundColor Green

# 2. WSL2 + Ubuntu 24.04 (the ML stack runs Linux-first; CUDA passes through WSL2)
Write-Host "[2/4] Installing WSL2 + Ubuntu 24.04 (skips if present)..."
$wslInstalled = (wsl -l -q 2>$null) -match 'Ubuntu'
if (-not $wslInstalled) {
  wsl --install -d Ubuntu-24.04 --no-launch
  Write-Host "      WSL installed. After reboot, run 'wsl' once to create your Linux user." -ForegroundColor Yellow
} else {
  Write-Host "      WSL/Ubuntu already present." -ForegroundColor Green
}

# 3. Report NVIDIA driver (CUDA-in-WSL needs a current driver; do NOT install CUDA toolkit on Windows)
Write-Host "[3/4] GPU / driver check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 4. Report Tailscale IP so the agent knows how to reach this box
Write-Host "[4/4] Network:"
try { tailscale ip -4 } catch { Write-Host "      Tailscale CLI not found - make sure the Tailscale app is running and set to start at login." -ForegroundColor Yellow }

Write-Host ""
Write-Host "=== Done. Please REBOOT, run 'wsl' once to finish Ubuntu user setup, ===" -ForegroundColor Cyan
Write-Host "=== then send the agent: this machine's Windows username.            ===" -ForegroundColor Cyan
