#requires -Version 5.1
<#
.SYNOPSIS
  Run the Astel web app and API gateway together for local development.
.DESCRIPTION
  Starts the FastAPI gateway (uv, port 8000) and the Vite dev server
  (pnpm, port 5173) as background jobs, streams both logs, and tears both
  down cleanly on Ctrl+C. Invoked by 'pnpm run dev:all'.
#>
[CmdletBinding()]
param(
  [int] $ApiPort = 8000,
  [int] $WebPort = 5173
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Astel dev -- API :$ApiPort  web :$WebPort" -ForegroundColor Cyan

$jobs = @()
try {
  $jobs += Start-Job -Name astel-api -ScriptBlock {
    param($root, $port)
    Set-Location (Join-Path $root 'services/api')
    uv run uvicorn astel_api.main:app --app-dir src --host 127.0.0.1 --port $port
  } -ArgumentList $repoRoot, $ApiPort

  $jobs += Start-Job -Name astel-web -ScriptBlock {
    param($root, $port)
    Set-Location (Join-Path $root 'apps/web')
    pnpm dev --port $port
  } -ArgumentList $repoRoot, $WebPort

  Write-Host 'Both services starting. Press Ctrl+C to stop.' -ForegroundColor Green
  while ($true) {
    Receive-Job -Job $jobs | ForEach-Object { $_ }
    if ($jobs | Where-Object { $_.State -ne 'Running' }) {
      Write-Warning 'A service exited; check logs above.'
      break
    }
    Start-Sleep -Milliseconds 500
  }
}
finally {
  Write-Host ''
  Write-Host 'Stopping Astel dev services...' -ForegroundColor Cyan
  $jobs | Stop-Job -ErrorAction SilentlyContinue
  $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
}
