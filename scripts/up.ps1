#requires -Version 5.1
<#
.SYNOPSIS
  astel up -- one-command local bring-up for the Astel stack.
.DESCRIPTION
  Default (dev) mode: starts the FastAPI gateway (SQLite + in-process stub
  engine) and the Vite web app, streams both logs, tears down on Ctrl+C. This
  is the verified, GPU-free path that works on the dev box today.

  -Temporal mode: additionally launches a local Temporal dev server and the
  Astel pipeline worker, and runs the API against the durable Temporal engine
  (ASTEL_ENGINE=temporal). Requires the `temporal` CLI on PATH; the script
  errors clearly if it is absent.

  For a full prod-shaped dependency stack (Postgres + MinIO + Temporal on
  Postgres), use Docker directly:  docker compose -f infra/docker-compose.yml up -d
.EXAMPLE
  pnpm run up
.EXAMPLE
  pnpm run up -- -Temporal
#>
[CmdletBinding()]
param(
  [switch] $Temporal,
  [int]    $ApiPort = 8000,
  [int]    $WebPort = 5173,
  [int]    $TemporalPort = 7233,
  [int]    $TemporalUiPort = 8233
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Test-OnPath([string] $name) {
  return [bool] (Get-Command $name -ErrorAction SilentlyContinue)
}

if ($Temporal -and -not (Test-OnPath 'temporal')) {
  Write-Error "Temporal mode requested but the 'temporal' CLI is not on PATH. Install the Temporal CLI (https://docs.temporal.io/cli) or run without -Temporal for dev/stub mode."
  exit 1
}

$mode = if ($Temporal) { 'temporal' } else { 'dev' }
Write-Host "Astel up [$mode]  --  API :$ApiPort  web :$WebPort" -ForegroundColor Cyan

$jobs = @()
try {
  if ($mode -eq 'temporal') {
    Write-Host "Starting Temporal dev server (frontend :$TemporalPort, UI :$TemporalUiPort)..." -ForegroundColor Cyan
    $dataDir = Join-Path $repoRoot '.astel-temporal'
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    $jobs += Start-Job -Name astel-temporal -ScriptBlock {
      param($port, $uiPort, $db)
      temporal server start-dev --port $port --ui-port $uiPort --db-filename $db
    } -ArgumentList $TemporalPort, $TemporalUiPort, (Join-Path $dataDir 'temporal.db')

    # Wait for the frontend gRPC port to accept connections before starting the worker.
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
      if ((Test-NetConnection -ComputerName '127.0.0.1' -Port $TemporalPort -WarningAction SilentlyContinue).TcpTestSucceeded) { break }
      Start-Sleep -Milliseconds 500
    }

    $jobs += Start-Job -Name astel-worker -ScriptBlock {
      param($root)
      Set-Location (Join-Path $root 'services/api')
      $env:PYTHONPATH = 'src'
      $env:ASTEL_ENGINE = 'temporal'
      uv run python -m astel_api.temporal.worker
    } -ArgumentList $repoRoot
  }

  $jobs += Start-Job -Name astel-api -ScriptBlock {
    param($root, $port, $engine)
    Set-Location (Join-Path $root 'services/api')
    $env:ASTEL_ENGINE = $engine
    uv run uvicorn astel_api.main:app --app-dir src --host 127.0.0.1 --port $port
  } -ArgumentList $repoRoot, $ApiPort, $mode

  $jobs += Start-Job -Name astel-web -ScriptBlock {
    param($root, $port)
    Set-Location (Join-Path $root 'apps/web')
    pnpm dev --port $port
  } -ArgumentList $repoRoot, $WebPort

  Write-Host "Services starting. Web: http://localhost:$WebPort  API: http://127.0.0.1:$ApiPort/healthz" -ForegroundColor Green
  if ($mode -eq 'temporal') {
    Write-Host "Temporal UI: http://localhost:$TemporalUiPort" -ForegroundColor Green
  }
  Write-Host 'Press Ctrl+C to stop.' -ForegroundColor Green

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
  Write-Host 'Stopping Astel services...' -ForegroundColor Cyan
  $jobs | Stop-Job -ErrorAction SilentlyContinue
  $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
}
