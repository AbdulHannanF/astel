#requires -Version 5.1
<#
.SYNOPSIS
  astel up -- one-command local bring-up for the Astel stack.
.DESCRIPTION
  Starts the FastAPI gateway and the Vite web app, streams both logs, and tears
  down on Ctrl+C.

  Producer selection (the real on-site splat generation):
    * By default the script AUTO-DETECTS the GPU: if nvidia-smi, the
      pipelines/gpu venv, and run-python.cmd are all present it runs the REAL
      generative producer (ASTEL_PRODUCER=gpu) -- prompt/image -> SDXL/TripoSplat
      -> 2DGS L3 -- so every generation is a real, prompt-conditioned splat.
    * Otherwise it falls back to the CPU stub (procedural placeholder geometry).
    * Force either with -Gpu or -Stub.

  Production runs ASYNCHRONOUSLY in the API: a generation returns immediately and
  streams real per-stage progress over SSE (no blocking request, no fake replay).

  Remote / laptop access (use this box's 4090s from another machine):
    pnpm run up -- -BindHost 0.0.0.0
  then browse to  http://<this-box-LAN-IP>:5173  from the laptop. The Vite dev
  server proxies /v1 to the local API, so the laptop drives generation on this
  box's GPUs with no extra config. See docs/REMOTE_ACCESS.md.

  -Temporal mode additionally launches a local Temporal dev server + worker and
  runs the API against the durable Temporal engine.

  For a full prod-shaped dependency stack (Postgres + MinIO + Temporal on
  Postgres), use Docker directly:  docker compose -f infra/docker-compose.yml up -d
.EXAMPLE
  pnpm run up
.EXAMPLE
  pnpm run up -- -BindHost 0.0.0.0      # expose on the LAN for laptop access
.EXAMPLE
  pnpm run up -- -Stub                  # force the CPU stub producer
.EXAMPLE
  pnpm run up -- -Temporal
#>
[CmdletBinding()]
param(
  [switch] $Temporal,
  [switch] $Gpu,
  [switch] $Stub,
  [string] $BindHost = '127.0.0.1',
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

function Test-GpuReady {
  $gpuVenv = Join-Path $repoRoot 'pipelines/gpu/.venv/Scripts/python.exe'
  $launcher = Join-Path $repoRoot 'pipelines/gpu/run-python.cmd'
  return ((Test-OnPath 'nvidia-smi') -and (Test-Path $gpuVenv) -and (Test-Path $launcher))
}

if ($Gpu -and $Stub) {
  Write-Error "Pass at most one of -Gpu / -Stub."
  exit 1
}

if ($Temporal -and -not (Test-OnPath 'temporal')) {
  Write-Error "Temporal mode requested but the 'temporal' CLI is not on PATH. Install the Temporal CLI (https://docs.temporal.io/cli) or run without -Temporal."
  exit 1
}

# Producer: GPU when forced or auto-detected; CPU stub otherwise.
if ($Stub) {
  $producer = 'stub'
} elseif ($Gpu) {
  if (-not (Test-GpuReady)) {
    Write-Error "-Gpu requested but the GPU pipeline isn't ready (need nvidia-smi + pipelines/gpu/.venv + run-python.cmd). See docs/MVP_TESTING.md."
    exit 1
  }
  $producer = 'gpu'
} elseif (Test-GpuReady) {
  $producer = 'gpu'
} else {
  $producer = 'stub'
}

$mode = if ($Temporal) { 'temporal' } else { 'dev' }
# The API's ASTEL_ENGINE only accepts 'stub' or 'temporal' (see config.py). The
# 'dev' label maps to the in-process engine; production runs in the async job
# manager regardless. Passing 'dev' straight through made pydantic reject it.
$engine = if ($Temporal) { 'temporal' } else { 'stub' }

$producerNote = if ($producer -eq 'gpu') { 'GPU (real generative pipeline)' } else { 'CPU stub (placeholder geometry)' }
Write-Host "Astel up [$mode]  --  API ${BindHost}:$ApiPort  web :$WebPort  producer: $producerNote" -ForegroundColor Cyan
if ($producer -eq 'stub' -and -not $Stub) {
  Write-Host "  (no GPU detected -- generations use the placeholder stub. Run on the 2x4090 box for real splats.)" -ForegroundColor DarkYellow
}

# When binding to all interfaces, surface the LAN URL for remote (laptop) use.
$exposed = ($BindHost -eq '0.0.0.0' -or $BindHost -eq '::')
if ($exposed) {
  $lanIp = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1).IPAddress
  if ($lanIp) {
    Write-Host "Remote access: open  http://${lanIp}:$WebPort  from another machine on this network." -ForegroundColor Green
  }
}

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

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
      if ((Test-NetConnection -ComputerName '127.0.0.1' -Port $TemporalPort -WarningAction SilentlyContinue).TcpTestSucceeded) { break }
      Start-Sleep -Milliseconds 500
    }

    $jobs += Start-Job -Name astel-worker -ScriptBlock {
      param($root, $producer)
      Set-Location (Join-Path $root 'services/api')
      $env:PYTHONPATH = 'src'
      $env:ASTEL_ENGINE = 'temporal'
      $env:ASTEL_PRODUCER = $producer
      uv run python -m astel_api.temporal.worker
    } -ArgumentList $repoRoot, $producer
  }

  $jobs += Start-Job -Name astel-api -ScriptBlock {
    param($root, $bindHost, $port, $engine, $producer)
    Set-Location (Join-Path $root 'services/api')
    $env:ASTEL_ENGINE = $engine
    $env:ASTEL_PRODUCER = $producer
    uv run uvicorn astel_api.main:app --app-dir src --host $bindHost --port $port
  } -ArgumentList $repoRoot, $BindHost, $ApiPort, $engine, $producer

  $jobs += Start-Job -Name astel-web -ScriptBlock {
    param($root, $port, $exposed)
    Set-Location (Join-Path $root 'apps/web')
    if ($exposed) {
      pnpm dev --host --port $port
    } else {
      pnpm dev --port $port
    }
  } -ArgumentList $repoRoot, $WebPort, $exposed

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
