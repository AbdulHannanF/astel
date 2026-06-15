#!/usr/bin/env pwsh
# Idempotent setup for pipelines/gpu (gsplat + torch cu128 on a 4090-class box).
#
# Steps:
#   1. uv sync the venv (torch cu128, gsplat, editable astel-splat-io).
#   2. Apply the two vendored-file patches required for gsplat 1.5.3 +
#      torch 2.11.0+cu128 to JIT-compile under MSVC on Windows (idempotent;
#      skipped if already applied).
#   3. Warm the gsplat JIT via run-python.cmd (the MSVC-env launcher) and
#      assert it succeeds.
#
# Safe to re-run. Prints a PASS/FAIL summary at the end.

$ErrorActionPreference = "Stop"
$gpuDir = "D:\Astel\pipelines\gpu"
$failed = $false

function Write-Step($msg) {
    Write-Host "`n== $msg ==" -ForegroundColor Cyan
}

# --- 1. uv sync -------------------------------------------------------
Write-Step "uv sync (pipelines/gpu)"
try {
    & cmd /c "cd /d $gpuDir && uv sync 2>&1" | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "uv sync exited $LASTEXITCODE" }
    Write-Host "uv sync OK" -ForegroundColor Green
}
catch {
    Write-Host "uv sync FAILED: $_" -ForegroundColor Red
    $failed = $true
}

# --- 2. Locate vendored package dirs -----------------------------------
Write-Step "Locating vendored torch/gsplat package dirs"
$torchDir = & cmd /c "cd /d $gpuDir && uv run python -c ""import torch,os;print(os.path.dirname(torch.__file__))"" 2>&1"
$gsplatDir = & cmd /c "cd /d $gpuDir && uv run python -c ""import gsplat,os;print(os.path.dirname(gsplat.__file__))"" 2>&1"
$torchDir = ($torchDir | Select-Object -Last 1).Trim()
$gsplatDir = ($gsplatDir | Select-Object -Last 1).Trim()
Write-Host "torch:  $torchDir"
Write-Host "gsplat: $gsplatDir"

# --- 2a. Patch CUDACachingAllocator.h (rename `small` ctor param) ------
Write-Step "Patch 1: torch CUDACachingAllocator.h (Windows `small` macro collision)"
$allocHeader = Join-Path $torchDir "include\c10\cuda\CUDACachingAllocator.h"
if (Test-Path $allocHeader) {
    $content = Get-Content -Raw -LiteralPath $allocHeader
    if ($content -match "bool\s+small\b") {
        $patched = $content `
            -replace "StreamSegmentSize\(cudaStream_t s, bool small, size_t sz\)", `
                     "StreamSegmentSize(cudaStream_t s, bool is_small_segment, size_t sz)" `
            -replace "is_small_pool\(small\)", "is_small_pool(is_small_segment)"
        if ($patched -ne $content) {
            Set-Content -LiteralPath $allocHeader -Value $patched -NoNewline
            Write-Host "Patched (renamed 'small' -> 'is_small_segment')" -ForegroundColor Green
        }
        else {
            Write-Host "Found 'bool small' but pattern didn't match expected form -- check manually" -ForegroundColor Yellow
            $failed = $true
        }
    }
    elseif ($content -match "is_small_segment") {
        Write-Host "Already patched" -ForegroundColor Green
    }
    else {
        Write-Host "Marker not found (neither 'small' nor 'is_small_segment') -- check manually" -ForegroundColor Yellow
        $failed = $true
    }
}
else {
    Write-Host "File not found: $allocHeader -- skipping" -ForegroundColor Yellow
    $failed = $true
}

# --- 2b. Patch gsplat/_backend.py (drop -Wno-attributes on win32) ------
Write-Step "Patch 2: gsplat/cuda/_backend.py (drop -Wno-attributes on win32)"
$backendPy = Join-Path $gsplatDir "cuda\_backend.py"
if (Test-Path $backendPy) {
    $content = Get-Content -Raw -LiteralPath $backendPy
    if ($content -match '\[opt_level\]\s+if\s+sys\.platform\s*==\s*"win32"\s+else\s+\[opt_level,\s*"-Wno-attributes"\]') {
        Write-Host "Already patched" -ForegroundColor Green
    }
    elseif ($content -match '"-Wno-attributes"') {
        $patched = $content -replace `
            'extra_cflags\s*=\s*\[opt_level,\s*"-Wno-attributes"\]', `
            "extra_cflags = (`n            [opt_level] if sys.platform == ""win32"" else [opt_level, ""-Wno-attributes""]`n        )"
        if ($patched -ne $content) {
            if ($content -notmatch "^import sys" -and $content -notmatch "`nimport sys") {
                $patched = $patched -replace "(?m)^(import .+\n)", "`$1import sys`n", 1
            }
            Set-Content -LiteralPath $backendPy -Value $patched -NoNewline
            Write-Host "Patched (dropped -Wno-attributes on win32)" -ForegroundColor Green
        }
        else {
            Write-Host "Found '-Wno-attributes' but pattern didn't match expected form -- check manually" -ForegroundColor Yellow
            $failed = $true
        }
    }
    else {
        Write-Host "Already patched (no -Wno-attributes literal present)" -ForegroundColor Green
    }
}
else {
    Write-Host "File not found: $backendPy -- skipping" -ForegroundColor Yellow
    $failed = $true
}

# --- 3. Warm the JIT via the launcher -----------------------------------
Write-Step "Warming gsplat JIT via run-python.cmd -m astel_gpu.env_check"
$envCheckOut = & cmd /c "$gpuDir\run-python.cmd -m astel_gpu.env_check 2>&1"
$envCheckOut | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "env_check FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    $failed = $true
}
else {
    Write-Host "env_check OK" -ForegroundColor Green
}

Write-Step "Warming gsplat JIT via run-python.cmd -m astel_gpu.smoke_refit --iters 50"
$smokeOut = & cmd /c "$gpuDir\run-python.cmd -m astel_gpu.smoke_refit --iters 50 --out out\setup-warm 2>&1"
$smokeOut | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "smoke_refit FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    $failed = $true
}
else {
    Write-Host "smoke_refit OK" -ForegroundColor Green
}

# --- Summary -------------------------------------------------------------
Write-Step "SUMMARY"
if ($failed) {
    Write-Host "FAIL: one or more steps failed -- see above" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "PASS: pipelines/gpu env is set up and gsplat JIT is warm" -ForegroundColor Green
    exit 0
}
