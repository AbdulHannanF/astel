@echo off
REM Launcher for the astel-gpu venv that ensures the MSVC compiler (cl.exe)
REM and CUDA build env vars are on PATH/set before invoking python.
REM
REM torch 2.11's cpp_extension JIT loader runs `where cl` on every gsplat
REM import, even when the compiled extension is already cached, so any
REM command that imports gsplat needs a VS dev shell. This script provides
REM that shell and then forwards all arguments to `uv run python`.
REM
REM Usage: run-python.cmd -m astel_gpu.produce --task-id ID ...

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul 2>&1

set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=8.9+PTX"
if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"

cd /d "%~dp0"
uv run python %*
