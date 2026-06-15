# astel-gpu

GPU pipeline for Astel: gsplat-backed differentiable rendering and
optimization. Standalone uv project (Python 3.11, CUDA-only — not part of any
workspace, and not on the API's import graph).

## Setup (Windows, native, MSVC + CUDA 12.9, RTX 4090 / sm_89)

**One-command setup:** from the repo root, run

```
pwsh scripts/setup-gpu-env.ps1
```

This `uv sync`s the venv, idempotently applies the two vendored-file patches
described below (skips them if already applied), and warms the gsplat JIT via
the launcher. Safe to re-run; prints a PASS/FAIL summary.

**Running any command:** torch 2.11's `cpp_extension` JIT loader runs `where
cl` on *every* gsplat import — even when the compiled extension is already
cached — so every process that imports `gsplat` needs `cl.exe` on PATH and
`CUDA_HOME` set, every time, with no cache-hit shortcut. Use the
`run-python.cmd` launcher in this directory, which sets up that environment
and then runs `uv run python` with whatever arguments you pass:

```
cmd /c "D:\Astel\pipelines\gpu\run-python.cmd -m astel_gpu.smoke_refit --iters 1500 --out out"
```

`services/api`'s GPU producer (`ASTEL_PRODUCER=gpu`) invokes this launcher
automatically via `subprocess.run(["cmd", "/c", "run-python.cmd", ...])`, so
no manual vcvars setup is needed to use the API's GPU path either.

The launcher is equivalent to manually running, from PowerShell:

```
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" -vcvars_ver=14.38 && set DISTUTILS_USE_SDK=1 && set TORCH_CUDA_ARCH_LIST=8.9+PTX && set CUDA_HOME=%CUDA_PATH% && cd /d D:\Astel\pipelines\gpu && uv run python -m astel_gpu.smoke_refit"
```

**Machine-specific gotcha (VS 2026 Community on THREADRIPPER-48):** plain
`vcvars64.bat` fails to put `cl.exe` on PATH here. The installed VC++ toolset
folder is `VC\Tools\MSVC\14.51.36231.hidden`, but `vcvars64.bat`'s default
version file points at `14.51.36231` (without `.hidden`), so it can't find the
toolset and silently skips adding it to PATH (`where cl` then fails with no
error). The fix is `-vcvars_ver=14.38`, which selects the other installed
toolset (`14.38.33130`) that vcvars *can* find. Also ensure
`C:\Program Files (x86)\Microsoft Visual Studio\Installer` is on PATH so
`vswhere.exe` resolves (otherwise vcvarsall prints a "Could not find files for
the given pattern(s)" warning, though that part is non-fatal).

**Watch for trailing spaces in `set VAR=value&&`.** If you build the `cmd /c`
command line in PowerShell with `&&` on its own token (`set "X=Y" && next`),
the trailing space before `&&` becomes part of the value (`X="Y "`). For
`TORCH_CUDA_ARCH_LIST` this turns `"8.9+PTX "` into `["8.9+PTX", ""]` after
torch's `.replace(' ', ';').split(';')`, and the empty string fails
`_get_cuda_arch_flags` with `ValueError: Unknown CUDA arch () or GPU not
supported`. Use `set "VAR=value"&& next` (no space before `&&`, value quoted)
to avoid this.

**Two local venv patches were required to get gsplat 1.5.3 + torch
2.11.0+cu128 to JIT-compile on this machine** (both are environment-only
patches to vendored third-party files in `.venv/`, not repo code; redo them
if the venv is recreated):

1. `torch/include/c10/cuda/CUDACachingAllocator.h`: `StreamSegmentSize`'s
   constructor takes a parameter literally named `small`. On Windows,
   `<rpcndr.h>` (pulled in transitively via CUDA/Windows headers) `#define
   small char`, so NVCC/MSVC sees `StreamSegmentSize(cudaStream_t s, bool
   char, size_t sz)` -> "invalid combination of type specifiers". This is a
   regression in this very-new torch build's header (the `small` identifier
   should never appear unguarded in a Windows-compiled header). Fix: rename
   the parameter to `is_small_segment`.
2. `gsplat/cuda/_backend.py`: hardcodes `extra_cflags = [opt_level,
   "-Wno-attributes"]`. `-Wno-attributes` is GCC/Clang-only; MSVC's `cl.exe`
   rejects it (`Command line error D8021 : invalid numeric argument
   '/Wno-attributes'`). This is an open upstream bug
   (nerfstudio-project/gsplat#809, unresolved as of 2025-09). Fix: on
   `sys.platform == "win32"`, drop `-Wno-attributes` from `extra_cflags`.

With both patches, `gsplat` JIT-compiled its CUDA extension in ~85s and a
tiny `rasterization()` call rendered correctly (first render 84.77s
including compile, second render 0.0008s).

## Commands

Run any of these via `run-python.cmd` (see above), e.g.
`cmd /c "D:\Astel\pipelines\gpu\run-python.cmd -m astel_gpu.env_check"`:

- `-m astel_gpu.env_check` — print torch/CUDA/GPU visibility.
- `-m astel_gpu.smoke_refit --iters 1500 --out out` — render-then-refit
  smoke test (self-consistency, not a ground-truth benchmark).
- `-m astel_gpu.synthetic_eval --iters 1500 --out out` — controlled
  synthetic-ground-truth eval: refits a gaussian cloud against a KNOWN
  sphere-shell target (longest axis exactly 0.20 m by construction) and
  reports a REAL measured Chamfer distance (mm) between the refit cloud's
  means and the known ground-truth points, plus PSNR. The headline
  `chamfer_mm_vs_l1` is measured over surface-defining gaussians (opacity >
  0.5); the raw all-means value is also reported (`chamfer_raw_all_means_mm`).
  Writes `l3.ply`, `synthetic-eval-metrics.json`, and a `quality-report.json`
  with non-`None` `geometric_error` and `scale` fields. This is a baseline
  for raw 3DGS without surface regularization — it covers the surface well
  (~15 mm) but leaves floaters (~165 mm precision), which is the empirical
  case for the surface-aligned L3 representation (2DGS/SuGaR).
- `-m astel_gpu.produce --task-id ID --modality text --prompt "..." --out DIR`
  — GPU producer used by `services/api` when `ASTEL_PRODUCER=gpu`.
- `-m astel_gpu.capture_eval --image-dir DIR --pos-dir DIR --gt-ply PLY --obsmask
  MAT --plane MAT --out DIR` — **real-world** geometry eval on a DTU scan: fits a
  gaussian cloud to the real photos using DTU's metric poses (mm, GT frame, no
  registration), then measures geometry vs the structured-light scan using DTU's
  **official ObsMask/Plane protocol** (`PointCompareMain.m`): `accuracy` = fitted
  gaussians in the observable volume → nearest GT; `completeness` = GT in the
  object volume → nearest gaussian; 60 mm cap; **held-out-view PSNR**. Example
  (scan1, paths under the gitignored `data/dtu/extracted/scan1/`): `--image-dir
  .../images --pos-dir .../pos --gt-ply .../stl001_total.ply --obsmask
  .../obsmask/ObsMask1_10.mat --plane .../obsmask/Plane1.mat --iters 3000
  --n-gaussians 200000`. Raw-3DGS baseline: held-out PSNR 21.5 dB, accuracy
  11.36 mm, completeness 6.10 mm, overall 8.73 mm.
- `-m astel_gpu.colmap_runner --image-dir DIR --work-dir DIR` — run COLMAP SfM
  (GPU SIFT → match → mapper → undistort) on an image folder → poses + sparse
  cloud (L0). On DTU scan1: 49/49 images registered, ~55 s.
- `-m astel_gpu.capture_sfm --colmap-model-dir DIR --pos-dir DIR` — align COLMAP
  poses to DTU GT poses (Umeyama) → pose accuracy (mm). scan1: RMSE 0.886 mm.
  (Pure numpy — runs without the launcher.)
- `-m pytest -q` — run the CUDA-backed test suite.

## Future hardening

Runtime currently requires a full MSVC dev environment (via `run-python.cmd`)
on every invocation because torch 2.11's `cpp_extension` JIT loader shells out
to `where cl` unconditionally, even for a cached extension. A future
hardening pass should either (a) AOT-build the gsplat CUDA extension into a
wheel that doesn't go through the JIT loader at runtime, or (b) move to a
torch version whose JIT loader takes the cache-hit shortcut without needing
`cl.exe` on PATH. Either would let the API invoke `astel_gpu.produce` with a
plain `uv run python` and drop the launcher/VS dependency entirely.

## Honesty note

The smoke test proves the differentiable rasterizer's forward+backward and
the optimization loop work on this hardware (gsplat renders a target, a fresh
random cloud is refit to match it). It is **not** a ground-truth-geometry
accuracy benchmark — that arrives with the COLMAP / real-capture path (M2).

`synthetic_eval` is a different, complementary measurement: it builds a
sphere-shell point cloud whose ground truth (positions AND a 0.20 m longest
axis) is KNOWN by construction, renders it, and refits a fresh cloud against
those renders. It reports the smoke-style self-consistency PSNR **and** a
REAL Chamfer distance (mm) between the refit cloud's means and the known
ground-truth points, plus `scale.confidence = 1.0` since the scale is fixed
by construction. This validates the Chamfer/scale measurement machinery and
measures the refit's geometric fidelity against a controlled target — it is
still **not** a real-world capture accuracy benchmark (real-world geometric
accuracy requires the COLMAP/MapAnything real-capture path on real photos or
video). It does not change `astel_gpu.produce`'s quality report, which keeps
`geometric_error` and `scale` honestly `None` for the API's GPU producer
path.
