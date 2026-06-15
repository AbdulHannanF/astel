# Session 7 retro (2026-06-14)

**First GPU session. The native-Windows pivot is validated: gsplat builds and
trains on the 2×4090 box, and the API can produce a real optimized splat.**

Mode (per founder directive — Opus plans/decides/verifies, Sonnet implements,
Haiku does small tasks):
- **Opus**: planning, the WSL2→native-Windows decision, and all
  verification (independently re-ran every headline claim — caught a real
  "works in the agent's shell only" gap, see §3).
- **Sonnet ×2**: (A) GPU env + gsplat smoke + producer wiring; (B) the
  launcher/setup-script hardening that fixed §3.
- **Haiku ×1**: baseline gate sweep of the existing CPU stack.

This box (`THREADRIPPER-48`) is **Box A** from `docs/setup/gpu-boxes.md` — 2×
RTX 4090 (24 GB). We are running **on** it directly (not over SSH).

## 1. Decision: WSL2 → native Windows (validated)

The repo's old plan (DECISIONS §"Dev environment"/C6) was WSL2-first. On this
box **WSL2 is hard-blocked** — virtualization is off in firmware and "Virtual
Machine Platform" is disabled (a physical BIOS action to enable). Meanwhile the
founder had provisioned **native CUDA 12.9 + Visual Studio 2026** and said so.
So we pivoted to a native-Windows GPU stack — ready now, vs WSL blocked on a
reboot. Recorded in DECISIONS.md (§ "2026-06-14 — GPU stack: native Windows").
Validated empirically this session (gsplat compiled and trained).

## 2. What landed (all measured on this hardware, not estimated)

- **GPU env** (`pipelines/gpu/`, standalone uv project, Python 3.12, not on the
  API's import graph): `torch 2.11.0+cu128`, `gsplat 1.5.3` (Apache-2.0). Both
  4090s visible (`device_count()==2`).
- **gsplat render-then-refit smoke** (`astel_gpu.smoke_refit`): a fresh random
  gaussian cloud refit to match gsplat-rendered target views.
  - 1500 iters: PSNR **8.19 → 45.63 dB**, 14.9 s, 0.17 GB peak VRAM.
  - 200 iters (launcher path): PSNR 8.19 → **39.87 dB**, 1.8 s.
  - **Honesty:** this is a self-consistency + convergence test (gsplat renders
    the target, gsplat refits it). It proves the differentiable rasterizer's
    forward+backward and the optimization loop work on this GPU. It is **not** a
    ground-truth-geometry accuracy benchmark — that arrives with the
    COLMAP/real-capture path (M2, session 8+).
- **Real GPU producer behind a flag**: `ASTEL_PRODUCER=gpu` makes the API
  (`gpu_producer.produce_artifacts_dispatch`) invoke the `pipelines/gpu`
  producer in a **subprocess** — torch/gsplat never enter the API's own import
  graph or dependency set. Default (no env var) is the unchanged stub, so CPU
  gates stay green. Produces `l3.ply` + an honest `astel.quality-report/v0`
  (`origin:"measured"`, real `fidelity.psnr_db`, `geometric_error`/`scale`
  explicitly `null` with reasons — never fabricated).
- **Reproducibility**: `pipelines/gpu/run-python.cmd` (the VS-env launcher) and
  `scripts/setup-gpu-env.ps1` (one-command, idempotent env + venv patches).

## 3. The real bug Opus caught (why verification matters)

Sonnet round A reported the GPU producer e2e "passed." Independent re-run from a
**clean shell** failed: torch 2.11's `cpp_extension` runs `where cl` on **every**
gsplat JIT import — even with the compiled `gsplat_cuda.pyd` already cached (it
does not take the cache-hit shortcut). So gsplat import requires the MSVC
compiler on PATH **at runtime, every time**. The agent's "e2e" only worked
because it manually vcvars-wrapped the call; the API's actual subprocess does
not — so `ASTEL_PRODUCER=gpu` would have failed in real operation.

Fix (Sonnet round B): a `run-python.cmd` launcher that activates the VS env
per-subprocess; `gpu_producer` now invokes it via `cmd /c`. Re-validated by Opus
from a clean shell (cl confirmed absent): the dispatcher produced `l3.ply`
(448 KB) + a measured `quality-report.json`. Works without any manual vcvars.

## 4. Windows/gsplat gotchas (documented in pipelines/gpu/README.md)

- **VS 2026 toolset quirk**: plain `vcvars64.bat` silently fails to put `cl` on
  PATH (installed toolset is `14.51.36231.hidden`, which vcvars can't resolve);
  `-vcvars_ver=14.38` selects the working `14.38.33130` toolset.
- **Two venv-local patches** (vendored third-party, re-applied by
  `setup-gpu-env.ps1`): (1) torch header `CUDACachingAllocator.h` param named
  `small` collides with Windows `<rpcndr.h>` `#define small char`; (2) gsplat
  `_backend.py` passes GCC-only `-Wno-attributes` to MSVC (upstream #809).
- Both are symptoms of torch 2.11 being bleeding-edge. See §6.

## 5. Gates (all re-run by Opus, green)

- Baseline CPU stack (Haiku, on this fresh box): api 28p/1s, astel_format 16,
  astel_splat_io 11, astel_eval 36, stub 14, web 18 (eslint+tsc), manifest 10.
  Only env fix: pnpm installed globally (corepack hit a permissions snag).
- This session's code: **api** ruff·mypy(21 files)·**28p/1s**; **pipelines/gpu**
  ruff·mypy(11 files)·**5p** (via launcher).

## 6. Honest gaps / carried forward

- **torch 2.11 fragility**: the stack works via the launcher, but runtime
  requires the VS compiler on PATH (a torch-2.11 JIT quirk). **Future hardening**
  (session 8 candidate): move to AOT-built gsplat or a stable torch (2.7/2.8
  cu126) so runtime needs no compiler and the two venv patches likely disappear.
- The two venv patches live in `.venv` (not repo code); `setup-gpu-env.ps1`
  re-applies them, but they're third-party-version-specific.
- **GPU producer is still the smoke geometry**, not a real reconstruction —
  it produces a converged splat of a synthetic target, not the user's prompt/
  capture. Real conditioning is the M2/M3 pipeline.
- **Deferred to session 8** (the actual M2 capture path): COLMAP/GLOMAP install
  + smoke; MapAnything (`-apache`) feed-forward on an orbit video; TRELLIS
  import-graph license check; the R-T1 distillation de-risk. Rungs 1–2 of the
  smoke ladder (CUDA sanity + gsplat reference) are done; rungs 3–5 remain.
- **TRELLIS/flash-attn on Windows** is an open risk for the M3 generative path
  (not M2). Flagged in DECISIONS; validate when M3 starts.

## 7. Next (session 8)

Start the real M2 capture path on this box: install COLMAP + smoke it; run
MapAnything on a real orbit video (founder to film a household object — see
`docs/eval/CORPUS.md`); wire L0→L1 from real data; report measured scale/Chamfer
(the first *ground-truth* numbers, replacing the self-consistency smoke). Then
the TRELLIS import check + R-T1 distillation for the generative path.
