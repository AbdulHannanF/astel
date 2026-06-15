# GPU Box Setup

Two machines carry all GPU work (dev box has only a 4 GB Quadro):

| Box | Hardware | Tailscale IP | Role | Status (2026-06-12) |
|---|---|---|---|---|
| **A** | 2× RTX 4090 (24 GB), 128 GB RAM, Threadripper (`THREADRIPPER-48`) | `100.87.142.33` | Refine pool, model experiments, TRELLIS/MapAnything | **ONLINE (session 7).** Native Windows; CUDA 12.9 + VS2026; gsplat builds + trains. Agent runs on the box directly. |
| **B** | 3× RTX 3080 (10–12 GB), 128 GB RAM | `100.70.127.42` | Preview pool (L0–L2), CPU-heavy stages (SfM, SDF, convex decomp) | Pings OK; **SSH already open** — needs username |

> **UPDATE 2026-06-14 (session 7):** the WSL2 plan below is **reversed for Box A** — see
> [DECISIONS.md](../research/DECISIONS.md) §"2026-06-14 — GPU stack: native Windows".
> WSL2 was hard-blocked on Box A (virtualization off in firmware). The GPU stack runs
> **native Windows** (CUDA Toolkit 12.9 + Visual Studio 2026 MSVC). No SSH/WSL bring-up
> was needed — reproduce the env with [`scripts/setup-gpu-env.ps1`](../../scripts/setup-gpu-env.ps1)
> and run gsplat commands via `pipelines/gpu/run-python.cmd`. The WSL2 procedure below is
> retained for Box B / historical reference only.

Both are Windows. Original decision (now superseded for Box A): **WSL2,
no dual-booting**. CUDA-in-WSL2 is mature; the founder runs one script once per box; the
agent does everything else over SSH (including all Ubuntu-side setup: drivers check, uv,
Python 3.11 env, CUDA toolchain inside WSL, repo clone, model downloads).

## Founder steps (once per box)

1. Copy [`scripts/setup-gpu-box.ps1`](../../scripts/setup-gpu-box.ps1) to the box, run as
   Administrator. It enables OpenSSH, installs WSL2 + Ubuntu 24.04, prints GPU/driver info.
2. Reboot, run `wsl` once (create the Linux user — any username/password).
3. If the NVIDIA driver is older than ~565: update via GeForce Experience / NVIDIA app
   (Game Ready or Studio — either is fine). Nothing CUDA-related to install on Windows.
4. Send the agent the **Windows username** of each box (and the Linux username if different).

## Agent steps (remote, automated — session 3)

1. SSH key install (`ssh-copy-id` equivalent for Windows OpenSSH; key generated on dev box).
2. Inside WSL: `uv` + Python 3.11 + CUDA userspace, clone repo, install gsplat (wheel or
   source build), pull MapAnything-apache + TRELLIS v1 + MoGe-2 checkpoints (~30 GB — confirm
   disk headroom on each box first).
3. Smoke tests, in order (see [LICENSE_AUDIT.md](../research/LICENSE_AUDIT.md) 🔍 items):
   1. `nvidia-smi` inside WSL (CUDA passthrough sanity).
   2. gsplat training run on a reference dataset (one 4090).
   3. MapAnything on an orbit video of a real object (filmed by founder — any household object).
   4. TRELLIS v1/v2 import-graph license check + image→gaussian generation.
   5. The distillation experiment (R-T1 de-risk): image → TRELLIS.2 geometry prior →
      2DGS fit → Chamfer + PSNR report.

## COLMAP (SfM front-end) — installed on Box A, session 8 (2026-06-14)

COLMAP **4.1.0.dev0** (official `colmap-x64-windows-cuda.zip`, CUDA build) is
installed to `D:\Astel\tools\colmap\` (the `tools/` dir is gitignored — binaries
never go in git). Reproduce:

```
curl -sL -o tools/colmap-cuda.zip \
  https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip
# (4.0.4 is the release tag; the bundled binary self-reports 4.1.0.dev0)
Expand-Archive tools/colmap-cuda.zip tools/colmap
tools/colmap/bin/colmap.exe help     # smoke: prints version + command list
```

The binary launches cleanly with CUDA on this box. A **functional** SfM smoke
(feature_extractor → exhaustive_matcher → mapper → registered-pose count) is
deferred to the real-capture session — it needs textured real images (founder's
orbit videos) to be representative; low-texture synthetic renders are not.

## Notes

- Disk: model checkpoints + datasets want ≥150 GB free per box, NVMe preferred. Check before step 2.
- Tailscale must auto-start on both boxes (it's the only network path the agent uses).
- Box B's 3080s (10–12 GB) cannot run TRELLIS.2 (needs 24 GB) — it gets preview/CPU duties.
- Power/thermals: long refine runs at full TGP; ensure the 2×4090 box has adequate cooling/PSU
  headroom (founder judgement; the agent staggers jobs across GPUs by default).
