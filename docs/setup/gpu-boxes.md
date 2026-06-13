# GPU Box Setup

Two machines carry all GPU work (dev box has only a 4 GB Quadro):

| Box | Hardware | Tailscale IP | Role | Status (2026-06-12) |
|---|---|---|---|---|
| **A** | 2× RTX 4090 (24 GB), 128 GB RAM, Threadripper | `100.87.142.33` | Refine pool, model experiments, TRELLIS/MapAnything | Pings OK; **SSH closed** — needs setup |
| **B** | 3× RTX 3080 (10–12 GB), 128 GB RAM | `100.70.127.42` | Preview pool (L0–L2), CPU-heavy stages (SfM, SDF, convex decomp) | Pings OK; **SSH already open** — needs username |

Both are Windows. Decision ([DECISIONS.md](../research/DECISIONS.md) §product C6): **WSL2,
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

## Notes

- Disk: model checkpoints + datasets want ≥150 GB free per box, NVMe preferred. Check before step 2.
- Tailscale must auto-start on both boxes (it's the only network path the agent uses).
- Box B's 3080s (10–12 GB) cannot run TRELLIS.2 (needs 24 GB) — it gets preview/CPU duties.
- Power/thermals: long refine runs at full TGP; ensure the 2×4090 box has adequate cooling/PSU
  headroom (founder judgement; the agent staggers jobs across GPUs by default).
