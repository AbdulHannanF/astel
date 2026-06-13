# Session 1 Retro — 2026-06-12 — Phase R kickoff

## What was produced

- Repo initialized at `E:\Downloads and Agreements\AURIGA` (user's choice), spec → `CLAUDE.md`,
  competitor analysis → `docs/meshy-analysis.md`.
- `docs/research/RESEARCH_PLAN.md` (the spec-mandated first output) and six research notes
  (RA1–RA6), every load-bearing dependency **web-verified on the day** per the verify-latest
  protocol.
- `DECISIONS.md` v0.1 — 20 stage/stack decisions (7 final, 10 draft, 3 deferred) + binding
  architecture decisions + license policy.
- `RISKS.md` v0.1 — 17 risks with mitigations.
- Positioning one-pager inside RA6.

## Headline findings

1. **The permissive stack exists end-to-end.** gsplat + 3dgrut (Apache) for splatting;
   TRELLIS/TRELLIS.2 (MIT) for generation; MapAnything `-apache` (Apache) for metric pose-free
   capture; MoGe-2 (MIT) for scale; Spark (MIT) for the viewer; Open3D/trimesh/CoACD for L5.
   The entire Inria-derived NC lineage is avoidable.
2. **TRELLIS.2 is mesh-only** — turned into an opportunity: internal O-Voxel prior distilled
   into surface splats (spec-legal scaffolding). This is the riskiest bet → first GPU experiment.
3. **DUSt3R/MASt3R are NC** — MapAnything (Apache ckpt, metric, May 2026 active) replaces them
   outright. Lucky timing.
4. **KHR_gaussian_splatting ratifies ~Q2 2026** — our standards bet lands exactly on schedule.
5. Hardware reality: dev box is GPU-weak; user's 2×4090 (Tailscale `100.87.142.33`, SSH not yet
   enabled) and 3×3080 (`100.70.127.42`, SSH already open) carry GPU work from session 3.

## What went well / what to fix

- ✅ Verify-latest protocol caught three things training knowledge had wrong or missing
  (TRELLIS.2 existence + mesh-only nature; MapAnything Apache checkpoint; VGGT relicense).
- ✅ One AskUserQuestion batch up front, autonomous after — matched user preference.
- ⚠ License verification is breadth-first only; ~12 ⚠ items remain (master list = RA-doc
  "open questions"). **Session 2 must close all of them before any M1 code.**
- ⚠ No paper deep-reads yet (method sections) — that is session 2's core job per RESEARCH_PLAN.

## Next session (Phase R, session 2)

1. License audit: close every ⚠ in DECISIONS.md → `LICENSE_AUDIT.md` v1.
2. Deep-read: 2DGS, PGSR, TRELLIS (SLAT), MapAnything, PhysGaussian, RTR-GS method sections;
   finalize 🟡 decisions; define per-layer accuracy metrics + targets.
3. Temporal vs Celery spike decision; `.auriga` manifest schema draft (provenance channel!).
4. Name proposal + USER decisions: GitHub remote? rename repo?
5. If GPU box ready: session-3 smoke tests move up (gsplat install, MapAnything orbit test,
   TRELLIS.2 distillation experiment).

## USER setup list (for GPU sessions)

On the **2×4090 box** (`100.87.142.33`, Windows by TTL):
1. Enable OpenSSH Server (Settings → System → Optional Features → OpenSSH Server; start service,
   set to automatic) — port 22 is currently closed.
2. Install **WSL2 + Ubuntu 24.04** (`wsl --install -d Ubuntu-24.04`) — the ML stack (gsplat,
   TRELLIS, MapAnything) is Linux-first; NVIDIA CUDA works natively in WSL2.
3. Current NVIDIA driver (Game Ready or Studio ≥ latest); no CUDA toolkit needed on Windows
   side (WSL2 handles it).
4. Confirm Tailscale stays logged in / auto-starts.

On the **3×3080 box** (`100.70.127.42`): SSH already reachable — send me the username (and
confirm it's Windows or Linux); same WSL2 + driver steps if Windows.
