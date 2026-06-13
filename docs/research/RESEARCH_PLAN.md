# Phase R — Research Plan

*Session 1, 2026-06-12. This is the spec-mandated first output (CLAUDE.md §2, §10.1). Budget: ~2–3 sessions total for Phase R.*

## Goal

For every stage of the AURIGA pipeline, choose a primary technique and a runner-up, with
evidence: current capability, maintenance status, **license** (commercial use is mandatory —
flag anything non-commercial), VRAM/runtime footprint on our hardware (2×RTX 4090 24 GB,
3×RTX 3080), and integration cost. Final deliverable: `DECISIONS.md` + positioning one-pager +
risk register (M0 exit criteria, CLAUDE.md §9).

## Method — the verify-latest protocol

Training-data knowledge is stale by definition (spec §10.1). For every candidate:

1. **Web-verify now**: repo activity (last release/commit), license file *as of today*, open
   issues that reveal dealbreakers, successor projects that obsolete it.
2. **Prefer permissive** (Apache-2.0/MIT/BSD). Non-commercial code (Inria 3DGS license,
   CC-BY-NC, research-only model weights) may inform *design* but cannot ship; every NC item
   gets an explicit flag and a permissive alternative or a reimplementation note.
3. **Distinguish code license from weights license** — they frequently differ.
4. Record citations (paper + repo URL) inline in each research note.

## Research areas → documents

| # | Area | Key questions | Output |
|---|---|---|---|
| RA1 | Core 3DGS + surface-accurate variants | Is gsplat the rasterization backbone? Which surface regularization (2DGS/SuGaR/RaDe-GS/GOF/PGSR) for L3, and is a permissive implementation available? Anti-aliasing, densification fixes. | `01-core-and-surface.md` |
| RA2 | Generative splats | Strongest open text/image→3D recipe today (TRELLIS-class and successors); feed-forward gaussian reconstructors (LGM-class) for L2; current multi-view diffusion SOTA; weights licenses. | `02-generative.md` |
| RA3 | Capture: photos/video → geometry | SfM (COLMAP/GLOMAP) vs pose-free feed-forward (DUSt3R/MASt3R/VGGT-class — license minefield); metric scale via monocular metric depth; dynamic video (4DGS-class) for L7. | `03-capture-video.md` |
| RA4 | Physics, lighting, print path | MPM on gaussians (PhysGaussian/Taichi) for L6 preview; BRDF decomposition for L4; splat→SDF→watertight for L5/print; convex decomposition (CoACD/V-HACD). | `04-physics-lighting-print.md` |
| RA5 | Formats, compression, engines | .spz/.sog/KSPLAT status; KHR_gaussian_splatting glTF standard status; web renderers (Three.js/PlayCanvas); Unity/UE5 plugin landscape; coordinate-convention table. | `05-formats-engines.md` |
| RA6 | Competitors & positioning | Current state (June 2026) of Meshy, Tripo, Rodin, Luma, Polycam, World Labs Marble; the open square AURIGA claims. | `06-competitors-positioning.md` |
| RA7 | Orchestration & infra | Temporal vs Celery+Redis for resumable multi-stage GPU workflows; queue-per-stage; progress events. (Desk decision — light verification.) | folded into `DECISIONS.md` |

## Session budgeting

- **Session 1 (this one)**: RA1–RA6 ecosystem sweep (web verification of every load-bearing
  dependency), draft `DECISIONS.md` v0.1, positioning one-pager, risk register v0.1.
- **Session 2**: deep-read the chosen papers/repos (not summaries — actual method sections and
  code), close open questions in DECISIONS.md, finalize per-layer accuracy targets and metric
  definitions, RA7 final call, name proposal.
- **Session 3 (if needed)**: prototype-level smoke checks on the 4090 box (install gsplat, run
  a reference reconstruction, validate the chosen feed-forward checkpoint loads) — exit Phase R
  with verified-runnable choices, not paper choices.

## Decision criteria (in priority order)

1. License-clean for commercial SaaS + self-host distribution.
2. Geometric accuracy potential (surface fidelity, normals) — the brand promise.
3. Runs on 24 GB VRAM for the self-host minimum (spec §6).
4. Maintenance health & community momentum.
5. Integration cost into a typed, tested Python/CUDA codebase.
