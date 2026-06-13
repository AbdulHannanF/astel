# Risk Register — v0.1 (2026-06-12)

*M0 deliverable. Severity = impact × likelihood, H/M/L. Owner is "founding engineer" (the
agent) unless marked USER. Reviewed at every milestone retro.*

## Technical

| ID | Risk | Sev | Mitigation / trigger |
|---|---|---|---|
| R-T1 | **Prior-distillation bet fails**: TRELLIS.2 mesh prior → 2DGS distillation loses geometric fidelity or PBR quality (DECISIONS #2) | H | De-risk experiment is the *first* GPU task (session 3): image → prior → distilled surfels → Chamfer + PSNR vs prior renders. Fallback: TRELLIS-v1 gaussian head + heavier L3 refinement. |
| R-T2 | 2DGS surfels over-smooth fuzzy/volumetric matter (hair, foliage) — splats' showcase content | M | A/B vs 3DGS+opacity-field extraction on a fixed corpus in M2; layer-mixed representation (surfels for surfaces, 3D kernels for fuzz) is the architectural escape hatch — manifest schema must not preclude mixed kernel types. |
| R-T3 | MapAnything underperforms on object-centric orbits (it's scene-trained) | M | Session 3 smoke test on real orbit footage; COLMAP/GLOMAP fallback is wired in from day one. |
| R-T4 | Metric scale consensus diverges wildly on textureless/ambiguous captures | M | That's *why* the CI is reported, not hidden (Truth Meter). UX: user override with one tap; log override telemetry as training signal. |
| R-T5 | Splat→SDF→watertight produces non-printable junk on thin/fuzzy regions | M | Printability checks + SDF erosion analysis before slicing; refuse-with-report rather than silently emit garbage (spec §1.3 honesty). |
| R-T6 | Interactive Physics Sandbox latency (server-sim + stream) unusable over WAN | L | Cap preview to small MPM grids; precompute canned interactions; WebRTC data channels; it's a demo of L5/L6 truth, not a game engine. |
| R-T7 | 24 GB VRAM ceiling breached by TRELLIS.2 (needs ≥24) + pipeline overhead co-resident | M | Stage isolation: one model resident per worker process; sequential stage scheduling on the 4090 box; the 2×4090 topology helps (model per GPU). |
| R-T8 | **Consumer-perceived quality below Meshy** despite TRELLIS.2-class raw model: cleanliness, reliability, retry behavior, print success are pipeline-maturity properties that take iteration | H | Finishing pipeline is the named consumer-quality workstream with M2/M3 acceptance metrics ([RA7b](07-free-tier-consumer-strategy.md) C4); blind-eval harness from M1 measures the real gap instead of guessing; launch positioning never rests on text-to-3D parity (C5). |

## Licensing / legal

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| R-L1 | A "permissive" dependency turns out NC on close read | H→M | **Materialized & contained 2026-06-12**: nvdiffrast/nvdiffrec (TRELLIS.2 deps) confirmed NVIDIA-NC; usage boundary + fallbacks defined in [LICENSE_AUDIT.md](LICENSE_AUDIT.md). All other audited deps clean. Remaining exposure: clone-time import-graph checks (session 3); CI license gate from M1. |
| R-L2 | Model weights trained on disputed data create downstream exposure for paying customers | M | Prefer vendors publishing training-data statements (Meta's `-apache` checkpoint exists precisely for this); enterprise tier docs state weight provenance; indemnification deferred to legal review pre-launch. |
| R-L3 | KHR_gaussian_splatting changes between RC and ratification | L | Exporter behind a versioned adapter; .ply/.spz carry products until ratified. |

## Competitive (from [RA6](06-competitors-positioning.md))

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| R-C1 | World Labs descends from worlds to assets | M | Speed on M2 measured-accuracy story + L5/L6 semantics; their center of gravity is worlds/spatial-AGI. |
| R-C2 | Meshy adds splat export as a checkbox | M | Layered accuracy + physics + Truth Meter is not a checkbox; our wedge is what their pipeline *can't* measure. |
| R-C3 | Meshy moves into physics-aware assets (founder = Taichi author) | M | Mesh representation still blocks their photoreal/volumetric ceiling; we get there first on splats — claim the territory publicly early (positioning doc). |

## Operational (USER-visible)

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| R-O1 | GPU boxes unavailable/unconfigured blocks sessions 3+ | M | USER: setup list delivered end of session 1 (SSH + driver/CUDA baseline). Until then, all work is CPU/docs/skeleton-safe. |
| R-O2 | Free-tier budget blocks LLM-layer prototyping (Generation Spec, L6 reasoning) | L | Design adapter + cached fixtures first; record per-call token costs in the credit ledger design from day one; ask USER before any paid usage (per agreement). |
| R-O3 | Repo lives in `E:\Downloads and Agreements\` (space in path, "Downloads" cleanup risk) | L | Git from day one (done); push to a remote (GitHub) early in M1 — USER decision pending; path-with-space issues addressed per-tool as they appear. |
| R-O4 | Solo-founder bus factor / agent session continuity | M | Everything written down (research docs, DECISIONS, retros); memory files maintained; every milestone ends green-CI + demo + retro (spec §9). |
| R-O5 | Generous free tier becomes a cost sink at public scale (GPU-hours for free users) | M | Preview-tier economics by construction (L0–L2 cheap); founder hardware carries closed beta at $0 cloud; local mode offloads power users; re-model unit economics before public launch (gate in M6 checklist). |
