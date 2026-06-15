# Session 8 retro (2026-06-14)

**The Truth Meter got its first real geometry number, and the SfM front-end
(COLMAP) is installed. The M2 capture path's *machinery* is now validated on
controlled ground truth; the *real-world* accuracy numbers still await the
founder's orbit videos (the one honest dependency).**

Mode (per founder directive — Opus plans/decides/verifies, Sonnet implements,
Haiku small tasks):
- **Opus**: planning, scope decisions, all verification (re-ran every claim
  from a clean shell), and the docs. Caught and fixed a scene-scale framing bug
  that was making the headline metric meaningless (see §3).
- **Sonnet ×1**: built the synthetic ground-truth harness (`metrics.py`,
  `synthetic.py`, `synthetic_eval.py` + tests). Reported honestly, including the
  large-Chamfer anomaly it could not fully explain.
- Note: `SendMessage` to continue a warm agent is unavailable in this harness,
  so the two small follow-up corrections (opacity-filtered Chamfer; the
  scene-scale fix) were done inline by Opus rather than paying a cold re-spawn —
  the bulk implementation stayed with Sonnet, honoring the directive's intent.

## 1. First: session 7 re-verified clean

Before new work, independently re-ran session 7 from a clean shell (`cl` absent):
CPU api gates green (ruff · mypy · 28 passed/1 skipped); GPU env + `produce` e2e
via the launcher produced `l3.ply` + an honest measured report; gpu pytest 5
passed. The session-7 stack is solid. (The continuation summary was a stale
mid-session checkpoint; the round-B launcher fix had in fact landed.)

## 2. What landed (all measured on this hardware)

- **Synthetic controlled-ground-truth eval** (`astel_gpu.synthetic_eval`):
  renders a KNOWN sphere-shell object (longest axis 0.20 m by construction) from
  known poses, refits a fresh gaussian cloud, and measures REAL Chamfer (mm) +
  scale against the known geometry. Produces the **first non-`None`
  `geometric_error`/`scale`** in the quality-report pipeline.
  - `metrics.py` — pure-torch bidirectional Chamfer (CPU-testable, no gsplat).
  - `synthetic.py` — deterministic sphere-shell GT cloud at a fixed 0.20 m scale.
  - Honesty: the API's GPU producer (`astel_gpu.produce`) is UNCHANGED — its
    `geometric_error`/`scale` stay honestly `None` (no ground truth). The
    synthetic report's caveats state plainly it is a controlled measurement, not
    real-world capture accuracy.
- **Measured baseline** (1500 iters, 4 k gaussians, 10 views, RTX 4090): PSNR
  6.7→32 dB; surface **coverage ≈ 15 mm**, **precision ≈ 165 mm** (opacity-
  filtered) / 220 mm (raw) on the 0.20 m object. Honest read: raw 3DGS covers
  the surface but leaves floaters — concrete evidence motivating the surface-
  aligned L3 representation (2DGS/PGSR). Recorded in DECISIONS.md (session-8 §).
- **COLMAP 4.1.0.dev0 (CUDA)** installed to `tools/` (gitignored); launches
  cleanly with CUDA on Box A. Install smoked; functional SfM reconstruction
  deferred to real captures (see §4).
- All gates green: `pipelines/gpu` ruff · mypy(10 files) · **11 pytest** (via
  launcher); CPU `api` unchanged and green.

## 3. The bug Opus caught (again, why verification matters)

Sonnet's harness reported a huge, asymmetric Chamfer (raw symmetric ~700 mm on a
200 mm object) and honestly flagged it couldn't fully explain it. Root cause on
inspection: the harness reused the smoke test's camera rig (`radius=3.0` world
units, tuned for its ~unit-scale torus) and init spread (±1.5 m) against a 0.20 m
object — so the object was a ~7% speck in-frame and the refit was wildly
under-constrained. The Chamfer was measuring **framing**, not geometry. Two
fixes: (a) **opacity-filter** the headline Chamfer to surface-defining gaussians
(raw kept as `chamfer_raw_all_means_mm`); (b) **metric-align the scene** —
camera orbit ≈ 2.5× the object, init spread ≈ object size. After the fix the
numbers became meaningful (coverage 15 mm / precision 165 mm) and the opacity
filter measurably helps (symmetric 117→91 mm). Methodology lesson recorded in
DECISIONS.md: capture-path evals must be metric-aligned or the error is an
artifact.

## 4. Honest gaps / carried forward

- **No real-world capture numbers yet.** The synthetic eval validates the
  measurement machinery and gives a baseline; the *ground-truth real-world*
  scale/Chamfer (the actual M2 deliverable) needs the founder's orbit videos.
  This is the one true dependency — unchanged from session 7.
- **COLMAP functional SfM smoke deferred** to real images (low-texture synthetic
  renders aren't a representative SfM test). MapAnything (L0/L1 feed-forward)
  likewise awaits real captures. Both are infra-ready.
- **torch-2.11 launcher fragility** (session-7 carryover) NOT yet hardened —
  runtime still needs the VS compiler on PATH via `run-python.cmd`. Still a
  session-9 candidate (AOT gsplat wheel or stable torch). The stack works
  reliably via the launcher; this is robustness debt, not a breakage.
- **L3 2DGS-vs-3DGS+GOF A/B** still open — the synthetic baseline strengthens the
  case for surface regularization generally but doesn't pick the variant (needs
  fuzzy real content on GPU).

## 5. Next (session 9 — the real M2 capture path)

The high-value, founder-gated step: film the orbit videos (`docs/eval/CORPUS.md`
§capture), then run COLMAP/GLOMAP + MapAnything on them, wire L0→L1 from real
data, and report the first *real-world* measured scale/Chamfer — the numbers the
synthetic harness was built to be compared against. Then the TRELLIS import-graph
license check + R-T1 distillation de-risk for the generative path.
