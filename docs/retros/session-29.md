# Session 29 retro (2026-06-19)

**M6 IMPLEMENTED — the final milestone: L7 dynamics core, scene seeds, LOD
streaming, and a hardening/security/launch pass.** M6 is the last milestone in
the build plan (§9); with it, M0–M6 are all closed. Run per the founder
directive: **Opus planned + reviewed/verified; Sonnet implemented; Haiku did the
mechanical doc.** Every subagent report was verified on disk + by re-running
gates myself — no summary was trusted (the session-23/28 rule).

## 0. Scope decision

M6 = "video→4DGS L7; scene seeds; LOD streaming; hardening + load tests +
security review + launch checklist." This is four distinct subsystems. I built
each the way every prior milestone built its GPU-boundary work: a **torch-free,
CPU-validated core**, **wired into the producer/package/web**, with **GPU-real
work honestly scaffolded + flagged** (the L5-session-18 pattern). Ground truth
verified before building: the `.astel` format ALREADY had the L7 `dynamics`
schema slot (JSON schema + Pydantic), and the producer was aliasing the video
modality to the render-then-refit smoke (caveated since s22, but still not a real
video path).

## 1. L7 Dynamics — `libs/astel_dynamics` (new, 40 tests)

Torch-free low-rank **Linear-Blend-Skinning deformation field**: K control nodes
(farthest-point sampled), per-gaussian Gaussian-RBF blend weights, per-frame
per-node weighted-least-squares **affine** transforms. `Timeline`, binary
`pack` (documented little-endian layout, lossless f32 round-trip), `baked.py`
(per-frame baking). **Validated against analytic ground truth** (the L5-s18
move): global rigid rotation (K=1) → mean err **1.6e-8** of scale; low-rank bend
(K=8) → **1.5%**; incompressible random per-point motion (K=8) → **9.0%** —
honestly large, ≥5× the bend, proving the fitter does NOT fabricate accuracy on
motion LBS can't compress. `FitReport` carries the REAL measured reconstruction
error, never a bound.

**Bound into `.astel`** (astel_format builder, +7 tests): `build_minimal_package`
gained `l7_deformation_path`/`l7_timeline_path`/`l7_representation` (all-or-
nothing validated), emitting a `dynamics` layer mirroring the L5/L6 pattern;
packages are byte-identical when L7 is absent. **Producer** (`packaging.py`):
public `write_dynamics_layer(field, timeline, out_dir)` writes
`l7-deformation.bin` + `l7-timeline.json`; `write_layer_stack` threads the L7
paths through. **Video honesty fix** (`produce.py`): `_produce_video` now runs
the REAL static reconstruction on a supplied frame (honest origin note: "static
L3; L7/4DGS tracking NOT performed — requires the GPU deformable-recon stage"),
or the caveated smoke fallback. **No fabricated motion** — L7 stays absent on the
video path; the capability exists + is tested, real per-frame tracking is the
deferred GPU stage.

## 2. Scene seeds — `libs/astel_scene` (new, 56 tests)

Torch-free multi-object composition operating on raw numpy gaussian-field arrays
(the `astel_appearance` convention). `SceneLayout`/`SceneObject`/`Placement`
schema (`astel.scene-layout/v0`); `apply_placement` (yaw about +Y, uniform
scale, translate — rotates gaussian quats + shifts log-scales); robust
**1st-percentile ground-drop** (ignores stray sub-base splats); greedy +X
**XZ-AABB no-overlap** resolution; `compose_scene` → combined cloud + per-object
index ranges. Tests assert real geometric conditions (each object's 1st-pct y at
ground; XZ AABBs disjoint). **Layout-LLM stage** (`llm_stage.py`, +17 tests of
the 56): reuses `astel_llm`'s offline `FixtureAdapter` (no key, no spend, like
the Generation Spec stage) → structured JSON → `SceneLayout`; honest single-
object fallback + zero-cost ledger on unseen prompts. Honest gaps: greedy
separator pushes +X only (documented); full API wiring of end-to-end scene
generation is a follow-on.

## 3. LOD streaming — `libs/astel_lod` (new, 53 tests) + producer + web

Importance = `opacity × exp(Σ log_scales)` (perceptual proxy, documented as
such). `generate_lod_indices` derives every tier from ONE global descending
sort, **structurally guaranteeing the nested-subset property** (top-k ⊂ top-K) so
a streaming client never re-downloads on upgrade — asserted across all tier
pairs. `TIER_BUDGETS` (lowpoly 100k / standard 1M / cinematic 5M) +
`PLATFORM_BUDGETS` (mobile/web/console/cinematic). **Producer** (`packaging.py`,
+7 tests): `_write_lod` always emits a "full" tier pointing at the master
`l3.ply` and writes `l3.lod.<name>.ply` + `l3.lod.json` for each budget strictly
< N (deduped; small clouds get only "full" — honest, no upsampling). **Web**
(`apps/web/src/viewer/lod.ts`, +20 vitest): `parseLodDescriptor` (strict-
ascending validation), `selectTierForBudget` (largest-fitting, smallest
fallback), `selectTierForPlatform`. Additive module — live render-loop wiring is
a follow-on.

## 4. Hardening / security / load test / launch checklist

**Security review (focused, by me).** New M6 attack surface = the untrusted-
`.astel` deserialization path. `read_deformation_bin` read N/K/F from the header
and sliced — Python slicing is memory-safe (no overread), but a tiny malicious
file could declare huge arrays and yield a confusing reshape error. **Fixed:** an
exact file-size check (`len(data) == 8 + header + n_floats·4`) rejects truncated,
trailing-junk, AND amplification-crafted headers up front with a clear message,
before any allocation (+2 adversarial tests: truncation, oversized-header). The
other surfaces (stdlib `zipfile`/`json` package reading; constant-named file
writes) are low-risk; the API itself was unchanged in M6. **Load-test harness**
(`tools/loadtest`): async httpx, semaphore concurrency cap, p50/p90/p99 latency,
`--health-only` probe, error-rate exit-gate; ruff+mypy clean + self-test (NOT
run end-to-end against a live server — stated honestly). **Launch checklist**
(`docs/LAUNCH_CHECKLIST.md`, Haiku-drafted from supplied facts): honest [x]/[ ]
across all subsystems + a P0/P1/P2 blocker list.

## 5. Honest finding: there is NO CI in the repo

`.github/` does not exist. The "green CI" referenced across retros/README is
aspirational — gates are run manually via `uv` / `pnpm`. Surfaced loudly in the
launch checklist (P0) rather than buried. Wiring real CI is a top launch blocker.

## 6. Process note (founder directive)

One Sonnet implementation agent (the load-test harness) **hit an account session
limit and returned no final report** — but it had already written the three
source files before dying. I verified them directly (read + ran ruff/mypy +
self-test) rather than trust a (nonexistent) summary, and completed the
remaining close-out myself. Two earlier Sonnet agents' "all green" reports were
independently re-verified by re-running every gate — all real this time.

## 7. Gates — all green (Opus-run, end-of-M6 sweep)

- **8 Python libs**: astel_dynamics **40** · astel_scene **56** · astel_lod **53**
  · astel_format **34** · astel_splat_io **37** · astel_appearance **25** ·
  astel_solid **37** · astel_eval **36** (= **318**), each ruff + mypy --strict.
- **pipelines/gpu**: ruff · mypy (43 files) · **112 passed, 3 skipped** (+13).
- **services/api** (untouched): ruff · mypy (26) · **71 passed, 1 skipped**.
- **apps/web**: **tsc -b** · eslint · **72 vitest** (+20).
- **@astel/manifest** · **@astel/sdk**: vitest green.
- **tools/loadtest**: ruff · mypy · self-test.

New tests this session: **+44** (dynamics 40, +7 format, +13 gpu, +17 scene-LLM
of the 56, +20 web, +2 security — overlaps counted in totals). `.gitignore`
covers all new `.venv`/cache dirs.

## 8. Honest gaps carried into post-M6

- L7 **real per-frame 4DGS tracking** from video is the deferred GPU stage; today
  video → static L3 + honest "dynamics not tracked" caveat. The L7 binding +
  `write_dynamics_layer` are real and tested, ready for that stage to call.
- Scene end-to-end + live LOD/scene **API/web render-loop wiring** are follow-ons
  (cores done + tested).
- **No CI, no production-deploy validation, no monitoring/rollback** — the real
  launch blockers (see [LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md) +
  [18-post-m6-roadmap](../research/18-post-m6-roadmap.md)).
- Engine plugins still not compiler-verified (no licensed runners; s27/s28 gap,
  unchanged).

**M6 is implemented + verified. The build plan is complete through M6. The next
phase is launch hardening + the GPU-real upgrades + the fine-tuning track —
see [18-post-m6-roadmap.md](../research/18-post-m6-roadmap.md).**
