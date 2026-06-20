# 18 — Post-M6 Roadmap: Launch Hardening, GPU-Real Upgrades, and the Fine-Tuning Track

*Written 2026-06-19 (session 29, end of M6). The build plan (CLAUDE.md §9) defines
M0–M6 and M6 is now implemented + verified — there is no M7 in the plan. This doc
defines the phase **beyond** M6: what it takes to go from "feature-complete +
tested at the library/producer level" to "launched, telemetry-driven, and
training our own models." It supersedes the "Next" pointers in earlier retros.*

Three parallel tracks. Track **N** (launch/infra) gates a public launch. Track
**G** (GPU-real) closes the honest gaps where M1–M6 scaffolded the torch/GPU
boundary. Track **T** (training/fine-tuning) is the §6 "later" item — explicitly
deferred until telemetry justifies it, planned here so it isn't a cold start.

---

## Track N — Launch hardening (the actual blockers)

These are infra, not research. They block a public/enterprise launch and are
mostly P0 in [LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md).

| # | Item | Why it blocks launch | Rough size |
|---|---|---|---|
| N1 | **CI pipeline** (`.github/workflows`) | ✅ **Wired** (this pass): `ci.yml` runs ruff·mypy·pytest (9 libs · api · sdk-python · loadtest) + tsc-b·eslint·vitest (web + TS pkgs) on push/PR; `gpu.yml` runs the CUDA gates on a self-hosted runner (manual). Authored + YAML-valid, mirrors the manual gates; **not yet executed on a remote** (repo is local-only). Remaining: push to a GitHub remote + confirm the first green run. | done (pending remote) |
| N2 | **`.astel` deserialization hardening** | Untrusted packages are parsed by `astel_format` (zip+json) and `astel_dynamics` (binary). The L7 `.bin` reader is now size-validated (s29); audit zip-bomb / path-traversal / oversized-accessor vectors across the whole package read path + add a `validate_untrusted()` entry point. | 1 session |
| N3 | **Production deploy validation** | Helm/K8s manifests + GPU-worker autoscale-on-queue-depth are designed but never load-validated. Run `tools/loadtest` against a real (stub then GPU) deploy; tune concurrency/timeouts; document rollback. | 1–2 sessions |
| N4 | **Monitoring / alerting / rollback runbook** | No SLIs/SLOs, no dashboards, no incident runbook. Define per-stage latency/error/$-cost SLIs (the per-stage telemetry already logged), wire metrics export, write the runbook. | 1 session |
| N5 | **Engine CI runners** | Unity/UE5 plugins are code-complete but never compiler-verified (no licensed runners). Add licensed build runners; compile-test plugins against a real `engine.json`. (Carried from s27/s28.) | infra-gated |

**Recommended order:** ~~N1~~ (wired) → N2 → N4 → N3 → N5. CI first so everything
after is guarded; security before any public exposure; monitoring before
load-validating the deploy.

---

## Track G — GPU-real upgrades (close the honest scaffolding gaps)

Each M1–M6 milestone built a CPU-validated core + an honest GPU-deferred note.
These are those deferrals, now first-class work. All run on the 2×4090 box (Box A)
and scale to the cloud A100/H100 pool.

| # | Item | Current honest state | What "real" needs |
|---|---|---|---|
| G1 | **Real 4DGS video L7** | Video → static L3 + "dynamics not tracked" caveat; `write_dynamics_layer` + L7 binding exist + tested on synthetic motion. | Frame selection/deblur → pose-free per-frame recon (MASt3R/VGGT-class) → deformable-3DGS / 4DGS fit on gsplat → feed the fitted field into the EXISTING `write_dynamics_layer`. The whole downstream (pack, bind, manifest, web) is already wired. (DECISIONS row 27.) |
| G2 | **Text→multiview bridge** | Text→3D works via SDXL/FLUX→TripoSplat (single image). | A real multi-view-diffusion stage (MV-Adapter / current SOTA — RE-VERIFY latest at build time) so text yields consistent multi-view conditioning, not one image. Modality #1 quality upgrade. (DECISIONS rows 22–23, both 🟡.) |
| G3 | **Live LOD streaming in the viewer** | `lod.ts` consumer + producer `l3.lod.json` tiers exist + tested. | Wire `selectTierForPlatform` into the `SplatScene` render loop: fetch the right tier file, hot-swap on budget/viewport change, progressive nested-tier upgrade. |
| G4 | **Scene generation end-to-end** | Layout-LLM + `compose_scene` cores exist + tested. | API producer stage: prompt → `build_scene_layout` → per-object generation (reuse the single-object path) → `compose_scene` → multi-object `.astel` (kernel_batches per object). Attack Meshy's single-object ceiling. |
| G5 | **L6 MPM physics sandbox / L4 GPU inverse-render** | L4 = achromatic low-freq decomposition; L6 sandbox = single rigid body. | The §3 upgrades: PhysGaussian MPM-on-gaussians over Warp (DECISIONS row 28, math settled) using the L5 SDF for interior fill; deferred-PBR-on-gsplat inverse render for L4 (row 30). Same data contracts. |

**Recommended order:** G1 (the headline M6 feature made real) → G3/G4 (cheap, the
cores are done) → G2 → G5. G1 and G2 are the two that most change the product
demo; G3/G4 are low-effort high-visibility because the hard parts already ship.

---

## Track T — Training / fine-tuning (the §6 "later", planned not started)

CLAUDE.md §6: *"Model training/fine-tuning (later): multi-node H100s; defer until
product telemetry justifies it — launch on adapted open checkpoints."* We launch
on TripoSplat (L2) + SDXL/FLUX + 2DGS (our code) — all permissive. Fine-tuning is
**not** started now; this is the plan so it's a warm start when the gate is met.

**Gate to START training (all must hold):**
1. Track N done (we have telemetry + a deploy that produces a labelled corpus).
2. ≥ ~10k real generations with Truth-Meter quality signals + user accept/reject
   logged (the credit ledger + quality reports are the corpus seed).
3. A measured, recurring quality deficit that a fine-tune would fix (e.g. a class
   of prompts where held-out PSNR / geometric error is consistently poor) — not a
   speculative "training would be nice."

**Candidate fine-tunes, in expected ROI order:**

| T# | Target | Data | Hardware | Payoff |
|---|---|---|---|---|
| T1 | **L2 generative prior (TripoSplat)** domain-adapt | Our accepted generations + their multi-view renders (self-distillation) + any licensed 3D corpus | multi-node H100 (LoRA/full per scope) | Better identity/geometry on Astel's actual prompt distribution; the single biggest quality lever (it seeds every asset). |
| T2 | **Multi-view diffusion** (the G2 bridge model) | Curated text→multiview pairs from accepted assets | H100 | Sharper, more consistent conditioning → fewer floaters at L3. |
| T3 | **Metric-scale / monocular-depth** head | Capture-path assets with SfM/EXIF scale ground truth | A100 | Tightens the Truth-Meter scale CI — our trust differentiator. |
| T4 | **L6 material/semantics** small model | LLM-labelled region/material data we already generate | A100 / even fine-tune a small open VLM | Cuts the per-generation LLM token cost (§5 budget) + offline capability. |

**Data/licensing discipline (carry from [LICENSE_AUDIT.md](LICENSE_AUDIT.md)):**
only train on permissively-licensed or self-generated data; never on NC corpora
(T&T/CO3D rejected earlier for exactly this). Log provenance of every training
sample. The honesty contract extends to models: a fine-tuned checkpoint's
eval-vs-baseline delta must be measured + published in `DECISIONS.md`, not
assumed.

**Cost flag (CLAUDE.md §10.2):** multi-node H100 training crosses the >$1k/mo
threshold — it is a **founder decision** with a written cost estimate before any
run. Nothing here spends until that approval + the start-gate above are both met.

---

## Single recommended next step

**Track N1 is now wired** (`.github/workflows/ci.yml` + `gpu.yml`). The immediate
follow-through is to **push the repo to a GitHub remote so CI actually executes**
and confirm the first green run — that converts the "all gates green" claim from a
manual ritual into an enforced invariant, the precondition for every other track.
After that: **N2** (`.astel` deserialization hardening) then **G1** (real 4DGS
video, the M6 feature made real).
