# NEXT_STEPS — Runway (updated 2026-06-15, end of session 22)

> **Session 22 done — TEXT→3D SHIPPED + verified live; full "nothing-unplugged"
> audit; honesty hardening; physics bug fixed.** The headline modality that was
> silently missing now works end-to-end. New `astel_gpu.text_to_image` stage:
> **prompt → canonicalized prompt → local text-to-image (default SDXL base 1.0,
> open/no-login/no-spend; FLUX.1-schnell Apache opt-in via `ASTEL_T2I_MODEL`) →
> TripoSplat L2 → 2DGS L3 → full `.astel` stack** (`diffusers
> AutoPipelineForText2Image`, no custom CUDA build, TripoSplat does its own bg
> removal so no segmentation stage). **Verified live on Box A** ("a worn brass
> astrolabe on a wooden base" → clean SDXL `text-reference.png` → 65,536-gaussian
> L2 → 2DGS L3 held-out **22.55 dB** → L5 solid → 12-artifact contract). Runs
> Opus-planned / Sonnet-implemented per founder directive.
> **Audits (so it can't recur):** [doc 15 wiring matrix](research/15-pipeline-wiring-audit.md)
> found the same silent-fallback class twice more — **video also aliased to the
> text-smoke path**, and the **stub SSE engine streamed "Asset ready" with fake
> metrics even on zero-artifact failures**. [doc 16 dead-code](research/16-dead-code-audit.md)
> → deleted `triposplat_spike.py` + `experiments/task-engine-spike/` (both
> graduated). **Honesty hardening:** SSE now reflects the REAL outcome (FAILED on
> no artifacts, real splat count on success; `Generation` gains
> `produced`/`splats`/`production_error`, migration `d4e5f6a7b8c9`); new
> structured **`conditioning`** field on `GenerationResource` (the guard that
> would've surfaced the text gap); producer logging; billing dead-config CI
> guard; honest web conditioning badge + real progress metrics. **Physics fix:**
> negative principal moments of inertia (impossible) when COM is far from origin
> — `compute_mass_properties` now integrates in the COM frame (no parallel-axis
> cancellation) + regression test. **Gates all green:** API ruff·mypy·**59
> pytest** · web **22 vitest** · manifest **10** · libs **98** · GPU
> ruff·mypy·**60 pytest**. **Tracked follow-up:** the typed-package `origin` enum
> (audit §2.4) is a versioned schema change for M5 plugins — done together with
> the v0-dict origin taxonomy + web pill in one pass, not rushed here. See
> [session-22 retro](retros/session-22.md). **Next: M4 — L6↔L5 mass join, bind
> L6 into `.astel`, origin-enum taxonomy pass, L4 relighting, metric-scale L5,
> CoACD+`.3mf`.**
>
> **Session 21 done — M3 VERIFICATION + MVP-polish pass; founder's "can I
> prompt it?" answered honestly; M4 entered (L6 physics-material on fixtures).**
> Full-stack re-verification at the founder's request: **every gate re-run
> green** (API ruff·mypy 23·51 pytest+1skip; web 18 vitest·tsc·lint;
> @astel/manifest 10; libs 87 = 14+10+16+11+36; GPU ruff·mypy 34·55 pytest+2skip)
> — and **two real regressions fixed in the act of verifying**: (1) the session-20
> billing migration tripped ruff E501; (2) the two GPU tests *hard-failed* under a
> plain `uv run pytest` on Box A (CUDA present but no MSVC on PATH → gsplat JIT
> error) — now they skip cleanly via a shared `requires_gsplat_runtime` fixture
> (new `conftest.py`), so the documented command is green everywhere and only
> *really* runs through `run-python.cmd`.
> **The honest answer to "can I give it a text prompt for model generation?":
> not for geometry that matches your prompt — not yet.** Verified live over HTTP:
> a text gen returns a valid layer stack + ledger, but the **shape is a
> prompt-independent procedural placeholder** (`origin: stub`) and the
> Generation Spec is `skipped` (no fixture/key). The **only** path that
> generates a real model from your input today is **image → TripoSplat L2 → 2DGS
> L3**, which I re-ran live on Box A this session (creature_butterfly, 200 iters:
> 65,536 gaussians, 11.1 s L2 / 3.3 s L3, held-out 19.0 dB, full contract incl
> `l5.stl`+mass). **text → 3D needs a text→multiview stage that is not built** —
> this is the real remaining M3-completion gap (see the recommendation at the end
> of this banner). Polish: stub/smoke quality reports now state explicitly that
> the geometry is *not* derived from the prompt; the web dock shows an honest,
> modality-aware hint of what each input actually produces. New guide:
> [MVP_TESTING.md](MVP_TESTING.md) (how to test today, both paths). No founder
> gate touched; no spend. See [session-21 retro](retros/session-21.md).
> **Recommendation for the founder:** the highest-value next build is arguably
> the **text→multiview bridge** (completes mission modality #1 so a text prompt
> yields a real model), *ahead of* finishing M4 — but per your instruction M4 is
> underway (L6 first). Say the word to reprioritize.
> **Next: continue M4 (L6 physics-material → L4 relighting → metric-scale L5 →
> CoACD+.3mf), or pivot to the text→multiview bridge.**
>
> **Session 20 done — M3 CLOSED: preview/refine credit-metering (billing
> semantics).** The generative path (s11–16) and Generation Spec stage (s15, 17)
> were already done; this session built the third/final M3 deliverable (build
> plan §9 M3: "preview/refine billing semantics"), so **M3 is now complete
> end-to-end.** New pure module `astel_api.billing`: the layer stack is metered
> as credits per CLAUDE.md §7 + meshy-analysis — **L0–L2 previews cheap (1/1/2),
> L3 the main spend (20), L4–L7 + print add-ons**, `1 credit == 1¢` (notional
> internal unit, no external spend / not a §10.2 cost item). Mirrors Meshy's
> two-stage model: `POST /v1/generations` gains `mode` (`preview`|`refine`,
> default `refine`) + optional `refine_of`; a **keyed refine bills only the L3+
> increment, never re-charging (or re-running the LLM spec for) the preview.**
> Every gen stores `credit-ledger.json` (`astel.credit-ledger/v0`) + returns a
> `billing` summary; `GET /v1/pricing` publishes the schedule; the measured
> Generation-Spec token cost folds in as an `LLM_SPEC` credit line (ceil ≥1
> credit). `generations` gains `mode`/`refine_of`/`credits` (Alembic
> `a1b2c3d4e5f6`, applies clean on a fresh DB). **Verified live over HTTP**
> (uvicorn): preview = 1 credit; standalone refine = 21 (L0+L3); keyed refine =
> 20 (L3 only); preview+keyed = 21 = standalone (no double-charge — the core
> invariant). **Honest gap:** the stub computes the full stack regardless of
> tier, so a preview has an unpaid L3 on disk → the ledger emits a caveat naming
> it rather than hiding it (the GPU path can tier-gate production later; billing
> is already correct for that). Also: this *prices* a generation, it does not yet
> *debit a balance* (needs accounts/auth — §7 follow-on). Gates green (API
> ruff·mypy 23·**51 pytest**, +16 new). No founder gate touched. Design doc:
> [architecture/billing.md](architecture/billing.md). See
> [session-20 retro](retros/session-20.md) + DECISIONS.md (§ session 20).
> **Next: M4 (L6 physics-material, L4 relighting, metric-scale L5, CoACD+.3mf).**
>
> **Session 19 done — L5 wired into the GPU producer: every generated asset now
> carries an `l5.stl` + mass properties.** `astel_gpu.packaging.write_layer_stack`
> derives `surfel_normals` from the L3 splats → `solidify` → writes `l5.stl` +
> `l5-mass.json` and threads a `solidity` block (volume / mass / COM / inertia
> diagonal + mesh & SDF stats) into the quality report. **Best-effort** (broad
> try/except like `.sog`): a cloud that won't solidify just skips L5 — never fails
> the asset (the surface is internal scaffolding; the asset stays splats, §1.2).
> `astel-solid` added as a torch-free GPU dep. **Verified on a real 65k cloud**
> (pirate-ship, self-consistency 28.6 dB): watertight 7,855-vert / 14,881-face
> mesh, `l5.stl` = 744,134 B = exactly `84+50·14881` (valid binary STL), volume
> 3.77 model-units³, **anisotropic inertia (4.61, 1.53, 5.42)** — physically
> correct (low about the long hull axis). Gates green (GPU ruff·mypy 33·**55 CPU
> pytest**, +1 seam). **Honest gaps:** mass/volume in MODEL units (metric grounding
> is a follow-on); L5 not yet a *bound* `.astel` manifest layer (loose artifacts +
> report block); star-shaped normal heuristic. See [session-19 retro](retros/session-19.md)
> + DECISIONS.md (§ session 19). **Next M4: L6 physics-material (reuse `astel_llm`);
> L4 relighting; metric-scale L5; CoACD + `.3mf` + printability; bind L5 layer.**
>
> **Session 18 done — M4 ENTERED: L5 solidification core (splat→SDF→watertight
> surface→mass properties→`.stl`), validated against analytic ground truth.** New
> torch-free, CPU-only lib `libs/astel_solid` (the print-path / physics-volume /
> collision spine from DECISIONS row 31): `oriented_point_sdf` (IMLS over scipy
> KDTree), `extract_isosurface` (skimage marching cubes, outward-wound),
> `compute_mass_properties` (volume/COM/inertia via divergence-theorem signed-tetra
> integrals), `write_binary_stl`, + `solidify`/`surfel_normals` (per-splat outward
> normal from the thinnest 2DGS axis). **Per §1.2 the derived surface is internal
> scaffolding only — never the asset.** Validated: **unit cube exact** (V=1, COM=0,
> I=diag(1/6) to 1e-6 — the math check); **sampled sphere r=0.5 through the full
> 64³ pipeline** V=0.5014 vs 0.5236 (4.2% low), COM≈4e-3, inertia ~14% low,
> near-isotropic — the honest discretization bias of a faceted MC sphere. Deps
> permissive (numpy/scipy/scikit-image, all BSD). Gates green (ruff·mypy 11·**10
> pytest**). No founder gate touched. **Honest gaps:** not yet wired into the
> producer/`.astel` package; `surfel_normals` uses the centroid outward heuristic
> (star-shaped only); Open3D-TSDF / CoACD convex-decomp / `.3mf` / printability
> checks deferred (row 31). See [session-18 retro](retros/session-18.md) +
> DECISIONS.md (§ session 18). **Next M4: wire L5 into the producer; L6
> physics-material LLM pass (reuse `astel_llm`); L4 relighting; then CoACD+`.3mf`.**
>
> **Session 17 done — M3 integration pt.2 (FINAL): Generation Spec LLM stage
> wired into the API text path.** The text pipeline now runs prompt →
> `GenerationSpec` on submit (new `astel_api.generation_spec_stage`): it stores
> `generation-spec.json` (spec + credit-ledger row) and threads the LLM's metric
> size estimate into the quality report's `scale` block (`method:"llm-estimate"`,
> confidence band) — the first non-`None` scale the Truth Meter can show for a
> generated asset, honestly flagged. **Founder gate R-O2 is double-gated:** offline
> `FixtureAdapter` by default (zero spend); live `AnthropicAdapter` only when BOTH
> `ASTEL_LLM_LIVE=1` AND `ANTHROPIC_API_KEY` are set, so a stray key can never
> trigger a paid call. Unseen prompts degrade gracefully to a `skipped` note.
> `astel-llm` added as an API dep (torch-free). Gates green (API ruff·mypy 21·**35+1
> pytest**, 5 new). No founder gate touched. **M3 integration is now COMPLETE in
> code** — the only remaining M3 item is the founder's API key (R-O2), not new code.
> See [session-17 retro](retros/session-17.md) + DECISIONS.md (§ session 17).
> **Next: M4 (world-awareness — L4 relighting / L5 collision+print / L6 physics-material).**
>
> **Session 16 done — M3 integration pt.1: GPU producer emits the full `.astel`
> contract + the real generative image path is wired through the API.** The GPU
> producer is now the true counterpart of the CPU stub: new torch-free,
> CPU-tested `astel_gpu.packaging.write_layer_stack(SplatCloud)` writes
> `l0.ply`/`l3.ply`/`l3.spz`/`l3.sog`/`package.astel`/`quality-report.json`
> (+`l2.ply` for generated assets), binding L0+L3 with per-gaussian provenance via
> `astel_format` (added as a GPU dep — pure-python, no CUDA). `astel_gpu.produce`
> dispatches by modality: **image+`--image` → real `run_l2_to_l3` (TripoSplat L2 →
> 2DGS L3)**; else the render-then-refit smoke. API `produce_artifacts_dispatch`
> gained `capture_id`, resolves the uploaded image from the store, and passes
> `--image`; the stub default path is byte-for-byte unchanged. **Measured on Box A
> (real CUDA):** smoke 8k/300it → 7-artifact contract, 41.8 dB, 2.4 s; generative
> (`creature_butterfly.webp`, 500it) → L2 65,536 gaussians (11.1 s, 4.59 GB, 0
> non-finite) → L3 65,536 surfels (8.1 s, 4.93 GB), held-out self-consistency
> 18.14 dB, 8-artifact contract incl `l2.ply`, all PLYs finite, `package.astel`
> round-trips honest. **REAL end-to-end (no mocking, `ASTEL_PRODUCER=gpu`):** the
> dispatch invoked the live `run-python.cmd` subprocess and stored all 7 artifacts.
> Gates green (GPU ruff·mypy 33·**56 pytest**; API ruff·mypy 19·**30+1**). No
> founder gate touched. **Honest gaps:** `astel_llm` Generation Spec still not
> wired into the API text path (session 17); text modality runs the smoke (no
> prompt conditioning until a text→multiview stage exists); generated geometric
> error/scale stay honestly `None`. See [session-16 retro](retros/session-16.md) +
> DECISIONS.md (§ session 16). **Next: session 17 — wire `astel_llm` into the API
> text path, then M4 (world-awareness — L4/L5/L6).**
>
> **Session 15 done — Generation Spec LLM stage scaffolded on fixtures (M3 step 5);
> M3 research/build arc complete.** New library `libs/astel_llm` implements CLAUDE.md §5's
> model-agnostic LLM layer + the text-pipeline prompt→`GenerationSpec` stage, built and tested
> **entirely offline** — no Anthropic API key, no spend (founder gate R-O2 untouched). External
> API facts re-verified live (Haiku 4.5 `claude-haiku-4-5` $1/$5; structured JSON via
> `output_config.format`; prompt caching; `count_tokens`). `FixtureAdapter` (default, replays
> cached completions by `(model,system,user)` hash) vs `AnthropicAdapter` (lazy SDK, optional
> `[live]` extra, key-gated) — stage code identical either way. `pricing.py` = verified rates +
> cache-discount math + credit `ledger_entry`. Gates green (ruff · mypy --strict 9 files ·
> **14 pytest**, offline). **The Anthropic API key is now the SINGLE remaining M3 gate** (R-O2):
> set `ANTHROPIC_API_KEY` + spend cap, `uv sync --extra live`, run one live call (~$0.02–0.035/gen
> Haiku, under the $1k/mo flag). **M3 steps 1–5 all complete in code; what's left is integration**
> (wire `astel_gpu.generative` + `astel_llm` into the API `produce` path + `.astel` packaging) and
> that key — not new research. See [session-15 retro](retros/session-15.md) + DECISIONS.md
> (§ session 15). **Next: integration, then M4 (world-awareness — L4/L5/L6).**
>
> **Session 14 done — generative L2→L3 wired end-to-end; DECISIONS #2 resolved; R-T1
> retired.** M3 step 4. New `astel_gpu.generative`: **image → TripoSplat L2 (native
> gaussians) → normalise → render orbit → 2DGS L3 distillation** (the session-13 L3
> representation). Generated objects have no GT scan, so the L3 is distilled from the L2's
> OWN multi-view renders; the number is held-out **self-consistency / distillation fidelity**,
> never accuracy-vs-reality (report keeps `geometric_error`/`scale` `None`,
> `generated_ratio=1.0`). New inverse converter `export.gaussian_params_from_splat_cloud` +
> pure `normalize_params` seam (CPU-tested). **Measured on Box A** (building_stone_house,
> 65,536 gaussians, 24 views, 1500 iters): L2 65,536 → L3 65,536 surfels, **held-out PSNR
> 23.13 dB**, refine 20.3 s, 4.93 GB, output PLYs finite. **DECISIONS #2 → L2 prior =
> TripoSplat** (evidence in hand; multi-model PSNR bake-off deferred — needs a multi-view
> generative corpus + TRELLIS-v1 install). **R-T1 retired** (TRELLIS.2 distillation off the
> critical path). Gates green (ruff · mypy --strict 31 files · **51 pytest**, 2 new). No
> founder gate touched. **Honest gaps:** no densification / 1500 iters (23 dB good not hero);
> not yet wired into the API `produce` path or `.astel` packaging; single test image. See
> [session-14 retro](retros/session-14.md) + DECISIONS.md (§ session 14). **Next: the
> Generation Spec LLM stage scaffolded on fixtures (M3 step 5), founder adds API key last.**
>
> **Session 13 done — L3 surface A/B RESOLVED: 2DGS beats raw 3DGS on real DTU.**
> The long-open L3 representation decision (DECISIONS #1, 🟡 since 2026-06-13) is now
> ✅ on a measured basis. New `astel_gpu.l3_refine` (gsplat-native 2DGS refine: normal
> consistency + L1 depth distortion over RGB L1+D-SSIM); `capture_eval` gained a
> `--representation {3dgs,2dgs}` switch so both arms share an identical init cloud + DTU
> ObsMask/Plane protocol + held-out split. **Measured on Box A (200k/3000, no densification):**
> raw 3DGS 8.76 mm overall (reproduces the session-10 8.73 mm baseline → fair); 2DGS with
> normal-only 9.48 mm (worse); 2DGS with normal + **scale-appropriate** distortion (λn=0.05,
> **λd=1e-4**) **8.53 mm overall / 10.91 mm accuracy** — beats 3DGS AND emits real surfel
> normals for L4/L5, at ~1 dB less PSNR. λd=1.0 collapses to 27 mm (distortion is
> depth²-scaled on a ~600 mm metric scene). **Decision: L3 = 2DGS + normal + distortion;
> GOF runner-up unneeded.** Gates green (ruff · mypy --strict 29 files · **49 pytest**, 4
> new). No founder gate touched. **Honest caveats:** λd is scene-scale-dependent (1e-4 is
> DTU-scan1-specific — a dimensionless scale-normalized λd is future work); no densification
> in either arm; single scan. See [session-13 retro](retros/session-13.md) + DECISIONS.md
> (§ session 13). **Next: L2→L3 wiring (TripoSplat → 2DGS surfelization), close DECISIONS #2,
> then the Generation Spec LLM stage on fixtures.**
>
> **Session 12 done — TripoSplat spike GRADUATED to a production L2 module; opacity
> defect fixed and measured.** M3 step 3a complete. The throwaway
> `triposplat_spike.py` is now typed/tested `astel_gpu.l2_triposplat`: it converts the
> vendored TripoSplat `Gaussian` into our `astel_splat_io` `SplatCloud` and exports via
> `write_ply`, **fixing the upstream `save_ply` inf-opacity defect** (~11% of points) by
> routing the activated `get_opacity` through our `[1e-6, 1-1e-6]` clamp — the one real
> defect flagged in session 11. Conversion is split into a pure CPU-testable
> `splat_cloud_from_fields` seam + a `gaussian_to_splat_cloud` adapter. **Measured on
> Box A:** 65,536 gaussians, 11.1 s, 4.59 GB peak, and decisively
> **`n_nonfinite_opacity_logit == 0`** (vs the spike's ~11% `inf`). Gates green (ruff ·
> mypy --strict 27 files · **45 pytest**, 2 new). No founder gate touched (no API key, no
> spend). **Still open:** the bake-off *scoring* half (resolves DECISIONS #2) and the L3
> surface A/B both remain — see [session-12 retro](retros/session-12.md). **Next: the L3
> 2DGS-vs-3DGS+GOF A/B on DTU scan1 (beat 8.73 mm), then L2→L3 wiring.**
>
> **Session 11 done — M3 ENTERED; TripoSplat = lead L2 prior.** Cleared the first
> two ordered/gated steps of [13-m3-readiness](research/13-m3-readiness.md) §4 with
> measured results, no founder gate touched (no API key, no spend). **(1) Triage GO**
> ([14-triposplat-triage](research/14-triposplat-triage.md)): `VAST-AI-Research/TripoSplat`
> is MIT (code **and** weights), 4 files / ~2.5k LOC, **zero** NC/build-heavy deps —
> cleaner than the TRELLIS-v1 head; single image → native 3D gaussians. **(2) Install
> spike PASS on Box A**: `torchvision 0.26.0+cu128` (from the existing index — no CUDA
> build, no flash-attn/xformers needed), weights downloaded, one inference produced
> **65,536 gaussians in 11.4 s at 4.6 GB peak VRAM**. **R-T9 resolved** for this
> candidate; **R-T1/R-T7 strongly de-risked**. **Decision: TripoSplat adopted as lead
> L2 prior**, pending the step-3 bake-off (resolves DECISIONS #2). One real defect to
> fix in the production wrapper: upstream `save_ply` emits `inf` opacity for ~11% of
> points (fp16 logit saturation) — clamp before export. See
> [session-11 retro](retros/session-11.md) + DECISIONS.md (§ session 11). **Next: L2
> bake-off (graduate spike → typed `l2_triposplat`) + the still-open L3 A/B.**
>
> **Session 10 done — M2 capture gaps closed.** (1) **SfM front-end validated**:
> ran COLMAP on the 49 real DTU images → 49/49 poses + 27k sparse cloud; Umeyama
> alignment to DTU GT poses gives **pose RMSE 0.886 mm** (sub-mm — closes the
> functional-SfM smoke deferred since session 8). (2) **DTU-protocol geometry**:
> rewrote `capture_eval` to DTU's official ObsMask/Plane masking + 60 mm cap +
> **held-out PSNR**. Protocol-correct result (scan1): held-out PSNR **21.5 dB**,
> **accuracy 11.36 mm, completeness 6.10 mm, overall 8.73 mm** vs the real scan
> (accuracy dropped from the box-proxy 18.9 mm once ObsMask excluded floaters).
> Raw-3DGS baseline the surface-aligned L3 must beat. New tested code:
> `capture_sfm`, `dtu.{umeyama,obsmask,plane}`, `metrics.nn_distances`; scipy
> added. Gates green (ruff · mypy 24 · 43 pytest). See
> [session-10 retro](retros/session-10.md) + DECISIONS.md (§ session 10).
> **Next: the L3 2DGS-vs-3DGS+GOF A/B on this scan** (beat 8.73 mm), then M3.
>
> **Session 9 done — FIRST REAL-WORLD geometry number.** Pivoted off the
> founder-gated orbit videos: per the founder's directive, sourced real capture
> data from public datasets. A Sonnet scout picked **DTU MVS** (real
> structured-light GT in metric mm, license-clean for internal benchmarking;
> T&T/CO3D rejected as non-commercial). Built the full real-capture path
> (`colmap_io`, `colmap_runner`, `dtu`, `capture_eval`, VRAM-safe chunked
> Chamfer) and measured, on real DTU scan1 photos vs the real scan: train PSNR
> 5.6→**23.3 dB**, **completeness (surface coverage) ≈ 3.85 mm**, accuracy ≈
> 18.9 mm (background-inflated — documented). Caught a scene-scale learning-rate
> bug (`spatial_lr_scale`: mm coords needed ~1000× larger position steps; 6→23
> dB). All gates green (ruff · mypy 23 files · 38 pytest). The number uses DTU's
> metric poses (no registration), so it validates splat-fitting geometry, not the
> pose-free story. See [session-09 retro](retros/session-09.md) and DECISIONS.md
> (§"2026-06-14 (session 9)"). Next: run COLMAP on the same images (front-end
> validation, runner ready), then exact ObsMask, then the L3 A/B on real data.
>
> **Session 8 done — first ground-truth geometry number + COLMAP installed.**
> Added a synthetic controlled-ground-truth eval (`astel_gpu.synthetic_eval`):
> it renders a KNOWN 0.20 m object, refits gaussians, and reports the **first
> real measured `geometric_error` (Chamfer) and `scale`** in the quality-report
> pipeline — the Truth Meter's first non-`None` numbers. Baseline: raw 3DGS
> covers the surface (~15 mm) but leaves floaters (~165 mm precision),
> empirically motivating the surface-aligned L3 (2DGS/PGSR) decision. COLMAP
> 4.1.0.dev0 (CUDA) installed to `tools/` (gitignored), launches clean. The
> API's GPU `produce` path is unchanged (its `geometric_error`/`scale` stay
> honestly `None`). See [session-08 retro](retros/session-08.md) and DECISIONS.md
> (§"2026-06-14 (session 8)"). **Still gated on the founder's orbit videos for
> the real-world capture numbers.**
>
> **Session 7 done — first GPU session.** The native-Windows GPU stack is
> validated on the 2×4090 box (`THREADRIPPER-48`): `torch 2.11+cu128` + `gsplat
> 1.5.3` compile and train (render-then-refit smoke PSNR 8.2→45.6 dB), and the
> API produces a real optimized `l3.ply` via `ASTEL_PRODUCER=gpu` (subprocess —
> torch stays out of the API env; stub remains the default so CPU gates stay
> green). **WSL2 was blocked (firmware), so we pivoted to native Windows** —
> see [session-07 retro](retros/session-07.md) and DECISIONS.md
> (§"2026-06-14 — GPU stack: native Windows"). Reproduce the env with
> `scripts/setup-gpu-env.ps1`; run any gsplat command via
> `pipelines/gpu/run-python.cmd`.
>
> **Session 6:** both session-5 carryover items closed — see
> [session-06 retro](retros/session-06.md). A real live-browser drag-drop of
> an image confirmed the `/v1/captures` round-trip and all 6 layer-stack
> artifacts serve correctly; Alembic migration scaffold added to `services/api`.

## Where we are

**Phase R closed. M1 CLOSED. M2 spine landed (CPU): real per-task artifacts flow end-to-end** (verified via a live browser round-trip in session 6), **including full `.astel` packages and `/v1/captures` uploads, with Alembic migrations now scaffolded.** Product named **Astel** (`.astel`).
- Stack chosen, deep-read, and license-audited ([DECISIONS.md](research/DECISIONS.md) v0.2,
  [LICENSE_AUDIT.md](research/LICENSE_AUDIT.md) v2).
- Task engine finalized = **Temporal** ([RA10](research/10-task-engine-spike.md)), now
  graduated into `services/api` behind the `TaskEngine` seam (stub default; `ASTEL_ENGINE=temporal`).
- `.astel` format implemented both sides: `libs/astel_format` (Python) + `packages/@astel/manifest` (TS).
- Splat exporters: `libs/astel_splat_io` — `.ply`/`.spz`/`.sog` + provenance sidecar.
- Blind-eval harness: `libs/astel_eval` — frozen corpus loader + Bradley-Terry + M3 gate (stub adapters).
- M1 monorepo runs end-to-end on CPU: web app (Spark viewer + Layer Inspector + Truth Meter),
  FastAPI + SSE, stub splat pipeline, CI (web · manifest · api · pipeline-stub · libs · license-gate),
  infra compose. See [ARCHITECTURE.md](architecture/ARCHITECTURE.md) and [session-03 retro](retros/session-03.md).
  Start it: `pnpm install` then **`pnpm run up`** (one-command bring-up; `-Temporal` for the durable engine)
  or `pnpm run dev:all` (web + API) / `pnpm dev` (web only).

## Two decisions still open — both GPU-gated (now UNBLOCKED — GPU work resumed session 7)

1. **L3 representation**: 2DGS surfels vs 3DGS + GOF extraction — needs a GPU A/B on fuzzy content.
   Session 8 added the first quantitative nudge: raw 3DGS on a synthetic 0.20 m object covers the
   surface (~15 mm) but leaves ~165 mm of floaters — surface regularization is clearly needed; the
   A/B (*which* variant) still wants fuzzy real content.
2. **Generative geometry prior** (R-T1, the riskiest bet): originally TRELLIS.2 O-Voxel → surfel
   distillation. **Session-10 M3 recon found a new option — TripoSplat (VAST-AI, MIT, native
   gaussian generator, reportedly SOTA) — which may de-risk R-T1 entirely.** The M3 plan is now a
   measured L2 bake-off (TripoSplat vs TRELLIS-v1 head vs TRELLIS.2 distill) on the eval harness.
   See [13-m3-readiness](research/13-m3-readiness.md).

The GPU is online; the L3 A/B now has real DTU content to test against (session 10). The L3
surface-aligned A/B (item 1) is the headline next GPU task; the L2 generative bake-off (item 2) is
M3, gated on the Windows-install spike + an API key for the Generation Spec stage (doc 13).

## Session 3 — finish M1 without a GPU ✅ DONE (2026-06-13)

All five items landed and green (see [session-03 retro](retros/session-03.md) for detail + honest gaps):
1. ✅ **Temporal engine integration** — `TemporalTaskEngine` behind the `TaskEngine` seam;
   `temporal server start-dev` managed via `astel up -Temporal`. Stub stays default; offline-safe tests.
2. ✅ **`.astel` packages** — `libs/astel_format` (Python) + `packages/@astel/manifest` (TS), round-trip tested.
3. ✅ **Export writers** — `libs/astel_splat_io`: `.ply`/`.spz`/`.sog` + provenance sidecar (SOG partial, documented).
4. ✅ **Blind-eval harness** — `libs/astel_eval`: corpus loader + Bradley-Terry + M3 gate + stub adapters.
5. ✅ **`astel up`** — `scripts/up.ps1` / `pnpm run up` (dev default; `-Temporal` opt-in).

M1 exit criteria met: green CI + browser demo + Temporal-backed resumable seam + eval-harness skeleton.

## Session 4 — real-artifact spine + first true browser SSE round-trip ✅ DONE (2026-06-13)

Closed the biggest "it's all simulated" gap from M1: the stub engine emitted progress events but
**never produced a file** — the viewer loaded a static checked-in `.ply` and the Truth Meter read a
static JSON. Now every generation produces and serves a **real, unique** asset on CPU, and the web
UI is driven by it. No GPU. All gates green (API: ruff·mypy-strict·17 pytest; web: eslint·tsc·15 vitest).

1. ✅ **Artifact store + producer** (`services/api`): `storage.py` (`LocalArtifactStore`, S3-swappable
   seam, path-traversal-guarded, `ASTEL_ARTIFACT_DIR`) + `producer.py` (deterministic per-task
   procedural splat seeded from `task_id`, reuses `libs/astel_splat_io.write_ply` via an editable path
   dep). On submit it writes `l3.ply` + an honest `quality-report.json` (`origin:"stub"` + explicit
   caveats — honesty channel intact; numbers flagged as illustrative, not measured).
2. ✅ **Serving route** `GET /v1/generations/{id}/artifacts/{name}` (FileResponse, 400 on bad name,
   404 if missing); `GenerationResource.artifacts[]` now lists what's on disk.
3. ✅ **Web wired to real output**: viewer loads the per-task `l3.ply` (sample = idle/fallback);
   Truth Meter renders the live API report with a mandatory **STUB** pill + caveat; Layer Stack
   L0–L3 reflect the SSE run ("4/8 ready").
4. ✅ **Fixed two real bugs found by the first live browser run** (M1 had only vitest + screenshots):
   (a) `App` and `GenerationDock` each held a *separate* `useGeneration()` instance → success never
   reached the viewer; lifted to one shared instance. (b) **SSE parser only split on `\n\n`** but
   sse-starlette emits **CRLF** (`\r\n\r\n`) → every event silently dropped, stream "completed" only
   on socket close. Parser is now CRLF/CR/LF-robust per the SSE spec; locked with a CRLF unit test.

Honest gaps: artifacts produced synchronously at submit in stub mode (fine for CPU stub; the durable
async path is the Temporal engine, unchanged); full `.astel` packaging + `.spz`/`.sog` exports +
`l0.ply` not yet wired into the producer (writers exist in `libs/astel_splat_io`); no `/v1/captures`
upload endpoint yet (Text path only — Image/Video tabs still send a placeholder string).

## Session 5 — close OPEN_ISSUES.md + full layer-stack artifacts + captures upload ✅ DONE (2026-06-13)

See [session-05 retro](retros/session-05.md) for full detail.

1. ✅ **P1** `astel_eval` suite 8m30s → ~4s: vectorized Bradley-Terry fixed-point fit +
   smoothing-tie prior (fixes a real MLE-divergence pathology on separated data) +
   relative early-stop.
2. ✅ **P2** `README.md` and `docs/architecture/ARCHITECTURE.md` de-staled (pnpm quickstart,
   `libs/` layout, all 6 CI jobs, real artifact flow, Temporal seam, current test counts).
3. ✅ **P3** producer now writes the full layer-stack artifact set per task: `l0.ply`,
   `l3.ply`, `l3.spz`, `l3.sog` (best-effort), `package.astel` (real `.astel` zip via
   `astel_format.builder.build_minimal_package`, fully-typed honest `QualityReport`), plus
   the existing Truth-Meter `quality-report.json`. New `POST /v1/captures` multipart upload
   (stores raw bytes, returns `capture_id`); web `GenerationDock` uploads Image/Video drops
   and threads `capture_id` into the generation request.

Honest gaps carried forward: captures are uploaded but not yet *consumed* (producer still
emits the stub splat regardless); no DB migration tooling (new `capture_id` column via
`create_all`, fine for dev SQLite); `.sog` remains best-effort per `astel_splat_io`'s own
docs; capture upload verified via automated tests, not yet a live-browser round trip.

## Session 6 — live-browser capture round-trip + Alembic scaffold ✅ DONE (2026-06-13)

See [session-06 retro](retros/session-06.md) for full detail.

1. ✅ Live browser pass: simulated drag-drop of an image onto the
   `GenerationDock`, confirmed `POST /v1/captures` (201) → `capture_id` →
   `POST /v1/generations` → SSE to "Asset ready" → Truth Meter STUB pill with
   `estimate (image)` → all 6 artifacts (`l0.ply`, `l3.ply`, `l3.spz`,
   `l3.sog`, `package.astel`, `quality-report.json`) served 200 with
   non-zero bodies. Added `"api"` entry to `.claude/launch.json`.
2. ✅ Alembic scaffold in `services/api`: `alembic.ini` +
   `migrations/` (async `env.py` reading `get_settings().database_url`,
   `target_metadata = Base.metadata`), baseline migration for the current
   `generations` table. `create_all` stays for dev/test; Alembic documented
   in ARCHITECTURE.md as the path for real schema changes.

Honest gap: Alembic is wired but no migration has run against persistent
data yet — first real test comes with the next schema change. Text- and
Video-modality live-browser passes still untested (only Image this session).

## M2 — Capture path (first GPU milestone; foundation landed sessions 7–8)

photos/video → L0→L1→L3 (reality first), quality report v1, exports. On the 2×4090 box.
Smoke ladder: ✅ rung 1 (CUDA sanity — both 4090s visible), ✅ rung 2 (gsplat reference
train — render-then-refit, measured), ✅ rung 3a (COLMAP 4.1 CUDA installed + launches; the
functional SfM smoke is deferred to real textured images), and ✅ a synthetic ground-truth
eval (first measured Chamfer/scale — the Truth Meter machinery is validated). **Session 9
delivered the first *real-world* number** on public **DTU** data (real structured-light GT,
metric mm): completeness ≈ 3.85 mm, accuracy ≈ 18.9 mm vs the real scan — the synthetic
baseline now has a real-world counterpart. **Session 10 closed the front-end + protocol
gaps**: COLMAP SfM validated (49/49 poses, **0.886 mm** pose RMSE vs DTU GT — closes the
session-8 SfM smoke) and `capture_eval` rewritten to DTU's official ObsMask/Plane protocol
+ held-out PSNR (scan1: **accuracy 11.36 mm, completeness 6.10 mm, overall 8.73 mm**,
held-out PSNR 21.5 dB). Remaining: (a) the **L3 2DGS-vs-3DGS+GOF A/B** on this real scan
(must beat 8.73 mm overall / 11.36 mm accuracy — the headline next step, resolves the
long-open L3 decision on real content); (b) a few more DTU scans for a corpus number; then
(c) M3: TRELLIS import check → R-T1 distillation. Founder's casual-phone orbit videos
remain the (non-blocking) way to prove the pose-free / scale-from-monocular-depth story
DTU's lab poses can't.

## What the founder does

**Nothing is blocking right now.** The 2×4090 box (`THREADRIPPER-48`) is online and the
GPU stack works natively (no WSL/SSH setup needed — we run on the box directly).

**Optional, now non-blocking (session 9 got the first real-world numbers from public
DTU data instead):**
- Film the 10 orbit videos in [eval/CORPUS.md](eval/CORPUS.md) §capture (phone, slow orbit,
  household objects). These are no longer needed for the *first* real-world accuracy numbers
  (DTU delivered those), but they're the one way to prove the **pose-free / casual-phone /
  scale-from-monocular-depth** story that DTU's lab-calibrated poses cannot — the part of the
  Truth Meter closest to the actual product use case.

**Optional:** bring the 3×3080 box (`100.70.127.42`) online when convenient (preview/CPU pool).

**M3 is prepped** (session 10) — external state re-verified live, plan + costs in
[13-m3-readiness](research/13-m3-readiness.md). Two founder gates before M3 LLM work:
- **Anthropic API key + a small spend cap** for the Generation Spec / L6 stage. Estimated
  **~$0.02–0.05 per generation** (Haiku 4.5 default, structured outputs + prompt caching) →
  ~$50–350/mo at 1k–10k generations — under the $1k/mo flag, but real spend needing your key.
  No paid call is made until approved (adapter built on cached fixtures first).
- **Generative-prior commitment** — recommend "evaluate TripoSplat first, then decide" (don't
  pre-commit to the TRELLIS.2 distillation). All candidate models are MIT (no licensing exposure).

**Decisions already settled (do not re-ask):** name = Astel; git stays local for now; GPU stack
is **native Windows** (WSL2 reversed — see DECISIONS.md 2026-06-14); GPU work has resumed.
Note: `scripts/setup-gpu-box.ps1` (the old WSL2 bootstrap) is superseded by native +
`scripts/setup-gpu-env.ps1` on this box.
