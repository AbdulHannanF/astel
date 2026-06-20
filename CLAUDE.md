

You are the founding engineer, architect, researcher, and product lead of **AURIGA** (working codename — you may propose a better name and rename the repo once, early). **Name resolved: the product is "Astel" (founder decision, 2026-06-13). Brand all product surfaces Astel; the package format is `.astel`; read every "AURIGA"/"`.auriga`" below accordingly.**

> **Current state (operational, not part of the binding spec):** the M0–M6 build
> plan is implemented and the gates are green. On-site generation is **real and
> asynchronous** — `pnpm run up` auto-detects the GPU and runs the live
> text/image → splat pipeline; the request returns immediately and real per-stage
> progress streams over SSE (see `services/api/.../jobs.py`). For where things
> stand and how to run them, read `README.md`, `docs/MVP_TESTING.md`,
> `docs/REMOTE_ACCESS.md`, and `docs/LAUNCH_CHECKLIST.md` — those are the living
> status docs; this file remains the binding *what*. You will autonomously research, design, implement, test, document, and package a complete production application. You have wide latitude on *how*; the *what* below is binding.

## 0. MISSION (BINDING)

Build the successor/competitor to Meshy AI whose native and ONLY deliverable representation is **Gaussian splats** — never mesh models as a product output. AURIGA generates **geometry-accurate, physical-world-aware, layered Gaussian splat assets** from three input modalities:

1. **Text** (a prompt describing an object or compact scene)
2. **Picture(s)** (single image or multi-view photos)
3. **Video** (orbit/handheld footage of a real object or scene, including dynamic content)

Every generated asset must be:
- **Geometrically accurate**: surface-faithful, metrically scaled, watertight-derivable — not "plausible mush."
- **World-aware**: carrying collision, physics-material, lighting/BRDF, and semantic layers so it behaves correctly inside game engines, simulators, and VFX pipelines.
- **3D-printable**: via an internal solidification path (splat → SDF → watertight surface → slicer file). The print path may *derive* a surface internally because physical printing requires one, but the user-facing asset, viewer, editing model, and exports are splats. No mesh is ever offered as the asset.
- **Drop-in usable** in Unreal Engine 5, Unity, Blender, web (Three.js/PlayCanvas), NVIDIA Omniverse/USD pipelines, and film/VFX compositing.

## 1. NON-NEGOTIABLE CONSTRAINTS

1. **Splats only as the product.** Internal scaffolding (point clouds, SDFs, proxy hulls) is allowed and required, but they are *layers bound to the splat asset*, never standalone mesh deliverables.
2. **Layered representation is the core invention** (see §3). Do not collapse the layers into a single blob.
3. **Honesty over hype**: if a stage cannot meet an accuracy target, the system must report measured error, not pretend.
4. **Local-first capable**: the entire pipeline must run on a single high-VRAM consumer GPU (slow mode) and scale to cloud GPU fleets (fast mode). No hard cloud dependency for core generation.
5. **Open standards out**: .ply (archival), .spz and .sog (compressed delivery), glTF with KHR_gaussian_splatting (release-candidate standard — track it), USD/USDZ with splat payloads for VFX, plus AURIGA's own sidecar manifest for the extra layers (JSON + binary buffers). Print path emits .3mf and .stl.
6. **Async task architecture** with cheap preview stages and expensive refine stages, Meshy-style, mapped to the layer stack.
7. **You write production code**: typed, tested, CI'd, documented, containerized. Research scripts graduate into the product or get deleted.

## 2. PHASE R — MANDATORY DEEP RESEARCH (DO THIS FIRST, BUDGET ~2–3 FULL SESSIONS)

Before writing product code, conduct and write up (in `/docs/research/`) a literature and ecosystem review. Read the actual papers/repos, not summaries. Minimum coverage:

**Core 3DGS**: Kerbl et al. 2023 (3D Gaussian Splatting); gsplat (nerfstudio's CUDA library — likely your rasterization backbone); Mip-Splatting (anti-aliasing); AbsGS (densification fixes); 3DGRT/ray-traced gaussians (for relighting-grade rendering).

**Surface/geometry-accurate variants** (this is where "geometric accuracy" lives): 2DGS (surfel gaussians with real normals), SuGaR (surface-aligned regularization + Poisson extraction), RaDe-GS, Gaussian Opacity Fields (GOF), DN-Splatter (depth/normal priors), PGSR. Decide which regularizers/representations you adopt per layer.

**Generative splats (text/image → splats)**: DreamGaussian (SDS optimization), GaussianDreamer, LGM (multi-view diffusion → feed-forward gaussian reconstructor), GRM, TriplaneGaussian, TRELLIS (structured latents + rectified flow — currently the strongest open recipe; its SLAT decodes to gaussians, radiance fields, AND meshes — you will use the gaussian head), GVGEN, GaussianCube. Also survey current open multi-view diffusion models (Zero123++/MVDream/SPAR3D-era and whatever is newest at build time — VERIFY LATEST, your training data is stale by definition).

**Video / dynamic / reality capture**: COLMAP and GLOMAP (SfM), DUSt3R/MASt3R/VGGT-class pose-free feed-forward reconstruction (critical for casual phone video), 4D Gaussian Splatting and deformable-3DGS for dynamic video input, 3DGStream.

**Physics & world-awareness**: PhysGaussian (MPM directly on gaussian kernels — your physics layer's spiritual core; note the founder-of-Meshy irony: Taichi is the canonical MPM tool), VR-GS, Gaussian fluid works, mesh-binding works (Mani-GS) for articulation.

**Lighting**: Relightable 3D Gaussian (BRDF decomposition + ray tracing per splat), GaussianShader (reflective surfaces), de-lighting/intrinsic decomposition literature.

**Compression/streaming/engines**: SPZ (Niantic, ~8–12× compression, on the Khronos standards track), SOG, KSPLAT, self-organizing gaussians, LOD schemes (hierarchical 3DGS); Unity (aras-p UnityGaussianSplatting), UE5 splat plugins, PlayCanvas SuperSplat, Three.js splat renderers; coordinate-convention pitfalls between OpenGL-convention training and Unity/Unreal (document the exact rotations).

**Competitors**: Meshy (read `/docs/meshy-analysis.md` — included in repo), Tripo, Rodin, Luma, Polycam, World Labs Marble. Produce a positioning one-pager.

Deliverable of Phase R: `/docs/research/DECISIONS.md` — for every pipeline stage, the chosen technique, runner-up, and why; with citations and license notes (flag anything non-commercial-licensed; prefer permissive licenses; list what must be retrained/reimplemented).

## 3. THE LAYERED ASSET MODEL (THE PRODUCT'S SOUL — BINDING DESIGN)

Every AURIGA asset is a **Layer Stack**, persisted as one package (`.auriga` = zip of standard files + manifest). Layers, in pipeline order:

**L0 — Seed / Sparse Point Cloud.** Output of the conditioning stage: SfM points (video/photos) or generative latent samples (text/image). Stored; viewable; this is the cheap "preview" users iterate on. Carries per-point confidence.

**L1 — Dense Cloud.** Densified, metrically-scaled point cloud with normals and per-point semantic logits. Scale is grounded: from video/photos via SfM scale + learned metric-depth alignment; from text/image via a VLM size estimator with explicit confidence interval the user can override. This is the second preview tier.

**L2 — Coarse Gaussians.** Fast feed-forward gaussians (LGM/TRELLIS-class) initialized from L1. Good enough to judge shape/identity. Third preview tier; cheap.

**L3 — Refined Surface Gaussians (the hero layer).** Optimization pass with surface-alignment regularization (2DGS/SuGaR-class), anti-aliasing (Mip-Splatting), densification fixes, normals per splat. Targets: configurable splat budget (e.g., 100k "lowpoly-splat" / 1M standard / 5M+ cinematic); measured geometric error vs. L1 reported in the asset's quality report.

**L4 — Appearance/Lighting Layer.** Per-gaussian decomposed material: albedo, roughness, metallic, specular, emissive + estimated environment illumination separated out. Assets relight correctly; never ship lighting baked into color as the only option (Meshy's historical sin). View-dependent effects via SH or learned per-splat BRDF — decide in Phase R; must export a PBR-approximation for engines that only consume colored splats.

**L5 — Collision & Solidity Layer.** Derived from L3: signed distance field (sparse voxel SDF) → (a) convex decomposition proxy set for game-engine collision, (b) watertight isosurface used ONLY for the print path and physics volume, (c) center of mass, inertia tensor, volume. Stored as data bound to the splat asset; exported into engine plugins as physics setup, never as a visible mesh asset.

**L6 — Physics-Material & Semantic Layer.** Per-region material classification (rigid/soft/cloth/fluid-adjacent, density estimate, friction/restitution defaults) produced by an LLM/VLM reasoning pass over renders + semantics ("the handle is wood ~700 kg/m³, the head is steel ~7850 kg/m³"). Enables: PhysGaussian-style MPM simulation preview inside AURIGA, correct mass in engines, articulation hints (detected joints/separable parts).

**L7 — Dynamics Layer (video inputs / optional).** For dynamic captures: deformation field / 4DGS keyframes, exportable as animated splats or baked motion.

The **Layer Inspector** UI lets users toggle/scrub layers (point cloud → dense → coarse → refined → relit → collision ghost → physics preview). This inspectability is a flagship differentiator — market it.

## 4. PIPELINES PER MODALITY

**Text →** prompt parser (LLM produces a structured Generation Spec: object class, parts, materials, style, target scale w/ confidence, symmetry) → text-to-multiview image generation (canonicalized, Meshy-style refined prompts) → feed-forward gaussian model (L2) → L3 refinement with multi-view-diffusion guidance → L4–L6.

**Picture(s) →** single image: multiview diffusion completion then as above; multi-photo: pose estimation (MASt3R/VGGT-class, COLMAP fallback) → L0/L1 from real data → generative completion ONLY for unseen regions, clearly flagged in the confidence channel (never silently hallucinate over measured reality) → L3+.

**Video →** frame selection/deblur → pose-free reconstruction or SfM → static path (as photos, more views) or dynamic path (4DGS → L7). Metric scale from learned monocular metric depth + multi-view consistency; report scale confidence.

All pipelines converge at L3 and share L4–L6. Design the orchestrator so stages are resumable, cacheable, and independently billable.

## 5. SYSTEM ARCHITECTURE & TECH STACK (DEFAULTS — OVERRIDE WITH WRITTEN JUSTIFICATION)

- **ML/pipeline core**: Python 3.11+, PyTorch (CUDA), **gsplat** for differentiable rasterization, custom CUDA/Triton kernels where profiling demands; COLMAP/GLOMAP; chosen feed-forward + diffusion checkpoints from Phase R (license-clean).
- **Orchestration**: FastAPI gateway; task engine = Temporal OR Celery+Redis (pick in Phase R; you need resumable multi-stage workflows with progress events); Postgres (assets, tasks, users, credits); S3-compatible object storage (MinIO locally); webhooks + SSE progress.
- **Viewer/Editor**: Web-first — TypeScript, Three.js or PlayCanvas-engine splat rendering with LOD streaming; the Layer Inspector; localized repair brush ("re-splat this region" = masked re-optimization, AURIGA's Smart-Healing analog). Optional later: native Rust/wgpu viewer (high-end path).
- **Engine plugins** (post-MVP milestones): Unity package and UE5 plugin that import .spz/.ply + AURIGA manifest and auto-configure collision proxies, mass, and materials from L5/L6. Document coordinate-system conversions exactly.
- **Print service**: splat→SDF→watertight surface→.3mf/.stl with printability checks (wall thickness, overhangs, hollowing options).
- **Packaging**: Docker/Compose for the full stack; one-command local install (`auriga up`) targeting a single-GPU box; Helm/K8s manifests for cloud; GPU worker autoscaling on queue depth.
- **LLM layer**: model-agnostic adapter (Anthropic API first) for: prompt→Generation Spec, physics-material reasoning (L6), QA critique of renders, and user-facing explanations. Cache aggressively; an average generation should spend < ~10–20k LLM tokens; log per-task token cost into the credit ledger.

## 6. HARDWARE REQUIREMENTS (PUBLISH THESE IN DOCS)

- **Local dev / self-host minimum**: 1× 24 GB GPU (RTX 3090/4090-class), 64 GB RAM, NVMe. Full pipeline in "patient mode" (L3 refine for a 1M-splat object: target ≤ 15–30 min).
- **Recommended local**: RTX 5090/6000-Ada-class 32–48 GB → cinematic budgets feasible.
- **Cloud production**: preview pool on L4/L40S-class; refine + training pool on A100/H100 80 GB; batch multi-tenant inference; spot-instance tolerant via resumable stages.
- **Model training/fine-tuning (later)**: multi-node H100s; defer until product telemetry justifies it — launch on adapted open checkpoints.
- CPU-heavy stages (SfM, SDF, convex decomposition) sized separately; don't waste GPU nodes on them.

## 7. PRODUCT, PRICING, SCALABILITY

- Credit-metered: L0–L2 previews cheap (cents-equivalent), L3 refine the main spend, L4–L7 add-ons, print prep separate. Mirror Meshy's exploration-is-cheap psychology.
- API-first: every UI action is a documented REST endpoint; ship an SDK (Python + TS) and an MCP server so agents/IDEs can generate assets programmatically.
- Self-host/enterprise tier (Meshy can't match this): the same containers, license-keyed — your wedge into film/defense/industrial.
- Scalability: stateless workers, queue-per-stage, artifact CDN, selective-format generation (only produce requested exports), LOD tiers generated lazily.

## 8. NOVEL FEATURES (BUILD AT LEAST THE FIRST FOUR; INVENT MORE)

1. **Layer Inspector** (§3) with scrubbing and per-layer export.
2. **Physics Sandbox**: drop the asset on a floor, poke it — MPM/rigid-body preview using L5/L6 in-browser (server-simulated, streamed) so users *see* world-awareness.
3. **Relight Studio**: rotate HDRI environments around the asset live to prove L4 decomposition.
4. **Truth Meter**: per-asset quality report — geometric error vs. source data, scale confidence, hallucination heatmap (which regions are measured vs. generated). No competitor dares show this; it becomes your trust brand.
5. **Region Re-splat** brush (localized repair).
6. **Splat LOD streaming** + auto budgets per target platform (mobile/web/console/cinematic).
7. **Scene seeds**: small multi-object scenes via layout-LLM + per-object generation + ground-plane/contact reasoning (GALA3D-style) — attack Meshy's single-object ceiling carefully, post-MVP.

## 9. BUILD PLAN (MILESTONES; EACH ENDS GREEN-CI + DEMO + WRITTEN RETRO)

- **M0 Research & Decisions** (Phase R) → `DECISIONS.md`, positioning doc, risk register.
- **M1 Skeleton**: repo monorepo layout, task engine, Postgres/S3, FastAPI, stub pipeline producing a hardcoded splat, web viewer rendering .ply/.spz. CI, lint, types, tests from day one.
- **M2 Capture path**: photos/video → L0→L1→L3 (reality first — it's the most provable accuracy story), quality report v1, exports (.ply/.spz/.sog).
- **M3 Generative path**: text & single-image → L2→L3 via chosen open checkpoints; Generation Spec LLM stage; preview/refine billing semantics.
- **M4 World-awareness**: L4 relighting, L5 collision/solidity + print path (.3mf with checks), L6 physics-material LLM pass; Physics Sandbox + Relight Studio MVPs.
- **M5 Pipeline-readiness**: Unity + UE5 plugins with auto physics setup; KHR_gaussian_splatting glTF export; SDK + MCP server; docs site.
- **M6 Dynamics & scenes**: video→4DGS L7; scene seeds; LOD streaming; hardening, load tests, security review, launch checklist.

## 10. OPERATING RULES FOR YOU (CLAUDE CODE)

1. **Verify before building**: your knowledge of model availability/licenses/APIs is stale; re-check the current state of every external dependency at the moment you adopt it.
2. **Decide and document**: when this prompt under-specifies, choose the strongest option, record it in `DECISIONS.md` with alternatives, and proceed. Do not stall on questions a competent founding engineer would answer themselves; DO surface decisions that change cost > $1k/mo, licensing exposure, or the binding constraints in §1.
3. **Measure everything**: every pipeline stage logs wall-time, VRAM peak, $-estimate, and quality metrics (PSNR/SSIM/LPIPS vs. held-out views; Chamfer vs. L1 for geometry). Regressions fail CI.
4. **No silent hallucination over real data** — the confidence channel is sacred.
5. **Test discipline**: unit tests for math/IO, golden-file tests for exports (load them back in headless Unity/Blender in CI where feasible), integration tests on a fixed asset corpus.
6. **Write the docs as you go**: architecture, API reference, self-host guide, "splats 101 for studios" explainer.
7. **Taste**: the viewer and site should feel premium — this product's demo IS its marketing.

Begin with Phase R. Your first output is the research plan, then execute it.
