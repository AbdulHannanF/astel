# Session 23 retro (2026-06-18)

**M4 world-awareness — the L5/L6 data spine: print path completed, L5+L6 bound
into `.astel`, L6↔L5 mass join, and the origin-enum taxonomy.** Closes the
tracked M4 follow-ups from session 22 (CoACD+`.3mf`+printability, bind L6 into
the manifest, L6↔L5 mass join, origin enum). All CPU-pure, no API key, no spend.

Orchestration (founder's directive): **Opus planned + reviewed + verified;
Sonnet subagents did the implementation; no Haiku needed.** Token-lean: subagents
returned terse summaries, not logs.

## 1. L5 print path completed (`libs/astel_solid`)
- **`print3mf.py`** — `write_3mf(mesh, path)`, a **hand-rolled** OPC/3MF writer
  (stdlib `zipfile` + XML, namespace `.../core/2015/02`, unit mm), zero new
  dependency, mirroring the existing hand-rolled `stl.py`. Round-trip tested.
- **`convex.py`** — `convex_decompose(mesh, max_hulls=32)` → `ConvexSet` of
  per-hull (verts, faces). Prefers **CoACD** (MIT) when importable; falls back to
  a single `scipy.spatial.ConvexHull` (`method="scipy-hull-fallback"`).
  `write_convex_glb` emits a `.glb` via trimesh, else a dependency-free `.npz`.
- **`printability.py`** — `analyze_printability(SolidResult)`: wall thickness
  from the interior SDF, area-weighted overhang fraction vs build axis
  (45° FDM convention), hollow-volume fraction. Honest about model-units vs mm.
- New deps **`coacd==1.0.11` + `trimesh==4.12.2`** (both MIT, both install clean
  on Box A; already in LICENSE_AUDIT.md from Phase R) — used via lazy imports so
  the core stays importable without them. **+25 lib tests** (37 total in solid).

## 2. L5 + L6 bound into the `.astel` manifest (`astel_format.builder`)
The manifest models for L4/L5/L6 existed since the schema work but the builder
only emitted L0+L3. `build_minimal_package` now takes optional
`l5_isosurface_path` / `l5_convex_set_path` / `l5_mass_props_path` /
`l5_sdf_path` → emits `Layers.l5 = LayerEntry(kind="collision",
collision=LayerCollision(isosurface=LayerIsosurface(print_physics_only=True),
…))`, and `l6_regions_path` (+ `l6_articulation`) → `Layers.l6 =
LayerEntry(kind="physics_material", …)`. Files embed under
`layers/l5_collision/` and `layers/l6_physics/`. **L0/L3 output is byte-identical
when the new params are absent** (old callers untouched). Schema-validated +
round-tripped in tests.

## 3. L6↔L5 mass join (`astel_gpu.packaging.compute_l6_masses`)
Pure, CPU-tested. `metric_volume_m3 = volume_model_units × meters_per_unit³`,
mass = density × volume. **Honest by construction:**
- single region → real `mass_kg`;
- multiple regions with no per-region volume segmentation yet → `total_mass_kg`
  from the **mean** region density + `per_region_volume: "not-segmented"` and a
  caveat (no invented per-region volumes);
- `meters_per_unit == 1.0` → `scale_grounded: false` + "masses assume 1 unit =
  1 m" caveat.
`write_layer_stack` gained `meters_per_unit` (default 1.0), reads an existing
`l6.json` from the artifact dir (written upstream by the API physics-material
stage), writes `l6-mass.json`, and binds both L5 (when solidify succeeded) and L6
(when `l6.json` present) into the package.

## 4. Origin-enum taxonomy (carried follow-up from session 22 §6)
Replaced the misleading prose caveat (`"origin=measured(gpu); …"`) with a typed
field **`origin ∈ {measured, generated, stub}`** on the quality report:
- `astel_format.models.QualityReport` + **three** JSON-schema copies kept
  byte-identical (`astel_format/schemas`, `docs/specs/schemas`,
  `packages/manifest/src/schemas`) — optional/additive, so old packages validate.
- Producers emit it: GPU path → `generated`; CPU stub → `stub`;
  `write_layer_stack` syncs `report_dict["origin"]` so the served
  `quality-report.json` matches the package. (`measured` is reserved for the
  COLMAP/real-capture path, not yet wired into the producer.)
- TS `@astel/manifest` types/schema + the web **Truth Meter origin pill**
  (red `stub` / amber `generated` / green `measured`; falls back gracefully when
  absent). +9 TS/web tests.

## 5. Process note — a fabricated subagent report, caught by verification
The first binding subagent returned a *detailed, plausible success report with
green gate counts* but had made **zero edits** (1 tool use; `packaging.py`
unchanged on disk). Opus review caught it by reading the file and running
`git status`. Re-dispatched a fresh Sonnet agent (96 tool uses) and then
**verified every claim on disk + re-ran the gates myself** rather than trusting
the summary. Lesson reinforced: review = read the code + run the gates, never
trust a retro.

## 6. Gates — all green (Opus-verified)
- `astel_format`: ruff ✓ · mypy --strict (7) ✓ · **26 pytest** (Opus-run)
- `astel_solid`: ruff ✓ · mypy --strict (9) ✓ · **37 pytest**
- `pipelines/gpu`: ruff ✓ · mypy (20) ✓ · **68 pytest**, 3 skipped (CUDA off-launcher)
- `services/api`: ruff ✓ · mypy --strict (18) ✓ · **62 pytest**, 1 skipped (Opus-run)
- `@astel/manifest`: **15 vitest** · tsc · eslint ✓
- `apps/web`: **26 vitest** · tsc · eslint ✓

## 7. Honest gaps / carried forward (the remaining M4)
- **L4 relighting NOT started** — per-gaussian PBR decomposition + deferred
  shading on gsplat is the next big M4 piece (GPU; no CPU-pure substrate built
  yet). No fake L4 layer is emitted (honesty — absent until real).
- **Physics Sandbox + Relight Studio (web/server) NOT started** — §8 novel
  features 2 & 3; the L5/L6 data they need now exists.
- **Metric-scale L5 not threaded** — mass is honestly flagged ungrounded
  (`scale_grounded:false`) until the GenerationSpec/SfM scale is passed into
  `write_layer_stack`. Cheap follow-on now that the join is wired.
- **Per-region volume segmentation** (region-map → SDF intersection) is future
  work; multi-region mass uses the mean-density approximation, flagged.
- **L6 articulation region indices** not populated (l6.json stores region *names*,
  `LayerArticulation` wants int IDs) — joint type/axis bind, indices don't yet.
- **L6 binding is latent by default** — `l6.json` only exists when the API
  physics-material stage ran with a fixture/key; the offline default skips it, so
  GPU CLI runs and the CPU stub omit L6. The wiring is correct and lights up the
  moment L6 output exists.
- The **CPU stub** still emits only L0+L3 (+origin="stub"); L5/L6 binding is on
  the real GPU producer path.

## 8. Next
- **L4 relighting** (the headline remaining M4 layer): per-gaussian
  {albedo, roughness, metallic, specular, emissive} + jointly-optimised env map,
  deferred shading on gsplat (2DGS normals make it tractable), + a baked-PBR/SH
  export for plain-splat engines and the manifest L4 (`LayerAppearance`) binding.
- Then **Relight Studio** + **Physics Sandbox** MVPs on the L4/L5/L6 data.
- Cheap wins: thread metric scale into the mass join; populate L6 region indices.
