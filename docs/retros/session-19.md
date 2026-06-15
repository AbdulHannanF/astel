# Session 19 retro (2026-06-15)

**M4 cont. — L5 wired into the GPU producer: generated assets now carry an
`l5.stl` + mass properties, verified on a real 65k-splat cloud.** The session-18
`astel_solid` lib is now product-integrated: every GPU generation derives its
internal watertight surface + physics data (best-effort) and threads a `solidity`
summary into the quality report.

Mode: Opus, inline. On the 2×4090 box. No founder gate touched.

## 1. What shipped

- `astel_gpu.packaging._try_solidify` + `write_layer_stack(..., solidify_l5=True)`:
  derives `surfel_normals` from the L3 splats → `solidify` (SDF → watertight mesh
  → mass props) → writes `l5.stl` + `l5-mass.json`, and injects a `solidity`
  block (volume, mass-at-unit-density, COM, inertia diagonal, mesh + SDF stats)
  into `quality-report.json`. **Best-effort** (broad try/except, like `.sog`): a
  noisy/open cloud that won't solidify simply skips the L5 artifacts — never fails
  the asset (the surface is scaffolding; the asset stays splats, §1.2).
- `astel-solid` added as a `pipelines/gpu` dep (torch-free; numpy/scipy/skimage).
- Both producer paths (smoke + generative) get L5 for free via the shared writer.

## 2. Measured — real 65k cloud (pirate-ship image, 400 refine iters)

- L2 TripoSplat 65,536 gaussians (0 non-finite) → L3 2DGS 65,536, self-consistency
  **28.56 dB** (a solid hull reproduces better than the butterfly's 18 dB).
- **L5 on the real cloud:** watertight mesh **7,855 verts / 14,881 faces**;
  `l5.stl` = 744,134 bytes = exactly `84 + 50·14881` (valid binary STL). Volume
  3.77 model-units³; **anisotropic inertia diagonal (4.61, 1.53, 5.42)** — the low
  value about the long axis is physically correct for an elongated hull, a good
  sanity check that the mass math yields meaningful per-object physics. SDF grid
  43×48×19 (auto-shaped to the flat/elongated bbox). Full artifact set now:
  `l0/l2/l3.ply`, `l3.spz`, `l3.sog`, `l5.stl`, `l5-mass.json`, `package.astel`,
  `quality-report.json`, `l2l3-metrics.json`.

Gates green: GPU ruff · mypy --strict (33 files) · **55 CPU pytest** (+1 solidify
seam test on a sphere cloud; existing exact-contract tests pin `solidify_l5=False`).

## 3. Honest gaps / carried forward

- **Mass/volume are in MODEL units, not metric** — labelled as such. Metric
  grounding needs the scale stage (Generation Spec `target_scale`, session 17, or
  SfM scale) applied to the cloud before solidify; wiring that conversion is a
  follow-on.
- **L5 is not yet a bound layer in the `.astel` manifest** — `l5.stl`/`l5-mass`
  ship as loose artifacts + a report block. Binding an `l5` layer (manifest schema
  + provenance) into the package is a format step.
- `surfel_normals` still uses the centroid outward heuristic (star-shaped only).
- Deferred (DECISIONS row 31): Open3D TSDF, **CoACD convex decomposition** (engine
  collision proxies), **`.3mf`**, printability checks (wall thickness / overhangs /
  hollowing).
- The L5 grid resolution is a fixed 48 for producer responsiveness; a finer
  print-grade pass on demand is future work.
- Still nothing committed (sessions 7–19 on the single "Beta" commit).

## 4. Next

Continue M4: (a) **L6 physics-material** LLM pass — reuse the `astel_llm` adapter
+ session-17 double-gate/graceful-degrade; assign per-region density/friction and
turn L5's unit-density volume into a real mass; (b) **L4** appearance/relighting
decomposition; (c) metric-scale the L5 volume via the Generation Spec scale; (d)
CoACD + `.3mf` + printability to complete the print path; (e) bind L5 as a proper
`.astel` layer.
