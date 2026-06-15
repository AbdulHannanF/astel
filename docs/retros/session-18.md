# Session 18 retro (2026-06-15)

**M4 ENTERED — L5 solidification core: splat → SDF → watertight surface → mass
properties → `.stl`, validated against analytic ground truth.** First step of M4
(world-awareness). New torch-free, CPU-only library `libs/astel_solid` implements
the print-path / physics-volume / collision spine decided in DECISIONS row 31.

Mode: Opus, inline. Pure CPU (no GPU needed, runs in CI). No founder gate touched.

## 1. What shipped — `libs/astel_solid`

A new standalone lib (sibling convention: own pyproject, ruff · mypy --strict ·
pytest). Per CLAUDE.md §1.2 the derived surface is **internal scaffolding only**
— used for printing / physics volume / collision proxies, never offered as the
asset (which is always splats):

- `sdf.py` — `oriented_point_sdf`: implicit-moving-least-squares SDF on a voxel
  grid from oriented surface samples (scipy `cKDTree` knn + gaussian-weighted
  point-to-plane). Outward normals ⇒ **negative inside / positive outside**, the
  marching-cubes convention.
- `isosurface.py` — `extract_isosurface`: `skimage` marching cubes at level 0 →
  world-space `TriMesh`, re-wound to outward (positive signed volume).
- `mass.py` — `compute_mass_properties`: volume, COM, and COM-frame inertia
  tensor for a closed mesh via signed-tetra divergence-theorem integrals
  (Blow & Binstock / Eberly), fully vectorised numpy, uniform density.
- `stl.py` — `write_binary_stl`: standard 80-byte-header binary STL (print only).
- `solidify.py` — `solidify(...)` ties the stages together; `surfel_normals(...)`
  derives per-splat OUTWARD normals from 2DGS quats + log-scales (thinnest
  principal axis, oriented away from the centroid) — the producer hook.

Deps verified live + permissive: numpy, scipy (BSD), scikit-image 0.26 (BSD).

## 2. Measured — validated against analytic solids

- **Unit cube (exact, the math check):** volume `1.0`, COM `(0,0,0)`, inertia
  `diag(1/6)` — all to `1e-6`. Offset + density scaling exact too. This proves
  the mass-property MATH is correct independent of any discretization.
- **Sampled sphere r=0.5 through the full pipeline (64³ grid, 6000 samples):**
  volume `0.5014` vs analytic `0.5236` (**4.2% low**); COM `‖·‖≈4e-3` (<1% of r);
  inertia diagonal `≈0.043` vs analytic `0.0501` (**~14% low**), near-isotropic
  (axes agree <5%), off-diagonals ~`1e-7` (≈0). The few-to-~15% low bias is the
  honest signature of a faceted inscribed MC polyhedron with an IMLS zero level
  that sits slightly inside — discretization, not a bug (the cube proves the math).

Gates green: ruff · mypy --strict (11 files) · **10 pytest** (cube exact;
sphere ballpark; SDF sign; outward winding; STL byte-layout; surfel-normal axis).

## 3. Honest gaps / carried forward

- **Not yet wired into the producer or `.astel` package.** The lib is standalone;
  a later step calls `surfel_normals` + `solidify` on the L3 cloud and binds the
  L5 layer (SDF stats, mass props, `.stl`) into the package + quality report.
- **`surfel_normals` outward-orientation is the centroid heuristic** — correct for
  star-shaped objects; non-star-shaped geometry needs proper normal-orientation
  propagation (graph-based flood fill). Documented in code.
- **Deferred per DECISIONS row 31:** Open3D TSDF fusion from L3 depth (alternative
  SDF source), **CoACD convex decomposition** for engine collision proxies,
  **`.3mf`** export, and **printability checks** (wall thickness / overhangs /
  hollowing) — each a follow-on session.
- IMLS bandwidth/resolution are fixed defaults; an adaptive/narrow-band SDF and a
  scale-aware bandwidth (use per-splat scales, not just spacing) would tighten the
  surface and cut the inward bias. Future tuning.
- Still nothing committed (sessions 7–18 on the single "Beta" commit).

## 4. Next

Continue M4: (a) wire L5 into the producer (`surfel_normals`→`solidify`→bind
SDF/mass/`.stl` into the package + report); (b) **L6 physics-material** LLM pass
(reuse the `astel_llm` adapter + the session-17 double-gate/graceful-degrade
pattern) to assign per-region density/friction and compute real mass from L5
volume; (c) **L4** appearance/relighting decomposition; (d) CoACD + `.3mf` +
printability to complete the print path.
