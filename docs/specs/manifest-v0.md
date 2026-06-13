# `.astel` Package Format — v0 Specification

*Status: DRAFT v0 · 2026-06-13 · binding for M1. Supersedes the `.auriga` working name
(product renamed **Astel** on 2026-06-13). The extension, manifest media type, and all
field names use `astel` / `astl` going forward.*

> **What this document is.** The on-disk container format for a single Astel asset: a Layer
> Stack (L0–L7, CLAUDE.md §3) persisted as one self-describing package. It covers the package
> layout, the `manifest.json` schema, per-layer file formats, the **provenance channel** (the
> per-gaussian measured↔generated confidence scalar — CLAUDE.md §10.4, DECISIONS.md binding
> architecture decision #1), the quality-report block, metric scale, mixed-kernel headroom,
> versioning, and how the manifest maps onto open-standard exports (glTF
> `KHR_gaussian_splatting`, USD, `.spz` sidecar) without losing layers.
>
> The authoritative machine-readable contract is
> [`schemas/manifest.schema.json`](schemas/manifest.schema.json) (JSON Schema draft 2020-12)
> plus the sub-schemas in [`schemas/`](schemas/). Where this prose and the schema disagree,
> **the schema wins** and the discrepancy is a bug.

---

## 0. Design principles (binding)

1. **Layers never collapse.** The package is a stack of independently-addressable layers
   (CLAUDE.md §1.2). The manifest references files; it does not inline splat geometry.
2. **Splats are the product; scaffolding is bound, never standalone.** Point clouds, SDFs,
   convex hulls, and isosurfaces are layers *bound to* the splat asset. The watertight
   isosurface (L5) is print-/physics-only and exporters refuse to emit it except as
   `.3mf`/`.stl` (CLAUDE.md §1.1, §3 L5).
3. **The provenance channel is sacred.** Every gaussian carries a measured↔generated
   confidence scalar from L0 through export. It is reserved in the manifest *before* any
   pipeline exists; retrofitting it is impossible (DECISIONS.md arch decision #1). No stage may
   silently overwrite measured reality with generated content (CLAUDE.md §10.4).
4. **Honesty is structural.** The quality-report block is mandatory; "unknown" is a legal,
   first-class value. Absence of a measurement is recorded as `null` + a reason, never faked.
5. **Ride standards.** Splat-layer files are the same open formats we export
   (`.ply`/`.spz`/`.sog`); manifest field naming tracks `KHR_gaussian_splatting` where a 1:1
   concept exists, so exporters are near-identity maps.
6. **Forward-migratable.** A reader that understands `format_version` *major* N can open any
   N.x package; unknown additive blocks are preserved on round-trip (§10).

---

## 1. Container & package layout

A `.astel` file is a **ZIP archive** (store or deflate; no encryption in v0). The first entry
SHOULD be an uncompressed `mimetype` file (OPC/ODF convention) so the format is sniffable
without unzipping.

```
asset.astel  (zip)
├── mimetype                         # ASCII: "application/vnd.astel.package+zip" (stored, first)
├── manifest.json                    # the contract — REQUIRED, validates against manifest.schema.json
├── thumbnail.webp                   # optional canonical preview render
├── layers/
│   ├── l0_seed/
│   │   ├── points.ply               # sparse point cloud (xyz, rgb, confidence)
│   │   └── provenance.bin           # provenance buffer for L0 (§5)
│   ├── l1_dense/
│   │   ├── points.ply               # dense cloud: xyz, normals, rgb, semantic logits ref
│   │   ├── semantics.bin            # per-point semantic logits (optional)
│   │   └── provenance.bin
│   ├── l2_coarse/
│   │   ├── splats.spz               # coarse feed-forward gaussians
│   │   └── provenance.bin
│   ├── l3_refined/                  # the hero layer
│   │   ├── splats.ply               # archival master (full precision)
│   │   ├── splats.spz               # compressed delivery copy (optional, derived)
│   │   ├── splats.sog               # compressed delivery copy (optional, derived)
│   │   ├── kernel_batches.json      # mixed-kernel batch table (§7) — optional if single-kernel
│   │   └── provenance.bin
│   ├── l4_appearance/
│   │   ├── materials.bin            # per-gaussian PBR channels (§ L4 below)
│   │   ├── env_map.hdr              # estimated environment illumination
│   │   └── pbr_bake.spz             # baked-preview SH derived from L4 (for plain-splat engines)
│   ├── l5_collision/
│   │   ├── sdf.npz                  # sparse-voxel SDF (§ L5 below)
│   │   ├── convex_set.glb           # convex decomposition proxy set (collision only)
│   │   ├── isosurface.ply           # watertight surface — PRINT/PHYSICS ONLY, never a deliverable
│   │   └── mass_props.json          # volume, center of mass, inertia tensor
│   ├── l6_physics/
│   │   ├── regions.json             # per-region material class, density, friction, restitution
│   │   └── region_map.bin           # per-gaussian region id (uint16)
│   └── l7_dynamics/                 # optional (video inputs)
│       ├── deformation.bin          # deformation field / 4DGS keyframe deltas
│       └── timeline.json            # frame timing, fps, loop metadata
├── quality/
│   ├── report.json                  # quality-report block (also summarized inline in manifest)
│   └── hallucination_heatmap.bin    # per-gaussian measured↔generated heatmap (= provenance, §5/§6)
└── exports/                         # optional pre-baked exports (lazily generated)
    ├── asset.gltf  + asset.bin      # glTF + KHR_gaussian_splatting
    ├── asset.usdz
    └── asset.spz   + asset.astl.json  # bare .spz + sidecar manifest subset
```

**Rules.**

- `mimetype`, `manifest.json` are REQUIRED; everything else is referenced *by the manifest* and
  optional at the container level (a preview-only package may contain only L0).
- All layer/quality/export files are referenced from the manifest by **POSIX relative path**
  from the package root. No absolute paths, no `..` traversal. Readers MUST reject paths that
  escape the root.
- A file present in the zip but unreferenced by the manifest is IGNORED (allows tooling
  scratch space); a manifest reference to a missing file is an ERROR.
- Binary buffers (`*.bin`) are little-endian, layout described by the manifest's `buffers`
  table (§4.3). This mirrors the glTF buffer/bufferView model so export is mechanical.

---

## 2. Top-level manifest structure

`manifest.json` is a single JSON object. Top-level keys (full contract in the schema):

| Key | Req | Purpose |
|---|---|---|
| `format_version` | ✅ | SemVer string of *this spec* the package conforms to (e.g. `"0.1.0"`). §10 |
| `astel` | ✅ | Asset identity: `id` (UUIDv7), `created`, `generator`, `source_modality` (`text`/`image`/`multi_image`/`video`), `name`, optional `prompt`/`seed`. |
| `coordinate_system` | ✅ | Handedness, up-axis, forward-axis, `meters_per_unit` — the basis every layer lives in (§8). Astel-native = RH, +Y up, −Z forward (glTF/OpenGL convention). |
| `scale` | ✅ | Metric scale block: `meters_per_unit`, `confidence` interval, `method`, `user_overridden` (§9). |
| `layers` | ✅ | Ordered map L0→L7. Each present layer = a Layer Entry (§3). Absent layers omitted. |
| `buffers` | ✅ | Table of binary buffers + bufferViews backing provenance/materials/etc. (§4.3). glTF-shaped. |
| `provenance` | ✅ | Provenance channel descriptor: which buffer(s), quantization, semantics (§5). |
| `quality_report` | ✅ | Inline summary of `quality/report.json`: geometric error, scale CI, hallucination stats (§6). |
| `exports` | ◻ | Records of generated exports + the lossy mapping applied (§11). |
| `extensions` | ◻ | Namespaced vendor blocks (`astel_*`, `vendor_*`); preserved on round-trip (§10). |
| `extras` | ◻ | Free-form, ignored by validators (glTF convention). |

---

## 3. Layer entries (L0–L7)

Every present layer is a **Layer Entry** object under `layers` keyed by `l0`…`l7`. Common
fields (schema: [`schemas/layer.schema.json`](schemas/layer.schema.json)):

| Field | Purpose |
|---|---|
| `kind` | Enum fixed per layer: `seed_pointcloud`, `dense_pointcloud`, `coarse_gaussians`, `refined_gaussians`, `appearance`, `collision`, `physics_material`, `dynamics`. |
| `status` | `present` / `pending` / `failed` / `skipped` — supports the async preview/refine pipeline (CLAUDE.md §1.6). |
| `files` | Array of `{ path, role, format, sha256, bytes }`. `role` disambiguates (e.g. `master`, `delivery`, `provenance`). |
| `count` | Primitive count (points or gaussians) where meaningful. |
| `derived_from` | Layer ids this layer was computed from (provenance graph; e.g. L3 `derived_from: ["l1"]`). |
| `metrics` | Per-stage telemetry: `wall_seconds`, `vram_peak_mb`, `usd_estimate`, plus layer-specific quality (CLAUDE.md §10.3). |
| `provenance_ref` | Index into the `provenance` descriptor binding this layer's per-primitive confidence buffer. |
| `extras` | Free-form. |

Per-layer specifics:

### L0 — Seed / Sparse Point Cloud
- `format`: `ply` (positions + `rgb` + `confidence` scalar). The cheap first preview.
- Provenance: `confidence` here is the **seed** of the provenance channel — for capture inputs
  it is SfM track reliability; for generative inputs it is sampler confidence. Carried forward.

### L1 — Dense Point Cloud
- `format`: `ply` with `nx,ny,nz` normals + `rgb`. Optional `semantics.bin` buffer of per-point
  class logits (referenced via `buffers`, with a `class_labels` array).
- **L1 is the geometric ground truth** against which L3 error is measured (§6).
- Metric scale grounded here (§9).

### L2 — Coarse Gaussians
- `format`: `spz` (or `ply`). Feed-forward gaussians; SH degree typically ≤1. Third preview tier.

### L3 — Refined Surface Gaussians (hero layer)
- `format`: `ply` master REQUIRED (full precision, archival), `spz`/`sog` delivery copies OPTIONAL
  and marked `role: delivery` + `derived: true`.
- Carries: position, rotation (quat), scale (3), opacity, SH (degree 0–3), **per-splat normal**
  (2DGS surfels — DECISIONS.md L3 = 2DGS), provenance scalar.
- `kernel_batches` (§7) describes 3DGS-ellipsoid vs 2DGS-surfel partitioning if mixed.
- `metrics.geometric` REQUIRED: Chamfer / mean / P95 distance to L1 (§6).
- `budget`: `{ tier: "lowpoly|standard|cinematic", target_count, actual_count }` (MCMC budget —
  DECISIONS.md arch decision #3).

### L4 — Appearance / Lighting
- Per-gaussian PBR channels in `materials.bin`: `albedo` (VEC3), `roughness` (SCALAR),
  `metallic` (SCALAR), `specular` (SCALAR), `emissive` (VEC3). Layout in `buffers`.
- `env_map`: estimated environment illumination (`.hdr`/`.exr`), separated from albedo.
- `pbr_bake`: a one-way-derived baked-preview splat file for engines that consume only colored
  splats (CLAUDE.md §3 L4; DECISIONS.md RA4 — "baked preview generated *from* L4"). Baked-only
  is forbidden; the decomposed channels are always present alongside.
- `bound_to`: the splat layer these materials index (always `l3`); per-gaussian arrays are
  **parallel and index-aligned** to that layer's gaussians.

### L5 — Collision & Solidity (derived from L3)
- `sdf`: sparse-voxel signed distance field. Format: `npz` with arrays
  `{ origin[3], voxel_size, dims[3], indices[N,3] (int32), values[N] (float32) }` (sparse).
- `convex_set`: convex decomposition proxy set for engine collision (CoACD — DECISIONS.md RA4).
  Stored as `.glb` of convex hulls; **collision data only**, never a visible deliverable.
- `isosurface`: watertight surface, `.ply`. **PRINT/PHYSICS ONLY.** Flagged
  `print_physics_only: true`; exporters MUST refuse to emit it except as `.3mf`/`.stl`
  (CLAUDE.md §1.1, §3 L5).
- `mass_props.json`: `volume_m3`, `center_of_mass[3]`, `inertia_tensor[3][3]` (kg·m², requires L6
  densities; `null` until L6 present).

### L6 — Physics-Material & Semantic
- `regions.json`: array of regions `{ id, label, material_class (rigid|soft|cloth|fluid_adjacent),
  density_kg_m3, friction, restitution, confidence, reasoning_ref }`. Produced by the LLM/VLM
  pass (DECISIONS.md RA4); `confidence` and a citation to the reasoning trace are mandatory.
- `region_map.bin`: per-gaussian `uint16` region id, index-aligned to L3.
- `articulation`: optional detected joints / separable parts hints.

### L7 — Dynamics (video inputs, optional)
- `deformation.bin`: deformation field or 4DGS keyframe deltas (DECISIONS.md L7 = own 4DGS).
- `timeline.json`: `fps`, `frame_count`, `duration_s`, `loop`, keyframe table.
- `representation`: `deformation_field` | `keyframes` | `baked_per_frame`.

---

## 4. Buffers, accessors, and binary layout

### 4.1 Why a buffer table
Per-gaussian channels (provenance, materials, region ids, deformation) are stored as flat
binary buffers, not JSON, for size and zero-copy load. The manifest's `buffers` table is
**deliberately glTF-shaped** (`buffers` → `bufferViews` → `accessors`) so the glTF exporter is a
near-identity transform and tooling can reuse glTF loaders.

### 4.2 Endianness & alignment
- Little-endian. Each bufferView is 4-byte aligned. Floats are IEEE-754 `float32` unless an
  accessor declares quantization.

### 4.3 Tables
- `buffers[]`: `{ uri (relative path to a .bin), byte_length }`.
- `buffer_views[]`: `{ buffer, byte_offset, byte_length, byte_stride? }`.
- `accessors[]`: `{ buffer_view, component_type, type (SCALAR|VEC2|VEC3|VEC4), count,
  normalized?, quantization? }`. `component_type` uses glTF GL enums
  (`5120`=BYTE … `5126`=FLOAT) extended with `UNORM8`/`UNORM16` aliases for clarity.
- Per-gaussian accessors MUST have `count` equal to the bound splat layer's gaussian count and
  are **index-aligned** to it (the i-th accessor element corresponds to the i-th gaussian).

---

## 5. The provenance channel (binding — the format's soul)

> One scalar per primitive, present from L0 to export, encoding **how measured vs. generated**
> that primitive is. This is the substrate of the Truth Meter (CLAUDE.md §8.4) and the §10.4
> "sacred" no-silent-hallucination guarantee.

### 5.1 Semantics
- **Definition**: `provenance ∈ [0, 1]`, where **`1.0` = fully measured** (backed by real sensor
  data: an SfM-triangulated point, a multi-view-consistent splat) and **`0.0` = fully
  generated** (hallucinated by a diffusion/feed-forward prior with no direct observation).
  Intermediate values = partially-constrained (e.g. a splat seen in one view but completed by a
  prior). The value is a **continuous confidence**, not a binary flag — fuzzy boundaries are the
  norm and must be representable.
- **One value per primitive per layer.** L0/L1 points and L2/L3 gaussians each get their own
  provenance buffer (`provenance_ref` on the layer entry). When a layer is derived, provenance
  propagates: a splat's provenance is bounded above by the provenance of the points/splats it
  was optimized from (you cannot become *more* measured than your evidence).
- **Direction of authority (the sacred rule)**: generative completion may only *lower-bound*
  provenance for regions it touches; it MUST NOT raise the provenance of a region that had
  measured evidence. Writing generated geometry over measured reality without dropping its
  provenance is a spec violation and a CI-failing condition.

### 5.2 Binary buffer format
The provenance channel is a per-primitive buffer described by an accessor:

- **Storage**: `SCALAR`, **`UNORM8`** (1 byte/primitive) by default — `q = round(p · 255)`,
  decode `p = q / 255`. 256 levels is ample for a confidence display and keeps the channel
  ~cheap (1 MB per 1 M splats).
- **High-precision option**: `UNORM16` (`q = round(p · 65535)`, `p = q / 65535`) when the asset
  declares `provenance.precision = "u16"` — used where the hallucination heatmap drives
  print/physics safety decisions.
- **Layout**: tightly packed (`byte_stride` = component size), `count` = primitive count of the
  bound layer, **index-aligned** to that layer. The buffer lives in a `*.bin` referenced via
  `buffers`/`buffer_views`/`accessors`; the layer's `provenance.bin` is the canonical home.
- **Reserved sentinel**: encoded value `0` is *fully generated*, NOT "unknown." Genuinely
  unknown provenance is represented by **omitting** the primitive from any optional auxiliary
  mask and is reported in `quality_report.provenance.unknown_fraction`; v0 does not reserve a
  separate NaN sentinel in the UNORM channel (a future minor version may add an `valid` mask
  bufferView).

### 5.3 The `provenance` manifest descriptor
A top-level `provenance` object lists one descriptor per bound layer:

```json
"provenance": {
  "semantic": "measured_vs_generated",
  "range": [0.0, 1.0],
  "convention": "1=measured, 0=generated",
  "precision": "u8",
  "channels": [
    { "layer": "l3", "accessor": 7, "count": 1048576 }
  ]
}
```

### 5.4 How provenance survives export
Provenance is **not** part of `KHR_gaussian_splatting` core. Astel carries it through exports by
three mechanisms, in priority order:

1. **glTF**: a namespaced vertex attribute `_ASTEL_PROVENANCE` (underscore-prefixed custom
   attribute, the glTF-sanctioned escape hatch) on the splat primitive, UNORM8 SCALAR,
   index-aligned to `POSITION`. Standard glTF loaders ignore it; Astel-aware loaders read it.
2. **`.spz` / `.sog` delivery**: provenance ships in the **sidecar** `*.astl.json` + companion
   `.bin` (the splat file itself stays standard). The sidecar's accessor is byte-identical to
   the in-package one.
3. **USD**: a `primvar` `primvars:astel:provenance` (`float[]`, `interpolation = vertex`) on the
   splat prim.

In every case provenance is **index-aligned to the exported splat order**; if an exporter
reorders splats (e.g. SPZ Morton sort), it MUST apply the same permutation to the provenance
accessor. Reordering geometry without reordering provenance is a CI-failing golden-file test.

---

## 6. Quality-report block (the Truth Meter substrate)

`quality/report.json` is the full report; `manifest.quality_report` is an inline summary so
consumers needn't unzip. Schema:
[`schemas/quality-report.schema.json`](schemas/quality-report.schema.json).

| Field | Meaning |
|---|---|
| `geometric_error` | L3-vs-L1 distance: `{ chamfer_mm, mean_mm, p95_mm, method, reference_layer: "l1", units: "mm" }`. `null` (with `reason`) when there is no measured reference (pure text-to-3D). |
| `scale_confidence` | `{ meters_per_unit, ci_low, ci_high, ci_method, sources[] }` — mirrors `scale` (§9); the CI is honest, possibly wide. |
| `hallucination` | `{ heatmap_ref, measured_fraction, generated_fraction, unknown_fraction }`. `heatmap_ref` points at the provenance accessor (the heatmap **is** the provenance channel rendered as a colour ramp — no separate data). |
| `view_metrics` | Held-out-view photometric quality where applicable: `{ psnr, ssim, lpips, n_holdout_views }` (CLAUDE.md §10.3). `null` for generative-only assets with no held-out views. |
| `stage_telemetry` | Roll-up of per-layer `wall_seconds` / `vram_peak_mb` / `usd_estimate`. |
| `caveats` | Free-text honesty notes surfaced in UI ("back face unseen; completed by prior"). |

**Honesty contract**: every numeric field is either a real measurement or explicit `null` with a
`reason`. There is no "0 means we didn't check." Regressions in these metrics fail CI
(CLAUDE.md §10.3).

---

## 7. Mixed-kernel-type headroom

The L3 representation is 2DGS surfels today (DECISIONS.md), but the format must not hard-code
one kernel. Splat layers carry an optional **kernel batch table** so a single layer can mix
kernel types and future kernels drop in without a format break.

- `kernel_type` enum (extensible): `gaussian_3d` (ellipsoid — `KHR` `kernel:"ellipse"`),
  `gaussian_2d` (surfel/disc), reserved future values `gaussian_spindle`, `gaussian_ray` (3DGRT),
  `custom:*`.
- A layer is either **homogeneous** (`kernel_type` set directly on the layer entry) or
  **batched** (`kernel_batches` array). Each batch:
  `{ kernel_type, first, count, attributes[] }` where `[first, first+count)` is a contiguous
  index range into the layer's splats and `attributes` names the per-splat fields that batch
  carries (e.g. surfels carry a 2-vector scale + normal; ellipsoids a 3-vector scale).
- Rationale: 2DGS surfels and 3DGS ellipsoids differ in scale dimensionality and normal
  semantics; batching keeps each kernel's attributes tight while letting them coexist in one
  ordered buffer set. Export to `KHR_gaussian_splatting` (which today only defines
  `kernel:"ellipse"`) converts surfels to thin ellipsoids and records the lossy mapping in
  `exports[].notes` (§11); the lossless surfel form stays in the package.

---

## 8. Coordinate system & conventions

`coordinate_system` is declared once and every layer/buffer lives in it.

- **Astel-native basis**: **right-handed, +Y up, −Z forward** (glTF / OpenGL convention) — chosen
  so glTF/Three.js export is identity and the documented-rotations matrix (RA5 §4) has Astel at
  the centre.
- Fields: `handedness` (`right`/`left`), `up_axis` (`+X|+Y|+Z|-X|-Y|-Z`), `forward_axis`,
  `meters_per_unit` (default `1.0`).
- **SH basis is part of the convention.** Any export that changes handedness/up (Unity LH+Y,
  Unreal LH+Z+cm) MUST rotate SH band-≥1 coefficients and resign as needed; the exporter records
  the applied transform in `exports[].coordinate_transform`. Unrotated SH under basis change is
  the classic silent-corruption bug (RA5 §4) and is golden-file tested.

---

## 9. Metric scale, units, confidence

`scale` makes the asset metrically grounded and honest about it.

| Field | Meaning |
|---|---|
| `meters_per_unit` | Conversion from native units to metres. |
| `confidence` | `{ ci_low, ci_high, distribution }` — the interval, not a point claim. May be wide. |
| `method` | `sfm_exif` / `metric_depth_consensus` / `vlm_size_estimate` / `user` (DECISIONS.md RA3 consensus). |
| `sources` | Array of `{ method, meters_per_unit, weight }` contributing to the consensus. |
| `user_overridden` | Bool — the user may override scale (CLAUDE.md §3 L1); original retained in `sources`. |

For pure text/image inputs scale comes from the VLM size estimator with an explicit, possibly
wide CI the user can override; the CI is never hidden (CLAUDE.md §3 L1, §4 Text).

---

## 10. Versioning & forward-migration policy

- `format_version` is SemVer (`MAJOR.MINOR.PATCH`) of **this spec**.
- **MAJOR** bump = breaking change (a field's meaning/required-ness changes). Readers MUST refuse
  a MAJOR they don't implement.
- **MINOR** bump = additive (new optional fields/layers/kernel types). A reader of N.x MUST open
  any N.y (y≥x) package, ignoring fields it doesn't know.
- **PATCH** = clarifications, no schema change.
- **Forward-migration**: unknown additive keys and unknown `extensions.*` / `extras` blocks MUST
  be **preserved on round-trip** (read→write) so older tools don't strip newer data. Validators
  warn, not error, on unknown additive keys within the same MAJOR.
- **Unknown enum values** (e.g. a future `kernel_type`) are preserved; a reader that can't render
  them falls back to its best-known kernel and flags the asset as partially-understood rather
  than failing.
- Each package records the producing `astel.generator` (name + version) so migrations are
  auditable.

---

## 11. Export mapping (no layer left behind)

Exports are **lossy projections** of the Layer Stack onto open standards. The package is the
lossless source of truth; every export records *what it dropped* in `exports[].notes` so the
round-trip story is honest.

### 11.1 glTF + `KHR_gaussian_splatting`
- Splat layer (L3, or L2 for previews) → a `POINTS` primitive with the
  `KHR_gaussian_splatting` extension. Direct field mapping:

  | Astel | `KHR_gaussian_splatting` |
  |---|---|
  | position | `POSITION` (VEC3 float) |
  | rotation (quat) | `KHR_gaussian_splatting:ROTATION` (VEC4, float / norm int8/16) |
  | scale | `KHR_gaussian_splatting:SCALE` (VEC3, float / (norm)uint8/16) |
  | opacity | `KHR_gaussian_splatting:OPACITY` (SCALAR, float / norm uint8/16) |
  | SH degree 0 | `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` (VEC3 float, RGB) |
  | SH degrees 1–3 | `KHR_gaussian_splatting:SH_DEGREE_ℓ_COEF_n` (VEC3 float) |
  | provenance | `_ASTEL_PROVENANCE` custom attribute (§5.4) |
  | colour fallback | `COLOR_0` (derived diffuse + alpha, for non-splat loaders) |

  Extension-level props set to match KHR: `kernel` (`"ellipse"`; surfels converted, §7),
  `colorSpace` (`"srgb_rec709_display"` or `"lin_rec709_display"`), `projection`
  (`"perspective"`), `sortingMethod` (`"cameraDistance"`). SPZ-compressed glTF uses the companion
  `KHR_gaussian_splatting_compression_spz` extension when emitting compressed buffers.
- **Dropped on export** (recorded in `notes`): all non-splat layers (L0/L1 clouds, L4 decomposed
  PBR beyond the SH/PBR bake, L5 SDF/convex/isosurface, L6 regions, L7 dynamics unless the target
  supports animated splats). The `.astel` package retains them.

### 11.2 USD / USDZ
- Splat prim with point-instancer-style attributes; provenance + (optionally) L4/L6 as
  `primvars:astel:*`. USDZ for AR delivery (VFX pipelines, CLAUDE.md §0).

### 11.3 Bare `.spz` / `.sog` + sidecar
- The splat file stays a standard `.spz`/`.sog`; a companion `*.astl.json` sidecar carries the
  manifest subset (scale, provenance accessor, quality summary, L4/L6 refs). Sidecar + splat
  reconstruct an Astel-aware view without a full `.astel` zip.

### 11.4 What engine plugins consume (Unity / UE5)
The Unity package and UE5 plugin (CLAUDE.md §5, M5) import **`.spz`/glTF + the `.astel`
manifest** and auto-configure, *without ever instantiating a visible mesh*:
- **Renderer**: splats from L3 (or L2 preview) via the engine's existing splat renderer
  (DECISIONS.md arch decision #5 — we wrap, not rewrite).
- **Collision**: L5 `convex_set` → physics colliders.
- **Mass/inertia**: L5 `mass_props` + L6 densities → rigidbody mass & inertia tensor.
- **Materials**: L4 channels → engine PBR params (or the baked-preview splat colours).
- **Physics material**: L6 `regions` → per-region friction/restitution.
- **Scale & axes**: `scale` + `coordinate_system` → correct unit/axis conversion (cm for UE5,
  LH for both; SH resign per §8).
- **Print**: L5 `isosurface` is *never* imported as engine geometry; it is print-path-only.

---

## 12. Open questions (tracked for v0.2)

1. Whether to reserve an explicit `valid` mask bufferView for provenance so genuinely-unknown
   primitives are distinguishable from fully-generated `0.0` in-band (§5.2). Leaning yes; defer to
   avoid over-building before the pipeline exists.
2. USD splat payload schema is still community-fluid (RA5 open-Q2); the `primvars:astel:*` choice
   may shift to whatever Omniverse standardizes.
3. `KHR_gaussian_splatting` is **Release Candidate** as of 2026-06-13 (ratification still
   targeted Q2 2026; not yet final — verified this session). Field names here track the RC; pin to
   the ratified spec text before the M5 exporter ships and add a conformance test.
4. Compression of provenance alongside SPZ (SPZ has no custom-attribute slot yet) — for now
   provenance rides the sidecar, not the SPZ stream.
