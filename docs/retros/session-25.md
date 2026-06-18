# Session 25 retro (2026-06-18)

**M4 finished off — the two tracked "cheap win" follow-ups landed, and one of
them turned out to be a latent crash, not a polish item.** Sessions 23/24 left M4
feature-complete (L4/L5/L6 + Truth Meter + Relight Studio + Physics Sandbox) but
flagged two carried-forward data-completeness gaps: (1) **L6 articulation region
indices** (the manifest wanted int indices; the binder dropped them) and (2)
**metric-scale grounding** of the L5/L6 mass + package scale. Both are now
implemented, tested, and green. All CPU-pure, no API key, no spend. Opus
end-to-end (planned, implemented, verified on disk + gates re-run).

## 0. Verification first (never trust a summary)

Before building, re-ran **every** gate sessions 23/24 claimed and confirmed the
counts are real, not fabricated: astel_appearance **25**, astel_format **28**,
astel_solid **37**, pipelines/gpu **70**+3skip, services/api **62**+1skip,
@astel/manifest **15**, apps/web **tsc -b**·eslint·**43** + production build. The
prior retros were honest. (This matters — session 23 itself caught a fabricated
subagent report, so the rule stands: review = read the files + run the gates.)

## 1. L6 articulation binding — region indices + joint-vocab map (fixed a latent crash)

`astel_gpu.packaging` bound L6 articulation by passing the LLM's raw
`joint_type` straight into the manifest `LayerArticulation` and dropping the
parent/child region links (`parent_region=None, child_region=None`). Two real
problems:

- **Vocabulary mismatch = latent crash.** The physics-material LLM emits the
  URDF-ish vocabulary `astel_llm.JOINT_TYPES = {fixed, hinge, slider, ball,
  free}`, but the manifest enum (layer.schema.json) is `{revolute, prismatic,
  fixed, free}`. A `hinge`/`slider`/`ball` joint therefore raised a pydantic
  `ValidationError` — and because the whole L6 join sits under a broad
  best-effort `except`, it **silently dropped the entire L6 mass join** for any
  articulated object. No test exercised a populated articulation, so it was green.
- **Dropped region links.** The manifest wants integer region *indices* (parallel
  to the per-gaussian region map); the binder hard-coded `None`.

New pure helper `build_l6_articulation(raw_articulation, raw_regions)`:
builds a region-name→index map, maps the joint vocabulary via `_JOINT_TYPE_MAP`
(`hinge`→`revolute`, `slider`→`prismatic`, `ball`→`free` — the manifest has no
spherical joint, so a 3-DOF ball is reported as `free` rather than
over-constrained), and resolves parent/child names to indices. **Honest by
construction** (CLAUDE.md §10.4): an unresolved region name or unmapped joint is
**omitted** (not a crash, not invented), and `axis` is never set — the LLM gives
no joint axis, so none is fabricated.

**A real schema finding while testing:** the manifest schema forbids *null*
members on articulation entries (each field is optional but typed). Passing
`axis=None` explicitly marked the field "set", so the builder serialized it as
`null` → `axis: None is not of type 'array'`. Fixed by only setting resolved
fields (the pattern the existing passing articulation test already used).

## 2. Metric-scale grounding (L1 metric scale → L5/L6 mass + package scale)

Mass + scale were ungrounded (`meters_per_unit = 1.0`, `scale_grounded: false`)
because nothing threaded the Generation Spec's VLM size estimate into packaging.
Closed end-to-end:

- New pure helper `meters_per_unit_from_longest_axis(longest_axis_m, positions)`:
  `meters_per_unit = longest_axis_m / (largest L3 AABB extent)`, so
  `model_extent × meters_per_unit == longest_axis_m`. Falls back to `1.0`
  (ungrounded) on a non-positive estimate or degenerate extent — a scale is never
  fabricated.
- `write_layer_stack` gained an optional `longest_axis_m`; when supplied it
  derives `meters_per_unit`, which flows into both `compute_l6_masses` (the L6↔L5
  mass join → `scale_grounded: true`) and the package manifest
  (`coordinate_system`/`scale`).
- `astel_gpu.produce` CLI gained `--longest-axis-m`, threaded through the two
  **generative** paths (image, text). The **smoke** path stays ungrounded on
  purpose — its geometry is not the object, so a metric scale would be meaningless.
- The API submit flow now runs the **Generation Spec stage first** (it conditions
  generation, CLAUDE.md §4) so its estimate can be passed to the producer via the
  dispatch (`produce_artifacts_dispatch(..., longest_axis_m=...)` →
  `--longest-axis-m`). `apply_spec_scale_to_report` + the L6 physics stage still
  run after produce (they patch the produced report / need the produced asset);
  the billing/refine semantics are unchanged. New helper `_spec_longest_axis_m`
  extracts the estimate only from a successful spec (else `None` → ungrounded).

**Honest scope:** in the current ordering the producer's `out_dir` still has no
`l6.json` at packaging time (the physics stage writes it to the store afterward),
so the GPU producer's L6 *mass-join* binding remains latent in production — it
lights up when `l6.json` is present (tested), exactly as session 23 documented.
What ships today is the grounded **package scale** and the corrected
**articulation** binding; the metric mass join is proven by unit + integration
tests and flows the moment the physics stage runs before packaging.

## 3. Tests (all CPU-pure, no GPU)

- `meters_per_unit_from_longest_axis`: basic ratio, longest-axis selection,
  non-positive estimate, degenerate/empty extent → ungrounded (5).
- `build_l6_articulation`: full joint-vocab map (parametrised over all 5
  `JOINT_TYPES`, the regression guard for the latent crash), region-index
  resolution, unknown region → omitted, unknown joint → omitted, empty (8).
- `write_layer_stack` integration: a pre-placed articulated multi-region
  `l6.json` binds hinge→revolute + region indices and a `longest_axis_m` grounds
  the package scale; plus grounded-vs-ungrounded package-scale tests (3 — kept
  CoACD-free so the suite stays fast).
- API: dispatch threads/omits `--longest-axis-m` (2); `_spec_longest_axis_m`
  extraction (3).

## 4. Gates — all green (Opus-run)

- `astel_appearance`: ruff · mypy --strict (13) · **25 pytest**
- `astel_format`: ruff · mypy · **28 pytest**
- `astel_solid`: ruff · mypy · **37 pytest**
- `pipelines/gpu`: ruff · mypy (37) · **87 pytest**, 3 skipped (CUDA off-launcher)
- `services/api`: ruff · mypy --strict (26) · **67 pytest**, 1 skipped
- `@astel/manifest`: typecheck · eslint · **15 vitest**
- `apps/web`: **tsc -b** · eslint · **43 vitest** + production build ✓

(Unchanged packages — appearance/format/solid/manifest/web — re-verified at the
top of the session; only `pipelines/gpu` (+17 tests) and `services/api` (+5) were
touched this session.)

## 5. Honest gaps / next

- The GPU producer's L6 **mass-join** binding is still latent in the live flow
  (physics-material `l6.json` is written to the store after packaging). The clean
  fix is to move the physics-material stage before packaging or to do a post-hoc
  store-side join; both are M5-adjacent and were out of scope for "finish M4"
  (they touch the billing-sensitive submit ordering more than a cheap win should).
- Metric grounding only meaningfully flows on **text + GPU producer + a successful
  spec** (fixture/key); image has no spec scale, and the default offline/stub
  paths stay honestly ungrounded.
- Per-region volume segmentation (region-map → SDF intersection) is still future
  work; multi-region mass uses the flagged mean-density approximation.
- No live-browser screenshot (no Playwright/launch harness present) — the studios
  remain covered by unit tests + a clean production build.
- **M4 is complete.** Next: **M5 pipeline-readiness** (Unity/UE5 plugins that
  consume the now-correct L5 collision + L6 mass/material/articulation,
  KHR_gaussian_splatting glTF export, SDK + MCP server) — the engine plugins are
  the direct consumer of the articulation indices fixed here — or the
  **text→multiview bridge** (mission modality #1).
