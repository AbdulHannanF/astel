# Session 28 retro (2026-06-18)

**M5 audit — found the engine plugins were fiction, fixed the unplugged glTF
export and two broken SDK models, locked the coordinate math with a real test,
and made the plugins consume a tested `engine.json` sidecar.** The task: review
the just-landed M5 (session 27) for anything fake / unplugged / bloat, and fix
it. Session 27's retro — unlike 24–26 — had **no "verification first" pass and
did not mark its gates Opus-run**, so it got the skeptical treatment. Opus
end-to-end (read every artifact on disk, re-ran the gates, fixed, re-ran).

## 0. Verification first (the rule sessions 23–26 set)

Re-ran the session-27 gate that is checkable here: `astel_splat_io` **35** —
real. The glTF exporter, SDK clients, and MCP server are genuine, well-built
code. But three things were fake/unplugged/wrong, and one was hollow.

## 1. glTF export was unplugged (HIGH) — fixed

`astel_splat_io.write_gltf` (KHR_gaussian_splatting GLB) was exported + tested,
but **no producer, API route, or CLI ever called it** — every shipped asset had
`.ply/.spz/.sog` and never a `.glb`. The "cleanest first win" of M5 produced
nothing in the product. Wired `l3.glb` into **both** producers
(`packaging.write_layer_stack` + the CPU stub `producer.produce_artifacts`),
mapped it as an L3 delivery format in billing, with round-trip tests on both
paths. The billing honesty-guard test (every `_ARTIFACT_LAYER` key must be
produced or tracked-missing) caught the new mapping — a good test; updated.

## 2. Engine plugins parsed a fictional manifest (CRITICAL) — fixed

Both the Unity and UE5 plugins — M5's stated core differentiator — could **not**
configure physics from a real `.astel`:

- They loaded **`manifest.astel`**; the real package member is **`manifest.json`**.
- They expected a **flat** schema (`l5.mass_props.mass_kg`, `l6.regions[].friction`,
  top-level `meters_per_unit`, `articulation[].joint_type/region_a`). The real
  manifest is **nested** and references mass/material by **file path**
  (`l5-mass.json`, `l6.json`), with articulation as
  `physics_material.articulation[].{type, parent_region, child_region}`.
- The Unity NUnit tests built the fictional object **in memory** and asserted on
  it; they never loaded a real asset. UE5 had no test at all.

**Decision (founder said "do what's best, document it"): emit a flat
`engine.json` sidecar from the producer, repoint the plugins to it.** Rationale:

- The load-bearing guarantee ("the data is real") then lives in **Python, which
  is testable in this environment**; the C#/C++ change shrinks to one filename.
- An engine importer should read a small denormalised descriptor, not walk a
  nested manifest + chase file-referenced sidecars in C#/C++. This is how real
  engine import pipelines work.
- Reading the two parsers proved the win: both **already parse exactly the
  nested shape `engine.json` emits** (`meters_per_unit`, `l5.mass_props.{…}`,
  `l6.regions[].{name,density_kg_m3,friction,restitution}`,
  `l6.articulation[].{joint_type,region_a,region_b}`). The *only* real code bug
  was the filename. So the fix is genuinely minimal, not a rewrite.

New pure `astel_gpu.packaging.build_engine_setup` (schema
`astel.engine-setup/v0`) assembles the flat descriptor from the real L5
(`l5-mass.json` solidify summary) + L6 (`l6.json` regions + the L6↔L5 metric
**mass join**) + the manifest scale. **Honest by construction:** `l5`/`l6` are
`null` when absent; `mass_kg` is the metric L6 mass when present, else `0.0` +
a caveat (a model-unit "mass at unit density" is never passed off as kg);
COM/inertia stay in model units (the plugin scales by `meters_per_unit`);
`scale_grounded` is surfaced. Emitted by `write_layer_stack` as a sibling
delivery artifact (alongside `l3.spz`, which it names in `splat_file`).

Plugin changes (repoint only): Unity + UE5 readers now load `engine.json`; the
Unity import window takes a downloaded-artifact **folder**; removed the UE5
zip-extraction `LoadFromAstelPackage`. Added a Unity NUnit test that parses a
**real `engine.json` payload** (the exact emitter shape), closing the
feeds-fiction-to-itself gap. Docs (unity-plugin, ue5-plugin, splats-101)
corrected — they had said `manifest.astel` and conflated package members with
sibling artifacts.

**Caveat (honest):** no Unity/UE5 toolchain in this environment, so the C#/C++
edits are **not compiler-verified** — only the `engine.json` contract is tested
(Python). The plugin changes are small and mechanical (filename + a parse test);
a real engine pass remains the outstanding verification, as session 27 already
flagged for engine CI.

## 3. Both SDKs' `PricingResource` model was wrong (MEDIUM) — fixed

SDK type was `{schedule, tiers}`; the real `/v1/pricing` returns
`{credit_usd_rate, layers[], modes, notes}`. Pydantic/TS ignore extras, so
`client.pricing()` and the MCP `list_pricing` tool returned an **empty** object.
The `test_pricing` test mocked the fictional shape and asserted on it. Rewrote
both SDK models to mirror the API, de-fictioned the tests to real payloads, and
generalised the MCP docstring (it hard-coded credit numbers that can drift).

## 4. Both SDKs' `BillingSummary` was wrong (LOW) — fixed

SDK had `{total_credits, total_usd, lines}`; the API sends
`{mode, refine_of, items[], total_credits, total_usd, credit_usd_rate, caveats}`
— so `billing.lines` was always empty and the per-line ledger was dropped. Added
`CreditLineItem`/`LayerPriceRef` and corrected both SDKs + tests.

## 5. Coordinate quat transforms had no correctness test (MEDIUM) — fixed

`conventions.py` tested only norm + component reorder — which cannot catch a
wrong handedness formula. Worked the math: the Unity `(-x,y,z,-w)` and Unreal
`(-z,x,y,-w)` quats are each the **negation** of the canonically-mirrored
quaternion (= same rotation), so they are correct — but the property was
unguarded. Added a parametrized test asserting
`rotmat(engine_quat) == M · rotmat(source_quat) · Mᵀ` for both engines. Passes.

## 6. Gates — all green (Opus-run)

- `astel_splat_io`: ruff · mypy · **37** (+2 quat-correctness)
- `sdk-python` (`astel-sdk`): ruff · mypy · **11** (+1 billing parse)
- `sdk-ts` (`@astel/sdk`): tsc -b · **9** (models corrected)
- `pipelines/gpu`: ruff · mypy · **97**+3skip (+3: l3.glb round-trip, build_engine_setup ×2)
- `services/api`: ruff · mypy --strict (18) · **71**+1skip
- Untouched (astel_format/solid/appearance, @astel/manifest, apps/web): not modified.

## 7. Honest gaps / next

- **Plugin C#/C++ is not compiler-verified here** (no Unity/UE5 toolchain). The
  `engine.json` contract is tested in Python; a real engine pass is the
  remaining verification — gate engine CI runners against §10.2 when added.
- `engine.json` only carries L5 mass when solidify ran **and** the L6 mass join
  fired (text + physics fixture/key); otherwise `l5`/`l6` are honestly `null`.
- The CPU **stub** producer emits `l3.glb` but no `engine.json` (it has no
  L5/L6) — engine physics auto-setup is a GPU-path feature, honestly.
- Per-region volume still `not-segmented` (mean-density mass, flagged) — unchanged.

**M5 is now real** (glTF export plugged, plugins consume a tested real sidecar,
SDK models match the API, coordinate math locked). Next: **M6** (video→4DGS L7,
scene seeds, LOD streaming) or the **text→multiview bridge** (mission modality #1).
