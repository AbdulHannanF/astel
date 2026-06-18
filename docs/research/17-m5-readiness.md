# M5 readiness — pipeline-readiness (engine plugins, glTF export, SDK + MCP)

**Status:** prep doc, written end of session 26 (2026-06-18). M4 is closed (world-
awareness layers real + bound, generation photorealistic, no packaging hangs).
This sequences M5 per CLAUDE.md §5 / §9 (M5) and records the **verified-current**
external state (training data is stale by definition — re-verify at build time).

## 0. What M5 is (binding, CLAUDE.md §9 M5)

> Unity + UE5 plugins with auto physics setup; KHR_gaussian_splatting glTF export;
> SDK + MCP server; docs site.

The thesis: Astel's differentiator in engines is **not** rendering splats (solved
by existing plugins) — it is **auto-configuring collision, mass, and materials**
from the L5/L6 layers no competitor ships. M5 is where the layered asset model
pays off in a real engine.

## 1. What is already real after session 26 (the substantive prep)

The data the engine plugins consume is now genuinely produced and bound — this was
the gating prerequisite and it is done:

- **Splat delivery formats:** `.ply` (archival), `.spz` (SPZ, on the Khronos
  compression track), `.sog` (best-effort) — all emitted by both producers.
- **L5 collision (now bound into `.astel`):** watertight isosurface (print/physics
  only), **convex hull set** (CoACD multi-hull for convex-friendly objects, scipy
  single-hull fallback — bounded, never hangs), mass / COM / inertia.
- **L6 physics-material (now bound + joined live):** per-region density →
  `l6-mass.json` (mass = density × L5 volume), and **articulation** mapped to the
  manifest joint enum with integer region indices (`hinge→revolute` etc.). Verified
  end-to-end: a produced package binds `l6: physics_material articulation=[…]`.
- **Quality report / Truth Meter** + `origin` taxonomy for honest provenance.

So a plugin reading an `.astel` can already find: splats + collision proxy + mass
properties + per-region material + articulation hints. **That is the M5 input
contract, and it is real now.**

## 2. External state — verified June 2026 (re-verify before adopting)

- **KHR_gaussian_splatting (glTF):** **release candidate announced 2026-02-03;
  ratification expected Q2 2026.** Stores position, orientation, scale, SH colour,
  opacity in glTF 2.0 mesh primitives; compression extensions for **SPZ** (Niantic)
  and **L-GSC** (Qualcomm) proposed on top. → Target the RC schema now, flag exports
  as RC until ratified; SPZ we already emit. (khronos.org press release; CG Channel
  2026-02; radiancefields.com.)
- **Unity:** aras-p **UnityGaussianSplatting 2.x** (Unity 6 LTS; D3D12/Metal/Vulkan)
  imports `.ply` **and** `.spz` natively with GPU sorting + compression presets. →
  We do not build a splat renderer; we ship a Unity package that imports our `.spz`
  and **auto-configures colliders (L5 convex set) + Rigidbody mass (L6) + physics
  materials (L6 friction/restitution)** from the sidecar manifest.
- **UE5:** multiple mature plugins import `.ply` (XScene/XVERSE, MLSLabs, **NanoGS**
  = Nanite-style LOD, UE 5.6+, free). → Same value-add: an Astel UE5 plugin that
  consumes our package and sets up collision + mass + physical materials.
- **MCP:** Anthropic's Model Context Protocol is the agent-integration surface
  (CLAUDE.md §7 "ship … an MCP server"). Re-verify the current MCP spec + SDK
  version at build time (see the `claude-api` skill).

## 3. Ordered M5 plan

1. **KHR_gaussian_splatting glTF export** (cleanest first win; pure-Python, no
   engine needed to test). Add a `.glb`/`.gltf` exporter for the L3 cloud against
   the RC schema (position/rotation/scale/SH/opacity), with a **golden-file test**
   that re-loads it headless. Coordinate convention: document the exact
   OpenGL(training) → glTF(+Y up, -Z forward) rotation **once**, with a test
   (CLAUDE.md §5 "document the exact rotations").
2. **Coordinate-convention reference doc + fixtures** for Unity (+Y up, left-handed,
   Z forward) and Unreal (+Z up, left-handed, cm units) — the documented rotation
   matrices + a tiny round-trip fixture each. This de-risks both plugins before any
   C#/C++ is written.
3. **Unity package** — import `.spz` via aras-p's importer (or our own), then read
   the `.astel` manifest sidecar and auto-create: MeshColliders/convex colliders
   from L5, Rigidbody.mass from L6, PhysicMaterial from L6 friction/restitution.
   Golden test: load a fixture asset in batch mode, assert the collider/mass values.
4. **UE5 plugin** — same contract against one of the existing splat plugins'
   import path; auto-configure UBodySetup collision + mass + physical material.
5. **SDK (Python + TS) + MCP server** — wrap the existing REST endpoints
   (`/v1/generations`, `/v1/captures`, `/v1/pricing`, artifacts) so agents/IDEs can
   generate assets programmatically. MCP server exposes "generate asset" / "get
   asset" tools over the same API. Re-verify the MCP spec first.
6. **Docs site** — self-host guide, API reference, "splats 101 for studios",
   coordinate-convention page.

## 4. Risks / founder gates

- **KHR RC churn:** the schema can change before ratification — keep the exporter
  behind a versioned module and a golden test so a spec bump is a localised fix.
- **Engine licensing / CI:** headless Unity/UE in CI needs licenses + large runners
  (a real cost item — flag if it crosses §10.2 thresholds). Until then, test the
  manifest→collision/mass mapping logic in pure unit tests + a manual engine pass.
- **No new founder spend** is required to *start* M5 (glTF export + convention docs +
  SDK are local); engine CI runners and any signed plugin distribution are the
  cost items to surface when reached.
- **L6 still gated on R-O2** for real material data (LLM key); the binding is real,
  but rich per-region materials only flow with a fixture/key. Convex-friendly
  objects get multi-hull collision; thin-featured ones get the single-hull fallback.

## 5. Recommended first M5 step

**KHR_gaussian_splatting glTF export** — it is pure-Python, testable headless with
a golden file, exercises the coordinate-convention work every later step needs, and
delivers the most broadly-useful interop artifact (any glTF viewer/engine) for the
least risk. The text→multiview bridge (mission modality #1) remains the alternative
priority if the founder wants modality completeness before interop.

Sources (verified 2026-06): khronos.org glTF Gaussian Splatting press release;
cgchannel.com (2026-02); radiancefields.com; github.com/aras-p/UnityGaussianSplatting;
github.com/xverse-engine/XScene-UEPlugin; cgchannel.com NanoGS (2026-03).
