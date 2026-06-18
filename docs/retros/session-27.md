# Session 27 retro (2026-06-18)

**M5 pipeline-readiness CLOSED: KHR_gaussian_splatting glTF export, coordinate-convention module, Unity + UE5 plugins, Python SDK + MCP server, TypeScript SDK, docs site.**  All gates green.

## What shipped

### 1. KHR_gaussian_splatting glTF / GLB export (new in `astel_splat_io`)

`libs/astel_splat_io/src/astel_splat_io/gltf.py` — `write_gltf` / `read_gltf`:

- Writes binary GLB (glTF 2.0) with the KHR_gaussian_splatting RC schema (Khronos, Feb 2026).
- 4 accessors per Gaussian: POSITION (VEC3), _ROTATION (VEC4, xyzw), _SCALE (VEC3, world-space σ), COLOR_0 (VEC4, rgba 0..1 from SH band-0 + sigmoid opacity).
- Quaternion reorder `(w,x,y,z) → (x,y,z,w)` only — no position transform (3DGS world == glTF frame).
- Round-trip is lossless to float32 precision.
- Golden-file + structural + round-trip tests: **14 new tests, 35 total** in astel_splat_io.
- Exported from `astel_splat_io.__init__`.

### 2. Coordinate-convention module + reference doc

`libs/astel_splat_io/src/astel_splat_io/conventions.py`:

- `gltf_positions/quats` — identity/reorder.
- `unity_positions/quats` — X-negate + quat handedness flip `(−qx, qy, qz, −qw)`.
- `unreal_positions/quats/scales` — axis remap `(x,y,z) → (−z×100, x×100, y×100)` cm.
- **10 new tests**, all passing.

`docs/architecture/coordinate-conventions.md` — canonical reference with the math, round-trip rules, and engine checklist.

### 3. Unity package (`plugins/unity/com.astel.importer/`)

- `AstelManifest.cs` — typed deserialisation of manifest.astel L5/L6 fields.
- `AstelPhysicsSetup.cs` — reads manifest, adds Rigidbody (mass + COM X-flipped), sets PhysicMaterial from L6 region 0, logs articulation hints.
- `AstelImportWindow.cs` — editor window: extracts .astel zip → assets folder, calls physics setup, logs splat file path.
- `AstelPhysicsSetupTests.cs` — 7 NUnit tests covering mass, COM X-flip, meters_per_unit scaling, null L5/L6 graceful handling.
- `package.json` for Unity Package Manager (Unity 6 LTS).

### 4. UE5 plugin (`plugins/unreal/AstelPlugin/`)

- `AstelPlugin.uplugin` — plugin descriptor (UE5.6, Editor module).
- `AstelManifestReader.h/.cpp` — FAstelManifest/FAstelMassProps/FAstelPhysicsRegion/FAstelArticulationHint USTRUCTs; JSON parsing via UE's TJsonReader; `ApplyPhysics(UStaticMeshComponent*, manifest)` sets mass override, COM in UE cm (axis remap), physical material override, logs articulation hints.
- `AstelPlugin.Build.cs` — module dependencies (Core, Json, UnrealEd, PhysicsCore).

### 5. Python SDK (`packages/sdk-python/`)

`astel-sdk` package:
- `AsyncAstelClient` — async httpx-based client with `generate`, `get_generation`, `wait_for_generation`, `upload_capture`, `download_artifact`, `download_all_artifacts`.
- `AstelClient` — sync wrapper via `asyncio.run`.
- `types.py` — `Generation`, `CaptureRef`, `ArtifactRef`, `BillingSummary`, `PricingResource` as Pydantic models; `gen.is_ready`, `gen.is_failed`, `gen.artifact_url(name)` helpers.
- **10 tests** (respx mock transport), ruff ✓ mypy ✓.

### 6. MCP server (`packages/sdk-python/src/astel_sdk/mcp_server.py`)

3 tools via `mcp.server.fastmcp.FastMCP`:
- `generate_asset(prompt, modality, mode, capture_id)` — submit generation, return ID + initial status.
- `get_asset(generation_id)` — poll status + artifacts + quality_report_url.
- `list_pricing()` — credit schedule.

`astel-mcp` console script; `--transport stdio` (default, Claude Desktop) or `--transport sse`; reads `ASTEL_API_URL` + `ASTEL_API_KEY`.

### 7. TypeScript SDK (`packages/sdk-ts/`)

`@astel/sdk` package:
- `AstelClient` — fetch-based, typed, async.
- `AstelError` — named error class with `status` + `body`.
- Full type coverage: `Generation`, `ArtifactRef`, `BillingSummary`, `CaptureRef`, `GenerateOptions`, etc.
- `waitForGeneration` with configurable poll interval + timeout.
- **9 vitest tests**, tsc clean.
- Integrated into pnpm workspace.

### 8. Docs site (`docs/site/`)

MkDocs Material config + 10 pages:
- `index.md` — product overview + feature comparison table.
- `self-host.md` — one-command start, env vars, Docker Compose.
- `api-reference.md` — full REST endpoint reference.
- `splats-101.md` — "Splats 101 for studios" explainer.
- `coordinate-conventions.md` — summary table + Python code.
- `unity-plugin.md` — install, import, scripting, caveats.
- `ue5-plugin.md` — install, C++ usage, Blueprint, caveats.
- `gltf-export.md` — schema, status, API.
- `sdk-python.md` — quick start + API reference.
- `mcp-server.md` — install, tools, Claude Desktop config.

## Gates (all green)

- `astel_splat_io`: ruff · mypy · **35** (+11 new)
- `astel_format`: **28** (unchanged)
- `astel_solid`: **37** (unchanged)
- `astel_appearance`: **25** (unchanged)
- `pipelines/gpu`: **94**+3skip (unchanged)
- `services/api`: **71**+1skip (unchanged)
- `@astel/manifest`: **15** (unchanged)
- `apps/web`: tsc -b · eslint · **43** (unchanged)
- `@astel/sdk` (new): tsc -b · vitest **9**
- `astel-sdk` (new): ruff · mypy · pytest **10**

## Honest gaps / next

- **Engine CI:** headless Unity / UE5 in CI requires licensed runners + large machines. The physics mapping logic is unit-tested in pure code; a manual engine pass is the remaining verification. Flag if engine CI runners cross the §10.2 cost threshold before adding them.
- **KHR_gaussian_splatting RC churn:** schema tagged `"sh_degree": 0` and RC status noted in docs. A spec change before ratification is a localized exporter fix.
- **`.astel` zip extraction in UE5:** UE5 ships no built-in zip reader; the UE5 plugin documents extracting externally before calling `LoadFromDirectory`.
- **MCP `[mcp]` extra:** mcp package is optional; `mcp_server.py` exits cleanly with a helpful message if not installed.
- **Docs site build:** `mkdocs serve` in `docs/site/` serves the site locally; CI hook and deployment not yet wired.

**M5 is closed.** Next: **M6 dynamics + scenes** (video→4DGS L7, scene seeds, LOD streaming) or the **text→multiview bridge** (mission modality #1).
