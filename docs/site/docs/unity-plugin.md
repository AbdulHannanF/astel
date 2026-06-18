# Unity plugin

The Astel Unity package (`com.astel.importer`) reads the **`engine.json`** sidecar and auto-configures physics from the L5/L6 layers. Splat rendering is delegated to the existing [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) package.

`engine.json` is a flat physics-setup descriptor (schema `astel.engine-setup/v0`) the producer emits **alongside** the package — it denormalises the L5 collision + L6 material/mass/articulation layers so the importer never has to walk the nested `manifest.json` or chase its file-referenced sidecars. It is a **sibling delivery artifact** (download it next to the splat it names in `splat_file`), not a member inside the `package.astel` zip.

## Install

1. Open **Package Manager → Add package from disk…**
2. Select `plugins/unity/com.astel.importer/package.json`.
3. Also install **UnityGaussianSplatting** (required for splat rendering).

## Import an asset

**Astel → Import Asset** opens the import window.

1. Download a generation's artifacts into a folder (it contains `engine.json` + the splat, e.g. `l3.spz`).
2. Browse to that **folder** and click **Import**.
3. A new `GameObject` is created with:
   - `Rigidbody.mass` from the L6↔L5 metric mass (`mass_kg`)
   - `Rigidbody.centerOfMass` from L5 COM (X-flipped for Unity's left-handed frame)
   - `PhysicMaterial` from L6 region 0 friction/restitution
5. The console logs the path to `l3.spz`. Drag it into a `GaussianSplatAsset`, then attach `GaussianSplatRenderer` to the same `GameObject`.

## Scripting

```csharp
using Astel;

// Load engine.json from a folder of downloaded artifacts
var manifest = AstelPhysicsSetup.LoadFromDirectory("Assets/AstelAssets/my-asset");

// Or point directly at the engine.json file
var manifest = AstelPhysicsSetup.LoadEngineSetup("path/to/engine.json");

// Apply physics to a GameObject
new AstelPhysicsSetup(manifest, gameObject).Apply();
```

## Coordinate convention

The manifest stores data in 3DGS world space (right-handed, +Y up, metres).
Unity is left-handed, +Y up, +Z forward. The plugin applies:

- `pos_unity.x = −pos.x` (X negated)
- Quaternion: `(−qx, qy, qz, −qw)` — handedness flip

See [Coordinate conventions](coordinate-conventions.md) for full details.

## Limitations

- Articulation hints from L6 are logged but not automatically wired to `ArticulationBody` — do this manually for articulated objects.
- Collision hulls from L5 must be imported as `MeshCollider` assets manually (the hull OBJ files are in the extracted directory).
- Tested on Unity 6 LTS (6000.x). Earlier versions may need minor adaptation.
