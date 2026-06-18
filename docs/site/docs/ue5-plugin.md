# Unreal Engine 5 plugin

The Astel UE5 plugin (`AstelPlugin`) reads the **`engine.json`** sidecar and auto-configures mass override, centre of mass, and physical material on a `UStaticMeshComponent`. Splat rendering is delegated to an existing plugin (NanoGS, XScene, or similar).

`engine.json` is a flat physics-setup descriptor (schema `astel.engine-setup/v0`) the producer emits **alongside** the package; it denormalises the L5/L6 layers so the plugin never has to walk the nested `manifest.json` or chase its file-referenced sidecars. Download it next to the splat it names in `splat_file` — it is a sibling delivery artifact, not a member inside the `package.astel` zip.

## Install

1. Copy `plugins/unreal/AstelPlugin` into your project's `Plugins/` folder.
2. Regenerate project files.
3. Enable the plugin in **Edit → Plugins → Astel Importer**.

## Usage (C++)

```cpp
#include "AstelManifestReader.h"

// Load engine.json from a folder of downloaded artifacts
FAstelManifest Manifest = UAstelManifestReader::LoadFromDirectory(
    FPaths::ProjectContentDir() + "AstelAssets/my-asset"
);

// Apply physics to a mesh component
UAstelManifestReader::ApplyPhysics(MyMeshComponent, Manifest);
```

`ApplyPhysics` sets:

- `FBodyInstance.bOverrideMass = true`, `SetMassOverride(Manifest.MassProps.MassKg)`
- `FBodyInstance.COMNudge` from L5 COM (converted to UE5 cm, axes remapped)
- `FBodyInstance.PhysMaterialOverride` from L6 region 0 friction/restitution
- Articulation hints are logged for manual `UPhysicsConstraintComponent` setup

## Coordinate convention

3DGS world → UE5:

```
pos_ue = (−z×100,  x×100,  y×100)   [cm]
q_ue   = (−qz,  qx,  qy,  −qw)
```

See [Coordinate conventions](coordinate-conventions.md).

## Blueprint usage

`FAstelManifest` and `FAstelMassProps` are `USTRUCT(BlueprintType)`, so you can read and use them in Blueprints via a custom Blueprint Function Library (not included — implement `UAstelBPLibrary` if needed).

## Limitations

- `engine.json` and the splat are sibling delivery artifacts — download them into a folder, then call `LoadFromDirectory` on it. No zip extraction is needed (the plugin does not read `package.astel`).
- Convex hull sets from L5 require manual import as `UConvexElem` or via a ProceduralMeshComponent.
- Tested against UE5.6.
