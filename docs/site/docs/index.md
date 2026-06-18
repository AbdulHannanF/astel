# Astel

**Layered Gaussian splat generation** — the successor to mesh-based 3D generators.

Astel turns text, photos, and video into `.astel` packages: photorealistic Gaussian splats with collision, physics materials, and BRDF decomposition already bound in. Drop them into Unreal, Unity, Blender, or Three.js, and they just work.

## What makes Astel different

| Astel | Competitors |
|---|---|
| Native Gaussian splats — no mesh conversion | Mesh output, then a splat bake |
| L5 collision + L6 physics material **in the file** | Manual physics setup in engine |
| Truth Meter — geometric error, hallucination heatmap | No accuracy reporting |
| Relight Studio — swap HDRI live | Lighting baked into colour |
| Physics Sandbox — poke the object in-browser | No physics preview |
| Self-hostable — one GPU, one command | Cloud-only |

## Quick start

```bash
# Start the API (requires a GPU for real generation)
pip install astel-sdk
astel-api

# Generate an asset
from astel_sdk import AstelClient
client = AstelClient()
gen = client.generate(prompt="a worn brass astrolabe on a wooden base")
client.download_all_artifacts(gen.id, "out/")
```

## Layer stack

Every asset is a stack of layers that you can inspect individually:

- **L0** Sparse point cloud (SfM or generative seed)
- **L1** Dense cloud with normals
- **L2** Coarse Gaussians (fast feed-forward)
- **L3** Refined surface Gaussians (hero layer)
- **L4** Appearance / BRDF decomposition
- **L5** Collision + solid geometry (physics only — never the asset)
- **L6** Physics material + articulation hints
- **L7** Dynamics keyframes (video input)

See [Layered asset model](layer-model.md) for the full spec.
