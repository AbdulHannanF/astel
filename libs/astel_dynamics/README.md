# astel-dynamics — L7 deformation timeline

Implements the **L7 Dynamics layer** (CLAUDE.md §3) for the `.astel` Gaussian-splat
format: a compact deformation timeline over a static base splat cloud using
**Linear-Blend Skinning (LBS)** with K control nodes.

Pure CPU / torch-free — validates the format and math against analytic ground
truth (global rigid rotation recovers zero reconstruction error; high-rank random
motion produces honestly large residuals).

## Representation

```
base_positions (N, 3) float32
  + weights        (N, K) float32   — per-gaussian blend weights, rows sum to 1
  + node_positions (K, 3) float32   — rest positions of control nodes
  + node_transforms (F, K, 3, 4) float32 — per-frame [R|t] per node
  → deformed_positions (N, 3) float32 per frame
```

The LBS deformation for gaussian *n* at frame *f*:

```
deformed[n] = Σ_k  weights[n,k] * (R[f,k] @ base[n] + t[f,k])
```

This is an **affine LBS approximation** — not strict rigid-body per node. Motion
that is well-described by LBS (global rotations, simple bends) fits tightly.
High-rank or incompressible motion produces honestly large residuals reported in
`FitReport` — Astel never fabricates accuracy.

## Modules

| Module | Contents |
|--------|----------|
| `timeline.py` | `Timeline` dataclass + JSON serialization |
| `field.py` | `DeformationField` dataclass + LBS `apply()` |
| `fit.py` | `fit_deformation_field()` + `FitReport` |
| `pack.py` | Binary `.bin` serialization (`ASTLDYN0` magic) |
| `baked.py` | `bake_per_frame()` — explicit (F,N,3) positions |

## Binary format (`ASTLDYN0`)

```
[8 bytes]  magic  b"ASTLDYN0"
[4 bytes]  uint32 N  (n_gaussians)
[4 bytes]  uint32 K  (n_nodes)
[4 bytes]  uint32 F  (n_frames)
[N*K*3 * 4 bytes]  node_positions  float32 C-contiguous
[N*K * 4 bytes]    weights         float32 C-contiguous
[F*K*3*4 * 4 bytes] node_transforms float32 C-contiguous
```

Round-trip is lossless to float32 precision.
