# Splat format notes — `astel_splat_io`

Verified 2026-06-13 against upstream sources (training-data-stale per
CLAUDE.md operating rule #1 — re-verify before relying on this for new
work).

## `.ply` — INRIA binary little-endian (archival master)

Re-derived from `pipelines/stub/make_sample_splat.py`. No external spec
dependency; this is the de-facto 3DGS ecosystem convention (header +
`x y z f_dc_0..2 opacity scale_0..2 rot_0..3`, float32, band-0 SH only).

Status: **fully implemented** (`ply.py`: `write_ply`, `read_ply`).

## `.spz` — Niantic SPZ

- Source: https://github.com/nianticlabs/spz (`src/cc/load-spz.h`,
  `src/cc/load-spz.cc`, HEAD as of 2026-06-13).
- License: **MIT** (Copyright 2025 Niantic Labs, Copyright 2025 Adobe Inc.).

Implemented: **version 3** container — 16-byte header
(`magic=0x5053474e "NGSP"`, `version`, `numPoints`, `shDegree=0`,
`fractionalBits=12`, `flags=0`, `reserved=0`), single gzip stream, attribute
order `positions, alphas, colors, scales, rotations, sh`. Positions: 24-bit
signed fixed-point (12 fractional bits). Scales: `uint8((log_scale+10)*16)`.
Alpha: `uint8(sigmoid(opacity)*255)`. Colors: `uint8(f_dc*0.15*255 +
0.5*255)`. Rotations: SPZ v3 "smallest three" (10-bit magnitude + sign per
non-largest quaternion component, 2-bit largest-index).

**Assumption / note**: although the repo's `LATEST_SPZ_HEADER_VERSION == 4`
(a newer 32-byte-header, multi-stream ZSTD/TOC container exists in the same
codebase as of the May 2026 "SPZ 4" release), the `saveSpz`/`loadSpz` /
`serializePackedGaussians`/`deserializePackedGaussians` entry points still
emit/parse the **legacy 16-byte-header gzip container** at HEAD. We target
that gzip container at version 3 (the most recent version of it, with
smallest-three quaternions) to avoid a ZSTD dependency. `read_spz` accepts
versions 1-4 of this gzip container but raises `NotImplementedError` for v1
(float16 positions) and v2 ("first three" rotations), since `SplatCloud`
round-trips are only exercised against v3 output. If the v4 TOC/ZSTD
container becomes the only one real-world `.spz` files use, this module
will need a follow-up.

Status: **fully implemented** for v3 (`spz.py`: `write_spz`, `read_spz`),
round-trip tested within quantization tolerance.

## `.sog` / SOGS — PlayCanvas Self-Organizing Gaussians

- Sources:
  - https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
    (format reference)
  - https://github.com/playcanvas/splat-transform (reference
    tooling/spec) — **MIT License**
  - https://github.com/playcanvas/sogs (original compressor,
    k-means/PLAS) — **Apache-2.0 License**

Container: ZIP of `meta.json` + lossless WebP textures, one pixel per
gaussian, row-major (`x = i % W`, `y = i // W`).

### Implemented
- `meta.json` (`version=2`, `count`, `width`, `height`, per-attribute
  blocks).
- `means_l.webp` / `means_u.webp`: signed-log-domain 16-bit position
  encoding with per-axis `mins`/`maxs`.
- `scales.webp`: per-axis 256-entry codebook + RGB indices.
- `sh0.webp`: 256-entry codebook for band-0 DC colour (RGB) + opacity in
  alpha (UNORM8).
- `quats.webp`: smallest-three quaternion packing (RGB = 3 components in
  `[-sqrt(1/2), sqrt(1/2)]`, A = `252 + index_of_largest`).
- Round-trip read/write for all of the above.

### NOT implemented (explicit `NotImplementedError` / documented gaps)

1. **Codebook generation**: the reference SOGS compressor builds the 256-
   entry `scales`/`sh0` codebooks via k-means (PLAS-ordered) over the whole
   cloud. This module uses **uniform quantile binning** — a documented
   simplification with higher quantization error than the reference, but a
   real codebook + real indices (not a stub). TODO: swap in k-means if
   fidelity becomes a product issue.
2. **`shN` (higher-order spherical harmonics)**: `SplatCloud` has no SH-rest,
   so `shN` is never emitted. `read_sog` raises `NotImplementedError` if a
   bundle declares `shN` — we do not silently drop real SH data.
3. **Spatial sort / PLAS re-ordering** for 2D-texture locality: not
   performed; splats are written in input order. (Provenance alignment is
   therefore trivial — identity permutation — for this writer.)
4. **`lod-meta.json` streaming bundles**: out of scope; only the
   single-file, non-streaming `meta.json` bundle is implemented.

## Provenance sidecar (`*.astl.json` + `.bin`)

Per `docs/specs/manifest-v0.md` section 11.3 / section 5: `SCALAR` `UNORM8`
(`q = round(p*255)`, `p = q/255`), tightly packed, `count` = splat count,
index-aligned to the *exported* splat order. `write_provenance_sidecar`
accepts an optional `permutation` so callers whose exporter reordered splats
(none of ours currently do — see SOG note above) can keep provenance
aligned. Fully implemented (`provenance.py`).
