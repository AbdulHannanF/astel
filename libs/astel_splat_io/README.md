# astel_splat_io

Splat export/import writers for Astel: `.ply` (archival INRIA layout), `.spz`
(Niantic SPZ v3, gzip), `.sog`/SOGS (PlayCanvas Self-Organizing Gaussians,
partial), and the `*.astl.json` + `.bin` provenance sidecar (manifest-v0
section 11.3).

See `FORMATS.md` for spec sources, license notes, and what is fully
implemented vs. stubbed.

## Usage

```python
from astel_splat_io import SplatCloud, write_ply, write_spz, write_provenance_sidecar

cloud = SplatCloud(positions=..., colors_dc=..., opacity=..., log_scales=..., quats=...)
write_ply(cloud, "asset.ply")
write_spz(cloud, "asset.spz")
write_provenance_sidecar(cloud, provenance, "asset.spz", "asset.astl.json")
```

## Gates

```
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest
```
