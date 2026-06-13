# astel_format

Python reader/writer for the `.astel` package format (manifest-v0).

See `docs/specs/manifest-v0.md` and `docs/specs/schemas/*.json` for the
authoritative spec; this package's pydantic models and JSON-schema copies
in `src/astel_format/schemas/` mirror that contract. Where prose and schema
disagree, the schema wins.

## Usage

```python
from astel_format import AstelPackage, build_minimal_package

pkg = build_minimal_package(
    l3_ply_path="splats.ply",
    provenance=provenance_array,  # float32, one per gaussian, in [0, 1]
    quality_report=quality_report,
    ...
)
pkg.write("asset.astel")

loaded = AstelPackage.read("asset.astel")
```

## Gates

```
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest
```
