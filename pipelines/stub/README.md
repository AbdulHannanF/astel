# pipelines/stub

The Astel **stub pipeline** for M1. It has no GPU dependency and produces a
procedural placeholder asset standing in for the real layered pipeline.

## `make_sample_splat.py`

Generates the checked-in sample Gaussian splat the web viewer loads:
`apps/web/public/samples/astel-sample.ply` — a torus knot of ~48k gaussians
with a brass→teal gradient and flattened, surfel-like splats.

The output is a standard **INRIA-layout** binary PLY (`x y z`, `f_dc_0..2`,
`opacity`, `scale_0..2`, `rot_0..3`), the format the 3DGS ecosystem and our
Spark-based viewer consume. `f_rest_*` (higher-order SH) is omitted.

### Regenerate the sample

```bash
uv sync                                   # first time, installs numpy + dev tools
uv run python make_sample_splat.py        # writes to apps/web/public/samples/
```

### Test / lint / type-check

```bash
uv run pytest        # golden-file + structure tests for the PLY writer
uv run ruff check .
uv run mypy
```
