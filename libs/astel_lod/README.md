# astel-lod — LOD streaming & splat budgets

Derives the **LOD (level-of-detail) streaming layer** from a Gaussian-splat cloud
(CLAUDE.md §8.6): ranks every Gaussian by perceptual importance and returns
index arrays for tier-capped subsamples that the caller uses to slice its own
cloud data.

**This library is torch-free, CPU-only, numpy-only.** It runs in CI and in the
packaging worker without any GPU context.

## Design: importance heuristic

The importance score for each Gaussian is:

```
importance(i) = opacity(i) * projected_footprint(i)
projected_footprint(i) = exp(log_scale_x) * exp(log_scale_y) * exp(log_scale_z)
                       = exp(log_scale_x + log_scale_y + log_scale_z)
```

This is a *perceptual proxy*, not an optimal saliency optimisation:

- **Opacity** gates whether the Gaussian contributes visibly at all (near-zero
  opacity Gaussians should always be culled first).
- **Projected footprint** (product of the three world-space semi-axis lengths)
  approximates the screen-space area at a fixed viewing distance.  Larger
  Gaussians cover more pixels and therefore carry more visual weight.

The formula is fast, deterministic, and monotone in both opacity and scale
individually. It does **not** account for viewing angle, occlusion, or
scene-level saliency — those require per-frame information unavailable at
asset-build time. The resulting LOD tiers are a reasonable first-pass budget
that downstream streaming logic can refine with view-dependent culling.

## Nested LOD guarantee

`generate_lod_indices` selects all tiers from the **same global importance
ranking**, so top-*k* ⊂ top-*K* for *k* < *K*. This is the property required
by progressive streaming: a client that already holds the lower tier never
needs to re-download the same splats when upgrading to a higher tier.

## Pipeline

```
opacity (N,)  +  log_scales (N, 3)
   → splat_importance(...)       importance.py  (N,) float64
   → select_lod_indices(...)     lod.py         top-k index array
   → generate_lod_indices(...)   lod.py         one array per tier
   → auto_target / tier_target   budgets.py     platform/tier caps
   → build_lod_descriptor(...)   descriptor.py  astel.lod/v0 JSON manifest
```

## Module summary

| Module | Public API |
|--------|-----------|
| `importance.py` | `splat_importance` |
| `lod.py` | `select_lod_indices`, `generate_lod_indices` |
| `budgets.py` | `TIER_BUDGETS`, `PLATFORM_BUDGETS`, `auto_target`, `tier_target` |
| `descriptor.py` | `build_lod_descriptor`, `write_descriptor`, `read_descriptor` |
