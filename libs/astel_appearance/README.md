# astel-appearance — L4 appearance / lighting (CLAUDE.md §3 L4)

Per-splat PBR material + **separated illumination**, so Astel assets relight
instead of shipping lighting baked into colour as the only option (Meshy's
historical sin). Torch-free, numpy-only — a CPU-testable seam, like
`astel-solid` (L5).

## What it does

- **`sh`** — real spherical harmonics (band 0–2), Lambertian irradiance
  (Ramamoorthi–Hanrahan), and a least-squares SH environment fit.
- **`brdf`** — Cook–Torrance / GGX microfacet model (the PBR-approximation
  forward shader for engines that consume coloured splats + a specular term for
  the Relight Studio).
- **`env`** — `EnvironmentSH` (9 RGB SH radiance coeffs) + studio presets.
- **`decompose`** — the L4 estimator: from baked L3 colours + surfel normals,
  split each splat into **albedo + an estimated SH environment** such that
  relighting under the estimated env reproduces the captured look exactly, and
  swapping the env relights the asset.
- **`webdata`** — a downsampled `{position, normal, albedo}` payload the
  Relight Studio re-shades live in the browser.

## Honesty (CLAUDE.md §1.3, §10.4)

A single baked observation **cannot** fully disambiguate albedo from
illumination (the intrinsic-image ambiguity). This estimator is explicit about
its limits:

- it attributes only the **low-frequency, normal-correlated** part of luminance
  to lighting (SH-L2 is band-limited); high-frequency / chromatic detail stays
  in albedo;
- the illumination estimate is **achromatic** (grayscale); coloured-light and
  multi-view inverse rendering are future work;
- **metallic / roughness are priors, not recovered** (a diffuse-baked
  observation has no reliable specular signal): metallic = 0 (dielectric),
  roughness = 0.6, both flagged;
- `lighting_confidence` reports the opacity-weighted fraction of luminance
  variance the SH-L2 lighting model explains — low values mean the look is
  mostly flat/albedo and the env estimate is weak.

The **relight round-trip invariant** (`relight(albedo, estimated_env) ==
observed`) is the structural guarantee and is enforced in tests.
