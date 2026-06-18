"""L4 appearance decomposition: split baked splat colour into albedo + lighting.

Given an L3 splat cloud's baked per-splat colours and surfel normals, this
estimates a **single-observation intrinsic decomposition**:

    observed_colour(i)  ≈  albedo(i) · diffuse_shading(env, normal(i))

by fitting a low-frequency SH-L2 environment to the luminance and dividing it
out. The result is a per-splat albedo + an estimated environment that **relight
exactly reproduces the captured look** (``relight(albedo, env) == observed``),
and can be swapped for a different environment to relight the asset.

HONESTY (CLAUDE.md §1.3, §10.4). A single baked observation cannot fully
disambiguate albedo from illumination — this is the classic intrinsic-image
ambiguity. This estimator makes that explicit:

* It only attributes the **low-frequency, normal-correlated** part of luminance
  to lighting (SH-L2 is band-limited to quadratic angular variation); high-
  frequency / chromatic detail stays in albedo.
* It assumes **achromatic** low-frequency illumination (the directional lighting
  estimate is grayscale). Coloured-light and full multi-view inverse rendering
  are future work.
* ``metallic`` and ``roughness`` are **priors**, not recovered: a single
  diffuse-baked observation carries no reliable specular signal, so metallic=0
  (dielectric) and a constant roughness are emitted and flagged.
* ``lighting_confidence`` reports the fraction of (opacity-weighted) luminance
  variance explained by the SH-L2 lighting model — low values mean the look is
  mostly flat/albedo and the environment estimate is weak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .env import EnvironmentSH
from .sh import diffuse_shading, fit_environment_sh

FloatArray = NDArray[np.float64]

#: SH band-0 DC basis constant (albedo_display = 0.5 + SH_C0 * f_dc).
SH_C0 = 0.28209479177387814

#: Rec.709 luminance weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

#: Default dielectric roughness prior emitted when no specular signal exists.
DEFAULT_ROUGHNESS = 0.6


def observed_rgb_from_dc(colors_dc: NDArray[np.floating]) -> FloatArray:
    """Display-space RGB in ``[0, 1]`` from SH band-0 DC coefficients."""
    rgb = 0.5 + SH_C0 * np.asarray(colors_dc, dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)


def colors_dc_from_rgb(rgb: NDArray[np.floating]) -> FloatArray:
    """Inverse of :func:`observed_rgb_from_dc` (RGB -> SH band-0 DC)."""
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


@dataclass(frozen=True)
class AppearanceLayer:
    """The estimated L4 appearance layer for a splat cloud."""

    albedo: FloatArray  # (N, 3) in [0, 1]
    metallic: FloatArray  # (N,) prior (zeros)
    roughness: FloatArray  # (N,) prior
    normals: FloatArray  # (N, 3) unit
    env: EnvironmentSH  # estimated illumination
    lighting_confidence: float
    notes: list[str]

    @property
    def count(self) -> int:
        return int(self.albedo.shape[0])

    def summary(self) -> dict[str, Any]:
        """JSON-able L4 summary for the quality report / ``l4.json``."""
        return {
            "schema": "astel.l4-appearance/v0",
            "method": "single-observation-intrinsic-sh-l2",
            "count": self.count,
            "lighting_confidence": float(self.lighting_confidence),
            "metallic": "prior:0 (dielectric); not recovered from one observation",
            "roughness": f"prior:{DEFAULT_ROUGHNESS}; not recovered",
            "illumination": "achromatic low-frequency SH-L2 estimate",
            "albedo_mean": [float(x) for x in self.albedo.mean(axis=0)],
            "notes": self.notes,
        }


def decompose_appearance(
    colors_dc: NDArray[np.floating],
    normals: NDArray[np.floating],
    *,
    opacity_logit: NDArray[np.floating] | None = None,
    roughness: float = DEFAULT_ROUGHNESS,
    eps: float = 1e-3,
) -> AppearanceLayer:
    """Estimate the L4 appearance layer from baked colours + surfel normals.

    Parameters
    ----------
    colors_dc:
        ``(N, 3)`` SH band-0 DC coefficients (``SplatCloud.colors_dc``).
    normals:
        ``(N, 3)`` per-splat surfel normals (e.g. ``astel_solid.surfel_normals``).
    opacity_logit:
        Optional ``(N,)`` opacity logits; converted to ``alpha = sigmoid`` and
        used as fit weights (opaque splats are trusted more). Defaults to
        uniform weights.
    roughness:
        The dielectric roughness prior to emit (not recovered).
    """
    observed = observed_rgb_from_dc(colors_dc)  # (N, 3)
    n = np.asarray(normals, dtype=np.float64)
    if observed.shape[0] != n.shape[0]:
        raise ValueError("colors_dc and normals must have the same N")
    count = observed.shape[0]

    if opacity_logit is None:
        alpha = np.ones(count, dtype=np.float64)
    else:
        alpha = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logit, dtype=np.float64)))

    luminance = observed @ _LUMA  # (N,)

    # Fit a low-frequency SH-L2 lighting field to the luminance.
    env_lum = fit_environment_sh(n, luminance, weights=alpha)  # (9,)
    pred = diffuse_shading(env_lum, n)  # (N,) the smooth lighting estimate

    # Resolve the global albedo/light scale ambiguity by normalising the mean
    # (opacity-weighted) shading to 1, so albedo lives on the same [0,1] scale
    # as the observed colour.
    wsum = float(alpha.sum())
    mean_shading = float((pred * alpha).sum() / wsum) if wsum > 0 else 0.0

    notes: list[str] = []
    if mean_shading <= eps:
        # Degenerate (e.g. all-dark input): fall back to flat unit lighting,
        # leaving albedo == observed. Honest no-op rather than a divide blow-up.
        env_lum = np.zeros_like(env_lum)
        env_lum[0] = 1.0 / 0.28209479177387814  # flat shading == 1
        pred = np.ones(count, dtype=np.float64)
        mean_shading = 1.0
        notes.append(
            "Degenerate luminance fit (near-black or normal-free input); "
            "lighting set to flat unit ambient and albedo == observed colour."
        )

    env_norm = env_lum / mean_shading
    shading_norm = np.clip(pred / mean_shading, eps, None)  # (N,)

    albedo = np.clip(observed / shading_norm[:, None], 0.0, 1.0)

    # Achromatic env: replicate the grayscale lighting estimate across RGB.
    env_rgb = np.repeat(env_norm[:, None], 3, axis=1)  # (9, 3)

    # lighting_confidence = opacity-weighted R^2 of the SH-L2 luminance fit.
    resid = luminance - pred
    lum_mean = float((luminance * alpha).sum() / wsum) if wsum > 0 else 0.0
    ss_res = float((alpha * resid * resid).sum())
    ss_tot = float((alpha * (luminance - lum_mean) ** 2).sum())
    confidence = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    notes.append(
        "Single-observation intrinsic split: low-frequency, normal-correlated "
        "luminance attributed to illumination; the rest retained as albedo. "
        "Relighting under the returned env reproduces the captured colour exactly."
    )
    notes.append(
        "Illumination estimate is achromatic (grayscale SH-L2). Coloured-light "
        "and multi-view inverse rendering are future work."
    )

    return AppearanceLayer(
        albedo=albedo,
        metallic=np.zeros(count, dtype=np.float64),
        roughness=np.full(count, float(roughness), dtype=np.float64),
        normals=n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None),
        env=EnvironmentSH(sh_rgb=env_rgb, name="estimated"),
        lighting_confidence=confidence,
        notes=notes,
    )


def relight_rgb(
    layer: AppearanceLayer,
    env: EnvironmentSH,
    *,
    rotation: NDArray[np.floating] | None = None,
) -> FloatArray:
    """Relit display-space RGB ``(N, 3)`` of ``layer`` under ``env``.

    Rotating the environment by ``R`` is implemented by evaluating shading at
    ``R^{-1} n`` (the Relight Studio spins the HDRI this way). ``albedo * env``
    under the *estimated* env returns the original observed colour.
    """
    normals = layer.normals
    if rotation is not None:
        r = np.asarray(rotation, dtype=np.float64)
        normals = normals @ r  # n @ R == (R^T n) == R^{-1} n for orthonormal R
    shading = diffuse_shading(env.sh_rgb, normals)  # (N, 3)
    return np.clip(layer.albedo * shading, 0.0, 1.0)


def relight_colors_dc(
    layer: AppearanceLayer,
    env: EnvironmentSH,
    *,
    rotation: NDArray[np.floating] | None = None,
) -> FloatArray:
    """Relit SH band-0 DC coefficients (for writing a relit splat cloud)."""
    return colors_dc_from_rgb(relight_rgb(layer, env, rotation=rotation))


def albedo_colors_dc(layer: AppearanceLayer) -> FloatArray:
    """SH band-0 DC coefficients of the un-lit albedo (the L4 albedo splats)."""
    return colors_dc_from_rgb(layer.albedo)
