"""astel-appearance — L4 appearance/lighting layer (CLAUDE.md §3 L4).

Per-splat PBR material + separated illumination. The product never ships
lighting baked into colour as the *only* option (Meshy's historical sin): this
library estimates a per-splat albedo + an SH environment from the baked L3
colours, so assets relight. It also provides a Cook-Torrance forward model for
the PBR-approximation export and a downsampled payload for the Relight Studio.

Torch-free, numpy-only — a CPU-testable seam, like ``astel_solid`` (L5).
"""

from __future__ import annotations

from .brdf import (
    DIELECTRIC_F0,
    cook_torrance,
    fresnel_schlick,
    ggx_ndf,
    smith_g,
)
from .decompose import (
    DEFAULT_ROUGHNESS,
    SH_C0,
    AppearanceLayer,
    albedo_colors_dc,
    colors_dc_from_rgb,
    decompose_appearance,
    observed_rgb_from_dc,
    relight_colors_dc,
    relight_rgb,
)
from .env import EnvironmentSH, directional_env, studio_presets
from .normals import surfel_normals
from .produce import AppearanceArtifacts, build_appearance
from .sh import (
    COSINE_CONV,
    N_SH_L2,
    diffuse_shading,
    fit_environment_sh,
    sh_eval_l2,
    yaw_rotation,
)
from .webdata import relight_payload

__all__ = [
    "COSINE_CONV",
    "DEFAULT_ROUGHNESS",
    "DIELECTRIC_F0",
    "N_SH_L2",
    "SH_C0",
    "AppearanceArtifacts",
    "AppearanceLayer",
    "EnvironmentSH",
    "albedo_colors_dc",
    "build_appearance",
    "colors_dc_from_rgb",
    "cook_torrance",
    "decompose_appearance",
    "diffuse_shading",
    "directional_env",
    "fit_environment_sh",
    "fresnel_schlick",
    "ggx_ndf",
    "observed_rgb_from_dc",
    "relight_colors_dc",
    "relight_payload",
    "relight_rgb",
    "sh_eval_l2",
    "smith_g",
    "studio_presets",
    "surfel_normals",
    "yaw_rotation",
]
