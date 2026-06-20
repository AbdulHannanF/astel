"""astel-dynamics — L7 deformation timeline for Gaussian splat assets.

Compact LBS (Linear-Blend Skinning) deformation field over a static splat cloud:
K control nodes, per-gaussian blend weights, per-frame affine node transforms.

Validates format and math on CPU (torch-free / numpy+scipy only).  Fit errors
are the REAL measured reconstruction errors — never fabricated (CLAUDE.md §10.4).
"""

from __future__ import annotations

from .baked import bake_per_frame
from .field import DeformationField
from .fit import FitReport, fit_deformation_field
from .pack import read_deformation_bin, write_deformation_bin
from .timeline import Timeline, read_timeline_json, write_timeline_json

__all__ = [
    "DeformationField",
    "FitReport",
    "Timeline",
    "bake_per_frame",
    "fit_deformation_field",
    "read_deformation_bin",
    "read_timeline_json",
    "write_deformation_bin",
    "write_timeline_json",
]
