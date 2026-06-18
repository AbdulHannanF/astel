"""Procedural per-task artifact producer (CPU, no GPU, no network).

This is an explicit STUB standing in for the future GPU reconstruction
pipeline (M2+). It writes a real, per-task layer stack into the
:class:`~astel_api.storage.ArtifactStore`:

* ``l3.ply`` — the hero refined-gaussian cloud (INRIA-layout 3DGS, via
  ``astel_splat_io``).
* ``l0.ply`` — a sparse "seed" point cloud (a deterministic subsample of L3),
  the cheap preview tier (CLAUDE.md §3 L0).
* ``l3.spz`` / ``l3.sog`` — compressed-delivery exports of the L3 cloud
  (Niantic SPZ v3 and PlayCanvas SOG, both via ``astel_splat_io``).
* ``l3.glb`` — KHR_gaussian_splatting glTF (RC) export of the L3 cloud, the
  broadly-interoperable interop artifact (via ``astel_splat_io.gltf``).
* ``package.astel`` — a real, schema-valid ``.astel`` package assembled by
  ``astel_format.build_minimal_package`` binding L0 + L3 with a per-gaussian
  provenance channel and an honest :class:`QualityReport`.
* ``quality-report.json`` — the ``astel.quality-report/v0`` dict the web Truth
  Meter consumes (a *different* report shape from the package's
  ``QualityReport`` — two consumers, two schemas, both built honestly).

Every report is explicitly marked ``origin: "stub"`` with caveats — never
silent hallucination over real data (CLAUDE.md §10.4). All numeric quality
fields are explicit placeholders (the dict) or explicit ``None`` + a ``reason``
(the package's typed report); nothing is fabricated as "measured".
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np
from astel_format.builder import build_minimal_package
from astel_format.models import (
    GeometricError,
    HallucinationReport,
    QualityReport,
    ScaleConfidence,
)
from astel_splat_io.cloud import SplatCloud
from astel_splat_io.gltf import write_gltf
from astel_splat_io.ply import write_ply
from astel_splat_io.sog import write_sog
from astel_splat_io.spz import write_spz

from . import __version__
from .storage import ArtifactStore

logger = logging.getLogger(__name__)

DEFAULT_COUNT: int = 48_000

# L0 seed cloud is a sparse preview: a fixed-fraction subsample of L3. The stub
# has no real SfM/conditioning stage, so "seed" == a deterministically strided
# slice of the refined cloud (documented as such in the quality report caveats).
L0_SUBSAMPLE_DIVISOR: int = 24

# SH band-0 (DC) basis constant. albedo = 0.5 + C0 * f_dc, so f_dc = (albedo-0.5)/C0.
_SH_C0: float = 0.28209479177387814


def stable_seed(task_id: str) -> int:
    """Deterministically derive a small positive int seed from ``task_id``."""
    digest = blake2b(task_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def _normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (v / norm).astype(np.float32)


def synth_cloud(seed: int, count: int = DEFAULT_COUNT) -> SplatCloud:
    """Build a procedural torus-knot-like ribbon :class:`SplatCloud`.

    Seeded so each task produces a visibly different cloud while remaining
    deterministic for a given ``seed``. Stands in for L3 "Refined Surface
    Gaussians" until the real reconstruction pipeline lands.
    """
    rng = np.random.default_rng(seed)

    # Vary the knot's winding numbers per-seed for visible per-task variation.
    p = 2 + int(seed % 3)
    q = 3 + int((seed // 3) % 4)
    tube_radius = 0.42
    knot_scale = 1.35

    t = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)

    cos_qt = np.cos(q * t)
    centre = knot_scale * np.stack(
        [
            (2.0 + cos_qt) * np.cos(p * t),
            (2.0 + cos_qt) * np.sin(p * t),
            np.sin(q * t),
        ],
        axis=1,
    ).astype(np.float32)

    tangent = knot_scale * np.stack(
        [
            -q * np.sin(q * t) * np.cos(p * t) - (2.0 + cos_qt) * p * np.sin(p * t),
            -q * np.sin(q * t) * np.sin(p * t) + (2.0 + cos_qt) * p * np.cos(p * t),
            q * np.cos(q * t),
        ],
        axis=1,
    ).astype(np.float32)
    tangent = _normalize(tangent)

    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
    cross_tangent_up = np.cross(tangent, up)
    normal = _normalize(cross_tangent_up)
    degenerate = np.linalg.norm(cross_tangent_up, axis=1) < 1e-4
    if np.any(degenerate):
        alt = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (count, 1))
        normal[degenerate] = _normalize(np.cross(tangent[degenerate], alt[degenerate]))
    binormal = _normalize(np.cross(tangent, normal))

    angle = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
    radial = (np.sqrt(rng.uniform(0.0, 1.0, size=count)) * tube_radius).astype(
        np.float32
    )
    offset = (
        np.cos(angle)[:, None] * normal + np.sin(angle)[:, None] * binormal
    ) * radial[:, None]
    positions = (centre + offset).astype(np.float32)

    # ---- Colour: a per-seed hue pair, gradient along t ----
    hue_a = float(rng.uniform(0.0, 1.0))
    hue_b = (hue_a + 0.5) % 1.0

    def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p_ = v * (1.0 - s)
        q_ = v * (1.0 - s * f)
        t_ = v * (1.0 - s * (1.0 - f))
        choices = {
            0: (v, t_, p_),
            1: (q_, v, p_),
            2: (p_, v, t_),
            3: (p_, q_, v),
            4: (t_, p_, v),
            5: (v, p_, q_),
        }
        return np.array(choices[i % 6], dtype=np.float32)

    color_a = _hsv_to_rgb(hue_a, 0.55, 0.85)
    color_b = _hsv_to_rgb(hue_b, 0.55, 0.85)
    grad = (0.5 + 0.5 * np.sin(q * t)).astype(np.float32)
    rgb = color_a[None, :] * (1.0 - grad[:, None]) + color_b[None, :] * grad[:, None]
    shade = (0.78 + 0.22 * (radial / tube_radius)).astype(np.float32)
    rgb = np.clip(rgb * shade[:, None], 0.0, 1.0).astype(np.float32)
    rgb = np.clip(
        rgb + rng.normal(0.0, 0.02, size=rgb.shape).astype(np.float32), 0.0, 1.0
    )
    colors_dc = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    # ---- Opacity: mostly opaque, softened near the surface ----
    alpha = (0.92 - 0.25 * (radial / tube_radius)).astype(np.float32)
    alpha = np.clip(alpha, 0.05, 0.995)
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    # ---- Scale: flattened, surfel-like gaussians ----
    base_long = 0.05 * (1.0 + 0.35 * rng.standard_normal(count).astype(np.float32))
    base_thin = 0.022 * (1.0 + 0.30 * rng.standard_normal(count).astype(np.float32))
    sig_long = np.clip(np.abs(base_long), 0.012, 0.12)
    sig_thin = np.clip(np.abs(base_thin), 0.006, 0.06)
    sigmas = np.stack([sig_long, sig_thin, sig_thin], axis=1).astype(np.float32)
    log_scales = np.log(sigmas).astype(np.float32)

    # ---- Orientation: identity-ish quats with small jitter (w, x, y, z) ----
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    quats[:, 1:] = rng.normal(0.0, 0.05, size=(count, 3)).astype(np.float32)
    quats = _normalize(quats)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=quats,
    )


def seed_cloud(cloud: SplatCloud, divisor: int = L0_SUBSAMPLE_DIVISOR) -> SplatCloud:
    """Derive a sparse L0 "seed" cloud by deterministic strided subsampling.

    The real pipeline's L0 is SfM points / generative latent samples
    (CLAUDE.md §3 L0). The stub has neither, so we take an evenly-strided
    slice of the refined L3 cloud — a faithful "sparse preview of the same
    object" that is fully deterministic for a given task. ``divisor`` is the
    stride; at least one point is always kept.
    """
    n = cloud.count
    step = max(1, divisor)
    order = np.arange(0, n, step, dtype=np.intp)
    return cloud.reordered(order)


def build_quality_report(*, count: int, modality: str) -> dict[str, Any]:
    """Build the honest quality-report dict (schema ``astel.quality-report/v0``).

    All metric values here are illustrative placeholders for the stub
    pipeline. ``origin`` and ``caveats`` are mandatory and must not be removed
    — measured metrics arrive with the GPU reconstruction path (M2).
    """
    return {
        "schema": "astel.quality-report/v0",
        "origin": "stub",
        "modality": modality,
        "splats": count,
        "geometric_error": {"chamfer_mm_vs_l1": 0.9, "method": "stub-placeholder"},
        "fidelity": {
            "psnr_db": 31.2,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": 0,
        },
        "scale": {
            "longest_axis_m": 0.182,
            "confidence": 0.41,
            "method": "estimate",
        },
        "provenance": {"measured_ratio": 0.0, "generated_ratio": 1.0},
        "caveats": [
            "Stub pipeline output: metrics are illustrative placeholders, not "
            "measured. Real geometric error and fidelity arrive with the GPU "
            "reconstruction path (M2).",
            "The geometry shown is a deterministic procedural placeholder seeded "
            "from the task id — it does NOT represent the prompt or any real "
            "object. Prompt/image-conditioned geometry comes from the GPU "
            "generative path (image -> TripoSplat today; text -> multiview is "
            "not built yet).",
        ],
    }


def build_package_quality_report(*, modality: str) -> QualityReport:
    """Build the typed :class:`QualityReport` embedded in the ``.astel`` package.

    This is the manifest-v0 ``quality_report`` shape (distinct from the
    ``astel.quality-report/v0`` dict the web Truth Meter reads). The honesty
    contract (manifest-v0 §6) requires every unmeasured numeric field to be
    explicit ``None`` *with a reason* — so for the stub every metric is ``None``
    and the reason names the stub. Nothing here is fabricated as measured: the
    illustrative placeholder numbers live only in the v0 dict, not in this
    schema-validated report.
    """
    stub_reason = (
        "Stub pipeline: no reconstruction performed, so geometric error is not "
        "measured. Real Chamfer-vs-L1 arrives with the GPU path (M2)."
    )
    return QualityReport(
        geometric_error=GeometricError(
            units="mm",  # explicit so exclude_unset emits the schema-required key
            reference_layer="l0",
            chamfer_mm=None,
            mean_mm=None,
            p95_mm=None,
            method="stub-placeholder",
            reason=stub_reason,
        ),
        # Scale is an honest "unknown": no metric grounding exists, so we report
        # the identity scale (1 unit == 1 metre) with a zero-width CI
        # (ci_low == ci_high == meters_per_unit). The schema forbids
        # non-positive bounds, so a degenerate point estimate at 1.0 is the
        # honest encoding of "ungrounded, treated as unit scale" — the
        # ci_method names it as a non-estimate rather than a real interval.
        scale_confidence=ScaleConfidence(
            meters_per_unit=1.0,
            ci_low=1.0,
            ci_high=1.0,
            ci_method="stub-no-estimate",
        ),
        # The stub cloud is 100% generated procedurally — 0% measured. This is
        # honest, not a placeholder: there is genuinely no measured input.
        hallucination=HallucinationReport(
            measured_fraction=0.0,
            generated_fraction=1.0,
        ),
        origin="stub",
        caveats=[
            f"Stub pipeline; modality={modality}. Procedurally generated, not "
            "reconstructed: the geometry is a prompt-independent placeholder "
            "seeded from the task id, not a model of the prompt. All quality "
            "metrics are unmeasured (null) by design until the GPU "
            "reconstruction path lands (M2).",
        ],
    )


def _write_appearance(
    cloud: SplatCloud,
    tmp_path: Path,
    task_id: str,
    store: ArtifactStore,
    artifacts: list[str],
) -> tuple[Path, Path, Path, dict[str, Any]] | None:
    """Best-effort L4: decompose the stub colour into albedo + an SH env.

    CPU-pure (torch-free) so the no-GPU demo asset relights. Writes/stores
    ``l4-albedo.ply`` (un-lit base colour), ``l4-env.json`` (estimated
    illumination), ``l4.json`` (summary) and ``l4-relight.json`` (web studio
    preview). Returns ``(env_path, albedo_path, summary_path, summary)`` or
    ``None`` — L4 must never fail an asset (the asset stays splats).
    """
    try:
        from astel_appearance import build_appearance

        art = build_appearance(
            cloud.positions,
            cloud.colors_dc,
            cloud.quats,
            cloud.log_scales,
            cloud.opacity,
        )
        albedo_cloud = SplatCloud(
            positions=cloud.positions,
            colors_dc=art.albedo_colors_dc.astype(np.float32),
            opacity=cloud.opacity,
            log_scales=cloud.log_scales,
            quats=cloud.quats,
        )
        albedo_ply = tmp_path / "l4-albedo.ply"
        write_ply(albedo_cloud, albedo_ply)
        store.put(task_id, "l4-albedo.ply", albedo_ply.read_bytes())
        artifacts.append("l4-albedo.ply")

        env_path = tmp_path / "l4-env.json"
        env_path.write_text(json.dumps(art.env, indent=2))
        store.put(task_id, "l4-env.json", env_path.read_bytes())
        artifacts.append("l4-env.json")

        summary_path = tmp_path / "l4.json"
        summary_path.write_text(json.dumps(art.summary, indent=2))
        store.put(task_id, "l4.json", summary_path.read_bytes())
        artifacts.append("l4.json")

        relight_path = tmp_path / "l4-relight.json"
        relight_path.write_text(json.dumps(art.relight_preview))
        store.put(task_id, "l4-relight.json", relight_path.read_bytes())
        artifacts.append("l4-relight.json")

        return env_path, albedo_ply, summary_path, art.summary
    except Exception:
        logger.exception("L4 appearance failed (best-effort); skipping l4 artifacts")
        return None


def produce_artifacts(
    task_id: str, modality: str, prompt: str, store: ArtifactStore
) -> dict[str, Any]:
    """Generate and store the L0/L3/L4 splat layer stack for ``task_id``.

    Writes ``l0.ply``, ``l3.ply``, ``l3.spz``, ``l3.sog``, the L4 appearance set
    (``l4-albedo.ply``, ``l4-env.json``, ``l4.json``, ``l4-relight.json``),
    ``package.astel`` and ``quality-report.json`` into ``store``. Returns
    ``{"splats": <l3 count>, "seed_splats": <l0 count>, "artifacts": [...]}``.
    """
    seed = stable_seed(task_id)
    cloud = synth_cloud(seed)
    l0 = seed_cloud(cloud)

    artifacts: list[str] = []

    appearance_summary: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        l3_ply = tmp_path / "l3.ply"
        write_ply(cloud, l3_ply)
        store.put(task_id, "l3.ply", l3_ply.read_bytes())
        artifacts.append("l3.ply")

        l0_ply = tmp_path / "l0.ply"
        write_ply(l0, l0_ply)
        store.put(task_id, "l0.ply", l0_ply.read_bytes())
        artifacts.append("l0.ply")

        # L4 appearance (CLAUDE.md §3 L4): the decomposition is CPU-pure
        # (torch-free), so even the stub asset relights — split the baked
        # procedural colour into albedo + an estimated SH environment. This is a
        # *real* operation on the stub geometry (not a measured object); the L4
        # summary inherits the stub's honesty caveats via the quality report.
        l4_paths = _write_appearance(cloud, tmp_path, task_id, store, artifacts)
        appearance_summary = l4_paths[3] if l4_paths else None

        # Compressed-delivery exports of the L3 cloud. SPZ is byte-exact to its
        # spec; SOG is best-effort (uniform-quantile codebooks, no spatial sort
        # — documented caveats in astel_splat_io.sog), real and loadable but
        # higher quantization error than reference k-means SOGS.
        l3_spz = tmp_path / "l3.spz"
        write_spz(cloud, l3_spz)
        store.put(task_id, "l3.spz", l3_spz.read_bytes())
        artifacts.append("l3.spz")

        l3_sog = tmp_path / "l3.sog"
        write_sog(cloud, l3_sog)
        store.put(task_id, "l3.sog", l3_sog.read_bytes())
        artifacts.append("l3.sog")

        # KHR_gaussian_splatting glTF (RC) — the broadly-interoperable interop
        # export. Same 3DGS frame as the .ply master (only the quaternion order
        # differs); see astel_splat_io.gltf for the coordinate convention.
        l3_glb = tmp_path / "l3.glb"
        write_gltf(cloud, l3_glb)
        store.put(task_id, "l3.glb", l3_glb.read_bytes())
        artifacts.append("l3.glb")

        l4_env = tmp_path / "l4-env.json" if l4_paths else None
        l4_albedo = tmp_path / "l4-albedo.ply" if l4_paths else None
        l4_summary = tmp_path / "l4.json" if l4_paths else None

        # Full .astel package binding L0 + L3 (+ L4 appearance) with per-gaussian
        # provenance. Every primitive is fully generated (provenance = 0.0 ==
        # "generated", per the manifest convention "1=measured, 0=generated"),
        # matching the honest 0%-measured hallucination report.
        package = build_minimal_package(
            asset_id=task_id,
            created=datetime.now(UTC).isoformat(),
            generator_name="astel-api-stub",
            generator_version=__version__,
            source_modality=modality,
            l3_ply_path=l3_ply,
            l3_count=cloud.count,
            l3_provenance=[0.0] * cloud.count,
            l0_ply_path=l0_ply,
            l0_count=l0.count,
            l0_provenance=[0.0] * l0.count,
            l4_env_path=l4_env,
            l4_albedo_path=l4_albedo,
            l4_summary_path=l4_summary,
            quality_report=build_package_quality_report(modality=modality),
            prompt=prompt or None,
            seed=seed,
        )
        astel_path = tmp_path / "package.astel"
        package.write(astel_path)
        store.put(task_id, "package.astel", astel_path.read_bytes())
        artifacts.append("package.astel")

    report = build_quality_report(count=cloud.count, modality=modality)
    if appearance_summary is not None:
        report["appearance"] = appearance_summary
    store.put(task_id, "quality-report.json", json.dumps(report).encode("utf-8"))
    artifacts.append("quality-report.json")

    return {
        "splats": cloud.count,
        "seed_splats": l0.count,
        "artifacts": artifacts,
    }
