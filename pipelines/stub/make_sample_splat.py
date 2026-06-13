"""Procedural sample Gaussian-splat generator for Astel.

Emits a standard INRIA-layout 3D Gaussian Splatting ``.ply`` of a torus knot,
shaded with a brass -> teal gradient and varied per-splat scales for a soft,
surface-like look. This is the hardcoded preview asset the M1 web viewer loads;
it stands in for the L3 "Refined Surface Gaussians" layer until the real
pipeline exists.

The output is a *binary little-endian* PLY with the exact field layout the
3DGS ecosystem (and our Spark-based web viewer) expects::

    x y z
    f_dc_0 f_dc_1 f_dc_2          # SH band-0 (DC) colour, NOT 0..1 RGB
    opacity                       # logit; sigmoid() -> alpha at render
    scale_0 scale_1 scale_2       # log-scale; exp() -> world-space sigma
    rot_0 rot_1 rot_2 rot_3       # quaternion (w, x, y, z), normalised

``f_rest_*`` (higher-order SH) are intentionally omitted: band-0 only is valid
and keeps the file small. Readers that expect them default to zero.

Run directly to (re)generate the checked-in sample::

    uv run python pipelines/stub/make_sample_splat.py

Determinism: a fixed RNG seed means the bytes are reproducible, so the golden
structure test can assert on the header and counts.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# SH band-0 (DC) basis constant. albedo = 0.5 + C0 * f_dc, so f_dc = (albedo-0.5)/C0.
SH_C0: float = 0.28209479177387814

# Default generation knobs. ~48k gaussians lands comfortably in the
# "tasteful, loads instantly on the web" range the brief asks for.
DEFAULT_COUNT: int = 48_000
DEFAULT_SEED: int = 20260613

# The PLY property order is load-bearing: many readers (incl. ours) key on it.
PLY_PROPERTIES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


@dataclass(frozen=True)
class SplatCloud:
    """A bundle of per-splat attributes ready to serialise to PLY.

    All arrays share the same leading dimension ``N`` (the splat count) and use
    the raw 3DGS parameterisation (log-scale, opacity logit, DC colour), not
    display-space values.
    """

    positions: NDArray[np.float32]  # (N, 3) world-space xyz
    colors_dc: NDArray[np.float32]  # (N, 3) SH band-0 DC coefficients
    opacity: NDArray[np.float32]  # (N,)   logit
    log_scales: NDArray[np.float32]  # (N, 3) log of world-space sigma
    quats: NDArray[np.float32]  # (N, 4) (w, x, y, z) normalised

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.positions.shape != (n, 3):
            raise ValueError("positions must be (N, 3)")
        if self.colors_dc.shape != (n, 3):
            raise ValueError("colors_dc must be (N, 3)")
        if self.opacity.shape != (n,):
            raise ValueError("opacity must be (N,)")
        if self.log_scales.shape != (n, 3):
            raise ValueError("log_scales must be (N, 3)")
        if self.quats.shape != (n, 4):
            raise ValueError("quats must be (N, 4)")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


def _rgb_to_dc(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear RGB in [0, 1] to SH band-0 DC coefficients."""
    return ((rgb - 0.5) / SH_C0).astype(np.float32)


def _lerp(
    a: NDArray[np.float32], b: NDArray[np.float32], t: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Per-row linear interpolation; ``t`` broadcast over the last axis."""
    out = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
    result: NDArray[np.float32] = np.asarray(out, dtype=np.float32)
    return result


def _normalize(v: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (v / norm).astype(np.float32)


def _quat_from_two_vectors(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Quaternion (w, x, y, z) rotating unit vector ``a`` onto unit ``b``, per row."""
    a = _normalize(a)
    b = _normalize(b)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    axis = np.cross(a, b)
    axis_len = np.linalg.norm(axis, axis=1, keepdims=True)

    # Stable half-angle form: w = 1 + dot, xyz = axis; then normalise.
    w = (1.0 + dot)[:, None]
    quat = np.concatenate([w, axis], axis=1).astype(np.float32)

    # Antiparallel (dot ~ -1): pick an arbitrary orthogonal axis.
    flipped = (axis_len[:, 0] < 1e-6) & (dot < 0.0)
    if np.any(flipped):
        quat[flipped] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return _normalize(quat)


def build_torus_knot(
    count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED
) -> SplatCloud:
    """Build a (p=2, q=3) torus-knot splat cloud with a brass -> teal gradient.

    The knot is a 1-D curve in 3-space; we scatter gaussians in a soft tube
    around it. Each gaussian is flattened along the tube's radial direction and
    oriented to the curve tangent, so the surface reads as a continuous, softly
    lit ribbon rather than a bag of round blobs.
    """
    rng = np.random.default_rng(seed)

    p, q = 2, 3  # knot winding
    tube_radius = 0.42
    knot_scale = 1.35

    # Parameter along the closed curve, with mild jitter so splats don't band.
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

    # Analytic tangent of the curve (un-normalised is fine; we normalise later).
    dq = q
    dp = p
    tangent = knot_scale * np.stack(
        [
            -dq * np.sin(q * t) * np.cos(p * t)
            - (2.0 + cos_qt) * dp * np.sin(p * t),
            -dq * np.sin(q * t) * np.sin(p * t)
            + (2.0 + cos_qt) * dp * np.cos(p * t),
            dq * np.cos(q * t),
        ],
        axis=1,
    ).astype(np.float32)
    tangent = _normalize(tangent)

    # Build a stable frame (normal, binormal) around the tangent.
    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
    normal = _normalize(np.cross(tangent, up))
    # Where tangent ~ up, the cross product degenerates; patch those rows.
    degenerate = np.linalg.norm(np.cross(tangent, up), axis=1) < 1e-4
    if np.any(degenerate):
        alt = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (count, 1))
        normal[degenerate] = _normalize(np.cross(tangent[degenerate], alt[degenerate]))
    binormal = _normalize(np.cross(tangent, normal))

    # Scatter within the tube cross-section. Bias toward the surface (sqrt) so
    # the shell looks dense and the core stays soft.
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
    radial = (np.sqrt(rng.uniform(0.0, 1.0, size=count)) * tube_radius).astype(
        np.float32
    )
    offset = (
        np.cos(angle)[:, None] * normal + np.sin(angle)[:, None] * binormal
    ) * radial[:, None]
    positions = (centre + offset).astype(np.float32)

    # ---- Colour: brass (instrument accent) -> teal (measured) along t ----
    brass = np.array([0.82, 0.55, 0.16], dtype=np.float32)
    teal = np.array([0.16, 0.62, 0.61], dtype=np.float32)
    grad = (0.5 + 0.5 * np.sin(q * t)).astype(np.float32)  # smooth 0..1 sweep
    rgb = _lerp(brass, teal, grad)
    # Subtle radial darkening toward the tube core for soft shading depth.
    shade = (0.78 + 0.22 * (radial / tube_radius)).astype(np.float32)
    rgb = np.clip(rgb * shade[:, None], 0.0, 1.0).astype(np.float32)
    # A touch of per-splat colour noise breaks up banding.
    rgb = np.clip(
        rgb + rng.normal(0.0, 0.02, size=rgb.shape).astype(np.float32), 0.0, 1.0
    )
    colors_dc = _rgb_to_dc(rgb)

    # ---- Opacity: mostly opaque, softened near the surface ----
    alpha = (0.92 - 0.25 * (radial / tube_radius)).astype(np.float32)
    alpha = np.clip(alpha, 0.05, 0.995)
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)  # inverse sigmoid

    # ---- Scale: flattened, surfel-like gaussians for a soft surface look ----
    # Anisotropic: longer along the tube, thin across it. Mild jitter per splat.
    base_long = 0.05 * (1.0 + 0.35 * rng.standard_normal(count).astype(np.float32))
    base_thin = 0.022 * (1.0 + 0.30 * rng.standard_normal(count).astype(np.float32))
    sig_long = np.clip(np.abs(base_long), 0.012, 0.12)
    sig_thin = np.clip(np.abs(base_thin), 0.006, 0.06)
    sigmas = np.stack([sig_long, sig_thin, sig_thin], axis=1).astype(np.float32)
    log_scales = np.log(sigmas).astype(np.float32)

    # ---- Orientation: align each splat's long axis to the curve tangent ----
    x_axis = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    quats = _quat_from_two_vectors(x_axis, tangent)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=quats,
    )


def _ply_header(count: int) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    lines += [f"property float {name}" for name in PLY_PROPERTIES]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def cloud_to_ply_bytes(cloud: SplatCloud) -> bytes:
    """Serialise a :class:`SplatCloud` to binary-little-endian PLY bytes."""
    interleaved = np.concatenate(
        [
            cloud.positions,
            cloud.colors_dc,
            cloud.opacity[:, None],
            cloud.log_scales,
            cloud.quats,
        ],
        axis=1,
    ).astype("<f4")  # explicit little-endian float32

    if interleaved.shape[1] != len(PLY_PROPERTIES):
        raise AssertionError(
            f"column count {interleaved.shape[1]} != {len(PLY_PROPERTIES)} properties"
        )

    header = _ply_header(cloud.count)
    return header + interleaved.tobytes(order="C")


def write_ply(cloud: SplatCloud, path: Path) -> int:
    """Write ``cloud`` to ``path`` as binary PLY. Returns bytes written."""
    data = cloud_to_ply_bytes(cloud)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def default_output_path() -> Path:
    """Repo-relative checked-in sample location: apps/web/public/samples/."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "apps" / "web" / "public" / "samples" / "astel-sample.ply"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Astel sample splat PLY.")
    parser.add_argument(
        "--out",
        type=Path,
        default=default_output_path(),
        help="Output .ply path (default: apps/web/public/samples/astel-sample.ply).",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cloud = build_torus_knot(count=args.count, seed=args.seed)
    size = write_ply(cloud, args.out)

    # struct import kept meaningful: sanity-check header parses as expected.
    magic = struct.unpack("<3s", b"ply")[0]
    assert magic == b"ply"

    print(
        f"Wrote {cloud.count:,} gaussians -> {args.out} "
        f"({size / 1_048_576:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
