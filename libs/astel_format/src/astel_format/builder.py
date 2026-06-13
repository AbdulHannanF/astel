"""Assemble a minimal, valid `.astel` package from L3 splat geometry.

Builds an :class:`~astel_format.package.AstelPackage` containing an L3
(refined gaussians) layer -- and optionally an L0 (seed point cloud) layer
-- each with a per-primitive provenance buffer (UNORM8, manifest-v0.md
section 5.2), plus the mandatory ``buffers``/``provenance``/
``quality_report`` blocks. This is the smallest package that satisfies
``manifest.schema.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from astel_format.models import (
    Accessor,
    AssetIdentity,
    BufferEntry,
    BufferTable,
    BufferView,
    CoordAxis,
    CoordinateSystem,
    FileRef,
    Generator,
    LayerEntry,
    Layers,
    Manifest,
    ProvenanceChannel,
    ProvenanceDescriptor,
    QualityReport,
    Scale,
    ScaleConfidenceInterval,
    ScaleMethod,
)
from astel_format.package import AstelPackage

_L3_PROVENANCE_PATH = "layers/l3_refined/provenance.bin"
_L3_SPLATS_PATH = "layers/l3_refined/splats.ply"
_L0_PROVENANCE_PATH = "layers/l0_seed/provenance.bin"
_L0_POINTS_PATH = "layers/l0_seed/points.ply"


def _encode_provenance_u8(values: Sequence[float]) -> bytes:
    """Encode per-primitive provenance floats in ``[0, 1]`` as UNORM8.

    ``q = round(p * 255)``, tightly packed (manifest-v0.md section 5.2).
    """
    out = bytearray(len(values))
    for i, p in enumerate(values):
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"provenance value out of [0, 1] range: {p!r} at index {i}"
            )
        out[i] = round(p * 255)
    return bytes(out)


def _provenance_accessor_and_buffer(
    *,
    buffer_index: int,
    buffer_view_index: int,
    count: int,
) -> tuple[BufferView, Accessor]:
    view = BufferView(buffer=buffer_index, byte_offset=0, byte_length=count)
    accessor = Accessor(
        buffer_view=buffer_view_index,
        component_type="UNORM8",
        type="SCALAR",
        count=count,
        normalized=True,
    )
    return view, accessor


def build_minimal_package(
    *,
    asset_id: str,
    created: str,
    generator_name: str,
    generator_version: str,
    source_modality: str,
    l3_ply_path: str | Path,
    l3_count: int,
    l3_provenance: Sequence[float],
    quality_report: QualityReport,
    l0_ply_path: str | Path | None = None,
    l0_count: int | None = None,
    l0_provenance: Sequence[float] | None = None,
    meters_per_unit: float = 1.0,
    handedness: str = "right",
    up_axis: CoordAxis = "+Y",
    forward_axis: CoordAxis = "-Z",
    scale_method: ScaleMethod = "user",
    scale_ci_low: float = 1.0,
    scale_ci_high: float = 1.0,
    asset_name: str | None = None,
    prompt: str | None = None,
    seed: int | None = None,
) -> AstelPackage:
    """Build a minimal valid :class:`AstelPackage` from L3 (+ optional L0).

    Parameters
    ----------
    l3_ply_path:
        Path to the L3 ``.ply`` master (read and embedded as
        ``layers/l3_refined/splats.ply``).
    l3_count:
        Gaussian count for L3; must equal ``len(l3_provenance)`` and the
        ``count`` of the provenance accessor (manifest-v0.md section 4.3 --
        per-gaussian accessors are index-aligned to the bound layer).
    l3_provenance:
        Per-gaussian provenance values in ``[0, 1]``, one per L3 gaussian,
        index-aligned (section 5).
    quality_report:
        A fully-formed :class:`QualityReport`. The honesty contract (section
        6) requires unmeasured numeric fields to be explicit ``None`` with a
        ``reason`` -- this builder does not fabricate values.
    l0_ply_path, l0_count, l0_provenance:
        Optional L0 seed point cloud + its provenance buffer. All three must
        be provided together or not at all.
    """
    if len(l3_provenance) != l3_count:
        raise ValueError(
            f"l3_provenance length ({len(l3_provenance)}) != l3_count ({l3_count})"
        )

    have_l0 = (
        l0_ply_path is not None or l0_count is not None or l0_provenance is not None
    )
    if have_l0:
        if l0_ply_path is None or l0_count is None or l0_provenance is None:
            raise ValueError(
                "l0_ply_path, l0_count, and l0_provenance must all be provided together"
            )
        if len(l0_provenance) != l0_count:
            raise ValueError(
                f"l0_provenance length ({len(l0_provenance)}) != l0_count ({l0_count})"
            )

    files: dict[str, bytes] = {}

    l3_ply_path = Path(l3_ply_path)
    files[_L3_SPLATS_PATH] = l3_ply_path.read_bytes()
    files[_L3_PROVENANCE_PATH] = _encode_provenance_u8(l3_provenance)

    buffers: list[BufferEntry] = []
    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []
    provenance_channels: list[ProvenanceChannel] = []

    # L3 provenance buffer -> buffer 0 / view 0 / accessor 0.
    buffers.append(
        BufferEntry(
            uri=_L3_PROVENANCE_PATH, byte_length=len(files[_L3_PROVENANCE_PATH])
        )
    )
    view, accessor = _provenance_accessor_and_buffer(
        buffer_index=0, buffer_view_index=0, count=l3_count
    )
    buffer_views.append(view)
    accessors.append(accessor)
    provenance_channels.append(
        ProvenanceChannel(layer="l3", accessor=0, count=l3_count)
    )

    l3_layer = LayerEntry(
        kind="refined_gaussians",
        status="present",
        files=[FileRef(path=_L3_SPLATS_PATH, role="master", format="ply")],
        count=l3_count,
        provenance_ref=0,
    )

    l0_layer: LayerEntry | None = None
    if have_l0:
        assert l0_ply_path is not None
        assert l0_count is not None
        assert l0_provenance is not None
        l0_ply_path = Path(l0_ply_path)
        files[_L0_POINTS_PATH] = l0_ply_path.read_bytes()
        files[_L0_PROVENANCE_PATH] = _encode_provenance_u8(l0_provenance)

        buf_idx = len(buffers)
        view_idx = len(buffer_views)
        buffers.append(
            BufferEntry(
                uri=_L0_PROVENANCE_PATH, byte_length=len(files[_L0_PROVENANCE_PATH])
            )
        )
        view, accessor = _provenance_accessor_and_buffer(
            buffer_index=buf_idx, buffer_view_index=view_idx, count=l0_count
        )
        buffer_views.append(view)
        accessor_idx = len(accessors)
        accessors.append(accessor)
        provenance_channels.append(
            ProvenanceChannel(layer="l0", accessor=accessor_idx, count=l0_count)
        )

        l0_layer = LayerEntry(
            kind="seed_pointcloud",
            status="present",
            files=[FileRef(path=_L0_POINTS_PATH, role="master", format="ply")],
            count=l0_count,
            provenance_ref=accessor_idx,
        )

    identity_kwargs: dict[str, object] = {
        "id": asset_id,
        "created": created,
        "generator": Generator(name=generator_name, version=generator_version),
        "source_modality": source_modality,
    }
    if asset_name is not None:
        identity_kwargs["name"] = asset_name
    if prompt is not None:
        identity_kwargs["prompt"] = prompt
    if seed is not None:
        identity_kwargs["seed"] = seed

    layers_kwargs: dict[str, object] = {"l3": l3_layer}
    if l0_layer is not None:
        layers_kwargs["l0"] = l0_layer

    manifest = Manifest(
        format_version="0.1.0",
        astel=AssetIdentity.model_validate(identity_kwargs),
        coordinate_system=CoordinateSystem(
            handedness=handedness,  # type: ignore[arg-type]
            up_axis=up_axis,
            forward_axis=forward_axis,
            meters_per_unit=meters_per_unit,
        ),
        scale=Scale(
            meters_per_unit=meters_per_unit,
            confidence=ScaleConfidenceInterval(
                ci_low=scale_ci_low, ci_high=scale_ci_high
            ),
            method=scale_method,
        ),
        layers=Layers.model_validate(layers_kwargs),
        buffers=BufferTable(
            buffers=buffers, buffer_views=buffer_views, accessors=accessors
        ),
        provenance=ProvenanceDescriptor(
            semantic="measured_vs_generated",
            range=[0.0, 1.0],
            convention="1=measured, 0=generated",
            precision="u8",
            channels=provenance_channels,
        ),
        quality_report=quality_report,
    )

    return AstelPackage(manifest=manifest, files=files)
