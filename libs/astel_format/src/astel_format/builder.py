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
    LayerAppearance,
    LayerArticulation,
    LayerCollision,
    LayerDynamics,
    LayerEntry,
    LayerIsosurface,
    LayerPhysicsMaterial,
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
    # L4 appearance / lighting layer (optional; emit only when an env or albedo
    # path is provided)
    l4_env_path: str | Path | None = None,
    l4_albedo_path: str | Path | None = None,
    l4_summary_path: str | Path | None = None,
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
    # L5 collision / solidity layer (all optional; emit the layer only when
    # at least l5_isosurface_path is provided)
    l5_isosurface_path: str | Path | None = None,
    l5_convex_set_path: str | Path | None = None,
    l5_mass_props_path: str | Path | None = None,
    l5_sdf_path: str | Path | None = None,
    # L6 physics-material layer (optional; emit only when l6_regions_path given)
    l6_regions_path: str | Path | None = None,
    l6_articulation: list[LayerArticulation] | None = None,
    # L7 dynamics layer (optional; both files must be supplied together)
    l7_deformation_path: str | Path | None = None,
    l7_timeline_path: str | Path | None = None,
    l7_representation: str | None = None,
) -> AstelPackage:
    """Build a minimal valid :class:`AstelPackage` from L3 (+ optional L0/L5/L6).

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
    l4_env_path:
        Optional path to the estimated SH environment JSON (``l4-env.json``).
        When provided (or ``l4_albedo_path``), an L4 appearance layer is
        emitted with ``LayerAppearance(bound_to="l3")``.
    l4_albedo_path:
        Optional path to the albedo / baked-PBR splat ``.ply`` (the un-lit
        base colour for engines that consume coloured splats).
    l4_summary_path:
        Optional path to the L4 summary JSON (method + confidence + notes).
    l5_isosurface_path:
        Optional path to the watertight surface file (.stl or .3mf). When
        provided, an L5 collision layer is emitted with
        ``LayerIsosurface(print_physics_only=True)``.
    l5_convex_set_path:
        Optional path to the convex decomposition file (.glb or .npz).
    l5_mass_props_path:
        Optional path to the mass properties JSON file.
    l5_sdf_path:
        Optional path to the SDF volume file.
    l6_regions_path:
        Optional path to the physics-material regions JSON (``l6.json``).
        When provided, an L6 physics-material layer is emitted.
    l6_articulation:
        Optional list of :class:`LayerArticulation` entries describing
        separable joints between regions.
    l7_deformation_path:
        Optional path to the deformation field / 4DGS keyframe deltas
        (``.bin``).  Must be supplied together with ``l7_timeline_path``; if
        only one of the pair is given a :exc:`ValueError` is raised.
    l7_timeline_path:
        Optional path to the dynamics timeline JSON (``timeline.json``).
        Must be supplied together with ``l7_deformation_path``.
    l7_representation:
        How motion is encoded; one of ``"deformation_field"``,
        ``"keyframes"``, or ``"baked_per_frame"``.  Defaults to
        ``"deformation_field"`` when the L7 layer is emitted.
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

    # --- L4 appearance / lighting layer ---
    # Emitted when an estimated environment and/or albedo (baked-PBR) artifact
    # is provided. The asset stays splats; L4 binds per-splat material +
    # separated illumination so engines/relight can re-shade (CLAUDE.md §3 L4).
    if l4_env_path is not None or l4_albedo_path is not None:
        l4_file_refs: list[FileRef] = []
        appearance_kwargs: dict[str, object] = {"bound_to": "l3"}

        if l4_env_path is not None:
            l4_env = Path(l4_env_path)
            l4_env_pkg_path = f"layers/l4_appearance/{l4_env.name}"
            files[l4_env_pkg_path] = l4_env.read_bytes()
            l4_file_refs.append(
                FileRef(path=l4_env_pkg_path, role="env_map", format="json")
            )
            appearance_kwargs["env_map_path"] = l4_env_pkg_path

        if l4_albedo_path is not None:
            l4_albedo = Path(l4_albedo_path)
            l4_albedo_pkg_path = f"layers/l4_appearance/{l4_albedo.name}"
            files[l4_albedo_pkg_path] = l4_albedo.read_bytes()
            l4_file_refs.append(
                FileRef(
                    path=l4_albedo_pkg_path,
                    role="baked_preview",
                    format="ply",
                    derived=True,
                )
            )
            appearance_kwargs["baked_pbr_path"] = l4_albedo_pkg_path

        if l4_summary_path is not None:
            l4_summary = Path(l4_summary_path)
            l4_summary_pkg_path = f"layers/l4_appearance/{l4_summary.name}"
            files[l4_summary_pkg_path] = l4_summary.read_bytes()
            l4_file_refs.append(
                FileRef(path=l4_summary_pkg_path, role="auxiliary", format="json")
            )

        layers_kwargs["l4"] = LayerEntry(
            kind="appearance",
            status="present",
            derived_from=["l3"],
            appearance=LayerAppearance.model_validate(appearance_kwargs),
            files=l4_file_refs,
        )

    # --- L5 collision layer ---
    if l5_isosurface_path is not None:
        l5_iso_path = Path(l5_isosurface_path)
        ext = l5_iso_path.suffix.lstrip(".").lower()
        iso_format = ext if ext in ("stl", "3mf") else "stl"
        l5_iso_pkg_path = f"layers/l5_collision/{l5_iso_path.name}"
        files[l5_iso_pkg_path] = l5_iso_path.read_bytes()

        l5_file_refs: list[FileRef] = [
            FileRef(
                path=l5_iso_pkg_path,
                role="isosurface",
                format=iso_format,  # type: ignore[arg-type]
            )
        ]

        iso_entry = LayerIsosurface(path=l5_iso_pkg_path, print_physics_only=True)
        collision_kwargs: dict[str, object] = {"isosurface": iso_entry}

        if l5_convex_set_path is not None:
            l5_convex = Path(l5_convex_set_path)
            cext = l5_convex.suffix.lstrip(".").lower()
            convex_fmt = cext if cext in ("glb", "npz") else "glb"
            l5_convex_pkg_path = f"layers/l5_collision/{l5_convex.name}"
            files[l5_convex_pkg_path] = l5_convex.read_bytes()
            l5_file_refs.append(
                FileRef(
                    path=l5_convex_pkg_path,
                    role="convex_set",
                    format=convex_fmt,  # type: ignore[arg-type]
                )
            )
            collision_kwargs["convex_set_path"] = l5_convex_pkg_path

        if l5_mass_props_path is not None:
            l5_mass = Path(l5_mass_props_path)
            l5_mass_pkg_path = f"layers/l5_collision/{l5_mass.name}"
            files[l5_mass_pkg_path] = l5_mass.read_bytes()
            l5_file_refs.append(
                FileRef(
                    path=l5_mass_pkg_path,
                    role="mass_props",
                    format="json",
                )
            )
            collision_kwargs["mass_props_path"] = l5_mass_pkg_path

        if l5_sdf_path is not None:
            l5_sdf = Path(l5_sdf_path)
            l5_sdf_pkg_path = f"layers/l5_collision/{l5_sdf.name}"
            files[l5_sdf_pkg_path] = l5_sdf.read_bytes()
            l5_file_refs.append(
                FileRef(
                    path=l5_sdf_pkg_path,
                    role="sdf",
                    format="npz",
                )
            )
            collision_kwargs["sdf_path"] = l5_sdf_pkg_path

        l5_layer = LayerEntry(
            kind="collision",
            status="present",
            derived_from=["l3"],
            collision=LayerCollision.model_validate(collision_kwargs),
            files=l5_file_refs,
        )
        layers_kwargs["l5"] = l5_layer

    # --- L6 physics-material layer ---
    if l6_regions_path is not None:
        l6_reg = Path(l6_regions_path)
        l6_reg_pkg_path = f"layers/l6_physics/{l6_reg.name}"
        files[l6_reg_pkg_path] = l6_reg.read_bytes()

        l6_file_refs: list[FileRef] = [
            FileRef(
                path=l6_reg_pkg_path,
                role="regions",
                format="json",
            )
        ]

        pm_kwargs: dict[str, object] = {"regions_path": l6_reg_pkg_path}
        if l6_articulation:
            pm_kwargs["articulation"] = l6_articulation
        pm = LayerPhysicsMaterial.model_validate(pm_kwargs)
        l6_layer = LayerEntry(
            kind="physics_material",
            status="present",
            derived_from=["l3"],
            physics_material=pm,
            files=l6_file_refs,
        )
        layers_kwargs["l6"] = l6_layer

    # --- L7 dynamics layer ---
    # Both deformation and timeline files are required together; a partial
    # supply is an error (mirrors the all-or-nothing L0 validation style).
    have_l7_deformation = l7_deformation_path is not None
    have_l7_timeline = l7_timeline_path is not None
    if have_l7_deformation != have_l7_timeline:
        missing = "l7_timeline_path" if have_l7_deformation else "l7_deformation_path"
        supplied = "l7_deformation_path" if have_l7_deformation else "l7_timeline_path"
        raise ValueError(
            f"{missing} must be provided when {supplied} is given; "
            "both files are required for a meaningful dynamics layer"
        )
    if have_l7_deformation and have_l7_timeline:
        l7_def = Path(l7_deformation_path)  # type: ignore[arg-type]
        l7_tl = Path(l7_timeline_path)  # type: ignore[arg-type]
        l7_def_pkg_path = f"layers/l7_dynamics/{l7_def.name}"
        l7_tl_pkg_path = f"layers/l7_dynamics/{l7_tl.name}"
        files[l7_def_pkg_path] = l7_def.read_bytes()
        files[l7_tl_pkg_path] = l7_tl.read_bytes()

        l7_file_refs: list[FileRef] = [
            FileRef(path=l7_def_pkg_path, role="deformation", format="bin"),
            FileRef(path=l7_tl_pkg_path, role="timeline", format="json"),
        ]

        dynamics = LayerDynamics.model_validate(
            {
                "representation": l7_representation
                if l7_representation is not None
                else "deformation_field",
                "deformation_path": l7_def_pkg_path,
                "timeline_path": l7_tl_pkg_path,
            }
        )
        layers_kwargs["l7"] = LayerEntry(
            kind="dynamics",
            status="present",
            derived_from=["l3"],
            dynamics=dynamics,
            files=l7_file_refs,
        )

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
