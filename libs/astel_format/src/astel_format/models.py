"""Pydantic v2 models mirroring docs/specs/schemas/*.json (manifest-v0).

Field names match the JSON Schemas exactly so ``model_dump(mode="json",
exclude_unset=True)`` round-trips byte-identically at the JSON level
(modulo key order/whitespace). ``exclude_unset=True`` is the round-trip
mechanism: a model loaded via ``model_validate`` marks every key present in
the source JSON as "set" -- including explicit ``null`` values, which the
honesty contract (manifest-v0.md section 6) requires to be preserved
alongside their ``reason`` -- while keys never assigned by a builder are
omitted from output, satisfying ``additionalProperties: false`` schemas.

Every model sets ``extra="allow"`` so unknown additive keys (manifest-v0.md
section 10 forward-migration policy) and vendor ``extensions``/``extras``
blocks survive read -> write untouched.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class AstelModel(BaseModel):
    """Base model: preserve unknown keys, round-trip via exclude_unset."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# manifest.schema.json $defs
# ---------------------------------------------------------------------------


class Generator(AstelModel):
    name: str
    version: str


class AssetIdentity(AstelModel):
    """``manifest.astel``: asset identity (manifest.schema.json $defs/assetIdentity)."""

    id: str
    created: str
    generator: Generator
    source_modality: Literal["text", "image", "multi_image", "video"]
    name: str | None = None
    prompt: str | None = None
    seed: int | None = None


CoordAxis = Literal["+X", "+Y", "+Z", "-X", "-Y", "-Z"]


class CoordinateSystem(AstelModel):
    """manifest.schema.json#/$defs/coordinateSystem."""

    handedness: Literal["right", "left"]
    up_axis: CoordAxis
    forward_axis: CoordAxis
    meters_per_unit: float = 1.0


ScaleMethod = Literal["sfm_exif", "metric_depth_consensus", "vlm_size_estimate", "user"]


class ScaleConfidenceInterval(AstelModel):
    ci_low: float
    ci_high: float
    distribution: Literal["lognormal", "normal", "uniform", "unknown"] = "unknown"


class ScaleSource(AstelModel):
    method: ScaleMethod
    meters_per_unit: float
    weight: float | None = None


class Scale(AstelModel):
    """manifest.schema.json#/$defs/scale -- the metric grounding block."""

    meters_per_unit: float
    confidence: ScaleConfidenceInterval
    method: ScaleMethod
    sources: list[ScaleSource] | None = None
    user_overridden: bool = False


ExportTarget = Literal[
    "gltf", "gltf_spz", "usd", "usdz", "spz_sidecar", "sog_sidecar", "ply", "3mf", "stl"
]
LayerId = Literal["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"]


class ExportRecord(AstelModel):
    """manifest.schema.json#/$defs/exportRecord."""

    target: ExportTarget
    path: str
    sidecar_path: str | None = None
    source_layer: Literal["l2", "l3"] | None = None
    dropped_layers: list[LayerId] | None = None
    kernel_conversion: str | None = None
    coordinate_transform: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# layer.schema.json
# ---------------------------------------------------------------------------

LayerKind = Literal[
    "seed_pointcloud",
    "dense_pointcloud",
    "coarse_gaussians",
    "refined_gaussians",
    "appearance",
    "collision",
    "physics_material",
    "dynamics",
]
LayerStatus = Literal["present", "pending", "failed", "skipped"]

FileRole = Literal[
    "master",
    "delivery",
    "provenance",
    "semantics",
    "env_map",
    "baked_preview",
    "sdf",
    "convex_set",
    "isosurface",
    "mass_props",
    "regions",
    "deformation",
    "timeline",
    "auxiliary",
]
FileFormat = Literal[
    "ply",
    "spz",
    "sog",
    "splat",
    "ksplat",
    "glb",
    "gltf",
    "npz",
    "bin",
    "json",
    "hdr",
    "exr",
    "webp",
    "3mf",
    "stl",
]


class FileRef(AstelModel):
    """layer.schema.json#/$defs/fileRef.

    ``path`` must be a POSIX-relative path with no leading ``/`` and no
    ``..`` traversal segment (enforced again, defensively, by
    :mod:`astel_format.package`).
    """

    path: str
    role: FileRole
    format: FileFormat
    derived: bool = False
    sha256: str | None = None
    bytes: int | None = None


# Kernel type: enum members or "custom:*" pattern -> plain str at the model
# level (validated loosely; the JSON Schema is the strict gate on write).
KernelType = Annotated[
    str, Field(description="gaussian_3d|gaussian_2d|... or custom:*")
]


class KernelBatch(AstelModel):
    kernel_type: KernelType
    first: int
    count: int
    attributes: list[str] | None = None


class LayerGeometricMetrics(AstelModel):
    chamfer_mm: float | None = None
    mean_mm: float | None = None
    p95_mm: float | None = None
    reference_layer: Literal["l0", "l1"] | None = None


class LayerMetrics(AstelModel):
    wall_seconds: float | None = None
    vram_peak_mb: float | None = None
    usd_estimate: float | None = None
    geometric: LayerGeometricMetrics | None = None


class LayerBudget(AstelModel):
    tier: Literal["lowpoly", "standard", "cinematic"]
    target_count: int | None = None
    actual_count: int | None = None


class LayerAppearance(AstelModel):
    bound_to: Literal["l2", "l3"] | None = None
    albedo_accessor: int | None = None
    roughness_accessor: int | None = None
    metallic_accessor: int | None = None
    specular_accessor: int | None = None
    emissive_accessor: int | None = None
    env_map_path: str | None = None
    baked_pbr_path: str | None = None


class LayerIsosurface(AstelModel):
    path: str
    print_physics_only: Literal[True] = True


class LayerCollision(AstelModel):
    sdf_path: str | None = None
    convex_set_path: str | None = None
    isosurface: LayerIsosurface | None = None
    mass_props_path: str | None = None


class LayerArticulation(AstelModel):
    type: Literal["revolute", "prismatic", "fixed", "free"] | None = None
    parent_region: int | None = None
    child_region: int | None = None
    axis: list[float] | None = None


class LayerPhysicsMaterial(AstelModel):
    regions_path: str | None = None
    region_map_accessor: int | None = None
    articulation: list[LayerArticulation] | None = None


class LayerDynamics(AstelModel):
    representation: (
        Literal["deformation_field", "keyframes", "baked_per_frame"] | None
    ) = None
    deformation_path: str | None = None
    timeline_path: str | None = None


class LayerEntry(AstelModel):
    """layer.schema.json -- one L0..L7 layer entry.

    ``kernel_type`` and ``kernel_batches`` are mutually exclusive per the
    schema's ``allOf``/``not`` rule; this is enforced in
    :meth:`astel_format.package.AstelPackage._validate_manifest` via the
    JSON Schema validator rather than re-implemented here, since the schema
    is authoritative.
    """

    kind: LayerKind
    status: LayerStatus
    files: list[FileRef] | None = None
    count: int | None = None
    derived_from: list[LayerId] | None = None
    metrics: LayerMetrics | None = None
    provenance_ref: int | None = None
    kernel_type: KernelType | None = None
    kernel_batches: list[KernelBatch] | None = None
    budget: LayerBudget | None = None
    appearance: LayerAppearance | None = None
    collision: LayerCollision | None = None
    physics_material: LayerPhysicsMaterial | None = None
    dynamics: LayerDynamics | None = None


# ---------------------------------------------------------------------------
# buffers.schema.json
# ---------------------------------------------------------------------------


class BufferEntry(AstelModel):
    uri: str
    byte_length: int


class BufferView(AstelModel):
    buffer: int
    byte_offset: int
    byte_length: int
    byte_stride: int | None = None


# component_type: glTF GL enums (ints) or "UNORM8"/"UNORM16" string aliases.
ComponentType = Literal[5120, 5121, 5122, 5123, 5125, 5126, "UNORM8", "UNORM16"]
AccessorType = Literal["SCALAR", "VEC2", "VEC3", "VEC4", "MAT3", "MAT4"]


class AccessorQuantization(AstelModel):
    scale: float | None = None
    offset: float | None = None


class Accessor(AstelModel):
    buffer_view: int
    component_type: ComponentType
    type: AccessorType
    count: int
    normalized: bool = False
    quantization: AccessorQuantization | None = None


class BufferTable(AstelModel):
    """buffers.schema.json -- glTF-shaped buffers/buffer_views/accessors."""

    buffers: list[BufferEntry]
    buffer_views: list[BufferView]
    accessors: list[Accessor]


# ---------------------------------------------------------------------------
# provenance.schema.json
# ---------------------------------------------------------------------------


class ProvenanceChannel(AstelModel):
    """One channel descriptor in ``provenance.channels``."""

    layer: Literal["l0", "l1", "l2", "l3"]
    accessor: int
    count: int


class ProvenanceExportCarriers(AstelModel):
    gltf_attribute: Literal["_ASTEL_PROVENANCE"] | None = None
    usd_primvar: Literal["primvars:astel:provenance"] | None = None
    spz_sidecar: bool | None = None


class ProvenanceDescriptor(AstelModel):
    """provenance.schema.json -- the top-level ``provenance`` block."""

    semantic: Literal["measured_vs_generated"] = "measured_vs_generated"
    range: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    convention: Literal["1=measured, 0=generated"] = "1=measured, 0=generated"
    precision: Literal["u8", "u16"]
    channels: list[ProvenanceChannel]
    export_carriers: ProvenanceExportCarriers | None = None


# ---------------------------------------------------------------------------
# quality-report.schema.json
# ---------------------------------------------------------------------------


class GeometricError(AstelModel):
    """``quality_report.geometric_error``.

    Honesty contract (manifest-v0.md section 6): ``chamfer_mm``/``mean_mm``/
    ``p95_mm`` are either real measurements or explicit ``null``, in which
    case ``reason`` is required. The schema does not make ``reason``
    conditionally required via JSON Schema machinery (no ``if``/``then``
    here), so this is enforced by convention / the builder, not re-validated
    structurally beyond the base schema.
    """

    units: Literal["mm"] = "mm"
    reference_layer: Literal["l0", "l1"]
    chamfer_mm: float | None = None
    mean_mm: float | None = None
    p95_mm: float | None = None
    method: str | None = None
    reason: str | None = None


class ScaleConfidence(AstelModel):
    """``quality_report.scale_confidence``."""

    meters_per_unit: float
    ci_low: float
    ci_high: float
    ci_method: str | None = None
    sources: list[ScaleMethod] | None = None


class HallucinationReport(AstelModel):
    """``quality_report.hallucination``."""

    measured_fraction: float
    generated_fraction: float
    heatmap_ref: int | str | None = None
    unknown_fraction: float | None = None


class ViewMetrics(AstelModel):
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    n_holdout_views: int | None = None


class StageTelemetry(AstelModel):
    total_wall_seconds: float | None = None
    peak_vram_mb: float | None = None
    total_usd_estimate: float | None = None


class QualityReport(AstelModel):
    """quality-report.schema.json -- the Truth Meter substrate.

    Used both as ``manifest.quality_report`` (inline summary) and as the
    full ``quality/report.json``; the schema is identical for both per
    manifest-v0.md section 6.
    """

    geometric_error: GeometricError
    scale_confidence: ScaleConfidence
    hallucination: HallucinationReport
    view_metrics: ViewMetrics | None = None
    stage_telemetry: StageTelemetry | None = None
    caveats: list[str] | None = None


# ---------------------------------------------------------------------------
# manifest.schema.json (root)
# ---------------------------------------------------------------------------


class Layers(AstelModel):
    """``manifest.layers`` -- ordered map of present L0..L7 entries.

    All keys optional (absent layers omitted entirely); at least one must be
    present (``minProperties: 1`` in the schema, checked at validation time
    against the JSON Schema, not re-implemented here).
    """

    l0: LayerEntry | None = None
    l1: LayerEntry | None = None
    l2: LayerEntry | None = None
    l3: LayerEntry | None = None
    l4: LayerEntry | None = None
    l5: LayerEntry | None = None
    l6: LayerEntry | None = None
    l7: LayerEntry | None = None


class Manifest(AstelModel):
    """manifest.schema.json -- the root ``manifest.json`` contract."""

    format_version: str
    astel: AssetIdentity
    coordinate_system: CoordinateSystem
    scale: Scale
    layers: Layers
    buffers: BufferTable
    provenance: ProvenanceDescriptor
    quality_report: QualityReport
    exports: list[ExportRecord] | None = None
    extensions: dict[str, dict[str, object]] | None = None
    extras: object | None = None
