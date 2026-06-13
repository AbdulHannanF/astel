/**
 * TypeScript types mirroring docs/specs/schemas/*.json (JSON Schema draft 2020-12).
 *
 * Field names, enums, and required-ness here MUST match the schemas exactly — the schema
 * wins (docs/specs/manifest-v0.md). Where the schema is ambiguous, a comment notes the
 * interpretation chosen.
 */

// ---------------------------------------------------------------------------------------------
// Shared / buffers.schema.json
// ---------------------------------------------------------------------------------------------

/** glTF GL component-type enums plus Astel's UNORM aliases. */
export type ComponentType = 5120 | 5121 | 5122 | 5123 | 5125 | 5126 | "UNORM8" | "UNORM16";

export type AccessorElementType = "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT3" | "MAT4";

export interface AccessorQuantization {
  /** Multiplier applied after reading the stored integer. */
  scale?: number;
  /** Value added after scaling. */
  offset?: number;
}

export interface Accessor {
  /** Index into the buffer_views array. */
  buffer_view: number;
  component_type: ComponentType;
  type: AccessorElementType;
  /** Number of elements. For per-gaussian channels, equals the bound layer's gaussian count. */
  count: number;
  /** Implied true for UNORM8/UNORM16 aliases. */
  normalized?: boolean;
  quantization?: AccessorQuantization;
}

export interface BufferView {
  /** Index into the buffers array. */
  buffer: number;
  byte_offset: number;
  byte_length: number;
  /** Stride in bytes for interleaved data; omit for tightly packed. */
  byte_stride?: number;
}

export interface BufferDescriptor {
  /** POSIX relative path to a .bin file inside the package. */
  uri: string;
  byte_length: number;
}

export interface BufferTable {
  buffers: BufferDescriptor[];
  buffer_views: BufferView[];
  accessors: Accessor[];
}

// ---------------------------------------------------------------------------------------------
// layer.schema.json
// ---------------------------------------------------------------------------------------------

export type LayerId = "l0" | "l1" | "l2" | "l3" | "l4" | "l5" | "l6" | "l7";

export type LayerKind =
  | "seed_pointcloud"
  | "dense_pointcloud"
  | "coarse_gaussians"
  | "refined_gaussians"
  | "appearance"
  | "collision"
  | "physics_material"
  | "dynamics";

export type LayerStatus = "present" | "pending" | "failed" | "skipped";

export type FileRole =
  | "master"
  | "delivery"
  | "provenance"
  | "semantics"
  | "env_map"
  | "baked_preview"
  | "sdf"
  | "convex_set"
  | "isosurface"
  | "mass_props"
  | "regions"
  | "deformation"
  | "timeline"
  | "auxiliary";

export type FileFormat =
  | "ply"
  | "spz"
  | "sog"
  | "splat"
  | "ksplat"
  | "glb"
  | "gltf"
  | "npz"
  | "bin"
  | "json"
  | "hdr"
  | "exr"
  | "webp"
  | "3mf"
  | "stl";

export interface FileRef {
  /** POSIX relative path from package root. No absolute paths, no '..' traversal. */
  path: string;
  /** Disambiguates multiple files of a layer. */
  role: FileRole;
  format: FileFormat;
  /** True if this file is a lossy derivative of a master in the same layer. */
  derived?: boolean;
  /** Lowercase hex SHA-256 of the file contents. */
  sha256?: string;
  /** File size in bytes. */
  bytes?: number;
}

/**
 * Splat kernel type. The schema constrains this with `anyOf` (a fixed enum OR a
 * `custom:<name>` pattern); represented here as a union of the known literals plus a
 * template-literal escape hatch for `custom:*` and any future reserved values.
 */
export type KernelType =
  | "gaussian_3d"
  | "gaussian_2d"
  | "gaussian_spindle"
  | "gaussian_ray"
  | `custom:${string}`;

export type JointType = "revolute" | "prismatic" | "fixed" | "free";

export interface ArticulationJoint {
  type?: JointType;
  /** Region id of the parent part. */
  parent_region?: number;
  /** Region id of the child part. */
  child_region?: number;
  /** Joint axis in native coordinates. Exactly 3 components per schema (minItems/maxItems 3). */
  axis?: [number, number, number];
}

export interface KernelBatch {
  kernel_type: KernelType;
  /** Index of the first splat in this batch within the layer's ordered splat buffer. */
  first: number;
  /** Number of splats in this batch; covers indices [first, first+count). */
  count: number;
  /** Names of per-splat fields this batch carries. */
  attributes?: string[];
}

export type BudgetTier = "lowpoly" | "standard" | "cinematic";

export interface LayerBudget {
  tier: BudgetTier;
  /** Requested gaussian count. */
  target_count?: number;
  /** Achieved gaussian count. */
  actual_count?: number;
}

export interface LayerGeometricMetrics {
  /** Symmetric Chamfer distance to reference layer, in mm. */
  chamfer_mm?: number | null;
  /** Mean point-to-surface distance, in mm. */
  mean_mm?: number | null;
  /** 95th-percentile distance, in mm. */
  p95_mm?: number | null;
  /** Layer used as the geometric reference. */
  reference_layer?: "l0" | "l1";
}

export interface LayerMetrics {
  /** Stage wall-clock time in seconds. */
  wall_seconds?: number;
  /** Peak VRAM in MB during this stage. */
  vram_peak_mb?: number;
  /** Estimated compute cost in USD for this stage. */
  usd_estimate?: number;
  /** Geometric quality of this layer vs its reference (primarily for L3 vs L1). */
  geometric?: LayerGeometricMetrics;
}

/** L4-specific block. */
export interface LayerAppearance {
  /** Splat layer these materials index (per-gaussian arrays are parallel to it). */
  bound_to?: "l2" | "l3";
  /** Accessor index for per-gaussian albedo (VEC3). */
  albedo_accessor?: number;
  /** Accessor index for per-gaussian roughness (SCALAR). */
  roughness_accessor?: number;
  /** Accessor index for per-gaussian metallic (SCALAR). */
  metallic_accessor?: number;
  /** Accessor index for per-gaussian specular (SCALAR). */
  specular_accessor?: number;
  /** Accessor index for per-gaussian emissive (VEC3). */
  emissive_accessor?: number;
  /** Relative path to estimated environment illumination (.hdr/.exr). */
  env_map_path?: string;
  /** Relative path to a baked-preview splat file derived from L4. */
  baked_pbr_path?: string;
}

/** L5-specific block. */
export interface LayerCollisionIsosurface {
  /** Relative path to the watertight surface (.ply). */
  path: string;
  /**
   * Always true. Marks the isosurface as never-exportable as an asset; only emitted as
   * .3mf/.stl print files. The schema fixes this with `"const": true`.
   */
  print_physics_only: true;
}

export interface LayerCollision {
  /** Relative path to the sparse-voxel SDF (.npz). */
  sdf_path?: string;
  /** Relative path to the convex decomposition proxy set (.glb). Collision data only. */
  convex_set_path?: string;
  /** Watertight surface for the print and physics-volume paths ONLY. */
  isosurface?: LayerCollisionIsosurface;
  /** Relative path to mass_props.json. Inertia is null until L6 is present. */
  mass_props_path?: string;
}

/** L6-specific block. */
export interface LayerPhysicsMaterial {
  /** Relative path to regions.json (array of region descriptors). */
  regions_path?: string;
  /** Accessor index for per-gaussian uint16 region id, index-aligned to L3. */
  region_map_accessor?: number;
  /** Optional detected joints / separable parts hints. */
  articulation?: ArticulationJoint[];
}

export type DynamicsRepresentation = "deformation_field" | "keyframes" | "baked_per_frame";

/** L7-specific block. */
export interface LayerDynamics {
  /** How motion is encoded. */
  representation?: DynamicsRepresentation;
  /** Relative path to the deformation field / 4DGS keyframe deltas (.bin). */
  deformation_path?: string;
  /** Relative path to timeline.json. */
  timeline_path?: string;
}

/**
 * One layer of the Astel Layer Stack (L0..L7).
 *
 * Note: the schema's `allOf`/`not` clause makes `kernel_type` and `kernel_batches` mutually
 * exclusive. TypeScript cannot express "at most one of these two optional properties" cleanly
 * alongside the many other optional fields without an unwieldy union, so both are modeled as
 * plain optional properties; the reader/validator enforces the mutual-exclusion constraint via
 * the JSON Schema at runtime.
 */
export interface LayerEntry {
  kind: LayerKind;
  status: LayerStatus;
  /** Files backing this layer. */
  files?: FileRef[];
  /** Primitive count for this layer (points for clouds, gaussians for splat layers). */
  count?: number;
  /** Layer ids this layer was computed from (provenance graph). */
  derived_from?: LayerId[];
  /** Per-stage telemetry and layer-specific quality. */
  metrics?: LayerMetrics;
  /** Index into provenance.channels binding this layer's per-primitive confidence buffer. */
  provenance_ref?: number;
  /** Set on a splat layer that is homogeneous in kernel type. Mutually exclusive with kernel_batches. */
  kernel_type?: KernelType;
  /** Mixed-kernel batch table. Mutually exclusive with kernel_type. */
  kernel_batches?: KernelBatch[];
  /** Splat budget for L3. */
  budget?: LayerBudget;
  /** L4-specific block. */
  appearance?: LayerAppearance;
  /** L5-specific block. */
  collision?: LayerCollision;
  /** L6-specific block. */
  physics_material?: LayerPhysicsMaterial;
  /** L7-specific block. */
  dynamics?: LayerDynamics;
  /** Free-form layer data, ignored by validators. */
  extras?: unknown;
}

// ---------------------------------------------------------------------------------------------
// provenance.schema.json
// ---------------------------------------------------------------------------------------------

export type ProvenancePrecision = "u8" | "u16";

export interface ProvenanceChannel {
  /** Layer id this provenance buffer is bound to. */
  layer: "l0" | "l1" | "l2" | "l3";
  /** Index into the manifest buffer table's accessors array. */
  accessor: number;
  /** Number of primitives, equal to the bound layer's count. */
  count: number;
}

export interface ProvenanceExportCarriers {
  /** Custom glTF vertex attribute carrying provenance on the splat primitive. Fixed by schema const. */
  gltf_attribute?: "_ASTEL_PROVENANCE";
  /** USD primvar carrying provenance. Fixed by schema const. */
  usd_primvar?: "primvars:astel:provenance";
  /** True if provenance ships in the *.astl.json sidecar for bare .spz/.sog exports. */
  spz_sidecar?: boolean;
}

/**
 * The provenance channel descriptor. `semantic`, `range`, and `convention` are fixed
 * (`const`) by the schema for v0 but typed as their literal values for documentation and
 * round-trip fidelity.
 */
export interface ProvenanceDescriptor {
  /** Fixed for v0. */
  semantic: "measured_vs_generated";
  /** Fixed [0,1] for v0. */
  range: [0.0, 1.0];
  /** Fixed for v0. */
  convention: "1=measured, 0=generated";
  precision: ProvenancePrecision;
  /** One descriptor per layer that carries a provenance buffer (typically L0-L3). */
  channels: ProvenanceChannel[];
  export_carriers?: ProvenanceExportCarriers;
}

// ---------------------------------------------------------------------------------------------
// quality-report.schema.json
// ---------------------------------------------------------------------------------------------

export interface GeometricError {
  /** Symmetric Chamfer distance L3 vs reference, in mm. Null if no measured reference. */
  chamfer_mm: number | null;
  /** Mean point-to-surface distance, in mm. */
  mean_mm: number | null;
  /** 95th-percentile distance, in mm. */
  p95_mm: number | null;
  /** Algorithm used to compute the distance. */
  method?: string;
  /** Layer used as geometric ground truth. */
  reference_layer: "l0" | "l1";
  /** Distance units; fixed to mm for v0. */
  units: "mm";
  /** Required when any distance field is null; explains why no measurement exists. */
  reason?: string;
}

export type ScaleMethod = "sfm_exif" | "metric_depth_consensus" | "vlm_size_estimate" | "user";

export interface ScaleConfidence {
  /** Best-estimate metres per native unit. */
  meters_per_unit: number;
  /** Lower bound of the confidence interval. */
  ci_low: number;
  /** Upper bound of the confidence interval. */
  ci_high: number;
  /** How the interval was derived (e.g. 'consensus_spread'). */
  ci_method?: string;
  /** Methods contributing to the scale estimate. */
  sources?: ScaleMethod[];
}

export interface Hallucination {
  /** Reference to the provenance accessor visualized as the heatmap. */
  heatmap_ref?: number | string;
  /** Fraction of primitives with provenance >= a 'measured' threshold. */
  measured_fraction: number;
  /** Fraction of primitives with provenance <= a 'generated' threshold. */
  generated_fraction: number;
  /** Fraction whose provenance could not be determined. Distinct from generated. */
  unknown_fraction?: number;
}

export interface ViewMetrics {
  /** Peak signal-to-noise ratio on held-out views, dB. */
  psnr: number | null;
  /** Structural similarity on held-out views. */
  ssim: number | null;
  /** Learned perceptual image patch similarity (lower is better). */
  lpips: number | null;
  /** Number of held-out views used. */
  n_holdout_views: number | null;
}

export interface StageTelemetry {
  /** Total pipeline wall-clock time. */
  total_wall_seconds?: number;
  /** Peak VRAM across all stages. */
  peak_vram_mb?: number;
  /** Total estimated compute cost in USD. */
  total_usd_estimate?: number;
}

export interface QualityReport {
  geometric_error: GeometricError;
  scale_confidence: ScaleConfidence;
  hallucination: Hallucination;
  /** Null for generative-only assets with no held-out views. */
  view_metrics?: ViewMetrics | null;
  /** Roll-up of per-layer telemetry (sum across stages). */
  stage_telemetry?: StageTelemetry;
  /** Free-text honesty notes surfaced in the UI. */
  caveats?: string[];
}

// ---------------------------------------------------------------------------------------------
// manifest.schema.json
// ---------------------------------------------------------------------------------------------

export type SourceModality = "text" | "image" | "multi_image" | "video";

export interface AssetGenerator {
  /** Generator name, e.g. 'astel-pipeline'. */
  name: string;
  /** Generator version string. */
  version: string;
}

export interface AssetIdentity {
  /** Stable unique id for the asset. UUIDv7 recommended (time-ordered). */
  id: string;
  /** ISO-8601 UTC timestamp of package creation. */
  created: string;
  /** Producing tool name and version, for auditable migrations. */
  generator: AssetGenerator;
  source_modality: SourceModality;
  /** Human-readable asset name. */
  name?: string;
  /** Original text prompt, if source_modality is text. */
  prompt?: string;
  /** Generation seed for reproducibility, if applicable. */
  seed?: number;
}

export type Handedness = "right" | "left";
export type Axis = "+X" | "+Y" | "+Z" | "-X" | "-Y" | "-Z";

export interface CoordinateSystem {
  handedness: Handedness;
  up_axis: Axis;
  forward_axis: Axis;
  /** Metres represented by one native unit. Default 1.0. */
  meters_per_unit: number;
}

export type ScaleDistribution = "lognormal" | "normal" | "uniform" | "unknown";

export interface ScaleConfidenceInterval {
  /** Lower bound of meters_per_unit confidence interval. */
  ci_low: number;
  /** Upper bound of meters_per_unit confidence interval. */
  ci_high: number;
  distribution?: ScaleDistribution;
}

export interface ScaleSource {
  /** Method of this individual source. */
  method: ScaleMethod;
  /** This source's scale estimate. */
  meters_per_unit: number;
  /** Relative weight in the consensus. */
  weight?: number;
}

export interface Scale {
  /** Best-estimate metres per native unit. */
  meters_per_unit: number;
  /** Confidence interval on the scale estimate. */
  confidence: ScaleConfidenceInterval;
  /** Primary method that produced the scale estimate. */
  method: ScaleMethod;
  /** Individual estimates feeding the consensus, each weighted. */
  sources?: ScaleSource[];
  /** True if the user manually set the scale. */
  user_overridden?: boolean;
}

export type ExportTarget =
  | "gltf"
  | "gltf_spz"
  | "usd"
  | "usdz"
  | "spz_sidecar"
  | "sog_sidecar"
  | "ply"
  | "3mf"
  | "stl";

export interface ExportRecord {
  target: ExportTarget;
  /** POSIX relative path to the primary exported file inside the package (or external). */
  path: string;
  /** Relative path to the companion Astel sidecar (*.astl.json). */
  sidecar_path?: string;
  /** Layer id whose splats were exported (e.g. 'l3' or 'l2' for previews). */
  source_layer?: "l2" | "l3";
  /** Layer ids present in the package but not representable in this export. */
  dropped_layers?: LayerId[];
  /** Kernel-type conversion applied. */
  kernel_conversion?: string;
  /** Description of the basis change applied. */
  coordinate_transform?: string;
  /** Free-text honesty notes about what this export loses. */
  notes?: string;
}

/**
 * Ordered map of present layers l0..l7. Absent layers are omitted entirely (the schema
 * forbids `additionalProperties` but every l0..l7 key is individually optional, with
 * `minProperties: 1`).
 */
export interface LayerMap {
  l0?: LayerEntry;
  l1?: LayerEntry;
  l2?: LayerEntry;
  l3?: LayerEntry;
  l4?: LayerEntry;
  l5?: LayerEntry;
  l6?: LayerEntry;
  l7?: LayerEntry;
}

/**
 * Root contract for an .astel package manifest.json (v0).
 *
 * `extensions` keys are namespaced (`astel_*`, `vendor_*`) and, like `extras`, MUST be
 * preserved verbatim on round-trip even when their shape is unknown to this reader
 * (forward-migration policy, manifest-v0.md section 10).
 */
export interface Manifest {
  /** SemVer string of the Astel package spec this manifest conforms to, e.g. "0.1.0". */
  format_version: string;
  astel: AssetIdentity;
  coordinate_system: CoordinateSystem;
  scale: Scale;
  /** Ordered map of present layers l0->l7. */
  layers: LayerMap;
  buffers: BufferTable;
  provenance: ProvenanceDescriptor;
  quality_report: QualityReport;
  /** Records of exports generated from this package. */
  exports?: ExportRecord[];
  /** Namespaced vendor extension blocks (astel_*, vendor_*). Preserved on round-trip. */
  extensions?: Record<string, Record<string, unknown>>;
  /** Free-form application data, ignored by validators. */
  extras?: unknown;
}
