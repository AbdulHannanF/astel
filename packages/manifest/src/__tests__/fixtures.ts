import type { Manifest } from "../types.js";

/**
 * A minimal valid manifest fixture: L0 seed cloud + L3 refined gaussians, one provenance
 * accessor bound to L3, and an honest quality report with a `null` geometric_error
 * (text-to-3D, no measured reference) plus a `reason`.
 *
 * Also carries an unknown additive top-level key, an `extensions.*` block, and `extras` to
 * exercise forward-compat preservation on round-trip.
 */
export function makeFixtureManifest(): Manifest {
  return {
    format_version: "0.1.0",
    astel: {
      id: "018f6f1e-7e6b-7c3a-8b3a-000000000001",
      created: "2026-06-13T00:00:00Z",
      generator: { name: "astel-pipeline", version: "0.1.0" },
      source_modality: "text",
      name: "test widget",
      prompt: "a small brass widget",
      seed: 42,
    },
    coordinate_system: {
      handedness: "right",
      up_axis: "+Y",
      forward_axis: "-Z",
      meters_per_unit: 1.0,
    },
    scale: {
      meters_per_unit: 0.1,
      confidence: { ci_low: 0.05, ci_high: 0.2, distribution: "lognormal" },
      method: "vlm_size_estimate",
      sources: [{ method: "vlm_size_estimate", meters_per_unit: 0.1, weight: 1.0 }],
      user_overridden: false,
    },
    layers: {
      l0: {
        kind: "seed_pointcloud",
        status: "present",
        count: 4096,
        files: [
          {
            path: "layers/l0_seed/points.ply",
            role: "master",
            format: "ply",
            bytes: 131072,
            sha256: "a".repeat(64),
          },
          {
            path: "layers/l0_seed/provenance.bin",
            role: "provenance",
            format: "bin",
            bytes: 4096,
          },
        ],
        provenance_ref: 0,
      },
      l3: {
        kind: "refined_gaussians",
        status: "present",
        count: 100000,
        derived_from: ["l0"],
        kernel_type: "gaussian_2d",
        budget: { tier: "lowpoly", target_count: 100000, actual_count: 100000 },
        metrics: {
          wall_seconds: 120.5,
          vram_peak_mb: 8192,
          usd_estimate: 0.05,
        },
        files: [
          {
            path: "layers/l3_refined/splats.ply",
            role: "master",
            format: "ply",
            bytes: 5242880,
            sha256: "b".repeat(64),
          },
          {
            path: "layers/l3_refined/provenance.bin",
            role: "provenance",
            format: "bin",
            bytes: 100000,
          },
        ],
        provenance_ref: 1,
      },
    },
    buffers: {
      buffers: [
        { uri: "layers/l0_seed/provenance.bin", byte_length: 4096 },
        { uri: "layers/l3_refined/provenance.bin", byte_length: 100000 },
      ],
      buffer_views: [
        { buffer: 0, byte_offset: 0, byte_length: 4096 },
        { buffer: 1, byte_offset: 0, byte_length: 100000 },
      ],
      accessors: [
        { buffer_view: 0, component_type: "UNORM8", type: "SCALAR", count: 4096, normalized: true },
        { buffer_view: 1, component_type: "UNORM8", type: "SCALAR", count: 100000, normalized: true },
      ],
    },
    provenance: {
      semantic: "measured_vs_generated",
      range: [0.0, 1.0],
      convention: "1=measured, 0=generated",
      precision: "u8",
      channels: [
        { layer: "l0", accessor: 0, count: 4096 },
        { layer: "l3", accessor: 1, count: 100000 },
      ],
      export_carriers: {
        gltf_attribute: "_ASTEL_PROVENANCE",
        usd_primvar: "primvars:astel:provenance",
        spz_sidecar: true,
      },
    },
    quality_report: {
      geometric_error: {
        chamfer_mm: null,
        mean_mm: null,
        p95_mm: null,
        reference_layer: "l1",
        units: "mm",
        reason: "pure text-to-3D input has no measured L1 reference",
      },
      scale_confidence: {
        meters_per_unit: 0.1,
        ci_low: 0.05,
        ci_high: 0.2,
        ci_method: "consensus_spread",
        sources: ["vlm_size_estimate"],
      },
      hallucination: {
        heatmap_ref: 1,
        measured_fraction: 0.0,
        generated_fraction: 0.9,
        unknown_fraction: 0.1,
      },
      view_metrics: null,
      stage_telemetry: {
        total_wall_seconds: 120.5,
        peak_vram_mb: 8192,
        total_usd_estimate: 0.05,
      },
      caveats: ["entire asset is generated from a text prompt; no held-out views exist"],
    },
    extensions: {
      vendor_example: { note: "preserved on round-trip", value: 123 },
    },
    extras: {
      internal_tag: "fixture",
    },
  } satisfies Manifest;
}
