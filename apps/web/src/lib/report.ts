/** Shape of the per-asset quality report (Truth Meter source). */

export interface QualityReport {
  asset_id: string;
  name: string;
  modality: "generated" | "captured" | "hybrid";
  splat_count: number;
  splat_budget: string;
  active_layer: string;
  scale: {
    longest_axis_m: number;
    confidence: number;
    method: string;
    note: string;
  };
  geometry: {
    chamfer_mm_vs_l1: number;
    psnr_db: number;
    ssim: number;
    normals_present: boolean;
  };
  provenance: {
    measured_ratio: number;
    generated_ratio: number;
    note: string;
  };
  /**
   * Honesty marker (CLAUDE.md §1.3). When set and not `"measured"`, the Truth
   * Meter MUST visibly flag the numbers as illustrative/stub, never as
   * measured fact. Absent on the static sample (treated as non-stub).
   */
  origin?: string;
  /** First caveat string from the API report, if any, shown alongside the
   * stub indicator. */
  caveat?: string | null;
}

export async function fetchSampleReport(
  signal?: AbortSignal,
): Promise<QualityReport> {
  const res = await fetch("/samples/astel-sample.report.json", {
    signal: signal ?? null,
  });
  if (!res.ok) throw new Error(`report ${res.status}`);
  return (await res.json()) as QualityReport;
}

/** Shape of `GET /v1/generations/{taskId}/artifacts/quality-report.json`. */
export interface ApiQualityReport {
  schema: string;
  origin: string;
  modality: string;
  splats: number;
  geometric_error: {
    chamfer_mm_vs_l1: number;
    method: string;
  };
  fidelity: {
    psnr_db: number;
    ssim: number | null;
    lpips: number | null;
    n_holdout_views: number;
  };
  scale: {
    longest_axis_m: number;
    confidence: number;
    method: string;
  };
  provenance: {
    measured_ratio: number;
    generated_ratio: number;
  };
  caveats: string[];
}

/**
 * Map the live API quality report onto the {@link QualityReport} shape the
 * Truth Meter renders. See CLAUDE.md task brief for the field mapping.
 */
export function mapApiReport(
  api: ApiQualityReport,
  assetId: string,
): QualityReport {
  return {
    asset_id: assetId,
    name: `Generation ${assetId}`,
    modality: api.modality === "text" ? "generated" : "hybrid",
    splat_count: api.splats,
    splat_budget: `${Math.round(api.splats / 1000)}k`,
    active_layer: "L3",
    scale: {
      longest_axis_m: api.scale.longest_axis_m,
      confidence: api.scale.confidence,
      method: api.scale.method,
      note: `${api.scale.method} (${api.modality})`,
    },
    geometry: {
      chamfer_mm_vs_l1: api.geometric_error.chamfer_mm_vs_l1,
      psnr_db: api.fidelity.psnr_db,
      ssim: api.fidelity.ssim ?? 0,
      normals_present: true,
    },
    provenance: {
      measured_ratio: api.provenance.measured_ratio,
      generated_ratio: api.provenance.generated_ratio,
      note: api.caveats[0] ?? "",
    },
    origin: api.origin,
    caveat: api.caveats[0] ?? null,
  };
}
