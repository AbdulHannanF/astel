/**
 * Origin taxonomy — must stay in sync with the Python schema and
 * `@astel/manifest` `AssetOrigin` type (three literals, optional/additive).
 *
 * - `measured`  = reconstructed from real capture with ground-truth comparison.
 * - `generated` = generative / self-consistency only, no ground truth.
 * - `stub`      = deterministic procedural placeholder, not derived from user input.
 */
export type AssetOrigin = "measured" | "generated" | "stub";

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
   * Typed origin taxonomy (CLAUDE.md §1.3 honesty contract). When set and not
   * `"measured"`, the Truth Meter MUST visibly flag the numbers as
   * illustrative/not-ground-truth. Optional/additive: absent on older packages.
   */
  origin?: AssetOrigin;
  /** First caveat string from the API report, if any, shown alongside the
   * origin indicator. */
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
  origin: AssetOrigin | string;
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

/** Known origin values from the API; anything else is treated as absent. */
const KNOWN_ORIGINS: ReadonlySet<AssetOrigin> = new Set<AssetOrigin>([
  "measured",
  "generated",
  "stub",
]);

function parseOrigin(raw: string): AssetOrigin | undefined {
  return KNOWN_ORIGINS.has(raw as AssetOrigin) ? (raw as AssetOrigin) : undefined;
}

/**
 * Map the live API quality report onto the {@link QualityReport} shape the
 * Truth Meter renders. See CLAUDE.md task brief for the field mapping.
 */
export function mapApiReport(
  api: ApiQualityReport,
  assetId: string,
): QualityReport {
  // `origin` is an optional field under exactOptionalPropertyTypes, so only
  // include the key when it resolves to a known value (never assign undefined).
  const origin = parseOrigin(api.origin);
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
    ...(origin !== undefined ? { origin } : {}),
    caveat: api.caveats[0] ?? null,
  };
}
