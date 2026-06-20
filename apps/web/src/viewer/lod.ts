/**
 * LOD descriptor parser and tier selector for the Astel splat LOD system.
 *
 * Consumes the `astel.lod/v0` descriptor emitted by the producer
 * (`libs/astel_lod`) and resolves the best-fit tier for a given splat budget
 * or named platform target.
 *
 * Design notes:
 * - Pure data logic — no Three.js or DOM imports. Safe to import from any
 *   context (worker, SSR, test, engine plugin).
 * - `selectTierForBudget` never returns undefined: if no tier fits within the
 *   budget it falls back to the SMALLEST tier and documents the overage via
 *   JSDoc. This matches the Python `astel_lod.select_tier` contract.
 * - Platform budgets mirror `astel_lod.budgets.PLATFORM_BUDGETS` exactly.
 *   Keep both in sync when adding new platforms.
 */

/** One LOD tier as stored in the `astel.lod/v0` descriptor. */
export interface LodTier {
  /** Human-readable name, e.g. "mobile", "web", "cinematic". */
  name: string;
  /** Gaussian splat count for this tier. */
  count: number;
  /** Relative path to the tier's PLY/SPZ file inside the .astel package. */
  file: string;
}

/** The top-level LOD descriptor stored as `lod.json` in an `.astel` package. */
export interface LodDescriptor {
  schema: "astel.lod/v0";
  /** Tiers sorted in strictly ascending order by `count`. */
  tiers: LodTier[];
}

/**
 * Canonical platform splat budgets.
 *
 * Mirrors `astel_lod.budgets.PLATFORM_BUDGETS` on the Python side.
 * All counts are maximum Gaussian splats that should comfortably render
 * at ≥60 fps on a representative mid-range device for each target.
 */
export const PLATFORM_BUDGETS: Record<string, number> = {
  mobile: 100_000,
  web: 500_000,
  console: 1_500_000,
  cinematic: 5_000_000,
};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function isLodTier(v: unknown): v is LodTier {
  if (typeof v !== "object" || v === null) return false;
  const obj = v as Record<string, unknown>;
  return (
    typeof obj["name"] === "string" &&
    typeof obj["count"] === "number" &&
    typeof obj["file"] === "string"
  );
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Parse and validate a raw JSON value as an `LodDescriptor`.
 *
 * Throws a descriptive `Error` if:
 * - `json.schema` is not `"astel.lod/v0"`.
 * - `tiers` is missing, not an array, or empty.
 * - Any tier is missing `name`, `count`, or `file`.
 * - `count` values are not strictly ascending (mirrors the Python
 *   `build_lod_descriptor` pre-condition check).
 */
export function parseLodDescriptor(json: unknown): LodDescriptor {
  if (typeof json !== "object" || json === null) {
    throw new Error("LOD descriptor must be a JSON object.");
  }
  const obj = json as Record<string, unknown>;

  if (obj["schema"] !== "astel.lod/v0") {
    throw new Error(
      `LOD descriptor schema must be "astel.lod/v0", got: ${JSON.stringify(obj["schema"])}.`,
    );
  }

  const tiers = obj["tiers"];
  if (!Array.isArray(tiers) || tiers.length === 0) {
    throw new Error("LOD descriptor must have a non-empty 'tiers' array.");
  }

  for (let i = 0; i < tiers.length; i++) {
    const tier = tiers[i];
    if (!isLodTier(tier)) {
      throw new Error(
        `LOD tier at index ${i} is missing required fields (name, count, file).`,
      );
    }
    if (i > 0) {
      const prev = tiers[i - 1] as LodTier;
      if ((tier as LodTier).count <= prev.count) {
        throw new Error(
          `LOD tier counts must be strictly ascending; tier[${i}].count (${(tier as LodTier).count}) ` +
            `is not greater than tier[${i - 1}].count (${prev.count}).`,
        );
      }
    }
  }

  return { schema: "astel.lod/v0", tiers: tiers as LodTier[] };
}

/**
 * Return the LARGEST tier whose `count` does not exceed `maxSplats`.
 *
 * Honest fallback: if ALL tiers exceed the budget, returns the SMALLEST tier
 * rather than failing. The caller should check `tier.count > maxSplats` to
 * detect this case and warn the user that rendering may be over-budget.
 *
 * Precondition: `desc` must be a valid descriptor (produced by
 * {@link parseLodDescriptor} or the Python `build_lod_descriptor`).
 */
export function selectTierForBudget(
  desc: LodDescriptor,
  maxSplats: number,
): LodTier {
  // Tiers are ascending by count. Walk from the end to find the largest
  // tier that fits within the budget.
  for (let i = desc.tiers.length - 1; i >= 0; i--) {
    const tier = desc.tiers[i]!;
    if (tier.count <= maxSplats) return tier;
  }
  // No tier fits — honest fallback to the smallest available tier.
  return desc.tiers[0]!;
}

/**
 * Return the best-fit tier for a named platform.
 *
 * Delegates to {@link selectTierForBudget} using the budget from
 * {@link PLATFORM_BUDGETS}. Throws on unknown platform names.
 */
export function selectTierForPlatform(
  desc: LodDescriptor,
  platform: keyof typeof PLATFORM_BUDGETS,
): LodTier {
  const budget = PLATFORM_BUDGETS[platform];
  if (budget === undefined) {
    const valid = Object.keys(PLATFORM_BUDGETS).join(", ");
    throw new Error(
      `Unknown platform "${String(platform)}". Valid platforms: ${valid}.`,
    );
  }
  return selectTierForBudget(desc, budget);
}
