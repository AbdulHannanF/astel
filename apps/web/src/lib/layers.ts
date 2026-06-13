import type { LayerStage, ProgressEvent } from "./api.ts";

/**
 * The Astel Layer Stack (CLAUDE.md §3). Every asset is L0->L7. The viewer's
 * Layer Inspector renders this model; the stub sample only has L3 available, so
 * the rest carry honest "pending" / "locked" availability states.
 */

export type LayerId =
  | "L0"
  | "L1"
  | "L2"
  | "L3"
  | "L4"
  | "L5"
  | "L6"
  | "L7";

/** Availability of a layer for the currently loaded asset. */
export type LayerAvailability =
  | "available" // produced and viewable
  | "pending" // will be produced by a later stage / add-on
  | "locked"; // not applicable to this asset (e.g. L7 for a static object)

export interface LayerDef {
  id: LayerId;
  name: string;
  /** One-line description, instrument-caption tone. */
  blurb: string;
  /** The kind of data this layer carries (shown as a chip). */
  kind: string;
  availability: LayerAvailability;
}

/**
 * Static layer definitions for the checked-in sample asset. The sample is a
 * generated object with only the refined surface layer (L3) materialised; the
 * collision/physics/dynamics layers are honestly marked pending/locked.
 */
export const SAMPLE_LAYERS: readonly LayerDef[] = [
  {
    id: "L0",
    name: "Seed",
    blurb: "Sparse point cloud from conditioning",
    kind: "point cloud",
    availability: "pending",
  },
  {
    id: "L1",
    name: "Dense",
    blurb: "Metric-scaled cloud, normals + semantics",
    kind: "point cloud",
    availability: "pending",
  },
  {
    id: "L2",
    name: "Coarse",
    blurb: "Feed-forward gaussians from L1",
    kind: "gaussians",
    availability: "pending",
  },
  {
    id: "L3",
    name: "Refined",
    blurb: "Surface-aligned gaussians — the hero layer",
    kind: "gaussians",
    availability: "available",
  },
  {
    id: "L4",
    name: "Appearance",
    blurb: "Per-splat PBR + separated illumination",
    kind: "material",
    availability: "pending",
  },
  {
    id: "L5",
    name: "Collision",
    blurb: "SDF, convex proxies, mass properties",
    kind: "solidity",
    availability: "pending",
  },
  {
    id: "L6",
    name: "Physics",
    blurb: "Per-region material & semantic class",
    kind: "semantics",
    availability: "pending",
  },
  {
    id: "L7",
    name: "Dynamics",
    blurb: "Deformation field for dynamic capture",
    kind: "4D",
    availability: "locked",
  },
] as const;

/** Pipeline order of the layers a live generation can produce (L0-L3). */
const STAGE_LAYER_ORDER: readonly LayerId[] = ["L0", "L1", "L2", "L3"];

const STAGE_TO_LAYER: Record<LayerStage, LayerId> = {
  L0_SEED: "L0",
  L1_DENSE: "L1",
  L2_COARSE: "L2",
  L3_REFINED: "L3",
};

/**
 * Derive Layer Stack availability from the live SSE progress event. As the
 * pipeline advances through L0_SEED -> L3_REFINED, mark the active layer
 * "available" (running, shown as the inspectable preview tier) and earlier
 * layers "available" too; later layers and L4-L6 stay "pending", L7 "locked".
 * On a succeeded terminal event, L0-L3 are all "available".
 */
export function liveLayers(
  last: ProgressEvent | null,
  succeeded: boolean,
): readonly LayerDef[] {
  const activeIndex = succeeded
    ? STAGE_LAYER_ORDER.length - 1
    : last?.stage
      ? STAGE_LAYER_ORDER.indexOf(STAGE_TO_LAYER[last.stage])
      : -1;

  return SAMPLE_LAYERS.map((layer) => {
    const idx = STAGE_LAYER_ORDER.indexOf(layer.id);
    if (idx === -1) return layer; // L4-L7 untouched: pending/locked as defined
    const availability: LayerAvailability =
      idx <= activeIndex ? "available" : "pending";
    return { ...layer, availability };
  });
}
