import { describe, expect, it } from "vitest";

import {
  type LodDescriptor,
  type LodTier,
  parseLodDescriptor,
  PLATFORM_BUDGETS,
  selectTierForBudget,
  selectTierForPlatform,
} from "./lod.ts";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** A valid three-tier descriptor (100k / 500k / 1M). */
const THREE_TIER: LodDescriptor = {
  schema: "astel.lod/v0",
  tiers: [
    { name: "mobile", count: 100_000, file: "lod0.ply" },
    { name: "web", count: 500_000, file: "lod1.ply" },
    { name: "cinematic", count: 1_000_000, file: "lod2.ply" },
  ],
};

// ---------------------------------------------------------------------------
// parseLodDescriptor
// ---------------------------------------------------------------------------

describe("parseLodDescriptor", () => {
  it("accepts a valid descriptor and returns a typed LodDescriptor", () => {
    const raw: unknown = {
      schema: "astel.lod/v0",
      tiers: [
        { name: "mobile", count: 100_000, file: "lod0.ply" },
        { name: "web", count: 500_000, file: "lod1.ply" },
      ],
    };
    const desc = parseLodDescriptor(raw);
    expect(desc.schema).toBe("astel.lod/v0");
    expect(desc.tiers).toHaveLength(2);
    expect(desc.tiers[0]!.count).toBe(100_000);
    expect(desc.tiers[1]!.name).toBe("web");
  });

  it("rejects a wrong schema string", () => {
    const raw: unknown = {
      schema: "astel.lod/v1",
      tiers: [{ name: "web", count: 500_000, file: "lod0.ply" }],
    };
    expect(() => parseLodDescriptor(raw)).toThrow(/"astel.lod\/v0"/);
  });

  it("rejects a missing schema field", () => {
    const raw: unknown = {
      tiers: [{ name: "web", count: 500_000, file: "lod0.ply" }],
    };
    expect(() => parseLodDescriptor(raw)).toThrow(/"astel.lod\/v0"/);
  });

  it("rejects an empty tiers array", () => {
    const raw: unknown = { schema: "astel.lod/v0", tiers: [] };
    expect(() => parseLodDescriptor(raw)).toThrow(/non-empty/);
  });

  it("rejects a missing tiers field", () => {
    const raw: unknown = { schema: "astel.lod/v0" };
    expect(() => parseLodDescriptor(raw)).toThrow(/non-empty/);
  });

  it("rejects non-ascending counts (equal counts)", () => {
    const raw: unknown = {
      schema: "astel.lod/v0",
      tiers: [
        { name: "a", count: 100_000, file: "a.ply" },
        { name: "b", count: 100_000, file: "b.ply" },
      ],
    };
    expect(() => parseLodDescriptor(raw)).toThrow(/strictly ascending/);
  });

  it("rejects non-ascending counts (descending)", () => {
    const raw: unknown = {
      schema: "astel.lod/v0",
      tiers: [
        { name: "a", count: 500_000, file: "a.ply" },
        { name: "b", count: 100_000, file: "b.ply" },
      ],
    };
    expect(() => parseLodDescriptor(raw)).toThrow(/strictly ascending/);
  });

  it("rejects a tier missing a required field", () => {
    const raw: unknown = {
      schema: "astel.lod/v0",
      tiers: [{ name: "web", count: 500_000 }], // missing 'file'
    };
    expect(() => parseLodDescriptor(raw)).toThrow(/missing required fields/);
  });

  it("rejects non-object input", () => {
    expect(() => parseLodDescriptor(null)).toThrow();
    expect(() => parseLodDescriptor("string")).toThrow();
    expect(() => parseLodDescriptor(42)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// selectTierForBudget
// ---------------------------------------------------------------------------

describe("selectTierForBudget", () => {
  it("returns the 500k tier for a 600k budget", () => {
    const tier: LodTier = selectTierForBudget(THREE_TIER, 600_000);
    expect(tier.count).toBe(500_000);
    expect(tier.name).toBe("web");
  });

  it("returns the 1M tier for a 2M budget (largest fits)", () => {
    const tier: LodTier = selectTierForBudget(THREE_TIER, 2_000_000);
    expect(tier.count).toBe(1_000_000);
  });

  it("returns the 100k tier (smallest fallback) for a 50k budget below all tiers", () => {
    const tier: LodTier = selectTierForBudget(THREE_TIER, 50_000);
    expect(tier.count).toBe(100_000);
    expect(tier.name).toBe("mobile");
  });

  it("returns the exact-match tier when budget equals a tier count", () => {
    const tier: LodTier = selectTierForBudget(THREE_TIER, 500_000);
    expect(tier.count).toBe(500_000);
  });

  it("single-tier descriptor always returns that tier regardless of budget", () => {
    const single: LodDescriptor = {
      schema: "astel.lod/v0",
      tiers: [{ name: "only", count: 999_000, file: "only.ply" }],
    };
    expect(selectTierForBudget(single, 1).count).toBe(999_000);
    expect(selectTierForBudget(single, 10_000_000).count).toBe(999_000);
  });
});

// ---------------------------------------------------------------------------
// selectTierForPlatform
// ---------------------------------------------------------------------------

describe("selectTierForPlatform", () => {
  it('"mobile" (100k budget) selects the 100k tier', () => {
    const tier: LodTier = selectTierForPlatform(THREE_TIER, "mobile");
    expect(tier.count).toBe(100_000);
    expect(tier.name).toBe("mobile");
  });

  it('"web" (500k budget) selects the 500k tier', () => {
    const tier: LodTier = selectTierForPlatform(THREE_TIER, "web");
    expect(tier.count).toBe(500_000);
  });

  it('"console" (1.5M budget) selects the 1M tier (largest that fits)', () => {
    const tier: LodTier = selectTierForPlatform(THREE_TIER, "console");
    expect(tier.count).toBe(1_000_000);
  });

  it('"cinematic" (5M budget) selects the 1M tier (largest available)', () => {
    const tier: LodTier = selectTierForPlatform(THREE_TIER, "cinematic");
    expect(tier.count).toBe(1_000_000);
  });

  it("throws on an unknown platform name and lists valid keys", () => {
    expect(() =>
      selectTierForPlatform(THREE_TIER, "holodeck" as keyof typeof PLATFORM_BUDGETS),
    ).toThrow(/holodeck/);
  });

  it("PLATFORM_BUDGETS has the four canonical entries", () => {
    expect(PLATFORM_BUDGETS["mobile"]).toBe(100_000);
    expect(PLATFORM_BUDGETS["web"]).toBe(500_000);
    expect(PLATFORM_BUDGETS["console"]).toBe(1_500_000);
    expect(PLATFORM_BUDGETS["cinematic"]).toBe(5_000_000);
  });
});
