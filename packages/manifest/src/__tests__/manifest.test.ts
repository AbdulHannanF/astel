import { describe, expect, it } from "vitest";

import { parseManifest, serializeManifest, validatePaths } from "../index.js";
import type { Manifest } from "../index.js";
import { makeFixtureManifest } from "./fixtures.js";

describe("round-trip", () => {
  it("parses, validates, serializes, and re-parses a known-good manifest", () => {
    const original = makeFixtureManifest();

    const text = serializeManifest(original);
    const firstParse = parseManifest(text);
    expect(firstParse.ok).toBe(true);
    if (!firstParse.ok) {
      throw new Error("expected ok");
    }

    const reSerialized = serializeManifest(firstParse.manifest);
    const secondParse = parseManifest(reSerialized);
    expect(secondParse.ok).toBe(true);
    if (!secondParse.ok) {
      throw new Error("expected ok");
    }

    // Deep equality across the full round-trip, including unknown/extensions/extras keys.
    expect(secondParse.manifest).toEqual(original);
    expect(text).toEqual(reSerialized);

    // Explicitly assert forward-compat blocks survived.
    expect(secondParse.manifest.extensions).toEqual({
      vendor_example: { note: "preserved on round-trip", value: 123 },
    });
    expect(secondParse.manifest.extras).toEqual({ internal_tag: "fixture" });

    // Honest-null quality report field survives with its reason.
    expect(secondParse.manifest.quality_report.geometric_error.chamfer_mm).toBeNull();
    expect(secondParse.manifest.quality_report.geometric_error.reason).toBe(
      "pure text-to-3D input has no measured L1 reference",
    );
  });

  it("accepts an already-parsed object (not just a JSON string)", () => {
    const original = makeFixtureManifest();
    const result = parseManifest(original);
    expect(result.ok).toBe(true);
  });
});

describe("negative: schema validation", () => {
  it("rejects a manifest missing a required top-level key with a useful message", () => {
    const broken = makeFixtureManifest() as unknown as Record<string, unknown>;
    delete broken["quality_report"];

    const result = parseManifest(broken);
    expect(result.ok).toBe(false);
    if (result.ok) {
      throw new Error("expected failure");
    }
    expect(result.errors.length).toBeGreaterThan(0);
    const messages = result.errors.map((e) => e.message);
    expect(messages.some((m) => m.includes("quality_report"))).toBe(true);
    expect(result.errors.some((e) => e.keyword === "required")).toBe(true);
  });

  it("rejects malformed JSON text", () => {
    const result = parseManifest("{ not json");
    expect(result.ok).toBe(false);
    if (result.ok) {
      throw new Error("expected failure");
    }
    expect(result.errors[0]?.keyword).toBe("json");
  });

  it("rejects a layer entry with an invalid kind enum value", () => {
    const broken = makeFixtureManifest();
    // @ts-expect-error -- intentionally invalid for the negative test
    broken.layers.l0!.kind = "not_a_real_kind";

    const result = parseManifest(broken);
    expect(result.ok).toBe(false);
  });
});

describe("negative: validatePaths", () => {
  it("rejects a file path containing '..' traversal", () => {
    const manifest = makeFixtureManifest();
    manifest.layers.l0!.files![0]!.path = "../escape/points.ply";

    const issues = validatePaths(manifest);
    expect(issues.length).toBeGreaterThan(0);
    expect(issues[0]).toMatchObject({
      location: "layers.l0.files[0].path",
      path: "../escape/points.ply",
      reason: "traversal",
    });
  });

  it("rejects an absolute file path", () => {
    const manifest = makeFixtureManifest();
    manifest.layers.l3!.files![0]!.path = "/etc/passwd";

    const issues = validatePaths(manifest);
    expect(issues.some((i) => i.reason === "absolute")).toBe(true);
  });

  it("rejects a Windows-style absolute path", () => {
    const manifest = makeFixtureManifest();
    manifest.layers.l3!.files![0]!.path = "C:\\Windows\\System32\\evil.ply";

    const issues = validatePaths(manifest);
    expect(issues.some((i) => i.reason === "absolute")).toBe(true);
  });

  it("passes for a well-formed manifest", () => {
    const manifest = makeFixtureManifest();
    const issues = validatePaths(manifest);
    expect(issues).toEqual([]);
  });
});

describe("type-level sanity", () => {
  it("Manifest type accepts the fixture shape (compile-time check via tsc --noEmit)", () => {
    const m: Manifest = makeFixtureManifest();
    expect(m.format_version).toBe("0.1.0");
    expect(m.layers.l0?.kind).toBe("seed_pointcloud");
    expect(m.layers.l3?.kind).toBe("refined_gaussians");
  });
});
