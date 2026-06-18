import { describe, expect, it } from "vitest";

import { computeColors, estimatedEnv, type RelightPayload } from "./relight.ts";
import { presetById } from "./sh.ts";

const payload: RelightPayload = {
  schema: "astel.l4-relight-preview/v0",
  count: 3,
  total: 3,
  downsampled: false,
  lighting_confidence: 0.5,
  center: [0, 0, 0],
  radius: 1,
  // A bright-ish grayscale environment (flat DC + a small key term).
  env_estimated: {
    sh_rgb: Array.from({ length: 9 }, (_, k) =>
      k === 0 ? [3.5, 3.5, 3.5] : [0.1, 0.1, 0.1],
    ),
  },
  positions: [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
  ],
  normals: [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ],
  albedo: [
    [0.4, 0.2, 0.1],
    [0.1, 0.5, 0.3],
    [0.6, 0.6, 0.6],
  ],
  notes: [],
};

describe("relight recolouring", () => {
  it("albedo mode returns the un-lit albedo", () => {
    const c = computeColors(payload, "albedo", estimatedEnv(payload), 0);
    expect(c[0]!).toBeCloseTo(0.4, 6);
    expect(c[1]!).toBeCloseTo(0.2, 6);
    expect(c[2]!).toBeCloseTo(0.1, 6);
  });

  it("relit mode applies shading (differs from raw albedo)", () => {
    const lit = computeColors(payload, "relit", presetById("studio").env, 0);
    const raw = computeColors(payload, "albedo", estimatedEnv(payload), 0);
    let diff = 0;
    for (let i = 0; i < lit.length; i++) diff += Math.abs(lit[i]! - raw[i]!);
    expect(diff).toBeGreaterThan(0.01);
  });

  it("different environments produce different colours", () => {
    const studio = computeColors(payload, "relit", presetById("studio").env, 0);
    const sunset = computeColors(payload, "relit", presetById("sunset").env, 0);
    let diff = 0;
    for (let i = 0; i < studio.length; i++) diff += Math.abs(studio[i]! - sunset[i]!);
    expect(diff).toBeGreaterThan(0.05);
  });

  it("yaw rotation changes the lit result", () => {
    const a = computeColors(payload, "relit", presetById("sunset").env, 0);
    const b = computeColors(payload, "relit", presetById("sunset").env, Math.PI / 2);
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += Math.abs(a[i]! - b[i]!);
    expect(diff).toBeGreaterThan(0.001);
  });

  it("all colours stay in [0,1]", () => {
    const c = computeColors(payload, "relit", presetById("noon").env, 0.7);
    for (const v of c) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
