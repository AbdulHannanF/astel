import { describe, expect, it } from "vitest";

import {
  COSINE_CONV,
  diffuseShading,
  directionalEnv,
  presetById,
  shEvalL2,
  yawRotate,
  type Vec3,
} from "./sh.ts";

// Golden values generated from astel_appearance (Python) — see sh.py / env.py.
const DIR: Vec3 = [0.3030457633656632, 0.5050762722761053, 0.8081220356417687];
const SH_EVAL = [
  0.282095, -0.246782, 0.39485, -0.148069, 0.167227, -0.445938, 0.302518,
  -0.267563, -0.089188,
];

function close(a: number, b: number, eps = 1e-5): void {
  expect(Math.abs(a - b)).toBeLessThan(eps);
}

describe("sh parity with astel_appearance", () => {
  it("evaluates the SH-L2 basis to match Python", () => {
    const y = shEvalL2(DIR);
    y.forEach((v, i) => close(v, SH_EVAL[i]!));
  });

  it("cosine-convolution constants match", () => {
    expect(COSINE_CONV[0]).toBe(1);
    close(COSINE_CONV[1]!, 2 / 3);
    expect(COSINE_CONV[4]).toBe(0.25);
  });

  it("studio preset shading matches Python golden", () => {
    const studio = presetById("studio");
    const n1: Vec3 = [0, 1, 0];
    const s1 = diffuseShading(studio.env, n1);
    [0.603418, 0.591899, 0.58038].forEach((v, i) => close(s1[i]!, v));

    const raw: Vec3 = [0.4, 0.2, -0.9];
    const len = Math.hypot(...raw);
    const n2: Vec3 = [raw[0] / len, raw[1] / len, raw[2] / len];
    const s2 = diffuseShading(studio.env, n2);
    [0.354023, 0.35384, 0.353657].forEach((v, i) => close(s2[i]!, v));
  });

  it("sunset preset shading matches Python golden", () => {
    const sunset = presetById("sunset");
    const s1 = diffuseShading(sunset.env, [0, 1, 0]);
    [0.292122, 0.236061, 0.212035].forEach((v, i) => close(s1[i]!, v));
  });

  it("flat ambient env shades uniformly", () => {
    const env = directionalEnv([0, 1, 0], [0, 0, 0], 0.4);
    for (const n of [[0, 0, 1], [1, 0, 0], [0, 1, 0]] as Vec3[]) {
      const s = diffuseShading(env, n);
      close(s[0], 0.4);
      close(s[1], 0.4);
      close(s[2], 0.4);
    }
  });

  it("yaw rotation preserves length", () => {
    const v: Vec3 = [0.3, 0.5, 0.8];
    const r = yawRotate(v, 0.9);
    close(Math.hypot(...r), Math.hypot(...v));
  });
});
