/**
 * Real spherical harmonics (band 0–2) + Lambertian shading — the TS port of
 * `libs/astel_appearance/src/astel_appearance/sh.py` (and the `env.py` presets).
 *
 * The Relight Studio re-shades the L4 albedo payload (`l4-relight.json`) live
 * with these functions, proving the albedo/illumination split. Parity with the
 * Python implementation is locked by golden values in `sh.test.ts`.
 *
 * SH order: [Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22].
 */

export type Vec3 = [number, number, number];
/** 9 RGB SH radiance coefficients (an environment). */
export type EnvSH = Vec3[]; // length 9

const C0 = 0.28209479177387814;
const C1 = 0.4886025119029199;
const C2 = 1.0925484305920792;
const C3 = 0.31539156525252005;
const C4 = 0.5462742152960396;

/** Folded cosine-convolution constants Â_l = A_l/π, per SH coefficient. */
export const COSINE_CONV: readonly number[] = [
  1, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25,
];

function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** Evaluate the 9 real SH basis functions for a unit direction. */
export function shEvalL2(dir: Vec3): number[] {
  const [x, y, z] = normalize(dir);
  return [
    C0,
    -C1 * y,
    C1 * z,
    -C1 * x,
    C2 * x * y,
    -C2 * y * z,
    C3 * (3 * z * z - 1),
    -C2 * x * z,
    C4 * (x * x - y * y),
  ];
}

/**
 * Lambertian diffuse shading factor E(n)/π for an RGB SH environment, returned
 * as an RGB triple. The reflected colour of a surface is `albedo * shading`.
 */
export function diffuseShading(env: EnvSH, normal: Vec3): Vec3 {
  const basis = shEvalL2(normal);
  let r = 0;
  let g = 0;
  let b = 0;
  for (let k = 0; k < 9; k++) {
    const e = env[k] ?? [0, 0, 0];
    const w = (basis[k] ?? 0) * (COSINE_CONV[k] ?? 0);
    r += w * e[0];
    g += w * e[1];
    b += w * e[2];
  }
  return [r, g, b];
}

/** Rotate a direction about +Y by `angle` radians (env spin == rotate normal). */
export function yawRotate(v: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // R^{-1} n for a +Y rotation R (see decompose.relight_rgb).
  return [c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]];
}

/** Build an SH env = flat ambient + a single soft key light (mirrors env.py). */
export function directionalEnv(dir: Vec3, color: Vec3, ambient: number): EnvSH {
  const y = shEvalL2(dir);
  const dc = y[0] || 1;
  const env: EnvSH = [];
  for (let k = 0; k < 9; k++) {
    const yk = y[k] ?? 0;
    const base = k === 0 ? ambient / dc : 0;
    env.push([base + yk * color[0], base + yk * color[1], base + yk * color[2]]);
  }
  return env;
}

export interface EnvPreset {
  id: string;
  label: string;
  env: EnvSH;
}

const STUDIO_ENV: EnvPreset = {
  id: "studio",
  label: "Studio",
  env: directionalEnv([0.4, 0.8, 0.6], [1.1, 1.05, 1.0], 0.35),
};

/** The studio relight environments (parity with env.py `studio_presets`). */
export const ENV_PRESETS: readonly EnvPreset[] = [
  STUDIO_ENV,
  { id: "noon", label: "Noon", env: directionalEnv([0.0, 1.0, 0.1], [1.3, 1.28, 1.2], 0.45) },
  { id: "sunset", label: "Sunset", env: directionalEnv([0.9, 0.25, 0.3], [1.4, 0.7, 0.4], 0.18) },
  { id: "rim", label: "Rim", env: directionalEnv([-0.6, 0.3, -0.7], [0.9, 0.95, 1.2], 0.12) },
];

/** The default relight environment (Studio) — a guaranteed non-undefined value. */
export const DEFAULT_PRESET: EnvPreset = STUDIO_ENV;

/** Look up a preset by id, falling back to {@link DEFAULT_PRESET}. */
export function presetById(id: string): EnvPreset {
  return ENV_PRESETS.find((p) => p.id === id) ?? DEFAULT_PRESET;
}
