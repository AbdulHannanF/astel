/**
 * L4 relight-preview data (`l4-relight.json`) + per-point recolouring.
 *
 * The Relight Studio fetches a downsampled `{position, normal, albedo}` payload
 * produced by `astel_appearance.webdata.relight_payload` and re-shades it with
 * the SH math in `sh.ts`, proving the albedo/illumination split.
 */

import { diffuseShading, type EnvSH, type Vec3, yawRotate } from "./sh.ts";

export interface RelightPayload {
  schema: string;
  count: number;
  total: number;
  downsampled: boolean;
  lighting_confidence: number;
  center: Vec3;
  radius: number;
  env_estimated: { sh_rgb: number[][]; name?: string };
  positions: number[][];
  normals: number[][];
  albedo: number[][];
  notes: string[];
}

export type RelightMode = "albedo" | "estimated" | "relit";

export async function fetchRelightPayload(
  url: string,
  signal?: AbortSignal,
): Promise<RelightPayload> {
  const res = await fetch(url, { signal: signal ?? null });
  if (!res.ok) throw new Error(`relight ${res.status}`);
  return (await res.json()) as RelightPayload;
}

/** The estimated environment as an `EnvSH` (9 RGB triples). */
export function estimatedEnv(payload: RelightPayload): EnvSH {
  return payload.env_estimated.sh_rgb.map(
    (c) => [c[0] ?? 0, c[1] ?? 0, c[2] ?? 0] as Vec3,
  );
}

/**
 * Compute the flat Float32 RGB array (3·N) for the given mode/env/rotation.
 *
 * - `albedo`   — un-lit base colour (lighting removed).
 * - otherwise  — albedo × diffuseShading(env, R⁻¹·normal): the relit image.
 */
export function computeColors(
  payload: RelightPayload,
  mode: RelightMode,
  env: EnvSH,
  yaw: number,
): Float32Array {
  const n = payload.count;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const aRaw = payload.albedo[i] ?? [0, 0, 0];
    const a: Vec3 = [aRaw[0] ?? 0, aRaw[1] ?? 0, aRaw[2] ?? 0];
    if (mode === "albedo") {
      out[i * 3] = clamp01(a[0]);
      out[i * 3 + 1] = clamp01(a[1]);
      out[i * 3 + 2] = clamp01(a[2]);
      continue;
    }
    const nRaw = payload.normals[i] ?? [0, 1, 0];
    const nrm: Vec3 = [nRaw[0] ?? 0, nRaw[1] ?? 1, nRaw[2] ?? 0];
    const rotated = yaw !== 0 ? yawRotate(nrm, yaw) : nrm;
    const s = diffuseShading(env, rotated);
    out[i * 3] = clamp01(a[0] * s[0]);
    out[i * 3 + 1] = clamp01(a[1] * s[1]);
    out[i * 3 + 2] = clamp01(a[2] * s[2]);
  }
  return out;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
