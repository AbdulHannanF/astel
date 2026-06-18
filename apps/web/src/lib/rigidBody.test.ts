import { describe, expect, it } from "vitest";

import {
  applyImpulse,
  type BodyConfig,
  createBody,
  massFromVolume,
  step,
  type Vec3,
} from "./rigidBody.ts";

const cfg = (over: Partial<BodyConfig> = {}): BodyConfig => ({
  radius: 0.5,
  mass: 1,
  restitution: 0.6,
  friction: 0.5,
  floorY: 0,
  gravity: -9.81,
  ...over,
});

function simulate(c: BodyConfig, start: Vec3, steps: number): ReturnType<typeof createBody> {
  const body = createBody(start);
  for (let i = 0; i < steps; i++) step(body, c, 1 / 120);
  return body;
}

describe("rigid body", () => {
  it("falls under gravity", () => {
    const body = createBody([0, 5, 0]);
    const c = cfg();
    step(body, c, 1 / 60);
    expect(body.velocity[1]).toBeLessThan(0);
    expect(body.position[1]).toBeLessThan(5);
  });

  it("never sinks below the floor", () => {
    const body = simulate(cfg(), [0, 5, 0], 2000);
    expect(body.position[1]).toBeGreaterThanOrEqual(cfg().radius - 1e-6);
  });

  it("comes to rest on the floor", () => {
    const body = simulate(cfg(), [0, 5, 0], 3000);
    expect(body.resting).toBe(true);
    expect(body.position[1]).toBeCloseTo(cfg().radius, 2);
    expect(Math.hypot(...body.velocity)).toBeLessThan(1e-6);
  });

  it("a bouncier material rests later (more bounces)", () => {
    const lowSteps = firstRestStep(cfg({ restitution: 0.1 }));
    const highSteps = firstRestStep(cfg({ restitution: 0.85 }));
    expect(highSteps).toBeGreaterThan(lowSteps);
  });

  it("impulse wakes a resting body and moves it", () => {
    const body = simulate(cfg(), [0, 5, 0], 3000);
    expect(body.resting).toBe(true);
    applyImpulse(body, cfg(), [2, 5, 0]);
    expect(body.resting).toBe(false);
    expect(body.velocity[0]).toBeCloseTo(2, 5);
    expect(body.velocity[1]).toBeCloseTo(5, 5);
  });

  it("mass scales with density (steel > wood)", () => {
    const vol = 0.001;
    expect(massFromVolume(vol, 7850)).toBeGreaterThan(massFromVolume(vol, 700));
  });
});

function firstRestStep(c: BodyConfig): number {
  const body = createBody([0, 5, 0]);
  for (let i = 0; i < 5000; i++) {
    step(body, c, 1 / 120);
    if (body.resting) return i;
  }
  return 5000;
}
