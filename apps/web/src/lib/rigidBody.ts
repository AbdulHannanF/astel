/**
 * A minimal single rigid-body integrator for the Physics Sandbox MVP.
 *
 * HONEST SCOPE (CLAUDE.md §8 feature 2): this is a *single rigid body* dropped
 * on a ground plane — gravity, a sphere–plane collision with restitution +
 * Coulomb friction, and tumbling induced by tangential contact. It is NOT the
 * MPM / PhysGaussian deformable simulation the full Physics Sandbox aspires to
 * (that runs server-side on the L5/L6 volume). It uses the asset's real L5 mass
 * (volume → mass via L6 density) + L6 restitution/friction so the *behaviour*
 * (how fast it falls, how high it bounces, when it rests) is driven by the
 * asset's measured/estimated physical properties, not faked.
 */

export type Vec3 = [number, number, number];

export interface Material {
  id: string;
  label: string;
  /** kg/m³ — drives mass = density · volume. */
  density: number;
  /** [0,1] bounciness. */
  restitution: number;
  /** Coulomb friction coefficient. */
  friction: number;
}

const RUBBER: Material = { id: "rubber", label: "Rubber", density: 1100, restitution: 0.8, friction: 0.9 };
const WOOD: Material = { id: "wood", label: "Wood", density: 700, restitution: 0.4, friction: 0.6 };
const STEEL: Material = { id: "steel", label: "Steel", density: 7850, restitution: 0.55, friction: 0.4 };
const STONE: Material = { id: "stone", label: "Stone", density: 2600, restitution: 0.2, friction: 0.8 };

/** A few defaults for when L6 doesn't name a material (honest fallbacks). */
export const MATERIALS: readonly Material[] = [RUBBER, WOOD, STEEL, STONE];

/** The default material (Wood) — a guaranteed non-undefined value. */
export const DEFAULT_MATERIAL: Material = WOOD;

/** Look up a material by id, falling back to {@link DEFAULT_MATERIAL}. */
export function materialById(id: string): Material {
  return MATERIALS.find((m) => m.id === id) ?? DEFAULT_MATERIAL;
}

export interface BodyConfig {
  /** Collision radius (model units) — the L5 bounding sphere of the asset. */
  radius: number;
  /** Mass (kg). Derived from L5 volume × L6 density upstream. */
  mass: number;
  restitution: number;
  friction: number;
  /** Floor plane height (model units). */
  floorY?: number;
  /** Gravity (m/s²), negative = down. */
  gravity?: number;
}

export interface BodyState {
  position: Vec3;
  velocity: Vec3;
  /** Angular velocity (rad/s) for visual tumbling. */
  angularVelocity: Vec3;
  /** Accumulated orientation (rad) about each axis, for rendering. */
  orientation: Vec3;
  resting: boolean;
}

const REST_SPEED = 0.05; // below this post-bounce speed, the body comes to rest.

export function createBody(position: Vec3): BodyState {
  return {
    position: [...position] as Vec3,
    velocity: [0, 0, 0],
    angularVelocity: [0, 0, 0],
    orientation: [0, 0, 0],
    resting: false,
  };
}

/**
 * Advance one fixed timestep (semi-implicit Euler). Returns the same state
 * object, mutated. Pure w.r.t. config; deterministic for a given input.
 */
export function step(state: BodyState, cfg: BodyConfig, dt: number): BodyState {
  const g = cfg.gravity ?? -9.81;
  const floorY = cfg.floorY ?? 0;

  if (!state.resting) {
    state.velocity[1] += g * dt;
    state.position[0] += state.velocity[0] * dt;
    state.position[1] += state.velocity[1] * dt;
    state.position[2] += state.velocity[2] * dt;
    state.orientation[0] += state.angularVelocity[0] * dt;
    state.orientation[1] += state.angularVelocity[1] * dt;
    state.orientation[2] += state.angularVelocity[2] * dt;
  }

  const bottom = state.position[1] - cfg.radius;
  if (bottom < floorY) {
    // Resolve penetration.
    state.position[1] = floorY + cfg.radius;

    if (state.velocity[1] < 0) {
      const impactSpeed = -state.velocity[1];
      state.velocity[1] = impactSpeed * cfg.restitution;

      // Coulomb friction on the tangential (x,z) velocity.
      const keep = Math.max(0, 1 - cfg.friction * dt * 10);
      state.velocity[0] *= keep;
      state.velocity[2] *= keep;

      // Tangential motion induces tumbling (rolling about the horizontal axes).
      state.angularVelocity[0] = -state.velocity[2] / cfg.radius;
      state.angularVelocity[2] = state.velocity[0] / cfg.radius;

      // Rest test: little vertical bounce and little horizontal drift.
      const horiz = Math.hypot(state.velocity[0], state.velocity[2]);
      if (state.velocity[1] < REST_SPEED && horiz < REST_SPEED) {
        state.velocity = [0, 0, 0];
        state.angularVelocity = [0, 0, 0];
        state.resting = true;
      }
    }
  }

  return state;
}

/** Apply an instantaneous impulse (N·s); wakes a resting body. */
export function applyImpulse(state: BodyState, cfg: BodyConfig, impulse: Vec3): void {
  state.velocity[0] += impulse[0] / cfg.mass;
  state.velocity[1] += impulse[1] / cfg.mass;
  state.velocity[2] += impulse[2] / cfg.mass;
  state.resting = false;
}

/** Mass (kg) from L5 volume (m³) and a material density (kg/m³). */
export function massFromVolume(volumeM3: number, density: number): number {
  return Math.max(1e-6, volumeM3 * density);
}
