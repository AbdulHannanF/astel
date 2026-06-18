/**
 * Spark + three.js Physics Sandbox controller.
 *
 * Renders the asset's splats and drives them with the single rigid-body
 * integrator (`lib/rigidBody.ts`): drop on a ground plane under gravity, bounce
 * with restitution, tumble with friction, poke with an impulse. Mass comes from
 * the asset's L5 volume × the selected L6 material density, so heavier materials
 * respond differently to a poke. Honest MVP scope — a single rigid body, not the
 * MPM deformable sim the full sandbox aspires to (documented in the UI).
 */

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import {
  applyImpulse,
  type BodyConfig,
  type BodyState,
  createBody,
  type Material,
  massFromVolume,
  step,
} from "../lib/rigidBody.ts";

const FIXED_DT = 1 / 120;
const DROP_HEIGHT = 2.2;

export interface PhysicsLoadInfo {
  numSplats: number;
  radius: number;
  volumeM3: number;
  volumeSource: "l5" | "bounding-sphere";
}

export class PhysicsScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly spark: SparkRenderer;
  private readonly clock = new THREE.Clock();
  private readonly resizeObserver: ResizeObserver;
  private readonly pivot = new THREE.Group();

  private splat: SplatMesh | null = null;
  private body: BodyState = createBody([0, DROP_HEIGHT, 0]);
  private cfg: BodyConfig = {
    radius: 1,
    mass: 1,
    restitution: 0.5,
    friction: 0.6,
    floorY: 0,
    gravity: -9.81,
  };
  private radius = 1;
  private volumeM3 = 1;
  private accumulator = 0;
  private running = false;
  private rafId = 0;
  private disposed = false;

  constructor(
    private readonly container: HTMLElement,
    private readonly onLoaded?: (info: PhysicsLoadInfo) => void,
    private readonly onError?: (err: Error) => void,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(0x000000, 0);
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = "block";

    this.camera = new THREE.PerspectiveCamera(45, this.aspect(), 0.01, 1000);
    this.camera.position.set(4, 2.6, 5);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.scene.add(this.makeGround());
    this.scene.add(this.pivot);

    this.spark = new SparkRenderer({ renderer: this.renderer, enable2DGS: true });
    this.scene.add(this.spark);

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);
    this.loop();
  }

  /**
   * Load a splat asset. `volumeM3` (from L5 mass-props) is used when given;
   * otherwise the bounding-sphere volume is a documented proxy.
   */
  async load(url: string, volumeM3?: number): Promise<void> {
    try {
      const mesh = new SplatMesh({ url });
      await mesh.initialized;
      if (this.disposed) {
        mesh.dispose();
        return;
      }
      mesh.quaternion.set(1, 0, 0, 0); // Spark OpenGL-convention flip.

      // Centre the splat at the pivot origin so physics rotation is about its
      // centroid; size the collision sphere from its bounding sphere.
      const box = new THREE.Box3().setFromObject(mesh);
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      const r = Math.max(sphere.radius, 0.05);
      mesh.position.sub(sphere.center);

      if (this.splat) {
        this.pivot.remove(this.splat);
        this.splat.dispose();
      }
      this.pivot.add(mesh);
      this.splat = mesh;

      this.radius = r;
      const volume = volumeM3 && volumeM3 > 0 ? volumeM3 : (4 / 3) * Math.PI * r ** 3;
      this.volumeM3 = volume;
      this.resetBody();
      this.frame(r);

      this.onLoaded?.({
        numSplats: mesh.packedSplats?.numSplats ?? 0,
        radius: r,
        volumeM3: volume,
        volumeSource: volumeM3 && volumeM3 > 0 ? "l5" : "bounding-sphere",
      });
    } catch (err) {
      this.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  }

  setMaterial(material: Material): void {
    this.cfg = {
      ...this.cfg,
      radius: this.radius,
      restitution: material.restitution,
      friction: material.friction,
      mass: massFromVolume(this.volumeM3, material.density),
    };
  }

  drop(): void {
    this.resetBody();
    this.running = true;
  }

  poke(): void {
    // A sideways + upward nudge, scaled so heavy materials barely budge.
    const dir = new THREE.Vector3(
      Math.cos(this.camera.rotation.y),
      0.55,
      Math.sin(this.camera.rotation.y),
    );
    const strength = 4 * Math.sqrt(this.cfg.mass);
    applyImpulse(this.body, this.cfg, [
      dir.x * strength,
      dir.y * strength,
      dir.z * strength,
    ]);
    this.running = true;
  }

  reset(): void {
    this.resetBody();
    this.running = false;
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.rafId);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.splat?.dispose();
    this.spark.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }

  private resetBody(): void {
    this.body = createBody([0, this.radius + DROP_HEIGHT, 0]);
    this.cfg = { ...this.cfg, radius: this.radius };
  }

  private makeGround(): THREE.Object3D {
    const grid = new THREE.GridHelper(20, 40, 0x7a6a4a, 0x352f24);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    return grid;
  }

  private frame(r: number): void {
    this.controls.target.set(0, r, 0);
    const dist = (r * 3.5) / Math.sin((this.camera.fov * Math.PI) / 360);
    this.camera.position.set(dist * 0.7, r + dist * 0.5, dist);
    this.camera.near = r / 100;
    this.camera.far = r * 200;
    this.camera.updateProjectionMatrix();
  }

  private handleResize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = this.aspect();
    this.camera.updateProjectionMatrix();
  }

  private aspect(): number {
    return this.container.clientWidth / Math.max(this.container.clientHeight, 1);
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.rafId = requestAnimationFrame(this.loop);

    if (this.running) {
      this.accumulator += Math.min(this.clock.getDelta(), 0.05);
      while (this.accumulator >= FIXED_DT) {
        step(this.body, this.cfg, FIXED_DT);
        this.accumulator -= FIXED_DT;
      }
      if (this.body.resting) this.running = false;
    } else {
      this.clock.getDelta(); // keep the clock current
    }

    const p = this.body.position;
    this.pivot.position.set(p[0], p[1], p[2]);
    const o = this.body.orientation;
    this.pivot.rotation.set(o[0], o[1], o[2]);

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
