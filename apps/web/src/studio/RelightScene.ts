/**
 * Three.js point-cloud renderer for the Relight Studio.
 *
 * Renders the L4 relight-preview payload as a coloured point cloud whose colours
 * are recomputed on the CPU (the same SH math the Python L4 uses) whenever the
 * environment, rotation or mode changes — demonstrating that albedo and
 * illumination are genuinely separated. This is a *downsampled inspector* (the
 * full asset is the splat viewer), labelled as such in the UI.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import {
  computeColors,
  estimatedEnv,
  type RelightMode,
  type RelightPayload,
} from "../lib/relight.ts";
import { type EnvSH } from "../lib/sh.ts";

export interface RelightView {
  mode: RelightMode;
  env: EnvSH;
  yaw: number;
}

export class RelightScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;

  private points: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private payload: RelightPayload | null = null;
  private rafId = 0;
  private disposed = false;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = "block";

    this.camera = new THREE.PerspectiveCamera(45, this.aspect(), 0.01, 1000);
    this.camera.position.set(2.2, 1.4, 2.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);
    this.loop();
  }

  load(payload: RelightPayload, view: RelightView): void {
    this.payload = payload;
    const n = payload.count;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = payload.positions[i] ?? [0, 0, 0];
      positions[i * 3] = p[0] ?? 0;
      positions[i * 3 + 1] = p[1] ?? 0;
      positions[i * 3 + 2] = p[2] ?? 0;
    }
    const colors = computeColors(payload, view.mode, view.env, view.yaw);

    if (this.points) {
      this.scene.remove(this.points);
      this.geometry?.dispose();
      (this.points.material as THREE.Material).dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const size = Math.max(0.006, payload.radius * 0.012);
    const material = new THREE.PointsMaterial({
      size,
      vertexColors: true,
      sizeAttenuation: true,
    });
    const pts = new THREE.Points(geom, material);
    this.scene.add(pts);
    this.points = pts;
    this.geometry = geom;

    this.frame(payload);
  }

  /** Recompute per-point colours for a new environment/rotation/mode. */
  setView(view: RelightView): void {
    if (!this.payload || !this.geometry) return;
    const colors = computeColors(
      this.payload,
      view.mode,
      view.mode === "estimated" ? estimatedEnv(this.payload) : view.env,
      view.yaw,
    );
    const attr = this.geometry.getAttribute("color") as THREE.BufferAttribute;
    (attr.array as Float32Array).set(colors);
    attr.needsUpdate = true;
  }

  resetView(): void {
    if (this.payload) this.frame(this.payload);
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.rafId);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.geometry?.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }

  private frame(payload: RelightPayload): void {
    const c = payload.center;
    this.controls.target.set(c[0], c[1], c[2]);
    const r = Math.max(payload.radius, 0.01);
    const dist = r / Math.sin((this.camera.fov * Math.PI) / 360);
    const dir = new THREE.Vector3(0.6, 0.45, 0.85).normalize();
    this.camera.position
      .set(c[0], c[1], c[2])
      .add(dir.multiplyScalar(dist * 1.25));
    this.camera.near = r / 100;
    this.camera.far = r * 100;
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
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
