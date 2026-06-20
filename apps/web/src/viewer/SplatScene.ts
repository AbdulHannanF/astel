/**
 * Framework-agnostic Spark + three.js splat scene controller.
 *
 * Owns the WebGLRenderer, SparkRenderer, camera, OrbitControls, and the loaded
 * SplatMesh. React just mounts/unmounts it and forwards intent (visibility,
 * opacity, reframe). Kept out of React so the render loop and GPU resources
 * have a single, explicit lifecycle.
 */

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export interface SceneCallbacks {
  onLoaded?: (info: { numSplats: number }) => void;
  onError?: (err: Error) => void;
}

const BG_TOP = new THREE.Color("#161310"); // warm near-black, brass-tinted
const BG_BOTTOM = new THREE.Color("#0d0b08");

export class SplatScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly spark: SparkRenderer;
  private readonly clock = new THREE.Clock();
  private readonly resizeObserver: ResizeObserver;

  private splat: SplatMesh | null = null;
  private rafId = 0;
  private disposed = false;
  private autoSpin = true;

  constructor(
    private readonly container: HTMLElement,
    private readonly callbacks: SceneCallbacks = {},
  ) {
    // Spark guidance: antialias OFF (WebGL MSAA doesn't help splats, costs perf).
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(0x000000, 0);
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = "block";

    this.scene.background = this.makeGradientBackground();

    this.camera = new THREE.PerspectiveCamera(
      42,
      this.aspect(),
      0.01,
      100,
    );
    this.camera.position.set(2.6, 1.6, 3.4);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 12;
    this.controls.target.set(0, 0, 0);
    // Stop auto-spin as soon as the user grabs the model.
    this.controls.addEventListener("start", () => (this.autoSpin = false));

    // enable2DGS: our 2DGS L3 assets are flattened, surfel-style splats.
    // minPixelRadius floors each splat to ~1.5px on screen: real generated
    // assets (262k surfels normalised to a unit frame) have very small
    // per-splat scales (~0.002 of the object radius), so with the default
    // floor of 0 they rasterize sub-pixel and the whole asset reads as blank.
    // Flooring makes the dense surfel cloud visible without affecting the
    // already-larger sample splats. (CLAUDE.md: the viewer is the product demo.)
    this.spark = new SparkRenderer({
      renderer: this.renderer,
      enable2DGS: true,
      minPixelRadius: 2.0,
    });
    this.scene.add(this.spark);

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);

    this.loop();
  }

  /**
   * Load a splat asset from a URL (e.g. /samples/astel-sample.ply), replacing
   * any previously loaded splat. Safe to call repeatedly (e.g. idle sample ->
   * real per-task artifact on generation completion).
   */
  async load(url: string): Promise<void> {
    try {
      const mesh = new SplatMesh({ url });
      await mesh.initialized;
      if (this.disposed) {
        mesh.dispose();
        return;
      }
      // Our PLY is authored Y-up; flip 180deg about X to match Spark's
      // OpenGL viewing convention so the knot sits upright.
      mesh.quaternion.set(1, 0, 0, 0);
      if (this.splat) {
        this.scene.remove(this.splat);
        this.splat.dispose();
      }
      this.scene.add(mesh);
      this.splat = mesh;
      this.frameObject(mesh);

      const numSplats = mesh.packedSplats?.numSplats ?? 0;
      this.callbacks.onLoaded?.({ numSplats });
    } catch (err) {
      this.callbacks.onError?.(
        err instanceof Error ? err : new Error(String(err)),
      );
    }
  }

  setVisible(visible: boolean): void {
    if (this.splat) this.splat.visible = visible;
  }

  setOpacity(opacity: number): void {
    if (this.splat) this.splat.opacity = opacity;
  }

  resetView(): void {
    this.camera.position.set(2.6, 1.6, 3.4);
    this.controls.target.set(0, 0, 0);
    this.autoSpin = true;
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

  private frameObject(object: THREE.Object3D): void {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.001);
    this.controls.target.copy(sphere.center);
    const dist = radius / Math.sin((this.camera.fov * Math.PI) / 360);
    const dir = new THREE.Vector3(0.62, 0.42, 0.86).normalize();
    this.camera.position
      .copy(sphere.center)
      .add(dir.multiplyScalar(dist * 1.15));
    this.camera.near = radius / 80;
    this.camera.far = radius * 80;
    this.camera.updateProjectionMatrix();
  }

  private makeGradientBackground(): THREE.Texture {
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = 2;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const grad = ctx.createLinearGradient(0, 0, 0, size);
      grad.addColorStop(0, `#${BG_TOP.getHexString()}`);
      grad.addColorStop(1, `#${BG_BOTTOM.getHexString()}`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 2, size);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  private handleResize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = this.aspect();
    this.camera.updateProjectionMatrix();
  }

  private aspect(): number {
    return (
      this.container.clientWidth / Math.max(this.container.clientHeight, 1)
    );
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.rafId = requestAnimationFrame(this.loop);
    const dt = this.clock.getDelta();
    if (this.autoSpin && this.splat) {
      // Slow, cinematic idle rotation until the user interacts.
      const angle = dt * 0.12;
      this.camera.position.applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        angle,
      );
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
