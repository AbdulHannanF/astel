import { useEffect, useRef, useState } from "react";

import {
  DEFAULT_MATERIAL,
  MATERIALS,
  type Material,
  materialById,
} from "../lib/rigidBody.ts";
import { PhysicsScene, type PhysicsLoadInfo } from "../studio/PhysicsScene.ts";
import "./PhysicsSandbox.css";

interface PhysicsSandboxProps {
  /** Splat asset URL (sample or per-task `l3.ply`). */
  splatUrl: string;
  /** Optional `l5-mass.json` URL for the real solidified volume. */
  massPropsUrl?: string | undefined;
}

/**
 * Physics Sandbox (CLAUDE.md §8 feature 2): drop the asset on a floor and poke
 * it, so users *see* world-awareness. Mass comes from L5 volume × L6 material
 * density. Honest MVP — a single rigid body, not the MPM deformable sim.
 */
export function PhysicsSandbox({
  splatUrl,
  massPropsUrl,
}: PhysicsSandboxProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<PhysicsScene | null>(null);
  const [info, setInfo] = useState<PhysicsLoadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [materialId, setMaterialId] = useState<string>(DEFAULT_MATERIAL.id);

  // Mount once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const scene = new PhysicsScene(
      el,
      (i) => setInfo(i),
      (err) => setError(err.message),
    );
    sceneRef.current = scene;
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Load the asset (+ optional real L5 volume).
  useEffect(() => {
    const ctrl = new AbortController();
    void Promise.resolve().then(() => {
      setInfo(null);
      setError(null);
    });

    const loadVolume = massPropsUrl
      ? fetch(massPropsUrl, { signal: ctrl.signal })
          .then((r) => (r.ok ? (r.json() as Promise<{ volume?: number }>) : null))
          .then((j) => j?.volume)
          .catch(() => undefined)
      : Promise.resolve<number | undefined>(undefined);

    loadVolume.then((volume) => {
      if (ctrl.signal.aborted) return;
      const scene = sceneRef.current;
      if (!scene) return;
      scene.setMaterial(materialById(materialId));
      void scene.load(splatUrl, volume);
    });

    return () => ctrl.abort();
    // materialId intentionally excluded: changing it shouldn't reload geometry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splatUrl, massPropsUrl]);

  const selectMaterial = (m: Material): void => {
    setMaterialId(m.id);
    sceneRef.current?.setMaterial(m);
  };

  const mass = info
    ? Math.max(1e-6, info.volumeM3 * materialById(materialId).density)
    : null;

  return (
    <div className="physics">
      <div className="physics-canvas" ref={containerRef} />

      <div className="physics-overlay physics-top">
        <span className="physics-chip">
          Physics Sandbox · <b>L5/L6</b>
        </span>
        {info && (
          <span className="physics-chip mono" title="mass = L5 volume × L6 density">
            {mass! < 1
              ? `${(mass! * 1000).toFixed(0)} g`
              : `${mass!.toFixed(1)} kg`}{" "}
            · vol {info.volumeM3.toExponential(2)} m³
            {info.volumeSource === "bounding-sphere" ? " (est.)" : ""}
          </span>
        )}
      </div>

      <div className="physics-panel">
        <div className="physics-group">
          <span className="physics-label">Material (L6)</span>
          <div className="physics-seg">
            {MATERIALS.map((m) => (
              <button
                key={m.id}
                type="button"
                className={m.id === materialId ? "active" : ""}
                onClick={() => selectMaterial(m)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="physics-actions">
          <button type="button" className="physics-btn primary" onClick={() => sceneRef.current?.drop()}>
            Drop
          </button>
          <button type="button" className="physics-btn" onClick={() => sceneRef.current?.poke()}>
            Poke
          </button>
          <button type="button" className="physics-btn" onClick={() => sceneRef.current?.reset()}>
            Reset
          </button>
        </div>

        <p className="physics-note">
          Single rigid body — gravity, restitution &amp; friction from the L6
          material, mass from the L5 volume.{" "}
          {info?.volumeSource === "bounding-sphere"
            ? "Volume is a bounding-sphere estimate (no L5 solidify for this asset)."
            : "Volume is the measured L5 watertight volume."}
        </p>
      </div>

      {error && (
        <div className="physics-overlay physics-center">
          <div className="physics-error">
            <h3>Couldn’t load asset</h3>
            <p className="mono">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
