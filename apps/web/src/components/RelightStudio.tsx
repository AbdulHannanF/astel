import { useEffect, useMemo, useRef, useState } from "react";

import {
  fetchRelightPayload,
  type RelightMode,
  type RelightPayload,
} from "../lib/relight.ts";
import { DEFAULT_PRESET, ENV_PRESETS, presetById } from "../lib/sh.ts";
import { RelightScene, type RelightView } from "../studio/RelightScene.ts";
import "./RelightStudio.css";

interface RelightStudioProps {
  /** URL of the `l4-relight.json` payload (sample or per-task artifact). */
  relightUrl: string;
}

type Status =
  | { kind: "loading" }
  | { kind: "ready"; payload: RelightPayload }
  | { kind: "error"; message: string };

/**
 * Relight Studio (CLAUDE.md §8 feature 3): rotate the environment around the
 * asset to prove the L4 albedo/illumination split. Re-shades the L4 albedo
 * preview live with the same SH math the pipeline uses.
 */
export function RelightStudio({ relightUrl }: RelightStudioProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<RelightScene | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [mode, setMode] = useState<RelightMode>("relit");
  const [presetId, setPresetId] = useState<string>("studio");
  const [yaw, setYaw] = useState(0);

  const preset = useMemo(() => presetById(presetId), [presetId]);

  const view: RelightView = useMemo(
    () => ({ mode, env: preset.env, yaw }),
    [mode, preset, yaw],
  );

  // Mount the scene once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const scene = new RelightScene(el);
    sceneRef.current = scene;
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Fetch the payload whenever the URL changes.
  useEffect(() => {
    const ctrl = new AbortController();
    void Promise.resolve().then(() => setStatus({ kind: "loading" }));
    fetchRelightPayload(relightUrl, ctrl.signal)
      .then((payload) => {
        setStatus({ kind: "ready", payload });
        sceneRef.current?.load(payload, {
          mode: "relit",
          env: DEFAULT_PRESET.env,
          yaw: 0,
        });
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => ctrl.abort();
  }, [relightUrl]);

  // Push view changes into the scene.
  useEffect(() => {
    if (status.kind === "ready") sceneRef.current?.setView(view);
  }, [view, status.kind]);

  const payload = status.kind === "ready" ? status.payload : null;
  const confidence = payload ? Math.round(payload.lighting_confidence * 100) : null;

  return (
    <div className="relight">
      <div className="relight-canvas" ref={containerRef} />

      <div className="relight-overlay relight-top">
        <span className="relight-chip">
          Relight Studio · <b>L4</b> appearance
        </span>
        {payload && (
          <span className="relight-chip mono" title="opacity-weighted R² of the SH-L2 lighting fit">
            lighting est. {confidence}%{payload.downsampled ? ` · ${payload.count.toLocaleString()} pts` : ""}
          </span>
        )}
      </div>

      <div className="relight-panel">
        <div className="relight-group">
          <span className="relight-label">Layer</span>
          <div className="relight-seg">
            {(["albedo", "estimated", "relit"] as RelightMode[]).map((m) => (
              <button
                key={m}
                type="button"
                className={m === mode ? "active" : ""}
                onClick={() => setMode(m)}
              >
                {m === "albedo" ? "Albedo" : m === "estimated" ? "As-captured" : "Relit"}
              </button>
            ))}
          </div>
        </div>

        <div className="relight-group">
          <span className="relight-label">Environment</span>
          <div className="relight-seg">
            {ENV_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                disabled={mode !== "relit"}
                className={p.id === presetId ? "active" : ""}
                onClick={() => setPresetId(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="relight-group">
          <span className="relight-label">Rotate {Math.round((yaw * 180) / Math.PI)}°</span>
          <input
            type="range"
            min={0}
            max={Math.PI * 2}
            step={0.01}
            value={yaw}
            disabled={mode === "albedo"}
            onChange={(e) => setYaw(Number(e.target.value))}
          />
        </div>

        <p className="relight-note">
          {mode === "albedo"
            ? "Un-lit per-splat albedo — illumination removed."
            : mode === "estimated"
              ? "Albedo re-lit by the environment estimated from the asset (reproduces the captured look)."
              : "Albedo re-lit by a swapped studio environment — drag to spin the light."}
        </p>
      </div>

      {status.kind === "loading" && (
        <div className="relight-overlay relight-center">
          <div className="spinner" /> <span>Loading appearance layer…</span>
        </div>
      )}
      {status.kind === "error" && (
        <div className="relight-overlay relight-center">
          <div className="relight-error">
            <h3>No L4 appearance layer</h3>
            <p>This asset hasn’t produced an <span className="mono">l4-relight.json</span> yet.</p>
            <p className="mono">{status.message}</p>
          </div>
        </div>
      )}
    </div>
  );
}
