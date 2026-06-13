import { useEffect, useRef, useState } from "react";

import { SplatScene } from "../viewer/SplatScene.ts";

interface ViewportProps {
  sampleUrl: string;
  /** URL to load. Defaults to {@link sampleUrl} when omitted. */
  splatUrl?: string;
  /** Falls back to this URL (the static sample) if {@link splatUrl} fails. */
  fallbackUrl?: string;
  splatVisible: boolean;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; numSplats: number }
  | { kind: "error"; message: string };

export function Viewport({
  sampleUrl,
  splatUrl,
  fallbackUrl,
  splatVisible,
}: ViewportProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SplatScene | null>(null);
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  const url = splatUrl ?? sampleUrl;

  // Create the scene once and tear it down on unmount.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const scene = new SplatScene(el, {
      onLoaded: ({ numSplats }) => setState({ kind: "ready", numSplats }),
      onError: (err) => setState({ kind: "error", message: err.message }),
    });
    sceneRef.current = scene;

    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // (Re)load whenever the target URL changes.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    Promise.resolve()
      .then(() => setState({ kind: "loading" }))
      .then(() => scene.load(url));
  }, [url]);

  // If loading the per-task artifact fails, fall back to the static sample
  // so a bad/missing artifact never leaves the viewer stuck on an error.
  useEffect(() => {
    const scene = sceneRef.current;
    const fallback = fallbackUrl ?? sampleUrl;
    if (!scene || state.kind !== "error" || url === fallback) return;

    Promise.resolve()
      .then(() => setState({ kind: "loading" }))
      .then(() => scene.load(fallback));
  }, [state.kind, url, fallbackUrl, sampleUrl]);

  // Reflect L3 visibility toggle from the Layer Inspector into the scene.
  useEffect(() => {
    sceneRef.current?.setVisible(splatVisible);
  }, [splatVisible]);

  const reset = () => sceneRef.current?.resetView();

  return (
    <div className="viewport-wrap">
      <div className="viewport-canvas" ref={containerRef} />

      <div className="viewport-hud">
        <div className="hud-row">
          <span className="hud-chip">
            <span className="swatch" />
            <span>
              Layer <b>L3</b> · Refined surface
            </span>
          </span>
          <div className="viewport-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={reset}
              title="Reset view"
              aria-label="Reset view"
            >
              <ResetIcon />
            </button>
          </div>
        </div>

        <div className="hud-row">
          <span className="hud-chip mono">
            {state.kind === "ready"
              ? `${state.numSplats.toLocaleString()} splats`
              : "—"}
          </span>
          <span className="hud-chip mono">drag · orbit / scroll · zoom</span>
        </div>
      </div>

      {state.kind === "loading" && (
        <div className="viewport-overlay">
          <div className="loading-card">
            <div className="spinner" />
            <span>Loading reference splat</span>
          </div>
        </div>
      )}

      {state.kind === "error" && (
        <div className="viewport-overlay">
          <div className="error-card">
            <h3>Viewport unavailable</h3>
            <p>
              Couldn&rsquo;t load the splat asset. This usually means WebGL2
              isn&rsquo;t available or the sample failed to fetch.
            </p>
            <p className="mono" style={{ color: "var(--text-faint)" }}>
              {state.message}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function ResetIcon(): React.JSX.Element {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M3 8a5 5 0 1 1 1.6 3.66"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <path
        d="M3 5v3h3"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
