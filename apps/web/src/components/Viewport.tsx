import { useEffect, useRef, useState } from "react";

import { SplatScene } from "../viewer/SplatScene.ts";

interface ViewportProps {
  sampleUrl: string;
  /** URL to load. Defaults to {@link sampleUrl} when omitted. */
  splatUrl?: string;
  /** Falls back to this URL (the static sample) if {@link splatUrl} fails. */
  fallbackUrl?: string;
  /**
   * True when the URL being shown is the bundled demo sample rather than a
   * real generated asset (i.e. the Studio is idle / no generation yet). Drives
   * an explicit "demo sample" badge so the placeholder torus is never mistaken
   * for the user's own result (CLAUDE.md §10.4: no silent substitution).
   */
  isSample?: boolean;
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
  isSample = false,
  splatVisible,
}: ViewportProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SplatScene | null>(null);
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  // Non-null once a real asset failed to load and we swapped in the static
  // sample: holds the underlying error so the HUD/console can say *why* rather
  // than passing the demo torus off as the user's asset. Reset on each fresh
  // target. (This is the path remote laptops hit when the box generates fine
  // but the generated splat can't be fetched/rendered client-side.)
  const [fallbackError, setFallbackError] = useState<string | null>(null);
  const fellBackToSample = fallbackError !== null;

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
      .then(() => {
        setFallbackError(null);
        setState({ kind: "loading" });
      })
      .then(() => scene.load(url));
  }, [url]);

  // If loading the per-task artifact fails, fall back to the static sample
  // so a bad/missing artifact never leaves the viewer stuck on an error. The
  // swap is flagged (fellBackToSample) so the HUD can say so out loud rather
  // than pass the demo torus off as the user's asset.
  useEffect(() => {
    const scene = sceneRef.current;
    const fallback = fallbackUrl ?? sampleUrl;
    if (!scene || state.kind !== "error" || url === fallback) return;

    const reason = state.message;
    // Surface the real failure to the console so a remote/laptop client can be
    // diagnosed (the on-screen result is the demo sample; without this the true
    // error is invisible).
    console.warn(
      `[Astel] Couldn't load the generated splat (${url}); showing the demo ` +
        `sample instead. Reason: ${reason}`,
    );
    Promise.resolve()
      .then(() => {
        setFallbackError(reason);
        setState({ kind: "loading" });
      })
      .then(() => scene.load(fallback));
  }, [state, url, fallbackUrl, sampleUrl]);

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
          <div className="hud-left">
            <span className="hud-chip">
              <span className="swatch" />
              <span>
                Layer <b>L3</b> · Refined surface
              </span>
            </span>
            {fellBackToSample ? (
              <span
                className="hud-chip hud-chip--warn"
                title={`The generated asset couldn't be loaded in this browser; the bundled demo sample is shown instead. Reason: ${fallbackError ?? "unknown"}`}
              >
                <span className="swatch" />
                <span>
                  Load failed — <b>demo</b>
                </span>
              </span>
            ) : isSample ? (
              <span
                className="hud-chip hud-chip--demo"
                title="This is the bundled demo sample, not a generated asset. Enter a prompt and Generate to create your own."
              >
                <span className="swatch" />
                <span>
                  <b>Demo sample</b>
                </span>
              </span>
            ) : null}
          </div>
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
