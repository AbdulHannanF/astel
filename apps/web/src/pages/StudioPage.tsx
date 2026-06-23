import { useCallback, useEffect, useState } from "react";

import { GenerationDock } from "../components/GenerationDock.tsx";
import { LayerInspector } from "../components/LayerInspector.tsx";
import { PhysicsSandbox } from "../components/PhysicsSandbox.tsx";
import { RelightStudio } from "../components/RelightStudio.tsx";
import { TopBar } from "../components/TopBar.tsx";
import { TruthMeter } from "../components/TruthMeter.tsx";
import { Viewport } from "../components/Viewport.tsx";
import { useGeneration } from "../hooks/useGeneration.ts";
import { artifactUrl } from "../lib/api.ts";
import { liveLayers, SAMPLE_LAYERS, type LayerId } from "../lib/layers.ts";
import {
  fetchSampleReport,
  mapApiReport,
  type ApiQualityReport,
  type QualityReport,
} from "../lib/report.ts";

const SAMPLE_URL = "/samples/astel-sample.ply";
const SAMPLE_RELIGHT_URL = "/samples/astrolabe.relight.json";

/** Layers visible in the viewport. Only L3 is available on the sample. */
const INITIAL_VISIBLE: ReadonlySet<LayerId> = new Set<LayerId>(["L3"]);

type StageMode = "viewer" | "relight" | "physics";

const STAGE_TABS: readonly { id: StageMode; label: string }[] = [
  { id: "viewer", label: "Viewer" },
  { id: "relight", label: "Relight Studio" },
  { id: "physics", label: "Physics Sandbox" },
];

export function StudioPage(): React.JSX.Element {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [reportError, setReportError] = useState(false);
  const [visibleLayers, setVisibleLayers] =
    useState<ReadonlySet<LayerId>>(INITIAL_VISIBLE);
  const [stage, setStage] = useState<StageMode>("viewer");
  const { state, start, cancel } = useGeneration();

  const taskId = state.last?.task_id ?? null;
  const succeeded = state.phase === "done" && state.last?.status === "succeeded";

  // Idle: keep loading the static sample so the app looks alive on first
  // paint. Once a generation succeeds, switch to its real per-task splat.
  const splatUrl =
    succeeded && taskId ? artifactUrl(taskId, "l3.ply") : SAMPLE_URL;
  const relightUrl =
    succeeded && taskId
      ? artifactUrl(taskId, "l4-relight.json")
      : SAMPLE_RELIGHT_URL;
  const massPropsUrl =
    succeeded && taskId ? artifactUrl(taskId, "l5-mass.json") : undefined;

  useEffect(() => {
    const ctrl = new AbortController();

    if (succeeded && taskId) {
      fetch(artifactUrl(taskId, "quality-report.json"), {
        signal: ctrl.signal,
      })
        .then((res) => {
          if (!res.ok) throw new Error(`report ${res.status}`);
          return res.json() as Promise<ApiQualityReport>;
        })
        .then((api) => {
          setReport(mapApiReport(api, taskId));
          setReportError(false);
        })
        .catch((err: unknown) => {
          if (!ctrl.signal.aborted) setReportError(true);
          if (!(err instanceof DOMException)) console.error(err);
        });
      return () => ctrl.abort();
    }

    fetchSampleReport(ctrl.signal)
      .then((sample) => {
        setReport(sample);
        setReportError(false);
      })
      .catch((err: unknown) => {
        if (!ctrl.signal.aborted) setReportError(true);
        if (!(err instanceof DOMException)) console.error(err);
      });
    return () => ctrl.abort();
  }, [succeeded, taskId]);

  const toggleLayer = useCallback((id: LayerId) => {
    setVisibleLayers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // The sample only renders L3; viewport visibility tracks whether L3 is on.
  const splatVisible = visibleLayers.has("L3");

  // Layer Stack: reflect the live SSE stage while a generation is in flight,
  // mark L0-L3 ready on completion, and fall back to the static sample
  // layers when idle.
  const layers =
    state.phase === "idle" ? SAMPLE_LAYERS : liveLayers(state.last, succeeded);

  return (
    <div className="app" data-page="studio">
      <TopBar />
      <div className="stage">
        <div className="stage-main">
          <nav className="stage-tabs" aria-label="Workspace">
            {STAGE_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={tab.id === stage ? "active" : ""}
                onClick={() => setStage(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {stage === "viewer" && (
            <Viewport
              sampleUrl={SAMPLE_URL}
              splatUrl={splatUrl}
              fallbackUrl={SAMPLE_URL}
              isSample={!(succeeded && taskId)}
              splatVisible={splatVisible}
            />
          )}
          {stage === "relight" && <RelightStudio relightUrl={relightUrl} />}
          {stage === "physics" && (
            <PhysicsSandbox splatUrl={splatUrl} massPropsUrl={massPropsUrl} />
          )}
        </div>

        <aside className="rail">
          <div className="rail__scroll">
            <LayerInspector
              layers={layers}
              visible={visibleLayers}
              onToggle={toggleLayer}
            />
            <TruthMeter
              report={report}
              errored={reportError}
              conditioning={state.conditioning}
            />
          </div>
        </aside>
      </div>
      <GenerationDock state={state} start={start} cancel={cancel} />
    </div>
  );
}
