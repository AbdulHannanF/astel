import { useId, useRef, useState } from "react";

import type { GenerationState } from "../hooks/useGeneration.ts";
import {
  uploadCapture,
  type LayerStage,
  type Modality,
  type ProgressEvent,
} from "../lib/api.ts";
import "./GenerationDock.css";

type UploadState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "uploaded"; captureId: string }
  | { phase: "error"; message: string };

const MODES: { id: Modality; label: string; icon: React.JSX.Element }[] = [
  { id: "text", label: "Text", icon: <TextIcon /> },
  { id: "image", label: "Image", icon: <ImageIcon /> },
  { id: "video", label: "Video", icon: <VideoIcon /> },
];

const STAGE_ORDER: { id: LayerStage; label: string }[] = [
  { id: "L0_SEED", label: "Seed" },
  { id: "L1_DENSE", label: "Dense" },
  { id: "L2_COARSE", label: "Coarse" },
  { id: "L3_REFINED", label: "Refine" },
];

const PLACEHOLDER: Record<Modality, string> = {
  text: "Describe an object — “a worn brass astrolabe on a wooden base”",
  image: "Drop a photo, or paste an image URL to reconstruct",
  video: "Drop an orbit clip to capture a real object",
};

export function GenerationDock({
  state,
  start,
  cancel,
}: {
  state: GenerationState;
  start: (
    modality: Modality,
    prompt: string,
    captureId?: string | null,
  ) => Promise<void>;
  cancel: () => void;
}): React.JSX.Element {
  const [modality, setModality] = useState<Modality>("text");
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadState>({ phase: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputId = useId();
  const dragDepth = useRef(0);

  const fileName = file?.name ?? null;
  const busy =
    state.phase === "submitting" ||
    state.phase === "running" ||
    upload.phase === "uploading";
  const canSubmit =
    !busy && (prompt.trim().length > 0 || file !== null);

  const acceptFile = (next: File): void => {
    setFile(next);
    setUpload({ phase: "idle" });
    if (modality === "text") setModality("image");
  };

  const submit = (): void => {
    if (!canSubmit) return;
    const text =
      prompt.trim() ||
      (fileName ? `capture: ${fileName}` : "untitled generation");

    // Text path: no capture to upload — submit immediately.
    if (file === null) {
      void start(modality, text);
      return;
    }

    // Image/Video path: upload the bytes to /v1/captures first, then thread
    // the returned capture_id into the generation request. Surface upload
    // progress/failure inline so the user sees the bytes really moved.
    setUpload({ phase: "uploading" });
    void uploadCapture(file)
      .then((ref) => {
        setUpload({ phase: "uploaded", captureId: ref.capture_id });
        return start(modality, text, ref.capture_id);
      })
      .catch((err: unknown) => {
        setUpload({
          phase: "error",
          message: err instanceof Error ? err.message : "Upload failed",
        });
      });
  };

  const onDrop = (e: React.DragEvent): void => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) acceptFile(dropped);
  };

  return (
    <div className="dock">
      <div className="dock__inner">
        <div className="modes" role="tablist" aria-label="Input modality">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="tab"
              type="button"
              aria-selected={modality === m.id}
              className={"mode " + (modality === m.id ? "mode--active" : "")}
              onClick={() => {
                setModality(m.id);
                setFile(null);
                setUpload({ phase: "idle" });
              }}
            >
              {m.icon}
              {m.label}
            </button>
          ))}
        </div>

        <div
          className={"prompt " + (dragging ? "prompt--drag" : "")}
          onDragEnter={(e) => {
            e.preventDefault();
            dragDepth.current += 1;
            setDragging(true);
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => {
            dragDepth.current = Math.max(0, dragDepth.current - 1);
            if (dragDepth.current === 0) setDragging(false);
          }}
          onDrop={onDrop}
        >
          <input
            id={inputId}
            type="text"
            value={prompt}
            placeholder={dragging ? "Release to attach" : PLACEHOLDER[modality]}
            onChange={(e) => setPrompt(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            aria-label={`${modality} prompt`}
          />
          {fileName ? (
            <span
              className={
                "prompt__file" +
                (upload.phase === "error" ? " prompt__file--error" : "")
              }
              title={fileName}
            >
              {fileName}
              {upload.phase === "uploading" && (
                <span className="prompt__file-status"> · uploading…</span>
              )}
              {upload.phase === "uploaded" && (
                <span className="prompt__file-status"> · uploaded ✓</span>
              )}
              {upload.phase === "error" && (
                <span className="prompt__file-status"> · {upload.message}</span>
              )}
            </span>
          ) : modality !== "text" ? (
            <span className="prompt__hint">drag &amp; drop</span>
          ) : null}
        </div>

        <button
          type="button"
          className="generate"
          disabled={!canSubmit}
          onClick={submit}
        >
          {busy ? "Generating…" : "Generate"}
          {!busy && <ArrowIcon />}
        </button>

        {state.phase !== "idle" && (
          <ProgressRail
            last={state.last}
            phase={state.phase}
            error={state.error}
            onCancel={cancel}
          />
        )}
      </div>
    </div>
  );
}

function ProgressRail({
  last,
  phase,
  error,
  onCancel,
}: {
  last: ProgressEvent | null;
  phase: string;
  error: string | null;
  onCancel: () => void;
}): React.JSX.Element {
  const activeIndex = last?.stage
    ? STAGE_ORDER.findIndex((s) => s.id === last.stage)
    : -1;
  const overall = last ? Math.round(last.progress * 100) : 0;
  const done = phase === "done";
  const errored = phase === "error";

  const metric = last?.metrics;
  const metricText =
    metric?.splats != null
      ? `${(metric.splats / 1000).toFixed(0)}k splats`
      : metric?.chamfer_mm != null
        ? `${metric.chamfer_mm.toFixed(1)} mm`
        : null;

  return (
    <div className="rail-progress">
      <div className="stages">
        {STAGE_ORDER.map((stage, i) => {
          const isDone = done || (activeIndex >= 0 && i < activeIndex);
          const isActive = !done && i === activeIndex;
          // Fraction within the active stage, derived from overall progress.
          const stageFill = isActive
            ? Math.max(
                0,
                Math.min(1, overall / 100 - i / STAGE_ORDER.length),
              ) * STAGE_ORDER.length
            : 0;
          return (
            <div
              key={stage.id}
              className={
                "stage-cell " +
                (isDone
                  ? "stage-cell--done"
                  : isActive
                    ? "stage-cell--active"
                    : "")
              }
            >
              <div className="stage-cell__top">
                <span className="stage-cell__id mono">{stage.id.slice(0, 2)}</span>
                <span className="stage-cell__label">{stage.label}</span>
              </div>
              <div className="stage-cell__track">
                <div
                  className="stage-cell__fill"
                  style={{ width: `${stageFill * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div
        className={
          "rail-status " +
          (errored
            ? "rail-status--error"
            : done
              ? "rail-status--done"
              : "")
        }
      >
        {errored ? (
          <span>{error ?? "Generation failed"}</span>
        ) : done ? (
          <>
            <span>Asset ready</span>
            {metricText && (
              <span className="rail-status__metric mono">· {metricText}</span>
            )}
          </>
        ) : (
          <>
            <span>{last?.message ?? "Queued"}</span>
            {metricText && (
              <span className="rail-status__metric mono">· {metricText}</span>
            )}
            <span className="rail-status__pct mono">{overall}%</span>
          </>
        )}
        <button type="button" className="rail-cancel" onClick={onCancel}>
          {done || errored ? "Clear" : "Cancel"}
        </button>
      </div>
    </div>
  );
}

/* ---- icons ---- */
function TextIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M3 4h8M3 7h8M3 10h5"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}
function ImageIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
      <rect
        x="2"
        y="2.5"
        width="10"
        height="9"
        rx="1.5"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <circle cx="5" cy="5.5" r="1" fill="currentColor" />
      <path
        d="M2.5 9.5 5.5 7l2 2 2-1.5 2 2"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function VideoIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
      <rect
        x="2"
        y="3.5"
        width="7"
        height="7"
        rx="1.5"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <path
        d="M9 6.2 12 4.5v5L9 7.8"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function ArrowIcon(): React.JSX.Element {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M3 7h8M7.5 3.5 11 7l-3.5 3.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
