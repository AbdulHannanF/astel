/**
 * Thin client for the Astel API. Mirrors services/api schemas. SSE is consumed
 * via the Fetch streaming API (not EventSource) so we can POST first, then read
 * the event stream with proper typing and abort support.
 */

export type Modality = "text" | "image" | "video";

export type TaskStatus = "queued" | "running" | "succeeded" | "failed";

export type LayerStage = "L0_SEED" | "L1_DENSE" | "L2_COARSE" | "L3_REFINED";

export interface StageMetrics {
  splats?: number | null;
  psnr_db?: number | null;
  chamfer_mm?: number | null;
  vram_gb?: number | null;
  wall_seconds?: number | null;
}

export interface ProgressEvent {
  task_id: string;
  status: TaskStatus;
  stage: LayerStage | null;
  stage_label: string | null;
  stage_index: number;
  stage_count: number;
  progress: number;
  message: string;
  metrics?: StageMetrics | null;
}

export interface GenerationArtifact {
  name: string;
  url: string;
  content_type: string;
  bytes: number;
}

/** A stored input capture (uploaded image/video), returned by POST /v1/captures. */
export interface CaptureRef {
  capture_id: string;
  filename: string;
  content_type: string;
  bytes: number;
}

/**
 * What the L3 geometry was actually conditioned on for this task (audit
 * recommendation #2 — see docs/research/15-pipeline-wiring-audit.md). "none"
 * means a prompt/capture-independent placeholder was produced; `null` means
 * the field is unknown (e.g. a row written before this field existed).
 */
export type Conditioning = "prompt" | "image" | "video" | "none";

export interface GenerationResource {
  id: string;
  modality: Modality;
  prompt: string;
  status: TaskStatus;
  created_at: string;
  events_url: string;
  artifacts?: GenerationArtifact[];
  conditioning?: Conditioning | null;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function createGeneration(
  body: { modality: Modality; prompt: string; capture_id?: string | null },
  signal?: AbortSignal,
): Promise<GenerationResource> {
  const res = await fetch("/v1/generations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: signal ?? null,
  });
  if (!res.ok) {
    throw new ApiError(
      `Failed to create generation (${res.status})`,
      res.status,
    );
  }
  return (await res.json()) as GenerationResource;
}

/**
 * Upload raw input bytes (an image or video) to POST /v1/captures and return
 * the stored {@link CaptureRef}. The returned `capture_id` is threaded into a
 * subsequent {@link createGeneration} call.
 */
export async function uploadCapture(
  file: File,
  signal?: AbortSignal,
): Promise<CaptureRef> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch("/v1/captures", {
    method: "POST",
    body: form,
    signal: signal ?? null,
  });
  if (!res.ok) {
    throw new ApiError(`Failed to upload capture (${res.status})`, res.status);
  }
  return (await res.json()) as CaptureRef;
}

/** Fetch the current state of a generation, including its artifact list. */
export async function fetchGeneration(
  taskId: string,
  signal?: AbortSignal,
): Promise<GenerationResource> {
  const res = await fetch(`/v1/generations/${taskId}`, {
    signal: signal ?? null,
  });
  if (!res.ok) {
    throw new ApiError(`Failed to fetch generation (${res.status})`, res.status);
  }
  return (await res.json()) as GenerationResource;
}

/** Same-origin URL for a named per-task artifact (e.g. `l3.ply`). */
export function artifactUrl(taskId: string, name: string): string {
  return `/v1/generations/${taskId}/artifacts/${name}`;
}

/**
 * Stream an SSE progress feed as an async iterable of {@link ProgressEvent}.
 * Parses the minimal SSE wire format (`event:` / `data:` lines, blank-line
 * delimited) — sufficient for our gateway and dependency-free.
 */
export async function* streamGenerationEvents(
  taskId: string,
  signal?: AbortSignal,
): AsyncGenerator<ProgressEvent> {
  const res = await fetch(`/v1/generations/${taskId}/events`, {
    headers: { accept: "text/event-stream" },
    signal: signal ?? null,
  });
  if (!res.ok || res.body === null) {
    throw new ApiError(`Failed to open event stream (${res.status})`, res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // Event records are blank-line delimited; per the SSE spec a line ending may
  // be LF, CRLF, or a lone CR. sse-starlette emits CRLF (`\r\n\r\n`), so we must
  // match all three rather than only `\n\n`.
  const recordDelim = /\r\n\r\n|\n\n|\r\r/;
  const lineDelim = /\r\n|\n|\r/;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let match: RegExpExecArray | null;
      while ((match = recordDelim.exec(buffer)) !== null) {
        const rawEvent = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const dataLine = rawEvent
          .split(lineDelim)
          .find((line) => line.startsWith("data:"));
        if (dataLine) {
          const json = dataLine.slice("data:".length).trim();
          if (json) yield JSON.parse(json) as ProgressEvent;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
