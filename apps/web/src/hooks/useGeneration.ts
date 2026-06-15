import { useCallback, useRef, useState } from "react";

import {
  createGeneration,
  streamGenerationEvents,
  type Conditioning,
  type Modality,
  type ProgressEvent,
} from "../lib/api.ts";

export type GenerationPhase =
  | "idle"
  | "submitting"
  | "running"
  | "done"
  | "error";

export interface GenerationState {
  phase: GenerationPhase;
  /** Latest progress event from the SSE stream, if any. */
  last: ProgressEvent | null;
  error: string | null;
  /**
   * What this generation's geometry was conditioned on, from the
   * POST /v1/generations response (audit recommendation #2). `null` while
   * idle/submitting or if the field was absent from the response.
   */
  conditioning: Conditioning | null;
}

const INITIAL: GenerationState = {
  phase: "idle",
  last: null,
  error: null,
  conditioning: null,
};

/**
 * Drives a generation: POST /v1/generations, then consume its SSE event stream,
 * surfacing the latest {@link ProgressEvent}. Abortable and re-entrant.
 */
export function useGeneration(): {
  state: GenerationState;
  start: (
    modality: Modality,
    prompt: string,
    captureId?: string | null,
  ) => Promise<void>;
  cancel: () => void;
  reset: () => void;
} {
  const [state, setState] = useState<GenerationState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState(INITIAL);
  }, []);

  const reset = useCallback(() => setState(INITIAL), []);

  const start = useCallback(
    async (
      modality: Modality,
      prompt: string,
      captureId?: string | null,
    ) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setState({ phase: "submitting", last: null, error: null, conditioning: null });

      try {
        const gen = await createGeneration(
          { modality, prompt, capture_id: captureId ?? null },
          ctrl.signal,
        );
        setState((s) => ({
          ...s,
          phase: "running",
          conditioning: gen.conditioning ?? null,
        }));

        for await (const evt of streamGenerationEvents(gen.id, ctrl.signal)) {
          setState((s) => ({ ...s, phase: "running", last: evt }));
          if (evt.status === "succeeded") {
            setState((s) => ({ ...s, phase: "done", last: evt }));
          } else if (evt.status === "failed") {
            setState((s) => ({
              ...s,
              phase: "error",
              error: evt.message,
            }));
          }
        }
      } catch (err) {
        if (ctrl.signal.aborted) return; // user-initiated cancel
        setState({
          phase: "error",
          last: null,
          error:
            err instanceof Error ? err.message : "Generation failed to start",
          conditioning: null,
        });
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    },
    [],
  );

  return { state, start, cancel, reset };
}
