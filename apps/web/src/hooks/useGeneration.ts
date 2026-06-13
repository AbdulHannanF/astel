import { useCallback, useRef, useState } from "react";

import {
  createGeneration,
  streamGenerationEvents,
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
}

const INITIAL: GenerationState = { phase: "idle", last: null, error: null };

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
      setState({ phase: "submitting", last: null, error: null });

      try {
        const gen = await createGeneration(
          { modality, prompt, capture_id: captureId ?? null },
          ctrl.signal,
        );
        setState((s) => ({ ...s, phase: "running" }));

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
        });
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    },
    [],
  );

  return { state, start, cancel, reset };
}
