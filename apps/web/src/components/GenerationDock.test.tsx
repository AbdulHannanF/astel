import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { GenerationState } from "../hooks/useGeneration.ts";
import { GenerationDock } from "./GenerationDock.tsx";

const IDLE: GenerationState = {
  phase: "idle",
  last: null,
  error: null,
  conditioning: null,
};

afterEach(() => {
  vi.restoreAllMocks();
});

function dropFile(file: File): void {
  // The drop target is the .prompt container; the file lands via dataTransfer.
  const target = document.querySelector(".prompt");
  expect(target).not.toBeNull();
  fireEvent.drop(target as Element, { dataTransfer: { files: [file] } });
}

describe("GenerationDock capture upload", () => {
  it("uploads the dropped file then submits with the returned capture_id", async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const cancel = vi.fn();
    const captureRef = {
      capture_id: "capture-xyz",
      filename: "orbit.png",
      content_type: "image/png",
      bytes: 3,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify(captureRef), { status: 201 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<GenerationDock state={IDLE} start={start} cancel={cancel} />);

    dropFile(new File(["png"], "orbit.png", { type: "image/png" }));
    expect(screen.getByText("orbit.png")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    // The capture upload really happens (POST /v1/captures).
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/v1/captures",
        expect.objectContaining({ method: "POST" }),
      );
    });
    // …and the returned capture_id is threaded into start().
    await waitFor(() => {
      expect(start).toHaveBeenCalledWith(
        "image",
        "capture: orbit.png",
        "capture-xyz",
      );
    });
    expect(await screen.findByText(/uploaded/)).toBeInTheDocument();
  });

  it("surfaces an upload error without calling start", async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const cancel = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("nope", { status: 500 })),
    );

    render(<GenerationDock state={IDLE} start={start} cancel={cancel} />);

    dropFile(new File(["mp4"], "clip.mp4", { type: "video/mp4" }));
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    expect(await screen.findByText(/upload capture/i)).toBeInTheDocument();
    expect(start).not.toHaveBeenCalled();
  });

  it("submits text without uploading when no file is attached", async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const cancel = vi.fn();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<GenerationDock state={IDLE} start={start} cancel={cancel} />);

    fireEvent.change(screen.getByLabelText(/text prompt/i), {
      target: { value: "a brass astrolabe" },
    });
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    expect(start).toHaveBeenCalledWith("text", "a brass astrolabe");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("GenerationDock progress rail", () => {
  it("shows an error, not 'Asset ready', when the SSE stream reports failure", () => {
    const failedState: GenerationState = {
      phase: "error",
      last: {
        task_id: "t1",
        status: "failed",
        stage: null,
        stage_label: null,
        stage_index: 0,
        stage_count: 4,
        progress: 0,
        message: "simulated CUDA OOM",
        metrics: null,
      },
      error: "simulated CUDA OOM",
      conditioning: "none",
    };

    render(
      <GenerationDock state={failedState} start={vi.fn()} cancel={vi.fn()} />,
    );

    expect(screen.getByText("simulated CUDA OOM")).toBeInTheDocument();
    expect(screen.queryByText("Asset ready")).not.toBeInTheDocument();
  });

  it("reports the real terminal splat count, not a fabricated one", () => {
    const doneState: GenerationState = {
      phase: "done",
      last: {
        task_id: "t1",
        status: "succeeded",
        stage: "L3_REFINED",
        stage_label: "Complete",
        stage_index: 4,
        stage_count: 4,
        progress: 1,
        message: "Asset ready",
        metrics: { splats: 8000 },
      },
      error: null,
      conditioning: "image",
    };

    render(
      <GenerationDock state={doneState} start={vi.fn()} cancel={vi.fn()} />,
    );

    expect(screen.getByText("Asset ready")).toBeInTheDocument();
    expect(screen.getByText(/8k splats/)).toBeInTheDocument();
  });
});
