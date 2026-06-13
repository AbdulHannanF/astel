import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  artifactUrl,
  createGeneration,
  fetchGeneration,
  streamGenerationEvents,
  type ProgressEvent,
} from "./api.ts";

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("createGeneration", () => {
  it("posts modality + prompt and returns the resource", async () => {
    const resource = {
      id: "abc",
      modality: "text",
      prompt: "teapot",
      status: "queued",
      created_at: "2026-06-13T00:00:00Z",
      events_url: "/v1/generations/abc/events",
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify(resource), { status: 201 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const out = await createGeneration({ modality: "text", prompt: "teapot" });
    expect(out.id).toBe("abc");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/v1/generations");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      modality: "text",
      prompt: "teapot",
    });
  });

  it("throws ApiError on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("nope", { status: 500 })),
    );
    await expect(
      createGeneration({ modality: "text", prompt: "x" }),
    ).rejects.toBeInstanceOf(ApiError);
  });
});

describe("streamGenerationEvents", () => {
  it("parses multiple SSE events split across chunk boundaries", async () => {
    const e1: ProgressEvent = {
      task_id: "t",
      status: "running",
      stage: "L0_SEED",
      stage_label: "Seeding",
      stage_index: 0,
      stage_count: 4,
      progress: 0.1,
      message: "seed",
    };
    const e2: ProgressEvent = {
      ...e1,
      status: "succeeded",
      stage: "L3_REFINED",
      progress: 1,
      message: "done",
    };
    // Deliberately split the wire format mid-event to exercise buffering.
    const wire = `event: progress\ndata: ${JSON.stringify(e1)}\n\nevent: progress\ndata: ${JSON.stringify(e2)}\n\n`;
    const mid = Math.floor(wire.length / 2);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(sseResponse([wire.slice(0, mid), wire.slice(mid)])),
    );

    const got: ProgressEvent[] = [];
    for await (const evt of streamGenerationEvents("t")) got.push(evt);

    expect(got).toHaveLength(2);
    expect(got[0]?.stage).toBe("L0_SEED");
    expect(got[1]?.status).toBe("succeeded");
    expect(got[1]?.progress).toBe(1);
  });

  it("parses CRLF-delimited events (the real sse-starlette wire format)", async () => {
    const e1: ProgressEvent = {
      task_id: "t",
      status: "running",
      stage: "L0_SEED",
      stage_label: "Seeding",
      stage_index: 0,
      stage_count: 4,
      progress: 0.1,
      message: "seed",
    };
    const e2: ProgressEvent = {
      ...e1,
      status: "succeeded",
      stage: "L3_REFINED",
      progress: 1,
      message: "done",
    };
    // sse-starlette terminates lines and records with CRLF, not LF.
    const wire = `event: progress\r\ndata: ${JSON.stringify(e1)}\r\n\r\nevent: progress\r\ndata: ${JSON.stringify(e2)}\r\n\r\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([wire])));

    const got: ProgressEvent[] = [];
    for await (const evt of streamGenerationEvents("t")) got.push(evt);

    expect(got).toHaveLength(2);
    expect(got[1]?.status).toBe("succeeded");
  });

  it("throws ApiError when the stream cannot open", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 404 })),
    );
    const iter = streamGenerationEvents("missing");
    await expect(iter.next()).rejects.toBeInstanceOf(ApiError);
  });
});

describe("artifactUrl", () => {
  it("builds the same-origin per-task artifact URL", () => {
    expect(artifactUrl("abc", "l3.ply")).toBe(
      "/v1/generations/abc/artifacts/l3.ply",
    );
    expect(artifactUrl("abc", "quality-report.json")).toBe(
      "/v1/generations/abc/artifacts/quality-report.json",
    );
  });
});

describe("fetchGeneration", () => {
  it("fetches the generation resource including artifacts", async () => {
    const resource = {
      id: "abc",
      modality: "text",
      prompt: "teapot",
      status: "succeeded",
      created_at: "2026-06-13T00:00:00Z",
      events_url: "/v1/generations/abc/events",
      artifacts: [
        {
          name: "l3.ply",
          url: "/v1/generations/abc/artifacts/l3.ply",
          content_type: "application/octet-stream",
          bytes: 1234,
        },
      ],
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(resource), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const out = await fetchGeneration("abc");
    expect(out.artifacts?.[0]?.name).toBe("l3.ply");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/generations/abc");
  });

  it("throws ApiError on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("nope", { status: 404 })),
    );
    await expect(fetchGeneration("missing")).rejects.toBeInstanceOf(ApiError);
  });
});
