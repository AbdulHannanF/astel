import { describe, it, expect, beforeEach, vi } from "vitest";
import { AstelClient, AstelError } from "./client.js";
import type { Generation } from "./types.js";

const BASE = "http://test-astel";

const GEN: Generation = {
  id: "abc-123",
  modality: "text",
  prompt: "a brass astrolabe",
  status: "succeeded",
  created_at: "2026-06-18T00:00:00",
  events_url: "/v1/generations/abc-123/events",
  artifacts: [
    { name: "l3.ply", url: "/v1/generations/abc-123/artifacts/l3.ply", content_type: "application/octet-stream", bytes: 1234 },
    { name: "quality-report.json", url: "/v1/generations/abc-123/artifacts/quality-report.json", content_type: "application/json", bytes: 512 },
  ],
  mode: "refine",
  refine_of: null,
  billing: {
    mode: "refine",
    refine_of: null,
    items: [
      { code: "L3", label: "Refined", tier: "refine", credits: 20, usd: 0.2, detail: "" },
    ],
    total_credits: 21,
    total_usd: 0.21,
    credit_usd_rate: 0.01,
    caveats: [],
  },
  conditioning: "prompt",
};

function mockFetch(payload: unknown, status = 200): void {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: status < 400,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(payload),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
  }));
}

describe("AstelClient", () => {
  let client: AstelClient;

  beforeEach(() => {
    client = new AstelClient(BASE);
    vi.restoreAllMocks();
  });

  it("health returns status", async () => {
    mockFetch({ status: "ok", service: "astel-api", version: "0.1.0" });
    const h = await client.health();
    expect(h.status).toBe("ok");
  });

  it("generate returns a Generation", async () => {
    mockFetch(GEN, 201);
    const gen = await client.generate({ prompt: "a brass astrolabe" });
    expect(gen.id).toBe("abc-123");
    expect(gen.status).toBe("succeeded");
  });

  it("getGeneration returns the generation", async () => {
    mockFetch(GEN);
    const gen = await client.getGeneration("abc-123");
    expect(gen.id).toBe("abc-123");
    expect(gen.artifacts).toHaveLength(2);
  });

  it("listArtifacts returns artifact list", async () => {
    mockFetch(GEN);
    const arts = await client.listArtifacts("abc-123");
    expect(arts.map((a) => a.name)).toContain("l3.ply");
  });

  it("downloadArtifact returns an ArrayBuffer", async () => {
    mockFetch({});
    const buf = await client.downloadArtifact("abc-123", "l3.ply");
    expect(buf).toBeInstanceOf(ArrayBuffer);
  });

  it("artifactUrl returns full URL for known artifact", () => {
    const url = client.artifactUrl(GEN, "l3.ply");
    expect(url).toContain("l3.ply");
    expect(url).toContain(BASE);
  });

  it("artifactUrl returns null for unknown artifact", () => {
    expect(client.artifactUrl(GEN, "missing.glb")).toBeNull();
  });

  it("throws AstelError on non-ok response", async () => {
    mockFetch({ detail: "not found" }, 404);
    await expect(client.getGeneration("bad-id")).rejects.toThrow(AstelError);
  });

  it("pricing returns the live schedule", async () => {
    mockFetch({
      credit_usd_rate: 0.01,
      layers: [
        { code: "L0", label: "Seed cloud", tier: "preview", credits: 1 },
        { code: "L3", label: "Refined", tier: "refine", credits: 20 },
      ],
      modes: { preview: ["L0", "L1", "L2"], refine: ["L3", "L4"] },
      notes: ["L0–L2 previews are cheap; L3 refine is the main spend."],
    });
    const p = await client.pricing();
    expect(p.credit_usd_rate).toBe(0.01);
    expect(p.layers.map((l) => l.code)).toContain("L3");
    expect(p.modes.refine).toContain("L3");
  });
});
