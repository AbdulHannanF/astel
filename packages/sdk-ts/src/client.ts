/**
 * Astel TypeScript SDK client.
 *
 * ```ts
 * import { AstelClient } from "@astel/sdk";
 * const client = new AstelClient("http://localhost:8000");
 * const gen = await client.generate({ prompt: "a brass astrolabe" });
 * const artifacts = await client.listArtifacts(gen.id);
 * ```
 */

import type {
  ArtifactRef,
  CaptureRef,
  Generation,
  GenerateOptions,
  GenerationStatus,
  PricingResource,
} from "./types.js";

export class AstelError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "AstelError";
  }
}

export interface AstelClientOptions {
  /** Base URL for the API (default: http://localhost:8000). */
  baseUrl?: string;
  /** Bearer token for authenticated endpoints. */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 120_000). */
  timeoutMs?: number;
}

export class AstelClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;

  constructor(baseUrl = "http://localhost:8000", opts: AstelClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.headers = { "Content-Type": "application/json" };
    if (opts.apiKey) this.headers["Authorization"] = `Bearer ${opts.apiKey}`;
    this.timeoutMs = opts.timeoutMs ?? 120_000;
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private async fetch<T>(
    method: string,
    path: string,
    body?: unknown,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    const init: RequestInit = {
      method,
      headers: { ...this.headers, ...extraHeaders },
      signal: ctrl.signal,
    };
    if (body !== undefined) init.body = JSON.stringify(body);
    try {
      const res = await fetch(this.url(path), init);
      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new AstelError(
          `Astel API error ${res.status}: ${res.statusText}`,
          res.status,
          errBody,
        );
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  // ------------------------------------------------------------------
  // Health
  // ------------------------------------------------------------------

  async health(): Promise<{ status: string; service: string; version: string }> {
    return this.fetch("GET", "/healthz");
  }

  // ------------------------------------------------------------------
  // Pricing
  // ------------------------------------------------------------------

  async pricing(): Promise<PricingResource> {
    return this.fetch("GET", "/v1/pricing");
  }

  // ------------------------------------------------------------------
  // Captures
  // ------------------------------------------------------------------

  async uploadCapture(file: Blob | ArrayBuffer, filename = "capture.jpg"): Promise<CaptureRef> {
    const form = new FormData();
    const blob = file instanceof Blob ? file : new Blob([file]);
    form.append("file", blob, filename);

    const res = await fetch(this.url("/v1/captures"), {
      method: "POST",
      headers: this.apiKey ? { Authorization: this.headers["Authorization"]! } : {},
      body: form,
    });
    if (!res.ok) throw new AstelError("capture upload failed", res.status, null);
    return (await res.json()) as CaptureRef;
  }

  private get apiKey(): string | undefined {
    return this.headers["Authorization"]?.replace("Bearer ", "");
  }

  // ------------------------------------------------------------------
  // Generations
  // ------------------------------------------------------------------

  async generate(opts: GenerateOptions = {}): Promise<Generation> {
    return this.fetch<Generation>("POST", "/v1/generations", {
      modality: opts.modality ?? "text",
      prompt: opts.prompt ?? null,
      capture_id: opts.captureId ?? null,
      mode: opts.mode ?? "refine",
      refine_of: opts.refineOf ?? null,
    });
  }

  async getGeneration(id: string): Promise<Generation> {
    return this.fetch("GET", `/v1/generations/${id}`);
  }

  async waitForGeneration(
    id: string,
    { pollMs = 2000, maxMs = 600_000 }: { pollMs?: number; maxMs?: number } = {},
  ): Promise<Generation> {
    const deadline = Date.now() + maxMs;
    while (Date.now() < deadline) {
      const gen = await this.getGeneration(id);
      if (isTerminal(gen.status)) return gen;
      await sleep(pollMs);
    }
    throw new Error(`generation ${id} did not complete within ${maxMs}ms`);
  }

  // ------------------------------------------------------------------
  // Artifacts
  // ------------------------------------------------------------------

  async listArtifacts(generationId: string): Promise<ArtifactRef[]> {
    const gen = await this.getGeneration(generationId);
    return gen.artifacts;
  }

  async downloadArtifact(generationId: string, name: string): Promise<ArrayBuffer> {
    const res = await fetch(
      this.url(`/v1/generations/${generationId}/artifacts/${name}`),
      { headers: this.headers },
    );
    if (!res.ok) throw new AstelError("artifact download failed", res.status, null);
    return res.arrayBuffer();
  }

  artifactUrl(generation: Generation, name: string): string | null {
    const art = generation.artifacts.find((a) => a.name === name);
    return art ? `${this.baseUrl}${art.url}` : null;
  }
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function isTerminal(status: GenerationStatus): boolean {
  return status === "succeeded" || status === "SUCCEEDED" ||
    status === "failed" || status === "FAILED";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
