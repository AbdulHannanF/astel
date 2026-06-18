import { useEffect, useState } from "react";

import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface OpenApiInfo {
  title: string;
  version: string;
  description?: string;
}

interface OpenApiOperation {
  summary?: string;
  tags?: string[];
  operationId?: string;
}

type HttpMethod = "get" | "post" | "put" | "patch" | "delete" | "head" | "options";

type OpenApiPathItem = Partial<Record<HttpMethod, OpenApiOperation>>;

interface OpenApiSpec {
  info: OpenApiInfo;
  paths: Record<string, OpenApiPathItem>;
}

// ---------------------------------------------------------------------------
// Static endpoint reference (derived from api.ts; ground-truth for the UI)
// ---------------------------------------------------------------------------

interface EndpointDef {
  method: string;
  path: string;
  summary: string;
  tag: "meta" | "captures" | "generations";
}

const ENDPOINTS: EndpointDef[] = [
  {
    method: "GET",
    path: "/healthz",
    summary: "Liveness probe — returns 200 when the gateway is up.",
    tag: "meta",
  },
  {
    method: "GET",
    path: "/v1/pipeline",
    summary: "Stage specs (StageSpec[]) for L0–L3: label, description, nominal_seconds.",
    tag: "meta",
  },
  {
    method: "GET",
    path: "/v1/pricing",
    summary: "Credit price schedule: layers[], credit_usd_rate, modes, notes.",
    tag: "meta",
  },
  {
    method: "POST",
    path: "/v1/captures",
    summary: "Multipart upload (image or video). Returns CaptureRef { capture_id, filename, content_type, bytes }.",
    tag: "captures",
  },
  {
    method: "POST",
    path: "/v1/generations",
    summary:
      'Body: { modality: "text"|"image"|"video", prompt, capture_id?, mode?: "preview"|"refine", refine_of? }. Returns GenerationResource.',
    tag: "generations",
  },
  {
    method: "GET",
    path: "/v1/generations/{id}",
    summary: "Fetch a generation and its artifact list.",
    tag: "generations",
  },
  {
    method: "GET",
    path: "/v1/generations/{id}/events",
    summary: "SSE progress stream of ProgressEvent objects (text/event-stream).",
    tag: "generations",
  },
  {
    method: "GET",
    path: "/v1/generations/{id}/artifacts/{name}",
    summary: 'Download a named artifact, e.g. "l3.ply" or "quality-report.json".',
    tag: "generations",
  },
];

const TAG_ORDER: EndpointDef["tag"][] = ["meta", "captures", "generations"];

const TAG_LABELS: Record<EndpointDef["tag"], string> = {
  meta: "Meta / discovery",
  captures: "Captures",
  generations: "Generations",
};

// ---------------------------------------------------------------------------
// Tooling / guides card data
// ---------------------------------------------------------------------------

interface GuideCard {
  name: string;
  desc: string;
  href: string;
}

const GUIDES: GuideCard[] = [
  {
    name: "Python + TypeScript SDK",
    desc: "Typed clients for every endpoint. pip install astel / npm install @astel/sdk. Access via self-host or cloud API.",
    href: "/self-host",
  },
  {
    name: "MCP server",
    desc: "Generate assets programmatically from any IDE or agent that supports the Model Context Protocol.",
    href: "/self-host",
  },
  {
    name: "glTF / USDZ export",
    desc: "KHR_gaussian_splatting glTF and USD/USDZ with splat payloads for VFX pipelines. Download from the artifacts endpoint.",
    href: "/docs",
  },
  {
    name: "Unity plugin",
    desc: "Import .spz / .ply + the Astel manifest. Auto-configures collision proxies, mass, and material from L5/L6.",
    href: "/self-host",
  },
  {
    name: "Unreal Engine 5 plugin",
    desc: "Same manifest-driven auto-setup as the Unity package. Coordinate-convention conversion handled automatically.",
    href: "/self-host",
  },
  {
    name: "Splats 101 for studios",
    desc: "What Gaussian splats are, how the layer model differs from meshes, and how to integrate Astel assets into your pipeline.",
    href: "/docs",
  },
];

// ---------------------------------------------------------------------------
// Method badge colour
// ---------------------------------------------------------------------------

function methodClass(method: string): string {
  if (method === "GET") return "endpoint-method--get";
  if (method === "POST") return "endpoint-method--post";
  return "endpoint-method--other";
}

// ---------------------------------------------------------------------------
// Live spec panel
// ---------------------------------------------------------------------------

function LiveSpecPanel(): React.JSX.Element {
  const [spec, setSpec] = useState<OpenApiSpec | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const res = await fetch("/openapi.json", { signal: ctrl.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as OpenApiSpec;
        setSpec(data);
      } catch (e) {
        if (!(e instanceof DOMException)) setError(true);
      }
    })();
    return () => ctrl.abort();
  }, []);

  if (!error && spec === null) {
    // Still loading — show nothing (gateway might be offline)
    return (
      <div className="docs-live-panel docs-live-panel--loading" aria-busy="true">
        <span className="docs-live-panel__loading-text mono">Connecting to gateway…</span>
      </div>
    );
  }

  if (error || spec === null) {
    return (
      <div className="docs-live-panel docs-live-panel--offline" role="status">
        <p className="docs-live-panel__offline-title">Gateway offline</p>
        <p className="docs-live-panel__offline-body">
          Start the gateway to load the live spec:{" "}
          <code className="docs-inline-code">pnpm run up</code>
        </p>
        <p className="docs-live-panel__offline-body">
          Once running, the live API title, version, and path list will appear here.
        </p>
        <div className="docs-live-panel__links">
          <a href="/docs" className="docs-external-link">
            Interactive API docs (Swagger UI) →
          </a>
          <a href="/openapi.json" className="docs-external-link">
            Raw OpenAPI JSON →
          </a>
        </div>
      </div>
    );
  }

  // Spec loaded successfully
  const pathEntries = Object.entries(spec.paths);

  return (
    <div className="docs-live-panel docs-live-panel--loaded">
      <div className="docs-live-panel__head">
        <div className="docs-live-panel__identity">
          <span className="docs-live-panel__badge mono">live</span>
          <span className="docs-live-panel__title">{spec.info.title}</span>
          <span className="docs-live-panel__version mono">v{spec.info.version}</span>
        </div>
        <div className="docs-live-panel__links">
          <a href="/docs" className="docs-external-link">
            Interactive API docs →
          </a>
          <a href="/openapi.json" className="docs-external-link">
            Raw OpenAPI →
          </a>
        </div>
      </div>
      <div className="docs-live-paths">
        {pathEntries.map(([path, item]) => {
          const methods = (Object.keys(item) as HttpMethod[]).filter(
            (m) => ["get", "post", "put", "patch", "delete"].includes(m),
          );
          return methods.map((method) => {
            const op = item[method];
            return (
              <div key={`${method}:${path}`} className="docs-live-path">
                <span className={`endpoint-method ${methodClass(method.toUpperCase())} mono`}>
                  {method.toUpperCase()}
                </span>
                <code className="docs-path-code">{path}</code>
                {op?.summary && (
                  <span className="docs-live-path__summary">{op.summary}</span>
                )}
              </div>
            );
          });
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DocsPage
// ---------------------------------------------------------------------------

export function DocsPage(): React.JSX.Element {
  return (
    <div data-page="docs" className="page docs-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Developer docs"
          title="Build with Astel."
          lede="The full API reference, quickstart examples, and guides for integrating layered Gaussian splat assets into your pipeline."
        />

        {/* 1. Live API panel */}
        <Section
          eyebrow="Live API"
          title="The gateway — live spec."
          lede="When the gateway is running, this panel loads the real OpenAPI spec and reflects any changes immediately."
          className="docs-live-section"
        >
          <Reveal>
            <LiveSpecPanel />
          </Reveal>
        </Section>

        {/* 2. Endpoint reference */}
        <Section
          eyebrow="Endpoint reference"
          title="All endpoints."
          lede="These are the only routes the gateway exposes today. Base URL: http://localhost:8000 (dev) or your self-hosted domain."
          className="docs-endpoints-section"
        >
          <div className="docs-endpoint-groups">
            {TAG_ORDER.map((tag) => {
              const endpoints = ENDPOINTS.filter((e) => e.tag === tag);
              return (
                <Reveal key={tag} className="docs-endpoint-group">
                  <p className="docs-endpoint-group__label mono">{TAG_LABELS[tag]}</p>
                  <div className="docs-endpoint-list">
                    {endpoints.map((ep) => (
                      <div key={`${ep.method}:${ep.path}`} className="docs-endpoint-row">
                        <span className={`endpoint-method ${methodClass(ep.method)} mono`}>
                          {ep.method}
                        </span>
                        <code className="docs-path-code">{ep.path}</code>
                        <span className="docs-endpoint-row__summary">{ep.summary}</span>
                      </div>
                    ))}
                  </div>
                </Reveal>
              );
            })}
          </div>
        </Section>

        {/* 3. Quickstart */}
        <Section
          eyebrow="Quickstart"
          title="From zero to splat in four calls."
          className="docs-quickstart-section"
        >
          <div className="docs-quickstart-steps">
            <Reveal className="docs-qs-step">
              <p className="docs-qs-step__label mono">1 — Create a text generation</p>
              <pre className="docs-code-block"><code>{`curl -s -X POST http://localhost:8000/v1/generations \\
  -H "Content-Type: application/json" \\
  -d '{
    "modality": "text",
    "prompt": "a ceramic teapot, matte glaze, studio lighting"
  }'
# → GenerationResource { id, status: "queued", events_url, … }`}</code></pre>
            </Reveal>

            <Reveal className="docs-qs-step" delay={80}>
              <p className="docs-qs-step__label mono">2 — Stream progress via SSE</p>
              <pre className="docs-code-block"><code>{`curl -s -H "Accept: text/event-stream" \\
  http://localhost:8000/v1/generations/<id>/events
# data: {"task_id":"…","status":"running","stage":"L1_DENSE",
#        "progress":0.22,"message":"Building dense cloud…"}`}</code></pre>
            </Reveal>

            <Reveal className="docs-qs-step" delay={160}>
              <p className="docs-qs-step__label mono">3 — Upload an image capture</p>
              <pre className="docs-code-block"><code>{`curl -s -X POST http://localhost:8000/v1/captures \\
  -F "file=@my-photo.jpg"
# → CaptureRef { capture_id, filename, content_type, bytes }

curl -s -X POST http://localhost:8000/v1/generations \\
  -H "Content-Type: application/json" \\
  -d '{
    "modality": "image",
    "prompt": "reconstruct object from photo",
    "capture_id": "<capture_id from above>"
  }'`}</code></pre>
            </Reveal>

            <Reveal className="docs-qs-step" delay={240}>
              <p className="docs-qs-step__label mono">4 — Download the L3 splat</p>
              <pre className="docs-code-block"><code>{`# After status reaches "succeeded":
curl -o output.ply \\
  http://localhost:8000/v1/generations/<id>/artifacts/l3.ply

# Or fetch the quality report:
curl http://localhost:8000/v1/generations/<id>/artifacts/quality-report.json`}</code></pre>
            </Reveal>

            <Reveal className="docs-qs-step" delay={320}>
              <p className="docs-qs-step__label mono">TypeScript fetch equivalent</p>
              <pre className="docs-code-block"><code>{`import { createGeneration, streamGenerationEvents } from "@astel/sdk";

const gen = await createGeneration({
  modality: "text",
  prompt: "a brass astrolabe on a velvet surface",
});

for await (const event of streamGenerationEvents(gen.id)) {
  console.log(event.stage, event.progress, event.message);
  if (event.status === "succeeded") break;
}

// Download artifact
const ply = await fetch(\`/v1/generations/\${gen.id}/artifacts/l3.ply\`);`}</code></pre>
            </Reveal>
          </div>
        </Section>

        {/* 4. Tooling & guides */}
        <Section
          eyebrow="Tooling & guides"
          title="SDKs, plugins, and formats."
          lede="Everything you need to integrate Astel assets into your workflow."
          className="docs-guides-section"
        >
          <div className="docs-guides-grid">
            {GUIDES.map((card, idx) => (
              <Reveal key={card.name} className="docs-guide-card-wrap" delay={idx * 60}>
                <a href={card.href} className="docs-guide-card">
                  <p className="docs-guide-card__name">{card.name}</p>
                  <p className="docs-guide-card__desc">{card.desc}</p>
                  <span className="docs-guide-card__arrow" aria-hidden>→</span>
                </a>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* CTA */}
        <Section className="docs-cta">
          <div className="docs-cta__inner">
            <h2>Deploy the gateway, then start building.</h2>
            <p>
              One command brings up the full stack locally. The interactive Swagger UI
              at <code className="docs-inline-code">/docs</code> lets you test every
              endpoint without writing a line of code.
            </p>
            <div className="docs-cta__buttons">
              <CtaLink to="/self-host" variant="primary">Self-host guide</CtaLink>
              <CtaLink to="/studio" variant="ghost">Open Studio</CtaLink>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
