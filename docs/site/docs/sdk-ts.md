# TypeScript SDK

`@astel/sdk` is a dependency-free, `fetch`-based client for the Astel REST API,
usable from Node ≥ 20 and modern browsers.

## Install

```bash
pnpm add @astel/sdk
```

## Generate an asset

```ts
import { AstelClient } from "@astel/sdk";

const client = new AstelClient("http://localhost:8000");

// Text generation
const gen = await client.generate({ prompt: "a worn brass astrolabe" });

// generate() returns immediately; poll until terminal:
const done = await client.waitForGeneration(gen.id, { pollMs: 3000 });
const artifacts = await client.listArtifacts(done.id);
```

## Image generation

```ts
const file = await fetch("photo.jpg").then((r) => r.blob());
const cap = await client.uploadCapture(file, "photo.jpg");
const gen = await client.generate({ modality: "image", captureId: cap.capture_id });
```

## Preview + refine

```ts
const preview = await client.generate({ prompt: "a teapot", mode: "preview" });
const refined = await client.generate({
  prompt: "a teapot",
  mode: "refine",
  refineOf: preview.id,
});
```

## Downloading artifacts

```ts
const bytes = await client.downloadArtifact(gen.id, "package.astel"); // ArrayBuffer
const url = client.artifactUrl(gen, "l3.ply"); // absolute URL, or null
```

## API surface

- `new AstelClient(baseUrl?, { apiKey?, timeoutMs? })`
- `health()` → `{ status, service, version }`
- `pricing()` → `PricingResource`
- `uploadCapture(file, filename?)` → `CaptureRef`
- `generate({ prompt?, modality?, captureId?, mode?, refineOf? })` → `Generation`
- `getGeneration(id)` → `Generation`
- `waitForGeneration(id, { pollMs?, maxMs? })` → `Generation`
- `listArtifacts(id)` → `ArtifactRef[]`
- `downloadArtifact(id, name)` → `ArrayBuffer`
- `artifactUrl(generation, name)` → `string | null`

Errors surface as `AstelError` (carrying the HTTP `status` and parsed `body`).
