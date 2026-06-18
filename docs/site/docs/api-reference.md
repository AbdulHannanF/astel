# API reference

The Astel API is a JSON REST API (FastAPI). Interactive docs at `/docs` when the server is running.

## Base URL

```
http://localhost:8000          # local / self-hosted
https://api.astel.ai/v1       # cloud (coming)
```

## Authentication

Pass a bearer token in the `Authorization` header (optional for self-hosted):

```
Authorization: Bearer <api_key>
```

---

## Health

### `GET /healthz`

Liveness probe.

**Response**
```json
{ "status": "ok", "service": "astel-api", "version": "0.1.0" }
```

---

## Pricing

### `GET /v1/pricing`

Returns the credit price schedule (`astel_api.billing.schedule_dict`).

**Response**
```json
{
  "schema": "astel.pricing/v0",
  "credit_usd_rate": 0.01,
  "layers": [
    { "code": "L0", "label": "Seed point cloud", "tier": "preview", "credits": 1.0 },
    { "code": "L3", "label": "Refined surface gaussians", "tier": "refine", "credits": 20.0 }
  ],
  "modes": { "preview": ["L0", "L1", "L2"], "refine": ["L3", "L4", "L5", "L6", "L7", "PRINT"] },
  "notes": ["L0–L2 previews are cheap; L3 refine is the main spend; ..."]
}
```

---

## Captures

### `POST /v1/captures`

Upload a raw image or video file. Returns a `capture_id` to use in a generation.

**Request**: multipart/form-data, field `file`.

**Response** `201`
```json
{
  "capture_id": "capture-<uuid>",
  "filename": "photo.jpg",
  "content_type": "image/jpeg",
  "bytes": 204800
}
```

---

## Generations

### `POST /v1/generations`

Submit a generation.

**Request body**
```json
{
  "modality": "text",
  "prompt": "a worn brass astrolabe on a wooden base",
  "mode": "refine",
  "capture_id": null,
  "refine_of": null
}
```

| Field | Type | Description |
|---|---|---|
| `modality` | `text` \| `image` \| `video` | Input type |
| `prompt` | string? | Text description (text modality) |
| `capture_id` | string? | From `POST /v1/captures` (image/video) |
| `mode` | `preview` \| `refine` | Preview = fast + cheap; refine = full pipeline |
| `refine_of` | string? | ID of a prior preview to refine (skips re-billing L0–L2) |

**Response** `201` → `GenerationResource`

### `GET /v1/generations/{id}`

Fetch a generation by ID.

### `GET /v1/generations/{id}/events`

Server-Sent Events stream of pipeline progress. Each `progress` event carries a `ProgressEvent` JSON:

```json
{ "stage": "L3_refine", "progress": 0.72, "status": "running", "splats": null }
```

### `GET /v1/generations/{id}/artifacts/{name}`

Download a named artifact. Common names:

| Name | Description |
|---|---|
| `l3.ply` | INRIA-layout PLY (archival) |
| `l3.spz` | Niantic SPZ compressed |
| `l3.sog` | PlayCanvas SOG |
| `package.astel` | Full layered package (zip) |
| `quality-report.json` | Truth Meter report |
| `l4.json` | Appearance layer descriptor |
| `l5.stl` | Watertight mesh (print/physics only) |
| `l6.json` | Physics material + articulation |
| `credit-ledger.json` | Billing breakdown |

---

## GenerationResource schema

```json
{
  "id": "uuid",
  "modality": "text",
  "prompt": "...",
  "status": "queued | running | succeeded | failed",
  "created_at": "ISO8601",
  "events_url": "/v1/generations/<id>/events",
  "artifacts": [{ "name": "...", "url": "...", "content_type": "...", "bytes": 0 }],
  "mode": "refine",
  "refine_of": null,
  "billing": { "total_credits": 21, "total_usd": 0.21, "lines": [] },
  "conditioning": "prompt | image | video | none"
}
```
