# Python SDK

`astel-sdk` wraps the Astel REST API for Python ≥ 3.12.

## Install

```bash
pip install astel-sdk
# With MCP server support:
pip install "astel-sdk[mcp]"
```

## Async client (recommended)

```python
from astel_sdk import AsyncAstelClient

async with AsyncAstelClient("http://localhost:8000") as client:
    # Text generation
    gen = await client.generate(prompt="a worn brass astrolabe")
    print(gen.id, gen.status)

    # Download all artifacts
    paths = await client.download_all_artifacts(gen.id, "out/")

    # Image generation
    with open("photo.jpg", "rb") as f:
        cap = await client.upload_capture(f, "photo.jpg")
    gen = await client.generate(modality="image", capture_id=cap.capture_id)
```

## Sync client

```python
from astel_sdk import AstelClient

client = AstelClient("http://localhost:8000")
gen = client.generate(prompt="a steel gear")
client.download_artifact(gen.id, "package.astel", "gear.astel")
```

## Preview + refine

```python
# Cheap preview (L0–L2 only, 3 credits)
preview = client.generate(prompt="a teapot", mode="preview")
print(f"Preview: {preview.id} — {preview.billing.total_credits} credits")

# Refine the preview (L3+, billed at 20 credits, not re-running L0–L2)
refined = client.generate(
    prompt="a teapot",
    mode="refine",
    refine_of=preview.id,
)
print(f"Refined: {refined.id} — {refined.billing.total_credits} credits")
```

## Polling for completion

```python
gen = client.generate(prompt="a ceramic vase")
# generate() returns immediately; poll until done:
gen = client.wait_for_generation(gen.id, poll_interval=3.0, max_wait=600.0)
if gen.is_ready:
    paths = client.download_all_artifacts(gen.id, "out/")
```

## API reference

- `AsyncAstelClient(base_url, *, timeout, api_key)` / `AstelClient(...)`
- `health() → dict`
- `pricing() → PricingResource`
- `upload_capture(file, filename, content_type) → CaptureRef`
- `generate(*, prompt, modality, capture_id, mode, refine_of) → Generation`
- `get_generation(id) → Generation`
- `wait_for_generation(id, poll_interval, max_wait) → Generation`
- `list_artifacts(id) → list[ArtifactRef]`
- `download_artifact(id, name, dest) → Path`
- `download_all_artifacts(id, dest_dir) → list[Path]`
