# Self-host guide

Run the full Astel stack on a single GPU box with one command.

## Requirements

- 1× GPU with ≥ 24 GB VRAM (RTX 3090/4090 or better)
- 64 GB RAM, NVMe storage
- CUDA 12.x + MSVC (Windows) or GCC (Linux)
- Docker (optional, for the full stack)
- Python 3.12+, Node 22+

See [Hardware requirements](hardware.md) for full specs.

## One-command start (dev mode)

```powershell
# Windows — GPU box native
pnpm run up           # starts API + web viewer
# or with the Temporal durable engine:
pnpm run up -Temporal
```

The web viewer opens at `http://localhost:5173`.  
The API is at `http://localhost:8000` (interactive docs at `/docs`).

## GPU producer

The GPU pipeline runs as a subprocess to keep torch out of the API env:

```powershell
# Enable GPU generation (requires CUDA + gsplat installed)
$env:ASTEL_PRODUCER = "gpu"
pnpm run up
```

Install the GPU env once:

```powershell
.\scripts\setup-gpu-env.ps1
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ASTEL_PRODUCER` | `stub` | `stub` (fast, no GPU) or `gpu` (real generation) |
| `ASTEL_ARTIFACT_DIR` | `./artifacts` | Where generated files are stored |
| `ASTEL_ENGINE` | `stub` | `stub` or `temporal` (durable task engine) |
| `ANTHROPIC_API_KEY` | — | Enables live LLM calls (Generation Spec + L6) |
| `ASTEL_LLM_LIVE` | `0` | Must be `1` AND key set to spend real LLM credits |
| `ASTEL_API_URL` | `http://localhost:8000` | Used by SDK and MCP server |

## Docker Compose (production)

```bash
docker compose -f infra/docker-compose.yml up
```

Includes: API, Postgres, MinIO, optional Temporal worker.

## Scaling

- **Preview pool**: L4/L40S-class GPUs for fast L2 previews.
- **Refine pool**: A100/H100 80 GB for L3 + L4 with high splat budgets.
- Worker autoscaling via queue depth (Temporal signals or Celery monitor).
- Stateless workers: all state in Postgres + S3 (MinIO locally).
