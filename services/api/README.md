# services/api

The Astel API gateway — FastAPI, async SQLAlchemy 2, SSE progress. M1 skeleton.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz` | Liveness probe |
| `GET`  | `/v1/pipeline` | Static stage specs (L0→L3) for the progress rail |
| `POST` | `/v1/generations` | Submit `{modality, prompt}` → task id + events URL |
| `GET`  | `/v1/generations/{id}` | Fetch a generation's status |
| `GET`  | `/v1/generations/{id}/events` | **SSE** stream of L0→L3 progress |

Interactive docs at `/docs` when running.

## Architecture seam

The pipeline runs behind a `TaskEngine` protocol (`engine.py`). M1 ships
`InProcessStubEngine`, which simulates the four preview/refine stages with
realistic durations and shaped metrics. A `TemporalTaskEngine` implementing the
same protocol lands next session — the routes do not change.

## Dev

```bash
uv sync
uv run uvicorn astel_api.main:app --reload --port 8000   # http://127.0.0.1:8000
```

SQLite by default. For Postgres set:

```bash
export ASTEL_DATABASE_URL=postgresql+asyncpg://user:pass@host/astel
```

## Test / lint / type-check

```bash
uv run pytest
uv run ruff check src tests
uv run mypy
```
