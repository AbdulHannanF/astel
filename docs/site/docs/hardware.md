# Hardware requirements

Astel is **local-first**: the entire pipeline runs on a single high-VRAM consumer
GPU ("patient mode") and scales out to cloud GPU fleets ("fast mode"). There is no
hard cloud dependency for core generation.

## Local dev / self-host minimum

- **1× 24 GB GPU** (RTX 3090 / 4090-class)
- **64 GB RAM**, NVMe storage
- Full pipeline in patient mode; an L3 refine for a ~1M-splat object targets
  **≤ 15–30 min**.

## Recommended local

- **RTX 5090 / 6000-Ada-class, 32–48 GB** → cinematic splat budgets become feasible.

## Cloud production

- **Preview pool** on L4 / L40S-class GPUs (cheap L0–L2 tiers, batch multi-tenant).
- **Refine + training pool** on A100 / H100 80 GB for the L3+ spend.
- Spot-instance tolerant via resumable, cacheable stages.

## CPU-bound stages

SfM (COLMAP/GLOMAP), SDF extraction, and convex decomposition are **CPU-heavy** and
sized on separate nodes — don't burn GPU nodes on them.

## Model training / fine-tuning (later)

Multi-node H100s, deferred until product telemetry justifies it. Astel launches on
adapted, permissively-licensed open checkpoints; the fine-tuning gate and cost flag
are tracked in the repo's post-M6 roadmap (`docs/research/18-post-m6-roadmap.md`,
Track T).

> The CPU **stub** producer (procedural placeholder geometry) needs none of the
> above and runs anywhere — it's the default so the web app, API, and tests work on
> any machine. Real geometry needs the GPU stack.
