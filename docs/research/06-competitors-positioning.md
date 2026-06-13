# RA6 — Competitive Map & Positioning One-Pager

*Verified 2026-06-12. Companion to [meshy-analysis](../meshy-analysis.md) (current through April 2026). M0 deliverable.*

## Competitive map, June 2026

| Player | Representation | Inputs | Their edge | Their ceiling |
|---|---|---|---|---|
| **Meshy** (~$40M ARR, Apr 2026) | Mesh+PBR | text, image(s) | Pipeline-ready assets, printing integrations, API, 10M+ users | Hallucinated (never measured) geometry; no video input; no physics; cloud-only; mesh representation ceiling on fuzzy/photoreal matter |
| Tripo, Rodin/Hyper3D, Hunyuan3D | Mesh | text, image | Model-quality race in Meshy's lane | Same lane, same ceilings; Hunyuan license excludes EU/UK/KR |
| Luma AI (Genie/capture) | NeRF/splats | video/photos | Photoreal real-world capture, $1/scene | Capture only — no generation; no asset semantics ([review](https://www.thefuture3d.com/software/luma-ai/)) |
| Polycam, Scaniverse | Splats/scan | photos/LiDAR | Export breadth (15+ formats), pro scanning | Capture only; no generation, no physics |
| **World Labs Marble** | **Splats (worlds)** | text, image, video, coarse layouts | Generative *worlds*; edit/expand/combine; exports splats+meshes+video; UE/LED-stage workflows; **NVIDIA Isaac Sim integration** (robotics environments); Marble Labs content hub ([blog](https://www.worldlabs.ai/blog/marble-world-model), [NVIDIA](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/)) | **Scene/world-centric, not asset-centric**: no per-asset metric accuracy claims, no collision/mass/material semantics per object, no print path, no measured-vs-generated honesty channel |
| Research recipes (TRELLIS et al.) | various | — | What we build on | Not productized — no pipeline, formats, API, QA |

World Labs' Isaac Sim move confirms the simulation direction matters commercially — and that
the *asset-level* version of it (collision + mass + materials bound to a generated splat) is
still unclaimed. They also ship Spark (MIT), the best web splat renderer — which we adopt.

## The open square (unchanged, sharpened)

**A production-grade, layered, physically-grounded, *generative + capture* Gaussian-splat
ASSET platform.** Capture tools have splats but no generation. Mesh generators have product
polish but the wrong representation, no measurement, no physics. World Labs has generative
splat *worlds* but not engine-ready *assets with physical semantics*. Nobody offers:

1. **Measured honesty** — Truth Meter: geometric error vs source, scale confidence,
   hallucination heatmap. No competitor dares show error bars. This is the trust brand.
2. **World-awareness per asset** — collision proxies, mass/inertia, physics materials,
   relightable decomposed appearance: drop into UE5/Unity/Isaac and it *behaves*.
3. **All three input modalities** converging on one layered representation (L0–L7), each
   layer inspectable and exportable.
4. **Local-first/self-host** — the film/defense/industrial wedge no cloud-only incumbent
   (Meshy *or* World Labs) can follow without rebuilding their business.
5. **Splat-native print path** — physical loop without ever selling a mesh.

## Positioning statement (draft v0.1)

> AURIGA generates engine-ready Gaussian-splat assets from text, photos, or video — with
> measured geometric accuracy, real-world scale, collision, mass, and relightable materials
> built into every asset. It tells you what it measured and what it imagined. It runs in our
> cloud or entirely inside yours.

Tagline candidates (pick at M1 naming pass): "Assets that know what they are." /
"Real geometry. Real physics. Real honesty." / "The world-aware asset engine."

**Launch-messaging guardrail** (added second pass, see [RA7b](07-free-tier-consumer-strategy.md)):
lead with capture + print + honesty + free-tier generosity. Never claim text-to-3D parity with
Meshy until the blind-eval harness proves it — the harness results themselves are the
marketing asset when they're ready.

## Strategic risks (feeds RISKS.md)

- **R-C1**: World Labs descends from worlds to assets (they have the model talent + Spark
  distribution). Mitigation: speed on M2 capture accuracy + L5/L6 semantics; their DNA and
  capital are pointed at worlds/AGI-spatial, not asset pipelines.
- **R-C2**: Meshy adds splat export as a checkbox feature (their repr ceiling becomes a
  marketing checkbox). Mitigation: layered accuracy + physics is not a checkbox; Truth Meter
  reframes the comparison.
- **R-C3**: KHR ratification slips → export-standard bet delayed. Mitigation: .ply/.spz/.sog
  carry us; glTF is M5 anyway.

## Sources

- https://www.worldlabs.ai/blog/marble-world-model · https://www.worldlabs.ai/labs · https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/ · https://www.worldlabs.ai/case-studies/bringing-marble-to-life
- https://www.thefuture3d.com/software/luma-ai/ · https://radiancefields.com/octanerender-2026-released-and-2027-roadmap-announced
- https://www.indiehackers.com/post/best-ai-3d-model-generator-in-2026-i-tested-9-of-the-best-and-here-is-what-i-found-70ecab1a0a
- ../meshy-analysis.md (in-repo, through Apr 2026)
