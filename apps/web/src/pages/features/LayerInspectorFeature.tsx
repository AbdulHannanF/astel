import { useState } from "react";

import { LayerInspector } from "../../components/LayerInspector.tsx";
import { CtaLink } from "../../components/site/CtaLink.tsx";
import { PageHeader } from "../../components/site/PageHeader.tsx";
import { Reveal } from "../../components/site/Reveal.tsx";
import { Section } from "../../components/site/Section.tsx";
import { SAMPLE_LAYERS } from "../../lib/layers.ts";
import type { LayerId } from "../../lib/layers.ts";

const LAYER_DESCRIPTIONS: readonly { id: LayerId; title: string; body: string }[] = [
  {
    id: "L0",
    title: "L0 — Seed / Sparse Point Cloud",
    body: "The conditioning stage output: SfM points from photos/video, or generative latent samples from text. Carries per-point confidence. This is the first cheap preview tier — users iterate here before spending on refine.",
  },
  {
    id: "L1",
    title: "L1 — Dense Cloud",
    body: "Densified, metrically-scaled point cloud with normals and per-point semantic logits. Scale is grounded from video SfM or a VLM size estimator with an explicit confidence interval the user can override.",
  },
  {
    id: "L2",
    title: "L2 — Coarse Gaussians",
    body: "Fast feed-forward gaussians (LGM/TRELLIS-class) initialised from L1. Good enough to judge shape and identity. Third preview tier; inexpensive.",
  },
  {
    id: "L3",
    title: "L3 — Refined Surface Gaussians",
    body: "The hero layer. Optimization pass with surface-alignment regularisation (2DGS-class), anti-aliasing (Mip-Splatting), densification fixes, and normals per splat. Geometric error vs. L1 is measured and reported in the quality report.",
  },
  {
    id: "L4",
    title: "L4 — Appearance / Lighting",
    body: "Per-gaussian decomposed material: albedo, roughness, metallic, specular, emissive, and estimated environment illumination separated out. Assets relight correctly — illumination is never permanently baked into colour.",
  },
  {
    id: "L5",
    title: "L5 — Collision & Solidity",
    body: "Sparse voxel SDF derived from L3, convex decomposition proxy set for game-engine collision, watertight isosurface for the print path, and mass properties (centre of mass, inertia tensor, volume).",
  },
  {
    id: "L6",
    title: "L6 — Physics-Material & Semantics",
    body: "Per-region material classification (rigid / soft / cloth / fluid-adjacent), density estimate, friction and restitution defaults — produced by a VLM reasoning pass over renders and semantic labels.",
  },
  {
    id: "L7",
    title: "L7 — Dynamics",
    body: "For dynamic video captures: deformation field / 4DGS keyframes, exportable as animated splats or baked motion. Static objects carry this layer as locked.",
  },
] as const;

export function LayerInspectorFeature(): React.JSX.Element {
  const [visible, setVisible] = useState<ReadonlySet<LayerId>>(
    new Set<LayerId>(["L3"]),
  );

  const onToggle = (id: LayerId): void => {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div data-page="feature-layer-inspector" className="page feature-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Feature — Layer Inspector"
          title="Every layer, inspectable."
          lede="The Layer Inspector is the core interface of the Astel asset model. Scrub from L0 seed cloud to L7 dynamics, toggle any layer on or off, and export each one independently. Nothing is collapsed into a single opaque blob."
        />

        {/* Live demo */}
        <Section
          eyebrow="Live demo"
          title="The sample asset — try it."
          lede="This is the real Layer Inspector component with the bundled sample asset. Toggle layers, scrub the timeline. The sample is fully synthetic (generated, no ground truth) — only L3 and L4 are available; L0–L2 and L5–L6 are pending add-ons."
          className="feature-demo-section"
        >
          <div className="feature-demo-panel feature-demo-panel--inspector">
            <LayerInspector
              layers={SAMPLE_LAYERS}
              visible={visible}
              onToggle={onToggle}
            />
          </div>
        </Section>

        {/* Layer stack explainer */}
        <Section
          eyebrow="The layer stack"
          title="L0 → L7: eight layers, one asset."
          lede="Every Astel asset accumulates layers as the pipeline runs. Preview layers are cheap; refine and add-on layers cost more. All layers travel together in a single .astel package."
          className="feature-layer-explainer"
        >
          <div className="feature-layer-list">
            {LAYER_DESCRIPTIONS.map((l, idx) => (
              <Reveal key={l.id} className="feature-layer-item" delay={idx * 50}>
                <span className="feature-layer-item__id mono">{l.id}</span>
                <div className="feature-layer-item__body">
                  <h3 className="feature-layer-item__title">{l.title}</h3>
                  <p className="feature-layer-item__body-text">{l.body}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* Export pitch */}
        <Section
          eyebrow="Per-layer export"
          title="Export what you need."
          className="feature-export"
        >
          <Reveal>
            <div className="feature-export__card">
              <p>
                Each layer is independently exportable. L3 refined gaussians ship
                as <span className="mono">.ply</span>,{" "}
                <span className="mono">.spz</span>, or{" "}
                <span className="mono">.sog</span> (compressed delivery). The L4
                appearance layer exports a PBR-approximation for engines that
                only consume coloured splats. L5 collision proxies export as a
                physics setup JSON consumed automatically by the Unity and UE5
                plugins. The full stack ships as a single{" "}
                <span className="mono">.astel</span> archive for archival.
              </p>
            </div>
          </Reveal>
        </Section>

        {/* CTAs */}
        <Section className="feature-cta">
          <div className="feature-cta__inner">
            <h2>Generate your first layered asset.</h2>
            <p>
              Open the Studio to watch the Layer Stack build live — seed in
              seconds, L3 refined surface in minutes.
            </p>
            <div className="feature-cta__buttons">
              <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
              <CtaLink to="/how-it-works" variant="ghost">Pipeline overview →</CtaLink>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
