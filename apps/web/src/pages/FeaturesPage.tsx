import { Link } from "react-router-dom";

import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";

interface FeatureEntry {
  id: string;
  label: string;
  description: string;
  link: string;
  accent: "brass" | "measured";
  layer: string;
}

const FEATURE_ENTRIES: readonly FeatureEntry[] = [
  {
    id: "layer-inspector",
    label: "Layer Inspector",
    description:
      "Scrub through L0→L7 — seed cloud, dense cloud, coarse gaussians, refined surface, appearance, collision, physics-material, dynamics. Every layer is independently visible, inspectable, and exportable. This is the asset model as a first-class UI surface, not an afterthought.",
    link: "/features/layer-inspector",
    accent: "brass",
    layer: "L0 – L7",
  },
  {
    id: "truth-meter",
    label: "Truth Meter",
    description:
      "Per-asset honesty report: geometric error (Chamfer distance) against the source point cloud, PSNR/SSIM fidelity score, scale confidence interval, and a provenance bar that separates what was measured from real data versus what was generated. No competitor shows you this.",
    link: "/features/truth-meter",
    accent: "measured",
    layer: "Quality report",
  },
  {
    id: "physics-sandbox",
    label: "Physics Sandbox",
    description:
      "Drop the asset on a floor and poke it. Mass is derived from the L5 watertight volume multiplied by the L6 physics-material density estimate, so the weight feels right. Rigid-body simulation preview runs live in the browser — the world-awareness of the asset made tactile.",
    link: "/features/physics-sandbox",
    accent: "brass",
    layer: "L5 + L6",
  },
  {
    id: "relight-studio",
    label: "Relight Studio",
    description:
      "Rotate a studio, daylight, or custom HDRI environment around the asset live. The L4 appearance layer separates per-splat albedo from baked illumination, so the asset responds correctly to any light source — not just the one it was captured under.",
    link: "/features/relight-studio",
    accent: "brass",
    layer: "L4",
  },
] as const;

export function FeaturesPage(): React.JSX.Element {
  return (
    <div data-page="features" className="page features-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Features"
          title="World-aware splats, inspectable by design."
          lede="Every Astel asset is a layered stack — geometry, appearance, collision, and physics-material bound together. These four tools surface that stack as interactive, demoable experiences."
        />

        <Section className="features-overview">
          <div className="features-deep-grid">
            {FEATURE_ENTRIES.map((f, idx) => (
              <Reveal key={f.id} delay={idx * 80}>
                <Link
                  to={f.link}
                  className={`feature-card feature-card--${f.accent}`}
                >
                  <div className="feature-card__meta">
                    <span className="feature-card__layer mono">{f.layer}</span>
                  </div>
                  <p className={`feature-card__name feature-card__name--${f.accent}`}>
                    {f.label}
                  </p>
                  <p className="feature-card__desc">{f.description}</p>
                  <span className="feature-card__arrow" aria-hidden>→</span>
                </Link>
              </Reveal>
            ))}
          </div>
        </Section>

        <Section className="features-cta">
          <div className="features-cta__inner">
            <h2>See every layer live.</h2>
            <p>
              Open the Studio to generate your own asset and watch the Layer
              Stack build in real time — L0 seed, L1 dense cloud, L2 coarse,
              L3 refined surface.
            </p>
            <div className="features-cta__buttons">
              <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
              <CtaLink to="/how-it-works" variant="ghost">How it works →</CtaLink>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
