import { Link } from "react-router-dom";

import { Viewport } from "../components/Viewport.tsx";
import { CtaLink } from "../components/site/CtaLink.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";
import { SAMPLE_LAYERS } from "../lib/layers.ts";

/** Modality cards: three input paths, honest about capability (not shipping dates). */
const MODALITIES = [
  {
    id: "text",
    label: "Text",
    eyebrow: "Modality 01",
    headline: "Describe it.",
    body: "A prompt becomes a structured Generation Spec — object class, parts, materials, target scale — which drives multi-view synthesis and feed-forward gaussian reconstruction.",
    icon: "T",
  },
  {
    id: "photos",
    label: "Photos",
    eyebrow: "Modality 02",
    headline: "Photograph it.",
    body: "One image or a multi-view set. Pose estimation via MASt3R-class feed-forward reconstruction initialises a dense point cloud from real data; generative completion is used only for unseen regions and is flagged in the provenance channel.",
    icon: "P",
  },
  {
    id: "video",
    label: "Video",
    eyebrow: "Modality 03",
    headline: "Film it.",
    body: "Orbit or handheld footage of any real object. Frame selection, metric depth alignment, and dynamic scene handling via 4DGS — with measured scale confidence reported honestly in the Truth Meter.",
    icon: "V",
  },
] as const;

/** Flagship features (all demoable in this app). */
const FEATURES = [
  {
    id: "layer-inspector",
    label: "Layer Inspector",
    description: "Scrub through L0→L7: seed cloud, dense cloud, coarse gaussians, refined surface, material, collision, physics, dynamics. Toggle layers live. Export any layer independently.",
    link: "/features/layer-inspector",
    accent: "brass",
  },
  {
    id: "truth-meter",
    label: "Truth Meter",
    description: "Per-asset honesty report: geometric error (Chamfer) vs. source data, PSNR/SSIM, scale confidence interval, and a provenance map distinguishing measured vs. generated regions.",
    link: "/features/truth-meter",
    accent: "measured",
  },
  {
    id: "physics-sandbox",
    label: "Physics Sandbox",
    description: "Drop the asset on a floor, poke it. MPM/rigid-body simulation preview using the L5 collision proxies and L6 physics-material assignments, streamed to the browser.",
    link: "/features/physics-sandbox",
    accent: "brass",
  },
  {
    id: "relight-studio",
    label: "Relight Studio",
    description: "Rotate HDRI environments around the asset live. The L4 appearance layer decomposes per-splat albedo from baked illumination — so the asset responds correctly to any light.",
    link: "/features/relight-studio",
    accent: "brass",
  },
] as const;

/** Availability labels for layer chips. */
const AVAIL_LABEL: Record<string, string> = {
  available: "live",
  pending: "add-on",
  locked: "N/A",
};

export function HomePage(): React.JSX.Element {
  return (
    <div data-page="home" className="page home-page">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="home-hero">
        <div className="home-hero__viewport">
          <Viewport sampleUrl="/samples/astel-sample.ply" splatVisible />
        </div>
        <div className="home-hero__overlay">
          <div className="home-hero__copy">
            <p className="home-hero__eyebrow mono">
              Gaussian splat assets — text · photos · video
            </p>
            <h1 className="home-hero__headline">
              The 3D asset platform<br />
              built on radical honesty.
            </h1>
            <p className="home-hero__lede">
              Astel generates geometry-accurate, world-aware Gaussian splat assets.
              Every dimension reported. Every claim verified. No mesh products.
              No hype.
            </p>
            <div className="home-hero__ctas">
              <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
              <CtaLink to="/how-it-works" variant="ghost">See how it works</CtaLink>
            </div>
          </div>
        </div>
      </section>

      {/* ── Modalities ───────────────────────────────────────────────────── */}
      <Section
        eyebrow="Three input paths"
        title="Text. Photos. Video."
        lede="All three converge at the same layered asset format — L0 seed through L7 dynamics. The path to that asset differs; the output contract doesn't."
        className="home-modalities"
      >
        <div className="modality-grid">
          {MODALITIES.map((m) => (
            <Reveal key={m.id} className="modality-card">
              <p className="modality-card__eyebrow mono">{m.eyebrow}</p>
              <div className="modality-card__icon mono" aria-hidden>{m.icon}</div>
              <h3 className="modality-card__headline">{m.headline}</h3>
              <p className="modality-card__body">{m.body}</p>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ── Features ─────────────────────────────────────────────────────── */}
      <Section
        eyebrow="Flagship features"
        title="What makes Astel different."
        className="home-features"
      >
        <div className="features-grid">
          {FEATURES.map((f) => (
            <Reveal key={f.id}>
              <Link to={f.link} className={`feature-card feature-card--${f.accent}`}>
                <h3 className="feature-card__name">{f.label}</h3>
                <p className="feature-card__desc">{f.description}</p>
                <span className="feature-card__arrow" aria-hidden>→</span>
              </Link>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ── Layer Stack ──────────────────────────────────────────────────── */}
      <Section
        eyebrow="The layered asset model"
        title="L0 → L7: every asset is a Layer Stack."
        lede="Layers are computed progressively. Preview layers are cheap; refinement and add-ons are metered separately. Every layer is independently inspectable and exportable."
        className="home-layers"
      >
        <div className="layer-stack">
          {SAMPLE_LAYERS.map((layer) => (
            <Reveal key={layer.id} className="layer-row">
              <div className="layer-row__id mono">{layer.id}</div>
              <div className="layer-row__body">
                <span className="layer-row__name">{layer.name}</span>
                <span className="layer-row__blurb">{layer.blurb}</span>
              </div>
              <div className="layer-row__meta">
                <span className="layer-row__kind mono">{layer.kind}</span>
                <span
                  className={`layer-row__avail layer-row__avail--${layer.availability}`}
                >
                  {AVAIL_LABEL[layer.availability] ?? layer.availability}
                </span>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ── Truth Meter / Honesty differentiator ─────────────────────────── */}
      <Section
        eyebrow="Radical honesty"
        title="We show you what we measured."
        lede="The Truth Meter is a per-asset quality report on every generation. These numbers are from the bundled reference asset — fully synthetic, no ground truth available."
        className="home-truth"
      >
        <div className="truth-banner">
          <p className="truth-banner__note mono">
            Reference asset · generated · no ground-truth comparison
          </p>
          <div className="truth-stats">
            <Reveal className="truth-stat">
              <span className="truth-stat__value mono">48,000</span>
              <span className="truth-stat__label">splats</span>
            </Reveal>
            <Reveal className="truth-stat" delay={60}>
              <span className="truth-stat__value mono">0.9 mm</span>
              <span className="truth-stat__label">Chamfer vs L1</span>
            </Reveal>
            <Reveal className="truth-stat" delay={120}>
              <span className="truth-stat__value mono">31.2 dB</span>
              <span className="truth-stat__label">PSNR</span>
            </Reveal>
            <Reveal className="truth-stat" delay={180}>
              <span className="truth-stat__value mono">0.946</span>
              <span className="truth-stat__label">SSIM</span>
            </Reveal>
            <Reveal className="truth-stat" delay={240}>
              <span className="truth-stat__value mono">0.182 m</span>
              <span className="truth-stat__label">longest axis</span>
            </Reveal>
            <Reveal className="truth-stat" delay={300}>
              <span className="truth-stat__value mono">0.41</span>
              <span className="truth-stat__label">scale confidence</span>
            </Reveal>
          </div>
          <p className="truth-banner__cta-note">
            No competitor shows this. It is our trust brand.
          </p>
          <CtaLink to="/features/truth-meter" variant="ghost">
            About the Truth Meter →
          </CtaLink>
        </div>
      </Section>

      {/* ── Closing CTA ──────────────────────────────────────────────────── */}
      <Section className="home-closing">
        <div className="closing-cta">
          <p className="closing-cta__eyebrow mono">Ready to start</p>
          <h2 className="closing-cta__headline">
            Build assets that behave in the real world.
          </h2>
          <p className="closing-cta__lede">
            Open the Studio to generate your first layered Gaussian splat — or read
            the docs to self-host the full pipeline on your own GPU.
          </p>
          <div className="closing-cta__buttons">
            <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
            <CtaLink to="/self-host" variant="ghost">Self-host guide</CtaLink>
          </div>
        </div>
      </Section>

    </div>
  );
}
