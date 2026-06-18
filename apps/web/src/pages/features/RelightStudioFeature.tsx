import { RelightStudio } from "../../components/RelightStudio.tsx";
import { CtaLink } from "../../components/site/CtaLink.tsx";
import { PageHeader } from "../../components/site/PageHeader.tsx";
import { Reveal } from "../../components/site/Reveal.tsx";
import { Section } from "../../components/site/Section.tsx";

const APPEARANCE_POINTS = [
  {
    id: "albedo-separation",
    title: "Albedo separation",
    body: "The L4 stage decomposes each gaussian's colour into albedo (the intrinsic surface colour) and an illumination component (the contribution of the environment light captured at generation time). Switch to the Albedo view in the demo to see the unlit surface.",
  },
  {
    id: "sh-lighting",
    title: "Spherical harmonics environment",
    body: "Illumination is represented as L2 spherical harmonics — a compact, rotationally continuous encoding of the environment light. Rotating the HDRI preset spins the SH coefficients in-place, re-shading the asset without reloading any geometry.",
  },
  {
    id: "pbr-export",
    title: "PBR approximation export",
    body: "For engines that only consume coloured splats (no per-gaussian BRDF), L4 exports a PBR-approximation baked at a user-chosen reference illumination. The unlit albedo is also exported as a separate channel for custom relighting in VFX compositing.",
  },
  {
    id: "anti-baking",
    title: "Why this matters",
    body: "Competitors bake illumination into colour as the only option. An asset trained under studio lighting looks wrong in outdoor VFX, looks wrong in a game with dynamic time-of-day, looks wrong anywhere the light changes. L4 corrects this structurally.",
  },
] as const;

export function RelightStudioFeature(): React.JSX.Element {
  return (
    <div data-page="feature-relight-studio" className="page feature-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Feature — Relight Studio"
          title="Illumination separated. Always."
          lede="The L4 appearance layer decomposes per-splat albedo from baked illumination. Rotate any studio or daylight environment around the asset live — no baking, no re-generation. Correct lighting in every scene, automatically."
        />

        {/* Live demo */}
        <Section
          eyebrow="Live demo"
          title="Rotate the environment."
          lede="This is the real Relight Studio mounted with the bundled astrolabe sample. Switch between Albedo (un-lit), As-captured (the environment estimated from the asset), and Relit (a swapped studio environment). Drag the slider to spin the light."
          className="feature-demo-section"
        >
          <div className="feature-demo-panel feature-demo-panel--webgl">
            <RelightStudio relightUrl="/samples/astrolabe.relight.json" />
          </div>
          <Reveal>
            <p className="feature-demo-note mono">
              L4 appearance layer. Per-splat albedo + SH-L2 environment
              decomposition. Sample asset is fully generated — no ground truth.
            </p>
          </Reveal>
        </Section>

        {/* Appearance layer explainer */}
        <Section
          eyebrow="L4 appearance layer"
          title="Per-splat material decomposition."
          lede="L4 runs after L3 refinement on every asset. It is not optional — baked-lighting-only assets cannot be correct in environments the pipeline has never seen."
          className="feature-relight-explainer"
        >
          <div className="feature-pillars-grid">
            {APPEARANCE_POINTS.map((p, idx) => (
              <Reveal key={p.id} className="feature-pillar" delay={idx * 70}>
                <h3 className="feature-pillar__title">{p.title}</h3>
                <p className="feature-pillar__body">{p.body}</p>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* VFX / engine callout */}
        <Section
          eyebrow="Drop-in for VFX"
          title="Any light, any scene."
          className="feature-relight-vfx"
        >
          <Reveal>
            <div className="feature-relight-callout">
              <p>
                VFX compositing pipelines expect assets to respond to the
                on-set lighting, not the training-time studio environment. With
                L4, an Astel asset imported into Nuke, Houdini, or Blender
                carries its albedo and a reference illumination estimate.
                Lighters can apply their own environment directly to the albedo
                without re-generating the asset. The same mechanism powers the
                dynamic day-night cycle in UE5 and the real-time relighting in
                Unity's HDRP — no rebake required.
              </p>
            </div>
          </Reveal>
        </Section>

        {/* CTAs */}
        <Section className="feature-cta">
          <div className="feature-cta__inner">
            <h2>Generate an asset that relights.</h2>
            <p>
              Open the Studio and generate or capture an object. L4 appearance
              is included in every standard generation — open the Relight Studio
              to verify the decomposition before you export.
            </p>
            <div className="feature-cta__buttons">
              <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
              <CtaLink to="/features" variant="ghost">← All features</CtaLink>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
