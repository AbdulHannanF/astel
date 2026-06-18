import { PhysicsSandbox } from "../../components/PhysicsSandbox.tsx";
import { CtaLink } from "../../components/site/CtaLink.tsx";
import { PageHeader } from "../../components/site/PageHeader.tsx";
import { Reveal } from "../../components/site/Reveal.tsx";
import { Section } from "../../components/site/Section.tsx";

const WORLD_AWARENESS_POINTS = [
  {
    id: "l5-collision",
    title: "L5 — Collision & Solidity",
    body: "A sparse voxel SDF is extracted from the L3 refined surface via marching cubes, then decomposed into a convex hull set via CoACD. The result: a game-engine-ready collision proxy that actually matches the visible geometry, not a hand-authored bounding box.",
  },
  {
    id: "l6-material",
    title: "L6 — Physics-Material",
    body: "A VLM reasoning pass over the L3 renders and L1 semantic labels classifies each region: wood, steel, ceramic, plastic, cloth. Density, friction, and restitution defaults follow. Mass is then volume (L5) × density (L6) — so the astrolabe weighs what an astrolabe should weigh.",
  },
  {
    id: "mpm",
    title: "MPM simulation (roadmap)",
    body: "PhysGaussian-style material point method simulation directly on gaussian kernels is the target for the full physics preview. The current sandbox uses a single rigid-body model — honest about where we are, clear about where we are going.",
  },
  {
    id: "engine-export",
    title: "Engine integration",
    body: "The L5 convex proxies and L6 material properties export automatically into the Unity and UE5 plugins as physics setups. Drop the .astel asset into a scene and the object already has correct mass, friction, and restitution — no manual configuration.",
  },
] as const;

export function PhysicsSandboxFeature(): React.JSX.Element {
  return (
    <div data-page="feature-physics-sandbox" className="page feature-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Feature — Physics Sandbox"
          title="World-awareness you can feel."
          lede="Drop the asset on a floor. Poke it. Mass is derived from the L5 watertight volume and L6 physics-material density — so the object responds the way the real thing would, not the way a random default would."
        />

        {/* Live demo */}
        <Section
          eyebrow="Live demo"
          title="Interact with the sample asset."
          lede="This is the real Physics Sandbox mounted with the bundled sample asset. Drop the object, poke it, change the material, reset. Volume is estimated from the L3 bounding sphere — L5 solidification is a pending add-on for this sample."
          className="feature-demo-section"
        >
          <div className="feature-demo-panel feature-demo-panel--webgl">
            <PhysicsSandbox splatUrl="/samples/astel-sample.ply" />
          </div>
          <Reveal>
            <p className="feature-demo-note mono">
              Rigid-body simulation. Volume from bounding-sphere estimate (L5
              solidification pending). Mass = volume × selected material density.
            </p>
          </Reveal>
        </Section>

        {/* World-awareness explainer */}
        <Section
          eyebrow="How it works"
          title="L5 + L6: collision and material."
          lede="World-awareness is not a post-process. It is a pipeline stage that derives collision geometry and physics material from the same data that produced the visible asset."
          className="feature-physics-explainer"
        >
          <div className="feature-pillars-grid">
            {WORLD_AWARENESS_POINTS.map((p, idx) => (
              <Reveal key={p.id} className="feature-pillar" delay={idx * 70}>
                <h3 className="feature-pillar__title">{p.title}</h3>
                <p className="feature-pillar__body">{p.body}</p>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* Engine integration callout */}
        <Section
          eyebrow="Drop-in usable"
          title="From sandbox to engine."
          className="feature-physics-engine"
        >
          <Reveal>
            <div className="feature-physics-callout">
              <p>
                The Physics Sandbox is a preview of what happens when you import
                the asset into Unreal Engine 5 or Unity. The same L5 convex
                proxies and L6 material assignments are consumed by the engine
                plugins automatically. A generated astrolabe imported into a
                physics scene falls, bounces, and slides with brass-appropriate
                friction — without any manual setup.
              </p>
            </div>
          </Reveal>
        </Section>

        {/* CTAs */}
        <Section className="feature-cta">
          <div className="feature-cta__inner">
            <h2>Generate a world-aware asset.</h2>
            <p>
              Open the Studio, generate or capture an object, and add the L5
              collision and L6 physics-material add-ons to see your asset behave
              correctly in any physics engine.
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
