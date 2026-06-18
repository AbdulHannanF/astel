import { useEffect, useState } from "react";

import { TruthMeter } from "../../components/TruthMeter.tsx";
import { CtaLink } from "../../components/site/CtaLink.tsx";
import { PageHeader } from "../../components/site/PageHeader.tsx";
import { Reveal } from "../../components/site/Reveal.tsx";
import { Section } from "../../components/site/Section.tsx";
import { fetchSampleReport } from "../../lib/report.ts";
import type { QualityReport } from "../../lib/report.ts";

const HONESTY_PILLARS = [
  {
    id: "geometric-error",
    title: "Geometric error",
    body: "Chamfer distance between the L3 refined surface and the L1 dense point cloud (the closest thing to ground truth in a generative pipeline). Reported in millimetres. If the pipeline cannot measure this, the field shows N/A — not a fabricated number.",
  },
  {
    id: "scale-confidence",
    title: "Scale confidence",
    body: "How confident the pipeline is that the physical scale is correct. For text-to-3D, scale is estimated by a VLM with an explicit confidence interval you can override. For photo/video inputs, scale is grounded by SfM metric depth alignment. The confidence percentage is shown honestly — not inflated.",
  },
  {
    id: "provenance",
    title: "Measured vs. generated",
    body: "A provenance bar splits what came from real capture data versus what was generated or completed by a diffusion model. Generative in-fill for unseen regions is flagged separately — never silently merged with measured data.",
  },
  {
    id: "fidelity",
    title: "Render fidelity",
    body: "PSNR (peak signal-to-noise ratio) and SSIM (structural similarity) against held-out views not used in the refine pass. Higher is better. Reported as a self-consistency estimate for fully generated assets with no ground truth.",
  },
] as const;

export function TruthMeterFeature(): React.JSX.Element {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [errored, setErrored] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        setReport(await fetchSampleReport(ctrl.signal));
      } catch (e) {
        if (!(e instanceof DOMException)) setErrored(true);
      }
    })();
    return () => ctrl.abort();
  }, []);

  return (
    <div data-page="feature-truth-meter" className="page feature-page feature-page--truth">
      <div className="page-inner">
        <PageHeader
          eyebrow="Feature — Truth Meter"
          title="Radical honesty, per asset."
          lede="No competitor shows you their error metrics. The Truth Meter does — geometric error, scale confidence, render fidelity, and a provenance bar separating measured reality from generative in-fill. It becomes your trust brand."
        />

        {/* Live demo */}
        <Section
          eyebrow="Live demo"
          title="The sample quality report."
          lede="This is the real Truth Meter component loaded with the bundled sample report. The sample asset is fully synthetic — generated, no ground-truth capture — so the provenance bar shows 0% measured. That is honest, and that is the point."
          className="feature-demo-section feature-demo-section--truth"
        >
          <div className="feature-demo-panel feature-demo-panel--meter">
            <TruthMeter
              report={report}
              errored={errored}
              conditioning="none"
            />
          </div>
          <Reveal>
            <p className="feature-demo-note mono">
              Sample asset: fully synthetic (generated, no ground truth). Numbers
              are self-consistency estimates — correctly labelled as such.
            </p>
          </Reveal>
        </Section>

        {/* Honesty pillars */}
        <Section
          eyebrow="What it measures"
          title="Four dimensions of honesty."
          lede="Each dimension is independently valuable. Together they let you decide whether an asset is suitable for your pipeline — before you export or integrate it."
          className="feature-truth-pillars"
        >
          <div className="feature-pillars-grid">
            {HONESTY_PILLARS.map((p, idx) => (
              <Reveal key={p.id} className="feature-pillar" delay={idx * 70}>
                <h3 className="feature-pillar__title">{p.title}</h3>
                <p className="feature-pillar__body">{p.body}</p>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* Why it matters */}
        <Section
          eyebrow="Positioning"
          title="Trust is the differentiator."
          className="feature-truth-why"
        >
          <Reveal>
            <div className="feature-truth-callout">
              <p className="feature-truth-callout__text">
                Generative 3D tools routinely ship plausible-looking assets that
                fall apart the moment you try to use them in a physics sim, a
                printable file, or a measured VFX pipeline. Astel measures and
                reports the error rather than hiding it. Studios, game teams, and
                industrial customers can make an informed decision about whether
                to accept an asset, refine it further, or submit a capture with
                real ground truth. The Truth Meter is also the mechanism that
                prevents Astel from silently hallucinating over measured reality —
                the confidence channel is a first-class output, not a footnote.
              </p>
            </div>
          </Reveal>
        </Section>

        {/* CTAs */}
        <Section className="feature-cta feature-cta--truth">
          <div className="feature-cta__inner">
            <h2>See the numbers for your own asset.</h2>
            <p>
              Generate or capture an asset in the Studio and the Truth Meter
              populates with real quality metrics — not placeholders.
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
