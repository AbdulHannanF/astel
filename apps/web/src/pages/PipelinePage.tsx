import { useEffect, useState } from "react";

import { fetchPipeline } from "../lib/api.ts";
import type { PipelineStageSpec } from "../lib/api.ts";
import { SAMPLE_LAYERS } from "../lib/layers.ts";
import type { LayerDef } from "../lib/layers.ts";
import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";

/** Availability badge text. */
const AVAIL_TEXT: Record<LayerDef["availability"], string> = {
  available: "live",
  pending: "add-on",
  locked: "N/A",
};

function formatDuration(seconds: number): string {
  if (seconds < 60) return `~${seconds}s`;
  const m = Math.round(seconds / 60);
  return `~${m}m`;
}

export function PipelinePage(): React.JSX.Element {
  const [stages, setStages] = useState<PipelineStageSpec[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        setStages(await fetchPipeline(ctrl.signal));
      } catch (e) {
        if (!(e instanceof DOMException)) setError(true);
      }
    })();
    return () => ctrl.abort();
  }, []);

  // Total nominal seconds for the live pipeline (L0–L3).
  const totalSeconds = stages?.reduce((s, st) => s + st.nominal_seconds, 0) ?? 0;

  return (
    <div data-page="pipeline" className="page pipeline-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="How it works"
          title="From prompt to world-aware splat."
          lede="Every asset passes through the same pipeline — cheap preview first, expensive refine on demand. The Layer Stack accumulates as the pipeline runs."
        />

        {/* Live pipeline: L0–L3 */}
        <Section
          eyebrow="Live pipeline"
          title="L0 → L3: the production stages."
          lede="These are the stages /v1/pipeline runs today, weighted by the nominal timing the API reports — previews are quick, the L3 refine dominates. Real wall-time scales with splat budget and GPU (a 1M-splat refine targets ~15–30 min on a 4090)."
          className="pipeline-live"
        >
          {error && (
            <div className="pipeline-error" role="alert">
              <p>Couldn&rsquo;t reach the gateway — pipeline schedule unavailable.</p>
            </div>
          )}

          {!stages && !error && (
            <div className="pipeline-skeleton" aria-busy="true" aria-label="Loading pipeline stages">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="skeleton-stage" />
              ))}
            </div>
          )}

          {stages && (
            <>
              <div className="pipeline-timeline">
                {stages.map((stage, idx) => {
                  const pct = totalSeconds > 0
                    ? Math.round((stage.nominal_seconds / totalSeconds) * 100)
                    : 25;
                  return (
                    <Reveal key={stage.stage} className="pipeline-step" delay={idx * 80}>
                      <div className="pipeline-step__connector" aria-hidden>
                        <div className="pipeline-step__line" />
                        <div className="pipeline-step__dot" />
                      </div>
                      <div className="pipeline-step__card">
                        <div className="pipeline-step__header">
                          <span className="pipeline-step__layer mono">{stage.layer}</span>
                          <span className="pipeline-step__duration mono">
                            {formatDuration(stage.nominal_seconds)}
                          </span>
                        </div>
                        <h3 className="pipeline-step__label">{stage.label}</h3>
                        <p className="pipeline-step__desc">{stage.description}</p>
                        <div className="pipeline-step__bar" aria-label={`${pct}% of pipeline time`}>
                          <div
                            className="pipeline-step__bar-fill"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <p className="pipeline-step__pct mono">{pct}% of pipeline time</p>
                      </div>
                    </Reveal>
                  );
                })}
              </div>

              <div className="pipeline-summary">
                <p className="pipeline-summary__text mono">
                  Nominal weighting from /v1/pipeline · real wall-time scales with splat budget + GPU
                </p>
              </div>
            </>
          )}
        </Section>

        {/* Full L0→L7 layer stack */}
        <Section
          eyebrow="The asset model"
          title="L0 → L7: the full Layer Stack."
          lede="The pipeline produces L0–L3. L4–L7 are add-ons applied after the refine stage. Each layer is independently viewable in the Layer Inspector and independently exportable."
          className="pipeline-stack"
        >
          <div className="layer-full-stack">
            {SAMPLE_LAYERS.map((layer, idx) => {
              const isLive = ["L0", "L1", "L2", "L3"].includes(layer.id);
              return (
                <Reveal key={layer.id} className="layer-full-row" delay={idx * 50}>
                  <div className={`layer-full-row__marker${isLive ? " layer-full-row__marker--live" : ""}`} aria-hidden />
                  <div className="layer-full-row__id mono">{layer.id}</div>
                  <div className="layer-full-row__body">
                    <div className="layer-full-row__top">
                      <span className="layer-full-row__name">{layer.name}</span>
                      {isLive && (
                        <span className="layer-full-row__badge layer-full-row__badge--live mono">
                          pipeline
                        </span>
                      )}
                      {!isLive && (
                        <span className="layer-full-row__badge layer-full-row__badge--addon mono">
                          add-on
                        </span>
                      )}
                    </div>
                    <p className="layer-full-row__blurb">{layer.blurb}</p>
                  </div>
                  <div className="layer-full-row__meta">
                    <span className="layer-full-row__kind mono">{layer.kind}</span>
                    <span className={`layer-full-row__avail layer-full-row__avail--${layer.availability}`}>
                      {AVAIL_TEXT[layer.availability]}
                    </span>
                  </div>
                </Reveal>
              );
            })}
          </div>

          <div className="pipeline-stack__note">
            <p className="mono">
              Availability reflects the bundled reference asset — fully synthetic,
              L3 + L4 produced, L5–L6 pending add-on, L7 locked (static object).
            </p>
          </div>
        </Section>

        {/* Modality summary */}
        <Section
          eyebrow="Three input paths"
          title="All paths converge at L3."
          className="pipeline-modalities"
        >
          <div className="modality-paths">
            <Reveal className="modality-path">
              <p className="modality-path__eyebrow mono">Text</p>
              <p className="modality-path__flow mono">
                prompt → Generation Spec → multi-view synthesis → L2 → L3
              </p>
              <p className="modality-path__desc">
                An LLM parses the prompt into a structured spec. Multi-view images are
                synthesised, then a feed-forward gaussian model produces L2. L3 refine
                runs with multi-view-diffusion guidance.
              </p>
            </Reveal>
            <Reveal className="modality-path" delay={80}>
              <p className="modality-path__eyebrow mono">Photos</p>
              <p className="modality-path__flow mono">
                images → pose estimation → L0 / L1 (real data) → L2 → L3
              </p>
              <p className="modality-path__desc">
                Pose estimation via MASt3R-class feed-forward reconstruction initialises
                the dense cloud from measured data. Generative completion fills unseen
                regions and is flagged in the provenance channel — never silently merged.
              </p>
            </Reveal>
            <Reveal className="modality-path" delay={160}>
              <p className="modality-path__eyebrow mono">Video</p>
              <p className="modality-path__flow mono">
                frames → pose-free recon → L0 / L1 / L2 → L3 (+ L7 if dynamic)
              </p>
              <p className="modality-path__desc">
                Frame selection and deblur precede metric depth alignment. Static scenes
                follow the photo path with more views; dynamic content produces L7
                deformation keyframes via 4DGS.
              </p>
            </Reveal>
          </div>
        </Section>

        {/* CTA */}
        <Section className="pipeline-cta">
          <div className="pipeline-cta__inner">
            <h2>Try it in the Studio.</h2>
            <p>
              Submit a text prompt or upload photos and watch the Layer Stack build
              live — L0 seed, L1 dense cloud, L2 coarse, L3 refined surface.
            </p>
            <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
          </div>
        </Section>
      </div>
    </div>
  );
}
