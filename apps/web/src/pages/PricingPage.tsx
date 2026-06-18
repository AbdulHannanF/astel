import { useEffect, useState } from "react";

import { fetchPricing } from "../lib/api.ts";
import type { PricingResource } from "../lib/api.ts";
import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";

/** Human-readable tier labels and ordering. */
const TIER_META: Record<string, { label: string; desc: string; order: number }> = {
  preview: {
    label: "Preview",
    desc: "L0 seed, L1 dense cloud, L2 coarse gaussians — fast, cheap iteration.",
    order: 0,
  },
  refine: {
    label: "Refine",
    desc: "L3 surface-aligned gaussians — the hero layer, the main spend.",
    order: 1,
  },
  addon: {
    label: "Add-ons",
    desc: "L4 appearance, L5 collision, L6 physics-material, L7 dynamics.",
    order: 2,
  },
  print: {
    label: "Print prep",
    desc: "Splat → SDF → watertight surface → .3mf / .stl with printability checks.",
    order: 3,
  },
};

function usd(credits: number, rate: number): string {
  return `$${(credits * rate).toFixed(2)}`;
}

export function PricingPage(): React.JSX.Element {
  const [data, setData] = useState<PricingResource | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        setData(await fetchPricing(ctrl.signal));
      } catch (e) {
        if (!(e instanceof DOMException)) setError(true);
      }
    })();
    return () => ctrl.abort();
  }, []);

  const rate = data?.credit_usd_rate ?? 0.01;

  // Group layers by tier, sorted by tier order then credits.
  const tierGroups: Array<{ key: string; meta: typeof TIER_META[string]; layers: typeof data extends null ? never[] : NonNullable<typeof data>["layers"] }> = [];
  if (data) {
    const byTier = new Map<string, NonNullable<typeof data>["layers"][number][]>();
    for (const layer of data.layers) {
      const arr = byTier.get(layer.tier) ?? [];
      arr.push(layer);
      byTier.set(layer.tier, arr);
    }
    for (const [key, layers] of byTier) {
      const meta = TIER_META[key] ?? { label: key, desc: "", order: 99 };
      tierGroups.push({ key, meta, layers });
    }
    tierGroups.sort((a, b) => a.meta.order - b.meta.order);
  }

  return (
    <div data-page="pricing" className="page pricing-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Pricing"
          title="Pay for what you generate."
          lede="Credits are the internal unit. 1 credit = 1 US cent (notional). Preview layers are cheap — L3 refine is the main spend. Add-ons and print prep are metered separately."
        />

        {/* Credit model explainer */}
        <Section className="pricing-explainer">
          <div className="credit-explainer">
            <div className="credit-explainer__card">
              <p className="credit-explainer__eyebrow mono">The credit model</p>
              <p className="credit-explainer__body">
                One generation runs through stages in order — seed, dense cloud, coarse
                gaussians, refined surface. Each stage has a credit cost. You pay only
                for the stages you request. Previewing a new idea (L0–L2) costs
                a few cents. Requesting a fully refined L3 surface costs more. Layer
                add-ons (relighting, collision, physics, dynamics) and print prep are
                purchased on top.
              </p>
              <p className="credit-explainer__note mono">
                1 credit = {rate === 0.01 ? "$0.01 notional" : `$${rate.toFixed(4)} notional`}
              </p>
            </div>
          </div>
        </Section>

        {/* Pricing schedule */}
        <Section
          eyebrow="Credit schedule"
          title="Layer pricing."
          className="pricing-schedule"
        >
          {error && (
            <div className="pricing-error" role="alert">
              <p>Couldn&rsquo;t reach the gateway — pricing schedule unavailable.</p>
            </div>
          )}

          {!data && !error && (
            <div className="pricing-skeleton" aria-busy="true" aria-label="Loading pricing data">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="skeleton-tier" />
              ))}
            </div>
          )}

          {data && (
            <div className="tier-groups">
              {tierGroups.map(({ key, meta, layers }) => (
                <Reveal key={key} className="tier-group">
                  <div className="tier-group__head">
                    <h3 className="tier-group__label">{meta.label}</h3>
                    <p className="tier-group__desc">{meta.desc}</p>
                  </div>
                  <table className="tier-table">
                    <thead>
                      <tr>
                        <th className="tier-table__col-layer">Layer</th>
                        <th className="tier-table__col-name">Name</th>
                        <th className="tier-table__col-credits mono">Credits</th>
                        <th className="tier-table__col-usd mono">≈ USD</th>
                      </tr>
                    </thead>
                    <tbody>
                      {layers.map((layer) => (
                        <tr key={layer.code}>
                          <td className="mono tier-table__code">{layer.code}</td>
                          <td>{layer.label}</td>
                          <td className="mono tier-table__credits">{layer.credits}</td>
                          <td className="mono tier-table__usd">{usd(layer.credits, rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Reveal>
              ))}
            </div>
          )}

          {/* Notes from API */}
          {data?.notes && data.notes.length > 0 && (
            <div className="pricing-notes">
              <p className="pricing-notes__head mono">Fine print</p>
              <ul className="pricing-notes__list">
                {data.notes.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </div>
          )}
        </Section>

        {/* Self-host callout */}
        <Section className="pricing-selfhost">
          <Reveal className="selfhost-callout">
            <div className="selfhost-callout__body">
              <p className="selfhost-callout__eyebrow mono">Enterprise / self-host</p>
              <h3 className="selfhost-callout__title">
                Run the full pipeline on your own infrastructure.
              </h3>
              <p className="selfhost-callout__desc">
                Astel is containerised: one command brings up the API and web
                viewer on a single-GPU box, and a Docker Compose stack (API,
                Postgres, MinIO, optional Temporal worker) scales to cloud GPU
                fleets. Self-hosted deployments are license-keyed — no per-credit
                metering on your own hardware.
              </p>
              <CtaLink to="/self-host" variant="ghost">
                Self-host guide →
              </CtaLink>
            </div>
          </Reveal>
        </Section>

        {/* Final CTA */}
        <Section className="pricing-cta">
          <div className="pricing-cta__inner">
            <h2>Start generating.</h2>
            <p>
              Open the Studio to create your first layered Gaussian splat asset.
              Preview stages cost only a few cents, so exploration stays cheap.
            </p>
            <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
          </div>
        </Section>
      </div>
    </div>
  );
}
