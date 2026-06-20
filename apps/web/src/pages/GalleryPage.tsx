import { useEffect, useState } from "react";

import { Link } from "react-router-dom";

import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";
import { fetchGallery } from "../lib/gallery.ts";
import type { GalleryEntry } from "../lib/gallery.ts";

/** Short display label for a modality string. */
function modalityLabel(modality: string): string {
  if (modality === "generated") return "Generated";
  if (modality === "captured") return "Captured";
  if (modality === "hybrid") return "Hybrid";
  return modality;
}

/** One-letter monogram for the asset tile face. */
function monogram(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return "A";
  const words = trimmed.split(/\s+/);
  const first = words[0] ?? "";
  const second = words[1] ?? "";
  if (words.length >= 2 && first && second) {
    return (first[0]! + second[0]!).toUpperCase();
  }
  return trimmed.slice(0, 2).toUpperCase();
}

interface GalleryTileProps {
  entry: GalleryEntry;
  index: number;
}

function GalleryTile({ entry, index }: GalleryTileProps): React.JSX.Element {
  return (
    <Reveal className="gallery-tile-wrap" delay={index * 60}>
      <Link
        to={`/gallery/${entry.id}`}
        className="gallery-tile"
        aria-label={`Open ${entry.name} in the viewer`}
      >
        {/* CSS-only art panel — brass gradient face with monogram */}
        <div className="gallery-tile__face" aria-hidden>
          <span className="gallery-tile__monogram mono">{monogram(entry.name)}</span>
          <div className="gallery-tile__scanlines" />
        </div>

        {/* Metadata */}
        <div className="gallery-tile__body">
          <div className="gallery-tile__top">
            <span className="gallery-tile__modality-chip mono">
              {modalityLabel(entry.modality)}
            </span>
          </div>
          <h3 className="gallery-tile__name">{entry.name}</h3>
          {entry.blurb && (
            <p className="gallery-tile__blurb">{entry.blurb}</p>
          )}
          <div className="gallery-tile__footer">
            <span className="gallery-tile__id mono">{entry.id}</span>
            <span className="gallery-tile__arrow" aria-hidden>→</span>
          </div>
        </div>
      </Link>
    </Reveal>
  );
}

export function GalleryPage(): React.JSX.Element {
  const [entries, setEntries] = useState<GalleryEntry[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        setEntries(await fetchGallery(ctrl.signal));
      } catch (e) {
        if (!(e instanceof DOMException)) setError(true);
      }
    })();
    return () => ctrl.abort();
  }, []);

  return (
    <div data-page="gallery" className="page gallery-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Gallery"
          title="Real assets, fully inspectable."
          lede="Every asset here was produced by the Astel pipeline and carries a live quality report — geometric error, scale confidence, and a provenance breakdown showing exactly what was measured versus generated. No mock renders, no retouched screenshots."
        />

        <Section
          eyebrow="Asset catalog"
          title="Open any asset in the live viewer."
          lede="Every generation the pipeline produces lands here automatically, newest first, alongside the bundled reference that sets the honesty baseline. Open any one to inspect its splats and quality report live."
          className="gallery-catalog"
        >
          {/* Error state */}
          {error && (
            <div className="gallery-error" role="alert">
              <p>Couldn&rsquo;t load the asset catalog. Check the network and reload.</p>
            </div>
          )}

          {/* Loading skeleton */}
          {!entries && !error && (
            <div
              className="gallery-skeleton"
              aria-busy="true"
              aria-label="Loading gallery"
            >
              {[1, 2, 3].map((i) => (
                <div key={i} className="gallery-skeleton__tile" />
              ))}
            </div>
          )}

          {/* Empty state (loaded but no assets) */}
          {entries && entries.length === 0 && (
            <div className="gallery-empty">
              <p className="gallery-empty__text">
                No assets in the catalog yet. The pipeline is generating the
                first batch — check back shortly.
              </p>
            </div>
          )}

          {/* Asset grid */}
          {entries && entries.length > 0 && (
            <>
              <div className="gallery-grid">
                {entries.map((entry, idx) => (
                  <GalleryTile key={entry.id} entry={entry} index={idx} />
                ))}
              </div>
              <Reveal>
                <p className="gallery-count mono">
                  {entries.length === 1
                    ? "1 asset · more as the pipeline produces them"
                    : `${entries.length} assets · catalog updated as the pipeline runs`}
                </p>
              </Reveal>
            </>
          )}
        </Section>

        <Section className="gallery-cta">
          <div className="gallery-cta__inner">
            <h2>Generate your own.</h2>
            <p>
              Submit a text prompt or upload photos and watch the Layer Stack
              build live — then inspect every quality metric in the Truth Meter.
            </p>
            <CtaLink to="/studio" variant="primary">Open Studio</CtaLink>
          </div>
        </Section>
      </div>
    </div>
  );
}
