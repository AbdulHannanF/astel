import { useEffect, useState } from "react";

import { Link, useParams } from "react-router-dom";

import { TruthMeter } from "../components/TruthMeter.tsx";
import { Viewport } from "../components/Viewport.tsx";
import { CtaLink } from "../components/site/CtaLink.tsx";
import { fetchGallery } from "../lib/gallery.ts";
import type { GalleryEntry } from "../lib/gallery.ts";
import { mapApiReport } from "../lib/report.ts";
import type { ApiQualityReport, QualityReport } from "../lib/report.ts";

/** Short display label for a modality string. */
function modalityLabel(modality: string): string {
  if (modality === "generated") return "Generated";
  if (modality === "captured") return "Captured";
  if (modality === "hybrid") return "Hybrid";
  return modality;
}

export function GalleryAssetPage(): React.JSX.Element {
  const { id } = useParams<{ id: string }>();
  const [entry, setEntry] = useState<GalleryEntry | null | "not-found">(null);
  const [report, setReport] = useState<QualityReport | null>(null);
  const [reportError, setReportError] = useState(false);

  // Resolve the entry from the gallery index.
  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const all = await fetchGallery(ctrl.signal);
        const found = all.find((e) => e.id === id);
        setEntry(found ?? "not-found");
      } catch (e) {
        if (!(e instanceof DOMException)) setEntry("not-found");
      }
    })();
    return () => ctrl.abort();
  }, [id]);

  // Load the quality report once we know the entry.
  useEffect(() => {
    if (!entry || entry === "not-found") return;
    const ctrl = new AbortController();
    (async () => {
      try {
        const res = await fetch(entry.reportUrl, { signal: ctrl.signal });
        if (!res.ok) throw new Error(`report ${res.status}`);
        if (entry.reportKind === "sample") {
          setReport((await res.json()) as QualityReport);
        } else {
          const api = (await res.json()) as ApiQualityReport;
          setReport(mapApiReport(api, entry.id));
        }
        setReportError(false);
      } catch (e) {
        if (!(e instanceof DOMException)) setReportError(true);
      }
    })();
    return () => ctrl.abort();
  }, [entry]);

  // Always render the data-page marker synchronously.
  return (
    <div data-page="gallery-asset" className="page gallery-asset-page">
      {/* Not-found state */}
      {entry === "not-found" && (
        <div className="gallery-asset-page__inner">
          <div className="gallery-asset-not-found">
            <p className="gallery-asset-not-found__eyebrow mono">404 · Asset not found</p>
            <h1 className="gallery-asset-not-found__title">This asset doesn&rsquo;t exist.</h1>
            <p className="gallery-asset-not-found__body">
              The ID <span className="mono">{id ?? "unknown"}</span> isn&rsquo;t in the
              gallery index. It may have been removed or the URL is incorrect.
            </p>
            <Link to="/gallery" className="cta-link cta-link--ghost">
              ← Back to Gallery
            </Link>
          </div>
        </div>
      )}

      {/* Loading state (entry still resolving) */}
      {entry === null && (
        <div className="gallery-asset-page__inner">
          <div
            className="gallery-asset-loading"
            aria-busy="true"
            aria-label="Loading asset"
          >
            <div className="gallery-asset-loading__bar" />
            <div className="gallery-asset-loading__bar gallery-asset-loading__bar--short" />
          </div>
        </div>
      )}

      {/* Full detail view */}
      {entry !== null && entry !== "not-found" && (
        <div className="gallery-asset-page__inner">
          {/* Back navigation */}
          <div className="gallery-asset-nav">
            <Link to="/gallery" className="gallery-asset-nav__back">
              ← Gallery
            </Link>
          </div>

          {/* Two-column layout */}
          <div className="gallery-asset-layout">
            {/* Left: live viewer */}
            <div className="gallery-asset-viewer">
              <Viewport
                sampleUrl="/samples/astel-sample.ply"
                splatUrl={entry.splatUrl}
                splatVisible
              />
            </div>

            {/* Right: metadata + Truth Meter */}
            <aside className="gallery-asset-sidebar">
              {/* Asset identity */}
              <div className="gallery-asset-identity">
                <p className="gallery-asset-identity__modality mono">
                  {modalityLabel(entry.modality)}
                </p>
                <h1 className="gallery-asset-identity__name">{entry.name}</h1>
                {entry.blurb && (
                  <p className="gallery-asset-identity__blurb">{entry.blurb}</p>
                )}
                <p className="gallery-asset-identity__id mono">{entry.id}</p>
              </div>

              {/* Truth Meter */}
              <div className="gallery-asset-meter">
                <p className="gallery-asset-meter__label mono">Quality report</p>
                <TruthMeter
                  report={report}
                  errored={reportError}
                  conditioning={null}
                />
              </div>

              {/* Actions */}
              <div className="gallery-asset-actions">
                <CtaLink to="/studio" variant="primary">
                  Make your own →
                </CtaLink>
                <Link to="/gallery" className="cta-link cta-link--ghost">
                  ← All assets
                </Link>
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}
