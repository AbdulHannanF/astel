import type { Conditioning } from "../lib/api.ts";
import type { AssetOrigin, QualityReport } from "../lib/report.ts";
import "./TruthMeter.css";

interface TruthMeterProps {
  report: QualityReport | null;
  errored: boolean;
  /**
   * What the current asset's L3 geometry was conditioned on (audit
   * recommendation #2). When `"none"`, the geometry is a
   * prompt/capture-independent placeholder — surfaced as an honest pill so
   * this never reads as "your input, generated".
   */
  conditioning?: Conditioning | null;
}

/** Pill config per origin value. */
const ORIGIN_PILL: Record<
  AssetOrigin,
  { className: string; label: string; title: string }
> = {
  stub: {
    className: "truth__origin-pill truth__origin-pill--stub",
    label: "STUB · placeholder geometry",
    title: "Stub pipeline output — illustrative, not derived from your input",
  },
  generated: {
    className: "truth__origin-pill truth__origin-pill--generated",
    label: "GENERATED · no ground truth",
    title: "Asset was generated without a real-world reference; numbers are self-consistency estimates",
  },
  measured: {
    className: "truth__origin-pill truth__origin-pill--measured",
    label: "MEASURED",
    title: "Asset was reconstructed from real capture data with ground-truth comparison",
  },
};

export function TruthMeter({
  report,
  errored,
  conditioning,
}: TruthMeterProps): React.JSX.Element {
  // Typed origin from the report (absent on older packages — no pill shown if missing).
  const origin: AssetOrigin | undefined = report?.origin;
  const isUnconditioned = conditioning === "none";

  const pillConfig = origin != null ? ORIGIN_PILL[origin] : null;

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="truth__title-group">
          <h2 className="panel__title">Truth Meter</h2>
          {pillConfig != null && (
            <span className={pillConfig.className} title={pillConfig.title}>
              {pillConfig.label}
            </span>
          )}
          {isUnconditioned && (
            <span
              className="truth__stub-pill"
              title="This geometry was not generated from your prompt or upload"
            >
              PLACEHOLDER — NOT FROM YOUR INPUT
            </span>
          )}
        </span>
        <span className="truth__badge">honesty report</span>
      </div>

      <div className="truth">
        {errored ? (
          <p className="truth__error">
            Quality report unavailable for this asset.
          </p>
        ) : report === null ? (
          <Skeleton />
        ) : (
          <Report report={report} />
        )}
      </div>
    </section>
  );
}

function Skeleton(): React.JSX.Element {
  return (
    <div aria-busy="true" aria-label="Loading quality report">
      <div className="truth__skeleton" />
      <div className="truth__skeleton" />
      <div className="truth__skeleton" style={{ width: "55%" }} />
    </div>
  );
}

function Report({ report }: { report: QualityReport }): React.JSX.Element {
  const measured = Math.round(report.provenance.measured_ratio * 100);
  const generated = 100 - measured;
  const confPct = Math.round(report.scale.confidence * 100);
  const confLevel = report.scale.confidence >= 0.7 ? "high" : "low";

  return (
    <>
      <div className="prov">
        <div className="prov__bar">
          <div className="prov__measured" style={{ width: `${measured}%` }} />
          <div
            className="prov__generated"
            style={{ width: `${generated}%` }}
          />
        </div>
        <div className="prov__legend mono">
          <span>
            <i className="dot dot--measured" />
            measured <b>{measured}%</b>
          </span>
          <span>
            <i className="dot dot--generated" />
            generated <b>{generated}%</b>
          </span>
        </div>
      </div>

      <div className="truth__grid">
        <div className="metric">
          <div className="metric__label">Geometric error</div>
          <div className="metric__value">
            {report.geometry.chamfer_mm_vs_l1.toFixed(1)}
            <small>mm chamfer</small>
          </div>
        </div>
        <div className="metric">
          <div className="metric__label">Fidelity</div>
          <div className="metric__value">
            {report.geometry.psnr_db.toFixed(1)}
            <small>dB psnr</small>
          </div>
        </div>
        <div className="metric">
          <div className="metric__label">Splats</div>
          <div className="metric__value">
            {(report.splat_count / 1000).toFixed(0)}
            <small>k</small>
          </div>
        </div>
        <div className="metric">
          <div className="metric__label">Longest axis</div>
          <div className="metric__value">
            {(report.scale.longest_axis_m * 100).toFixed(1)}
            <small>cm</small>
          </div>
        </div>
      </div>

      <div className="confidence">
        <div className="confidence__head mono">
          <span className="label">scale confidence</span>
          <span className="pct">{confPct}%</span>
        </div>
        <div className="confidence__track">
          <div
            className={`confidence__fill confidence__fill--${confLevel}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
        <p className="confidence__note">{report.scale.note}</p>
      </div>

      {report.origin != null && report.origin !== "measured" && report.caveat && (
        <p className="truth__caveat">{report.caveat}</p>
      )}
    </>
  );
}
