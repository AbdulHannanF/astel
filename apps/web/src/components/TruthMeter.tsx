import type { QualityReport } from "../lib/report.ts";
import "./TruthMeter.css";

interface TruthMeterProps {
  report: QualityReport | null;
  errored: boolean;
}

export function TruthMeter({
  report,
  errored,
}: TruthMeterProps): React.JSX.Element {
  const isStub =
    report !== null && report.origin != null && report.origin !== "measured";

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="truth__title-group">
          <h2 className="panel__title">Truth Meter</h2>
          {isStub && (
            <span
              className="truth__stub-pill"
              title="Stub pipeline output — illustrative, not measured"
            >
              STUB
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
