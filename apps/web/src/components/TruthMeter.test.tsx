import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { mapApiReport, type ApiQualityReport, type QualityReport } from "../lib/report.ts";
import { TruthMeter } from "./TruthMeter.tsx";

const REPORT: QualityReport = {
  asset_id: "x",
  name: "Sample",
  modality: "generated",
  splat_count: 48000,
  splat_budget: "100k",
  active_layer: "L3",
  scale: {
    longest_axis_m: 0.18,
    confidence: 0.41,
    method: "VLM",
    note: "estimate",
  },
  geometry: {
    chamfer_mm_vs_l1: 0.9,
    psnr_db: 31.2,
    ssim: 0.946,
    normals_present: true,
  },
  provenance: { measured_ratio: 0, generated_ratio: 1, note: "synthetic" },
};

describe("TruthMeter", () => {
  it("shows a skeleton while loading", () => {
    render(<TruthMeter report={null} errored={false} />);
    expect(screen.getByLabelText(/loading quality report/i)).toBeInTheDocument();
  });

  it("renders measured/generated provenance and metrics", () => {
    render(<TruthMeter report={REPORT} errored={false} />);
    expect(screen.getByText("100%")).toBeInTheDocument(); // generated
    expect(screen.getByText(/0\.9/)).toBeInTheDocument(); // chamfer
    expect(screen.getByText(/31\.2/)).toBeInTheDocument(); // psnr
    expect(screen.getByText("41%")).toBeInTheDocument(); // scale confidence
  });

  it("shows an honest fallback on error", () => {
    render(<TruthMeter report={null} errored={true} />);
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });

  it("does not show a STUB pill for the static sample report", () => {
    render(<TruthMeter report={REPORT} errored={false} />);
    expect(screen.queryByText("STUB")).not.toBeInTheDocument();
  });

  it("shows an honest placeholder pill when conditioning is 'none'", () => {
    render(
      <TruthMeter report={REPORT} errored={false} conditioning="none" />,
    );
    expect(screen.getByText(/not from your input/i)).toBeInTheDocument();
  });

  it("does not show the placeholder pill when conditioning is real input", () => {
    render(
      <TruthMeter report={REPORT} errored={false} conditioning="image" />,
    );
    expect(screen.queryByText(/not from your input/i)).not.toBeInTheDocument();
  });

  it("shows a STUB pill and caveat for a non-measured API report", () => {
    const api: ApiQualityReport = {
      schema: "astel.quality-report/v0",
      origin: "stub",
      modality: "text",
      splats: 48000,
      geometric_error: { chamfer_mm_vs_l1: 0.9, method: "stub-placeholder" },
      fidelity: { psnr_db: 31.2, ssim: null, lpips: null, n_holdout_views: 0 },
      scale: { longest_axis_m: 0.182, confidence: 0.41, method: "estimate" },
      provenance: { measured_ratio: 0, generated_ratio: 1 },
      caveats: ["Stub pipeline output: illustrative only."],
    };
    const mapped = mapApiReport(api, "task-123");

    render(<TruthMeter report={mapped} errored={false} />);
    expect(screen.getByText("STUB")).toBeInTheDocument();
    expect(
      screen.getByText(/Stub pipeline output: illustrative only\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/0\.9/)).toBeInTheDocument(); // chamfer
    expect(screen.getByText(/31\.2/)).toBeInTheDocument(); // psnr
  });
});
