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
    // New typed pill: "STUB · placeholder geometry"
    expect(screen.getByText(/^STUB/)).toBeInTheDocument();
    expect(
      screen.getByText(/Stub pipeline output: illustrative only\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/0\.9/)).toBeInTheDocument(); // chamfer
    expect(screen.getByText(/31\.2/)).toBeInTheDocument(); // psnr
  });
});

describe("TruthMeter origin taxonomy pills", () => {
  const BASE_REPORT: QualityReport = {
    asset_id: "pill-test",
    name: "Pill Test",
    modality: "generated",
    splat_count: 48000,
    splat_budget: "100k",
    active_layer: "L3",
    scale: { longest_axis_m: 0.18, confidence: 0.8, method: "VLM", note: "est" },
    geometry: { chamfer_mm_vs_l1: 1.0, psnr_db: 30.0, ssim: 0.9, normals_present: true },
    provenance: { measured_ratio: 0, generated_ratio: 1, note: "" },
  };

  it("renders red STUB pill for origin='stub'", () => {
    render(<TruthMeter report={{ ...BASE_REPORT, origin: "stub" }} errored={false} />);
    const pill = screen.getByText(/STUB · placeholder geometry/i);
    expect(pill).toBeInTheDocument();
    expect(pill.className).toContain("truth__origin-pill--stub");
  });

  it("renders amber GENERATED pill for origin='generated'", () => {
    render(<TruthMeter report={{ ...BASE_REPORT, origin: "generated" }} errored={false} />);
    const pill = screen.getByText(/GENERATED · no ground truth/i);
    expect(pill).toBeInTheDocument();
    expect(pill.className).toContain("truth__origin-pill--generated");
  });

  it("renders green MEASURED pill for origin='measured'", () => {
    render(<TruthMeter report={{ ...BASE_REPORT, origin: "measured" }} errored={false} />);
    // The origin pill has class truth__origin-pill--measured; find via role=generic / title
    const pill = document.querySelector(".truth__origin-pill--measured");
    expect(pill).not.toBeNull();
    expect(pill?.textContent).toBe("MEASURED");
  });

  it("renders no origin pill when origin is absent (backward-compat)", () => {
    render(<TruthMeter report={BASE_REPORT} errored={false} />);
    expect(screen.queryByText(/placeholder geometry/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/no ground truth/i)).not.toBeInTheDocument();
    // No .truth__origin-pill element at all
    expect(document.querySelector(".truth__origin-pill")).toBeNull();
  });
});
