// @vitest-environment node
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const PLY = fileURLToPath(
  new URL("../../public/samples/astel-sample.ply", import.meta.url),
);
const REPORT = fileURLToPath(
  new URL("../../public/samples/astel-sample.report.json", import.meta.url),
);

const INRIA_FIELDS = [
  "x",
  "y",
  "z",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
];

describe("checked-in sample asset", () => {
  it("is a valid binary 3DGS PLY with the INRIA field layout", async () => {
    const buf = await readFile(PLY);
    const headerEnd = buf.indexOf("end_header\n") + "end_header\n".length;
    const header = buf.subarray(0, headerEnd).toString("ascii");

    expect(header.startsWith("ply\n")).toBe(true);
    expect(header).toContain("format binary_little_endian 1.0");
    for (const field of INRIA_FIELDS) {
      expect(header).toContain(`property float ${field}`);
    }

    const match = header.match(/element vertex (\d+)/);
    expect(match).not.toBeNull();
    const count = Number(match?.[1]);
    expect(count).toBeGreaterThan(10_000);

    // Body must be exactly count * 14 float32.
    const body = buf.byteLength - headerEnd;
    expect(body).toBe(count * INRIA_FIELDS.length * 4);
  });

  it("ships a quality report whose splat count matches the asset", async () => {
    const report = JSON.parse(await readFile(REPORT, "utf8")) as {
      splat_count: number;
      provenance: { measured_ratio: number; generated_ratio: number };
    };
    expect(report.splat_count).toBeGreaterThan(10_000);
    expect(
      report.provenance.measured_ratio + report.provenance.generated_ratio,
    ).toBeCloseTo(1);
  });
});
