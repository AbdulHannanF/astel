/**
 * Bundled copies of docs/specs/schemas/*.json (JSON Schema draft 2020-12).
 *
 * These are imported as JSON modules so the package is dependency-light and isomorphic
 * (no filesystem access required at runtime, so it works in the browser). Keep these in sync
 * with docs/specs/schemas/ — the schema there is authoritative; the schema wins.
 */

import manifestSchema from "./manifest.schema.json" with { type: "json" };
import layerSchema from "./layer.schema.json" with { type: "json" };
import buffersSchema from "./buffers.schema.json" with { type: "json" };
import provenanceSchema from "./provenance.schema.json" with { type: "json" };
import qualityReportSchema from "./quality-report.schema.json" with { type: "json" };

export {
  manifestSchema,
  layerSchema,
  buffersSchema,
  provenanceSchema,
  qualityReportSchema,
};

/** All sub-schemas referenced by `$ref` from manifest.schema.json, keyed by their `$id`. */
export const allSchemas = [
  manifestSchema,
  layerSchema,
  buffersSchema,
  provenanceSchema,
  qualityReportSchema,
];
