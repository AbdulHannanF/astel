import Ajv2020 from "ajv/dist/2020.js";
import type { ErrorObject, ValidateFunction } from "ajv";
import addFormats from "ajv-formats";

import { allSchemas, manifestSchema } from "./schemas/index.js";
import type { Manifest } from "./types.js";

/** A single structured validation problem. */
export interface ManifestValidationIssue {
  /** JSON-pointer-ish path to the offending value (ajv `instancePath`). */
  path: string;
  /** Human-readable description of the problem. */
  message: string;
  /** The JSON Schema keyword that failed (e.g. "required", "pattern"). */
  keyword: string;
  /** Additional ajv-provided parameters about the failure. */
  params?: Record<string, unknown>;
}

export type ParseManifestResult =
  | { ok: true; manifest: Manifest }
  | { ok: false; errors: ManifestValidationIssue[] };

let cachedValidator: ValidateFunction | undefined;

function getValidator(): ValidateFunction {
  if (cachedValidator) {
    return cachedValidator;
  }
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv);
  for (const schema of allSchemas) {
    if (schema.$id !== manifestSchema.$id) {
      ajv.addSchema(schema);
    }
  }
  cachedValidator = ajv.compile(manifestSchema);
  return cachedValidator;
}

function toIssues(errors: ErrorObject[] | null | undefined): ManifestValidationIssue[] {
  if (!errors) {
    return [];
  }
  return errors.map((err) => {
    const issue: ManifestValidationIssue = {
      path: err.instancePath || "/",
      message: describeError(err),
      keyword: err.keyword,
    };
    if (err.params !== undefined) {
      issue.params = err.params as Record<string, unknown>;
    }
    return issue;
  });
}

function describeError(err: ErrorObject): string {
  const at = err.instancePath || "(root)";
  switch (err.keyword) {
    case "required": {
      const missing = (err.params as { missingProperty?: string }).missingProperty;
      return `${at}: missing required property "${missing}"`;
    }
    case "additionalProperties": {
      const extra = (err.params as { additionalProperty?: string }).additionalProperty;
      return `${at}: unexpected additional property "${extra}"`;
    }
    default:
      return `${at}: ${err.message ?? "validation failed"}`;
  }
}

/**
 * Parse and validate a `manifest.json` payload against `manifest.schema.json` (draft 2020-12).
 *
 * Accepts either a JSON string or an already-parsed value. On success, returns the value typed
 * as {@link Manifest}. On failure (malformed JSON or schema violations), returns a structured
 * list of issues — never throws for ordinary validation failures.
 */
export function parseManifest(input: string | unknown): ParseManifestResult {
  let value: unknown;
  if (typeof input === "string") {
    try {
      value = JSON.parse(input);
    } catch (err) {
      return {
        ok: false,
        errors: [
          {
            path: "/",
            message: `invalid JSON: ${err instanceof Error ? err.message : String(err)}`,
            keyword: "json",
          },
        ],
      };
    }
  } else {
    value = input;
  }

  const validate = getValidator();
  const valid = validate(value);
  if (!valid) {
    return { ok: false, errors: toIssues(validate.errors) };
  }
  return { ok: true, manifest: value as Manifest };
}

/** A path-validation problem found by {@link validatePaths}. */
export interface PathValidationIssue {
  /** Where in the manifest the bad path was found (e.g. "layers.l3.files[0].path"). */
  location: string;
  /** The offending path value. */
  path: string;
  /** Why the path is rejected. */
  reason: "absolute" | "traversal";
}

const LAYER_IDS = ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"] as const;

function isBadPath(path: string): "absolute" | "traversal" | null {
  // POSIX-relative only: reject leading "/" and any ".." path segment, regardless of
  // separator style (the manifest spec requires POSIX relative paths, but be defensive
  // about stray backslashes from Windows-authored manifests).
  if (path.startsWith("/")) {
    return "absolute";
  }
  // Windows drive-letter absolute paths (e.g. "C:\\...") or UNC paths.
  if (/^[A-Za-z]:[\\/]/.test(path) || path.startsWith("\\\\")) {
    return "absolute";
  }
  const segments = path.split(/[\\/]+/);
  if (segments.some((seg) => seg === "..")) {
    return "traversal";
  }
  return null;
}

/**
 * Walk every file/buffer `path`/`uri` referenced by the manifest and report any that are
 * absolute or contain a `..` traversal segment.
 *
 * Per manifest-v0.md section 1, all layer/quality/export files are referenced by POSIX
 * relative path from the package root; readers MUST reject paths that escape the root. This
 * function performs that check independent of whether the referenced file actually exists in
 * a package (the caller may not have the zip at hand, e.g. in a browser before download).
 */
export function validatePaths(manifest: Manifest): PathValidationIssue[] {
  const issues: PathValidationIssue[] = [];

  const check = (location: string, path: string | undefined): void => {
    if (path === undefined) {
      return;
    }
    const bad = isBadPath(path);
    if (bad) {
      issues.push({ location, path, reason: bad });
    }
  };

  for (const layerId of LAYER_IDS) {
    const layer = manifest.layers[layerId];
    if (!layer) {
      continue;
    }
    for (const [i, file] of (layer.files ?? []).entries()) {
      check(`layers.${layerId}.files[${i}].path`, file.path);
    }
    if (layer.appearance) {
      check(`layers.${layerId}.appearance.env_map_path`, layer.appearance.env_map_path);
      check(`layers.${layerId}.appearance.baked_pbr_path`, layer.appearance.baked_pbr_path);
    }
    if (layer.collision) {
      check(`layers.${layerId}.collision.sdf_path`, layer.collision.sdf_path);
      check(`layers.${layerId}.collision.convex_set_path`, layer.collision.convex_set_path);
      check(`layers.${layerId}.collision.mass_props_path`, layer.collision.mass_props_path);
      if (layer.collision.isosurface) {
        check(`layers.${layerId}.collision.isosurface.path`, layer.collision.isosurface.path);
      }
    }
    if (layer.physics_material) {
      check(`layers.${layerId}.physics_material.regions_path`, layer.physics_material.regions_path);
    }
    if (layer.dynamics) {
      check(`layers.${layerId}.dynamics.deformation_path`, layer.dynamics.deformation_path);
      check(`layers.${layerId}.dynamics.timeline_path`, layer.dynamics.timeline_path);
    }
  }

  for (const [i, buf] of manifest.buffers.buffers.entries()) {
    check(`buffers.buffers[${i}].uri`, buf.uri);
  }

  for (const hallucination of [manifest.quality_report.hallucination]) {
    if (typeof hallucination.heatmap_ref === "string") {
      check("quality_report.hallucination.heatmap_ref", hallucination.heatmap_ref);
    }
  }

  for (const [i, exp] of (manifest.exports ?? []).entries()) {
    check(`exports[${i}].path`, exp.path);
    check(`exports[${i}].sidecar_path`, exp.sidecar_path);
  }

  return issues;
}
