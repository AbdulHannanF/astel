import type { Manifest } from "./types.js";

/**
 * Serialize a {@link Manifest} to canonical `manifest.json` text.
 *
 * This is a thin, faithful `JSON.stringify` — it does not strip, reorder, or normalize any
 * keys, so unknown additive keys, `extensions.*`, and `extras` blocks carried on the typed
 * object round-trip byte-for-byte in content (modulo JSON formatting), per the
 * forward-migration policy (manifest-v0.md section 10).
 *
 * @param manifest - the manifest to serialize.
 * @param pretty - if true (default), pretty-print with 2-space indentation and a trailing
 *   newline, matching typical checked-in `manifest.json` formatting.
 */
export function serializeManifest(manifest: Manifest, pretty = true): string {
  const json = pretty ? JSON.stringify(manifest, null, 2) : JSON.stringify(manifest);
  return pretty ? `${json}\n` : json;
}
