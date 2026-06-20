/**
 * Gallery catalog source.
 *
 * Every generation the pipeline produces shows up here by default: we fetch the
 * live catalog from `GET /v1/generations` and map each produced asset to a
 * {@link GalleryEntry}. The bundled static sample is appended as an honesty
 * baseline so the gallery is never empty (and still works in a static deploy
 * with no API reachable).
 */
import { artifactUrl, type Conditioning, type Modality } from "./api.ts";

export type GalleryReportKind = "sample" | "api";

export interface GalleryEntry {
  id: string;
  name: string;
  modality: string;
  blurb?: string;
  splatUrl: string;
  reportUrl: string;
  reportKind: GalleryReportKind;
  relightUrl?: string;
  massUrl?: string;
}

export interface GalleryIndex {
  schema: string;
  assets: GalleryEntry[];
}

/** One record from `GET /v1/generations` (mirror of API `GenerationSummary`). */
interface GenerationSummary {
  id: string;
  modality: Modality;
  prompt: string;
  created_at: string;
  produced: boolean;
  splats?: number | null;
  conditioning?: Conditioning | null;
  has_asset: boolean;
}

/** A short, tidy display name derived from a generation prompt. */
function nameFromPrompt(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return "Untitled generation";
  if (trimmed.length <= 48) return trimmed;
  return trimmed.slice(0, 47).trimEnd() + "…";
}

/** Map a live generation summary onto a gallery entry backed by API artifacts. */
function entryFromSummary(s: GenerationSummary): GalleryEntry {
  const splatCount =
    typeof s.splats === "number" ? `${Math.round(s.splats / 1000)}k splats · ` : "";
  return {
    id: s.id,
    name: nameFromPrompt(s.prompt),
    // Text/image generations are "generated"; captures would be "captured".
    modality: s.modality === "video" ? "captured" : "generated",
    blurb: `${splatCount}${s.conditioning === "prompt" ? "from your prompt" : s.modality}`,
    splatUrl: artifactUrl(s.id, "l3.ply"),
    reportUrl: artifactUrl(s.id, "quality-report.json"),
    reportKind: "api",
    relightUrl: artifactUrl(s.id, "l4-relight.json"),
    massUrl: artifactUrl(s.id, "l5-mass.json"),
  };
}

/** Fetch the live catalog from the API. Returns [] if the API is unreachable. */
async function fetchLiveGenerations(
  signal?: AbortSignal,
): Promise<GalleryEntry[]> {
  try {
    const res = await fetch("/v1/generations", { signal: signal ?? null });
    if (!res.ok) return [];
    const list = (await res.json()) as GenerationSummary[];
    return list.filter((s) => s.has_asset).map(entryFromSummary);
  } catch {
    return [];
  }
}

/** Fetch the bundled static sample catalog. Returns [] if unavailable. */
async function fetchSampleGallery(
  signal?: AbortSignal,
): Promise<GalleryEntry[]> {
  try {
    const res = await fetch("/samples/gallery.json", { signal: signal ?? null });
    if (!res.ok) return [];
    const data = (await res.json()) as GalleryIndex;
    return data.assets;
  } catch {
    return [];
  }
}

/**
 * The full gallery catalog: live generations first (newest first, courtesy of
 * the API ordering), then the static sample baseline. De-duplicated by id.
 */
export async function fetchGallery(
  signal?: AbortSignal,
): Promise<GalleryEntry[]> {
  const [live, sample] = await Promise.all([
    fetchLiveGenerations(signal),
    fetchSampleGallery(signal),
  ]);
  const seen = new Set(live.map((e) => e.id));
  const merged = [...live, ...sample.filter((e) => !seen.has(e.id))];
  return merged;
}
