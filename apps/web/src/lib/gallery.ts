/** Loads the gallery asset index served as a static file. Real assets only —
 *  the index grows as the pipeline produces more. */
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

export async function fetchGallery(
  signal?: AbortSignal,
): Promise<GalleryEntry[]> {
  const res = await fetch("/samples/gallery.json", { signal: signal ?? null });
  if (!res.ok) throw new Error(`Failed to load gallery (${res.status})`);
  const data = (await res.json()) as GalleryIndex;
  return data.assets;
}
