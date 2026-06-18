/** Typed models mirroring the Astel REST API schemas. */

export type Modality = "text" | "image" | "video";
export type GenerationMode = "preview" | "refine";
export type GenerationStatus = "queued" | "running" | "succeeded" | "failed" | "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED";
export type Conditioning = "prompt" | "image" | "video" | "none";

export interface ArtifactRef {
  name: string;
  url: string;
  content_type: string;
  bytes: number;
}

export interface CreditLineItem {
  code: string;
  label: string;
  tier: string;
  credits: number;
  usd: number;
  detail: string;
}

export interface BillingSummary {
  mode: GenerationMode;
  refine_of: string | null;
  items: CreditLineItem[];
  total_credits: number;
  total_usd: number;
  credit_usd_rate: number;
  caveats: string[];
}

export interface Generation {
  id: string;
  modality: Modality;
  prompt: string | null;
  status: GenerationStatus;
  created_at: string;
  events_url: string;
  artifacts: ArtifactRef[];
  mode: GenerationMode;
  refine_of: string | null;
  billing: BillingSummary | null;
  conditioning: Conditioning | null;
}

export interface CaptureRef {
  capture_id: string;
  filename: string;
  content_type: string;
  bytes: number;
}

export interface LayerPriceRef {
  code: string;
  label: string;
  tier: string;
  credits: number;
}

export interface PricingResource {
  credit_usd_rate: number;
  layers: LayerPriceRef[];
  modes: Record<string, string[]>;
  notes: string[];
}

export interface CreateGenerationRequest {
  modality?: Modality;
  prompt?: string;
  capture_id?: string;
  mode?: GenerationMode;
  refine_of?: string;
}

export interface GenerateOptions {
  prompt?: string;
  modality?: Modality;
  captureId?: string;
  mode?: GenerationMode;
  refineOf?: string;
}

export interface DownloadOptions {
  /** Directory to save the file. */
  dir?: string;
}
