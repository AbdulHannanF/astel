import { CtaLink } from "../components/site/CtaLink.tsx";
import { PageHeader } from "../components/site/PageHeader.tsx";
import { Reveal } from "../components/site/Reveal.tsx";
import { Section } from "../components/site/Section.tsx";

// ---------------------------------------------------------------------------
// Environment variable table data (verified from docs/site/docs/self-host.md)
// ---------------------------------------------------------------------------

interface EnvVar {
  name: string;
  default: string;
  desc: string;
}

const ENV_VARS: EnvVar[] = [
  {
    name: "ASTEL_PRODUCER",
    default: "stub",
    desc: "stub (fast, no GPU) or gpu (real generation via CUDA + gsplat).",
  },
  {
    name: "ASTEL_ARTIFACT_DIR",
    default: "./artifacts",
    desc: "Filesystem path where generated files are stored.",
  },
  {
    name: "ASTEL_ENGINE",
    default: "stub",
    desc: "stub or temporal — enables the durable Temporal task engine.",
  },
  {
    name: "ANTHROPIC_API_KEY",
    default: "—",
    desc: "Enables live LLM calls for the Generation Spec parser and L6 physics-material reasoning.",
  },
  {
    name: "ASTEL_LLM_LIVE",
    default: "0",
    desc: "Must be 1 AND key set to spend real LLM credits. 0 uses fixture responses.",
  },
  {
    name: "ASTEL_API_URL",
    default: "http://localhost:8000",
    desc: "Base URL used by the SDK and MCP server to reach the API.",
  },
];

// ---------------------------------------------------------------------------
// Hardware tier data (CLAUDE.md §6)
// ---------------------------------------------------------------------------

interface HardwareTier {
  label: string;
  eyebrow: string;
  specs: string[];
}

const HARDWARE_TIERS: HardwareTier[] = [
  {
    eyebrow: "Local / self-host minimum",
    label: "Single-GPU workstation",
    specs: [
      "1× GPU ≥ 24 GB VRAM (RTX 3090 / 4090-class)",
      "64 GB RAM, NVMe storage",
      "CUDA 12.x + MSVC (Windows) or GCC (Linux)",
      "Python 3.12+, Node 22+",
      "L3 refine (1 M-splat asset): ≤ 15–30 min patient mode",
    ],
  },
  {
    eyebrow: "Recommended local",
    label: "High-VRAM workstation",
    specs: [
      "RTX 5090 / NVIDIA 6000-Ada-class (32–48 GB VRAM)",
      "Cinematic splat budgets (5 M+) feasible locally",
      "Same one-command start; no config changes required",
    ],
  },
  {
    eyebrow: "Cloud production",
    label: "GPU fleet",
    specs: [
      "Preview pool: L4 / L40S-class for fast L2 previews",
      "Refine + training pool: A100 / H100 80 GB",
      "Spot-instance tolerant via resumable pipeline stages",
      "CPU-heavy stages (SfM, SDF, convex decomp) on separate CPU nodes",
    ],
  },
];

// ---------------------------------------------------------------------------
// SelfHostPage
// ---------------------------------------------------------------------------

export function SelfHostPage(): React.JSX.Element {
  return (
    <div data-page="self-host" className="page selfhost-page">
      <div className="page-inner">
        <PageHeader
          eyebrow="Self-host & Enterprise"
          title="Run Astel on your own hardware."
          lede="The same containers that power the cloud service — license-keyed, no per-credit metering on your own infrastructure. Your data never leaves your hardware."
        />

        {/* Positioning */}
        <Section
          eyebrow="Why self-host"
          title="Your hardware. Your data. No metering."
          className="selfhost-positioning"
        >
          <div className="selfhost-positioning__grid">
            <Reveal className="selfhost-pillar">
              <p className="selfhost-pillar__title">Same containers as the cloud</p>
              <p className="selfhost-pillar__body">
                The self-hosted stack is not a cut-down edition. It is the same
                Docker Compose setup (API, Postgres, MinIO, Temporal) that the
                cloud service runs on — license-keyed for on-premises use.
              </p>
            </Reveal>
            <Reveal className="selfhost-pillar" delay={80}>
              <p className="selfhost-pillar__title">No per-credit metering on-prem</p>
              <p className="selfhost-pillar__body">
                Cloud credits pay for GPU time we provide. When you supply the
                GPU, there is no per-generation charge. You pay a flat license
                fee and run unlimited generations on your own hardware.
              </p>
            </Reveal>
            <Reveal className="selfhost-pillar" delay={160}>
              <p className="selfhost-pillar__title">Data sovereignty</p>
              <p className="selfhost-pillar__body">
                Film, defence, and industrial pipelines often cannot send assets
                to a third-party cloud. Astel self-hosted keeps every capture,
                generation, and artifact inside your network.
              </p>
            </Reveal>
          </div>
        </Section>

        {/* Hardware requirements */}
        <Section
          eyebrow="Hardware requirements"
          title="What you need."
          lede="All sizes of Astel run on consumer NVIDIA GPUs. The cloud production tier scales to A100/H100 fleets."
          className="selfhost-hardware"
        >
          <div className="selfhost-hw-tiers">
            {HARDWARE_TIERS.map((tier, idx) => (
              <Reveal key={tier.eyebrow} className="selfhost-hw-tier" delay={idx * 80}>
                <p className="selfhost-hw-tier__eyebrow mono">{tier.eyebrow}</p>
                <h3 className="selfhost-hw-tier__label">{tier.label}</h3>
                <ul className="selfhost-hw-tier__specs">
                  {tier.specs.map((s) => (
                    <li key={s} className="selfhost-hw-tier__spec">{s}</li>
                  ))}
                </ul>
              </Reveal>
            ))}
          </div>
        </Section>

        {/* One-command start */}
        <Section
          eyebrow="Quick start"
          title="One command, full stack."
          lede="Dev mode starts the FastAPI gateway and the web viewer together. No Docker required."
          className="selfhost-quickstart"
        >
          <div className="selfhost-qs-steps">
            <Reveal className="selfhost-qs-block">
              <p className="selfhost-qs-block__label mono">Start the API + viewer</p>
              <pre className="docs-code-block"><code>{`pnpm run up            # starts API + web viewer
pnpm run up -Temporal  # with the durable Temporal task engine`}</code></pre>
              <p className="selfhost-qs-block__note">
                Web viewer at{" "}
                <code className="docs-inline-code">http://localhost:5173</code>
                {" "}— API at{" "}
                <code className="docs-inline-code">http://localhost:8000</code>
                {" "}(interactive Swagger docs at{" "}
                <a href="/docs" className="selfhost-inline-link">/docs</a>).
              </p>
            </Reveal>

            <Reveal className="selfhost-qs-block" delay={80}>
              <p className="selfhost-qs-block__label mono">Enable the GPU producer</p>
              <pre className="docs-code-block"><code>{String.raw`# Install the GPU env once (CUDA + gsplat):
.\scripts\setup-gpu-env.ps1

# Then run with GPU generation:
$env:ASTEL_PRODUCER = "gpu"
pnpm run up`}</code></pre>
              <p className="selfhost-qs-block__note">
                The GPU pipeline runs as a subprocess so PyTorch stays out of
                the API process. The stub producer (default) returns realistic
                fixture data instantly — useful for UI development without a GPU.
              </p>
            </Reveal>

            <Reveal className="selfhost-qs-block" delay={160}>
              <p className="selfhost-qs-block__label mono">Docker Compose (production)</p>
              <pre className="docs-code-block"><code>{`docker compose -f infra/docker-compose.yml up`}</code></pre>
              <p className="selfhost-qs-block__note">
                Includes: API, Postgres, MinIO, optional Temporal worker.
              </p>
            </Reveal>
          </div>
        </Section>

        {/* Environment variables */}
        <Section
          eyebrow="Configuration"
          title="Environment variables."
          lede="All configuration is via environment variables. No config files to edit."
          className="selfhost-env"
        >
          <Reveal>
            <div className="selfhost-env-table-wrap">
              <table className="selfhost-env-table">
                <thead>
                  <tr>
                    <th className="selfhost-env-table__col-name">Variable</th>
                    <th className="selfhost-env-table__col-default">Default</th>
                    <th className="selfhost-env-table__col-desc">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {ENV_VARS.map((v) => (
                    <tr key={v.name}>
                      <td className="mono selfhost-env-table__name">{v.name}</td>
                      <td className="mono selfhost-env-table__default">{v.default}</td>
                      <td className="selfhost-env-table__desc">{v.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>
        </Section>

        {/* Scaling */}
        <Section
          eyebrow="Scaling"
          title="From single GPU to cloud fleet."
          className="selfhost-scaling"
        >
          <div className="selfhost-scaling-grid">
            <Reveal className="selfhost-scale-card">
              <p className="selfhost-scale-card__eyebrow mono">Preview pool</p>
              <p className="selfhost-scale-card__body">
                L4 / L40S-class GPUs handle L0–L2 cheaply and quickly. Multiple
                workers can run in parallel — workers are stateless, so scaling
                horizontally is safe.
              </p>
            </Reveal>
            <Reveal className="selfhost-scale-card" delay={80}>
              <p className="selfhost-scale-card__eyebrow mono">Refine pool</p>
              <p className="selfhost-scale-card__body">
                A100 / H100 80 GB nodes handle L3 surface refinement and L4
                appearance at high splat budgets. Spot-instance tolerant: all
                stages are resumable from the last checkpoint.
              </p>
            </Reveal>
            <Reveal className="selfhost-scale-card" delay={160}>
              <p className="selfhost-scale-card__eyebrow mono">Worker autoscaling</p>
              <p className="selfhost-scale-card__body">
                Queue depth drives autoscaling — Temporal signals or Celery
                monitor. State lives in Postgres + S3 (MinIO locally), so
                workers can be added or removed at any time.
              </p>
            </Reveal>
            <Reveal className="selfhost-scale-card" delay={240}>
              <p className="selfhost-scale-card__eyebrow mono">CPU-heavy stages</p>
              <p className="selfhost-scale-card__body">
                SfM (structure from motion), SDF extraction, and convex
                decomposition are CPU-bound. Size these nodes separately —
                do not waste GPU nodes on them.
              </p>
            </Reveal>
          </div>
        </Section>

        {/* CTAs */}
        <Section className="selfhost-cta">
          <div className="selfhost-cta__inner">
            <h2>Ready to deploy?</h2>
            <p>
              Clone the repo, run{" "}
              <code className="docs-inline-code">pnpm run up</code>, and the
              full stack is live in under a minute. The interactive API docs
              are at <code className="docs-inline-code">/docs</code>.
            </p>
            <div className="selfhost-cta__buttons">
              <a href="/docs" className="cta-link cta-link--primary">
                API reference
              </a>
              <CtaLink to="/studio" variant="ghost">Open Studio</CtaLink>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
