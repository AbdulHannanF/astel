# Astel — Launch Readiness Checklist

Status as of M6 (session 29, 2026-06-19). Honest by construction (CLAUDE.md §3 honesty).

## 1. Pipeline & layers (L0–L7)

- [x] L0 seed / L1 dense / L2 coarse gaussians — capture + generative paths (M2/M3)
- [x] L3 refined surface gaussians (2DGS) — hero layer (M3)
- [x] L4 appearance / relighting (M4)
- [x] L5 collision / solidity + print path .3mf/.stl (M4)
- [x] L6 physics-material + articulation (M4)
- [x] L7 dynamics core (deformation field + timeline + binary pack), bound into `.astel`, validated vs analytic ground-truth motion (M6)
- [ ] L7 real per-frame 4DGS tracking from video — NOT done (requires the GPU deformable-recon stage; the `write_dynamics_layer` capability + manifest binding exist and are tested, but video produces a static L3 + honest "dynamics not tracked" caveat)

## 2. Modalities

- [x] Text → 3D (SDXL/FLUX → TripoSplat L2 → 2DGS L3)
- [x] Image → 3D (TripoSplat L2 → 2DGS L3)
- [x] Video upload handled honestly — produces the render-then-refit placeholder explicitly labeled `conditioning: none` (no fabricated motion). The CLI `_produce_video` runs a real static reconstruction **when supplied a frame**, but API frame-extraction + sharpest-frame selection are NOT wired (roadmap G1), so the product does not yet reconstruct video content.
- [ ] Video → static reconstruction wired end-to-end through the API (frame extraction + `--image` for the video modality) — NOT done (roadmap G1)
- [ ] Higher-quality text→multiview bridge — standing enhancement, not built

## 3. Scene seeds (M6)

- [x] Multi-object scene layout schema + ground-contact/no-overlap composition core (tested)
- [x] Layout-LLM stage on offline fixtures (no API key, no spend)
- [ ] End-to-end scene generation wired into the API producer — composition + layout cores exist and are tested; full API wiring is a follow-on

## 4. LOD streaming (M6)

- [x] Importance-weighted LOD tiering + per-platform budgets (mobile/web/console/cinematic), nested-subset guarantee (tested)
- [x] Producer emits `l3.lod.json` descriptor + downsampled tier PLYs when the cloud is large enough
- [x] Web LOD consumer module (parse descriptor, select tier for budget/platform) — tested
- [ ] Live LOD streaming wired into the web viewer render loop — consumer module exists + tested; render-loop integration is a follow-on

## 5. Formats & exports

- [x] .ply (archival), .spz + .sog (compressed), .glb (KHR_gaussian_splatting), `.astel` package + manifest
- [x] Print: .3mf + .stl with printability checks
- [x] engine.json sidecar for Unity/UE5 plugins

## 6. Engine & pipeline integrations

- [x] Unity package + UE5 plugin (consume engine.json) — code complete
- [ ] Engine plugins compiler-verified in CI — NOT done (no licensed Unity/UE5 runners in this environment; only the engine.json contract is tested in Python)
- [x] Python SDK + TypeScript SDK + MCP server

## 7. Quality, testing & gates

- [x] All library + service gates green (ruff · mypy --strict · pytest / tsc -b · eslint · vitest), re-run and verified
- [x] Honesty contract: unmeasured metrics are explicit null + reason; no fabrication over real data
- [x] Automated CI pipeline wired — `.github/workflows/ci.yml` runs the CPU gates (9 libs · api · sdk-python · loadtest · web + TS packages) on push/PR; `gpu.yml` runs the CUDA gates on a self-hosted runner (manual). The workflows are authored + YAML-valid and mirror the manually-verified gates, but have **not executed on a remote runner yet** (the repo is local-only — push to a GitHub remote to activate). Until then, gates are still run by hand.
- [x] Load-test harness available (`tools/loadtest`) — note: not yet executed end-to-end against a running server

## 8. Security

- [x] Security review of M6 changes — done (session 29); see **Appendix A**. The `.bin` deformation reader is now size-validated against truncated/oversized-header (amplification) input. Broader untrusted-package hardening (zip-bomb / path-traversal / oversized-accessor across the full `astel_format` read path) remains a launch **P0** (roadmap N2).

## 9. Hardware & deployment

- [x] Local self-host minimum documented (1× 24GB GPU, 64GB RAM) and cloud tiers (docs/setup)
- [ ] Production deploy (Helm/K8s autoscaling on queue depth) — designed, not load-validated
- [ ] Monitoring / alerting / rollback runbook — not yet written

## 10. Outstanding before public launch (summary)

### P0 (Critical blockers)

- **CI pipeline** (wired — pending remote execution): `.github/workflows/ci.yml` + `gpu.yml` are authored and run the full gate suite (`ruff`, `mypy --strict`, `pytest`, `tsc -b`, `eslint`, `vitest`); they need a GitHub remote to execute and a first green run to confirm
- **Security hardening**: Audit + harden `.astel` package deserialization (zip + JSON + binary `.bin` deformation reader) against malicious inputs
- **Production deployment**: Validate Helm/K8s autoscaling under realistic load; document rollback runbook

### P1 (High-priority, post-MVP)

- **Engine CI verification**: Add licensed Unity/UE5 build runners to validate plugin compiler correctness
- **Live LOD wiring**: Integrate LOD consumer module into web viewer render loop for streaming on large assets
- **Scene API wiring**: Connect layout-LLM + composition stages into the API producer for end-to-end multi-object generation
- **Monitoring & alerting**: Define SLIs/SLOs; add logging aggregation, metrics dashboards, incident runbook

### P2 (Strategic enhancements)

- **Text→multiview bridge**: Implement higher-quality text→multiview bridge (beyond SDXL/FLUX)
- **Real 4DGS video tracking**: GPU deformable-reconstruction stage to extract L7 dynamics from video (currently: static L3 + honest caveat)

---

**Summary**: M6 features (L7 dynamics core, scene composition, LOD tiering, integrity hardening) are implemented and tested at the library/producer level. The remaining launch blockers are infrastructure (CI pipeline now wired — pending a remote to execute; production deployment validation; broader untrusted-package hardening) and GPU-deferred paths (real-time 4DGS video tracking, live LOD streaming in the UI). Full plan: [research/18-post-m6-roadmap.md](research/18-post-m6-roadmap.md).

---

## Appendix A — M6 security review (session 29)

**Scope reviewed:** the new M6 attack surface. M6 added no API endpoints or
request handling — the API was unchanged. The genuinely new surface is the
**untrusted-`.astel` deserialization path** introduced by the L7 binary buffer.

**Finding (fixed) — `astel_dynamics.read_deformation_bin` amplification.** The
reader took `N/K/F` from the file header and sliced the body accordingly. Python
slicing is memory-safe (no buffer overread is possible), but a tiny crafted file
could declare enormous arrays, producing a confusing `reshape` failure rather
than a clean rejection, and the declared sizes were unbounded. **Fix:** an exact
file-size check — `len(data) == 8 (magic) + 12 (header) + n_floats·4` — now runs
**before** any allocation/slice, rejecting truncated, trailing-junk, and
amplification-crafted headers with a clear message. A small file can never pass
while claiming large arrays. Regression tests added: truncated body, and a
~few-byte file declaring `N=K=F=2^20`.

**Reviewed, low-risk (no change needed in M6):**
- `astel_format` package read uses stdlib `zipfile` + `json` with constant member
  paths; LOD/dynamics file writes use constant names (no user-controlled paths).
- The scene layout-LLM stage runs offline (`FixtureAdapter`) by default — no
  network, no key, no spend.

**Deferred to launch P0 (roadmap N2):** a full hardening pass of the untrusted
`astel_format` package reader — zip-bomb (decompressed-size caps), path-traversal
on member names, and oversized-accessor/buffer-count validation — plus a single
`validate_untrusted()` entry point. This is required before accepting
third-party `.astel` uploads in production.
