# RA7b — Consumer Free-Tier Strategy & the Open-Model Quality Gap

*Added 2026-06-12 (session 1, second pass) in response to founder direction: "first we need a
Meshy competitor that can do free tier better than Meshy." Verifies the quality-gap critique,
adapts the plan, and records what changed.*

## 1. The critique, verified honestly

**Claim:** "Open models like TRELLIS can do what Meshy does but are very far behind — not
clean, not realistic, not printable."

**Verdict: half true, and the half that's true is the half we already planned to build.**

What June-2026 third-party comparisons actually say
([trellis2.app comparison](https://trellis2.app/blog/best-ai-3d-model-generator),
[3DAI Studio review](https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-image-to-3d-tools-2026),
[Meshy's own comparison page](https://www.meshy.ai/compare/meshy-vs-trellis-2)):

- **Raw geometry/texture quality:** TRELLIS.2 is called "the quality king of image-to-3D"
  with the highest texture resolution and color accuracy. The *raw-model* gap vs Meshy has
  closed since the TRELLIS-v1 era (the critique is accurate for v1, stale for v2).
- **Cleanliness/printability:** Meshy still wins — "cleanest, most watertight meshes, native
  STL/3MF," safest for printing. TRELLIS.2's O-Voxel embraces open surfaces and non-manifold
  geometry — great for fidelity, *hostile to printing* out of the box.
- **Product experience:** raw TRELLIS.2 is a research checkpoint: no cleanup, no retry UX,
  no scale estimation, no print checks, no asset management, no editing. That delta — not the
  neural network — is most of what a consumer perceives as "Meshy is better."

**Conclusion (unchanged from meshy-analysis §7.1, now evidence-backed):** Meshy's moat is the
boring pipeline. The model race has been substantially closed *for us, for free, under MIT* by
Microsoft. What stands between an open checkpoint and a Meshy-grade consumer product is
exactly the finishing pipeline AURIGA's layer stack already specifies: cleanup → solidify
(L5 SDF → watertight) → real-world scale (L1 metric) → quality report (Truth Meter) → print
checks (.3mf). **Printability in particular is a pipeline property, not a model property** —
our splat→SDF→watertight path makes *any* generator's output printable, which directly
neutralizes the "not exactly printable" weakness of open models.

## 2. Meshy's free tier — the target, verified

([pricing](https://www.meshy.ai/pricing), [help center](https://help.meshy.ai/en/articles/12062933-what-are-your-prices-and-plans-offered-and-do-you-have-monthly-annual-plans), [costbench](https://costbench.com/software/ai-3d-generation/meshy/))

| Meshy Free constraint | The opening it leaves |
|---|---|
| 100 credits/mo (~5–10 generations) | Generosity: splat L0–L2 previews cost us cents — we can offer 5–10× more free exploration |
| **Assets forced public, CC BY 4.0** | **Private assets on free** — huge for hobbyists; costs us nothing |
| ~10 downloads/mo, low queue priority, 1 queued task | Less artificial scarcity at preview tier |
| No API on free | Free API key with preview-tier quota (devs = our flywheel) |
| **No capture input at any price** | Photo/video → asset on the FREE tier = a capability their $60/mo tier doesn't have |
| Cloud-only | **Local mode**: bring your own GPU → unlimited free generation. Structurally impossible for Meshy's business model |

## 3. The free-tier doctrine (new product decision, binding)

"Better free tier than Meshy" ≠ more free text-to-3D credits (a cash-burning race we'd lose).
It means winning on three axes simultaneously:

1. **Generosity** — unlimited-feeling preview exploration (L0–L2 are cheap by design — this is
   why the layer stack maps to credit psychology); private assets; meaningful free API quota.
2. **Capability** — things free AURIGA does that paid Meshy cannot: video/photo capture,
   relight-correct splats, physics-aware export, Truth Meter, local mode.
3. **Trust** — free users get the honest quality report too. Trust is a free-tier feature that
   costs compute ~nothing and converts professionals.

**Beta economics:** the founder's own hardware (2×4090 + 3×3080) serves a closed beta free
tier comfortably — preview pool on the 3080s, refine on the 4090s — $0 cloud cost until
demand proves itself. This *is* the spec's local-first architecture earning its keep early.

## 4. What changes in the plan

| # | Change | Where |
|---|---|---|
| C1 | **Free-tier doctrine** (generosity/capability/trust axes) adopted as binding product decision | DECISIONS.md §product |
| C2 | **Blind-eval harness added to M1 exit criteria**: fixed corpus (20 prompts + 20 images + 10 captures), blind side-by-side vs Meshy free, Tripo, raw TRELLIS.2. M3 cannot exit until AURIGA ≥ raw TRELLIS.2 on all and ≥ Meshy-free on majority, blind-judged. Also becomes marketing material (honesty brand). | DECISIONS.md, M1/M3 gates |
| C3 | **Consumer UX is first-class from M2**, not API-first-only: drag-drop video/photos → asset with zero settings shown by default. (API still ships underneath, per spec §7.) | M2 scope |
| C4 | **"Finishing pipeline" reframed as the consumer-quality workstream** and pulled earlier: cleanup, auto-retry policies, scale estimation, watertight/print checks get their own M2/M3 acceptance metrics instead of waiting for M4 polish | M2–M4 scope |
| C5 | **Marketing/positioning guardrail**: do NOT lead with text-to-3D-parity claims at launch; lead with capture + print + honesty + free generosity. Text-to-3D is offered, improves behind the blind-eval gate | 06-competitors-positioning.md |
| C6 | **Dev environment decision**: GPU boxes stay Windows; we use **WSL2** (not dual-boot) — CUDA-mature, no reboot friction, agent-driveable over SSH. One-time setup per box; founder does ~3 commands, agent does the rest remotely | DECISIONS.md §infra |

## 5. What stays the same (and why the critique doesn't break it)

- **Splats-only layered product** — binding (spec §1); also the only defensible position vs
  Meshy's mesh incumbency.
- **All RA1–RA6 stack choices** — the critique *strengthens* TRELLIS.2-prior distillation
  (D#2): we inherit the "quality king" geometry and add the cleanliness layer it lacks.
- **Capture-first M2** — reinforced: capture is the free-tier capability wedge AND builds the
  finishing pipeline that generative quality needs. Generative (M3) follows immediately on
  shared infrastructure. (Founder can override ordering; recommendation is strongly capture-first.)
- **Honesty over hype** — now also the answer to "are open models really good enough":
  the blind-eval harness measures it instead of us asserting it.

## Sources

- https://www.meshy.ai/pricing · https://help.meshy.ai/en/articles/12062933 · https://costbench.com/software/ai-3d-generation/meshy/ · https://help.meshy.ai/en/articles/9991982-can-i-get-free-credits
- https://trellis2.app/blog/best-ai-3d-model-generator · https://trellis2.app/blog/meshy-ai-vs-trellis-vs-tripo · https://www.meshy.ai/compare/meshy-vs-trellis-2
- https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-image-to-3d-tools-2026 · https://www.3daistudio.com/blog/best-3d-model-generation-apis-2026
