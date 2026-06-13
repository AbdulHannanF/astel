# RA8 — Paper Deep-Reads: Finalizing the Draft Decisions

*Session 2, 2026-06-13. Method-section reads of the six papers that gate the 🟡 draft rows in
[DECISIONS.md](DECISIONS.md). For each: the exact losses/regularizers/representations we adopt
(equations in plain text), how they map onto **gsplat** modes/flags, what we must reimplement,
and gotchas. GPU smoke tests deferred per user decision 2026-06-13 — these reads settle the
**design**, not the empirical A/Bs. License lineage per [LICENSE_AUDIT.md](LICENSE_AUDIT.md):
all original repos below are Inria-NC-derived and **never vendored**; we reimplement the
published math on gsplat (Apache-2.0).*

---

## 1. 2DGS — Huang et al. (arXiv:2403.17888)

**Representation we adopt (L3 base):** each splat is a 2D oriented disk (surfel), centred at
`p_k`, with two orthogonal tangent vectors `t_u, t_v`, scales `(s_u, s_v)`, and an **explicit
normal `t_w = t_u × t_v`**. Points on the disk: `P(u,v) = p_k + s_u·t_u·u + s_v·t_v·v`, packed
into a homogeneous `H = [[RS | p_k],[0|1]]` with `R=[t_u,t_v,t_w]`, `S=diag(s_u,s_v,0)`. Pixels
are evaluated by **ray–splat intersection** (intersect the two pixel planes with the disk plane,
solve for `(u,v)`), not by projecting a 3D covariance — this is what makes the normal exact and
view-consistent. A low-pass filter `Ĝ(x)=max{G(u(x)), G((x−c)/σ)}`, σ=√2/2, prevents aliasing
when the disk is viewed edge-on.

**Losses we adopt:**
- **Depth distortion** `L_d = Σ_{i,j} ω_i ω_j |z_i − z_j|` (ω = `T_i·α_i·Ĝ_i` blend weight along
  a ray). Concentrates weight to a thin surface band. Implemented in one pass via prefix sums
  (`A_i, D_i, D²_i`) as in the paper — cheap.
- **Normal consistency** `L_n = Σ_i ω_i (1 − n_i^T N)` where `n_i` is the splat normal and `N`
  is the normal from the rendered depth gradient `N = normalize(∇_x p_s × ∇_y p_s)`, `p_s` at the
  **median-depth** surface point (opacity-accumulation 0.5).
- **Total:** `L = L_c + α·L_d + β·L_n`, **α=1000 bounded / 100 unbounded, β=0.05**. `L_c` =
  3DGS's `(1−λ)L1 + λ·D-SSIM`, λ=0.2.

**Render outputs we rely on:** mean depth `Σω_i z_i/(Σω_i+ε)`, **median depth** (outlier-robust,
used for meshing/SDF), and the depth-gradient normal map.

**gsplat mapping:** gsplat ships a **2DGS rasterization mode** with depth+normal render and a
~4 GB VRAM overhead (RA1). The depth-distortion and normal-consistency terms are **available as
gsplat utilities** but we wrap our own loss module so weights and the median-depth surface point
are under our control (product needs configurable α per bounded/unbounded auto-detect). **Anti-
aliased mode** = gsplat's AA flag (Mip-Splatting equivalent). The low-pass `σ=√2/2` filter is in
gsplat's 2DGS kernel — do not re-add it.

**We reimplement:** nothing in the rasterizer (gsplat has it). We **own**: the loss-weight
schedule, bounded/unbounded auto-detection, and the median-depth → TSDF handoff to L5.

**TSDF extraction (feeds L5, never exported as asset):** fuse per-view rendered depth with Open3D
TSDF, **voxel 0.004, truncation 0.02** (object-scale defaults; we rescale by metric extent).

**Gotcha:** 2DGS **over-smooths fuzzy/volumetric content** (hair, foliage) — confirmed across
2024–2026 comparisons. This is the one thing the deep-read does *not* settle; it needs the
GPU A/B vs 3DGS+GOF on the fixed corpus (deferred). So the L3-**representation** row stays 🟡.

**Reported numbers (for §09 targets):** DTU mean Chamfer **0.80 mm**; Tanks&Temples F1 **0.32**;
Mip-NeRF360 indoor PSNR 30.40 / SSIM 0.916 / LPIPS 0.195, outdoor 24.34 / 0.717 / 0.246.

---

## 2. PGSR — Chen et al. (arXiv:2406.06521, TVCG 2024)

**Why it matters:** **published surface-accuracy leader** — DTU mean Chamfer **0.52 mm** (0.47
with tuned hyperparams per 2026 follow-ups), Tanks&Temples F1 **0.52**, Mip-NeRF360 PSNR ~27.25.
This is the regularizer set that takes us from "2DGS-good" to "best-in-class geometry." We adopt
its **losses on top of the 2DGS surfel base** (not its NC code).

**Representation:** flatten gaussians by penalising the smallest scale → planar primitive whose
**normal `n_i` = min-scale axis**, plane distance `d_i = (R_c^T(μ_i − T_c))^T (R_c^T n_i)`. Render
an α-blended **normal map** `N` and **distance map** `D`; recover **unbiased depth**
`D(p) = D / (N(p)·K^{-1}·p̃)` — dividing distance by normal removes the blend-weight depth bias
that plain 3DGS depth has.

**Losses we adopt:**
- **Flattening** `L_s = ||min(s1,s2,s3)||_1`, **λ₁ = 100**. (Our 2DGS base already has a zero
  third scale, so this term is near-redundant for us — keep at low weight or drop; it's the
  bridge term for a 3DGS base. Decision: **we run the 2DGS base, so L_s is optional**; keep PGSR's
  *consistency* terms, which are the real value.)
- **Single-view geometric** (edge-aware local-plane normal):
  `L_sv = (1/|W|) Σ_p (1−∇Ī_p)² · ||N_d(p) − N(p)||_1`, **λ₂ = 0.015**. `N_d` from neighbouring-
  pixel depth cross products; `(1−∇Ī)²` down-weights image edges so we don't flatten true creases.
- **Multi-view geometric** (forward–backward reprojection): homography
  `H_{rn} = K_n(R_{rn} − T_{rn} N_r^T/d_r) K_r^{-1}`, loss
  `L_mvgeom = (1/V) Σ w(p_r)·||p_r − H_{nr}H_{rn}p_r||`, weight `w = 1/exp(φ)` for φ<1 else 0,
  **λ₄ = 0.03**.
- **Multi-view photometric** (7×7 patch NCC on grayscale):
  `L_mvrgb = (1/V) Σ w(p_r)·(1 − NCC(I_r(p_r), I_n(H_{rn}p_r)))`, **λ₃ = 0.15**.
- **Exposure compensation** per image `I_a = exp(a_i)·I + b_i`; SSIM-switched RGB loss with λ=0.2.

**gsplat mapping:** gsplat gives us the rasterizer + 2DGS depth/normal renders; **PGSR's
multi-view consistency, single-view edge-aware normal, and exposure terms are NOT in gsplat** —
they are **our reimplementation**, the largest custom-loss lift in L3. The homography warp +
patch-NCC needs a small CUDA/Triton kernel (per-pixel 7×7 gather across a paired view) or a
vectorised PyTorch `grid_sample` version first (correctness before speed).

**We reimplement:** single-view edge-aware normal loss, multi-view geometric (FB-reprojection),
multi-view photometric (patch-NCC + homography), per-image exposure affine. **All published math,
all on gsplat — zero NC code.** This is the RA1 "PGSR-style multi-view consistency" line made
concrete.

**Interaction with DN-Splatter priors:** PGSR's single-view normal term is **redundant with
monocular-normal priors** when capture data has a good normal predictor; keep both behind a flag
and let RA1-Q4's ablation (deferred) decide the minimal set. So the **L3-losses row stays 🟡** on
the *exact weight subset*, but the **technique set is settled** → we note that.

**Gotcha:** multi-view terms need ≥2 overlapping views — they **silently no-op on single-image
generative path** (only capture/multi-view has them). The generative path leans on the geometry
prior (TRELLIS.2) + single-view normal instead. Document per-modality which terms are active.

---

## 3. TRELLIS — Xiang et al. (arXiv:2412.01506, CVPR'25) + TRELLIS.2 (2026)

**SLAT representation:** a sparse set of **~20 K active voxels** on a 64³ grid, each active voxel
`i` carrying position `p_i` + a **local latent `z_i`** (encoded by a sparse VAE from multiview
DINOv2 features). Generation = **two rectified-flow transformers**: (1) sparse-structure flow
`G_S` over a 3D-conv-VAE-compressed occupancy grid → `{p_i}`; (2) SLAT flow `G_L` → `{z_i}`,
conditioned by **DINOv2** (image) / CLIP (text) via cross-attention.

**The Gaussian head (what we use for L2):** each `z_i` decodes to **K gaussians** with predicted
**position offset `o` (tanh-bounded to the parent voxel: `x_i^k = p_i + tanh(o_i^k)`), color,
scale, opacity, rotation**; supervised by **L1 + D-SSIM + LPIPS** vs ground-truth renders. This
is a **native gaussian generator** — exactly the L2 coarse-splat producer we want, **MIT** (v1
code+weights).

**TRELLIS.2 (4B, 2026) — verified mesh-only:** the new "O-Voxel" field-free sparse-voxel rep is
SOTA image-to-3D with **full PBR (base color/metallic/roughness/alpha)** but **bidirectional
O-Voxel↔mesh ⇒ output is mesh** (the v1 gaussian/RF decoders were dropped). PLY export is point-
cloud/mesh-derived, **not a trained gaussian head**. **MIT code+weights.**

**Architecture decision this settles (was 🟡):**
- **L2 generator = TRELLIS-image-large gaussian head (v1, MIT).** Settled: it is the only
  open *native* gaussian generator with this quality and a permissive license.
- **L3 geometry/PBR prior = TRELLIS.2 internal O-Voxel.** We render its O-Voxel/mesh to multi-
  view depth+normal+PBR and **distill into our 2DGS surfels** (the prior is internal scaffolding,
  spec §1 — never exported). This **"generate-prior-then-distill"** path is the one bet that the
  deep-read **cannot** fully de-risk — distillation fidelity (does TRELLIS.2 geometry survive the
  surfel fit?) is the scheduled GPU experiment (deferred). **Stays 🟡** as a *de-risk* item, but
  the **model selection underneath it is settled** (no open alternative competes).

**gsplat mapping:** TRELLIS gaussians load directly as gsplat gaussians (same attribute set:
mean, scale, rot-quat, opacity, SH/color) → **L2 output is a gsplat init for the L3 optimizer**
with no format bridge. TRELLIS's own rendering uses `diffoctreerast` (RF head) — **we never invoke
it**; we only call the gaussian decoder, sidestepping the NC submodule entirely (LICENSE_AUDIT
boundary).

**We reimplement:** nothing in TRELLIS; we **own** the distillation harness (TRELLIS.2 render-set
→ 2DGS fit + Chamfer-vs-prior metric) and the L2→L3 init bridge.

**Gotcha:** TRELLIS.2 nvdiffrast/nvdiffrec deps are **NVIDIA-NC** — our usage boundary (geometry
decode only, our renderer for prior views, never nvdiffrast) is defined in LICENSE_AUDIT.md; the
**clone-time import-graph check is still required** before we ship (deferred to GPU session).

---

## 4. MapAnything — Keetha et al. (arXiv:2509.13414, Meta)

**Factored representation (this is the keystone for L0/L1 and the Truth Meter scale channel):**
per view it predicts **(a) ray directions `R_i`** (unit, = intrinsics), **(b) up-to-scale ray
depth `D̃_i`**, **(c) camera pose** (quaternion `Q_i` + up-to-scale translation `T̃_i`), and **one
global metric scale factor `m`**. Local up-to-scale points `L̃_i = R_i·D̃_i` → world
`X̃_i = O_i·L̃_i + T̃_i` → **metric `X_i = m·X̃_i`**. The single-`m` factoring is *exactly* the
"one honest scale with a confidence" design L1 needs — metric and pose are separable, so we get a
**directly reportable scale estimate** (Truth Meter input) instead of scale baked into geometry.

**Input flexibility (matches RA3's "consumes whatever exists"):** optional **ray
directions/intrinsics, poses, ray depth** are each encoded (shallow conv for ray/depth maps,
4-layer MLP for pose/scale; fused by **LayerNorm + summation** with image tokens) and **dropped
out probabilistically in training** (geometric-input prob 0.9; each factor 0.5; metric-scale
withheld 0.05) — which is *why* one checkpoint serves pose-free phone video, posed multi-photo,
and depth-completion without retraining.

**Architecture:** **DINOv2 ViT-L** patch features (1024-d at H/14×W/14) → **24-layer alternating-
attention transformer** (12 heads, dim 768, MLP×4) → **DPT head** for dense per-view
(rays/depth/mask/**confidence**), pooled conv pose head, 2-layer MLP for `m` (exp-scaled). Trained
2–4 views, generalises to **100+** views.

**Losses (informs our trust-channel design):** factored, **log-space** geometry
(`flog(x)=(x/‖x‖)·log(1+‖x‖)`); ray-direction L2; **geodesic** rotation; normalized translation;
**confidence-weighted pointmap loss (10×)**; ambiguity-mask BCE (0.1×); a **factored scale loss**
with stop-gradient on the up-to-scale branch (so scale errors don't corrupt shape).

**Reported (for §09 targets):** images-only multi-view metric depth **rel ≈ 0.057**; with
intrinsics+poses+depth **rel ≈ 0.042**; two-view rel 0.18 (beats VGGT 0.20, MASt3R 0.25);
single-image calibration **1.18° angular** (beats MoGe-2 1.95°).

**License:** **Apache-2.0 code + `map-anything-apache` checkpoint** (6 permissively-licensed
datasets); CC-BY-NC variant exists and is **not used**. This row was already ✅ in DECISIONS;
the deep-read **confirms** the metric-scale + confidence channel is first-class, strengthening the
L1 scale-consensus design — no change needed beyond §09 target numbers.

**gsplat mapping:** MapAnything is upstream of gsplat — its metric point cloud + poses are the
**L0/L1 init and camera set** the gsplat L3 optimizer consumes. Its per-pixel **confidence** seeds
our per-gaussian provenance scalar (spec §10.4 sacred channel).

**Gotcha:** MapAnything is **scene-oriented**; our M2 case is **orbit-around-a-single-object**.
Object-centric accuracy + GLOMAP/COLMAP BA refinement on dense orbits is the deferred smoke test
(RA3-Q2). The **front-end choice is settled** (already ✅); only object-centric tuning is empirical.

---

## 5. PhysGaussian — Xie et al. (arXiv:2311.12198, CVPR'24)

**Method we adopt for L6 (MPM-on-gaussians, reimplemented on NVIDIA Warp — Apache):** each
gaussian **is** an MPM material point. Kinematics: position `x_p(t)=φ(X_p,t)`, **covariance
`a_p(t)=F_p A_p F_p^T`** where `F_p` is the deformation gradient (local-affine assumption keeps a
deformed gaussian gaussian → **direct splat render, no mesh**). This is "**what you see is what you
simulate**" — the same kernels simulate and render, the property that makes the Physics Sandbox
(spec §8.2) honest.

**MLS-MPM step (explicit):** grid momentum
`m_i/Δt (v_i^{n+1}−v_i^n) = −Σ_p V_p^0 (∂Ψ/∂F)(F_p^E)(F_p^E)^T ∇w_{ip} + f_i^{ext}`; deformation
update `F_p^{E,n+1} = (I + Δt Σ_i v_i^{n+1} ∇w_{ip}^T) F_p^{E,n}`; plasticity return-map
`F^E ← Z(F^E)`. B-spline `w_{ip}` (C¹).

**Three things we must implement that aren't "just MPM":**
1. **Internal particle filling** — reconstructed gaussians are surface-shells (hollow); extract a
   density field `d(x)=Σ_p σ_p exp(−½(x−x_p)^T A_p^{-1}(x−x_p))`, ray-march low→high opacity
   transitions, insert interior particles inheriting σ/SH from nearest surface gaussian, covariance
   `diag(r²,r²,r²)`, `r=(3V_p^0/4π)^{1/3}`. **This couples directly to our L5 SDF** — the SDF gives
   us the interior far more robustly than ray-marching opacity, so **AURIGA fills from the L5 SDF,
   not PhysGaussian's opacity ray-march** (our improvement; SDF already exists for print).
2. **SH rotation under deformation** — polar-decompose `F_p=R_p S_p`, rotate query view dir
   `f^t(d)=f^0(R_p^T d)` so texture follows the object. Must implement in the sandbox renderer.
3. **Anisotropy regularizer** `L_aniso = mean_p(max(max S_p/min S_p, r) − r)` to stop over-
   elongated splats "plushing" under large strain. Optional, used at reconstruction time.

**Constitutive models to port:** fixed-corotated (elastic), von-Mises (metal), Drucker-Prager
(sand/granular), Herschel-Bulkley (viscoplastic). The **material assignment is L6's LLM/VLM job**
(per-region density/friction → which constitutive model + parameters).

**gsplat mapping:** **none — MPM is not a gsplat concern.** gsplat only renders the deformed
gaussians each frame; the simulator is a **separate Warp module** consuming gsplat gaussian state
(mean, covariance via scale+rot, opacity, SH) and writing back updated mean+covariance per step.
Clean separation: Warp owns dynamics, gsplat owns the frame.

**We reimplement:** the **entire MPM-on-gaussian coupling on Warp** (PhysGaussian's repo is
Taichi/NC-adjacent; the math is published). Largest single L6 build. **Stays 🟡** — Warp-MPM
example maturity + CUDA-graph capture for the interactive loop is a GPU evaluation (RA4-Q2,
deferred). The **engine choice (Warp) and the math are settled**; the empirical "Warp vs Taichi
sandbox latency" is not.

**Gotcha:** explicit MPM needs small Δt; i-PhysGaussian (arXiv:2602.17117) implicit integrator
gives larger stable steps for interactivity — keep as a **drop-in integrator upgrade** behind the
same coupling layer (note, not adopt-now).

---

## 6. RTR-GS — (arXiv:2507.07733, ACM MM'25)

**Role:** the **published relighting quality bar** for L4 and the runner-up architecture to our
chosen deferred-PBR-on-gsplat. Read to decide *how much* of it we adopt vs our deferred-shading
plan.

**What it does:** two simultaneous branches — a **hybrid radiance-transfer** branch (geometry +
view-dependent appearance, fast to converge) and a **PBR-decomposition** branch (the relightable
output). Diffuse `C_d ≈ ρ_d Σ c_j c_j^t` (global SH light `c_j`, per-gaussian transfer `c_j^t`);
view-dependent specular via a 3-layer MLP `c_j^t(o)=G(f_t,o)`. Final composite
`I = C_r(1−R_i) + C_ref·R_i`.

**PBR branch (what we actually adopt the structure of):** per-gaussian **albedo `c`, metallic
`m`, roughness `r`, indirect light `L_ind`**; **split-sum IBL** from a filtered env cubemap +
BRDF LUT (GGX NDF, Schlick-Beckmann geometry); diffuse
`L_d ≈ (c/π)[V·L_d^dir + (1−V)·L_d^ind]`, specular split-sum; **visibility `V` baked into an SH
voxel grid** (no per-frame ray trace). This is a **deferred-shading PBR pipeline on gaussians** —
i.e. it **validates our "deferred shading on gsplat" choice** rather than competing with it.

**Losses worth porting:** `L_n=||n−n̂_d||_2` (normal vs depth-derived pseudo-normal — **we already
have real 2DGS normals, so this is a consistency check, weaker need**); **white-light reg**
`L_light=Σ_c(L_c − mean_c L)` λ=0.003; **metal-reflection prior** `L_m=||m−R_i||` λ=0.1;
normal λ=0.02; staged `λ_PBR` 0→1 (geometry first, then decompose).

**Reported (for §09 targets):** TensoIR relight PSNR **30.10** (vs GShader 26.86, GS-IR ?);
Shiny-Blender relight **26.16**; Stanford-ORB relight **28.93**. NVS PSNR up to 41.4 (HR branch).

**Decision this settles (was 🟡 for L4):** **adopt deferred-shading PBR on gsplat with split-sum
IBL + jointly-optimised env map**, structure per RTR-GS's PBR branch, but **drive normals from our
2DGS L3** (RTR-GS estimates pseudo-normals; we have real ones → simpler, more accurate). Port the
white-light, metal-prior, and staged-PBR schedule. **Generative path seeds materials from
TRELLIS.2 PBR**; capture path decomposes during L3→L4. The **L4 technique is settled** → flip.
The **hybrid radiance-transfer branch is NOT adopted** (extra MLP + transfer vectors; our value is
honest PBR export to engines, not max NVS PSNR) — recorded as runner-up.

**Newer context (verified 2026-06-13):** GOGS (arXiv:2508.14563, glossy relighting on **Gaussian
surfels** — same 2DGS base as us) and IRGS (CVPR'25, 2D-gaussian ray tracing) are aligned
successors; **3DGRT (Apache)** still supplies our ray-traced visibility ground truth for the
Relight Studio. No permissive turnkey exists → L4 is **our code** regardless; RTR-GS settles the
*recipe*, not a dependency.

**gsplat mapping:** L4 = a **deferred shading pass** over gsplat's G-buffer (2DGS gives
position+normal+albedo render targets) + an optimisable env map (torch parameter) + split-sum
LUT. No gsplat fork — a render-target read + a torch shading module. 3DGRT/3DGUT (gsplat mode or
3dgrut) supplies secondary-ray ground truth for validation only.

**We reimplement:** the whole L4 (no permissive dependency exists). RTR-GS gives the loss/schedule
blueprint.

---

## Summary: what flips, what stays 🟡, and why

| Row (DECISIONS) | Deep-read verdict |
|---|---|
| L3 representation (2DGS) | **stays 🟡** — 2DGS over-smooths fuzzy content; A/B vs 3DGS+GOF needs GPU (deferred). 2DGS as the *default* base is sound; the gate is empirical. |
| L3 refinement losses (PGSR + DN-Splatter) | **technique set settled → ✅**; the exact minimal weight subset (PGSR vs monocular-prior overlap) is an ablation (noted, deferred). |
| Generative foundation L2 (TRELLIS v1 GS head) | **flip ✅** — only open native gaussian generator at this quality, MIT; TRELLIS.2 confirmed mesh-only so v1 is the head. |
| Generative geometry prior (TRELLIS.2 distill) | **model choice ✅, distillation fidelity stays 🟡** — the one bet a paper read can't settle; GPU de-risk deferred. |
| Multi-view guidance (MV-Adapter) | unchanged 🟡 — license verify pending (LICENSE_AUDIT shows MV-Adapter closed ✅ permissive); kept 🟡 only on "current best MV diffusion" empirical Q. |
| L6 physics (MPM on Warp) | **math + engine settled → ✅** for design; Warp example maturity / sandbox latency is GPU eval (stays noted). |
| L4 appearance (deferred PBR on gsplat) | **flip ✅** — RTR-GS validates deferred-PBR-on-gaussians; we improve it with real 2DGS normals + TRELLIS.2 PBR seed. |

Rows we **do not** touch: Task engine (other workstream), anything in LICENSE_AUDIT.md.

## Sources (arXiv IDs)
- 2DGS — arXiv:2403.17888 · https://arxiv.org/html/2403.17888v2
- PGSR — arXiv:2406.06521 · https://arxiv.org/html/2406.06521v2 · https://github.com/zju3dv/PGSR
- TRELLIS — arXiv:2412.01506 · https://arxiv.org/html/2412.01506v1 · TRELLIS.2: https://github.com/microsoft/TRELLIS.2 · https://microsoft.github.io/TRELLIS.2/
- MapAnything — arXiv:2509.13414 · https://arxiv.org/html/2509.13414v1
- PhysGaussian — arXiv:2311.12198 · i-PhysGaussian arXiv:2602.17117
- RTR-GS — arXiv:2507.07733 · related: GOGS arXiv:2508.14563, IRGS (CVPR'25)
