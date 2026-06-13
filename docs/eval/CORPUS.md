# AURIGA Blind-Eval Corpus v1 (FROZEN)

*Adopted 2026-06-13. This corpus and protocol are frozen on adoption — see §5 Freeze Policy.
It gates M3 exit (per `docs/research/07-free-tier-consumer-strategy.md` C2: AURIGA must score
≥ raw TRELLIS.2 on all 50 cases and ≥ Meshy-free on a majority, blind-judged) and is reused
verbatim as marketing material (the Truth Meter / honesty brand, CLAUDE.md §8).*

**Status of this document**: spec + protocol, ready to execute. Image/video assets themselves
are *not* checked into git (binary, license-tracked separately in `docs/eval/assets/` with a
manifest — see §2.3/§3.3). Prompt strings and scripts below ARE the frozen artifact for the
text and capture sections; image case *specs* are frozen, the actual image files are pinned
by content-hash once collected.

---

## 1. Twenty Text Prompts (T01–T20)

Each prompt is verbatim input to the text pipeline (CLAUDE.md §4 "Text →"). No paraphrasing
across releases — same string, every version, forever. Ordered roughly by difficulty axis,
not by importance; all 20 are in-scope for every score.

| ID | Prompt (verbatim) | Stresses | Expected failure modes |
|---|---|---|---|
| T01 | "A worn cast-iron skillet with a long handle, matte black surface, slight rust spots near the rim" | Hard-surface + material variation, slightly concave interior | Interior of pan under-resolved/flat; rust treated as texture-only with no surface pitting; handle-to-body joint geometry merges into a blob |
| T02 | "A pair of wire-frame round eyeglasses with thin metal temples and clear glass lenses" | Thin structures (temples, frame wire), transparency | Temples vanish or thicken into solid bars; lenses opaque or missing; frame topology disconnected from temples |
| T03 | "A wooden dining chair with four thin tapered legs, a slatted backrest, and a woven rattan seat" | Thin structures (legs), repeated fine geometry (slats, weave), articulated multi-part | Legs fuse to seat as a single mass; slats/weave collapse into a texture instead of geometry; legs of unequal length (no symmetry enforcement) |
| T04 | "A chrome-plated robot action figure with articulated joints at shoulders, elbows, and knees, mirror-polished surface" | Reflective/specular material, articulated multi-part, view-dependent appearance | Mirror surface baked as flat gray or environment-reflection ghosting baked into albedo; joints not separable as parts; L4 fails to decompose specular from albedo |
| T05 | "A clear glass wine bottle, empty, standing upright, with a paper label wrapped around the body" | Transparent/refractive material, text-on-object (label) | Glass rendered opaque/solid; label text illegible or smeared; refraction artifacts baked as opacity holes |
| T06 | "A small ceramic coffee mug with a painted logo reading 'CAFE' on the side and a glossy glaze finish" | Text-on-object, glossy/specular, simple convex hard-surface (sanity baseline) | Logo text garbled or illegible (known diffusion-model weakness); glossy highlight baked into texture as a fixed bright patch |
| T07 | "A fluffy orange tabby cat sitting upright, fur visible, tail curled around its paws" | Organic/fuzzy geometry, volumetric fine detail | Fur reduced to smooth solid surface; tail merges with body; splat budget concentrated on body leaving fur "bald" patches |
| T08 | "A leafy potted fern with many thin, overlapping fronds and visible soil texture in a terracotta pot" | Thin structures (fronds), high part-count repetition, concave container | Fronds clump into a green mass; pot interior (concave) not modeled — appears solid; soil texture flat |
| T09 | "A minimalist gold wedding band, plain, no stones, size 7 (approx 18mm inner diameter)" | Scale extreme (very small), reflective metal, simplicity stress-test | Scale estimator defaults to "generic object" size (way off from ~18mm); ring hole (concave/hollow) filled in as solid disc; metallic reflection baked flat |
| T10 | "A modern L-shaped sectional sofa in light gray fabric with six rectangular cushions" | Scale extreme (large), multi-part (cushions), soft/cloth material class for L6 | Cushions fused into the sofa body as one part; fabric texture/material misclassified as rigid (wrong L6 density/friction); L-shape proportions distorted |
| T11 | "A hollow ceramic vase, cylindrical, open at the top, with a narrow neck and a wide rounded base" | Concave/hollow object, thin-wall structure | Vase generated as a solid cylinder (opening filled in); neck-to-base wall thickness inconsistent, breaking printability checks at L5 |
| T12 | "An old wooden treasure chest with a curved lid, metal latch, and visible wood grain, lid open revealing an empty dark interior" | Concave/hollow (open interior), multi-part (lid hinge), articulation | Lid fused shut or interior filled solid; hinge not detected as an articulation joint in L6; interior left as a black hole (no geometry, just dark texture) |
| T13 | "A red bicycle with thin spoked wheels, a chain, and thin cable housings for brakes" | Thin structures (spokes, chain, cables) — known hard case across all generators | Spokes vanish entirely or merge into solid disc wheels; chain becomes a textured band; cables disappear below splat-density floor |
| T14 | "A stack of three hardcover books of different sizes and colors, slightly offset, with visible page edges" | Multi-part (separable objects), thin structures (page edges), scale relationships between parts | Books fuse into a single multi-colored block; page-edge thin geometry smoothed away; relative scale between the three books distorted |
| T15 | "A desk lamp with a flexible articulated arm made of multiple jointed segments, a conical metal shade, and a round base" | Articulated multi-part, thin segments, mixed materials (metal shade + cloth/rigid base) | Arm segments fuse into a single curved tube (joints not separable); shade interior (concave) not modeled; base material misclassified |
| T16 | "A garden gnome figurine, painted in bright primary colors, standing on a small rock" | Known Meshy failure-mode analog: small painted figurine with fine facial/hat detail | Face/hat fine detail smeared to "melted wax" look (documented Meshy weakness per `docs/meshy-analysis.md` §5.2); colors bleed across part boundaries |
| T17 | "The Utah teapot: a classic glossy white ceramic teapot with a curved spout, ring handle, and lid with a small knob" | Canonical-object correctness check — Meshy is documented to sometimes return the *wrong object* for "Utah teapot" (meshy-analysis.md §5.2) | Model returns a generic teapot ignoring "Utah teapot" canonical shape, or returns an unrelated object entirely; ring handle (concave/hollow) filled solid |
| T18 | "A construction crane in a simplified toy style, with a tall lattice tower, horizontal jib arm, and thin steel cables holding a hook" | Scale extreme (tall/thin proportions), thin structures (lattice, cables), hard-surface mechanical | Lattice tower collapses into a solid tapered column; cables disappear; overall proportions squashed toward a "default cube" aspect ratio |
| T19 | "A transparent plastic water bottle, half full of water, with a blue screw cap and a wrapped paper label" | Transparent material + a *visible internal volume* (water) — compound transparency/concave case | Water level not represented (bottle appears either fully solid or fully empty); cap material/color bleeds into bottle body; label text illegible |
| T20 | "A pair of scissors, open at a 45-degree angle, with metal blades and orange plastic handles, a visible pivot screw" | Thin structures (blades), articulated multi-part (pivot), mixed material (metal + plastic) at fine scale | Blades fuse into a single flat shape; pivot not detected as a joint (L6 articulation hint fails); handles and blades not separable as distinct L6 material regions |

**Coverage check** (must remain true for any v2 superset, see §5): every prompt maps to at
least one of {hard-surface, organic/fuzzy, thin structures, reflective/transparent,
text-on-object, articulated/multi-part, scale extreme, concave/hollow, known competitor
failure mode}. T02, T13, T20 are the "thin structures" anchor trio — if AURIGA cannot beat
raw TRELLIS.2 on these three, that is the headline finding regardless of aggregate score.

---

## 2. Twenty Image Cases (I01–I20)

Each case specifies what image(s) to collect, the source plan, and what it stresses. Sourcing
priority order: **(a) founder's own photos** (full rights, preferred — also doubles as capture
corpus seed data), **(b) CC0/CC-BY sources** (Wikimedia Commons, Polyhaven, NASA, Smithsonian
Open Access), **(c) only if (a)/(b) cannot cover the case, a licensed stock image with an
explicit license record**. No image is added to the corpus without a license entry in
`docs/eval/assets/MANIFEST.md` (path, source URL, license, capture/download date, SHA-256).

| ID | Image spec | Source plan | Stresses |
|---|---|---|---|
| I01 | Single photo, 3/4 view, of a matte ceramic mug on a plain table, no visible logo | Own photo (kitchen) | Single-view ambiguity for a simple convex shape — baseline sanity case |
| I02 | Single photo of a stapler from directly above (top-down only) | Own photo | Severe single-view ambiguity — generator must hallucinate the entire underside; Truth Meter confidence channel must flag the bottom as unmeasured |
| I03 | Single photo of a houseplant where one side of the pot is occluded by another object (e.g., a book leaning against it) | Own photo, staged | Occlusion — generator must complete the occluded region and the confidence channel must mark it low |
| I04 | Single photo of a plain white ceramic bowl, no pattern, even studio-ish lighting | Own photo, white tablecloth backdrop | Textureless surface — depth/SfM cues absent, pure shape-prior reliance |
| I05 | Single photo of a stainless-steel kettle under indoor lighting, strong specular highlight visible on the body | Own photo | Metallic/reflective single image — tests whether L4 separates the highlight from albedo or bakes it |
| I06 | Single photo of a smartphone lying flat, screen off (black glossy slab) | Own photo | Flat reflective surface + thin profile — easy to get "melted slab" geometry |
| I07 | Single photo of a knitted wool beanie on a flat surface | Own photo or Wikimedia (CC0 textile photos) | Soft/cloth material, fine surface texture (knit pattern) for L6 material classification |
| I08 | Single photo of a potted succulent with multiple distinct rosettes (multi-part organic) | Own photo | Multi-part organic separation — does the model treat each rosette as a separable part for L6? |
| I09 | Single photo, low-angle, of a wooden bar stool showing only 2 of its 3 legs (third occluded behind the visible legs) | Own photo, staged angle | Occlusion + thin structures combined — leg count must not be hallucinated wrong |
| I10 | Single photo of a glass tumbler, empty, on a dark background, photographed against backlight (silhouette-heavy) | Own photo | Transparency under adverse lighting — high failure-mode density |
| I11 | Single photo of an action figure (articulated joints visible) viewed from the front only | Own photo (toy) or Wikimedia Commons toy photography (CC0) | Articulated multi-part from single view — back-side joints must be flagged low-confidence, not silently invented |
| I12 | Single photo of a brick (or concrete block), plain gray, no markings | Own photo | Extreme textureless + simple geometry — tests whether scale estimator defaults sanely (bricks have a near-universal real size, good ground-truth check) |
| I13 | Single photo of a hardback book, cover visible, with title text on the spine and cover | Own photo | Text-on-object from a single view — spine vs. cover text must both attempt to render, back cover flagged unmeasured |
| I14 | Single photo of a small succulent in a tiny terracotta pot, photographed close-up so the pot fills most of the frame | Own photo | Scale extreme (small) — absolute-scale estimation with a real, measurable ground truth (founder measures the pot with calipers for L1 confidence-interval validation) |
| I15 | Single photo of an office chair (large, on casters) photographed from across a room | Own photo | Scale extreme (large) — same ground-truth validation approach as I14 at the opposite end |
| I16 | Two photos (front + back) of a framed picture frame, empty, glass front | Own photos, two angles | Multi-photo path with transparency — tests MASt3R/VGGT-class pose estimation (CLAUDE.md §4) on a thin, mostly-flat, reflective object |
| I17 | Three photos (front, left 3/4, right 3/4) of a teapot with a patterned (non-plain) glaze | Own photos | Multi-photo path, moderate texture — the "easy" multi-view case as a baseline against I02/I09 |
| I18 | Single photo of a wicker/rattan basket (dense repeating weave texture) empty | Own photo or Wikimedia (CC0 craft photography) | Repetitive fine geometry vs. texture — does the model attempt actual weave geometry or flatten to a texture map (relevant to L3 surface fidelity scoring) |
| I19 | Single photo of a pair of leather shoes, one partially occluding the other, both visible but overlapping | Own photo | Multi-object/occlusion ambiguity — does the generator correctly segment two objects vs. fusing them into one asset? |
| I20 | Single photo of a chrome bicycle bell or similar small mirror-finish object against a complex/cluttered background | Own photo | Reflective material + background-removal/segmentation stress — environment reflections in the chrome will literally contain the background, a classic confound |

**Sourcing note**: I01, I04–I09, I11–I20 are achievable with the founder's own phone + common
household objects within a single shooting session; ~3 (I07, I11, I18 if no suitable owned
object exists) may fall back to Wikimedia Commons CC0. Every fallback must be logged in
`docs/eval/assets/MANIFEST.md` before the corpus is marked frozen-and-populated.

---

## 3. Ten Capture Scenarios (C01–C10)

Orbit/handheld video scripts, phone-shootable. Each ~15–30 seconds, 1080p/4K @ 30fps minimum,
steady pace, full 360° unless noted. These double as the M2 capture-path test corpus
(CLAUDE.md §4 "Video →").

| ID | Object / scene | Environment | Lighting | Motion pattern | Stresses |
|---|---|---|---|---|---|
| C01 | Plain white ceramic mug (textureless) | Indoor, plain table | Soft diffuse daylight, no harsh shadow | Full 360° orbit at constant ~1m radius, camera height level with object | Textureless-surface SfM/pose failure (mug has almost no SIFT-style features) — primary stress test for the pose-free reconstruction path |
| C02 | Stainless kettle (shiny/specular) | Indoor kitchen counter | Mixed — one window + one overhead light (multiple specular highlights that move as camera moves) | Full 360° orbit, slightly varying radius (0.8–1.2m) | Specular highlight "swimming" across the surface during orbit — tests whether reconstruction separates true geometry from moving highlights (L4 relevance) |
| C03 | Potted plant with thin overlapping leaves (e.g., pothos or fern) | Indoor near window | Natural daylight, slight motion in leaves from airflow possible — note if any movement occurs | Full 360° orbit, two passes: one wide (full plant), one close (single leaf cluster) | Thin-structure capture (leaves) + potential dynamic content (leaf flutter) — relevant to L7 dynamics layer triggering correctly (should NOT trigger for near-static leaves, confidence channel should note minor inconsistency) |
| C04 | Outdoor wooden bench | Outdoor, garden/yard/park | Natural daylight, partly cloudy preferred (even lighting) | Full 360° orbit at ~1.5–2m radius, camera height ~1m | Outdoor scale/lighting variability, larger object, ground-plane contact reasoning (relevant to scene-seed/ground-plane work) |
| C05 | Bicycle (spoked wheels, chain, cables) | Outdoor or garage, against plain wall | Even outdoor shade or garage lighting | Full 360° orbit at ~2m radius; second close pass on one wheel only | Thin-structure capture stress at video scale — spokes/chain/cables are at/below typical SfM feature-survival resolution from a distance |
| C06 | Glass drinking glass (empty, transparent) | Indoor, on a table with a patterned background visible through the glass | Indoor ambient, avoid direct reflections of the camera/phone in the glass | Full 360° orbit, slow pace, ~1m radius | Transparent-object capture — refraction breaks standard SfM correspondence; tests pose-free (DUSt3R/MASt3R-class) robustness vs. COLMAP fallback |
| C07 | Small handheld object — wristwatch or jewelry item | Indoor, on a rotating turntable or lazy-susan if available, else handheld orbit | Soft box / diffuse desk lamp, close-up | Close orbit ~30cm radius, slow, may need 2 passes (top half, bottom half) | Scale extreme (small object), fine detail at macro distance, metric-scale ground truth (founder measures the object with calipers for L1 validation) |
| C08 | Hardback book lying open, pages visible with text and an illustration | Indoor, flat surface | Even overhead lighting, no glare on pages | Partial arc (not full 360° — book has a clear "front" only), ~120° sweep over the open book | Text-on-object at video scale (page text legibility in reconstructed texture); thin geometry (paper-thin pages, individually) |
| C09 | A person's hand holding a small object (e.g., an apple), object rotated by hand rather than camera moving | Indoor, plain background behind hand | Soft even lighting | Camera mostly static, object rotated through ~360° by hand, occasional brief hand occlusion of part of the object | Dynamic/occlusion content within an otherwise static scene — tests segmentation of the target object from a moving occluder (hand) and whether occluded frames degrade or are correctly down-weighted |
| C10 | A simple two-part toy or mechanism with one moving part articulated through its range during the video (e.g., a pair of pliers opened/closed, a hinged box lid raised/lowered) | Indoor, plain table | Even indoor lighting | Camera orbits slowly (~180°) while the part is articulated partway through the capture (held in 2–3 distinct poses, a few seconds each) | The genuine L7 dynamics-layer trigger case — deliberately dynamic content, tests 4DGS/deformable path activation vs. the static-scene default, and whether AURIGA correctly reports which frames correspond to which articulation state |

---

## 4. Scoring Protocol

### 4.1 Conditions and watermark stripping
Every case (T01–T20, I01–I20, C01–C10 — 50 total) is run through each of: **AURIGA**,
**raw TRELLIS.2** (unmodified checkpoint, no AURIGA finishing pipeline), **Meshy free tier**,
and **Tripo free tier** (secondary baseline, included opportunistically — not required for the
M3 gate but tracked). Outputs are converted to a common neutral render (turntable video or
fixed-camera multi-angle still set, rendered in a shared neutral-lit scene) with:

- All UI chrome, watermarks, and platform logos cropped or removed.
- Filenames/metadata stripped and replaced with random tokens before rater assignment.
- Background normalized to the same neutral gray for every output.

### 4.2 Blind pairwise A/B
Raters see two anonymized renders side by side (labels "A"/"B", randomized left/right per
trial, randomized which system is A vs B) and the **original prompt/image/video** that
generated both. For each pair, raters answer the per-axis rubric questions below plus an
overall preference. Every system pair that includes AURIGA is compared against each baseline
at least once per case; with 3–4 systems per case this is 3–6 pairs per case.

### 4.3 Per-axis rubrics (1–5 Likert per output, plus pairwise preference per axis)
For each output, independently of the pairwise step, a rater scores 1 (very poor) to 5
(excellent) on:

1. **Geometry fidelity** — does the silhouette/proportions/part structure match the
   prompt/source? (For T17, score 0 if the wrong canonical object is returned, recorded
   separately as a binary "correct object identity" flag.)
2. **Texture/appearance** — color, material look, text legibility where applicable, absence
   of baked-lighting artifacts.
3. **Thin-structure survival** — for cases tagged "thin structures" (T02, T03, T13, T14, T15,
   T18, T20, C03, C05, C08): are the thin elements present at all, and are they
   topologically/visually plausible (not vanished, not fused into solid blobs)? Score 1 if
   the element is entirely absent.
4. **Printability after solidification** — run the AURIGA L5 splat→SDF→watertight→.3mf path
   on *every* system's output (AURIGA's pipeline can solidify any splat/mesh input — this is
   the universal-printability claim from `docs/research/07-free-tier-consumer-strategy.md`
   §1). Score: did it produce a watertight, manifold, printable .3mf without manual repair?
   1 = failed entirely, 5 = clean pass with no warnings.
5. **Metric-scale accuracy** — *only* for cases with an independently measured ground truth
   (T09 ring ~18mm, I12 brick standard dims, I14/I15/C07 calipers-measured objects). Score =
   |predicted scale − measured scale| / measured scale, bucketed: <5% = 5, <10% = 4, <20% = 3,
   <40% = 2, ≥40% or no scale reported = 1. Cases without ground truth are marked N/A for this
   axis and excluded from its aggregate, not scored as 1.

### 4.4 Aggregation
- **Per-axis and overall pairwise results** feed a **Bradley-Terry model** (preferred over
  raw Elo for its principled handling of unequal comparison counts and confidence intervals)
  to produce a per-system strength score per axis and overall, with 95% CIs.
- **Per-axis Likert scores** are reported as medians + IQR per system per case-category
  (text/image/capture × failure-mode tag), not just a single grand mean — the category
  breakdown IS the headline result (e.g., "AURIGA wins thin-structure survival by X but ties
  on textureless capture").
- The **M3 gate** (per C2) is evaluated as: for each of the 50 cases, does AURIGA's
  Bradley-Terry strength exceed raw TRELLIS.2's on overall preference (required: all 50), and
  exceed Meshy-free's on overall preference (required: ≥26 of 50, a simple majority)?

### 4.5 Minimum rater counts
- **Minimum 5 independent raters** per pairwise comparison for the frozen M3-gate run (50
  cases × up to 6 pairs × 5 raters = up to 1,500 pairwise judgments — large but mostly
  automatable via a lightweight web rating tool; budget accordingly).
- Raters should include a mix: at least 2 "general consumer" raters (no 3D background) and
  at least 1 rater with 3D/CG production experience, reflecting both the consumer free-tier
  audience and the prosumer/studio audience.
- For interim development checkpoints (not the formal M3 gate), a reduced **minimum of 2
  raters** per pair is acceptable for directional signal, but the formal gate and any
  published marketing numbers require the full 5-rater run.

### 4.6 Publication format
Results are published as a static report (`docs/eval/results/v1-<date>.md` + an interactive
HTML page reusing the web viewer, CLAUDE.md §5) containing:
- The full corpus (this document, linked/embedded) so readers can inspect exactly what was
  tested — no cherry-picked subset.
- Side-by-side renders for every case and every system (the anonymized renders used for
  rating, re-labeled post-hoc).
- Bradley-Terry strength scores with CIs, per axis and overall, plus the category breakdown.
- Raw per-rater scores (anonymized rater IDs) in a downloadable CSV for independent
  verification — this transparency IS the Truth Meter brand applied to AURIGA's own
  marketing, per CLAUDE.md §8.
- An explicit pass/fail statement against the M3 gate criteria (§4.4), including cases where
  AURIGA loses, with no spin — "honesty over hype" (CLAUDE.md §1.3) applies to self-evaluation
  too.

---

## 5. Freeze and Versioning Policy

1. **This corpus (T01–T20, I01–I20, C01–C10; 50 cases total) is v1 and immutable upon
   adoption.** The exact prompt strings, image specs, and capture scripts above do not
   change, get reworded, "improved," or removed — ever — once the first formal scoring run
   using them has been published.
2. **No case is ever removed because AURIGA scores badly on it.** A bad score on, e.g., T13
   (bicycle/thin structures) is a finding to publish and work on, not a reason to drop T13.
   Removing a case after an unfavorable result would invalidate cross-release comparability
   and contradict the honesty brand (CLAUDE.md §1.3, §8).
3. **Additions are versioned supersets only**: a v2 corpus, if created, is v1 (all 50 cases,
   unchanged) **plus** new cases (e.g., T21+, I21+, C11+, or entirely new categories like
   multi-object scenes once scene-seed work lands per CLAUDE.md §8.7). v2 results report both
   "v1 subset" scores (directly comparable to all historical v1 runs) and "v2 full" scores
   separately — never blend them into one number that breaks comparability.
4. **The scoring protocol (§4) may be refined between versions** (e.g., adding a rater pool,
   adding a new baseline system, improving the rating UI) but any protocol change must be
   logged in this file's changelog (below) with the date and reason, and — where it affects
   comparability — historical results must be re-run or footnoted as "scored under protocol
   v1.x" vs "v1.y".
5. **Image/video asset files** referenced in §2/§3 are pinned by SHA-256 in
   `docs/eval/assets/MANIFEST.md` once collected. If a source image becomes unavailable
   (e.g., a Wikimedia file is taken down), the manifest entry is updated with a replacement
   that satisfies the *same spec* and the change is logged in the changelog — the spec (what
   the image must depict and stress) is the frozen artifact, not necessarily the exact
   original file, but replacements must be flagged and old vs. new results compared
   side-by-side, never silently substituted.

### Changelog
- **2026-06-13**: v1 corpus and protocol adopted (this document). 20 text prompts, 20 image
  case specs, 10 capture scenarios, Bradley-Terry pairwise protocol with 5-axis rubric, 5
  minimum raters for the formal M3 gate.
