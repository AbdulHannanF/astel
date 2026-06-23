"""Reference-image critic — score a generated image for TripoSplat suitability.

The text->3D path hinges on ONE reference image (``text_to_image`` -> TripoSplat).
TripoSplat is a single-image feed-forward generator: it removes the background,
crops to the subject's bounding box, and reconstructs from that one view. So the
quality *and the identity* of the whole 3D asset are decided by whether that image
is a clean, single, centred, fully-in-frame product shot. When the text-to-image
model instead paints a cropped object, an off-centre 3/4 view, two objects, or a
busy background, the 3D collapses — which is exactly the "same prompt, sometimes
right, sometimes wrong" failure: each task draws a fresh seed (``produce.py``), so
each task draws a different image, and the bad draws ship.

This module scores an image against TripoSplat's known failure modes so the
caller can pick the best of N draws (see
:func:`astel_gpu.text_to_image.generate_image_best_of_n`) and reject degenerate
ones, *before* paying for the expensive L2->L3 stage. It is pure ``numpy`` (the
file-loading wrapper lazy-imports PIL), fully CPU-testable, and deterministic.

Sub-scores (each in ``[0, 1]``, higher = better), and the failure each guards:

* **plainness** — border ring colour uniformity. Busy/scene backgrounds confuse
  background removal and bleed into the splats.
* **coverage** — foreground area fraction, scored as a trapezoid: too small =
  detail-starved tiny object; too large = object filling/exceeding the frame.
* **centering** — foreground centroid distance from the image centre.
* **margin** — bounding-box distance to the nearest frame edge. Zero margin means
  the object is *cropped*, the single worst input for single-view reconstruction.
* **single** — largest-connected-component share of the foreground. Penalises
  two-object / fragmented compositions TripoSplat cannot reconstruct as one body.

HONESTY: this is a *suitability* heuristic for the generator, not a measure of the
final asset's accuracy. It never inspects real captured data — it only ranks
generated candidate images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Border ring colour std (0..255, mean over channels) at/above which the
#: background is considered fully "busy" (plainness -> 0). A plain studio sweep
#: sits well under this; a photographed scene exceeds it.
_PLAIN_STD_REF = 42.0
#: Foreground/background separation: a pixel is foreground if its distance from
#: the estimated background colour exceeds the Otsu threshold of the distance map.
_RGB_MAX_DIST = float(np.sqrt(3.0) * 255.0)


@dataclass(frozen=True)
class ImageQAConfig:
    """Thresholds + weights for :func:`score_image`. Conservative defaults."""

    #: Outer-ring width (fraction of the shorter side) used to model the
    #: background colour and measure its uniformity.
    border_frac: float = 0.06
    #: Coverage trapezoid (foreground-area fraction): score ramps 0->1 across
    #: ``[reject_low, ideal_low]``, holds 1 across the ideal band, ramps 1->0
    #: across ``[ideal_high, reject_high]``.
    coverage_reject_low: float = 0.06
    coverage_ideal_low: float = 0.16
    coverage_ideal_high: float = 0.58
    coverage_reject_high: float = 0.82
    #: Bounding-box margin (fraction of the frame) that earns a full margin score;
    #: a box touching the edge (margin 0) scores 0 — the object is cropped.
    margin_full_frac: float = 0.03
    #: Overall acceptance threshold.
    accept_threshold: float = 0.55
    #: Sub-score weights (need not sum to 1; the overall is weight-normalised).
    w_plain: float = 0.28
    w_coverage: float = 0.24
    w_centering: float = 0.14
    w_margin: float = 0.22
    w_single: float = 0.12


@dataclass(frozen=True)
class ImageScore:
    """Per-image scorecard from :func:`score_image`."""

    overall: float
    accept: bool
    plainness: float
    coverage: float
    centering: float
    margin: float
    single: float
    coverage_fraction: float
    foreground_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_rgb(arr: NDArray[Any]) -> NDArray[np.float64]:
    """Coerce an image array to ``(H, W, 3)`` float64 in ``[0, 255]``."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3:
        raise ValueError(f"score_image: expected H x W [x C] array, got {a.shape}")
    # Take RGB, or broadcast a single explicit channel up to 3.
    a = a[:, :, :3] if a.shape[2] >= 3 else np.repeat(a[:, :, :1], 3, axis=2)
    return a.astype(np.float64)


def _alpha_mask(arr: NDArray[Any]) -> NDArray[np.bool_] | None:
    """Foreground mask from a real alpha channel, or ``None`` if absent/trivial."""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 4:
        return None
    alpha = a[:, :, 3].astype(np.float64)
    if float(alpha.min()) >= 255.0:  # fully opaque -> alpha carries no matte
        return None
    return alpha > 127.5


def _otsu_threshold(values: NDArray[np.float64], bins: int = 64) -> float:
    """Otsu's threshold over a 1-D array (pure numpy, deterministic)."""
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return hi
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return hi
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * centers)
    total_mean = cum_mean[-1]
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return hi
    mean_bg = np.where(weight_bg > 0, cum_mean / np.maximum(weight_bg, 1e-9), 0.0)
    mean_fg = np.where(
        weight_fg > 0, (total_mean - cum_mean) / np.maximum(weight_fg, 1e-9), 0.0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def _border_ring_mask(h: int, w: int, border_frac: float) -> NDArray[np.bool_]:
    """Boolean mask of the outer-ring pixels used to model the background."""
    bw = max(1, int(round(min(h, w) * border_frac)))
    ring = np.zeros((h, w), dtype=bool)
    ring[:bw, :] = True
    ring[-bw:, :] = True
    ring[:, :bw] = True
    ring[:, -bw:] = True
    return ring


def _foreground_mask(
    rgb: NDArray[np.float64], ring: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], float]:
    """Estimate foreground from background-colour distance. Returns (mask, ring_std).

    The background colour is the median of the border ring; the per-pixel
    distance to it is Otsu-thresholded. ``ring_std`` (mean per-channel std of the
    ring) feeds the plainness score.
    """
    ring_pixels = rgb[ring]
    bg_color = np.median(ring_pixels, axis=0)
    ring_std = float(np.mean(np.std(ring_pixels, axis=0)))
    dist = np.sqrt(((rgb - bg_color[None, None, :]) ** 2).sum(axis=2))
    thr = _otsu_threshold(dist.ravel())
    # A near-flat distance map (uniform image) has a tiny Otsu threshold; require
    # a minimum absolute separation so a blank image yields an EMPTY foreground
    # rather than half the frame.
    min_sep = 0.04 * _RGB_MAX_DIST
    mask = dist > max(thr, min_sep)
    return mask, ring_std


def _largest_component_fraction(mask: NDArray[np.bool_]) -> float:
    """Share of foreground pixels in its largest 4-connected component (``[0,1]``)."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    try:
        from scipy.ndimage import label  # noqa: PLC0415 (optional heavy dep)
    except ImportError:
        return 1.0
    labels, n = label(mask)
    if n <= 1:
        return 1.0
    counts = np.bincount(labels.ravel())[1:]  # drop background label 0
    return float(counts.max()) / float(total)


def _coverage_score(frac: float, cfg: ImageQAConfig) -> float:
    """Trapezoidal coverage score (see :class:`ImageQAConfig`)."""
    if frac <= cfg.coverage_reject_low or frac >= cfg.coverage_reject_high:
        return 0.0
    if frac < cfg.coverage_ideal_low:
        return (frac - cfg.coverage_reject_low) / (
            cfg.coverage_ideal_low - cfg.coverage_reject_low
        )
    if frac > cfg.coverage_ideal_high:
        return (cfg.coverage_reject_high - frac) / (
            cfg.coverage_reject_high - cfg.coverage_ideal_high
        )
    return 1.0


def score_image(
    arr: NDArray[Any], config: ImageQAConfig | None = None
) -> ImageScore:
    """Score an ``(H, W, 3|4)`` image array for single-view-generator suitability.

    Pure and deterministic. Uses a real alpha matte when present, otherwise a
    border-ring background model. Returns an :class:`ImageScore`; ``overall`` is
    the weight-normalised mean of the five sub-scores and ``accept`` compares it to
    ``config.accept_threshold``.
    """
    cfg = config or ImageQAConfig()
    rgb = _as_rgb(arr)
    h, w = rgb.shape[:2]
    ring = _border_ring_mask(h, w, cfg.border_frac)

    alpha = _alpha_mask(arr)
    if alpha is not None:
        mask = alpha
        ring_std = float(np.mean(np.std(rgb[ring], axis=0)))
    else:
        mask, ring_std = _foreground_mask(rgb, ring)

    fg_pixels = int(mask.sum())
    total_pixels = h * w
    coverage_fraction = fg_pixels / total_pixels if total_pixels else 0.0

    plainness = float(np.clip(1.0 - ring_std / _PLAIN_STD_REF, 0.0, 1.0))
    coverage = _coverage_score(coverage_fraction, cfg)

    if fg_pixels == 0:
        centering = 0.0
        margin = 0.0
        single = 0.0
    else:
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        # Centroid distance from centre, normalised by the half-diagonal.
        half_diag = 0.5 * float(np.hypot(h, w))
        dist_center = float(np.hypot(cy - h / 2.0, cx - w / 2.0))
        centering = float(np.clip(1.0 - dist_center / half_diag, 0.0, 1.0))

        # Bounding-box margin to the nearest edge (fraction of frame).
        top, bottom = int(ys.min()), int(h - 1 - ys.max())
        left, right = int(xs.min()), int(w - 1 - xs.max())
        min_margin_frac = min(top / h, bottom / h, left / w, right / w)
        margin = float(np.clip(min_margin_frac / cfg.margin_full_frac, 0.0, 1.0))

        single = _largest_component_fraction(mask)

    weights = np.array(
        [cfg.w_plain, cfg.w_coverage, cfg.w_centering, cfg.w_margin, cfg.w_single]
    )
    scores = np.array([plainness, coverage, centering, margin, single])
    overall = float((weights * scores).sum() / weights.sum())

    return ImageScore(
        overall=overall,
        accept=overall >= cfg.accept_threshold,
        plainness=plainness,
        coverage=coverage,
        centering=centering,
        margin=margin,
        single=single,
        coverage_fraction=coverage_fraction,
        foreground_pixels=fg_pixels,
    )


def score_image_file(
    path: str | Path, config: ImageQAConfig | None = None
) -> ImageScore:
    """Load an image file (lazy PIL) and score it with :func:`score_image`."""
    from PIL import Image  # noqa: PLC0415 (lazy: keep numpy-only core importable)

    with Image.open(path) as img:
        mode = "RGBA" if "A" in img.getbands() else "RGB"
        arr = np.asarray(img.convert(mode))
    return score_image(arr, config)
