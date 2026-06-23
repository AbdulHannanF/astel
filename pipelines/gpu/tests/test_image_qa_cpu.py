"""CPU tests for the reference-image critic (pure numpy; no diffusers/CUDA)."""

from __future__ import annotations

import numpy as np
import pytest

from astel_gpu.image_qa import ImageQAConfig, score_image

_SIZE = 256


def _blank(color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.uint8)
    img[:] = color
    return img


def _disc(
    *,
    cx: float,
    cy: float,
    radius: float,
    bg: tuple[int, int, int] = (245, 245, 245),
    fg: tuple[int, int, int] = (180, 60, 40),
    size: int = _SIZE,
) -> np.ndarray:
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = bg
    yy, xx = np.mgrid[0:size, 0:size]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    img[inside] = fg
    return img


def test_centered_object_on_plain_bg_scores_high_and_accepts() -> None:
    img = _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.28)
    score = score_image(img)

    assert score.accept
    assert score.overall > 0.7
    assert score.centering > 0.9
    assert score.plainness > 0.9
    assert score.single > 0.95


def test_blank_image_is_rejected() -> None:
    score = score_image(_blank())

    assert not score.accept
    assert score.foreground_pixels == 0
    assert score.coverage == pytest.approx(0.0)


def test_cropped_object_touching_edge_loses_margin() -> None:
    # Disc centred on the left edge: bbox touches x=0 -> cropped.
    cropped = _disc(cx=0.0, cy=_SIZE / 2, radius=_SIZE * 0.3)
    centered = _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.3)

    assert score_image(cropped).margin < score_image(centered).margin
    assert score_image(cropped).margin == pytest.approx(0.0)


def test_tiny_object_is_detail_starved_low_coverage() -> None:
    tiny = _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.03)
    score = score_image(tiny)

    assert score.coverage < 0.5
    assert score.overall < score_image(
        _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.28)
    ).overall


def test_two_objects_penalise_single_score() -> None:
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.uint8)
    img[:] = (245, 245, 245)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    r = _SIZE * 0.12
    left = (xx - _SIZE * 0.3) ** 2 + (yy - _SIZE / 2) ** 2 <= r**2
    right = (xx - _SIZE * 0.7) ** 2 + (yy - _SIZE / 2) ** 2 <= r**2
    img[left | right] = (180, 60, 40)

    two = score_image(img)
    one = score_image(_disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=r))
    assert two.single < 0.65
    assert two.single < one.single


def test_busy_background_lowers_plainness() -> None:
    rng = np.random.default_rng(0)
    busy = rng.integers(0, 256, size=(_SIZE, _SIZE, 3), dtype=np.uint8)
    # Stamp a centred object so it is not purely noise.
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    inside = (xx - _SIZE / 2) ** 2 + (yy - _SIZE / 2) ** 2 <= (_SIZE * 0.25) ** 2
    busy[inside] = (180, 60, 40)

    plain = _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.25)
    assert score_image(busy).plainness < score_image(plain).plainness


def test_alpha_matte_is_used_when_present() -> None:
    rgba = np.zeros((_SIZE, _SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = (120, 120, 120)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    inside = (xx - _SIZE / 2) ** 2 + (yy - _SIZE / 2) ** 2 <= (_SIZE * 0.26) ** 2
    rgba[inside, 3] = 255  # opaque subject, transparent elsewhere
    score = score_image(rgba)

    assert score.foreground_pixels > 0
    assert score.centering > 0.9
    assert score.single > 0.95


def test_accept_threshold_is_configurable() -> None:
    # A small, off-centre object scores well under 1.0; a strict threshold should
    # reject it while the default accepts a clean centred shot.
    mediocre = _disc(cx=_SIZE * 0.35, cy=_SIZE * 0.4, radius=_SIZE * 0.1)
    good = _disc(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.28)
    strict = ImageQAConfig(accept_threshold=0.95)

    assert not score_image(mediocre, strict).accept
    assert score_image(good).accept
