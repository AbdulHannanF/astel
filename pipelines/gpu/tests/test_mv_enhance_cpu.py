"""CPU tests for the multi-view enhancer (pure torch + injected fake enhancer)."""

from __future__ import annotations

import torch

from astel_gpu.mv_enhance import (
    apply_black_background,
    build_enhance_prompt,
    detail_transfer,
    enhance_views,
    foreground_mask,
)


def _orbit(v: int = 3, s: int = 8) -> torch.Tensor:
    """A black-background batch with a bright central square 'object'."""
    img = torch.zeros(v, s, s, 3)
    img[:, 2:6, 2:6, :] = 0.8
    return img


def test_build_enhance_prompt() -> None:
    assert build_enhance_prompt("a red mug").startswith("a red mug,")
    assert "sharp focus" in build_enhance_prompt("")  # generic fallback


def test_foreground_mask_picks_object_pixels() -> None:
    img = _orbit()
    mask = foreground_mask(img)
    assert mask.shape == (3, 8, 8, 1)
    assert float(mask[:, 2:6, 2:6].min()) == 1.0  # object kept
    assert float(mask[:, 0, 0].max()) == 0.0  # corner background dropped


def test_apply_black_background_zeros_non_foreground() -> None:
    img = torch.ones(1, 4, 4, 3)
    mask = torch.zeros(1, 4, 4, 1)
    mask[:, 1:3, 1:3] = 1.0
    out = apply_black_background(img, mask)
    assert float(out[:, 0, 0].sum()) == 0.0
    assert float(out[:, 1, 1].sum()) == 3.0


def test_enhance_views_strength_zero_is_masked_base() -> None:
    base = _orbit()

    def _boom(*a: object, **k: object) -> torch.Tensor:
        raise AssertionError("enhancer must NOT run at strength 0")

    out, m = enhance_views(base, prompt="x", strength=0.0, enhancer=_boom)
    # strength 0 short-circuits: masked base, enhancer untouched.
    assert torch.allclose(out, apply_black_background(base, foreground_mask(base)))
    assert m["strength"] == 0.0


def test_enhance_views_replace_mode_applies_enhancer_and_masks() -> None:
    base = _orbit()

    def fake(images: torch.Tensor, **kw: object) -> torch.Tensor:
        # Pretend to "enhance" by brightening everything, including background.
        return (images + 0.2).clamp(0, 1)

    out, m = enhance_views(
        base, prompt="a mug", strength=0.3, combine="replace", enhancer=fake
    )
    # Background was brightened by the fake enhancer but the mask restores black.
    assert float(out[:, 0, 0].max()) == 0.0
    # Foreground retains the enhancement.
    assert float(out[:, 3, 3].max()) > 0.8
    assert m["n_views"] == 3
    assert m["combine"] == "replace"
    assert "mug" in m["prompt_used"]


def test_enhance_views_keeps_shape_and_range() -> None:
    base = _orbit(v=4, s=16)

    def fake(images: torch.Tensor, **kw: object) -> torch.Tensor:
        return images * 1.5  # would exceed 1.0 without the clamp

    out, _ = enhance_views(base, prompt="x", strength=0.5, enhancer=fake)
    assert out.shape == base.shape
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


def test_detail_transfer_rejects_low_frequency_shift() -> None:
    # The core robustness property: detail_transfer is INVARIANT to a pure
    # exposure/colour (low-frequency) shift in the enhancement, so per-view exposure
    # inconsistency cannot darken/collapse the refine. (It is deliberately NOT
    # invariant to the enhancement's high-frequency structure — transferring that is
    # the whole point — so the old `out == base` assert was wrong: `base` carries its
    # own high frequencies which detail_transfer reinforces.)
    base = _orbit(v=2, s=32)
    enh = base.clone()
    enh[:, 10:14, 18:22, :] = 0.5  # real high-frequency detail to transfer (interior)
    shifted = enh + 0.3  # SAME structure, +0.3 exposure = a pure low-frequency shift
    out_a = detail_transfer(base, enh, gain=1.0, blur_kernel=9)
    out_b = detail_transfer(base, shifted, gain=1.0, blur_kernel=9)
    # Interior (>= kernel//2 from every border) — the box-blur is shift-exact there,
    # so the +0.3 exposure shift produces an identical target.
    inner = (slice(None), slice(8, 24), slice(8, 24), slice(None))
    assert torch.allclose(out_a[inner], out_b[inner], atol=1e-5)


def test_detail_transfer_adds_high_frequency_structure() -> None:
    base = torch.full((1, 16, 16, 3), 0.4)
    enhanced = base.clone()
    enhanced[:, 8, 8, :] = 1.0  # a sharp high-frequency spike
    out = detail_transfer(base, enhanced, gain=1.0, blur_kernel=5)
    # The spike location gains intensity over the flat base.
    assert float(out[:, 8, 8].mean()) > float(base[:, 8, 8].mean())


def test_enhance_views_detail_mode_default_rejects_exposure() -> None:
    base = _orbit(v=2, s=32)

    def darken(images: torch.Tensor, **kw: object) -> torch.Tensor:
        return (images * 0.4).clamp(0, 1)  # the real failure: enhancer darkens views

    out, m = enhance_views(base, prompt="x", strength=0.3, enhancer=darken)
    assert m["combine"] == "detail"
    # Detail mode keeps the base exposure despite the darkening enhancer. The mask is
    # (V, H, W, 1); broadcast it to the 3 colour channels before boolean-indexing the
    # (V, H, W, 3) targets (a bare (...,1) mask raises IndexError on a (...,3) tensor).
    fg = foreground_mask(base).expand_as(out) > 0
    assert float(out[fg].mean()) > 0.5 * float(base[fg].mean())
