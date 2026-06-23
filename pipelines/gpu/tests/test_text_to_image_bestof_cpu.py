"""CPU tests for best-of-N image selection (fake generator; no diffusers/CUDA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astel_gpu.text_to_image import (
    ImageCandidate,
    active_best_of_n,
    generate_image_best_of_n,
)

_SIZE = 128


def _save_disc(path: Path, *, cx: float, cy: float, radius: float) -> None:
    from PIL import Image

    img = np.full((_SIZE, _SIZE, 3), 245, dtype=np.uint8)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    img[inside] = (180, 60, 40)
    Image.fromarray(img).save(path)


def _fake_generator(
    prompt: str,
    out_dir: Path,
    *,
    seeds: list[int],
    steps: int | None,
    size: int,
    device: str | None,
) -> list[ImageCandidate]:
    """Seed 0 -> clean centred disc; later seeds -> cropped / tiny (worse)."""
    recipes = {
        0: dict(cx=_SIZE / 2, cy=_SIZE / 2, radius=_SIZE * 0.28),  # good
        1: dict(cx=0.0, cy=_SIZE / 2, radius=_SIZE * 0.30),  # cropped (edge)
        2: dict(cx=_SIZE * 0.4, cy=_SIZE * 0.4, radius=_SIZE * 0.02),  # tiny
    }
    out: list[ImageCandidate] = []
    for i, seed in enumerate(seeds):
        path = out_dir / f"cand_{i}_seed{seed}.png"
        _save_disc(path, **recipes.get(seed, recipes[2]))
        out.append(ImageCandidate(seed=seed, path=path, metrics={"seed": seed}))
    return out


def test_best_of_n_picks_the_clean_centered_image(tmp_path: Path) -> None:
    out = tmp_path / "reference.png"
    result = generate_image_best_of_n(
        "a red mug",
        out,
        base_seed=0,
        n=3,
        candidate_generator=_fake_generator,
    )

    assert result.chosen_seed == 0  # the clean centred disc beat cropped + tiny
    assert result.chosen_score.accept
    assert out.is_file()
    assert len(result.candidates) == 3


def test_best_of_n_cleans_up_losing_candidates(tmp_path: Path) -> None:
    out = tmp_path / "reference.png"
    generate_image_best_of_n(
        "a red mug", out, base_seed=0, n=3, candidate_generator=_fake_generator
    )

    leftover = list(tmp_path.glob("cand_*"))
    assert leftover == []  # only the chosen image survives, at out_path


def test_best_of_n_sidecar_is_serialisable(tmp_path: Path) -> None:
    import json

    out = tmp_path / "reference.png"
    result = generate_image_best_of_n(
        "a red mug", out, base_seed=0, n=3, candidate_generator=_fake_generator
    )
    sidecar = result.to_sidecar()
    json.dumps(sidecar)  # must not raise

    assert sidecar["chosen_seed"] == 0
    assert sidecar["n"] == 3
    assert {c["seed"] for c in sidecar["candidates"]} == {0, 1, 2}


def test_best_of_n_seeds_are_offset_from_base(tmp_path: Path) -> None:
    seen: list[int] = []

    def recording_gen(
        prompt: str, out_dir: Path, *, seeds: list[int], **kw: object
    ) -> list[ImageCandidate]:
        seen.extend(seeds)
        return _fake_generator(prompt, out_dir, seeds=seeds, steps=None, size=128,
                               device=None)

    generate_image_best_of_n(
        "x", tmp_path / "r.png", base_seed=100, n=3,
        candidate_generator=recording_gen,
    )
    assert seen == [100, 101, 102]


def test_active_best_of_n_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTEL_T2I_BEST_OF", raising=False)
    assert active_best_of_n(default=4) == 4
    monkeypatch.setenv("ASTEL_T2I_BEST_OF", "1")
    assert active_best_of_n() == 1
    monkeypatch.setenv("ASTEL_T2I_BEST_OF", "garbage")
    assert active_best_of_n(default=3) == 3
