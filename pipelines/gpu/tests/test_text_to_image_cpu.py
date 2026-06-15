"""CPU tests for text_to_image helpers (no weights/CUDA/diffusers run needed)."""

from __future__ import annotations

import pytest

from astel_gpu.text_to_image import _T2I_PROMPT_SUFFIX, build_t2i_prompt


def test_build_t2i_prompt_wraps_user_prompt() -> None:
    result = build_t2i_prompt("a worn brass astrolabe on a wooden base")

    assert result.startswith("a worn brass astrolabe on a wooden base")
    assert _T2I_PROMPT_SUFFIX in result


def test_build_t2i_prompt_strips_whitespace() -> None:
    result = build_t2i_prompt("  a red mug  ")

    assert result.startswith("a red mug")
    assert "  " not in result.split(",")[0]


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\n\t"])
def test_build_t2i_prompt_rejects_empty(bad_prompt: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_t2i_prompt(bad_prompt)
