"""Shared pytest fixtures for the GPU pipeline tests.

The ``*_cpu.py`` tests run anywhere. The two GPU tests (``test_smoke_refit``,
``test_synthetic_eval``) actually launch a gsplat kernel, so they need both a
CUDA GPU *and* the MSVC build toolchain on PATH — torch 2.11's cpp_extension
JIT loader runs ``where cl`` on every gsplat import. Use the
:func:`requires_gsplat_runtime` fixture to skip them cleanly anywhere that
cannot run them, so a plain ``uv run pytest`` is green both on a CPU box and on
the GPU box when not launched through ``run-python.cmd``.
"""

from __future__ import annotations

import shutil

import pytest
import torch


@pytest.fixture
def requires_gsplat_runtime() -> None:
    """Skip the test unless a gsplat kernel can actually be compiled + run."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    if shutil.which("cl") is None:
        pytest.skip(
            "MSVC compiler (cl.exe) not on PATH — run GPU tests via "
            "run-python.cmd so gsplat's JIT loader can compile"
        )


@pytest.fixture
def requires_flux_runtime() -> None:
    """Skip unless CUDA + diffusers + locally-cached FLUX.1-schnell weights exist.

    The full ~24 GB FLUX.1-schnell checkpoint must never be downloaded as a
    side effect of running the test suite, so this fixture only proceeds when
    the weights are ALREADY present in the local Hugging Face cache.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    try:
        from diffusers import FluxPipeline  # noqa: F401
    except ImportError:
        pytest.skip("diffusers not importable")

    from huggingface_hub import scan_cache_dir

    try:
        cached_repos = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        pytest.skip("could not scan Hugging Face cache")

    if "black-forest-labs/FLUX.1-schnell" not in cached_repos:
        pytest.skip("FLUX.1-schnell weights not cached locally")
