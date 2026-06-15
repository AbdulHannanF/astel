"""Short version of the render-then-refit smoke test.

Skips on any machine that cannot actually run a gsplat kernel (no CUDA GPU, or
CUDA present but the MSVC toolchain not on PATH) via the
``requires_gsplat_runtime`` fixture — see ``conftest.py``.
"""

from __future__ import annotations


def test_psnr_climbs_with_short_refit(requires_gsplat_runtime: None) -> None:
    from astel_gpu.smoke_refit import run_smoke

    _final_params, metrics = run_smoke(
        iters=300,
        n_gaussians=2_000,
        n_views=6,
        image_size=128,
        seed=12345,
    )

    assert metrics["origin"] == "measured"
    assert metrics["final_psnr_db"] > metrics["init_psnr_db"]
    # A short 300-iter refit should still show meaningful convergence.
    assert metrics["final_psnr_db"] - metrics["init_psnr_db"] > 3.0
