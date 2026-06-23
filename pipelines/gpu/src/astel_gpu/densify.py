"""Adaptive Density Control (ADC) — the engine that lets refinement ADD detail.

The current generative L3 (:mod:`astel_gpu.generative`) is a *distillation*: a
FIXED-count, FROZEN-position 2DGS surfel cloud re-fit to reproduce the TripoSplat
L2 generator's own renders. It can only ever lose information relative to L2. A
*real* refinement (Tier 1) needs three things the distillation lacks: external
multi-view supervision (see :mod:`astel_gpu.refine`), moving positions, and the
ability to grow/shrink the cloud where the data demands it. This module is the
third: Kerbl et al. (2023) adaptive density control, as used by every production
3DGS optimiser and by DreamGaussian's progressive densification.

The four operations, gated on the running view-space positional gradient:

* **clone** — a high-gradient *small* gaussian is under-reconstructing a region;
  duplicate it so two gaussians cover the detail.
* **split** — a high-gradient *large* gaussian is over-reconstructing; replace it
  with ``split_n`` smaller children jittered within its own extent.
* **prune** — drop near-transparent gaussians and pathologically oversized ones.
* **opacity reset** — periodically push opacities down so persistent floaters
  fall below the prune threshold (separate, see :func:`reset_opacity`).

DESIGN: every selection mask and every tensor-surgery step here is a pure
``torch`` function, CPU-runnable and unit-tested without gsplat. The stateful
:class:`DensityController` only accumulates per-gaussian gradient statistics
between densifications. Because densification changes ``N``, the caller rebuilds
the Adam optimiser around the new tensors after each :meth:`DensityController.step`
(a fresh optimiser per ~100-iter densification is simpler and safer than the
surgical optimiser-state editing of the reference CUDA implementation, and the lost
momentum is immaterial over a short refine).

HONESTY: the gradient signal here is the L2 norm of the 3D ``means`` gradient, a
documented approximation of the reference implementation's screen-space gradient.
It is a sound densification driver; it is not bit-for-bit the INRIA criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .gaussians import GaussianParams


@dataclass(frozen=True)
class DensifyConfig:
    """Thresholds for adaptive density control. Defaults target a unit-radius cloud.

    The generative refiner normalises every asset to unit radius
    (:func:`astel_gpu.generative.normalize_params`), so ``scene_extent`` defaults
    to ``1.0`` and the scale thresholds are expressed as fractions of it.
    """

    #: Mean positional-gradient magnitude above which a gaussian is densified.
    grad_threshold: float = 2.0e-4
    #: Scale boundary (fraction of ``scene_extent``) separating clone (small) from
    #: split (large) among the high-gradient gaussians.
    percent_dense: float = 0.01
    #: Children produced per split.
    split_n: int = 2
    #: Each split child's scale = parent scale / this factor.
    split_scale_factor: float = 1.6
    #: Prune gaussians with activated opacity below this.
    min_opacity: float = 0.05
    #: Prune gaussians whose largest extent exceeds this fraction of ``scene_extent``
    #: (runaway blobs).
    max_scale_factor: float = 0.5
    #: Hard cap on cloud size — densification is skipped once reached (VRAM guard).
    max_gaussians: int = 1_500_000
    #: Spatial extent the fractional thresholds are relative to.
    scene_extent: float = 1.0


def _max_extent(scales: torch.Tensor) -> torch.Tensor:
    """Largest per-gaussian extent, ``(N,)``."""
    return scales.abs().max(dim=1).values


def clone_mask(
    grads: torch.Tensor, scales: torch.Tensor, cfg: DensifyConfig
) -> torch.Tensor:
    """High-gradient *small* gaussians (under-reconstruction) — to duplicate."""
    big = cfg.percent_dense * cfg.scene_extent
    return (grads >= cfg.grad_threshold) & (_max_extent(scales) <= big)


def split_mask(
    grads: torch.Tensor, scales: torch.Tensor, cfg: DensifyConfig
) -> torch.Tensor:
    """High-gradient *large* gaussians (over-reconstruction) — to split."""
    big = cfg.percent_dense * cfg.scene_extent
    return (grads >= cfg.grad_threshold) & (_max_extent(scales) > big)


def prune_mask(
    opacities: torch.Tensor, scales: torch.Tensor, cfg: DensifyConfig
) -> torch.Tensor:
    """``True`` where a gaussian should be PRUNED (too faint or too large)."""
    too_faint = opacities < cfg.min_opacity
    too_big = _max_extent(scales) > cfg.max_scale_factor * cfg.scene_extent
    return too_faint | too_big


def _subset(params: GaussianParams, mask: torch.Tensor) -> GaussianParams:
    return GaussianParams(
        means=params.means[mask],
        scales=params.scales[mask],
        quats=params.quats[mask],
        opacities=params.opacities[mask],
        colors=params.colors[mask],
    )


def _build_split_children(
    params: GaussianParams,
    sel: torch.Tensor,
    cfg: DensifyConfig,
    generator: torch.Generator | None,
) -> GaussianParams:
    """``split_n`` jittered, shrunk children for each selected gaussian.

    Children are offset by Gaussian noise scaled to the parent's per-axis extent
    (an axis-aligned approximation of sampling within the rotated ellipsoid) and
    given ``scale / split_scale_factor``; opacity/colour/orientation are inherited.
    """
    idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
    k = int(idx.numel())
    n = cfg.split_n
    base_scales = params.scales[idx].abs()  # (k, 3)
    noise = torch.randn(
        k, n, 3, generator=generator, device=params.means.device,
        dtype=params.means.dtype,
    )
    child_means = (
        params.means[idx][:, None, :] + noise * base_scales[:, None, :]
    ).reshape(-1, 3)
    child_scales = (
        (base_scales / cfg.split_scale_factor)[:, None, :].expand(-1, n, -1)
    ).reshape(-1, 3)
    child_quats = params.quats[idx][:, None, :].expand(-1, n, -1).reshape(-1, 4)
    child_op = params.opacities[idx][:, None].expand(-1, n).reshape(-1)
    child_col = params.colors[idx][:, None, :].expand(-1, n, -1).reshape(-1, 3)
    return GaussianParams(
        means=child_means,
        scales=child_scales,
        quats=child_quats,
        opacities=child_op,
        colors=child_col,
    )


def _cat(parts: list[GaussianParams]) -> GaussianParams:
    parts = [p for p in parts if p.count > 0]
    if not parts:
        return parts_empty_like(parts)
    return GaussianParams(
        means=torch.cat([p.means for p in parts], dim=0),
        scales=torch.cat([p.scales for p in parts], dim=0),
        quats=torch.cat([p.quats for p in parts], dim=0),
        opacities=torch.cat([p.opacities for p in parts], dim=0),
        colors=torch.cat([p.colors for p in parts], dim=0),
    )


def parts_empty_like(_parts: list[GaussianParams]) -> GaussianParams:
    """A zero-row :class:`GaussianParams` (used only on the empty-cat edge case)."""
    return GaussianParams(
        means=torch.zeros(0, 3),
        scales=torch.zeros(0, 3),
        quats=torch.zeros(0, 4),
        opacities=torch.zeros(0),
        colors=torch.zeros(0, 3),
    )


def densify_and_prune(
    params: GaussianParams,
    grads: torch.Tensor,
    cfg: DensifyConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[GaussianParams, dict[str, Any]]:
    """One adaptive-density-control step. Pure; CPU-runnable.

    Clones high-gradient small gaussians, splits high-gradient large ones (parent
    replaced by ``split_n`` children), then prunes faint/oversized gaussians from
    the combined cloud. Returns ``(new_params, stats)``. When the cloud already
    exceeds ``max_gaussians`` only the prune pass runs (no growth), so the VRAM cap
    is never breached.
    """
    cfg = cfg or DensifyConfig()
    n_in = params.count
    clone_sel = clone_mask(grads, params.scales, cfg)
    split_sel = split_mask(grads, params.scales, cfg)

    at_cap = n_in >= cfg.max_gaussians
    if at_cap:
        clone_sel = torch.zeros_like(clone_sel)
        split_sel = torch.zeros_like(split_sel)

    survivors = _subset(params, ~split_sel)  # split parents are replaced
    clones = _subset(params, clone_sel)
    children = _build_split_children(params, split_sel, cfg, generator)
    combined = _cat([survivors, clones, children])

    keep = ~prune_mask(combined.opacities, combined.scales, cfg)
    # Never let a prune empty the cloud (a misconfiguration, not a result).
    if int(keep.sum()) == 0:
        keep = torch.ones_like(keep)
    final = _subset(combined, keep)

    stats: dict[str, Any] = {
        "input": n_in,
        "cloned": int(clone_sel.sum()),
        "split": int(split_sel.sum()),
        "split_children": int(children.count),
        "pruned": int(combined.count - final.count),
        "output": final.count,
        "at_cap": at_cap,
    }
    return final, stats


def reset_opacity(
    params: GaussianParams, value: float = 0.05
) -> GaussianParams:
    """Clamp every opacity DOWN to at most ``value`` (the periodic floater cull).

    Pushing opacities low forces persistent floaters below ``min_opacity`` so the
    next prune removes them, while genuine surface gaussians quickly re-learn a
    high opacity. Only opacity changes.
    """
    return GaussianParams(
        means=params.means,
        scales=params.scales,
        quats=params.quats,
        opacities=params.opacities.clamp(max=value),
        colors=params.colors,
    )


class DensityController:
    """Accumulates positional-gradient stats and applies ADC on demand.

    Usage inside an optimisation loop::

        ctrl = DensityController(params.count, cfg, device)
        for it in range(iters):
            ... forward/backward producing means.grad ...
            ctrl.record(means.grad)
            if ctrl.should_densify(it):
                params, stats = ctrl.step(params)   # N changes
                ... rebuild trainable params + optimiser around new params ...
        # opacity reset is driven separately by should_reset_opacity(it)

    The controller is state only — it never touches the optimiser, keeping the
    GPU-runtime concerns in the caller and this logic CPU-testable.
    """

    def __init__(
        self,
        n: int,
        cfg: DensifyConfig | None = None,
        device: torch.device | None = None,
        *,
        warmup: int = 100,
        interval: int = 100,
        stop: int = 10_000,
        opacity_reset_interval: int = 300,
        generator: torch.Generator | None = None,
    ) -> None:
        self.cfg = cfg or DensifyConfig()
        self.device = device or torch.device("cpu")
        self.warmup = warmup
        self.interval = interval
        self.stop = stop
        self.opacity_reset_interval = opacity_reset_interval
        self.generator = generator
        self._reset_stats(n)

    def _reset_stats(self, n: int) -> None:
        self.grad_accum = torch.zeros(n, device=self.device)
        self.denom = torch.zeros(n, device=self.device)

    def record(self, means_grad: torch.Tensor | None) -> None:
        """Accumulate the per-gaussian gradient magnitude for this iteration."""
        if means_grad is None:
            return
        g = means_grad.detach()
        if g.shape[0] != self.grad_accum.shape[0]:  # post-densify mismatch guard
            self._reset_stats(g.shape[0])
        self.grad_accum += g.norm(dim=-1)
        self.denom += 1.0

    def avg_grad(self) -> torch.Tensor:
        """Mean per-gaussian gradient magnitude since the last densification."""
        return self.grad_accum / self.denom.clamp_min(1.0)

    def should_densify(self, iteration: int) -> bool:
        return (
            self.warmup <= iteration < self.stop
            and iteration % self.interval == 0
        )

    def should_reset_opacity(self, iteration: int) -> bool:
        return (
            self.warmup <= iteration < self.stop
            and iteration > 0
            and iteration % self.opacity_reset_interval == 0
        )

    def step(self, params: GaussianParams) -> tuple[GaussianParams, dict[str, Any]]:
        """Run one densify/prune step and reset the gradient accumulators."""
        new_params, stats = densify_and_prune(
            params, self.avg_grad(), self.cfg, generator=self.generator
        )
        self._reset_stats(new_params.count)
        return new_params, stats


def config_summary(cfg: DensifyConfig) -> dict[str, Any]:
    """JSON-serialisable view of a :class:`DensifyConfig` (for metrics sidecars)."""
    return asdict(cfg)
