"""Splat-budget constants and capping helpers.

Product tiers (CLAUDE.md §3 L3)
--------------------------------
``TIER_BUDGETS`` maps the three named quality tiers to their splat-count
ceiling.  These numbers match the per-tier targets stated in the spec; they
are first-pass defaults and can be overridden per-generation.

Per-platform budgets
--------------------
``PLATFORM_BUDGETS`` maps a deployment target to a sensible default splat
count.  Values are deliberately conservative for bandwidth/memory-limited
targets (mobile, web) and generous for high-end targets (console, cinematic).
Override via API parameters; do not hardcode these in the pipeline.

Platform  |  Budget   |  Rationale
----------|-----------|---------------------------------------------
mobile    |  100 000  |  WebGL/Metal on phones; memory & GPU limits
web       |  500 000  |  Desktop browser; ~2-4 GB VRAM headroom
console   |  1 500 000|  PS5/XSX-class; 10-16 GB GDDR; 60 fps target
cinematic |  5 000 000|  Full-budget render farm; offline render path
"""

from __future__ import annotations

#: Named quality-tier splat budgets (CLAUDE.md §3 L3).
TIER_BUDGETS: dict[str, int] = {
    "lowpoly": 100_000,
    "standard": 1_000_000,
    "cinematic": 5_000_000,
}

#: Per-platform splat-count targets.  Sensible first-pass defaults; override
#: via API.  See module docstring for rationale.
PLATFORM_BUDGETS: dict[str, int] = {
    "mobile": 100_000,
    "web": 500_000,
    "console": 1_500_000,
    "cinematic": 5_000_000,
}


def auto_target(n_splats: int, platform: str) -> int:
    """Return the effective splat count for ``platform``, capped at ``n_splats``.

    Parameters
    ----------
    n_splats:
        Total number of Gaussians in the cloud.
    platform:
        One of the keys in :data:`PLATFORM_BUDGETS`.

    Returns
    -------
    int
        ``min(n_splats, PLATFORM_BUDGETS[platform])``.

    Raises
    ------
    ValueError
        On an unknown platform key, with a helpful message listing valid keys.
    """
    if platform not in PLATFORM_BUDGETS:
        valid = ", ".join(sorted(PLATFORM_BUDGETS))
        msg = f"Unknown platform {platform!r}.  Valid platforms: {valid}"
        raise ValueError(msg)
    return min(n_splats, PLATFORM_BUDGETS[platform])


def tier_target(n_splats: int, tier: str) -> int:
    """Return the effective splat count for quality ``tier``, capped at ``n_splats``.

    Parameters
    ----------
    n_splats:
        Total number of Gaussians in the cloud.
    tier:
        One of the keys in :data:`TIER_BUDGETS`.

    Returns
    -------
    int
        ``min(n_splats, TIER_BUDGETS[tier])``.

    Raises
    ------
    ValueError
        On an unknown tier key, with a helpful message listing valid keys.
    """
    if tier not in TIER_BUDGETS:
        valid = ", ".join(sorted(TIER_BUDGETS))
        msg = f"Unknown tier {tier!r}.  Valid tiers: {valid}"
        raise ValueError(msg)
    return min(n_splats, TIER_BUDGETS[tier])
