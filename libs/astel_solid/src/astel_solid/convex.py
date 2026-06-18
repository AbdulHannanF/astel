"""Convex decomposition of a watertight mesh into a set of convex hulls.

Used to populate the L5 collision layer (game-engine proxy shapes). The hull
set is INTERNAL scaffolding bound to the splat asset — never the product mesh
(CLAUDE.md §1.2).

Preferred backend: **CoACD** (PyPI ``coacd``, MIT-licensed), lazily imported so
it remains an optional dependency. Fallback: a single scipy convex hull when
CoACD is absent or fails.

Output format:
- ``ConvexSet``: frozen dataclass holding one ``ConvexHull`` per hull.
- ``write_convex_glb``: serialise the hull set. If ``trimesh`` is importable,
  writes a ``.glb``; otherwise writes a ``.npz`` (numpy-only fallback) and
  names the file accordingly.
"""

from __future__ import annotations

import importlib.util
import logging
import queue as _queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .isosurface import TriMesh

logger = logging.getLogger(__name__)

#: Fast, collision-grade CoACD configuration. CoACD's defaults
#: (``resolution=2000``, ``mcts_iterations=150``) run an MCTS search that can take
#: tens of minutes on a detailed mesh; a *collision proxy* does not need that
#: fidelity, so we use a coarser, bounded search. The dominant cost is
#: ``preprocess_resolution`` (CoACD voxel-remeshes the input to a manifold before
#: searching — measured: 50 -> ~286k working triangles, ~30s/MCTS iteration), so
#: it is kept low. Convex-friendly objects decompose in seconds; thin-featured
#: ones (e.g. insect wings) may still exceed the timeout below and fall back.
#: (``max_convex_hull`` is passed separately as the ``max_hulls`` arg, so it is
#: intentionally not duplicated here.)
_COACD_FAST_PARAMS: dict[str, float | int] = {
    "threshold": 0.1,
    "preprocess_resolution": 30,
    "resolution": 512,
    "mcts_iterations": 50,
    "mcts_max_depth": 2,
    "mcts_nodes": 12,
}

#: Hard wall-clock cap (seconds) on the CoACD subprocess. On timeout the worker
#: is terminated and we fall back to a single scipy convex hull, so the producer
#: can never hang in packaging (CLAUDE.md §10: stages must complete). Measured:
#: convex-friendly meshes finish well under this; pathologically thin/concave
#: surfaces (where CoACD's MCTS does not converge) hit the cap and fall back.
DEFAULT_COACD_TIMEOUT_S = 45.0


@dataclass(frozen=True)
class ConvexHull:
    """A single convex hull.

    ``vertices`` ``(V,3)`` float32, ``faces`` ``(F,3)`` int32.
    """

    vertices: NDArray[np.float32]
    faces: NDArray[np.int32]


@dataclass(frozen=True)
class ConvexSet:
    """A set of convex hulls decomposing a mesh (L5 collision proxy).

    ``hulls``: list of per-hull ``ConvexHull``.
    ``method``: either ``"coacd"`` or ``"scipy-hull-fallback"``.
    """

    hulls: list[ConvexHull]
    method: str

    @property
    def n_hulls(self) -> int:
        return len(self.hulls)


def _scipy_single_hull(mesh: TriMesh) -> ConvexSet:
    """Single convex hull via scipy (always available, MIT-compatible)."""
    from scipy.spatial import ConvexHull as _ScipyHull

    hull = _ScipyHull(mesh.vertices.astype(np.float64))
    verts = mesh.vertices[hull.vertices].astype(np.float32)
    # scipy gives simplices in the full-vertex index space; re-index to local
    old_to_new = np.full(mesh.n_vertices, -1, dtype=np.int32)
    old_to_new[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
    faces = old_to_new[hull.simplices].astype(np.int32)
    return ConvexSet(
        hulls=[ConvexHull(vertices=verts, faces=faces)],
        method="scipy-hull-fallback",
    )


def _coacd_worker(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    max_hulls: int,
    params: dict[str, float | int],
    out_q: Any,
) -> None:
    """Run CoACD in a child process; put ``("ok", hulls)`` / ``("err", msg)``.

    Top-level so it is importable under the spawn start method (Windows). Returns
    nothing — the result travels back over ``out_q``. The input mesh is fed
    as-is: CoACD voxel-remeshes it internally, so pre-decimation does not help
    (and welding vertices makes the mesh non-manifold, forcing a *finer* remesh).
    """
    try:
        import coacd  # type: ignore[import-untyped]  # noqa: PLC0415

        mesh = coacd.Mesh(vertices, faces)
        parts = coacd.run_coacd(mesh, max_convex_hull=max_hulls, **params)
        hulls = [
            (np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32))
            for v, f in parts
        ]
        out_q.put(("ok", hulls))
    except BaseException as exc:  # noqa: BLE001 - report any failure to the parent
        out_q.put(("err", repr(exc)))


def _coacd_decompose(
    mesh: TriMesh, max_hulls: int, timeout_s: float
) -> list[tuple[NDArray[np.float32], NDArray[np.int32]]] | None:
    """Run CoACD in a spawned, time-bounded subprocess.

    Returns the per-hull ``(vertices, faces)`` list, or ``None`` on timeout /
    failure (the caller then uses the scipy fallback). The subprocess is hard-
    terminated if it exceeds ``timeout_s`` so a runaway MCTS search cannot hang
    the producer.
    """
    import multiprocessing as mp  # noqa: PLC0415

    ctx = mp.get_context("spawn")
    out_q: Any = ctx.Queue()
    proc = ctx.Process(
        target=_coacd_worker,
        args=(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int32),
            max_hulls,
            dict(_COACD_FAST_PARAMS),
            out_q,
        ),
    )
    proc.start()
    try:
        status, payload = out_q.get(timeout=timeout_s)
    except _queue.Empty:
        logger.warning(
            "CoACD exceeded %.0fs; terminating and using scipy hull fallback",
            timeout_s,
        )
        status, payload = "timeout", None
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5.0)

    if status == "ok":
        return payload  # type: ignore[no-any-return]
    if status == "err":
        logger.warning("CoACD failed (%s); using scipy hull fallback", payload)
    return None


def convex_decompose(
    mesh: TriMesh,
    *,
    max_hulls: int = 32,
    timeout_s: float = DEFAULT_COACD_TIMEOUT_S,
) -> ConvexSet:
    """Decompose ``mesh`` into at most ``max_hulls`` convex hulls.

    Tries CoACD (MIT, optional) in a **time-bounded subprocess** first; falls
    back to a single scipy convex hull if CoACD is absent, raises, or exceeds
    ``timeout_s``. The ``ConvexSet.method`` field records which path was taken so
    callers / the quality report stay honest about it. Isolating CoACD in a
    subprocess is what makes the timeout enforceable: its heavy work is a C++
    extension that an in-process watchdog thread could not interrupt.
    """
    if importlib.util.find_spec("coacd") is None:
        return _scipy_single_hull(mesh)

    try:
        parts = _coacd_decompose(mesh, max_hulls, timeout_s)
    except Exception:  # spawning/IPC failure must never be fatal
        logger.warning(
            "CoACD subprocess error; using scipy hull fallback", exc_info=True
        )
        parts = None

    if not parts:
        return _scipy_single_hull(mesh)

    hulls = [
        ConvexHull(vertices=verts, faces=faces) for verts, faces in parts
    ]
    return ConvexSet(hulls=hulls, method="coacd")


# ---------------------------------------------------------------------------
# Hull set serialisation
# ---------------------------------------------------------------------------


def _write_glb(cset: ConvexSet, path: Path) -> Path:
    """Write the hull set as GLB using trimesh. Raises if trimesh absent/fails."""
    import trimesh  # type: ignore[import-untyped,unused-ignore]
    import trimesh.scene  # type: ignore[import-untyped,unused-ignore]

    scene = trimesh.scene.Scene()
    for i, hull in enumerate(cset.hulls):
        tm = trimesh.Trimesh(
            vertices=hull.vertices.astype(np.float64),
            faces=hull.faces.astype(np.int64),
            process=False,
        )
        scene.add_geometry(tm, node_name=f"hull_{i:03d}")
    out = path.with_suffix(".glb")
    scene.export(str(out))  # type: ignore[no-untyped-call]
    return out


def _write_npz(cset: ConvexSet, path: Path) -> Path:
    """Write the hull set as NPZ (numpy-only fallback)."""
    out = path.with_suffix(".npz")
    arrays: list[tuple[str, NDArray[Any]]] = []
    for i, hull in enumerate(cset.hulls):
        arrays.append((f"verts_{i}", hull.vertices))
        arrays.append((f"faces_{i}", hull.faces))
    arrays.append(("method", np.array(cset.method)))
    np.savez_compressed(str(out), **dict(arrays))  # type: ignore[arg-type]
    return out


def write_convex_glb(cset: ConvexSet, path: str | Path) -> Path:
    """Write the convex hull set to ``path``.

    Tries ``trimesh`` (optional) → writes a ``.glb`` combining all hulls as
    separate mesh geometries in one scene.  If trimesh is not importable,
    writes a ``.npz`` (numpy-only) with arrays ``verts_N`` / ``faces_N`` for
    each hull ``N``, and the ``method`` key, returning the ``.npz`` path.

    Returns the *actual* path written (the suffix may differ from ``path`` if
    the fallback is used and ``path`` ended in ``.glb``).
    """
    p = Path(path)
    try:
        return _write_glb(cset, p)
    except Exception:
        pass  # trimesh absent or failed; fall through to npz
    return _write_npz(cset, p)
