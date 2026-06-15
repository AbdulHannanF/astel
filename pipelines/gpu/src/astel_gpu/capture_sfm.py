"""COLMAP SfM front-end validation against DTU's ground-truth poses.

Runs (or loads) a COLMAP reconstruction of a DTU scan's images, then aligns the
scale-free COLMAP camera centres to DTU's metric GT centres with a similarity
transform (Umeyama). Reports how well the SfM front-end recovers the real rig:
pose RMSE in mm + the recovered metric scale. Pure numpy (no gsplat/torch) ->
runs without the MSVC launcher.

This is the *pose-from-images* validation that complements
:mod:`astel_gpu.capture_eval` (which uses DTU's GT poses): together they cover
"can we recover poses from real photos?" and "given good poses, how accurate is
the splat geometry?".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .colmap_io import load_colmap_model
from .dtu import camera_center, decompose_projection, read_pos_matrix, umeyama

_VIEW_RE = re.compile(r"rect_(\d+)_")


def run_sfm_pose_eval(colmap_model_dir: Path, pos_dir: Path) -> dict[str, Any]:
    """Align COLMAP camera centres to DTU GT centres; report pose accuracy (mm)."""
    model = load_colmap_model(Path(colmap_model_dir))
    pos_dir = Path(pos_dir)

    colmap_centres: list[np.ndarray] = []
    gt_centres: list[np.ndarray] = []
    matched: list[str] = []
    for im in model.images:
        match = _VIEW_RE.search(im.name)
        if match is None:
            continue
        pos_path = pos_dir / f"pos_{match.group(1)}.txt"
        if not pos_path.exists():
            continue
        vm = im.viewmat()
        colmap_centres.append(-vm[:3, :3].T @ vm[:3, 3])
        _k, rot_gt, t_gt = decompose_projection(read_pos_matrix(pos_path))
        gt_centres.append(camera_center(rot_gt, t_gt))
        matched.append(match.group(1))

    if len(matched) < 3:
        raise RuntimeError(
            f"only {len(matched)} matched cameras; need >= 3 for Umeyama"
        )

    src = np.stack(colmap_centres)
    dst = np.stack(gt_centres)
    scale, rot, trans, rmse = umeyama(src, dst)
    aligned = (scale * (rot @ src.T)).T + trans
    resid = np.linalg.norm(aligned - dst, axis=1)

    return {
        "origin": "measured",
        "n_registered": len(model.images),
        "n_matched_to_gt": len(matched),
        "pose_rmse_mm": rmse,
        "pose_median_mm": float(np.median(resid)),
        "pose_max_mm": float(resid.max()),
        "recovered_scale": scale,
        "n_sparse_points": int(model.points_xyz.shape[0]),
        "note": (
            "COLMAP camera centres (scale-free) aligned to DTU GT centres via "
            "Umeyama similarity; RMSE is the residual after alignment (mm). "
            "Proves the SfM front-end recovers the real camera rig; the splat "
            "geometry accuracy given good poses is measured by capture_eval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="COLMAP-vs-DTU pose accuracy.")
    parser.add_argument("--colmap-model-dir", type=Path, required=True)
    parser.add_argument("--pos-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_sfm_pose_eval(args.colmap_model_dir, args.pos_dir)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "sfm-pose-eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
