"""Drive COLMAP's SfM pipeline on a folder of images -> a sparse model (L0).

Wraps the COLMAP 4.1 CLI (the CUDA build in ``tools/colmap`` on Box A) as a
sequence of subprocess stages: feature extraction -> exhaustive matching ->
mapping -> undistortion. The undistorter rewrites cameras as PINHOLE, so the
resulting model loads losslessly via :func:`astel_gpu.colmap_io.load_colmap_model`
(no distortion dropped).

Flag names are pinned to COLMAP 4.1.0.dev0 as verified on Box A (e.g. the GPU
toggles are ``--FeatureExtraction.use_gpu`` / ``--FeatureMatching.use_gpu`` --
renamed from the older ``SiftExtraction``/``SiftMatching`` namespaces). The
command builders below are pure (no I/O) and unit-tested; :func:`run_sfm`
executes them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .colmap_io import ColmapModel, load_colmap_model

_DEFAULT_COLMAP = Path("D:/Astel/tools/colmap/bin/colmap.exe")


def feature_extractor_cmd(
    colmap_exe: Path,
    database_path: Path,
    image_path: Path,
    *,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    use_gpu: bool = True,
) -> list[str]:
    """Build the ``colmap feature_extractor`` command."""
    return [
        str(colmap_exe),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
        "--FeatureExtraction.use_gpu",
        "1" if use_gpu else "0",
    ]


def exhaustive_matcher_cmd(
    colmap_exe: Path, database_path: Path, *, use_gpu: bool = True
) -> list[str]:
    """Build the ``colmap exhaustive_matcher`` command (right for ~tens of views)."""
    return [
        str(colmap_exe),
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "1" if use_gpu else "0",
    ]


def mapper_cmd(
    colmap_exe: Path, database_path: Path, image_path: Path, output_path: Path
) -> list[str]:
    """Build the ``colmap mapper`` command (writes a model into ``output_path/0``)."""
    return [
        str(colmap_exe),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
    ]


def image_undistorter_cmd(
    colmap_exe: Path, image_path: Path, input_path: Path, output_path: Path
) -> list[str]:
    """Build the ``colmap image_undistorter`` command (rewrites cameras as PINHOLE)."""
    return [
        str(colmap_exe),
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "COLMAP",
    ]


@dataclass
class SfmResult:
    """Outputs of a successful SfM run."""

    sparse_model_dir: Path  # the chosen (largest) sparse model component
    undistorted_dir: Path  # contains images/ and sparse/ (PINHOLE)
    undistorted_model_dir: Path  # undistorted_dir/sparse, loads losslessly
    model: ColmapModel
    n_registered: int
    n_input_images: int
    stage_seconds: dict[str, float]


def _run_stage(name: str, cmd: list[str], log_dir: Path) -> float:
    """Run one COLMAP stage; tee output to a log; raise with context on failure."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"colmap-{name}.log"
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    log_path.write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"COLMAP {name} failed (exit {proc.returncode}). Log: {log_path}\n{tail}"
        )
    return elapsed


def _largest_model_dir(sparse_dir: Path) -> Path:
    """Pick the sparse component with the most registered images.

    The mapper can split a reconstruction into several components
    (``sparse/0``, ``sparse/1``, ...); we keep the largest.
    """
    candidates = sorted(
        p for p in sparse_dir.iterdir() if p.is_dir() and (p / "images.bin").exists()
    )
    if not candidates:
        raise RuntimeError(f"no COLMAP model produced under {sparse_dir}")
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda d: len(load_colmap_model(d).images))


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    colmap_exe: Path,
    *,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    use_gpu: bool = True,
) -> SfmResult:
    """Run the full COLMAP SfM pipeline on ``image_dir`` -> :class:`SfmResult`.

    Layout under ``work_dir``: ``database.db``, ``sparse/`` (mapper output),
    ``undistorted/`` (PINHOLE images + model). Idempotency is the caller's
    responsibility -- pass a clean ``work_dir`` for a fresh run.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir = work_dir / "undistorted"

    n_input = sum(
        1
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    stage_seconds: dict[str, float] = {}
    stage_seconds["feature_extractor"] = _run_stage(
        "feature_extractor",
        feature_extractor_cmd(
            colmap_exe,
            database,
            image_dir,
            camera_model=camera_model,
            single_camera=single_camera,
            use_gpu=use_gpu,
        ),
        work_dir,
    )
    stage_seconds["exhaustive_matcher"] = _run_stage(
        "exhaustive_matcher",
        exhaustive_matcher_cmd(colmap_exe, database, use_gpu=use_gpu),
        work_dir,
    )
    stage_seconds["mapper"] = _run_stage(
        "mapper",
        mapper_cmd(colmap_exe, database, image_dir, sparse_dir),
        work_dir,
    )

    model_dir = _largest_model_dir(sparse_dir)
    stage_seconds["image_undistorter"] = _run_stage(
        "image_undistorter",
        image_undistorter_cmd(colmap_exe, image_dir, model_dir, undistorted_dir),
        work_dir,
    )

    undistorted_model_dir = undistorted_dir / "sparse"
    model = load_colmap_model(undistorted_model_dir)
    return SfmResult(
        sparse_model_dir=model_dir,
        undistorted_dir=undistorted_dir,
        undistorted_model_dir=undistorted_model_dir,
        model=model,
        n_registered=len(model.images),
        n_input_images=n_input,
        stage_seconds=stage_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP SfM on an image folder.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--colmap-exe", type=Path, default=_DEFAULT_COLMAP)
    parser.add_argument("--camera-model", type=str, default="OPENCV")
    parser.add_argument("--no-single-camera", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="disable GPU SIFT/matching")
    args = parser.parse_args()

    result = run_sfm(
        args.image_dir,
        args.work_dir,
        args.colmap_exe,
        camera_model=args.camera_model,
        single_camera=not args.no_single_camera,
        use_gpu=not args.cpu,
    )
    print(
        json.dumps(
            {
                "n_registered": result.n_registered,
                "n_input_images": result.n_input_images,
                "stage_seconds": result.stage_seconds,
                "undistorted_model_dir": str(result.undistorted_model_dir),
                "any_distortion": result.model.any_distortion,
                "n_sparse_points": int(result.model.points_xyz.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
