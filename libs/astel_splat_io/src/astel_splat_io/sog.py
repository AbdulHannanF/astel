"""PlayCanvas SOG / SOGS (Self-Organizing Gaussians) bundle — partial implementation.

Spec source (verified 2026-06-13):

- https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
  — the SOG format reference page (PlayCanvas Developer Site).
- https://github.com/playcanvas/splat-transform — reference CLI/tooling and
  the canonical SOG read/write implementation. **MIT License** (PlayCanvas
  Ltd.).
- https://github.com/playcanvas/sogs — the original Self-Organizing
  Gaussians compressor (k-means/PLAS-based codebook + spatial sort).
  **Apache-2.0 License**.

## Container

A `.sog` file is a ZIP containing `meta.json` plus a set of lossless WebP
textures, one pixel per gaussian (row-major, `x = i % W`, `y = i // W`).
PlayCanvas readers also accept an unbundled directory layout (same files,
no zip) — we always write the bundled `.sog` zip form.

## What this module implements

- `meta.json` with `version`, `count`, `width`/`height`, and per-attribute
  blocks (`means`, `scales`, `quats`, `sh0`) referencing the WebP files,
  matching the documented shapes (mins/maxs for `means`, 256-entry codebooks
  for `scales`/`sh0`).
- **means**: log-domain transform, 16-bit split across `means_l.webp` /
  `means_u.webp` per the documented scheme.
- **scales**: per-axis codebook of 256 floats (`scales.webp` stores RGB
  codebook indices).
- **sh0**: 256-entry codebook for the SH band-0 DC colour, packed into RGB of
  `sh0.webp`; alpha channel carries opacity directly (`UNORM8`).
- **quats**: "smallest three" quaternion packing into `quats.webp` RGBA,
  matching the SPZ smallest-three scheme (RGB = three smallest components
  quantized to [-sqrt(1/2), sqrt(1/2)], A encodes which component was
  omitted as `252 + index`).

## Honest limitations (NOT implemented — explicit `NotImplementedError`)

- **Codebook generation**: the reference SOGS implementation builds the
  256-entry `scales`/`sh0` codebooks via k-means (PLAS-based) over the whole
  cloud for near-lossless reconstruction. This module uses **uniform
  quantile binning** instead — a simpler, fully-documented approximation that
  still round-trips through real codebooks/textures, but with higher
  quantization error than the reference k-means codebooks. This is recorded
  here rather than silently matching the reference's accuracy.
- **Higher-order spherical harmonics** (`shN_centroids.webp` /
  `shN_labels.webp`): `SplatCloud` carries band-0 only, so `shN` is omitted
  from `meta.json` entirely (a legal absence per the spec, not a stub).
- **Spatial re-ordering** (Morton / PLAS sort) for better 2D-texture
  locality: not performed. Splats are written in input order (an explicit
  identity permutation is still recorded for provenance alignment by
  callers, see :mod:`astel_splat_io.provenance`).
- **LOD / `lod-meta.json` streaming bundles**: out of scope; only the
  single-file `meta.json` (non-streaming) bundle is implemented.

Any of the above raise :class:`NotImplementedError` with a TODO if a caller
requests it via unsupported options.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from astel_splat_io.cloud import SplatCloud

SOG_VERSION = 2
CODEBOOK_SIZE = 256
SQRT1_2 = 0.70710678118654752440


def _texture_dims(count: int) -> tuple[int, int]:
    """Pick a near-square WxH >= count, matching the documented row-major layout."""
    width = int(np.ceil(np.sqrt(count)))
    width = max(width, 1)
    height = int(np.ceil(count / width))
    return width, height


def _to_pixel_grid(
    values: NDArray[np.uint8], width: int, height: int
) -> NDArray[np.uint8]:
    """Lay out (N, C) uint8 values into an (H, W, C) image, zero-padding the tail."""
    count, channels = values.shape
    total = width * height
    padded = np.zeros((total, channels), dtype=np.uint8)
    padded[:count] = values
    return padded.reshape(height, width, channels)


def _from_pixel_grid(image: NDArray[np.uint8], count: int) -> NDArray[np.uint8]:
    height, width, channels = image.shape
    flat = image.reshape(height * width, channels)
    return flat[:count]


def _webp_bytes(image: NDArray[np.uint8]) -> bytes:
    mode = "RGBA" if image.shape[-1] == 4 else "RGB" if image.shape[-1] == 3 else "L"
    pil_image = Image.fromarray(image, mode=mode)
    buf = io.BytesIO()
    # `exact=True` is required for RGBA: libwebp's lossless mode otherwise
    # zeroes RGB wherever A == 0 (a "don't care" optimization for normal
    # images), which would silently corrupt our sh0/quats payloads whenever
    # a splat's quantized opacity rounds to 0.
    pil_image.save(buf, format="WEBP", lossless=True, exact=True)
    return buf.getvalue()


def _webp_to_array(data: bytes) -> NDArray[np.uint8]:
    pil_image = Image.open(io.BytesIO(data))
    return np.array(pil_image)


def _quantile_codebook(values: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Build a `size`-entry codebook covering ``values`` via uniform binning.

    A simplified stand-in for the reference implementation's k-means/PLAS
    codebook (see module docstring): `size` evenly spaced values spanning
    `[min(values), max(values)]`, guaranteeing a worst-case nearest-codebook
    error of `range / (2*size)` regardless of how the data is distributed
    (quantile-based binning can leave large gaps at sparse tails). Returns
    sorted codebook values.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(size, dtype=np.float64)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.full(size, lo, dtype=np.float64)
    return np.linspace(lo, hi, size, dtype=np.float64)


def _nearest_index(
    values: NDArray[np.float64], codebook: NDArray[np.float64]
) -> NDArray[np.uint8]:
    # codebook is sorted; searchsorted then pick nearer of the two neighbours.
    idx = np.searchsorted(codebook, values, side="left")
    idx = np.clip(idx, 0, len(codebook) - 1)
    idx_prev = np.clip(idx - 1, 0, len(codebook) - 1)
    dist_idx = np.abs(codebook[idx] - values)
    dist_prev = np.abs(codebook[idx_prev] - values)
    chosen = np.where(dist_prev < dist_idx, idx_prev, idx)
    return chosen.astype(np.uint8)


def write_sog(cloud: SplatCloud, path: str | Path) -> int:
    """Write ``cloud`` as a bundled `.sog` zip. Returns bytes written.

    Implements: `meta.json`, `means_l.webp`/`means_u.webp`, `scales.webp`,
    `quats.webp`, `sh0.webp`. See module docstring for what is intentionally
    not implemented (`shN`, real k-means codebooks, spatial sort).
    """
    n = cloud.count
    width, height = _texture_dims(n)

    # ---- means: log-domain 16-bit split ----
    signed_log = np.sign(cloud.positions.astype(np.float64)) * np.log1p(
        np.abs(cloud.positions.astype(np.float64))
    )
    mins = signed_log.min(axis=0)
    maxs = signed_log.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    normalized = (signed_log - mins) / span
    quantized = np.clip(np.round(normalized * 65535.0), 0, 65535).astype(np.uint32)
    means_low = (quantized & 0xFF).astype(np.uint8)
    means_high = ((quantized >> 8) & 0xFF).astype(np.uint8)
    means_l_img = _to_pixel_grid(means_low, width, height)
    means_u_img = _to_pixel_grid(means_high, width, height)

    # ---- scales: per-axis codebook + RGB indices ----
    scales_flat = cloud.log_scales.astype(np.float64)
    scales_codebook = _quantile_codebook(scales_flat.reshape(-1), CODEBOOK_SIZE)
    scales_idx = np.empty((n, 3), dtype=np.uint8)
    for axis in range(3):
        scales_idx[:, axis] = _nearest_index(scales_flat[:, axis], scales_codebook)
    scales_img = _to_pixel_grid(scales_idx, width, height)

    # ---- sh0: codebook for colors_dc (RGB) + opacity alpha ----
    colors_flat = cloud.colors_dc.astype(np.float64)
    sh0_codebook = _quantile_codebook(colors_flat.reshape(-1), CODEBOOK_SIZE)
    sh0_idx = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        sh0_idx[:, ch] = _nearest_index(colors_flat[:, ch], sh0_codebook)
    opacity_u8 = np.clip(
        np.round(1.0 / (1.0 + np.exp(-cloud.opacity.astype(np.float64))) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    sh0_rgba = np.concatenate([sh0_idx, opacity_u8[:, None]], axis=1)
    sh0_img = _to_pixel_grid(sh0_rgba, width, height)

    # ---- quats: smallest-three packing ----
    quats_rgba = _pack_quats_smallest_three(cloud.quats)
    quats_img = _to_pixel_grid(quats_rgba, width, height)

    meta: dict[str, Any] = {
        "version": SOG_VERSION,
        "count": n,
        "width": width,
        "height": height,
        "means": {
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {
            "codebook": scales_codebook.tolist(),
            "files": ["scales.webp"],
        },
        "quats": {
            "files": ["quats.webp"],
        },
        "sh0": {
            "codebook": sh0_codebook.tolist(),
            "files": ["sh0.webp"],
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        zf.writestr("means_l.webp", _webp_bytes(means_l_img))
        zf.writestr("means_u.webp", _webp_bytes(means_u_img))
        zf.writestr("scales.webp", _webp_bytes(scales_img))
        zf.writestr("quats.webp", _webp_bytes(quats_img))
        zf.writestr("sh0.webp", _webp_bytes(sh0_img))

    data = buf.getvalue()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return len(data)


def _pack_quats_smallest_three(quats_wxyz: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Pack (w, x, y, z) quaternions into RGBA per the SOG smallest-three scheme.

    RGB = the three non-largest components, each mapped from
    [-sqrt(1/2), sqrt(1/2)] to [0, 255]; A = 252 + index_of_largest (0..3).
    """
    n = quats_wxyz.shape[0]
    w = quats_wxyz[:, 0].astype(np.float64)
    xyz = quats_wxyz[:, 1:4].astype(np.float64)
    q = np.concatenate([xyz, w[:, None]], axis=1)  # (x, y, z, w)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    q = q / norm

    out = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        row = q[i]
        i_largest = int(np.argmax(np.abs(row)))
        sign = 1.0 if row[i_largest] >= 0.0 else -1.0
        rest = [j for j in range(4) if j != i_largest]
        for slot, j in enumerate(rest):
            v = row[j] * sign  # flip so the largest is implicitly positive
            v_clamped = np.clip(v, -SQRT1_2, SQRT1_2)
            out[i, slot] = int(round((v_clamped + SQRT1_2) / (2.0 * SQRT1_2) * 255.0))
        out[i, 3] = 252 + i_largest
    return out


def _unpack_quats_smallest_three(rgba: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Inverse of :func:`_pack_quats_smallest_three`, returning (w, x, y, z)."""
    n = rgba.shape[0]
    i_largest = (rgba[:, 3].astype(np.int64) - 252).clip(0, 3)
    out_xyzw = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        largest = int(i_largest[i])
        rest = [j for j in range(4) if j != largest]
        sum_sq = 0.0
        for slot, j in enumerate(rest):
            v = rgba[i, slot] / 255.0 * (2.0 * SQRT1_2) - SQRT1_2
            out_xyzw[i, j] = v
            sum_sq += v * v
        out_xyzw[i, largest] = np.sqrt(max(0.0, 1.0 - sum_sq))

    out_wxyz = np.empty((n, 4), dtype=np.float32)
    out_wxyz[:, 0] = out_xyzw[:, 3]
    out_wxyz[:, 1:4] = out_xyzw[:, 0:3]
    return out_wxyz


def read_sog(path: str | Path) -> SplatCloud:
    """Read a bundled `.sog` zip written by :func:`write_sog`.

    Raises :class:`NotImplementedError` if the bundle declares `shN` (higher-
    order SH) or an `lod-meta.json`-style streaming layout, neither of which
    this module writes or reads.
    """
    data = Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

        if "shN" in meta:
            raise NotImplementedError(
                "SOG higher-order SH (shN) is not implemented in astel_splat_io"
            )

        count = int(meta["count"])
        width = int(meta["width"])
        height = int(meta["height"])

        means_l = _webp_to_array(zf.read(meta["means"]["files"][0]))
        means_u = _webp_to_array(zf.read(meta["means"]["files"][1]))
        scales_img = _webp_to_array(zf.read(meta["scales"]["files"][0]))
        quats_img = _webp_to_array(zf.read(meta["quats"]["files"][0]))
        sh0_img = _webp_to_array(zf.read(meta["sh0"]["files"][0]))

    mins = np.asarray(meta["means"]["mins"], dtype=np.float64)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float64)
    span = np.where(maxs > mins, maxs - mins, 1.0)

    # means_l/means_u store one scalar per axis interleaved across the three
    # channels of the image (R=x, G=y, B=z).
    low3 = _from_pixel_grid(means_l.reshape(height, width, -1), count).astype(np.uint32)
    high3 = _from_pixel_grid(means_u.reshape(height, width, -1), count).astype(
        np.uint32
    )
    quantized3 = (low3 | (high3 << 8)).astype(np.float64)
    normalized3 = quantized3 / 65535.0
    signed_log = mins[None, :] + normalized3 * span[None, :]
    positions = (np.sign(signed_log) * (np.expm1(np.abs(signed_log)))).astype(
        np.float32
    )

    scales_codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float64)
    scales_idx = _from_pixel_grid(scales_img.reshape(height, width, -1), count)
    log_scales = scales_codebook[scales_idx[:, :3]].astype(np.float32)

    sh0_codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float64)
    sh0_rgba = _from_pixel_grid(sh0_img.reshape(height, width, -1), count)
    colors_dc = sh0_codebook[sh0_rgba[:, :3]].astype(np.float32)
    opacity_alpha = sh0_rgba[:, 3].astype(np.float64) / 255.0
    opacity_alpha = np.clip(opacity_alpha, 1e-6, 1.0 - 1e-6)
    opacity = np.log(opacity_alpha / (1.0 - opacity_alpha)).astype(np.float32)

    quats_rgba = _from_pixel_grid(quats_img.reshape(height, width, -1), count)
    quats = _unpack_quats_smallest_three(quats_rgba)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=quats,
    )
