"""CPU-only tests for the INRIA PLY export conversion and camera rig math.

These do not require CUDA/gsplat and run on any machine.
"""

from __future__ import annotations

import numpy as np
import torch
from astel_splat_io.ply import read_ply

from astel_gpu.cameras import build_camera_rig
from astel_gpu.export import to_splat_cloud
from astel_gpu.gaussians import GaussianParams, build_target_cloud


def test_to_splat_cloud_roundtrip_albedo_and_alpha() -> None:
    params = build_target_cloud(n=50, seed=1, device=torch.device("cpu"))
    cloud = to_splat_cloud(params)

    # albedo = 0.5 + SH_C0 * f_dc should recover the original colors.
    sh_c0 = 0.28209479177387814
    albedo = 0.5 + sh_c0 * cloud.colors_dc
    np.testing.assert_allclose(
        albedo, params.colors.numpy(), atol=1e-5
    )

    # alpha = sigmoid(opacity_logit) should recover the original opacities.
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacity))
    np.testing.assert_allclose(alpha, params.opacities.numpy(), atol=1e-5)

    # log_scales should exponentiate back to the original linear scales.
    np.testing.assert_allclose(
        np.exp(cloud.log_scales), params.scales.numpy(), atol=1e-5
    )

    assert cloud.count == 50


def test_to_splat_cloud_quats_normalized() -> None:
    params = GaussianParams(
        means=torch.zeros(4, 3),
        scales=torch.ones(4, 3) * 0.1,
        quats=torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 4),
        opacities=torch.full((4,), 0.5),
        colors=torch.full((4, 3), 0.5),
    )
    cloud = to_splat_cloud(params)
    norms = np.linalg.norm(cloud.quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_write_and_read_back_ply(tmp_path: object) -> None:
    from pathlib import Path

    from astel_gpu.export import write_gaussian_ply

    params = build_target_cloud(n=20, seed=2, device=torch.device("cpu"))
    out = Path(str(tmp_path)) / "l3.ply"
    write_gaussian_ply(params, out)

    cloud = read_ply(out)
    assert cloud.count == 20


def test_build_camera_rig_shapes() -> None:
    viewmats, ks = build_camera_rig(n_views=8, image_size=64)
    assert viewmats.shape == (8, 4, 4)
    assert ks.shape == (8, 3, 3)

    # Each viewmat's rotation block should be orthonormal (a valid rotation).
    rot = viewmats[:, :3, :3]
    should_be_identity = torch.bmm(rot, rot.transpose(1, 2))
    eye = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
    torch.testing.assert_close(should_be_identity, eye, atol=1e-4, rtol=1e-4)
