"""SH-L2 basis + irradiance correctness against analytic ground truth."""

from __future__ import annotations

import numpy as np

from astel_appearance.sh import (
    COSINE_CONV,
    N_SH_L2,
    diffuse_shading,
    fit_environment_sh,
    sh_eval_l2,
    yaw_rotation,
)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0**0.5)
    theta = golden * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def test_dc_constant() -> None:
    dirs = _fibonacci_sphere(64)
    y = sh_eval_l2(dirs)
    assert np.allclose(y[:, 0], 0.28209479177387814)


def test_basis_orthonormal_over_sphere() -> None:
    dirs = _fibonacci_sphere(8000)
    y = sh_eval_l2(dirs)
    # mean(Y_i Y_j) ~= delta_ij / (4pi)  =>  *4pi ~= identity
    gram = (y.T @ y) / dirs.shape[0] * 4.0 * np.pi
    assert np.allclose(gram, np.eye(N_SH_L2), atol=2e-2)


def test_white_furnace_constant_shading() -> None:
    # Constant radiance c -> only DC coeff non-zero -> shading == c everywhere.
    c = 0.7
    env = np.zeros(N_SH_L2)
    env[0] = c / 0.28209479177387814
    dirs = _fibonacci_sphere(200)
    shading = diffuse_shading(env, dirs)
    assert np.allclose(shading, c, atol=1e-6)


def test_cosine_conv_constants() -> None:
    assert COSINE_CONV[0] == 1.0
    assert np.allclose(COSINE_CONV[1:4], 2.0 / 3.0)
    assert np.allclose(COSINE_CONV[4:9], 0.25)


def test_fit_recovers_known_env() -> None:
    rng = np.random.default_rng(1)
    dirs = rng.standard_normal((4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    env_true = rng.standard_normal(N_SH_L2)
    shading = diffuse_shading(env_true, dirs)
    env_fit = fit_environment_sh(dirs, shading, ridge=1e-6)
    assert np.allclose(env_fit, env_true, atol=2e-3)


def test_fit_multichannel() -> None:
    rng = np.random.default_rng(2)
    dirs = rng.standard_normal((2000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    env_true = rng.standard_normal((N_SH_L2, 3))
    values = diffuse_shading(env_true, dirs)  # (N, 3)
    env_fit = fit_environment_sh(dirs, values, ridge=1e-6)
    assert env_fit.shape == (N_SH_L2, 3)
    assert np.allclose(env_fit, env_true, atol=3e-3)


def test_yaw_rotation_orthonormal() -> None:
    r = yaw_rotation(0.7)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)
