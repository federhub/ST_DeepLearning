"""Smoke tests for SVD decomposition."""

from __future__ import annotations

import numpy as np


def test_svd_shapes():
    """SVDResult tensors have correct shapes."""
    from stdl.decomposition import svd_decompose

    n_samples, n_time, n_eofs = 20, 50, 5
    Z = np.random.default_rng(0).standard_normal((n_samples, n_time)).astype(np.float32)
    result = svd_decompose(Z, n_eofs=n_eofs)

    assert tuple(result.s.shape) == (n_eofs,)
    assert tuple(result.u.shape) == (n_samples, n_eofs)
    assert tuple(result.v.shape) == (n_time, n_eofs)
    assert tuple(result.time_mean.shape) == (1, n_time)


def test_svd_singular_values_descending():
    """Singular values are in descending order."""
    from stdl.decomposition import svd_decompose

    Z = np.random.default_rng(1).standard_normal((15, 30)).astype(np.float32)
    result = svd_decompose(Z, n_eofs=5)
    s = np.array(result.s)
    assert np.all(s[:-1] >= s[1:])


def test_svd_full_rank_reconstruction():
    """Full-rank decomposition reconstructs Z with negligible error."""
    from stdl.decomposition import svd_decompose

    rng = np.random.default_rng(42)
    n, T = 12, 25
    Z = rng.standard_normal((n, T)).astype(np.float32)
    n_eofs = min(n, T) - 1
    result = svd_decompose(Z, n_eofs=n_eofs)

    sqrt_nS = np.sqrt(float(result.nS) - 1)
    sv_t = np.diag(np.array(result.s)) @ np.array(result.v).T
    Z_approx = (np.array(result.u) / sqrt_nS @ sv_t) * sqrt_nS + np.array(
        result.time_mean
    )

    np.testing.assert_allclose(Z_approx, Z, atol=1e-4)
