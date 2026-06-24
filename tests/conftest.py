"""Shared pytest fixtures for stdl tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session")
def tiny_svd():
    """Minimal SVDResult on 10 samples × 20 timesteps, 2 EOFs."""
    from stdl.decomposition import svd_decompose

    rng = np.random.RandomState(0)
    Z = rng.randn(10, 20).astype(np.float32)
    return svd_decompose(Z, n_eofs=2)


@pytest.fixture(scope="session")
def tiny_bundle(tiny_svd):
    """Minimal DataBundle matching tiny_svd dimensions."""
    from stdl.data import DataBundle

    rng = np.random.RandomState(1)
    n_train, n_val, n_test, T = 6, 4, 3, 20
    X_raw = rng.randn(n_train + n_val, 2).astype(np.float32)
    scaler = StandardScaler().fit(X_raw[:n_train])

    y_svd = rng.randn(n_train + n_val, T).astype(np.float32)
    return DataBundle(
        X_train=scaler.transform(X_raw[:n_train]).astype(np.float32),
        X_val=scaler.transform(X_raw[n_train:]).astype(np.float32),
        X_test=scaler.transform(rng.randn(n_test, 2)).astype(np.float32),
        X_mesh=scaler.transform(rng.randn(5, 2)).astype(np.float32),
        y_svd=y_svd,
        y_train=y_svd[:n_train],
        y_val=y_svd[n_train:],
        y_test=rng.randn(n_test, T).astype(np.float32),
        raw_coords_train=X_raw[:n_train],
        raw_coords_test=rng.randn(n_test, 2).astype(np.float32),
        raw_coords_mesh=rng.randn(5, 2).astype(np.float32),
        scaler=scaler,
    )
