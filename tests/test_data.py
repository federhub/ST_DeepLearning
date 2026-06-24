"""Smoke tests for data loading."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data" / "simulated"


def _tiny_cfg():
    from stdl.config import DataConfig

    return DataConfig(
        dataset="simulated",
        spatial_basis_path=DATA_DIR / "SIM1_spatialbasis.txt",
        time_coeff_path=DATA_DIR / "SIM1_timecoeff.txt",
        n_spatial_samples=120,
        n_timesteps=10,
        train_val_size=60,
        val_split_idx=40,
        test_size=20,
        random_state=0,
    )


@pytest.mark.skipif(
    not (DATA_DIR / "SIM1_spatialbasis.txt").exists(),
    reason="Simulated data files not found",
)
def test_load_simulated_shapes():
    """DataBundle arrays have the expected shapes."""
    from stdl.data import load_simulated_data

    cfg = _tiny_cfg()
    bundle = load_simulated_data(cfg)

    assert bundle.X_train.shape == (cfg.val_split_idx, 2)
    assert bundle.X_val.shape == (cfg.train_val_size - cfg.val_split_idx, 2)
    assert bundle.X_test.shape == (cfg.test_size, 2)
    assert bundle.y_train.shape == (cfg.val_split_idx, cfg.n_timesteps)
    assert bundle.y_svd.shape == (cfg.train_val_size, cfg.n_timesteps)
    assert bundle.y_test.shape == (cfg.test_size, cfg.n_timesteps)


@pytest.mark.skipif(
    not (DATA_DIR / "SIM1_spatialbasis.txt").exists(),
    reason="Simulated data files not found",
)
def test_load_simulated_dtypes():
    """All coordinate and target arrays are float32."""
    import numpy as np

    from stdl.data import load_simulated_data

    bundle = load_simulated_data(_tiny_cfg())
    for arr in (bundle.X_train, bundle.X_val, bundle.X_test, bundle.y_train):
        assert arr.dtype == np.float32


@pytest.mark.skipif(
    not (DATA_DIR / "SIM1_spatialbasis.txt").exists(),
    reason="Simulated data files not found",
)
def test_load_data_dispatch():
    """load_data() correctly dispatches to the simulated loader."""
    from stdl.data import load_data

    bundle = load_data(_tiny_cfg())
    assert bundle.X_train.shape[1] == 2
