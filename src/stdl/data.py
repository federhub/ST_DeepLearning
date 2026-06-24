"""Data loading and preprocessing for simulated and temperature experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from stdl.config import DataConfig


@dataclass
class DataBundle:
    """All arrays needed for one experiment, with coordinates already scaled."""

    X_train: np.ndarray  # [n_train, input_dim]
    X_val: np.ndarray  # [n_val, input_dim]
    X_test: np.ndarray  # [n_test, input_dim]
    X_mesh: np.ndarray  # [n_mesh, input_dim]

    # y_svd = y_train concat y_val — the full training set used for SVD
    y_svd: np.ndarray  # [n_train + n_val, T]
    y_train: np.ndarray  # [n_train, T]
    y_val: np.ndarray  # [n_val, T]
    y_test: np.ndarray  # [n_test, T]

    raw_coords_train: np.ndarray
    raw_coords_test: np.ndarray
    raw_coords_mesh: np.ndarray
    scaler: StandardScaler


def load_simulated_data(cfg: DataConfig) -> DataBundle:
    """Load and preprocess the simulated spatio-temporal dataset.

    Reads pre-generated spatial basis functions and temporal coefficients,
    reconstructs the field, adds Gaussian noise, then splits into train/val/test.
    """
    assert cfg.spatial_basis_path is not None, "spatial_basis_path required"
    assert cfg.time_coeff_path is not None, "time_coeff_path required"

    spat_basis = pd.read_csv(cfg.spatial_basis_path, sep=" ")
    time_coeff = pd.read_csv(cfg.time_coeff_path, index_col=0, sep=" ")
    time_coeff = time_coeff.iloc[: cfg.n_timesteps, :]

    samples = spat_basis.sample(n=cfg.n_spatial_samples, random_state=cfg.random_state)
    coords = samples[["X", "Y"]].values
    basis = samples.iloc[:, -20:].values
    field_coords = spat_basis[["X", "Y"]].values

    # Full-grid field for computing noise sigma
    full_field = (time_coeff.values @ spat_basis.iloc[:, -20:].values.T).T

    sigma = float(cfg.noise_fraction * np.std(full_field))
    tf.random.set_seed(cfg.random_state)

    reconstructed = (time_coeff.values @ basis.T).T  # [n_spatial_samples, T]
    noise = tf.random.normal(reconstructed.shape, 0.0, sigma, dtype=tf.float64)
    reconstructed = (reconstructed + noise.numpy()).astype(np.float32)

    n_tv = cfg.train_val_size
    vi = cfg.val_split_idx
    n_te = cfg.test_size

    X_tv_raw = coords[:n_tv]
    y_svd = reconstructed[:n_tv]
    X_test_raw = coords[n_tv : n_tv + n_te]
    y_test = reconstructed[n_tv : n_tv + n_te]

    scaler = StandardScaler().fit(X_tv_raw[:vi])
    X_train = scaler.transform(X_tv_raw[:vi])
    X_val = scaler.transform(X_tv_raw[vi:])
    X_test_scaled = scaler.transform(X_test_raw)
    X_mesh = scaler.transform(field_coords)

    return DataBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test_scaled.astype(np.float32),
        X_mesh=X_mesh.astype(np.float32),
        y_svd=y_svd,
        y_train=y_svd[:vi],
        y_val=y_svd[vi:],
        y_test=y_test,
        raw_coords_train=X_tv_raw,
        raw_coords_test=X_test_raw,
        raw_coords_mesh=field_coords,
        scaler=scaler,
    )


def load_temperature_data(cfg: DataConfig) -> DataBundle:
    """Load and preprocess Swiss temperature station data.

    Expects pre-split CSV files (train/test) at the paths defined in cfg.
    Data files are not included in the repository; place them under data/temperature/.
    """
    assert cfg.y_train_path is not None, "y_train_path required"
    assert cfg.y_test_path is not None, "y_test_path required"
    assert cfg.X_train_path is not None, "X_train_path required"
    assert cfg.X_test_path is not None, "X_test_path required"
    assert cfg.mesh_path is not None, "mesh_path required"

    y_train_df = pd.read_csv(cfg.y_train_path, index_col=0).T
    y_test_df = pd.read_csv(cfg.y_test_path, index_col=0).T
    X_train_df = pd.read_csv(cfg.X_train_path, index_col=0).iloc[:, 1:4]
    X_test_df = pd.read_csv(cfg.X_test_path, index_col=0).iloc[:, 1:4]
    mesh_df = pd.read_csv(cfg.mesh_path, header=0).iloc[:, 0:4]
    mesh_df = mesh_df.reindex(columns=["X", "Y", "Z", "Switzerland"])

    vi = cfg.val_split_idx
    X_tv_raw = X_train_df.values
    y_svd = y_train_df.values.astype(np.float32)
    X_test_raw = X_test_df.values
    y_test = y_test_df.values.astype(np.float32)

    scaler = StandardScaler().fit(X_tv_raw[:vi])
    X_train = scaler.transform(X_tv_raw[:vi])
    X_val = scaler.transform(X_tv_raw[vi:])
    X_test_scaled = scaler.transform(X_test_raw)
    X_mesh = scaler.transform(mesh_df.iloc[:, :3].values)

    return DataBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test_scaled.astype(np.float32),
        X_mesh=X_mesh.astype(np.float32),
        y_svd=y_svd,
        y_train=y_svd[:vi],
        y_val=y_svd[vi:],
        y_test=y_test,
        raw_coords_train=X_tv_raw,
        raw_coords_test=X_test_raw,
        raw_coords_mesh=mesh_df.iloc[:, :3].values,
        scaler=scaler,
    )


def load_data(cfg: DataConfig) -> DataBundle:
    """Dispatch to the correct loader based on cfg.dataset."""
    if cfg.dataset == "simulated":
        return load_simulated_data(cfg)
    if cfg.dataset == "temperature":
        return load_temperature_data(cfg)
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
