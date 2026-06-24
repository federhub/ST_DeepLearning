"""Evaluation metrics and diagnostic statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def temporal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE at each timestep across all spatial locations.

    Args:
        y_true: Shape [n_locations, T].
        y_pred: Shape [n_locations, T].

    Returns:
        Array of shape [T].
    """
    return np.array(
        [
            mean_absolute_error(y_true[:, t], y_pred[:, t])
            for t in range(y_true.shape[1])
        ]
    )


def spatial_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE at each spatial location across all timesteps.

    Args:
        y_true: Shape [n_locations, T].
        y_pred: Shape [n_locations, T].

    Returns:
        Array of shape [n_locations].
    """
    return np.array(
        [
            mean_absolute_error(y_true[i, :], y_pred[i, :])
            for i in range(y_true.shape[0])
        ]
    )


def control_stats(y_train: np.ndarray, y_hat_mesh: np.ndarray) -> pd.DataFrame:
    """Per-timestep distributional comparison between training data and mesh prediction.

    Useful for diagnosing the smoothing effect of the neural network: predicted
    maps typically underestimate the range seen in training data.

    Args:
        y_train: Training targets, shape [n_train, T].
        y_hat_mesh: Mesh predictions, shape [n_mesh, T].

    Returns:
        DataFrame with columns [y_mean, y_var, y_min, y_max,
        mesh_mean, mesh_var, mesh_min, mesh_max], one row per timestep.
    """
    T = y_train.shape[1]
    rows = [
        {
            "y_mean": float(np.mean(y_train[:, t])),
            "y_var": float(np.var(y_train[:, t])),
            "y_min": float(np.min(y_train[:, t])),
            "y_max": float(np.max(y_train[:, t])),
            "mesh_mean": float(np.mean(y_hat_mesh[:, t])),
            "mesh_var": float(np.var(y_hat_mesh[:, t])),
            "mesh_min": float(np.min(y_hat_mesh[:, t])),
            "mesh_max": float(np.max(y_hat_mesh[:, t])),
        }
        for t in range(T)
    ]
    return pd.DataFrame(rows)
