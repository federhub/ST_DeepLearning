"""SVD/EOF decomposition of spatio-temporal matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class SVDResult:
    """Output of the EOF decomposition."""

    s: tf.Tensor  # singular values, shape [n_eofs]
    u: tf.Tensor  # spatial coefficients, shape [n_samples, n_eofs]
    v: tf.Tensor  # temporal basis functions, shape [n_timesteps, n_eofs]
    time_mean: tf.Tensor  # temporal mean, shape [1, n_timesteps]
    nS: tf.Tensor  # number of samples (scalar float)


def svd_decompose(Z_array: np.ndarray, n_eofs: int) -> SVDResult:
    """Decompose a spatio-temporal matrix Z via truncated SVD (EOF analysis).

    Follows the Wikle (2019) convention: time-demean Z, normalise by
    1/sqrt(nS-1), compute full SVD, then rescale U.

    Args:
        Z_array: Shape [n_samples, n_timesteps].
        n_eofs: Number of leading EOF components to retain.

    Returns:
        SVDResult with truncated components.
    """
    Z = tf.cast(Z_array, tf.float32)
    nS = tf.constant(Z.shape[0], dtype=tf.float32)

    time_mean_vec = tf.reduce_mean(Z, axis=0)
    time_mean_tiled = tf.reshape(
        tf.tile(time_mean_vec, [tf.cast(nS, tf.int32)]),
        [tf.cast(nS, tf.int32), time_mean_vec.shape[0]],
    )

    Ztilde = (Z - time_mean_tiled) / tf.sqrt(nS - 1)
    s_full, u_full, v_full = tf.linalg.svd(Ztilde)

    u_full = u_full * tf.sqrt(nS - 1)

    return SVDResult(
        s=s_full[:n_eofs],
        u=u_full[:, :n_eofs],
        v=v_full[:, :n_eofs],
        time_mean=time_mean_tiled[:1, :],
        nS=nS,
    )
