"""Keras model definition for spatio-temporal field prediction."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from stdl.config import ModelConfig
from stdl.decomposition import SVDResult


@keras.utils.register_keras_serializable(package="stdl")
class RecomposeLayer(keras.layers.Layer):
    """Reconstructs the full spatio-temporal field from predicted EOF coefficients.

    Given predicted spatial coefficients U (shape [batch, n_eofs]), applies the
    inverse EOF transform to recover the original field at all timesteps.
    """

    def __init__(
        self,
        s: list[float],
        v: list[list[float]],
        time_mean_row: list[float],
        nS: float,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        s_arr = np.array(s, dtype=np.float32)
        v_arr = np.array(v, dtype=np.float32)
        mean_arr = np.array(time_mean_row, dtype=np.float32)
        sqrt_nS = np.sqrt(nS - 1, dtype=np.float32)

        # Precompute diag(s) @ V^T  →  [n_eofs, n_timesteps]
        sv_t = (np.diag(s_arr) @ v_arr.T).astype(np.float32)

        self._sqrt_nS = tf.constant(sqrt_nS, dtype=tf.float32)
        self._sv_t = tf.constant(sv_t, dtype=tf.float32)
        self._time_mean = tf.constant(mean_arr[np.newaxis, :], dtype=tf.float32)

        # Keep originals for get_config
        self._cfg_s = s_arr.tolist()
        self._cfg_v = v_arr.tolist()
        self._cfg_mean = mean_arr.tolist()
        self._cfg_nS = float(nS)

    @classmethod
    def from_svd(cls, svd: SVDResult, **kwargs: object) -> RecomposeLayer:
        """Construct from an SVDResult dataclass."""
        return cls(
            s=np.array(svd.s).tolist(),
            v=np.array(svd.v).tolist(),
            time_mean_row=np.array(svd.time_mean[0]).tolist(),
            nS=float(svd.nS),
            **kwargs,
        )

    def call(self, U: tf.Tensor) -> tf.Tensor:
        U = tf.cast(U, tf.float32)
        return (
            tf.matmul(U / self._sqrt_nS, self._sv_t) * self._sqrt_nS + self._time_mean
        )

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            s=self._cfg_s,
            v=self._cfg_v,
            time_mean_row=self._cfg_mean,
            nS=self._cfg_nS,
        )
        return cfg


def build_model(cfg: ModelConfig, svd: SVDResult) -> keras.Model:
    """Build the spatio-temporal deep learning model.

    Architecture: input coordinates → stacked Dense blocks (optionally with
    BatchNorm) → auxiliary output of spatial EOF coefficients → RecomposeLayer
    that reconstructs the full time series at each location.

    Args:
        cfg: Model hyperparameters.
        svd: SVD decomposition result used to initialise the RecomposeLayer.

    Returns:
        Compiled-ready Keras Model with outputs [field, spatial_coefficients].
    """
    inp = keras.layers.Input(shape=(cfg.input_dim,), name="coords")
    x = inp
    for i, units in enumerate(cfg.hidden_sizes):
        if cfg.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = keras.layers.Dense(
            units,
            activation=cfg.activation,
            kernel_initializer="he_normal",
            name=f"dense_{i}",
        )(x)
    if cfg.use_batch_norm:
        x = keras.layers.BatchNormalization(name="bn_out")(x)
    aux_output = keras.layers.Dense(cfg.n_eofs, name="spatial_coefficients")(x)
    main_output = RecomposeLayer.from_svd(svd, name="field")(aux_output)

    return keras.Model(inputs=inp, outputs=[main_output, aux_output])
