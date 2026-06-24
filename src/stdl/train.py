"""Training loop and learning rate scheduling."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from stdl.config import TrainConfig
from stdl.data import DataBundle
from stdl.decomposition import SVDResult


class OneCycleScheduler(keras.callbacks.Callback):
    """1-cycle learning rate policy (Smith, 2018).

    Linearly warms up LR from start_rate to max_rate over the first half of
    training, then linearly cools down back to start_rate, and finally decays
    to last_rate over the final phase.
    """

    def __init__(
        self,
        iterations: int,
        max_rate: float,
        start_rate: float | None = None,
        last_iterations: int | None = None,
        last_rate: float | None = None,
    ) -> None:
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1: int, iter2: int, rate1: float, rate2: float) -> float:
        return (rate2 - rate1) * (self.iteration - iter1) / (iter2 - iter1) + rate1

    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        if self.iteration < self.half_iteration:
            rate = self._interpolate(
                0, self.half_iteration, self.start_rate, self.max_rate
            )
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(
                self.half_iteration,
                2 * self.half_iteration,
                self.max_rate,
                self.start_rate,
            )
        else:
            rate = self._interpolate(
                2 * self.half_iteration,
                self.iterations,
                self.start_rate,
                self.last_rate,
            )
            rate = max(rate, self.last_rate)
        self.iteration += 1
        self.model.optimizer.learning_rate.assign(rate)


def train(
    model: keras.Model,
    bundle: DataBundle,
    svd: SVDResult,
    cfg: TrainConfig,
) -> keras.callbacks.History:
    """Compile and train the model.

    The auxiliary output (spatial coefficients) is included in model.fit with
    zero loss weight by default, matching the paper's training setup.

    Args:
        model: Keras model returned by build_model().
        bundle: Data bundle with scaled train/val arrays.
        svd: SVD result; u[:n_train] and u[n_train:] are used as auxiliary targets.
        cfg: Training hyperparameters.

    Returns:
        Keras History object.
    """
    tf.random.set_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    n_train = bundle.X_train.shape[0]
    u_train = np.array(svd.u[:n_train])
    u_val = np.array(svd.u[n_train:])

    n_steps = (n_train // cfg.batch_size) * cfg.n_epochs
    callbacks = [
        OneCycleScheduler(n_steps, max_rate=cfg.max_lr),
        keras.callbacks.EarlyStopping(
            patience=cfg.patience,
            restore_best_weights=True,
        ),
    ]

    model.compile(
        loss=["mae", "mae"],
        loss_weights=[cfg.loss_weight_main, cfg.loss_weight_aux],
        optimizer=cfg.optimizer,
    )

    return model.fit(
        bundle.X_train,
        [bundle.y_train, u_train],
        epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        validation_data=(bundle.X_val, [bundle.y_val, u_val]),
        callbacks=callbacks,
        verbose=1,
    )
