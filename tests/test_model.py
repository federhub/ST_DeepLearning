"""Smoke tests for model construction and training."""

from __future__ import annotations

import numpy as np


def _small_model_cfg(n_eofs: int):
    from stdl.config import ModelConfig

    return ModelConfig(
        n_eofs=n_eofs,
        input_dim=2,
        hidden_sizes=[16, 8],
        activation="relu",
        use_batch_norm=False,
    )


def test_build_model_output_shapes(tiny_svd):
    """Forward pass produces outputs with correct shapes."""
    from stdl.model import build_model

    n_eofs = int(tiny_svd.s.shape[0])
    n_time = int(tiny_svd.v.shape[0])
    model = build_model(_small_model_cfg(n_eofs), tiny_svd)

    batch = np.random.default_rng(0).standard_normal((5, 2)).astype(np.float32)
    main_out, aux_out = model(batch, training=False)

    assert tuple(main_out.shape) == (5, n_time)
    assert tuple(aux_out.shape) == (5, n_eofs)


def test_build_model_with_batch_norm(tiny_svd):
    """Model builds successfully with batch normalisation enabled."""
    from stdl.config import ModelConfig
    from stdl.model import build_model

    n_eofs = int(tiny_svd.s.shape[0])
    cfg = ModelConfig(
        n_eofs=n_eofs,
        input_dim=2,
        hidden_sizes=[16],
        activation="elu",
        use_batch_norm=True,
    )
    model = build_model(cfg, tiny_svd)
    assert model is not None


def test_train_one_epoch(tiny_bundle, tiny_svd):
    """One epoch of training completes and history contains 'loss'."""
    from stdl.config import TrainConfig
    from stdl.model import build_model
    from stdl.train import train

    n_eofs = int(tiny_svd.s.shape[0])
    cfg_train = TrainConfig(
        batch_size=2,
        n_epochs=1,
        patience=1,
        max_lr=1e-3,
        random_seed=0,
    )
    model = build_model(_small_model_cfg(n_eofs), tiny_svd)
    history = train(model, tiny_bundle, tiny_svd, cfg_train)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
