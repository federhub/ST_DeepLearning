"""Pydantic v2 configuration models for stdl experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    dataset: Literal["simulated", "temperature"] = "simulated"

    # --- Simulated dataset ---
    spatial_basis_path: Path | None = None
    time_coeff_path: Path | None = None
    n_spatial_samples: int = 7500
    n_timesteps: int = 1008
    train_val_size: int = 3000
    test_size: int = 1000
    noise_fraction: float = 0.1
    random_state: int = 42

    # --- Temperature dataset ---
    y_train_path: Path | None = None
    y_test_path: Path | None = None
    X_train_path: Path | None = None
    X_test_path: Path | None = None
    mesh_path: Path | None = None

    # Shared: index within train_val_size that separates train from val
    val_split_idx: int = 2000


class ModelConfig(BaseModel):
    n_eofs: int = 19
    input_dim: int = 2
    hidden_sizes: list[int] = [300, 100, 100, 100, 100]
    activation: Literal["elu", "relu", "tanh"] = "elu"
    use_batch_norm: bool = True


class TrainConfig(BaseModel):
    batch_size: int = 32
    n_epochs: int = 2000
    patience: int = 500
    max_lr: float = 0.005
    optimizer: Literal["nadam", "adam", "sgd"] = "nadam"
    loss_weight_main: float = 1.0
    loss_weight_aux: float = 0.0
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    experiment_name: str
    output_dir: Path = Path("outputs")
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        path = Path(path).resolve()
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls.model_validate(raw)
        # Resolve relative data paths against the config file's directory so
        # the config works regardless of where the process (notebook, CLI) runs.
        base = path.parent
        d = cfg.data
        for field in (
            "spatial_basis_path", "time_coeff_path",
            "y_train_path", "y_test_path", "X_train_path", "X_test_path", "mesh_path",
        ):
            val = getattr(d, field)
            if val is not None and not val.is_absolute():
                setattr(d, field, (base / val).resolve())
        if not cfg.output_dir.is_absolute():
            cfg.output_dir = (base / cfg.output_dir).resolve()
        return cfg
