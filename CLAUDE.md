# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for the paper ["A Novel Framework for Spatio-Temporal Prediction of Environmental Data Using Deep Learning"](https://www.nature.com/articles/s41598-020-79148-7) (Amato, Guignard, Robert, Kanevski). The `stdl` package implements an SVD+deep-learning pipeline for spatio-temporal field interpolation.

## Development setup

```bash
pip install uv            # once
uv sync --all-extras      # installs runtime + dev dependencies
```

## Common commands

```bash
uv run pytest                                   # all tests
uv run pytest tests/test_model.py -v            # single test file
uv run ruff check src/ tests/                   # lint
uv run black src/ tests/                        # format
uv run stdl train configs/simulated.yaml        # run simulated experiment
uv run stdl train configs/simulated.yaml --n-eofs 10 --n-epochs 100  # with overrides
```

## Package structure (`src/stdl/`)

Seven flat modules — no sub-packages:

| Module | Responsibility |
|--------|---------------|
| `config.py` | Pydantic v2 `ExperimentConfig` (loaded from YAML) |
| `data.py` | `load_simulated_data()`, `load_temperature_data()`, `DataBundle` dataclass |
| `decomposition.py` | `svd_decompose(Z, n_eofs)` → `SVDResult` (Wikle 2019 convention) |
| `model.py` | `build_model(cfg, svd)` → Keras model; `RecomposeLayer` custom layer |
| `train.py` | `train(model, bundle, svd, cfg)` → History; `OneCycleScheduler` callback |
| `evaluate.py` | `temporal_mae`, `spatial_mae`, `control_stats` |
| `cli.py` | Typer CLI; entry point `stdl train <config.yaml>` |

## Architecture

The pipeline is always: `load_data` → `svd_decompose` → `build_model` → `train` → `model.predict`.

**SVD decomposition** (`decomposition.py`): `Z` (shape `[n_train+n_val, T]`) is time-demeaned, normalised by `1/sqrt(nS-1)`, and decomposed via `tf.linalg.svd`. `u` (spatial coefficients) and `v` (temporal bases) are the key outputs.

**Model** (`model.py`): Input = spatial coordinates → stacked Dense+BatchNorm blocks → `aux_output` (spatial EOF coefficients, shape `[batch, n_eofs]`) → `RecomposeLayer` that applies the inverse SVD transform to reconstruct the full time series at each point. The `RecomposeLayer` stores the precomputed `diag(s) @ V^T` matrix for efficiency and supports Keras model serialisation via `get_config`.

**Key design constraint**: `SVDResult.u[:n_train]` are the auxiliary targets for training; `u[n_train:]` for validation. The model learns to predict `u` given coordinates, then `RecomposeLayer` reconstructs the field.

## Data

- **Simulated** (included): `data/simulated/SIM1_spatialbasis.txt`, `data/simulated/SIM1_timecoeff.txt`
- **Temperature** (not in repo): place CSV files in `data/temperature/` — see `configs/temperature.yaml` for expected filenames
- Original notebooks remain in `Code/` (pre-refactor reference)

## Configs

`configs/simulated.yaml` and `configs/temperature.yaml` are the canonical experiment configs. Override any field at runtime: `--n-eofs`, `--n-epochs`, `--output-dir`.

## Tests

Smoke tests only — shape checks and no-crash guarantees on tiny synthetic data (10 samples × 20 timesteps, 2 EOFs). Tests hitting the actual data files are marked `skipif` if files are absent. No ground-truth metric assertions (fragile across TF versions).

## TF/Keras version note

TF 2.21 (Keras 3) is installed. `optimizer.lr` no longer exists — use `optimizer.learning_rate.assign(value)`.
