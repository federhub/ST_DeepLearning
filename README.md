# stdl — Spatio-Temporal Deep Learning

Code for the paper:

> **A Novel Framework for Spatio-Temporal Prediction of Environmental Data Using Deep Learning**  
> Federico Amato, Fabian Guignard, Sylvain Robert, Mikhail Kanevski  
> *Scientific Reports*, 2020. [Read the paper →](https://www.nature.com/articles/s41598-020-79148-7)

---

## Overview

`stdl` decomposes a spatio-temporal field into temporal basis functions (via SVD/EOF analysis) and spatial coefficients, then trains a deep feedforward neural network to map spatial coordinates to those coefficients. Predictions at new locations reconstruct the full time series via the learned EOF expansion.

Two experiments are provided:

| Experiment | Data | Config |
|------------|------|--------|
| Simulated  | Gaussian random fields + AR(1) time series (included) | `configs/simulated.yaml` |
| Temperature | Hourly Swiss station data 2016–2018 (not included) | `configs/temperature.yaml` |

---

## Installation

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
pip install uv          # install uv if needed
git clone <repo-url>
cd ST_DeepLearning
uv sync                 # installs all runtime dependencies
uv sync --all-extras    # also installs dev tools (pytest, black, ruff, jupyter)
```

---

## Running experiments

### CLI

```bash
# Train on simulated data (data files are included)
uv run stdl train configs/simulated.yaml

# Override specific hyperparameters at runtime
uv run stdl train configs/simulated.yaml --n-eofs 10 --n-epochs 500

# Temperature experiment (requires local data — see below)
uv run stdl train configs/temperature.yaml --output-dir outputs/temp_run
```

Outputs written to `output_dir` from config:

| File | Contents |
|------|----------|
| `y_hat_test.npy` | Predictions at test locations `[n_test, T]` |
| `y_hat_mesh.npy` | Predictions on full prediction grid `[n_mesh, T]` |
| `control_stats.csv` | Per-timestep distributional statistics |
| `<name>_model.keras` | Saved Keras model |

### Python API

```python
from pathlib import Path
from stdl.config import ExperimentConfig
from stdl.data import load_data
from stdl.decomposition import svd_decompose
from stdl.model import build_model
from stdl.train import train
from stdl.evaluate import temporal_mae

cfg = ExperimentConfig.from_yaml(Path("configs/simulated.yaml"))
bundle = load_data(cfg.data)
svd    = svd_decompose(bundle.y_svd, cfg.model.n_eofs)
model  = build_model(cfg.model, svd)
train(model, bundle, svd, cfg.train)

y_hat_test, _ = model.predict(bundle.X_test)
print(f"Mean temporal MAE: {temporal_mae(bundle.y_test, y_hat_test).mean():.4f}")
```

---

## Reproducing paper experiments

### Simulated data

Data files are included under `data/simulated/`. Run end-to-end:

```bash
uv run stdl train configs/simulated.yaml
```

Then open `notebooks/experiment_simulated.ipynb` to reproduce the paper figures.

### Temperature data

The temperature dataset is not distributed with this repository. Contact the authors to access the data and place them under `data/temperature/`:

```
Temperature_f1Hour_20162018_FilledData_train.txt
Temperature_f1Hour_20162018_CleanedData_test.txt
Temperature_Stn_1Hour_20162018_train.txt
Temperature_Stn_1Hour_20162018_test.txt
DEM_2500_Stnid.csv
```

Then:

```bash
uv run stdl train configs/temperature.yaml
```

---

## Development

```bash
uv run pytest               # run tests
uv run ruff check src/ tests/   # lint
uv run black src/ tests/        # format
```

---

## Repository structure

```
stdl/
├── src/stdl/            # installable package
│   ├── config.py        # Pydantic experiment config (YAML ↔ dataclasses)
│   ├── data.py          # data loading: simulated + temperature
│   ├── decomposition.py # SVD/EOF analysis (Wikle 2019 convention)
│   ├── model.py         # Keras model + custom RecomposeLayer
│   ├── train.py         # training loop + OneCycleScheduler callback
│   ├── evaluate.py      # temporal/spatial MAE + control_stats
│   └── cli.py           # Typer CLI entry point (stdl train …)
├── configs/             # per-experiment YAML configs
├── data/simulated/      # included simulated data files
├── data/temperature/    # NOT in repo — add locally
├── notebooks/           # thin demo/analysis notebooks
└── tests/               # pytest smoke tests
```
