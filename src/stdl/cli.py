"""Typer CLI entry point for stdl experiments."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="stdl",
    help="Spatio-Temporal Deep Learning — train and evaluate models from YAML configs.",
    no_args_is_help=True,
)


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML config."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override output directory from config."
    ),
    n_eofs: int | None = typer.Option(None, "--n-eofs", help="Override number of EOFs."),
    n_epochs: int | None = typer.Option(
        None, "--n-epochs", help="Override maximum training epochs."
    ),
) -> None:
    """Train a spatio-temporal model from a YAML config file."""
    # Lazy imports keep --help fast even without TF loaded
    from stdl.config import ExperimentConfig
    from stdl.data import load_data
    from stdl.decomposition import svd_decompose
    from stdl.evaluate import control_stats, spatial_mae, temporal_mae
    from stdl.model import build_model
    from stdl.train import train as _train

    import numpy as np

    cfg = ExperimentConfig.from_yaml(config_path)
    if output_dir is not None:
        cfg.output_dir = output_dir
    if n_eofs is not None:
        cfg.model.n_eofs = n_eofs
    if n_epochs is not None:
        cfg.train.n_epochs = n_epochs

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[stdl] Experiment: {cfg.experiment_name}")
    typer.echo(f"[stdl] Loading data ({cfg.data.dataset})…")
    bundle = load_data(cfg.data)

    typer.echo(f"[stdl] SVD decomposition (n_eofs={cfg.model.n_eofs})…")
    svd = svd_decompose(bundle.y_svd, cfg.model.n_eofs)

    typer.echo("[stdl] Building model…")
    model = build_model(cfg.model, svd)
    model.summary()

    typer.echo("[stdl] Training…")
    _train(model, bundle, svd, cfg.train)

    typer.echo("[stdl] Evaluating on test set…")
    y_hat_test, _ = model.predict(bundle.X_test, verbose=0)
    y_hat_mesh, _ = model.predict(bundle.X_mesh, verbose=0)

    mae_t = temporal_mae(bundle.y_test, y_hat_test)
    mae_s = spatial_mae(bundle.y_test, y_hat_test)
    typer.echo(f"[stdl] Mean temporal MAE : {mae_t.mean():.4f}")
    typer.echo(f"[stdl] Mean spatial MAE  : {mae_s.mean():.4f}")

    np.save(cfg.output_dir / "y_hat_test.npy", y_hat_test)
    np.save(cfg.output_dir / "y_hat_mesh.npy", y_hat_mesh)

    stats = control_stats(bundle.y_train, y_hat_mesh)
    stats.to_csv(cfg.output_dir / "control_stats.csv", index=False)

    out_path = cfg.output_dir / f"{cfg.experiment_name}_model.keras"
    model.save(str(out_path))
    typer.echo(f"[stdl] Model saved → {out_path}")


@app.command()
def evaluate(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML config."),
    model_path: Path = typer.Argument(..., help="Path to a saved .keras model file."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override output directory from config."
    ),
) -> None:
    """Evaluate a saved model on the test set without retraining."""
    import numpy as np
    from tensorflow import keras

    from stdl.config import ExperimentConfig
    from stdl.data import load_data
    from stdl.evaluate import control_stats, spatial_mae, temporal_mae
    from stdl.model import RecomposeLayer

    cfg = ExperimentConfig.from_yaml(config_path)
    if output_dir is not None:
        cfg.output_dir = output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[stdl] Loading data ({cfg.data.dataset})…")
    bundle = load_data(cfg.data)

    typer.echo(f"[stdl] Loading model from {model_path}…")
    model = keras.models.load_model(
        str(model_path), custom_objects={"RecomposeLayer": RecomposeLayer}
    )

    typer.echo("[stdl] Evaluating…")
    y_hat_test, _ = model.predict(bundle.X_test, verbose=0)
    y_hat_mesh, _ = model.predict(bundle.X_mesh, verbose=0)

    mae_t = temporal_mae(bundle.y_test, y_hat_test)
    mae_s = spatial_mae(bundle.y_test, y_hat_test)
    typer.echo(f"[stdl] Mean temporal MAE : {mae_t.mean():.4f}")
    typer.echo(f"[stdl] Mean spatial MAE  : {mae_s.mean():.4f}")

    np.save(cfg.output_dir / "y_hat_test.npy", y_hat_test)
    np.save(cfg.output_dir / "y_hat_mesh.npy", y_hat_mesh)

    stats = control_stats(bundle.y_train, y_hat_mesh)
    stats.to_csv(cfg.output_dir / "control_stats.csv", index=False)
    typer.echo(f"[stdl] Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    app()
