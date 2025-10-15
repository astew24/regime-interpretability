"""Run systematic hyperparameter sweeps for the regime embedding and sparse decomposition pipeline."""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest import compute_metrics, regime_conditioned_strategy
from analysis.interpretability import (
    build_indicator_panel,
    compute_feature_activations,
    correlate_with_indicators,
    label_interpretable_features,
)
from data.download import run_data_pipeline
from models.sparse_autoencoder import SparseAutoencoder
from train.train_ae import train_autoencoder
from train.train_sparse import train_sparse_autoencoder
from utils.helpers import load_checkpoint, load_config
from viz.plots import plot_ablation_table


def run_ablation_sweep(config: Dict) -> pd.DataFrame:
    """Run the full ablation grid and save reconstruction, interpretability, and Sharpe results.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        DataFrame containing one row per ablation configuration.
    """

    project_root = Path(__file__).resolve().parents[1]
    ablation_root = Path(config["paths"]["results_dir"]) / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)

    # Full grid can be slow on CPU — consider running on a subset of sparsity/topk
    # values first to identify the responsive region before launching the full sweep.
    results: List[Dict[str, float | int | str]] = []
    num_assets = len(config["data"]["tickers"])

    grid = itertools.product(
        config["ablation"]["latent_dims"],
        config["ablation"]["topk_values"],
        config["ablation"]["dict_sizes"],
        config["ablation"]["window_sizes"],
        config["ablation"]["sparsity_lambdas"],
    )

    for latent_dim, topk, dict_size, window_size, sparsity_lambda in grid:
        run_name = (
            f"latent_{latent_dim}_topk_{topk}_dict_{dict_size}_window_{window_size}_"
            f"lambda_{str(sparsity_lambda).replace('.', 'p')}"
        )
        run_dir = ablation_root / run_name
        run_config = _build_run_config(
            config=config,
            project_root=project_root,
            run_dir=run_dir,
            latent_dim=latent_dim,
            topk=topk,
            dict_size=dict_size,
            window_size=window_size,
            sparsity_lambda=sparsity_lambda,
            input_dim=window_size * num_assets,
        )

        run_data_pipeline(run_config)
        train_autoencoder(run_config)
        train_sparse_autoencoder(run_config)

        market_payload = torch.load(Path(run_config["paths"]["data_dir"]) / "market_data.pt", map_location="cpu")
        embeddings_payload = torch.load(Path(run_config["paths"]["results_dir"]) / "embeddings.pt", map_location="cpu")
        sparse_outputs = torch.load(Path(run_config["paths"]["results_dir"]) / "sparse_outputs.pt", map_location="cpu")
        sparse_model = load_checkpoint(
            Path(run_config["paths"]["checkpoint_dir"]) / "sparse_autoencoder.pt",
            SparseAutoencoder,
        )

        val_dates = pd.to_datetime(embeddings_payload["val"]["dates"])
        val_embeddings = embeddings_payload["val"]["embeddings"].float()
        val_embedding_frame = pd.DataFrame(val_embeddings.numpy(), index=val_dates)

        indicator_panel = build_indicator_panel(
            market_payload=market_payload,
            window_dates=val_dates,
            window_size=window_size,
        )
        indicator_panel = indicator_panel.reindex(val_embedding_frame.index).dropna()
        val_embedding_frame = val_embedding_frame.reindex(indicator_panel.index)

        activations = compute_feature_activations(sparse_model=sparse_model, embeddings=val_embedding_frame)
        correlations = correlate_with_indicators(activations=activations, indicators=indicator_panel)
        feature_labels = label_interpretable_features(correlations=correlations, threshold=0.5)

        returns_frame = pd.DataFrame(
            market_payload["return_values"].numpy(),
            index=pd.to_datetime(market_payload["return_dates"]),
            columns=market_payload["return_columns"],
        )
        val_returns = returns_frame["SPY"].reindex(indicator_panel.index).dropna()
        aligned_activations = activations.reindex(val_returns.index)

        dominant_features = aligned_activations.abs().idxmax(axis=1)
        label_lookup = feature_labels.set_index("feature")["label"].to_dict()
        semantic_labels = dominant_features.map(lambda name: label_lookup.get(name, "unlabeled"))
        strategy_returns = regime_conditioned_strategy(val_returns, semantic_labels)
        metrics = compute_metrics(strategy_returns)

        reconstructions = sparse_outputs["val"]["reconstructions"].float()
        aligned_embeddings = val_embeddings[: reconstructions.shape[0]]
        reconstruction_error = float(F.mse_loss(reconstructions, aligned_embeddings).item())

        row = {
            "latent_dim": latent_dim,
            "topk": topk,
            "dict_size": dict_size,
            "window_size": window_size,
            "sparsity_lambda": sparsity_lambda,
            "reconstruction_error": reconstruction_error,
            "interpretable_features": int(feature_labels["interpretable"].sum()),
            "val_sharpe": float(metrics["sharpe"]),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(ablation_root / "ablation_results.csv", index=False)

    summary_frame = summarize_ablation_results(results)
    summary_frame.to_csv(ablation_root / "ablation_results_sorted.csv", index=False)

    importance_rows = []
    for parameter in ("latent_dim", "topk", "dict_size", "window_size", "sparsity_lambda"):
        grouped = (
            summary_frame.groupby(parameter)[["reconstruction_error", "interpretable_features", "val_sharpe"]]
            .mean()
            .reset_index()
        )
        grouped.insert(0, "parameter", parameter)
        importance_rows.append(grouped)
    importance_frame = pd.concat(importance_rows, ignore_index=True)
    importance_frame.to_csv(ablation_root / "ablation_parameter_importance.csv", index=False)

    fig, _ax = plot_ablation_table(summary_frame)
    fig.savefig(ablation_root / "ablation_table.png", dpi=200, bbox_inches="tight")
    fig.clf()

    return summary_frame


def summarize_ablation_results(results) -> pd.DataFrame:
    """Sort and summarize ablation results for easy downstream inspection.

    Args:
        results: List-like collection or DataFrame of ablation result records.

    Returns:
        DataFrame sorted by validation Sharpe and interpretability score.
    """

    results_frame = pd.DataFrame(results).copy()
    return results_frame.sort_values(
        by=["val_sharpe", "interpretable_features", "reconstruction_error"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_run_config(
    config: Dict,
    project_root: Path,
    run_dir: Path,
    latent_dim: int,
    topk: int,
    dict_size: int,
    window_size: int,
    sparsity_lambda: float,
    input_dim: int,
) -> Dict:
    """Create a deep-copied configuration for a single ablation run."""

    run_config = copy.deepcopy(config)
    run_config["data"]["window_size"] = window_size
    run_config["autoencoder"]["input_dim"] = input_dim
    run_config["autoencoder"]["latent_dim"] = latent_dim
    run_config["sparse_autoencoder"]["input_dim"] = latent_dim
    run_config["sparse_autoencoder"]["topk"] = topk
    run_config["sparse_autoencoder"]["dict_size"] = dict_size
    run_config["sparse_autoencoder"]["sparsity_lambda"] = sparsity_lambda
    run_config["paths"]["data_dir"] = str(project_root / "data" / f"processed_window_{window_size}")
    run_config["paths"]["results_dir"] = str(run_dir)
    run_config["paths"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    return run_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation sweep entry point."""

    parser = argparse.ArgumentParser(description="Run ablation sweeps for the regime detection pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    run_ablation_sweep(loaded_config)
