"""Backtest regime-conditioned SPY strategies against learned and baseline regime labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.baselines import run_baseline_suite
from analysis.interpretability import run_interpretability
from utils.helpers import load_config
from viz.plots import plot_equity_curve


def regime_conditioned_strategy(returns, regime_labels):
    """Convert regime labels into positions and produce the resulting daily strategy returns.

    Args:
        returns: Daily return series for the traded asset.
        regime_labels: Semantic regime labels or explicit position values aligned to ``returns``.

    Returns:
        Series of daily strategy returns after applying a one-day signal lag.
    """

    returns_series = pd.Series(returns).astype(float).copy()
    labels_series = pd.Series(regime_labels, index=getattr(regime_labels, "index", returns_series.index))
    labels_series = labels_series.reindex(returns_series.index)

    if pd.api.types.is_numeric_dtype(labels_series) and set(labels_series.dropna().unique()).issubset({-1, 0, 1}):
        positions = labels_series.astype(float)
    else:
        positions = labels_series.apply(_label_to_position).astype(float)

    # Keeping this intentionally simple for now: one-day lag, no costs, no leverage cap.
    lagged_positions = positions.shift(1).fillna(0.0)
    return lagged_positions * returns_series


def compute_metrics(strategy_returns) -> Dict[str, float]:
    """Compute core performance statistics for a daily return series.

    Args:
        strategy_returns: Daily strategy return series.

    Returns:
        Dictionary containing cumulative return, annualized Sharpe, max drawdown, and win rate.
    """

    returns = pd.Series(strategy_returns).fillna(0.0)
    cumulative_curve = (1.0 + returns).cumprod()
    drawdown = cumulative_curve / cumulative_curve.cummax() - 1.0
    volatility = returns.std(ddof=0)

    sharpe = float(np.sqrt(252.0) * returns.mean() / volatility) if volatility > 1e-12 else float("nan")
    return {
        "cumulative_return": float(cumulative_curve.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0.0).mean()),
    }


def run_backtest(config: Dict) -> pd.DataFrame:
    """Run sparse, HMM, and K-means regime-conditioned SPY backtests on the test split.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        DataFrame of backtest performance metrics by method on the held-out test split.
    """

    results_dir = Path(config["paths"]["results_dir"])
    data_dir = Path(config["paths"]["data_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_labels_path = results_dir / "feature_labels.csv"
    sparse_outputs_path = results_dir / "sparse_outputs.pt"
    baseline_labels_path = results_dir / "baseline_labels.pt"
    market_data_path = data_dir / "market_data.pt"

    if not feature_labels_path.exists() or not sparse_outputs_path.exists():
        run_interpretability(config)
    if not baseline_labels_path.exists():
        run_baseline_suite(config)

    feature_labels = pd.read_csv(feature_labels_path)
    sparse_outputs = torch.load(sparse_outputs_path, map_location="cpu")
    baseline_labels = torch.load(baseline_labels_path, map_location="cpu")
    market_payload = torch.load(market_data_path, map_location="cpu")

    returns_frame = pd.DataFrame(
        market_payload["return_values"].numpy(),
        index=pd.to_datetime(market_payload["return_dates"]),
        columns=market_payload["return_columns"],
    )
    spy_returns = returns_frame["SPY"]

    val_dates = pd.to_datetime(sparse_outputs["val"]["dates"])
    test_dates = pd.to_datetime(sparse_outputs["test"]["dates"])
    val_spy_returns = spy_returns.reindex(val_dates).dropna()
    test_spy_returns = spy_returns.reindex(test_dates).dropna()

    metrics_rows = []

    benchmark_metrics = compute_metrics(test_spy_returns)
    metrics_rows.append({"method": "Buy and Hold SPY", **benchmark_metrics})

    label_lookup = feature_labels.set_index("feature")["label"].to_dict()
    sparse_test_features = pd.Series(sparse_outputs["test"]["dominant_features"].numpy(), index=test_dates)
    sparse_test_labels = sparse_test_features.map(lambda idx: label_lookup.get(f"feature_{int(idx):03d}", "unlabeled"))
    sparse_strategy_returns = regime_conditioned_strategy(test_spy_returns, sparse_test_labels.reindex(test_spy_returns.index))
    metrics_rows.append({"method": "Sparse Autoencoder", **compute_metrics(sparse_strategy_returns)})

    fig, _ax = plot_equity_curve(sparse_strategy_returns, test_spy_returns)
    fig.savefig(results_dir / "equity_curve_sparse_vs_buyhold.png", dpi=200, bbox_inches="tight")
    fig.clf()

    baseline_dates = pd.to_datetime(baseline_labels["dates"])
    for method_name, tensor in baseline_labels.items():
        if method_name in {"dates", "sparse_dominant_features", "vix_events"}:
            continue

        full_labels = pd.Series(tensor.numpy(), index=baseline_dates)
        val_labels = full_labels.reindex(val_spy_returns.index)
        test_labels = full_labels.reindex(test_spy_returns.index)
        position_map = _infer_numeric_state_positions(val_labels=val_labels, returns=val_spy_returns)
        test_positions = test_labels.map(position_map).fillna(0.0)
        strategy_returns = regime_conditioned_strategy(test_spy_returns, test_positions.reindex(test_spy_returns.index))

        metrics_rows.append({"method": method_name, **compute_metrics(strategy_returns)})

        fig, _ax = plot_equity_curve(strategy_returns, test_spy_returns)
        fig.savefig(results_dir / f"equity_curve_{method_name}.png", dpi=200, bbox_inches="tight")
        fig.clf()

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)
    metrics_frame.to_csv(results_dir / "backtest_metrics.csv", index=False)

    with (results_dir / "backtest_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_rows, handle, indent=2)

    return metrics_frame


def _label_to_position(label) -> float:
    """Map a semantic regime label to a tradable position signal."""

    if pd.isna(label):
        return 0.0

    text = str(label).lower()
    risk_on_keywords = ("risk-on", "calm", "trend-following")
    risk_off_keywords = ("risk-off", "volatility-spike", "correlation-shock", "momentum-reversal", "flattening")

    if any(keyword in text for keyword in risk_on_keywords):
        return 1.0
    if any(keyword in text for keyword in risk_off_keywords):
        return -1.0
    return 0.0


def _infer_numeric_state_positions(val_labels: pd.Series, returns: pd.Series) -> Dict[float, float]:
    """Infer long, short, or flat mappings for numeric regime states using validation returns."""

    aligned = pd.concat(
        [
            pd.Series(val_labels, name="label"),
            pd.Series(returns, name="return"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {}

    mean_returns = aligned.groupby("label")["return"].mean().sort_values()
    position_map = {state: 0.0 for state in mean_returns.index}
    best_state = mean_returns.index[-1]
    worst_state = mean_returns.index[0]

    if mean_returns.loc[best_state] > 0.0:
        position_map[best_state] = 1.0
    if mean_returns.loc[worst_state] < 0.0 and worst_state != best_state:
        position_map[worst_state] = -1.0
    return position_map


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the backtest entry point."""

    parser = argparse.ArgumentParser(description="Run regime-conditioned portfolio backtests.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    run_backtest(loaded_config)
