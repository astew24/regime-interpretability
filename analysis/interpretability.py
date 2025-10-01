"""Interpret sparse features by correlating them with external market regime indicators."""

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

from models.sparse_autoencoder import SparseAutoencoder
from train.train_sparse import train_sparse_autoencoder
from utils.helpers import load_checkpoint, load_config
from viz.plots import plot_feature_heatmap, plot_umap_embeddings


def compute_feature_activations(sparse_model: SparseAutoencoder, embeddings) -> pd.DataFrame:
    """Compute sparse feature activation time series for a matrix of latent embeddings.

    Args:
        sparse_model: Trained sparse autoencoder used to transform embeddings into sparse codes.
        embeddings: Embedding matrix as a pandas DataFrame, NumPy array, or torch Tensor.

    Returns:
        DataFrame of sparse feature activations with one column per learned feature.
    """

    if isinstance(embeddings, pd.DataFrame):
        index = pd.DatetimeIndex(embeddings.index)
        array = embeddings.to_numpy(dtype=np.float32)
    elif isinstance(embeddings, torch.Tensor):
        index = pd.RangeIndex(0, embeddings.shape[0])
        array = embeddings.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(embeddings, dtype=np.float32)
        index = pd.RangeIndex(0, array.shape[0])

    device = next(sparse_model.parameters()).device
    sparse_model.eval()
    with torch.no_grad():
        tensor = torch.tensor(array, dtype=torch.float32, device=device)
        _, sparse_code, _ = sparse_model(tensor)

    columns = [f"feature_{idx:03d}" for idx in range(sparse_code.shape[1])]
    return pd.DataFrame(sparse_code.cpu().numpy(), index=index, columns=columns)


def correlate_with_indicators(activations: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """Measure linear correlation between sparse feature activations and external indicators.

    Args:
        activations: Sparse feature activation DataFrame indexed by date.
        indicators: External indicator DataFrame indexed by date.

    Returns:
        Correlation matrix with sparse features in rows and indicators in columns.
    """

    aligned_activations, aligned_indicators = activations.align(indicators, join="inner", axis=0)
    correlations = {
        feature: aligned_indicators.corrwith(aligned_activations[feature]) for feature in aligned_activations.columns
    }
    return pd.DataFrame(correlations).T


def label_interpretable_features(correlations: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Assign descriptive labels to sparse features with strong indicator correlations.

    Args:
        correlations: Correlation matrix between sparse features and external indicators.
        threshold: Minimum absolute correlation required to flag a feature as interpretable.

    Returns:
        DataFrame containing each feature's strongest indicator match and descriptive label.
    """

    records = []
    for feature_name, row in correlations.iterrows():
        abs_row = row.abs()
        top_indicator = abs_row.idxmax()
        top_correlation = row[top_indicator]
        interpretable = bool(abs(top_correlation) >= threshold)
        # The label mapping is intentionally heuristic. It is mostly there to make
        # notebook review easier, not to claim the feature semantics are settled.
        label = _indicator_to_label(indicator=top_indicator, correlation=top_correlation) if interpretable else "unlabeled"
        records.append(
            {
                "feature": feature_name,
                "top_indicator": top_indicator,
                "correlation": float(top_correlation),
                "interpretable": interpretable,
                "label": label,
            }
        )
    return pd.DataFrame(records)


def generate_event_heatmap(activations: pd.DataFrame, event_dates: Dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Aggregate sparse feature activations across named market event windows.

    Args:
        activations: Sparse feature activation DataFrame indexed by date.
        event_dates: Mapping from event name to ``(start_date, end_date)`` tuples.

    Returns:
        DataFrame of average feature activation by event with features in rows.
    """

    activation_frame = activations.copy()
    activation_frame.index = pd.to_datetime(activation_frame.index)
    event_matrix = {}
    for event_name, (start_date, end_date) in event_dates.items():
        mask = (activation_frame.index >= pd.Timestamp(start_date)) & (activation_frame.index <= pd.Timestamp(end_date))
        event_matrix[event_name] = activation_frame.loc[mask].mean(axis=0)
    return pd.DataFrame(event_matrix)


def build_indicator_panel(market_payload: Dict, window_dates, window_size: int) -> pd.DataFrame:
    """Build the external indicator panel aligned to rolling-window sample dates.

    Args:
        market_payload: Serialized market data payload from ``data/download.py``.
        window_dates: Iterable of rolling-window endpoint dates.
        window_size: Rolling lookback used to construct the windowed dataset.

    Returns:
        DataFrame of indicator values aligned to the supplied rolling-window dates.
    """

    prices = pd.DataFrame(
        market_payload["price_values"].numpy(),
        index=pd.to_datetime(market_payload["price_dates"]),
        columns=market_payload["price_columns"],
    )
    returns = pd.DataFrame(
        market_payload["return_values"].numpy(),
        index=pd.to_datetime(market_payload["return_dates"]),
        columns=market_payload["return_columns"],
    )
    external = pd.DataFrame(
        market_payload["indicator_values"].numpy(),
        index=pd.to_datetime(market_payload["indicator_dates"]),
        columns=market_payload["indicator_columns"],
    )

    indicator_panel = external.copy()
    indicator_panel["spy_20d_momentum"] = prices["SPY"].pct_change(window_size)
    indicator_panel["cross_asset_correlation"] = _average_pairwise_correlation(returns=returns, window_size=window_size)

    aligned_dates = pd.to_datetime(window_dates)
    return indicator_panel.reindex(aligned_dates).ffill().dropna()


def run_interpretability(config: Dict) -> Dict[str, str]:
    """Run the full sparse feature interpretability analysis and save outputs to disk.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary of saved artifact paths for the interpretability analysis.
    """

    results_dir = Path(config["paths"]["results_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    sparse_ckpt = checkpoint_dir / "sparse_autoencoder.pt"
    embeddings_path = results_dir / "embeddings.pt"
    market_data_path = Path(config["paths"]["data_dir"]) / "market_data.pt"

    if not sparse_ckpt.exists() or not embeddings_path.exists():
        train_sparse_autoencoder(config)

    sparse_model = load_checkpoint(sparse_ckpt, SparseAutoencoder)
    embeddings_payload = torch.load(embeddings_path, map_location="cpu")
    market_payload = torch.load(market_data_path, map_location="cpu")

    full_dates = pd.to_datetime(embeddings_payload["full"]["dates"])
    full_embeddings = pd.DataFrame(embeddings_payload["full"]["embeddings"].numpy(), index=full_dates)
    indicator_panel = build_indicator_panel(
        market_payload=market_payload,
        window_dates=full_dates,
        window_size=config["data"]["window_size"],
    )
    full_embeddings = full_embeddings.reindex(indicator_panel.index)

    activations = compute_feature_activations(sparse_model=sparse_model, embeddings=full_embeddings)
    correlations = correlate_with_indicators(activations=activations, indicators=indicator_panel)
    feature_labels = label_interpretable_features(correlations=correlations, threshold=0.5)

    activations_path = results_dir / "feature_activations.csv"
    correlations_path = results_dir / "feature_indicator_correlations.csv"
    labels_path = results_dir / "feature_labels.csv"
    heatmap_csv_path = results_dir / "event_heatmap.csv"

    activations.to_csv(activations_path)
    correlations.to_csv(correlations_path)
    feature_labels.to_csv(labels_path, index=False)

    if indicator_panel["vix_level"].nunique() >= 3:
        vix_terciles = pd.qcut(
            indicator_panel["vix_level"],
            q=3,
            labels=["Low VIX", "Mid VIX", "High VIX"],
            duplicates="drop",
        )
    else:
        vix_terciles = pd.Series(["VIX Regime"] * len(indicator_panel), index=indicator_panel.index)
    dominant_features = activations.abs().idxmax(axis=1)
    dominant_feature_labels = dominant_features.map(feature_labels.set_index("feature")["label"]).fillna(dominant_features)
    calendar_years = pd.Series(activations.index.year.astype(str), index=activations.index)

    fig, _ax = plot_umap_embeddings(full_embeddings.values, vix_terciles.astype(str), "UMAP Colored by VIX Tercile")
    fig.savefig(results_dir / "umap_vix_tercile.png", dpi=200, bbox_inches="tight")
    fig.clf()

    fig, _ax = plot_umap_embeddings(
        full_embeddings.values,
        dominant_feature_labels.astype(str),
        "UMAP Colored by Dominant Sparse Feature",
    )
    fig.savefig(results_dir / "umap_dominant_feature.png", dpi=200, bbox_inches="tight")
    fig.clf()

    fig, _ax = plot_umap_embeddings(full_embeddings.values, calendar_years.astype(str), "UMAP Colored by Calendar Year")
    fig.savefig(results_dir / "umap_calendar_year.png", dpi=200, bbox_inches="tight")
    fig.clf()

    event_windows = {
        "COVID Crash": ("2020-02-20", "2020-03-31"),
        "2022 Hiking Cycle": ("2022-03-16", "2022-10-31"),
    }
    event_windows.update(_detect_major_drawdowns(market_payload=market_payload, sample_dates=full_dates))
    event_heatmap = generate_event_heatmap(activations=activations, event_dates=event_windows)
    event_heatmap.to_csv(heatmap_csv_path)

    fig, _ax = plot_feature_heatmap(
        activations=activations,
        event_dates=event_windows,
        feature_labels=feature_labels["label"].tolist(),
    )
    fig.savefig(results_dir / "event_feature_heatmap.png", dpi=200, bbox_inches="tight")
    fig.clf()

    summary = {
        "num_features": int(len(feature_labels)),
        "num_interpretable_features": int(feature_labels["interpretable"].sum()),
        "saved_outputs": [
            str(activations_path),
            str(correlations_path),
            str(labels_path),
            str(heatmap_csv_path),
        ],
    }
    with (results_dir / "interpretability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "activations": str(activations_path),
        "correlations": str(correlations_path),
        "labels": str(labels_path),
        "heatmap": str(heatmap_csv_path),
    }


def _average_pairwise_correlation(returns: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute the rolling mean off-diagonal correlation across the asset universe."""

    values = []
    for end_idx in range(len(returns)):
        if end_idx + 1 < window_size:
            values.append(np.nan)
            continue
        window = returns.iloc[end_idx - window_size + 1 : end_idx + 1]
        corr = window.corr().to_numpy()
        tri_upper = corr[np.triu_indices_from(corr, k=1)]
        values.append(float(np.nanmean(tri_upper)))
    return pd.Series(values, index=returns.index, name="cross_asset_correlation")


def _indicator_to_label(indicator: str, correlation: float) -> str:
    """Map a strongly correlated indicator to a descriptive feature label."""

    positive = correlation >= 0.0
    label_map = {
        "vix_level": ("risk-off", "calm/risk-on"),
        "vix_5d_change": ("volatility-spike", "volatility-compression"),
        "yield_spread": ("yield-curve-steepening", "yield-curve-flattening"),
        "spy_20d_momentum": ("trend-following", "momentum-reversal"),
        "cross_asset_correlation": ("correlation-shock", "dispersion/rotation"),
    }
    positive_label, negative_label = label_map.get(indicator, ("indicator-linked", "indicator-linked"))
    return positive_label if positive else negative_label


def _detect_major_drawdowns(market_payload: Dict, sample_dates) -> Dict[str, tuple[str, str]]:
    """Identify large drawdown periods in the sample range to annotate the event heatmap."""

    prices = pd.DataFrame(
        market_payload["price_values"].numpy(),
        index=pd.to_datetime(market_payload["price_dates"]),
        columns=market_payload["price_columns"],
    )
    sample_index = pd.to_datetime(sample_dates)
    spy = prices["SPY"].reindex(sample_index).ffill()
    drawdown = spy / spy.cummax() - 1.0
    stressed = drawdown <= min(-0.08, drawdown.quantile(0.1))

    events: Dict[str, tuple[str, str]] = {}
    if stressed.sum() == 0:
        return events

    current_group = []
    grouped_periods = []
    for date, is_stressed in stressed.items():
        if is_stressed:
            current_group.append(date)
        elif current_group:
            grouped_periods.append(current_group)
            current_group = []
    if current_group:
        grouped_periods.append(current_group)

    # This keeps only the bigger drawdown blocks. It's a little blunt, but it gives
    # the event heatmap a sane default when I don't want to hand-curate dates.
    for idx, period in enumerate(sorted(grouped_periods, key=len, reverse=True)[:2], start=1):
        events[f"Test Drawdown {idx}"] = (period[0].date().isoformat(), period[-1].date().isoformat())
    return events


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the interpretability analysis entry point."""

    parser = argparse.ArgumentParser(description="Run sparse feature interpretability analysis.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    run_interpretability(loaded_config)
