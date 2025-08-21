"""Plotting utilities for training diagnostics, embeddings, feature analysis, and backtests."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap


def plot_training_curves(train_losses: Iterable[float], val_losses: Iterable[float]):
    """Plot train and validation loss curves across epochs.

    Args:
        train_losses: Sequence of per-epoch training losses.
        val_losses: Sequence of per-epoch validation losses.

    Returns:
        Matplotlib ``(figure, axis)`` tuple containing the rendered loss chart.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(train_losses), label="Train", linewidth=2)
    ax.plot(list(val_losses), label="Validation", linewidth=2)
    ax.set_title("Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_umap_embeddings(embeddings, labels, title: str):
    """Project latent embeddings into two dimensions with UMAP and color by labels.

    Args:
        embeddings: Array-like latent embedding matrix with shape ``(num_samples, latent_dim)``.
        labels: Array-like label vector used to color each point.
        title: Figure title describing the coloring scheme.

    Returns:
        Matplotlib ``(figure, axis)`` tuple containing the scatter plot.
    """

    embedding_array = np.asarray(embeddings)
    label_series = pd.Series(labels, name="label")
    reducer = umap.UMAP(random_state=42)
    projected = reducer.fit_transform(embedding_array)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = label_series.dropna().unique()

    if len(unique_labels) <= 12:
        palette = sns.color_palette("tab20", n_colors=max(len(unique_labels), 1))
        color_map = {label: palette[idx] for idx, label in enumerate(unique_labels)}
        colors = label_series.map(color_map).apply(
            lambda value: value if isinstance(value, tuple) else (0.7, 0.7, 0.7)
        )
        ax.scatter(projected[:, 0], projected[:, 1], c=list(colors), s=18, alpha=0.85)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=str(label))
            for label in unique_labels
        ]
        ax.legend(handles=handles, loc="best", frameon=True)
    else:
        numeric_codes = pd.factorize(label_series.fillna("missing"))[0]
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=numeric_codes, cmap="viridis", s=18, alpha=0.85)
        fig.colorbar(scatter, ax=ax, label="Label Code")

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    return fig, ax


def plot_feature_heatmap(activations, event_dates, feature_labels):
    """Plot mean sparse feature activations across named market event windows.

    Args:
        activations: Activation DataFrame indexed by date with sparse features in columns.
        event_dates: Mapping from event name to ``(start_date, end_date)`` tuples.
        feature_labels: Optional sequence of human-readable labels for feature rows.

    Returns:
        Matplotlib ``(figure, axis)`` tuple containing the event-by-feature heatmap.
    """

    activation_frame = pd.DataFrame(activations).copy()
    activation_frame.index = pd.to_datetime(activation_frame.index)
    event_matrix = {}
    for event_name, (start_date, end_date) in event_dates.items():
        mask = (activation_frame.index >= pd.Timestamp(start_date)) & (activation_frame.index <= pd.Timestamp(end_date))
        event_matrix[event_name] = activation_frame.loc[mask].mean(axis=0)

    heatmap_data = pd.DataFrame(event_matrix)
    if feature_labels is not None and len(feature_labels) == heatmap_data.shape[0]:
        heatmap_data.index = list(feature_labels)

    fig_height = max(6, int(heatmap_data.shape[0] * 0.18))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(heatmap_data, cmap="mako", center=0.0, ax=ax)
    ax.set_title("Sparse Feature Activation by Event")
    ax.set_xlabel("Event")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig, ax


def plot_equity_curve(strategy_returns, benchmark_returns):
    """Plot cumulative equity curves for a strategy and benchmark return stream.

    Args:
        strategy_returns: Daily strategy return series.
        benchmark_returns: Daily benchmark return series.

    Returns:
        Matplotlib ``(figure, axis)`` tuple containing the cumulative return comparison.
    """

    strategy = pd.Series(strategy_returns).fillna(0.0)
    benchmark = pd.Series(benchmark_returns).fillna(0.0)

    strategy_curve = (1.0 + strategy).cumprod()
    benchmark_curve = (1.0 + benchmark).cumprod()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strategy_curve.index, strategy_curve.values, label="Strategy", linewidth=2)
    ax.plot(benchmark_curve.index, benchmark_curve.values, label="Benchmark", linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_ablation_table(results):
    """Render the leading ablation configurations as a compact matplotlib table.

    Args:
        results: DataFrame or list-like collection of ablation result records.

    Returns:
        Matplotlib ``(figure, axis)`` tuple containing the rendered results table.
    """

    results_frame = pd.DataFrame(results).copy()
    if "val_sharpe" in results_frame.columns:
        results_frame = results_frame.sort_values("val_sharpe", ascending=False)
    display_frame = results_frame.head(20).round(4)

    fig_height = max(4, 0.45 * (len(display_frame) + 1))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title("Ablation Results", pad=16)
    fig.tight_layout()
    return fig, ax
