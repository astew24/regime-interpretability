"""Baseline regime detection methods and transition timing comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import run_data_pipeline
from train.train_sparse import train_sparse_autoencoder
from utils.helpers import load_config


def fit_hmm(data, n_states: int):
    """Fit a Gaussian hidden Markov model and infer latent state labels.

    Args:
        data: Array-like input matrix with shape ``(num_samples, num_features)``.
        n_states: Number of hidden states to fit.

    Returns:
        Tuple ``(model, labels)`` containing the fitted HMM and inferred state sequence.
    """

    array = np.asarray(data, dtype=np.float64)
    # GaussianHMM can be a little temperamental on these windows, so keeping the
    # covariance simple has been more stable than a fully flexible setup.
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=250, random_state=42)
    model.fit(array)
    labels = model.predict(array)
    return model, labels


def fit_kmeans(data, n_clusters: int):
    """Fit PCA followed by K-means clustering for regime assignment.

    Args:
        data: Array-like input matrix with shape ``(num_samples, num_features)``.
        n_clusters: Number of clusters to fit.

    Returns:
        Tuple ``(pipeline, labels)`` where the pipeline stores PCA and K-means objects.
    """

    array = np.asarray(data, dtype=np.float64)
    # This is a rough PCA cutoff, not something deeply tuned.
    n_components = min(32, array.shape[0], array.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(array)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = model.fit_predict(reduced)
    return {"pca": pca, "kmeans": model}, labels


def detect_regime_transitions(labels) -> Sequence[pd.Timestamp] | Sequence[int]:
    """Identify sample indices or dates where the inferred regime label changes.

    Args:
        labels: Array-like sequence or pandas Series of regime assignments.

    Returns:
        Sequence of transition positions preserving datetime indices when available.
    """

    if isinstance(labels, pd.Series):
        shifted = labels.shift(1)
        transition_mask = labels.ne(shifted)
        return list(labels.index[transition_mask.fillna(False)][1:])

    array = np.asarray(labels)
    return [idx for idx in range(1, len(array)) if array[idx] != array[idx - 1]]


def compare_transition_timing(method_transitions, vix_events) -> Dict[str, float]:
    """Measure the lead or lag of detected regime changes relative to VIX spike events.

    Args:
        method_transitions: Sequence of transition dates or positions produced by a regime model.
        vix_events: Sequence of VIX spike event dates.

    Returns:
        Dictionary of summary timing statistics measured in days.
    """

    method_dates = pd.to_datetime(list(method_transitions))
    event_dates = pd.to_datetime(list(vix_events))
    if len(method_dates) == 0 or len(event_dates) == 0:
        return {
            "avg_lead_lag_days": float("nan"),
            "median_lead_lag_days": float("nan"),
            "num_events": int(len(event_dates)),
            "detections_before_event": 0,
        }

    deltas = []
    for event_date in event_dates:
        candidate_days = np.array([(transition - event_date).days for transition in method_dates], dtype=float)
        deltas.append(float(candidate_days[np.argmin(np.abs(candidate_days))]))

    delta_array = np.asarray(deltas, dtype=float)
    return {
        "avg_lead_lag_days": float(delta_array.mean()),
        "median_lead_lag_days": float(np.median(delta_array)),
        "num_events": int(len(event_dates)),
        "detections_before_event": int((delta_array < 0).sum()),
    }


def run_baseline_suite(config: Dict) -> Dict[str, str]:
    """Fit HMM and K-means baselines, detect transitions, and save comparison outputs.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary mapping saved baseline artifact names to output paths.
    """

    data_dir = Path(config["paths"]["data_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    window_data_path = data_dir / "window_data.pt"
    train_split_path = data_dir / "train.pt"
    market_data_path = data_dir / "market_data.pt"
    sparse_outputs_path = results_dir / "sparse_outputs.pt"

    if not window_data_path.exists() or not train_split_path.exists() or not market_data_path.exists():
        run_data_pipeline(config)
    if not sparse_outputs_path.exists():
        train_sparse_autoencoder(config)

    window_payload = torch.load(window_data_path, map_location="cpu")
    train_payload = torch.load(train_split_path, map_location="cpu")
    market_payload = torch.load(market_data_path, map_location="cpu")
    sparse_outputs = torch.load(sparse_outputs_path, map_location="cpu")

    full_windows = window_payload["windows"].numpy()
    train_windows = train_payload["windows"].numpy()
    sample_dates = pd.to_datetime(window_payload["dates"])
    sparse_dominant = sparse_outputs["full"]["dominant_features"].numpy()

    indicator_frame = pd.DataFrame(
        market_payload["indicator_values"].numpy(),
        index=pd.to_datetime(market_payload["indicator_dates"]),
        columns=market_payload["indicator_columns"],
    )
    vix_series = indicator_frame["vix_level"].reindex(sample_dates).ffill().dropna()
    vix_threshold = vix_series.quantile(0.8)
    vix_spike_mask = vix_series > vix_threshold
    vix_events = vix_series.index[vix_spike_mask & ~vix_spike_mask.shift(1, fill_value=False)]

    label_payload: Dict[str, object] = {
        "dates": [timestamp.isoformat() for timestamp in sample_dates],
        "sparse_dominant_features": torch.tensor(sparse_dominant),
        "vix_events": [timestamp.isoformat() for timestamp in vix_events],
    }
    summary_rows = []

    sparse_transitions = detect_regime_transitions(pd.Series(sparse_dominant, index=sample_dates))
    sparse_timing = compare_transition_timing(method_transitions=sparse_transitions, vix_events=vix_events)
    summary_rows.append({"method": "Sparse Autoencoder", "config": "dominant_feature", **sparse_timing})

    for n_states in (3, 4, 5):
        hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=250, random_state=42)
        hmm_model.fit(train_windows)
        full_labels = hmm_model.predict(full_windows)
        label_payload[f"hmm_{n_states}"] = torch.tensor(full_labels)
        transitions = detect_regime_transitions(pd.Series(full_labels, index=sample_dates))
        timing = compare_transition_timing(method_transitions=transitions, vix_events=vix_events)
        summary_rows.append({"method": "Gaussian HMM", "config": f"{n_states}_states", **timing})

    for n_clusters in (3, 4, 5):
        n_components = min(32, train_windows.shape[0], train_windows.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        train_reduced = pca.fit_transform(train_windows)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(train_reduced)
        full_labels = kmeans.predict(pca.transform(full_windows))
        label_payload[f"kmeans_{n_clusters}"] = torch.tensor(full_labels)
        transitions = detect_regime_transitions(pd.Series(full_labels, index=sample_dates))
        timing = compare_transition_timing(method_transitions=transitions, vix_events=vix_events)
        summary_rows.append({"method": "KMeans", "config": f"{n_clusters}_clusters", **timing})

    labels_path = results_dir / "baseline_labels.pt"
    summary_path = results_dir / "baseline_transition_summary.csv"
    torch.save(label_payload, labels_path)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    with (results_dir / "baseline_transition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    return {"labels": str(labels_path), "summary": str(summary_path)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline comparison entry point."""

    parser = argparse.ArgumentParser(description="Fit baseline regime detection models.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    run_baseline_suite(loaded_config)
