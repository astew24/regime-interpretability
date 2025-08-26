"""Baseline regime detection methods and transition timing checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import run_data_pipeline
from train.train_sparse import train_sparse_autoencoder
from utils.helpers import load_config


def fit_hmm(data, n_states: int):
    array = np.asarray(data, dtype=np.float64)
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=250, random_state=42)
    model.fit(array)
    labels = model.predict(array)
    return model, labels


def fit_kmeans(data, n_clusters: int):
    raise NotImplementedError('KMeans baseline comes next.')


def detect_regime_transitions(labels):
    if isinstance(labels, pd.Series):
        shifted = labels.shift(1)
        mask = labels.ne(shifted)
        return list(labels.index[mask.fillna(False)][1:])
    array = np.asarray(labels)
    return [idx for idx in range(1, len(array)) if array[idx] != array[idx - 1]]


def compare_transition_timing(method_transitions, vix_events):
    method_dates = pd.to_datetime(list(method_transitions))
    event_dates = pd.to_datetime(list(vix_events))
    if len(method_dates) == 0 or len(event_dates) == 0:
        return {'avg_lead_lag_days': float('nan')}
    deltas = []
    for event_date in event_dates:
        candidate_days = np.array([(transition - event_date).days for transition in method_dates], dtype=float)
        deltas.append(float(candidate_days[np.argmin(np.abs(candidate_days))]))
    return {'avg_lead_lag_days': float(np.mean(deltas))}


def run_baseline_suite(config):
    data_dir = Path(config['paths']['data_dir'])
    results_dir = Path(config['paths']['results_dir'])
    if not (data_dir / 'window_data.pt').exists():
        run_data_pipeline(config)
    if not (results_dir / 'sparse_outputs.pt').exists():
        train_sparse_autoencoder(config)

    window_payload = torch.load(data_dir / 'window_data.pt', map_location='cpu')
    market_payload = torch.load(data_dir / 'market_data.pt', map_location='cpu')
    sample_dates = pd.to_datetime(window_payload['dates'])
    indicator_frame = pd.DataFrame(
        market_payload['indicator_values'].numpy(),
        index=pd.to_datetime(market_payload['indicator_dates']),
        columns=market_payload['indicator_columns'],
    )
    vix_series = indicator_frame['vix_level'].reindex(sample_dates).ffill().dropna()
    threshold = vix_series.quantile(0.8)
    vix_events = vix_series.index[(vix_series > threshold) & ~(vix_series > threshold).shift(1, fill_value=False)]

    rows = []
    label_payload = {'dates': [timestamp.isoformat() for timestamp in sample_dates]}
    full_windows = window_payload['windows'].numpy()
    for n_states in (3, 4, 5):
        model, labels = fit_hmm(full_windows, n_states=n_states)
        label_payload[f'hmm_{n_states}'] = torch.tensor(labels)
        transitions = detect_regime_transitions(pd.Series(labels, index=sample_dates))
        timing = compare_transition_timing(transitions, vix_events)
        rows.append({'method': 'Gaussian HMM', 'config': f'{n_states}_states', **timing})

    torch.save(label_payload, results_dir / 'baseline_labels.pt')
    pd.DataFrame(rows).to_csv(results_dir / 'baseline_transition_summary.csv', index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run regime baselines.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_baseline_suite(load_config(args.config))
