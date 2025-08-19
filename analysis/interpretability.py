"""Interpret sparse features against known market indicators."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sparse_autoencoder import SparseAutoencoder
from train.train_sparse import train_sparse_autoencoder
from utils.helpers import load_checkpoint, load_config


def compute_feature_activations(sparse_model: SparseAutoencoder, embeddings) -> pd.DataFrame:
    if isinstance(embeddings, pd.DataFrame):
        index = pd.DatetimeIndex(embeddings.index)
        array = embeddings.to_numpy(dtype=np.float32)
    else:
        array = np.asarray(embeddings, dtype=np.float32)
        index = pd.RangeIndex(0, array.shape[0])

    with torch.no_grad():
        tensor = torch.tensor(array, dtype=torch.float32)
        _, sparse_code, _ = sparse_model(tensor)
    columns = [f'feature_{idx:03d}' for idx in range(sparse_code.shape[1])]
    return pd.DataFrame(sparse_code.numpy(), index=index, columns=columns)


def correlate_with_indicators(activations: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    aligned_activations, aligned_indicators = activations.align(indicators, join='inner', axis=0)
    correlations = {
        feature: aligned_indicators.corrwith(aligned_activations[feature])
        for feature in aligned_activations.columns
    }
    return pd.DataFrame(correlations).T


def label_interpretable_features(correlations: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for feature_name, row in correlations.iterrows():
        top_indicator = row.abs().idxmax()
        top_value = row[top_indicator]
        rows.append({
            'feature': feature_name,
            'top_indicator': top_indicator,
            'correlation': float(top_value),
            'interpretable': bool(abs(top_value) >= threshold),
        })
    return pd.DataFrame(rows)


def generate_event_heatmap(activations, event_dates):
    activation_frame = pd.DataFrame(activations).copy()
    activation_frame.index = pd.to_datetime(activation_frame.index)
    event_matrix = {}
    for name, (start, end) in event_dates.items():
        mask = (activation_frame.index >= pd.Timestamp(start)) & (activation_frame.index <= pd.Timestamp(end))
        event_matrix[name] = activation_frame.loc[mask].mean(axis=0)
    return pd.DataFrame(event_matrix)


def build_indicator_panel(market_payload, window_dates, window_size: int) -> pd.DataFrame:
    prices = pd.DataFrame(
        market_payload['price_values'].numpy(),
        index=pd.to_datetime(market_payload['price_dates']),
        columns=market_payload['price_columns'],
    )
    returns = pd.DataFrame(
        market_payload['return_values'].numpy(),
        index=pd.to_datetime(market_payload['return_dates']),
        columns=market_payload['return_columns'],
    )
    external = pd.DataFrame(
        market_payload['indicator_values'].numpy(),
        index=pd.to_datetime(market_payload['indicator_dates']),
        columns=market_payload['indicator_columns'],
    )
    indicator_panel = external.copy()
    indicator_panel['spy_20d_momentum'] = prices['SPY'].pct_change(window_size)
    indicator_panel['cross_asset_correlation'] = returns.rolling(window_size).corr().groupby(level=0).mean().mean(axis=1)
    return indicator_panel.reindex(pd.to_datetime(window_dates)).ffill().dropna()


def run_interpretability(config):
    results_dir = Path(config['paths']['results_dir'])
    sparse_ckpt = Path(config['paths']['checkpoint_dir']) / 'sparse_autoencoder.pt'
    if not sparse_ckpt.exists():
        train_sparse_autoencoder(config)

    sparse_model = load_checkpoint(sparse_ckpt, SparseAutoencoder)
    embeddings_payload = torch.load(results_dir / 'embeddings.pt', map_location='cpu')
    market_payload = torch.load(Path(config['paths']['data_dir']) / 'market_data.pt', map_location='cpu')

    dates = pd.to_datetime(embeddings_payload['full']['dates'])
    embeddings = pd.DataFrame(embeddings_payload['full']['embeddings'].numpy(), index=dates)
    indicators = build_indicator_panel(market_payload, dates, config['data']['window_size'])
    activations = compute_feature_activations(sparse_model, embeddings.reindex(indicators.index))
    correlations = correlate_with_indicators(activations, indicators)
    labels = label_interpretable_features(correlations)

    activations.to_csv(results_dir / 'feature_activations.csv')
    correlations.to_csv(results_dir / 'feature_indicator_correlations.csv')
    labels.to_csv(results_dir / 'feature_labels.csv', index=False)
    return {'activations': str(results_dir / 'feature_activations.csv')}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run sparse feature interpretation.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_interpretability(load_config(args.config))
