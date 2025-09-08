"""Backtest regime-conditioned SPY strategies."""

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

from analysis.interpretability import run_interpretability
from utils.helpers import load_config


def regime_conditioned_strategy(returns, regime_labels):
    returns_series = pd.Series(returns).astype(float).copy()
    labels_series = pd.Series(regime_labels, index=getattr(regime_labels, 'index', returns_series.index)).reindex(returns_series.index)
    positions = labels_series.apply(_label_to_position).astype(float)
    return positions.shift(1).fillna(0.0) * returns_series


def compute_metrics(strategy_returns):
    returns = pd.Series(strategy_returns).fillna(0.0)
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    vol = returns.std(ddof=0)
    sharpe = float(np.sqrt(252.0) * returns.mean() / vol) if vol > 1e-12 else float('nan')
    return {
        'cumulative_return': float(curve.iloc[-1] - 1.0),
        'sharpe': sharpe,
        'max_drawdown': float(drawdown.min()),
        'win_rate': float((returns > 0.0).mean()),
    }


def run_backtest(config):
    results_dir = Path(config['paths']['results_dir'])
    if not (results_dir / 'feature_labels.csv').exists():
        run_interpretability(config)

    feature_labels = pd.read_csv(results_dir / 'feature_labels.csv').set_index('feature')['label'].to_dict()
    sparse_outputs = torch.load(results_dir / 'sparse_outputs.pt', map_location='cpu')
    market_payload = torch.load(Path(config['paths']['data_dir']) / 'market_data.pt', map_location='cpu')

    returns_frame = pd.DataFrame(
        market_payload['return_values'].numpy(),
        index=pd.to_datetime(market_payload['return_dates']),
        columns=market_payload['return_columns'],
    )
    test_dates = pd.to_datetime(sparse_outputs['test']['dates'])
    test_returns = returns_frame['SPY'].reindex(test_dates).dropna()
    dominant = pd.Series(sparse_outputs['test']['dominant_features'].numpy(), index=test_dates)
    labels = dominant.map(lambda idx: feature_labels.get(f'feature_{int(idx):03d}', 'unlabeled'))

    strategy_returns = regime_conditioned_strategy(test_returns, labels.reindex(test_returns.index))
    metrics = pd.DataFrame([
        {'method': 'Sparse Autoencoder', **compute_metrics(strategy_returns)},
        {'method': 'Buy and Hold SPY', **compute_metrics(test_returns)},
    ])
    metrics.to_csv(results_dir / 'backtest_metrics.csv', index=False)
    return metrics


def _label_to_position(label):
    if pd.isna(label):
        return 0.0
    text = str(label).lower()
    if 'risk-on' in text or 'calm' in text or 'trend-following' in text:
        return 1.0
    if 'risk-off' in text or 'volatility-spike' in text or 'correlation-shock' in text:
        return -1.0
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run sparse regime backtest.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_backtest(load_config(args.config))
