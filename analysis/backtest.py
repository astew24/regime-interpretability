"""Backtest regime-conditioned SPY strategies."""

from __future__ import annotations


def regime_conditioned_strategy(returns, regime_labels):
    """Turn regime labels into tradable SPY positions."""

    raise NotImplementedError


def compute_metrics(strategy_returns):
    """Compute cumulative return, Sharpe, drawdown, and win rate."""

    raise NotImplementedError


def run_backtest(config):
    """Run the held-out test set backtest."""

    raise NotImplementedError
