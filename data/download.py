"""Download helpers for raw ETF prices and regime indicator series."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def download_etf_data(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the configured ETF universe."""

    raw = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=False, group_by='column')
    if raw.empty:
        raise ValueError('No price data returned from yfinance.')

    if isinstance(raw.columns, pd.MultiIndex):
        field = 'Adj Close' if 'Adj Close' in raw.columns.get_level_values(0) else 'Close'
        prices = pd.DataFrame(raw[field])
    else:
        prices = pd.DataFrame(raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close'])
    return prices.sort_index().ffill().dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices into log returns."""

    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def download_external_indicators(start: str, end: str) -> pd.DataFrame:
    """Fetch VIX and yield-curve proxies for later validation."""

    raw = yf.download(['^VIX', '^TNX', '^IRX'], start=start, end=end, progress=False, auto_adjust=False, group_by='column')
    if isinstance(raw.columns, pd.MultiIndex):
        field = 'Adj Close' if 'Adj Close' in raw.columns.get_level_values(0) else 'Close'
        panel = pd.DataFrame(raw[field]).sort_index().ffill().dropna()
    else:
        field = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        panel = pd.DataFrame(raw[field]).sort_index().ffill().dropna()

    indicators = pd.DataFrame(index=panel.index)
    indicators['vix_level'] = panel['^VIX']
    indicators['vix_5d_change'] = indicators['vix_level'].diff(5)
    indicators['yield_spread'] = (panel['^TNX'] - panel['^IRX']) / 10.0
    return indicators.dropna()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download raw market data.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f'Data config path: {args.config}')


if __name__ == '__main__':
    main()
