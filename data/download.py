"""Download helpers for raw ETF prices and regime indicator series."""

from __future__ import annotations

import argparse
from typing import Iterable


def download_etf_data(tickers: Iterable[str], start: str, end: str):
    """Download adjusted close prices for the configured ETF universe."""

    # TODO: call yfinance and return a clean price frame.
    raise NotImplementedError


def compute_log_returns(prices):
    """Convert daily prices into log returns."""

    # TODO: take one-period logs and drop the initial NaN row.
    raise NotImplementedError


def download_external_indicators(start: str, end: str):
    """Fetch VIX and yield-curve proxies for later validation."""

    # TODO: download VIX and treasury proxies used in interpretability checks.
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download raw market data.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f'Loading config from {args.config}')


if __name__ == '__main__':
    main()
