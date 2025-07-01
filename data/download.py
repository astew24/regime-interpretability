"""Download, preprocess, and persist multi-asset market data for regime experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RollingWindowDataset, create_splits
from utils.helpers import load_config, set_seed


def download_etf_data(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close prices for the configured market universe.

    Args:
        tickers: Iterable of ticker symbols to download in column order.
        start: Inclusive start date in ``YYYY-MM-DD`` format.
        end: Exclusive end date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame of adjusted close prices indexed by trading date and ordered by ticker.
    """

    ticker_list = list(tickers)
    raw = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("No market price data was returned by yfinance.")

    prices = _extract_price_panel(raw)
    prices = prices.reindex(columns=ticker_list)
    prices = prices.sort_index().ffill().dropna()
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price panel.

    Args:
        prices: Daily adjusted close prices indexed by date.

    Returns:
        DataFrame of one-period log returns aligned to the input dates after the first row.
    """

    log_returns = np.log(prices / prices.shift(1))
    return log_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def download_external_indicators(start: str, end: str) -> pd.DataFrame:
    """Download external validation indicators used for regime interpretation.

    Args:
        start: Inclusive start date in ``YYYY-MM-DD`` format.
        end: Exclusive end date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame containing VIX level, five-day VIX change, and a term-spread proxy.
    """

    raw = yf.download(
        tickers=["^VIX", "^TNX", "^IRX"],
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("No external indicator data was returned by yfinance.")

    levels = _extract_price_panel(raw).sort_index().ffill().dropna()
    indicators = pd.DataFrame(index=levels.index)
    indicators["vix_level"] = levels["^VIX"]
    indicators["vix_5d_change"] = indicators["vix_level"].diff(5)
    indicators["yield_spread"] = (levels["^TNX"] - levels["^IRX"]) / 10.0
    return indicators.dropna()


def build_rolling_windows(log_returns: pd.DataFrame, window_size: int) -> Tuple[torch.Tensor, pd.DatetimeIndex]:
    """Construct flattened standardized rolling windows from daily returns.

    Args:
        log_returns: Asset return DataFrame with shape ``(num_days, num_assets)``.
        window_size: Number of trading days per sample window.

    Returns:
        Tuple of ``(windows, dates)`` where windows are float tensors and dates align to window endpoints.
    """

    dataset = RollingWindowDataset(log_returns=log_returns, window_size=window_size)
    return dataset.windows, dataset.dates


def run_data_pipeline(config: Dict) -> Dict[str, Path]:
    """Run the full data pipeline and save processed artifacts to disk.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary mapping artifact names to their saved file paths.
    """

    set_seed(config["autoencoder"]["seed"])

    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    window_size = config["data"]["window_size"]

    prices = download_etf_data(tickers=tickers, start=start_date, end=end_date)
    log_returns = compute_log_returns(prices)
    indicators = download_external_indicators(start=start_date, end=end_date)

    dataset = RollingWindowDataset(log_returns=log_returns, window_size=window_size)
    train_set, val_set, test_set = create_splits(
        dataset=dataset,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    full_payload = {
        "windows": dataset.windows,
        "dates": [timestamp.isoformat() for timestamp in dataset.dates],
        "tickers": list(log_returns.columns),
        "window_size": window_size,
    }
    market_payload = {
        "price_values": torch.tensor(prices.to_numpy(dtype=np.float32)),
        "price_dates": [timestamp.isoformat() for timestamp in prices.index],
        "price_columns": list(prices.columns),
        "return_values": torch.tensor(log_returns.to_numpy(dtype=np.float32)),
        "return_dates": [timestamp.isoformat() for timestamp in log_returns.index],
        "return_columns": list(log_returns.columns),
        "indicator_values": torch.tensor(indicators.to_numpy(dtype=np.float32)),
        "indicator_dates": [timestamp.isoformat() for timestamp in indicators.index],
        "indicator_columns": list(indicators.columns),
    }

    artifact_paths = {
        "window_data": data_dir / "window_data.pt",
        "market_data": data_dir / "market_data.pt",
        "train_split": data_dir / "train.pt",
        "val_split": data_dir / "val.pt",
        "test_split": data_dir / "test.pt",
    }

    torch.save(full_payload, artifact_paths["window_data"])
    torch.save(market_payload, artifact_paths["market_data"])
    torch.save(_subset_payload(dataset, train_set), artifact_paths["train_split"])
    torch.save(_subset_payload(dataset, val_set), artifact_paths["val_split"])
    torch.save(_subset_payload(dataset, test_set), artifact_paths["test_split"])

    return artifact_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data pipeline entry point."""

    parser = argparse.ArgumentParser(description="Download and preprocess market regime data.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def _extract_price_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract an adjusted close price panel from a raw yfinance response."""

    if isinstance(raw.columns, pd.MultiIndex):
        top_level = raw.columns.get_level_values(0)
        for field in ("Adj Close", "Close"):
            if field in top_level:
                panel = raw[field].copy()
                return pd.DataFrame(panel)
        raise ValueError("Unable to locate adjusted close prices in the yfinance response.")

    if "Adj Close" in raw.columns:
        return pd.DataFrame(raw["Adj Close"])
    if "Close" in raw.columns:
        return pd.DataFrame(raw["Close"])
    return pd.DataFrame(raw)


def _subset_payload(dataset: RollingWindowDataset, subset) -> Dict[str, object]:
    """Serialize a dataset subset into tensors plus date metadata."""

    indices = list(subset.indices)
    dates = [dataset.dates[idx].isoformat() for idx in indices]
    return {
        "windows": dataset.windows[indices].clone(),
        "dates": dates,
        "indices": indices,
    }


def main() -> None:
    """Run the configured market data pipeline from the command line."""

    args = parse_args()
    config = load_config(args.config)
    artifact_paths = run_data_pipeline(config)
    for name, path in artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
