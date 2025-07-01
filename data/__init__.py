"""Data ingestion, preprocessing, and dataset utilities for market regime modeling."""

from .dataset import RollingWindowDataset, create_splits
from .download import compute_log_returns, download_etf_data, download_external_indicators

__all__ = [
    "RollingWindowDataset",
    "create_splits",
    "download_etf_data",
    "compute_log_returns",
    "download_external_indicators",
]
