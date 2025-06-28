"""PyTorch dataset utilities for rolling standardized return windows."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


class RollingWindowDataset(Dataset):
    """Dataset of flattened rolling return windows for unsupervised regime learning.

    Args:
        log_returns: Daily asset log returns as a pandas DataFrame, NumPy array,
            or torch Tensor with shape ``(num_days, num_assets)``.
        window_size: Number of trading days to include in each rolling sample.

    Attributes:
        windows: Tensor of shape ``(num_samples, window_size * num_assets)``.
        dates: Datetime index aligned to the final day of each rolling window.
        num_assets: Number of assets in each return observation.
    """

    def __init__(self, log_returns: pd.DataFrame | np.ndarray | torch.Tensor, window_size: int) -> None:
        self.window_size = int(window_size)
        values, dates = self._coerce_inputs(log_returns)
        self.num_assets = values.shape[1]
        self.windows, self.dates = self._build_windows(values=values, dates=dates)

    def __len__(self) -> int:
        """Return the number of rolling windows in the dataset."""

        return self.windows.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a flattened standardized window for the given sample index.

        Args:
            index: Dataset index for the requested rolling window.

        Returns:
            A float tensor with shape ``(window_size * num_assets,)``.
        """

        return self.windows[index]

    def _coerce_inputs(
        self, log_returns: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """Normalize supported return containers into arrays plus aligned dates."""

        if isinstance(log_returns, pd.DataFrame):
            values = log_returns.to_numpy(dtype=np.float32)
            dates = pd.DatetimeIndex(log_returns.index)
            return values, dates

        if isinstance(log_returns, np.ndarray):
            if log_returns.ndim != 2:
                raise ValueError("NumPy inputs must have shape (num_days, num_assets).")
            dates = pd.date_range(start="2000-01-01", periods=log_returns.shape[0], freq="B")
            return log_returns.astype(np.float32), dates

        if isinstance(log_returns, torch.Tensor):
            if log_returns.ndim != 2:
                raise ValueError("Tensor inputs must have shape (num_days, num_assets).")
            array = log_returns.detach().cpu().numpy().astype(np.float32)
            dates = pd.date_range(start="2000-01-01", periods=array.shape[0], freq="B")
            return array, dates

        raise TypeError("log_returns must be a pandas DataFrame, NumPy array, or torch Tensor.")

    def _build_windows(self, values: np.ndarray, dates: pd.DatetimeIndex) -> Tuple[torch.Tensor, pd.DatetimeIndex]:
        """Create z-scored rolling windows and align them to end-of-window dates."""

        if values.shape[0] < self.window_size:
            raise ValueError("Not enough observations to build the requested rolling windows.")

        windows = []
        sample_dates = []
        for end_idx in range(self.window_size - 1, values.shape[0]):
            window = values[end_idx - self.window_size + 1 : end_idx + 1]
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0
            standardized = (window - mean) / std
            windows.append(standardized.reshape(-1))
            sample_dates.append(dates[end_idx])

        tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
        return tensor, pd.DatetimeIndex(sample_dates)


def create_splits(
    dataset: Dataset, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[Subset, Subset, Subset]:
    """Create chronological train, validation, and test splits for time-series data.

    Args:
        dataset: Dataset containing rolling windows ordered chronologically.
        train_ratio: Fraction of the earliest samples used for training.
        val_ratio: Fraction of samples used for validation after training.
        test_ratio: Fraction of the latest samples reserved for testing.

    Returns:
        A tuple containing ``(train_subset, val_subset, test_subset)`` in temporal order.
    """

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, received {total_ratio:.4f}.")

    num_samples = len(dataset)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, num_samples))

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Chronological split produced an empty subset; adjust ratios or dataset size.")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )
