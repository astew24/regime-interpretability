"""Dataset helpers for rolling windows of multi-asset returns."""

from __future__ import annotations

from torch.utils.data import Dataset


class RollingWindowDataset(Dataset):
    """Hold rolling windows of returns for autoencoder training."""

    def __init__(self, log_returns, window_size: int) -> None:
        self.log_returns = log_returns
        self.window_size = window_size

    def __len__(self) -> int:
        # TODO: return the number of valid rolling windows.
        raise NotImplementedError

    def __getitem__(self, index: int):
        # TODO: slice a rolling window, standardize it, and flatten it.
        raise NotImplementedError


def create_splits(dataset, train_ratio: float, val_ratio: float, test_ratio: float):
    """Split a dataset chronologically into train, validation, and test segments."""

    # TODO: perform a deterministic time-series split with no shuffling.
    raise NotImplementedError
