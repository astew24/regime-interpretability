"""Shared configuration, reproducibility, and checkpoint helpers for the project."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducible experiments.

    Args:
        seed: Integer seed value used across supported libraries.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Save a model checkpoint containing weights and constructor arguments.

    Args:
        model: Instantiated PyTorch model to persist.
        path: Output checkpoint file path.

    Returns:
        None.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if hasattr(model, "get_config") and callable(model.get_config):
        payload["model_kwargs"] = model.get_config()

    torch.save(payload, checkpoint_path)


def load_checkpoint_safe(
    path: str | Path, model_class: Type[torch.nn.Module]
) -> torch.nn.Module | None:
    """Attempt to load a checkpoint, returning None if the file does not exist.

    Convenience wrapper around :func:`load_checkpoint` for evaluation scripts
    that should degrade gracefully when a training run hasn't completed yet.

    Args:
        path: Path to the checkpoint file on disk.
        model_class: PyTorch module class used to re-instantiate the model.

    Returns:
        Loaded model, or ``None`` if the checkpoint file is absent.
    """

    if not Path(path).exists():
        return None
    return load_checkpoint(path, model_class)


def load_checkpoint(path: str | Path, model_class: Type[torch.nn.Module]) -> torch.nn.Module:
    """Restore a model instance from a checkpoint saved by ``save_checkpoint``.

    Args:
        path: Path to the checkpoint file on disk.
        model_class: PyTorch module class used to re-instantiate the model.

    Returns:
        Model instance loaded onto CPU with checkpoint weights restored.
    """

    checkpoint = torch.load(Path(path), map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = model_class(**model_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model
