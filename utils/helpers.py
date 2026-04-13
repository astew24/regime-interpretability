"""Shared config, seed, and checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load project config yaml into a dict."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set all the random seeds — needed for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Save model weights + constructor args so it can be reloaded without
    having to reconstruct the class manually."""
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
    """Like load_checkpoint but returns None instead of crashing if the file
    doesn't exist yet. Handy for eval scripts that run mid-training."""
    if not Path(path).exists():
        return None
    return load_checkpoint(path, model_class)


def load_checkpoint(path: str | Path, model_class: Type[torch.nn.Module]) -> torch.nn.Module:
    """Reload a model from a checkpoint saved by save_checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = model_class(**model_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model
