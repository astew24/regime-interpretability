"""Train the dense autoencoder and save frozen latent embeddings for downstream analysis."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import run_data_pipeline
from models.autoencoder import MLPAutoencoder, get_encoder
from utils.helpers import load_config, save_checkpoint, set_seed
from viz.plots import plot_training_curves


def train_one_epoch(
    model: MLPAutoencoder, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Run one optimization epoch for the dense autoencoder.

    Args:
        model: Autoencoder instance to optimize.
        dataloader: Training dataloader yielding flattened rolling windows.
        optimizer: Optimizer configured for the autoencoder parameters.
        device: Device used for training.

    Returns:
        Mean training loss for the epoch.
    """

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        inputs = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, _ = model(inputs)
        loss = F.mse_loss(reconstructed, inputs)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def validate(model: MLPAutoencoder, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate dense autoencoder reconstruction loss on a validation loader.

    Args:
        model: Autoencoder instance in evaluation mode.
        dataloader: Validation dataloader yielding flattened rolling windows.
        device: Device used for evaluation.

    Returns:
        Mean validation loss across the dataloader.
    """

    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            reconstructed, _ = model(inputs)
            loss = F.mse_loss(reconstructed, inputs)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_autoencoder(config: Dict) -> Dict[str, object]:
    """Train the configured dense autoencoder and persist checkpoints plus embeddings.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Summary dictionary describing the best validation loss and saved artifact paths.
    """

    set_seed(config["autoencoder"]["seed"])
    data_paths = _ensure_processed_data(config)
    results_dir = Path(config["paths"]["results_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_payload = torch.load(data_paths["train_split"], map_location="cpu")
    val_payload = torch.load(data_paths["val_split"], map_location="cpu")
    test_payload = torch.load(data_paths["test_split"], map_location="cpu")
    full_payload = torch.load(data_paths["window_data"], map_location="cpu")

    train_windows = train_payload["windows"].float()
    val_windows = val_payload["windows"].float()
    test_windows = test_payload["windows"].float()
    full_windows = full_payload["windows"].float()

    ae_config = config["autoencoder"]
    batch_size = ae_config["batch_size"]
    train_loader = DataLoader(
        train_windows,
        batch_size=batch_size,
        shuffle=False,
        drop_last=(len(train_windows) % batch_size == 1),
    )
    val_loader = DataLoader(val_windows, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPAutoencoder(
        input_dim=ae_config["input_dim"],
        hidden_dims=ae_config["hidden_dims"],
        latent_dim=ae_config["latent_dim"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=ae_config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ae_config["max_epochs"])

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(ae_config["max_epochs"]):
        train_loss = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
        val_loss = validate(model=model, dataloader=val_loader, device=device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= ae_config["patience"]:
            break

    model.load_state_dict(best_state)

    autoencoder_ckpt = checkpoint_dir / "autoencoder.pt"
    encoder_ckpt = checkpoint_dir / "encoder.pt"
    save_checkpoint(model, autoencoder_ckpt)
    torch.save(
        {
            "state_dict": get_encoder(model).state_dict(),
            "input_dim": ae_config["input_dim"],
            "hidden_dims": ae_config["hidden_dims"],
            "latent_dim": ae_config["latent_dim"],
        },
        encoder_ckpt,
    )

    full_embeddings = _encode_tensor(model=model, windows=full_windows, batch_size=batch_size, device=device)
    train_embeddings = _encode_tensor(model=model, windows=train_windows, batch_size=batch_size, device=device)
    val_embeddings = _encode_tensor(model=model, windows=val_windows, batch_size=batch_size, device=device)
    test_embeddings = _encode_tensor(model=model, windows=test_windows, batch_size=batch_size, device=device)

    embedding_path = results_dir / "embeddings.pt"
    torch.save(
        {
            "full": {"embeddings": full_embeddings.cpu(), "dates": full_payload["dates"]},
            "train": {"embeddings": train_embeddings.cpu(), "dates": train_payload["dates"]},
            "val": {"embeddings": val_embeddings.cpu(), "dates": val_payload["dates"]},
            "test": {"embeddings": test_embeddings.cpu(), "dates": test_payload["dates"]},
        },
        embedding_path,
    )

    history_frame = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    history_frame.to_csv(results_dir / "autoencoder_history.csv", index=False)

    fig, _ax = plot_training_curves(train_losses=train_losses, val_losses=val_losses)
    fig.savefig(results_dir / "autoencoder_training_curves.png", dpi=200, bbox_inches="tight")
    fig.clf()

    summary = {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "checkpoint": str(autoencoder_ckpt),
        "encoder_checkpoint": str(encoder_ckpt),
        "embedding_path": str(embedding_path),
        "hyperparameters": ae_config,
    }
    with (results_dir / "autoencoder_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _ensure_processed_data(config: Dict) -> Dict[str, Path]:
    """Ensure processed window and split artifacts exist before model training."""

    data_dir = Path(config["paths"]["data_dir"])
    expected = {
        "window_data": data_dir / "window_data.pt",
        "market_data": data_dir / "market_data.pt",
        "train_split": data_dir / "train.pt",
        "val_split": data_dir / "val.pt",
        "test_split": data_dir / "test.pt",
    }
    if any(not path.exists() for path in expected.values()):
        run_data_pipeline(config)
    return expected


def _encode_tensor(model: MLPAutoencoder, windows: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode a full tensor of windows into latent embeddings without gradient tracking."""

    model.eval()
    embeddings = []
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device)
            _, latent = model(inputs)
            embeddings.append(latent.cpu())
    return torch.cat(embeddings, dim=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the autoencoder training entry point."""

    parser = argparse.ArgumentParser(description="Train the dense market regime autoencoder.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    train_autoencoder(loaded_config)
