"""Train the sparse autoencoder on frozen dense embeddings and save interpretable features."""

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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sparse_autoencoder import SparseAutoencoder, compute_sparse_loss
from train.train_ae import train_autoencoder
from utils.helpers import load_config, save_checkpoint, set_seed
from viz.plots import plot_training_curves


def train_one_epoch(
    model: SparseAutoencoder, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Run one training epoch and return mean loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        inputs = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, sparse_code, _ = model(inputs)
        loss = compute_sparse_loss(recon_x=reconstructed, x=inputs, sparse_code=sparse_code, sparsity_lambda=_SPARSITY_LAMBDA)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def validate(model: SparseAutoencoder, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate sparse autoencoder loss on a validation loader.

    Args:
        model: Sparse autoencoder instance in evaluation mode.
        dataloader: Validation dataloader yielding frozen embeddings.
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
            reconstructed, sparse_code, _ = model(inputs)
            reconstruction_loss = F.mse_loss(reconstructed, inputs)
            sparsity_penalty = sparse_code.abs().mean()
            loss = reconstruction_loss + (_SPARSITY_LAMBDA * sparsity_penalty)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_sparse_autoencoder(config: Dict) -> Dict[str, object]:
    """Train the sparse autoencoder on frozen dense embeddings and save dictionary artifacts.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Summary dictionary describing the best sparse validation loss and artifact paths.
    """

    global _SPARSITY_LAMBDA

    set_seed(config["autoencoder"]["seed"])
    results_dir = Path(config["paths"]["results_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = results_dir / "embeddings.pt"
    if not embeddings_path.exists():
        train_autoencoder(config)

    embeddings = torch.load(embeddings_path, map_location="cpu")
    sparse_config = config["sparse_autoencoder"]
    _SPARSITY_LAMBDA = float(sparse_config["sparsity_lambda"])

    train_embeddings = embeddings["train"]["embeddings"].float()
    val_embeddings = embeddings["val"]["embeddings"].float()
    test_embeddings = embeddings["test"]["embeddings"].float()
    full_embeddings = embeddings["full"]["embeddings"].float()

    train_loader = DataLoader(train_embeddings, batch_size=sparse_config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_embeddings, batch_size=sparse_config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(
        input_dim=sparse_config["input_dim"],
        dict_size=sparse_config["dict_size"],
        topk=sparse_config["topk"],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=sparse_config["learning_rate"])
    # TODO: try the same cosine schedule used in the dense autoencoder and check if it
    # helps the larger dictionary sweeps settle down a bit faster.

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = config["autoencoder"].get("patience", 15)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(sparse_config["max_epochs"]):
        train_loss = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
        val_loss = validate(model=model, dataloader=val_loader, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    sparse_ckpt = checkpoint_dir / "sparse_autoencoder.pt"
    dictionary_path = results_dir / "sparse_dictionary.pt"
    save_checkpoint(model, sparse_ckpt)
    torch.save(model.decoder.weight.detach().cpu(), dictionary_path)

    full_outputs = _encode_sparse_outputs(model=model, inputs=full_embeddings, batch_size=sparse_config["batch_size"], device=device)
    train_outputs = _encode_sparse_outputs(model=model, inputs=train_embeddings, batch_size=sparse_config["batch_size"], device=device)
    val_outputs = _encode_sparse_outputs(model=model, inputs=val_embeddings, batch_size=sparse_config["batch_size"], device=device)
    test_outputs = _encode_sparse_outputs(model=model, inputs=test_embeddings, batch_size=sparse_config["batch_size"], device=device)

    outputs_path = results_dir / "sparse_outputs.pt"
    torch.save(
        {
            "full": {**full_outputs, "dates": embeddings["full"]["dates"]},
            "train": {**train_outputs, "dates": embeddings["train"]["dates"]},
            "val": {**val_outputs, "dates": embeddings["val"]["dates"]},
            "test": {**test_outputs, "dates": embeddings["test"]["dates"]},
        },
        outputs_path,
    )

    history_frame = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    history_frame.to_csv(results_dir / "sparse_autoencoder_history.csv", index=False)

    fig, _ax = plot_training_curves(train_losses=train_losses, val_losses=val_losses)
    fig.savefig(results_dir / "sparse_autoencoder_training_curves.png", dpi=200, bbox_inches="tight")
    fig.clf()

    summary = {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "checkpoint": str(sparse_ckpt),
        "dictionary_path": str(dictionary_path),
        "outputs_path": str(outputs_path),
        "hyperparameters": sparse_config,
    }
    with (results_dir / "sparse_autoencoder_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _encode_sparse_outputs(
    model: SparseAutoencoder, inputs: torch.Tensor, batch_size: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Run a sparse autoencoder across a tensor and collect outputs for later analysis."""

    model.eval()
    reconstructions = []
    sparse_codes = []
    active_indices = []
    dominant_features = []

    loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            embeddings = batch.to(device)
            reconstructed, sparse_code, indices = model(embeddings)
            reconstructions.append(reconstructed.cpu())
            sparse_codes.append(sparse_code.cpu())
            active_indices.append(indices.cpu())
            dominant_features.append(sparse_code.abs().argmax(dim=1).cpu())

    return {
        "reconstructions": torch.cat(reconstructions, dim=0),
        "sparse_codes": torch.cat(sparse_codes, dim=0),
        "active_indices": torch.cat(active_indices, dim=0),
        "dominant_features": torch.cat(dominant_features, dim=0),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sparse training entry point."""

    parser = argparse.ArgumentParser(description="Train the sparse autoencoder on frozen embeddings.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


_SPARSITY_LAMBDA = 0.01


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    train_sparse_autoencoder(loaded_config)
