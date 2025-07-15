"""Train the dense autoencoder used for market regime embeddings."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import run_data_pipeline
from models.autoencoder import MLPAutoencoder
from utils.helpers import load_config, save_checkpoint, set_seed


def train_one_epoch(model, dataloader, optimizer, device):
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
        total_loss += loss.item() * inputs.size(0)
        total_examples += inputs.size(0)
    return total_loss / max(total_examples, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            reconstructed, _ = model(inputs)
            loss = F.mse_loss(reconstructed, inputs)
            total_loss += loss.item() * inputs.size(0)
            total_examples += inputs.size(0)
    return total_loss / max(total_examples, 1)


def _ensure_processed_data(config):
    data_dir = Path(config['paths']['data_dir'])
    expected = {
        'train': data_dir / 'train.pt',
        'val': data_dir / 'val.pt',
    }
    if any(not path.exists() for path in expected.values()):
        run_data_pipeline(config)
    return expected


def train_autoencoder(config):
    set_seed(config['autoencoder']['seed'])
    paths = _ensure_processed_data(config)
    train_windows = torch.load(paths['train'], map_location='cpu')['windows'].float()
    val_windows = torch.load(paths['val'], map_location='cpu')['windows'].float()

    batch_size = config['autoencoder']['batch_size']
    train_loader = DataLoader(train_windows, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_windows, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPAutoencoder(
        input_dim=config['autoencoder']['input_dim'],
        hidden_dims=config['autoencoder']['hidden_dims'],
        latent_dim=config['autoencoder']['latent_dim'],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config['autoencoder']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['autoencoder']['max_epochs'])

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience = config['autoencoder']['patience']
    patience_counter = 0

    for _epoch in range(config['autoencoder']['max_epochs']):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    save_checkpoint(model, Path(config['paths']['checkpoint_dir']) / 'autoencoder.pt')
    return {'best_val_loss': best_val_loss}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the dense autoencoder.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_autoencoder(load_config(args.config))
