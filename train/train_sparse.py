"""Train the sparse autoencoder on frozen dense embeddings."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sparse_autoencoder import SparseAutoencoder, compute_sparse_loss
from train.train_ae import train_autoencoder
from utils.helpers import load_config, save_checkpoint, set_seed


_SPARSITY_LAMBDA = 0.01


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        inputs = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, sparse_code, _ = model(inputs)
        loss = compute_sparse_loss(reconstructed, inputs, sparse_code, _SPARSITY_LAMBDA)
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
            reconstructed, sparse_code, _ = model(inputs)
            loss = compute_sparse_loss(reconstructed, inputs, sparse_code, _SPARSITY_LAMBDA)
            total_loss += loss.item() * inputs.size(0)
            total_examples += inputs.size(0)
    return total_loss / max(total_examples, 1)


def train_sparse_autoencoder(config):
    global _SPARSITY_LAMBDA
    set_seed(config['autoencoder']['seed'])
    results_dir = Path(config['paths']['results_dir'])
    embeddings_path = results_dir / 'embeddings.pt'
    if not embeddings_path.exists():
        train_autoencoder(config)

    embeddings = torch.load(embeddings_path, map_location='cpu')
    train_embeddings = embeddings['train']['embeddings'].float()
    val_embeddings = embeddings['val']['embeddings'].float()

    sparse_cfg = config['sparse_autoencoder']
    _SPARSITY_LAMBDA = float(sparse_cfg['sparsity_lambda'])

    model = SparseAutoencoder(
        input_dim=sparse_cfg['input_dim'],
        dict_size=sparse_cfg['dict_size'],
        topk=sparse_cfg['topk'],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=sparse_cfg['learning_rate'])

    train_loader = DataLoader(train_embeddings, batch_size=sparse_cfg['batch_size'], shuffle=False)
    val_loader = DataLoader(val_embeddings, batch_size=sparse_cfg['batch_size'], shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience = config['autoencoder'].get('patience', 15)
    patience_counter = 0

    for _epoch in range(sparse_cfg['max_epochs']):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    save_checkpoint(model, Path(config['paths']['checkpoint_dir']) / 'sparse_autoencoder.pt')
    torch.save(model.decoder.weight.detach().cpu(), results_dir / 'sparse_dictionary.pt')
    return {'best_val_loss': best_val_loss}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the sparse autoencoder.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_sparse_autoencoder(load_config(args.config))
