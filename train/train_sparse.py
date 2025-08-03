"""Train the sparse autoencoder on frozen dense embeddings."""

from __future__ import annotations

import argparse


def train_one_epoch(model, dataloader, optimizer, device):
    """Run one sparse autoencoder optimization epoch."""

    raise NotImplementedError


def validate(model, dataloader, device):
    """Evaluate sparse reconstruction quality on a validation loader."""

    raise NotImplementedError


def train_sparse_autoencoder(config):
    """Main sparse training entry point."""

    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the sparse autoencoder.')
    parser.add_argument('--config', default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    parse_args()
