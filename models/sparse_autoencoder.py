"""Sparse autoencoder used to decompose dense regime embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    """Placeholder sparse autoencoder with an overcomplete hidden layer."""

    def __init__(self, input_dim: int, dict_size: int, topk: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.topk = topk
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x: torch.Tensor):
        return x, x, None


def compute_sparse_loss(recon_x, x, sparse_code, sparsity_lambda: float):
    """Placeholder sparse autoencoder loss."""

    raise NotImplementedError
