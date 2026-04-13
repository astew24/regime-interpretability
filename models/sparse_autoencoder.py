"""Sparse autoencoder architecture for interpretable decomposition of latent embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SparseAutoencoder(nn.Module):
    """Overcomplete sparse autoencoder with TopK activation.

    Sits on top of frozen dense AE embeddings and learns a larger dictionary
    where each input only activates a few features (controlled by topk).
    """

    def __init__(self, input_dim: int, dict_size: int, topk: int) -> None:
        super().__init__()
        if topk > dict_size:
            raise ValueError("topk must be less than or equal to dict_size.")

        self.input_dim = int(input_dim)
        self.dict_size = int(dict_size)
        self.topk = int(topk)

        self.encoder = nn.Linear(self.input_dim, self.dict_size)
        self.decoder = nn.Linear(self.dict_size, self.input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstructed_x, sparse_code, active_indices).
        Only the top-k activations are kept; the rest are zeroed out."""

        dense_code = self.encoder(x)
        active_indices = torch.topk(dense_code.abs(), k=self.topk, dim=1).indices
        sparse_code = torch.zeros_like(dense_code)
        sparse_values = torch.gather(dense_code, dim=1, index=active_indices)
        sparse_code.scatter_(dim=1, index=active_indices, src=sparse_values)
        reconstructed_x = self.decoder(sparse_code)
        return reconstructed_x, sparse_code, active_indices

    def get_config(self) -> dict[str, int]:
        """Return constructor arguments needed to recreate the sparse model."""

        return {
            "input_dim": self.input_dim,
            "dict_size": self.dict_size,
            "topk": self.topk,
        }


def compute_sparse_loss(
    recon_x: torch.Tensor, x: torch.Tensor, sparse_code: torch.Tensor, sparsity_lambda: float
) -> torch.Tensor:
    """MSE reconstruction loss + L1 sparsity penalty weighted by sparsity_lambda."""

    reconstruction_loss = F.mse_loss(recon_x, x)
    sparsity_penalty = sparse_code.abs().mean()
    return reconstruction_loss + (sparsity_lambda * sparsity_penalty)
