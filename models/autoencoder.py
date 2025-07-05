"""Dense autoencoder for rolling multi-asset return windows."""

from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Minimal scaffold for the dense regime autoencoder."""

    def __init__(self, input_dim: int, hidden_dims, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x: torch.Tensor):
        latent_z = self.encoder(x)
        reconstructed_x = self.decoder(latent_z)
        return reconstructed_x, latent_z


def get_encoder(model: MLPAutoencoder):
    """Return the encoder block from the autoencoder."""

    return model.encoder
