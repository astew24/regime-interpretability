"""Dense MLP autoencoder architecture for compressing rolling return windows."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Multi-layer perceptron autoencoder for market regime embeddings.

    Args:
        input_dim: Dimensionality of each flattened rolling return window.
        hidden_dims: Widths of the encoder hidden layers.
        latent_dim: Size of the learned latent regime embedding.

    Returns:
        Instantiated PyTorch module with mirrored encoder and decoder stacks.
    """

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], latent_dim: int) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width.")

        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.latent_dim = int(latent_dim)

        encoder_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = self.latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an input batch and reconstruct it from the latent representation.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tuple ``(reconstructed_x, latent_z)`` containing the reconstruction and latent embedding.
        """

        latent_z = self.encoder(x)
        reconstructed_x = self.decoder(latent_z)
        return reconstructed_x, latent_z

    def get_config(self) -> dict[str, object]:
        """Return constructor arguments needed to recreate the module."""

        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
        }


def get_encoder(model: MLPAutoencoder) -> nn.Module:
    """Return the encoder half of a trained autoencoder model.

    Args:
        model: Instantiated ``MLPAutoencoder``.

    Returns:
        Encoder submodule that maps input windows to latent embeddings.
    """

    return model.encoder
