"""Model architectures for regime embedding and sparse decomposition experiments."""

from .autoencoder import MLPAutoencoder, get_encoder
from .sparse_autoencoder import SparseAutoencoder, compute_sparse_loss

__all__ = [
    "MLPAutoencoder",
    "get_encoder",
    "SparseAutoencoder",
    "compute_sparse_loss",
]
