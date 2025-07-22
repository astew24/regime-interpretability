"""Plotting helpers for model training diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(train_losses), label='Train', linewidth=2)
    ax.plot(list(val_losses), label='Validation', linewidth=2)
    ax.set_title('Training Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax
