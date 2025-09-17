"""Run ablation sweeps across the dense and sparse model settings."""

from __future__ import annotations


def run_ablation_sweep(config):
    """Evaluate a grid of embedding and sparsity hyperparameters."""

    raise NotImplementedError


def summarize_ablation_results(results):
    """Summarize the most promising ablation configurations."""

    raise NotImplementedError
