"""Interpret sparse features against known market indicators."""

from __future__ import annotations


def compute_feature_activations(sparse_model, embeddings):
    """Project dense embeddings into sparse activations."""

    raise NotImplementedError


def correlate_with_indicators(activations, indicators):
    """Correlate sparse feature activity with external market signals."""

    raise NotImplementedError


def label_interpretable_features(correlations, threshold: float = 0.5):
    """Assign descriptive labels to interpretable sparse features."""

    raise NotImplementedError


def generate_event_heatmap(activations, event_dates):
    """Aggregate feature activity around named market events."""

    raise NotImplementedError
