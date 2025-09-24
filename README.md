# Interpretable Market Regime Detection via Sparse Autoencoder Decomposition

This project studies whether rolling multi-asset return windows can be compressed into regime embeddings with a neural autoencoder and then decomposed with a sparse autoencoder into interpretable market features. The pipeline downloads cross-asset ETF data, constructs standardized rolling windows, trains latent representations, benchmarks unsupervised regime detectors, evaluates sparse features against external indicators, and tests whether the learned regimes have value in a simple SPY timing strategy.

## Table of Contents

- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Analysis](#analysis)
- [Results](#results)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All experiment settings live in [`config.yaml`](./config.yaml). Update paths, date ranges, or hyperparameters there before running the project.

## Data

Build the processed rolling-window dataset and external indicator panel with:

```bash
python data/download.py --config config.yaml
```

The data pipeline downloads daily market prices, computes log returns, creates rolling standardized windows, performs chronological train/validation/test splits, and saves processed `.pt` artifacts under the configured data directory.

## Training

Train the base autoencoder:

```bash
python train/train_ae.py --config config.yaml
```

Train the sparse autoencoder on frozen latent embeddings:

```bash
python train/train_sparse.py --config config.yaml
```

## Analysis

Run the downstream analysis modules after training:

```bash
python analysis/interpretability.py --config config.yaml
python analysis/baselines.py --config config.yaml
python analysis/backtest.py --config config.yaml
python analysis/ablation.py --config config.yaml
```

Saved figures, tables, and checkpoints are written beneath the configured results directory.

## Results

TODO
