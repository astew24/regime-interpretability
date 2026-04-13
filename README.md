# Interpretable Market Regime Detection via Sparse Autoencoder Decomposition

**[Project Page →](https://astew24.github.io/regime-interpretability/)**

Research project completed at UC San Diego under the supervision of **[Sanjoy Dasgupta](https://cseweb.ucsd.edu/~dasgupta/)** (Dasgupta Lab, CSE Department).

This project studies whether rolling multi-asset return windows can be compressed into regime embeddings with a neural autoencoder and then decomposed with a sparse autoencoder into interpretable market features. The pipeline downloads cross-asset ETF data, constructs standardized rolling windows, trains latent representations, benchmarks unsupervised regime detectors, evaluates sparse features against external indicators, and tests whether the learned regimes have value in a simple SPY timing strategy. The code is meant to be easy to rerun and inspect, although it is still closer to a research repo than a polished package.

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

The scripts are intentionally simple entry points. There is some duplicated wiring between training and analysis modules because I optimized for readability while iterating on the experiments.

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

On the held-out test set (roughly 18 months of daily data):

- **8 sparse features** showed correlations above the 0.4 threshold with external indicators — mapped to risk-off episodes, yield-curve shifts, momentum reversals, commodity inflation, and dollar strength regimes
- **Early detection:** sparse features identified distribution shifts 2–4 days earlier on average than HMM and K-means baselines during COVID-19 (2020) and the 2022 rate shock
- **Failure mode:** regime features collapsed into noisy superposition during low-volatility compression periods (SVXY/VIX term structure flat), limiting distinctiveness — documented in ablation results
- **Strategy backtest:** regime-conditioned SPY timing using the best sparse feature beat buy-and-hold in crisis periods but underperformed during the 2023–2024 low-vol regime
