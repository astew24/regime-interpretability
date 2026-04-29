# Regime Interpretability

Research code for studying whether market regime embeddings can be made easier to interpret with a sparse autoencoder.

[Project page](https://astew24.github.io/regime-interpretability/) | [GitHub repo](https://github.com/astew24/regime-interpretability)

## Overview

This project builds a small research pipeline around rolling multi-asset return windows:

- download daily ETF and market-indicator data with `yfinance`
- turn returns into standardized rolling windows
- train a dense autoencoder to compress each window into a latent regime embedding
- train a sparse autoencoder on the frozen embeddings
- compare sparse regime features against external indicators such as VIX, yield spread, momentum, and cross-asset correlation
- benchmark against HMM and K-means regime labels
- run a simple SPY timing backtest for downstream sanity checking

The repo is intentionally closer to a research notebook/codebase than a polished package. Most scripts are meant to be run from the command line and inspected directly.

## Why I Built This

Market regime models are often hard to explain. I wanted to test whether sparse decomposition could make learned regime features easier to inspect than dense embeddings or cluster labels.

This project was completed at UC San Diego under the supervision of [Sanjoy Dasgupta](https://cseweb.ucsd.edu/~dasgupta/) in the Dasgupta Lab.

## How It Works

```text
daily ETF prices
      |
      v
log returns + rolling windows
      |
      v
dense autoencoder
      |
      v
frozen latent embeddings
      |
      v
sparse autoencoder with TopK activation
      |
      v
feature correlations, regime baselines, and simple backtest
```

Key settings live in `config.yaml`. The current config uses 15 market ETFs/indicators, 20-day rolling windows, a 32-dimensional dense latent space, and a 128-feature sparse dictionary with 8 active features per sample.

## Project Structure

```text
regime-interpretability/
|-- config.yaml
|-- data/
|   |-- dataset.py
|   `-- download.py
|-- models/
|   |-- autoencoder.py
|   `-- sparse_autoencoder.py
|-- train/
|   |-- train_ae.py
|   `-- train_sparse.py
|-- analysis/
|   |-- interpretability.py
|   |-- baselines.py
|   |-- backtest.py
|   `-- ablation.py
|-- viz/
|   `-- plots.py
|-- notebooks/
|   `-- results.ipynb
|-- docs/index.html
`-- requirements.txt
```

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and process the market data:

```bash
python data/download.py --config config.yaml
```

Train the dense autoencoder:

```bash
python train/train_ae.py --config config.yaml
```

Train the sparse autoencoder:

```bash
python train/train_sparse.py --config config.yaml
```

Run the analysis scripts:

```bash
python analysis/interpretability.py --config config.yaml
python analysis/baselines.py --config config.yaml
python analysis/backtest.py --config config.yaml
python analysis/ablation.py --config config.yaml
```

Outputs are written under the configured `results/` directory. Generated artifacts are not committed except for a `.gitkeep` placeholder.

## Example Outputs

Depending on which scripts you run, the repo can produce:

- trained autoencoder checkpoints
- dense embeddings and sparse activations
- feature-to-indicator correlation CSVs
- UMAP plots and feature heatmaps
- baseline transition summaries
- ablation tables
- a simple backtest summary

## Limitations

- The data source is `yfinance`, so the pipeline depends on Yahoo availability and symbol coverage.
- Feature labels are heuristic and based on correlations with external indicators.
- HMM and K-means baselines are simple comparison points, not heavily tuned models.
- The SPY timing backtest is a sanity check, not a deployable strategy.
- Generated results should be rerun locally before making claims from them.

## Next Steps

- Add pinned sample outputs or a small fixture so reviewers can reproduce the project quickly.
- Improve experiment tracking around config, random seed, and output metadata.
- Add tests for dataset construction and analysis helpers.
- Try calendar-aware event definitions for crisis and rate-shock windows.
- Compare sparse autoencoder features against simpler PCA or factor-model baselines.
