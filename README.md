# First Vibe ML Lab

A small PyTorch-based machine learning experiment workflow for CIFAR-10.

This repository is my first vibe coding project. It is meant to be a compact, reproducible baseline for running small ML experiments with a clear training, inference, and evaluation pipeline.

## What It Includes

- `train.py`: training loop with validation and WandB logging
- `inference.py`: sample predictions and visualization output
- `evaluation.py`: test metrics, per-class metrics, and confusion matrices
- `dataset.py`: CIFAR-10 data loading, transforms, and train/val/test split
- `model.py`: `SimpleCNN` baseline model
- `config.py`: centralized experiment config
- `reproducibility.py`: shared seed and deterministic setup

## Project Goals

- Keep the workflow simple enough to iterate on quickly
- Make experiments reproducible and trackable
- Separate training-time metrics from final evaluation metrics
- Serve as a base repo for future small ML experiments

## Current Setup

- Framework: PyTorch
- Dataset: CIFAR-10
- Tracking: Weights & Biases (`First-Vibe-Project`)
- Default training config:
  - Batch size: `64`
  - Learning rate: `0.001`
  - Epochs: `5`
  - Validation split: `0.1`
  - Seed: `42`

## Usage

Make sure the CIFAR-10 dataset already exists under `./data`.

Train:

```bash
python train.py
```

Train with custom checkpoints:

```bash
python train.py --model-path custom_last.pth --best-model-path custom_best.pth
```

Run inference samples:

```bash
python inference.py --model-path cifar_model.pth
```

Run evaluation:

```bash
python evaluation.py
```

Resume evaluation metrics into an existing WandB run:

```bash
python evaluation.py --resume-run-id <run_id>
```

## Outputs

- Final checkpoint: `cifar_model.pth`
- Best validation checkpoint: `cifar_model_best.pth`
- Inference visualization: `inference_samples.png`
- Confusion matrix figures: `confusion_matrix_counts_*.png`, `confusion_matrix_normalized_*.png`

## Notes

- The dataset is loaded with `download=False` by design.
- Random behavior is controlled through `SEED` for reproducibility.
- This repo intentionally keeps the model simple so the workflow is the main focus.
