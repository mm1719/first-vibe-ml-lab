# First Vibe ML Lab

A small PyTorch-based machine learning experiment workflow for CIFAR-10.

This repository is my first vibe coding project. It is meant to be a compact, reproducible baseline for running small ML experiments with a clear training, inference, and evaluation pipeline.

## Environment Setup

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

## What It Includes

- `scripts/train.py`: training loop with validation and WandB logging
- `scripts/inference.py`: sample predictions and visualization output
- `src/utils/evaluation.py`: test metrics, per-class metrics, and confusion matrices
- `src/data/dataset.py`: CIFAR-10 data loading, transforms, and train/val/test split
- `src/models/model.py`: `SimpleCNN` experiment model
- `configs/config.py`: centralized experiment config
- `src/utils/reproducibility.py`: shared seed and deterministic setup

## Project Goals

- Keep the workflow simple enough to iterate on quickly
- Make experiments reproducible and trackable
- Separate training-time metrics from final evaluation metrics
- Serve as a base repo for future small ML experiments

## Current Setup

- Framework: PyTorch
- Dataset: CIFAR-10
- Tracking: Weights & Biases (`First-Vibe-Project`)
- Current default training config:
  - Batch size: `128`
  - Learning rate: `3e-4`
  - Epochs: `12`
  - Validation split: `0.1`
  - Seed: `42`
  - Optimizer: `AdamW`
  - Weight decay: `1e-4`
  - Label smoothing: `0.1`

## Usage

Make sure the CIFAR-10 dataset already exists under `./data`.

Train:

```bash
python scripts/train.py
```

Train with custom checkpoints:

```bash
python scripts/train.py --model-path checkpoints/custom_last.pth --best-model-path checkpoints/custom_best.pth
```

Run inference samples:

```bash
python scripts/inference.py --model-path checkpoints/cifar_model.pth
```

Run evaluation:

```bash
python scripts/evaluation.py
```

Evaluate a specific checkpoint:

```bash
python scripts/evaluation.py --model-path checkpoints/cifar_model_agent_best.pth
```

Resume evaluation metrics into an existing WandB run:

```bash
python scripts/evaluation.py --resume-run-id <run_id>
```

## Outputs

- Final checkpoint: `cifar_model.pth`
- Best validation checkpoint: `cifar_model_best.pth`
- Agent experiment checkpoints: `cifar_model_agent_last.pth`, `cifar_model_agent_best.pth`
- Inference visualization: `inference_samples.png`
- Confusion matrix figures: `confusion_matrix_counts_*.png`, `confusion_matrix_normalized_*.png`

## Notes

- The dataset is loaded with `download=False` by design.
- Random behavior is controlled through `SEED` for reproducibility.
- This repo intentionally keeps the model simple so the workflow is the main focus.

## Project Conventions

- `train.py`, `inference.py`, and `evaluation.py` are intentionally separated and should remain separate.
- Training metrics belong in `wandb.log`; final evaluation metrics belong in `wandb.run.summary`.
- `train/val/test` must share the same normalization, while augmentation is only applied to training data.
- Reproducibility is part of the project contract: keep all random behavior under `SEED`.
- `.cursorrules_original` stores the original Cursor rules for this project.
- `.cursorrules` stores the rewritten, project-finalized rules that reflect the current reproducible workflow.

## Experiment History

This repository currently records two experiment setups:

- A manual baseline configuration, treated as the initial human-authored setup
- An agent-improved configuration, chosen after reviewing the baseline evaluation results

### Baseline Experiment

This run corresponds to the original setup recorded under `wandb/run-20260320_211051-mka09767/files`.

Configuration:

- Model: 3 convolution layers, 1 max-pooling layer, large fully connected classifier
- Batch size: `64`
- Learning rate: `0.001`
- Epochs: `5`
- Optimizer: `Adam`
- Weight decay: none
- Label smoothing: none
- Augmentation: `RandomHorizontalFlip`, `ColorJitter`
- Evaluation checkpoint: `cifar_model.pth`

Results:

- Validation accuracy: `0.7384`
- Test top-1 accuracy: `0.7369`
- Test top-3 accuracy: `0.9371`
- Test top-5 accuracy: `0.9819`
- Macro F1: `0.7360`
- Weighted F1: `0.7360`

Notable weaknesses from the baseline evaluation:

- `bird`: `0.5650` per-class accuracy
- `cat`: `0.5700` per-class accuracy
- `dog`: `0.6520` per-class accuracy
- `deer`: `0.6840` per-class accuracy

### Agent-Improved Experiment

This run corresponds to W&B run `19uxt7z9` (`young-elevator-2`).

Why it was changed:

- The baseline model pooled too late, which left a very large classifier head and a weak hierarchical feature extractor.
- The baseline training schedule was too short for CIFAR-10.
- The weakest classes were visually fine-grained animal classes, which usually benefit from stronger local feature extraction and stronger augmentation.

Configuration:

- Model: 3 convolution blocks, 6 convolution layers total
- Normalization layers: `BatchNorm2d` after each convolution
- Pooling: `MaxPool2d` after each block
- Head: `AdaptiveAvgPool2d(1)` + dropout + linear classifier
- Batch size: `128`
- Learning rate: `3e-4`
- Epochs: `12`
- Optimizer: `AdamW`
- Weight decay: `1e-4`
- Label smoothing: `0.1`
- Scheduler: `CosineAnnealingLR`
- Augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `ColorJitter`
- Training checkpoint: `cifar_model_agent_last.pth`
- Best validation checkpoint: `cifar_model_agent_best.pth`

Results:

- Best validation accuracy: `0.8640`
- Test top-1 accuracy: `0.8548`
- Test top-3 accuracy: `0.9748`
- Test top-5 accuracy: `0.9932`
- Macro precision: `0.8544`
- Macro recall: `0.8548`
- Macro F1: `0.8545`
- Weighted F1: `0.8545`

Selected per-class improvements over baseline:

- `bird`: `0.5650 -> 0.7980`
- `cat`: `0.5700 -> 0.7070`
- `dog`: `0.6520 -> 0.8060`
- `deer`: `0.6840 -> 0.8480`

### Summary Comparison

| Metric | Baseline | Agent-improved | Delta |
| --- | ---: | ---: | ---: |
| Validation accuracy | 0.7384 | 0.8640 | +0.1256 |
| Test top-1 accuracy | 0.7369 | 0.8548 | +0.1179 |
| Test top-3 accuracy | 0.9371 | 0.9748 | +0.0377 |
| Test top-5 accuracy | 0.9819 | 0.9932 | +0.0113 |
| Macro F1 | 0.7360 | 0.8545 | +0.1185 |

The second setup is the better default for this repository because it improves both overall accuracy and the weakest animal classes without making the workflow significantly more complicated.
