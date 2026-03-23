# First Vibe ML Lab

A small PyTorch-based machine learning experiment workflow for CIFAR-10.

This repository is my first vibe coding project. It is meant to be a compact, reproducible baseline for running small ML experiments with a clear training, inference, and evaluation pipeline.

## Environment Setup

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

## Project Layout And Conventions

- `scripts/train.py`, `scripts/inference.py`, and `scripts/evaluation.py` are the CLI entrypoints.
- `src/data/dataset.py` owns data loading, transforms, and dataset splits.
- `src/models/model.py` owns the model definition.
- `src/utils/evaluation.py` owns evaluation metrics, per-class reporting, and confusion matrices.
- `src/utils/reproducibility.py` owns shared seed setup.
- `configs/config.py` centralizes shared project configuration.
- Training metrics belong in `wandb.log`; final evaluation metrics belong in `wandb.run.summary`.
- `train/val/test` share normalization, while augmentation is only applied to training data.
- Reproducibility is part of the project contract: keep random behavior under `SEED`.
- `.cursorrules_original` stores the initial Cursor rules, while `.cursorrules` stores the maintained project rules.

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

- Final checkpoint: `checkpoints/cifar_model.pth`
- Best validation checkpoint: `checkpoints/cifar_model_best.pth`
- Agent experiment checkpoints: `checkpoints/cifar_model_agent_last.pth`, `checkpoints/cifar_model_agent_best.pth`
- Inference visualization: `outputs/plots/inference_samples.png`
- Confusion matrix figures: `outputs/plots/confusion_matrix_counts_*.png`, `outputs/plots/confusion_matrix_normalized_*.png`

## Notes

- The dataset is loaded with `download=False` by design.
- Random behavior is controlled through `SEED` for reproducibility.
- This repo intentionally keeps the model simple so the workflow is the main focus.

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
- Evaluation checkpoint: `checkpoints/cifar_model.pth`

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
- Training checkpoint: `checkpoints/cifar_model_agent_last.pth`
- Best validation checkpoint: `checkpoints/cifar_model_agent_best.pth`

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

## Development Notes

The full development process of this project was a useful reminder that a small ML lab is not just about getting a model to train once. The first stage was building a minimal end-to-end baseline that could train, run inference, and produce evaluation outputs. After that, the work shifted toward reproducibility: consolidating configuration, making data handling deterministic, separating training metrics from final evaluation metrics, and writing the rules down clearly enough that the workflow could be repeated instead of rediscovered.

The second stage was model improvement. Looking at the baseline evaluation results made it clear that the first model and training schedule were too weak for several visually similar CIFAR-10 classes, so the project moved from a rough proof of concept to a more deliberate experiment setup with a stronger CNN backbone, better augmentation, and a longer, better-regularized training process. That improvement mattered not only because the top-line metrics got better, but because the project started to behave more like a real experiment workflow instead of a one-off script.

The last stage was project hardening. Once the codebase was refactored into `configs`, `src`, and `scripts`, the main challenge was no longer model quality but operational reliability: import paths, script entrypoints, WandB artifact compatibility, output locations, Git hygiene, and the reality that the project is run from an Anaconda terminal rather than an abstract clean environment. The main takeaway from the whole process is that good project structure is not just aesthetic. It has to line up with how experiments are actually executed, tracked, debugged, and handed off in practice.
