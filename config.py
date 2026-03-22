"""
Project-level configuration shared across scripts.
"""

BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 12
VAL_SPLIT = 0.1
SEED = 42

NUM_CLASSES = 10
MODEL_PATH = "cifar_model.pth"
MODEL_PATH_BEST = "cifar_model_best.pth"

WANDB_PROJECT = "First-Vibe-Project"

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# CIFAR-10 normalization constants (commonly used defaults)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Keep workers conservative on Windows to avoid multiprocessing issues.
NUM_WORKERS = 0
