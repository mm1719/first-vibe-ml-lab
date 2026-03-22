from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

from config import BATCH_SIZE, CIFAR10_MEAN, CIFAR10_STD, NUM_WORKERS, SEED, VAL_SPLIT


def _build_cifar10_dataset(*, train: bool, transform):
    try:
        return datasets.CIFAR10(
            root="./data",
            train=train,
            download=False,
            transform=transform,
        )
    except RuntimeError as exc:
        split = "train" if train else "test"
        raise RuntimeError(
            f"CIFAR-10 {split} split not found under ./data. "
            "Download the dataset manually before running this script."
        ) from exc


def _build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    # No augmentation for validation/test, only normalization.
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    # For visualization we don't want normalization.
    display_transform = transforms.Compose([transforms.ToTensor()])

    return train_transform, eval_transform, display_transform


def get_dataloaders(batch_size: int = BATCH_SIZE):
    train_transform, eval_transform, _ = _build_transforms()

    # Create two datasets over the same underlying samples:
    # - train transform includes augmentation
    # - val transform includes only normalization
    full_train_dataset_aug = _build_cifar10_dataset(train=True, transform=train_transform)
    full_train_dataset_eval = _build_cifar10_dataset(train=True, transform=eval_transform)

    # Random split: use 10% of train as validation.
    n = len(full_train_dataset_aug)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset_aug, train_indices)
    val_dataset = Subset(full_train_dataset_eval, val_indices)

    test_dataset = _build_cifar10_dataset(train=False, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_test_datasets_for_inference():
    _, test_transform, display_transform = _build_transforms()

    test_dataset_pred = _build_cifar10_dataset(train=False, transform=test_transform)
    test_dataset_disp = _build_cifar10_dataset(train=False, transform=display_transform)

    return test_dataset_pred, test_dataset_disp
