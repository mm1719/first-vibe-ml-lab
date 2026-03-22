import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import wandb

from config import BATCH_SIZE, CIFAR10_MEAN, CIFAR10_STD, MODEL_PATH, NUM_CLASSES, NUM_WORKERS, SEED, WANDB_PROJECT
from model import SimpleCNN
from reproducibility import set_seed


TOPK_LIST = (1, 3, 5)


def _build_eval_dataset(transform):
    try:
        return datasets.CIFAR10(
            root="./data",
            train=False,
            download=False,
            transform=transform,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "CIFAR-10 test split not found under ./data. "
            "Download the dataset manually before running evaluation."
        ) from exc


def _build_eval_dataloader(batch_size: int = BATCH_SIZE) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    test_dataset = _build_eval_dataset(transform=transform)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def _confusion_matrix_from_preds(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels, preds: shape [N], values in [0, num_classes)
    return: confusion matrix shape [C, C] where row=GT, col=Pred
    """
    idx = labels * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def evaluate(resume_run_id: str | None = None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = _build_eval_dataloader(batch_size=BATCH_SIZE)

    total = 0
    correct_top1 = 0
    correct_topk = {k: 0 for k in TOPK_LIST}

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # [B, 10]
        preds_top1 = torch.argmax(logits, dim=1)  # [B]

        confusion += _confusion_matrix_from_preds(labels.cpu(), preds_top1.cpu(), NUM_CLASSES)

        total += labels.size(0)
        correct_top1 += (preds_top1 == labels).sum().item()

        max_k = max(TOPK_LIST)
        topk_indices = torch.topk(logits, k=max_k, dim=1).indices  # [B, max_k]

        for k in TOPK_LIST:
            in_topk = (topk_indices[:, :k] == labels.unsqueeze(1)).any(dim=1)
            correct_topk[k] += in_topk.sum().item()

    # Metrics
    test_acc_top1 = correct_top1 / total
    topk_acc = {k: correct_topk[k] / total for k in TOPK_LIST}

    # classification report derived from confusion matrix
    # row=GT, col=Pred
    cm = confusion.float()
    row_sum = cm.sum(dim=1)  # GT counts per class
    col_sum = cm.sum(dim=0)  # Pred counts per class

    tp = torch.diag(cm)
    fp = col_sum - tp
    fn = row_sum - tp

    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(tp),
    )

    per_class_acc = torch.where(row_sum > 0, tp / row_sum, torch.zeros_like(tp))

    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    weighted_f1 = (f1 * (row_sum / row_sum.sum().clamp_min(1.0))).sum().item()

    # Print summary
    print(f"Test accuracy (top-1): {test_acc_top1:.4f}")
    for k in TOPK_LIST:
        print(f"Test accuracy (top-{k}): {topk_acc[k]:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    for i in range(NUM_CLASSES):
        print(
            f"[Class {i}] "
            f"Acc: {per_class_acc[i].item():.4f}, "
            f"Precision: {precision[i].item():.4f}, "
            f"Recall: {recall[i].item():.4f}, "
            f"F1: {f1[i].item():.4f}"
        )

    # Confusion matrix plot
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    cm_counts = confusion.cpu().numpy()
    cm_norm = cm_counts.astype(float)
    cm_row = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (cm_row + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes[0].imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (counts)")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(classes, rotation=45, ha="right")
    axes[0].set_yticklabels(classes)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, int(cm_counts[i, j]), ha="center", va="center", fontsize=8)

    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title("Confusion Matrix (normalized by GT)")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_yticklabels(classes)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    counts_path = f"confusion_matrix_counts_{ts}.png"
    norm_path = f"confusion_matrix_normalized_{ts}.png"

    # Save two separate images (for nicer wandb display)
    # 1) Counts only
    plt.figure(figsize=(7, 7))
    plt.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (counts)")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), classes, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), classes)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, int(cm_counts[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(counts_path, dpi=150)
    plt.close()

    # 2) Normalized only
    plt.figure(figsize=(7, 7))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (normalized by GT)")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), classes, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), classes)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=150)
    plt.close("all")

    # WandB logging
    if resume_run_id:
        # Resume the existing training run so evaluation metrics show up in the same Run.
        wandb.init(project=WANDB_PROJECT, id=resume_run_id, resume="allow")
    else:
        wandb.init(project=WANDB_PROJECT, name=f"test_evaluation_{ts}")

    # Scalar metrics: store as run-level summary (not as time-series charts).
    wandb.run.summary.update(
        {
            "test_acc_top1": test_acc_top1,
            "test_acc_top3": topk_acc[3],
            "test_acc_top5": topk_acc[5],
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    )

    # Per-class metrics: store as a table (one row per class).
    per_class_table = wandb.Table(
        columns=["class", "per_class_acc", "precision", "recall", "f1"],
    )
    for i in range(NUM_CLASSES):
        per_class_table.add_data(
            classes[i],
            per_class_acc[i].item(),
            precision[i].item(),
            recall[i].item(),
            f1[i].item(),
        )

    wandb.log(
        {
            "per_class_metrics": per_class_table,
            "confusion_matrix_counts": wandb.Image(counts_path),
            "confusion_matrix_normalized": wandb.Image(norm_path),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="When provided, evaluation logs will be appended to the existing W&B run id (resume).",
    )
    args = parser.parse_args()
    evaluate(resume_run_id=args.resume_run_id)
