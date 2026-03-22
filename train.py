import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from config import BATCH_SIZE, EPOCHS, LR, MODEL_PATH, MODEL_PATH_BEST, NUM_CLASSES, SEED, WANDB_PROJECT
from dataset import get_dataloaders
from model import SimpleCNN
from reproducibility import set_seed


def train(model_path: str = MODEL_PATH, best_model_path: str = MODEL_PATH_BEST):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

    # CrossEntropyLoss:
    # L = - sum_{i=1}^{n} y_i * log(ŷ_i)
    # In PyTorch, it expects raw logits (no softmax), and internally applies log-softmax + NLL.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    wandb.init(project=WANDB_PROJECT)
    wandb.config.update(
        {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "device": str(device),
            "num_classes": NUM_CLASSES,
            "seed": SEED,
            "model_path": model_path,
            "best_model_path": best_model_path,
        }
    )

    best_val_acc = float("-inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_running_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # 每個 Epoch 結束後上傳訓練/驗證準確度與損失。
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), model_path)
    wandb.finish()
    print(f"Saved final model weights to {model_path}")
    print(f"Saved best model weights to {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to save the final model checkpoint.",
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default=MODEL_PATH_BEST,
        help="Path to save the best validation checkpoint.",
    )
    args = parser.parse_args()
    train(model_path=args.model_path, best_model_path=args.best_model_path)
