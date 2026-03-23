import argparse
import random
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import torch
import matplotlib.pyplot as plt

from configs.config import MODEL_PATH, NUM_CLASSES, SEED
from src.models.model import SimpleCNN
from src.data.dataset import get_test_datasets_for_inference
from src.utils.reproducibility import set_seed


def main(model_path: str = MODEL_PATH):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

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

    test_dataset_pred, test_dataset_disp = get_test_datasets_for_inference()

    # Randomly sample 4 images from test set.
    indices = random.sample(range(len(test_dataset_pred)), k=4)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax_i, idx in zip(axes, indices):
        img_norm, true_label = test_dataset_pred[idx]  # normalized tensor
        img_disp, _ = test_dataset_disp[idx]  # unnormalized tensor for display

        with torch.no_grad():
            logits = model(img_norm.unsqueeze(0).to(device))
            pred_label = int(torch.argmax(logits, dim=1).item())

        img_np = img_disp.permute(1, 2, 0).numpy()
        img_np = img_np.clip(0.0, 1.0)

        ax_i.imshow(img_np)
        ax_i.axis("off")
        ax_i.set_title(f"GT: {classes[true_label]}\nPred: {classes[pred_label]}", fontsize=10)

    plt.tight_layout()
    out_path = Path("outputs/plots/inference_samples.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to the model checkpoint used for inference.",
    )
    args = parser.parse_args()
    main(model_path=args.model_path)
