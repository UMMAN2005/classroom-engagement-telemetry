import csv
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


CLASS_NAMES = ["0_oriented", "1_diverted", "2_obscured"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels = []
    all_preds = []
    all_confs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confs, preds = probs.max(1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_confs)


def save_predictions_csv(dataset, y_true, y_pred, confs, class_names, save_path: Path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label", "confidence"])
        for i in range(len(y_true)):
            filepath = Path(dataset.samples[i][0]).name
            writer.writerow([
                filepath,
                class_names[y_true[i]],
                class_names[y_pred[i]],
                f"{confs[i]:.4f}",
            ])
    print(f"Predictions CSV saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Baseline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    input_size = config["model"]["classifier_input_size"]
    batch_size = config["model"]["batch_size"]

    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    test_dataset = datasets.ImageFolder(str(crops_dir / "test"), transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    weights_path = project_root / "src" / "models" / "best_baseline.pth"
    model = build_model(weights_path, num_classes=len(CLASS_NAMES), device=device)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    y_true, y_pred, confs = collect_predictions(model, test_loader, device)

    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(
        y_true,
        y_pred,
        CLASS_NAMES,
        save_path=results_dir / "baseline_confusion_matrix.png",
    )

    save_predictions_csv(
        test_dataset, y_true, y_pred, confs, CLASS_NAMES,
        save_path=results_dir / "test_predictions.csv",
    )


if __name__ == "__main__":
    main()
