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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm


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


def build_resnet18(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def build_vgg16(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
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


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def evaluate_model(model_name, model, test_loader, test_dataset, class_names, device, results_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    y_true, y_pred, confs = collect_predictions(model, test_loader, device)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title=f"{model_name} Confusion Matrix",
        save_path=results_dir / f"{model_name.lower()}_confusion_matrix.png",
    )

    save_predictions_csv(
        test_dataset, y_true, y_pred, confs, class_names,
        save_path=results_dir / f"{model_name.lower()}_test_predictions.csv",
    )

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    class_names = config["classes"]
    num_classes = len(class_names)
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

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    models_dir = project_root / "src" / "models"

    comparison = []

    resnet_path = models_dir / "best_resnet18.pth"
    if resnet_path.exists():
        resnet = build_resnet18(resnet_path, num_classes, device)
        comparison.append(
            evaluate_model("ResNet18", resnet, test_loader, test_dataset, class_names, device, results_dir)
        )

    vgg_path = models_dir / "best_vgg16.pth"
    if vgg_path.exists():
        vgg = build_vgg16(vgg_path, num_classes, device)
        comparison.append(
            evaluate_model("VGG16", vgg, test_loader, test_dataset, class_names, device, results_dir)
        )

    if comparison:
        print(f"\n{'='*60}")
        print("Side-by-Side Model Comparison")
        print(f"{'='*60}")
        header = f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        print(header)
        print("-" * len(header))
        for row in comparison:
            print(
                f"{row['model']:<12} {row['accuracy']:>10.4f} "
                f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}"
            )

        comp_path = results_dir / "model_comparison.csv"
        with open(comp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall", "f1"])
            writer.writeheader()
            writer.writerows(comparison)
        print(f"\nComparison saved to: {comp_path}")


if __name__ == "__main__":
    main()
