import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
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


def build_feature_extractor(
    weights_path: Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract 512-dim features from ResNet18 avgpool."""
    all_features = []
    all_labels = []

    def _forward_features(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x.flatten(1)

    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        feats = _forward_features(images)
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    class_names = config["classes"]
    num_classes = len(class_names)
    input_size = config["model"]["classifier_input_size"]
    batch_size = config["model"]["batch_size"]
    seed = config["project"]["random_seed"]

    test_tf = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    test_dir = str(crops_dir / "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    weights_path = project_root / "src" / "models" / "best_resnet18.pth"
    model = build_feature_extractor(weights_path, num_classes, device)

    features, labels = extract_embeddings(model, test_loader, device)
    print(f"Feature matrix shape: {features.shape}")

    perplexity = min(30, len(features) - 1)
    print(f"Running t-SNE (perplexity={perplexity}, seed={seed})...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(features)

    colors = ["#95a5a6", "#e67e22", "#2ecc71", "#f1c40f", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(9, 7))

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[cls_idx % len(colors)],
            label=cls_name,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(
        "t-SNE: ResNet18 Features (Test Set)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", fontsize=10)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "tsne_clusters.png"
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"t-SNE visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
