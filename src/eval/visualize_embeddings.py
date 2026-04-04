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


def build_feature_extractor(
    weights_path: Path, device: torch.device,
) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_feat = model.last_channel
    model.classifier[1] = nn.Linear(in_feat, len(CLASS_NAMES))
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract 1280-dim bottleneck features."""
    all_features = []
    all_labels = []

    pool = nn.AdaptiveAvgPool2d(1)

    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        feat_maps = model.features(images)
        pooled = pool(feat_maps).flatten(1)
        all_features.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    input_size = config["model"]["classifier_input_size"]
    batch_size = config["model"]["batch_size"]
    seed = config["project"]["random_seed"]

    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    test_dir = str(crops_dir / "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    weights_path = project_root / "src" / "models" / "best_baseline.pth"
    model = build_feature_extractor(weights_path, device)

    features, labels = extract_embeddings(model, test_loader, device)
    print(f"Feature matrix shape: {features.shape}")

    perplexity = min(30, len(features) - 1)
    print(f"Running t-SNE (perplexity={perplexity}, seed={seed})...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(features)

    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[cls_idx],
            label=cls_name,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(
        "t-SNE: MobileNetV2 Bottleneck Features (Test Set)",
        fontsize=13, fontweight="bold",
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
