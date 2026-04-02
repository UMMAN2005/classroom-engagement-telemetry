import yaml
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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


def build_transforms(input_size: int) -> tuple:
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    """weight_i = total_samples / (num_classes * count_i)"""
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    print(f"  Class counts : {dict(counts)}")
    print(f"  Class weights: {[round(w, 3) for w in weights]}")
    return torch.FloatTensor(weights).to(device)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    torch.manual_seed(config["project"]["random_seed"])
    device = get_device()
    print(f"Device: {device}")

    input_size = config["model"]["classifier_input_size"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    lr = config["model"]["learning_rate"]

    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    train_tf, val_tf = build_transforms(input_size)

    train_dataset = datasets.ImageFolder(str(crops_dir / "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(crops_dir / "val"), transform=val_tf)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=len(CLASS_NAMES), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = project_root / "src" / "models" / "best_baseline.pth"
    best_val_acc = 0.0

    print(f"\nStarting training for {epochs} epochs (batch_size={batch_size}, lr={lr})\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_path))
            tag = "  <-- best"

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}{tag}"
        )

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best weights saved to: {save_path}")


if __name__ == "__main__":
    main()
