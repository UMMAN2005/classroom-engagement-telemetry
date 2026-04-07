import csv
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
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


def build_transforms(input_size: int) -> tuple:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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


def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def build_vgg16(num_classes: int, device: torch.device) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features, num_classes,
    )
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


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    epochs: int,
    save_path: Path,
    history_path: Path,
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True,
    )
    best_val_acc = 0.0
    history = []

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {epochs} epochs")
    print(f"  Trainable params: {trainable:,} / {total:,}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_acc)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_path))
            tag = "  <-- best"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"LR: {cur_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}{tag}"
        )

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writerows(history)
    print(f"Training history saved to: {history_path}")
    print(f"{model_name} complete. Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best weights saved to: {save_path}")

    return best_val_acc


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    torch.manual_seed(config["project"]["random_seed"])
    device = get_device()
    print(f"Device: {device}")

    class_names = config["classes"]
    num_classes = len(class_names)
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

    targets = torch.tensor(train_dataset.targets)
    sample_weights = class_weights.cpu()[targets]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset) * 2, replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    models_dir = project_root / "src" / "models"

    resnet = build_resnet18(num_classes, device)
    train_model(
        "ResNet18", resnet, train_loader, val_loader, criterion, device, lr, epochs,
        save_path=models_dir / "best_resnet18.pth",
        history_path=results_dir / "history_resnet18.csv",
    )

    vgg = build_vgg16(num_classes, device)
    train_model(
        "VGG16", vgg, train_loader, val_loader, criterion, device, lr, epochs,
        save_path=models_dir / "best_vgg16.pth",
        history_path=results_dir / "history_vgg16.csv",
    )

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
