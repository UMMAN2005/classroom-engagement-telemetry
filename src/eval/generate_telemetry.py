import cv2
import yaml
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torchvision import transforms, models
from ultralytics import YOLO
from tqdm import tqdm


PERSON_CLASS = 0
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


def build_classifier(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def get_test_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def classify_crops(crops, transform, model, device, num_classes):
    counts = [0] * num_classes
    if not crops:
        return counts

    batch = torch.stack([transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)
        _, preds = outputs.max(1)

    for p in preds.cpu().numpy():
        counts[p] += 1

    return counts


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    class_names = config["classes"]
    num_classes = len(class_names)
    confidence = config["pipeline"]["detection_confidence_threshold"]
    input_size = config["model"]["classifier_input_size"]

    detector_path = project_root / config["model"]["detector"]
    detector = YOLO(str(detector_path))

    weights_path = project_root / "src" / "models" / "best_resnet18.pth"
    classifier = build_classifier(weights_path, num_classes=num_classes, device=device)
    transform = get_test_transform(input_size)

    frames_dir = project_root / config["data"]["extracted_frames_dir"]

    all_frames = sorted(frames_dir.glob("*.jpg"))
    if not all_frames:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Processing {len(all_frames)} frames")

    frame_indices = []
    per_class_counts = {cls: [] for cls in class_names}

    for idx, frame_path in enumerate(tqdm(all_frames, desc="Telemetry inference")):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        results = detector(str(frame_path), conf=confidence, verbose=False)

        crops = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crops.append(image[y1:y2, x1:x2])

        counts = classify_crops(crops, transform, classifier, device, num_classes)

        frame_indices.append(idx)
        for i, cls in enumerate(class_names):
            per_class_counts[cls].append(counts[i])

    df_data = {"frame_index": frame_indices}
    df_data.update(per_class_counts)
    df = pd.DataFrame(df_data)

    total_per_frame = df[class_names].sum(axis=1).replace(0, 1)
    for cls in class_names:
        df[f"{cls}_ratio"] = df[cls] / total_per_frame

    print("\nTelemetry summary:")
    print(f"  Frames processed: {len(df)}")
    for cls in class_names:
        mean_ratio = df[f"{cls}_ratio"].mean()
        print(f"  Mean {cls} ratio: {mean_ratio:.3f}")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    colors = ["#95a5a6", "#e67e22", "#2ecc71", "#f1c40f", "#e74c3c"]

    for i, cls in enumerate(class_names):
        axes[0].plot(
            df["frame_index"], df[cls],
            color=colors[i % len(colors)], linewidth=1.2, label=cls, alpha=0.8,
        )
    axes[0].set_title("Reaction Counts per Frame", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right", fontsize=9)

    for i, cls in enumerate(class_names):
        smoothed = df[f"{cls}_ratio"].rolling(window=5, min_periods=1).mean()
        axes[1].plot(
            df["frame_index"], smoothed,
            color=colors[i % len(colors)], linewidth=2, label=cls,
        )
    axes[1].set_title("Smoothed Reaction Distribution over Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frame Index (chronological)")
    axes[1].set_ylabel("Proportion")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle("Classroom Reaction Telemetry", fontsize=14, fontweight="bold")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / "classroom_telemetry_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Dashboard saved to: {save_path}")


if __name__ == "__main__":
    main()
