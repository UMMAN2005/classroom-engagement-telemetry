import cv2
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torchvision import transforms, models
from ultralytics import YOLO
from tqdm import tqdm


CLASS_NAMES = ["0_oriented", "1_diverted", "2_obscured"]
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
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
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


def classify_crops(crops, transform, model, device):
    """Classify a list of BGR crop arrays. Returns counts per class."""
    counts = [0, 0, 0]
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

    confidence = config["pipeline"]["detection_confidence_threshold"]
    input_size = config["model"]["classifier_input_size"]

    detector_path = project_root / config["model"]["detector"]
    detector = YOLO(str(detector_path))

    weights_path = project_root / "src" / "models" / "best_baseline.pth"
    classifier = build_classifier(weights_path, num_classes=len(CLASS_NAMES), device=device)
    transform = get_test_transform(input_size)

    frames_dir = project_root / config["data"]["extracted_frames_dir"]
    val_test_stems = [Path(v).stem for v in config["split"]["val_test_video"]]

    all_frames = sorted(frames_dir.glob("*.jpg"))
    target_frames = [f for f in all_frames if any(f.name.startswith(s) for s in val_test_stems)]

    if not target_frames:
        print(f"No frames found for val_test videos {val_test_stems}")
        return

    print(f"Processing {len(target_frames)} frames from {val_test_stems}")

    frame_indices = []
    oriented_counts = []
    diverted_counts = []
    obscured_counts = []

    for idx, frame_path in enumerate(tqdm(target_frames, desc="Telemetry inference")):
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

        counts = classify_crops(crops, transform, classifier, device)

        frame_indices.append(idx)
        oriented_counts.append(counts[0])
        diverted_counts.append(counts[1])
        obscured_counts.append(counts[2])

    df = pd.DataFrame({
        "frame_index": frame_indices,
        "oriented": oriented_counts,
        "diverted": diverted_counts,
        "obscured": obscured_counts,
    })

    df["Engagement_Ratio"] = df["oriented"] / (df["oriented"] + df["diverted"] + 0.0001)
    df["Smoothed_Engagement"] = df["Engagement_Ratio"].rolling(window=5, min_periods=1).mean()

    print(f"\nTelemetry summary:")
    print(f"  Frames processed : {len(df)}")
    print(f"  Mean engagement  : {df['Engagement_Ratio'].mean():.3f}")
    print(f"  Min engagement   : {df['Engagement_Ratio'].min():.3f}")
    print(f"  Max engagement   : {df['Engagement_Ratio'].max():.3f}")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df["frame_index"], df["Engagement_Ratio"],
        color="steelblue", alpha=0.3, linewidth=1, label="Raw Engagement",
    )
    ax.plot(
        df["frame_index"], df["Smoothed_Engagement"],
        color="steelblue", linewidth=2.5, label="Smoothed (window=5)",
    )

    ax.set_title("Classroom Engagement Telemetry over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frame Index (chronological)")
    ax.set_ylabel("Engagement Ratio (oriented / engaged)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / "classroom_telemetry_dashboard.png"
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Dashboard saved to: {save_path}")


if __name__ == "__main__":
    main()
