import cv2
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from torchvision import transforms, models
from ultralytics import YOLO
from tqdm import tqdm


PERSON_CLASS = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DURATION_SECONDS = 30


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_classifier(
    weights_path: Path, num_classes: int, device: torch.device,
) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
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


def draw_hud(frame, frame_idx, total_frames, counts, class_names):
    overlay = frame.copy()
    num_classes = len(class_names)
    hud_w, hud_h = 340, 40 + num_classes * 25 + 10
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = 20, 35
    line_h = 25

    cv2.putText(
        frame, f"Frame: {frame_idx + 1}/{total_frames}",
        (x0, y0), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )

    box_colors = [
        (180, 180, 180),
        (0, 100, 255),
        (0, 200, 0),
        (0, 255, 255),
        (0, 0, 220),
    ]

    for i, cls in enumerate(class_names):
        color = box_colors[i % len(box_colors)]
        cv2.putText(
            frame, f"{cls}: {counts[i]}",
            (x0, y0 + (i + 1) * line_h), font, 0.50, color, 1, cv2.LINE_AA,
        )


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    class_names = sorted(config["classes"])
    num_classes = len(class_names)
    confidence = config["pipeline"]["detection_confidence_threshold"]
    input_size = config["model"]["classifier_input_size"]

    video_dir = project_root / config["data"]["video_dir"]
    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"ERROR: no videos found in {video_dir}")
        return
    video_path = video_files[0]

    if not video_path.exists():
        print(f"ERROR: test video not found at {video_path}")
        return

    detector_path = project_root / config["model"]["detector"]
    detector = YOLO(str(detector_path))

    weights_path = project_root / "src" / "models" / "best_resnet18.pth"
    classifier = build_classifier(weights_path, num_classes=num_classes, device=device)
    transform = get_test_transform(input_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(fps * DURATION_SECONDS)

    print(f"Video: {video_path.name} ({width}x{height} @ {fps:.1f} FPS)")
    print(f"Processing first {DURATION_SECONDS}s ({total_frames} frames)")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "classroom_demo.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    box_colors = [
        (180, 180, 180),
        (0, 100, 255),
        (0, 200, 0),
        (0, 255, 255),
        (0, 0, 220),
    ]

    for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = detector(frame, conf=confidence, verbose=False)

        counts = [0] * num_classes
        crops_with_boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                crops_with_boxes.append((crop, x1, y1, x2, y2))

        if crops_with_boxes:
            crop_tensors = []
            for crop, *_ in crops_with_boxes:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_tensors.append(transform(rgb))
            batch = torch.stack(crop_tensors).to(device)

            with torch.no_grad():
                outputs = classifier(batch)
                _, preds = outputs.max(1)
                for i in range(len(preds)):
                    if preds[i] == 3:
                        outputs[i, 3] = float('-inf')
                        preds[i] = outputs[i].argmax()

            for i, (_, bx1, by1, bx2, by2) in enumerate(crops_with_boxes):
                cls = preds[i].item()
                counts[cls] += 1

                color = box_colors[cls % len(box_colors)]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                label = class_names[cls]
                cv2.putText(
                    frame, label, (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

        draw_hud(frame, frame_idx, total_frames, counts, class_names)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {out_path}")


if __name__ == "__main__":
    main()
