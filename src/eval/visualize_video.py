import cv2
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from torchvision import transforms, models
from ultralytics import YOLO
from tqdm import tqdm


CLASS_NAMES = ["0_oriented", "1_diverted", "2_obscured"]
PERSON_CLASS = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BOX_COLORS = {
    0: (0, 200, 0),
    1: (0, 0, 220),
    2: (180, 180, 180),
}
CLASS_LABELS = {0: "Oriented", 1: "Diverted", 2: "Obscured"}

DURATION_SECONDS = 30
FACE_RATIO = 0.30
BLUR_KSIZE = (99, 99)
BLUR_SIGMA = 30


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
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
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


def blur_face_region(frame, x1, y1, x2, y2):
    """Apply Gaussian blur to the top portion of a box."""
    face_y2 = y1 + int((y2 - y1) * FACE_RATIO)
    face_y2 = min(face_y2, y2)
    if face_y2 > y1 and x2 > x1:
        face_roi = frame[y1:face_y2, x1:x2]
        frame[y1:face_y2, x1:x2] = cv2.GaussianBlur(
            face_roi, BLUR_KSIZE, BLUR_SIGMA
        )


def draw_hud(frame, frame_idx, total_frames, counts, engagement):
    """Draw a semi-transparent telemetry HUD in the top-left corner."""
    overlay = frame.copy()
    hud_w, hud_h = 320, 160
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = 20, 35
    line_h = 25

    cv2.putText(
        frame, f"Frame: {frame_idx + 1}/{total_frames}",
        (x0, y0), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Oriented:  {counts[0]}",
        (x0, y0 + line_h), font, 0.55, BOX_COLORS[0], 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Diverted:  {counts[1]}",
        (x0, y0 + 2 * line_h), font, 0.55, BOX_COLORS[1], 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Obscured:  {counts[2]}",
        (x0, y0 + 3 * line_h), font, 0.55, BOX_COLORS[2], 1, cv2.LINE_AA,
    )

    bar_x, bar_y = x0, y0 + 4 * line_h
    bar_w, bar_max = 200, 200
    filled = int(bar_max * engagement)
    bg_end = (bar_x + bar_w, bar_y + 4)
    cv2.rectangle(
        frame, (bar_x, bar_y - 12), bg_end,
        (80, 80, 80), -1,
    )
    if filled > 0:
        fill_end = (bar_x + filled, bar_y + 4)
        cv2.rectangle(
            frame, (bar_x, bar_y - 12), fill_end,
            (0, 200, 0), -1,
        )
    txt_x = bar_x + bar_w + 8
    cv2.putText(
        frame, f"Engagement: {engagement:.0%}",
        (txt_x, bar_y + 2), font, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    confidence = config["pipeline"]["detection_confidence_threshold"]
    input_size = config["model"]["classifier_input_size"]

    val_test_videos = config["split"]["val_test_video"]
    video_dir = project_root / config["data"]["video_dir"]
    video_path = video_dir / val_test_videos[0]

    if not video_path.exists():
        print(f"ERROR: test video not found at {video_path}")
        return

    detector_path = project_root / config["model"]["detector"]
    detector = YOLO(str(detector_path))

    weights_path = project_root / "src" / "models" / "best_baseline.pth"
    classifier = build_classifier(
        weights_path, num_classes=len(CLASS_NAMES), device=device,
    )
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
    out_path = results_dir / "classroom_demo_anonymized.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = detector(frame, conf=confidence, verbose=False)

        counts = [0, 0, 0]
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

            for i, (_, bx1, by1, bx2, by2) in enumerate(crops_with_boxes):
                cls = preds[i].item()
                counts[cls] += 1

                blur_face_region(frame, bx1, by1, bx2, by2)

                color = BOX_COLORS[cls]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                label = CLASS_LABELS[cls]
                cv2.putText(
                    frame, label, (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

        oriented = counts[0]
        diverted = counts[1]
        engagement = oriented / (oriented + diverted + 1e-4)
        draw_hud(frame, frame_idx, total_frames, counts, engagement)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {out_path}")


if __name__ == "__main__":
    main()
