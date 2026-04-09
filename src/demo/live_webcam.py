"""Real-time classroom engagement dashboard (ONNX + YOLOv8).

Usage:  python src/demo/live_webcam.py
Keys:   Q/ESC quit, S screenshot
"""

import sys
import time
import types
from collections import deque
from pathlib import Path

if "_bz2" not in sys.modules:
    try:
        import _bz2  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("_bz2")
        _stub.BZ2Compressor = None
        _stub.BZ2Decompressor = None
        sys.modules["_bz2"] = _stub

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISPLAY_NAMES = ["Neutral", "Confused", "Smiling / Amused", "Surprised", "Bored / Tired"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BAR_COLORS = [
    (180, 180, 180),
    (0, 140, 255),
    (0, 220, 120),
    (0, 200, 255),
    (200, 80, 80),
]

ENGAGEMENT_WEIGHTS = {0: 50, 1: 15, 2: 95, 3: 75, 4: 5}
ALERT_THRESHOLD = 35
HISTORY_SECONDS = 30
WINDOW = "Classroom Engagement Dashboard"


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def preprocess(crop_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1)[np.newaxis]


def draw_confidence_bars(frame, probs, x, y, w=160, h=14, gap=4):
    for i, (prob, name, color) in enumerate(zip(probs, DISPLAY_NAMES, BAR_COLORS)):
        yy = y + i * (h + gap)
        bar_w = int(prob * w)
        cv2.rectangle(frame, (x, yy), (x + w, yy + h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, yy), (x + bar_w, yy + h), color, -1)
        cv2.putText(
            frame, f"{name}: {prob:.0%}", (x + 4, yy + h - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )


def engagement_color(score: float) -> tuple:
    if score < 50:
        r = 0
        g = int(score / 50 * 200)
        b = int((1 - score / 50) * 255)
    else:
        r = 0
        g = int(200 + (score - 50) / 50 * 55)
        b = 0
    return (b, g, r)


def draw_gauge(frame, score, x, y, w=220, h=28):
    label = f"Engagement: {score:.0f}%"
    color = engagement_color(score)

    cv2.rectangle(frame, (x - 4, y - 22), (x + w + 4, y + h + 6), (0, 0, 0), -1)
    cv2.putText(
        frame, label, (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(score / 100 * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def draw_summary(frame, counts, total, y=30):
    parts = []
    for i, name in enumerate(DISPLAY_NAMES):
        if counts[i] > 0:
            short = name.split("/")[0].strip()
            parts.append(f"{counts[i]} {short}")
    summary = f"{total} detected" + (f":  {', '.join(parts)}" if parts else "")

    (tw, th), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cx = frame.shape[1] // 2 - tw // 2
    cv2.rectangle(frame, (cx - 8, y - th - 6), (cx + tw + 8, y + 6), (0, 0, 0), -1)
    cv2.putText(
        frame, summary, (cx, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_history(frame, history, x, y, w=260, h=80):
    if len(history) < 2:
        return

    panel = frame.copy()
    cv2.rectangle(panel, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame, f"Last {HISTORY_SECONDS}s", (x + 4, y + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
    )

    for pct in (25, 50, 75):
        gy = y + h - int(pct / 100 * (h - 20)) - 2
        cv2.line(frame, (x, gy), (x + w, gy), (50, 50, 50), 1)

    pts = list(history)
    n = len(pts)
    step = w / max(n - 1, 1)
    points = []
    for i, val in enumerate(pts):
        px = int(x + i * step)
        py = int(y + h - val / 100 * (h - 20) - 2)
        points.append((px, py))

    for i in range(len(points) - 1):
        color = engagement_color(pts[i + 1])
        cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_alert_border(frame, score):
    if score >= ALERT_THRESHOLD:
        return
    pulse = int(abs(np.sin(time.time() * 4)) * 200) + 55
    color = (0, 0, pulse)
    thickness = 8
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    label = "LOW ENGAGEMENT"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cx = w // 2 - tw // 2
    cy = h - 30
    cv2.putText(
        frame, label, (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, pulse), 3, cv2.LINE_AA,
    )


def main():
    config = load_config()
    input_size = config["model"]["classifier_input_size"]
    det_conf = config["pipeline"]["detection_confidence_threshold"]

    onnx_path = PROJECT_ROOT / "src" / "models" / "model.onnx"
    if not onnx_path.exists():
        sys.exit(
            f"ONNX model not found at {onnx_path}.\n"
            "Run  make benchmark  first to export it."
        )

    detector = YOLO(str(PROJECT_ROOT / config["model"]["detector"]))
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam (device 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Dashboard started. Press Q or ESC to quit, S to screenshot.")

    fps_avg = 0.0
    alpha = 0.1
    engagement_smooth = 50.0

    max_history = HISTORY_SECONDS * 10  # ~10 fps estimate
    history = deque(maxlen=max_history)

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, conf=det_conf, verbose=False)[0]

        reaction_counts = [0] * 5
        person_scores = []

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tensor = preprocess(crop, input_size)
            logits = session.run(None, {input_name: tensor})[0][0]
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()

            top_idx = int(probs.argmax())
            reaction_counts[top_idx] += 1

            person_engagement = sum(
                float(probs[c]) * ENGAGEMENT_WEIGHTS[c] for c in range(5)
            )
            person_scores.append(person_engagement)

            color = BAR_COLORS[top_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{DISPLAY_NAMES[top_idx]} {probs[top_idx]:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

            bar_x = x2 + 8 if x2 + 170 < frame.shape[1] else x1 - 170
            draw_confidence_bars(frame, probs, bar_x, y1)

        total = sum(reaction_counts)
        if person_scores:
            raw_engagement = np.mean(person_scores)
        else:
            raw_engagement = engagement_smooth
        engagement_smooth = 0.3 * raw_engagement + 0.7 * engagement_smooth
        history.append(engagement_smooth)

        h, w = frame.shape[:2]

        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_avg = alpha * fps + (1 - alpha) * fps_avg if fps_avg else fps
        cv2.putText(
            frame, f"FPS: {fps_avg:.1f}", (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )

        draw_summary(frame, reaction_counts, total)
        draw_gauge(frame, engagement_smooth, w - 234, 8)
        draw_history(frame, history, w - 274, h - 100)
        draw_alert_border(frame, engagement_smooth)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            shot_path = PROJECT_ROOT / "results" / f"dashboard_{int(time.time())}.png"
            cv2.imwrite(str(shot_path), frame)
            print(f"Screenshot saved: {shot_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
