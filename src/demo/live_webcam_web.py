"""Browser-based webcam demo (Gradio). Usage: make live-demo-web"""

import os
import sys
import types

os.environ["NNPACK_SUPPRESS_WARNINGS"] = "1"

if "_bz2" not in sys.modules:
    try:
        import _bz2  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("_bz2")
        _stub.BZ2Compressor = None
        _stub.BZ2Decompressor = None
        sys.modules["_bz2"] = _stub

import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")

import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_REPO = "ummanmm/classroom-reaction-resnet18"
ONNX_REPO = "ummanmm/classroom-reaction-resnet18-onnx"

DISPLAY_NAMES = ["Neutral", "Confused", "Smiling / Amused", "Surprised", "Bored / Tired"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BAR_COLORS_RGB = [
    (180, 180, 180),
    (255, 140, 0),
    (120, 220, 0),
    (255, 200, 0),
    (80, 80, 200),
]

USE_ONNX = False

onnx_path = PROJECT_ROOT / "src" / "models" / "model.onnx"
if not onnx_path.exists():
    try:
        onnx_path = Path(hf_hub_download(repo_id=ONNX_REPO, filename="model.onnx"))
    except Exception:
        onnx_path = None

if onnx_path and onnx_path.exists():
    import onnxruntime as ort
    SESSION = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    INPUT_NAME = SESSION.get_inputs()[0].name
    USE_ONNX = True
    print("Using ONNX runtime.")
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    weights_path = PROJECT_ROOT / "src" / "models" / "best_resnet18.pth"
    if not weights_path.exists():
        weights_path = Path(hf_hub_download(repo_id=PYTORCH_REPO, filename="best_resnet18.pth"))

    PT_MODEL = models.resnet18(weights=None)
    PT_MODEL.fc = nn.Linear(PT_MODEL.fc.in_features, 5)
    PT_MODEL.load_state_dict(torch.load(str(weights_path), map_location="cpu", weights_only=True))
    PT_MODEL.eval()

    PT_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Using PyTorch runtime.")


def preprocess_onnx(frame_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame_rgb, (224, 224)).astype(np.float32) / 255.0
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)[np.newaxis]


def infer(frame_rgb: np.ndarray) -> np.ndarray:
    if USE_ONNX:
        tensor = preprocess_onnx(frame_rgb)
        logits = SESSION.run(None, {INPUT_NAME: tensor})[0][0]
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()
    else:
        tensor = PT_TRANSFORM(frame_rgb).unsqueeze(0)
        with torch.no_grad():
            return F.softmax(PT_MODEL(tensor)[0], dim=0).numpy()


def draw_overlay(frame: np.ndarray, probs: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    top_idx = int(probs.argmax())
    label = f"{DISPLAY_NAMES[top_idx]}  {probs[top_idx]:.0%}"

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)
    cv2.putText(
        overlay, label, (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, BAR_COLORS_RGB[top_idx], 2, cv2.LINE_AA,
    )

    bar_w, bar_h, gap = 200, 18, 5
    x0, y0 = 12, h - len(DISPLAY_NAMES) * (bar_h + gap) - 10

    panel = overlay.copy()
    cv2.rectangle(panel, (x0 - 6, y0 - 6), (x0 + bar_w + 6, h - 4), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, overlay, 0.5, 0, overlay)

    for i, (prob, name, color) in enumerate(zip(probs, DISPLAY_NAMES, BAR_COLORS_RGB)):
        yy = y0 + i * (bar_h + gap)
        fill_w = int(prob * bar_w)
        cv2.rectangle(overlay, (x0, yy), (x0 + bar_w, yy + bar_h), (60, 60, 60), -1)
        cv2.rectangle(overlay, (x0, yy), (x0 + fill_w, yy + bar_h), color, -1)
        cv2.putText(
            overlay, f"{name}: {prob:.0%}", (x0 + 4, yy + bar_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return overlay


def classify_frame(frame):
    if frame is None:
        return None
    return draw_overlay(frame, infer(frame))


runtime = "ONNX Runtime" if USE_ONNX else "PyTorch"

demo = gr.Interface(
    fn=classify_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=gr.Image(label="Reaction Detection"),
    live=True,
    title="Classroom Reaction -- Live Demo",
    description=(
        f"Stand in front of your webcam. The ResNet18 model ({runtime}) classifies "
        "your facial reaction in real-time. No GPU required."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
