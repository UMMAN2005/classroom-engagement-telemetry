"""Create and deploy a Gradio Space on Hugging Face with GradCAM."""

import tempfile
from huggingface_hub import HfApi


SPACE_REQUIREMENTS = """\
torch
torchvision
gradio
huggingface-hub
Pillow
grad-cam
opencv-python-headless
numpy
"""

APP_PY = """\
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_REPO = "ummanmm/classroom-reaction-resnet18"
CLASSES = [
    "Neutral",
    "Confused",
    "Smiling / Amused",
    "Surprised",
    "Bored / Tired",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

model_path = hf_hub_download(
    repo_id=MODEL_REPO, filename="best_resnet18.pth"
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(
    torch.load(model_path, map_location=torch.device("cpu"))
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict_reaction(image):
    if image is None:
        return {}, None

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs[0], dim=0)

    confidences = {CLASSES[i]: float(probs[i]) for i in range(5)}

    rgb = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale = cam(input_tensor=tensor, targets=None)[0, :]
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)

    return confidences, heatmap


demo = gr.Interface(
    fn=predict_reaction,
    inputs=gr.Image(label="Upload Student Crop"),
    outputs=[
        gr.Label(num_top_classes=5, label="Reaction Prediction"),
        gr.Image(label="Grad-CAM Attention Heatmap"),
    ],
    title="Classroom Reaction Recognition",
    description=(
        "Upload a cropped image of a student to classify their "
        "facial reaction using a ResNet18 model. "
        "The Grad-CAM heatmap shows which regions the CNN focuses on."
    ),
    examples=[],
)

demo.launch()
"""


def main():
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return

    space_name = "Classroom-Reaction-Demo"
    repo_id = f"{username}/{space_name}"
    api = HfApi()

    print(f"Creating Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(APP_PY)
        app_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write(SPACE_REQUIREMENTS)
        req_path = f.name

    print("Uploading app.py...")
    api.upload_file(
        path_or_fileobj=app_path,
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
    )

    print("Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj=req_path,
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\nDone! Your Space is building at:")
    print(f"  {url}")
    print(
        "\nIt takes 2-3 minutes for Hugging Face to install "
        "dependencies and launch the app."
    )


if __name__ == "__main__":
    main()
