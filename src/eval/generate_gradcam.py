import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from torchvision import datasets, transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


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


def build_model(
    weights_path: Path, num_classes: int, device: torch.device
) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel, num_classes
    )
    model.load_state_dict(
        torch.load(str(weights_path), map_location=device)
    )
    model.to(device)
    model.eval()
    return model


def find_candidate_images(model, dataset, transform, device):
    """Find a True Positive Diverted and a False Positive Diverted image."""
    tp_diverted = None  # true=1, pred=1
    fp_diverted = None  # true=0, pred=1

    for idx in range(len(dataset)):
        img_path, true_label = dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor).argmax(1).item()

        if true_label == 1 and pred == 1 and tp_diverted is None:
            tp_diverted = (idx, img_path)
        elif true_label == 0 and pred == 1 and fp_diverted is None:
            fp_diverted = (idx, img_path)

        if tp_diverted and fp_diverted:
            break

    return tp_diverted, fp_diverted


def generate_cam_overlay(model, target_layer, img_path, transform, device):
    """Generate a GradCAM heatmap overlay for a single image."""
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    rgb_img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay, rgb_img


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")

    device = get_device()
    print(f"Device: {device}")

    input_size = config["model"]["classifier_input_size"]

    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    test_dataset = datasets.ImageFolder(str(crops_dir / "test"))

    weights_path = (
        project_root / "src" / "models" / "best_baseline.pth"
    )
    model = build_model(
        weights_path, num_classes=len(CLASS_NAMES), device=device
    )

    print("Searching for candidate images...")
    tp, fp = find_candidate_images(model, test_dataset, test_tf, device)

    if tp is None:
        print("WARNING: No True Positive Diverted image found in test set.")
        return
    if fp is None:
        print("WARNING: No False Positive Diverted image found in test set.")
        return

    print(f"  TP Diverted: {Path(tp[1]).name}")
    print(f"  FP Diverted: {Path(fp[1]).name}")

    target_layer = model.features[-1]

    tp_overlay, tp_orig = generate_cam_overlay(
        model, target_layer, tp[1], test_tf, device
    )
    fp_overlay, fp_orig = generate_cam_overlay(
        model, target_layer, fp[1], test_tf, device
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0][0].imshow(tp_orig)
    axes[0][0].set_title("Original", fontsize=11)
    axes[0][0].set_ylabel(
        "Success: Correctly\nIdentified Diverted",
        fontsize=11, fontweight="bold",
    )
    axes[0][0].axis("off")

    axes[0][1].imshow(tp_overlay)
    axes[0][1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[0][1].axis("off")

    axes[1][0].imshow(fp_orig)
    axes[1][0].set_title("Original", fontsize=11)
    axes[1][0].set_ylabel(
        "Failure: Oriented\nMisclassified as Diverted",
        fontsize=11, fontweight="bold",
    )
    axes[1][0].axis("off")

    axes[1][1].imshow(fp_overlay)
    axes[1][1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1][1].axis("off")

    fig.suptitle(
        "Grad-CAM Analysis: MobileNetV2 Attention Maps",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / "gradcam_analysis.png"
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Grad-CAM analysis saved to: {save_path}")


if __name__ == "__main__":
    main()
