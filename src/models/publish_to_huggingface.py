import tempfile
from pathlib import Path
from huggingface_hub import HfApi


REPO_NAME = "classroom-reaction-resnet18"

MODEL_CARD = """\
---
license: mit
tags:
  - pytorch
  - resnet18
  - image-classification
  - reaction-recognition
---

# Classroom Reaction Recognition - ResNet18

A ResNet18 classifier trained to recognize facial reactions of students
from cropped person bounding boxes in classroom lecture videos.

## Classes

| Index | Label            | Description                        |
|-------|------------------|------------------------------------|
| 0     | `Neutral`        | No visible expression              |
| 1     | `Confused`       | Furrowed brow, squinting           |
| 2     | `Smiling_Amused` | Visible smile, laughter            |
| 3     | `Surprised`      | Raised eyebrows, open mouth        |
| 4     | `Bored_Tired`    | Yawning, blank stare               |

## Architecture

- **Backbone:** ResNet18 (ImageNet pre-trained, all layers frozen except fc)
- **Classifier head:** `nn.Linear(512, 5)`
- **Loss:** Weighted `CrossEntropyLoss` to handle class imbalance
- **Optimizer:** Adam (lr=0.001)
- **Input size:** 224 x 224 RGB, ImageNet-normalized

## Training

Trained on manually annotated person crops extracted from classroom
lecture videos using YOLOv8-nano detection. 70/10/20 stratified
train/val/test split.

## Usage

```python
import torch
from torchvision import models

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 5)
model.load_state_dict(torch.load("best_resnet18.pth", map_location="cpu"))
model.eval()
```
"""


def main():
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "src" / "models" / "best_resnet18.pth"

    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        return

    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return

    repo_id = f"{username}/{REPO_NAME}"
    api = HfApi()

    print(f"Creating repo: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True)

    print("Uploading model weights...")
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="best_resnet18.pth",
        repo_id=repo_id,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        card_path = f.name

    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    print("\nDone! Model published at:")
    print(f"  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
