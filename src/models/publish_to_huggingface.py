import tempfile
from pathlib import Path
from huggingface_hub import HfApi


REPO_NAME = "classroom-engagement-mobilenet"

MODEL_CARD = """\
---
license: mit
tags:
  - pytorch
  - mobilenetv2
  - image-classification
  - classroom-engagement
---

# Classroom Engagement MobileNetV2

A lightweight MobileNetV2 classifier trained to estimate classroom
engagement from cropped person bounding boxes.

## Classes

| Index | Label         | Description                              |
|-------|---------------|------------------------------------------|
| 0     | `0_oriented`  | Student facing the instructor/board      |
| 1     | `1_diverted`  | Student turned away or looking elsewhere |
| 2     | `2_obscured`  | Student occluded or not clearly visible  |

## Architecture

- **Backbone:** MobileNetV2 (ImageNet pre-trained, features frozen)
- **Classifier head:** `nn.Linear(1280, 3)`
- **Loss:** Weighted `CrossEntropyLoss` to handle class imbalance
- **Optimizer:** Adam (lr=0.001)
- **Input size:** 224 x 224 RGB, ImageNet-normalized

## Training

Trained on ~200 manually annotated person crops extracted from
classroom lecture videos using YOLOv8-nano detection. Strict
video-level train/val/test split to prevent data leakage.

## Usage

```python
import torch
from torchvision import models, transforms

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 3)
model.load_state_dict(torch.load("best_baseline.pth", map_location="cpu"))
model.eval()
```
"""


def main():
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "src" / "models" / "best_baseline.pth"

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
        path_in_repo="best_baseline.pth",
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
