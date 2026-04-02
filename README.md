# Classroom Engagement Telemetry

A lightweight deep learning pipeline for estimating classroom engagement from video telemetry using YOLOv8 person detection and MobileNetV2 posture classification.

## Pipeline Overview

```plain
Raw Videos → Frame Extraction → Person Detection/Cropping → Train/Val/Test Split
→ MobileNetV2 Training → Evaluation → Engagement Telemetry Dashboard
```

## Setup

```bash
make setup
```

## Pipeline Steps (run in order)

| Step | Command                    | Description                                                     |
| ---- | -------------------------- | --------------------------------------------------------------- |
| 1    | `make extract-frames`      | Extract one frame every 10 seconds from raw videos              |
| 2    | `make crop-students`       | Run YOLOv8n person detection and crop bounding boxes            |
| 3    | `make split-data`          | Split crops into train/val/test by video source                 |
| 4    | *(manual)*                 | Annotate crops into `0_oriented/`, `1_diverted/`, `2_obscured/` |
| 5    | `make train-baseline`      | Train MobileNetV2 classifier with weighted loss                 |
| 6    | `make evaluate-baseline`   | Evaluate on test set, generate confusion matrix and CSV         |
| 7    | `make generate-telemetry`  | Run chronological inference and plot engagement dashboard       |
| 8    | `make generate-gradcam`    | Generate Grad-CAM attention heatmaps for error analysis         |

## Directory Structure

```plain
classroom_engagement_telemetry/
├── data/
│   ├── raw_videos/                     # Source .mp4 files (git-ignored)
│   ├── extracted_frames/               # One frame per 10 seconds
│   └── crops/
│       ├── train/
│       │   ├── 0_oriented/
│       │   ├── 1_diverted/
│       │   └── 2_obscured/
│       ├── val/
│       │   ├── 0_oriented/
│       │   ├── 1_diverted/
│       │   └── 2_obscured/
│       └── test/
│           ├── 0_oriented/
│           ├── 1_diverted/
│           └── 2_obscured/
├── src/
│   ├── data/
│   │   ├── extract_frames.py           # Downsample video to 1 frame / 10s
│   │   ├── crop_students.py            # YOLOv8n person detection and cropping
│   │   └── split_data.py               # Strict video-level train/val/test split
│   ├── models/
│   │   ├── train_baseline.py           # MobileNetV2 training with weighted loss
│   │   ├── publish_to_huggingface.py   # Upload weights to Hugging Face Hub
│   │   ├── best_baseline.pth           # Saved best weights (git-ignored)
│   │   └── yolov8n.pt                  # YOLOv8-nano detector (git-ignored, auto-downloads)
│   └── eval/
│       ├── evaluate_baseline.py        # Test-set metrics, confusion matrix, CSV
│       ├── generate_telemetry.py       # Chronological engagement dashboard
│       └── generate_gradcam.py         # Grad-CAM attention heatmaps
├── results/                            # Generated pipeline outputs
│   ├── baseline_confusion_matrix.png
│   ├── test_predictions.csv
│   ├── classroom_telemetry_dashboard.png
│   └── gradcam_analysis.png
├── notebooks/
│   └── EDA_and_Testing.ipynb           # Exploratory data analysis notebook
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions lint pipeline
├── config.yaml                         # Centralized hyperparameters and paths
├── requirements.txt                    # Python dependencies
├── Makefile                            # Pipeline orchestration commands
├── LICENSE
├── .gitignore
└── README.md
```

## Outputs

- `src/models/best_baseline.pth` -- Best model weights (by val accuracy)
- `results/baseline_confusion_matrix.png` -- Per-class precision/recall visualization
- `results/test_predictions.csv` -- Per-image predictions with confidence scores for error analysis
- `results/classroom_telemetry_dashboard.png` -- Engagement ratio over time with temporal smoothing
- `results/gradcam_analysis.png` -- Grad-CAM attention heatmaps comparing correct vs. misclassified crops

## Configuration

All hyperparameters and paths are centralized in `config.yaml`.

## Requirements

Python 3.10+ with packages listed in `requirements.txt`. Install via `make setup`.

## Design Rationale (Edge/CPU Constraints)

This pipeline was intentionally designed to run on local, consumer-grade hardware (e.g., AMD Ryzen 7 CPU) without requiring cloud GPU instances.

- **Extraction:** 10-second downsampling reduces 4GB of raw video to lightweight chronological batches.
- **Models:** YOLOv8-nano and MobileNetV2 were explicitly chosen for their highly optimized, depthwise-separable convolutions, allowing for rapid CPU inference.

## Ethical Considerations & Privacy

This pipeline is designed strictly for **Observability**, not surveillance.

- **No Biometrics:** The system does not perform facial recognition or identity tracking.
- **Posture-Only:** Classification is based purely on geometric posture heuristics (Oriented vs. Diverted).
- **Aggregate Telemetry:** Outputs are smoothed into macro-level classroom engagement ratios. No individual student data or images are stored in the final telemetry outputs.
