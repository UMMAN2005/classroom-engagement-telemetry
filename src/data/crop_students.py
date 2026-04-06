import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS = 0


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def crop_students(config: dict, project_root: Path) -> None:
    frames_dir = project_root / config["data"]["extracted_frames_dir"]
    staging_dir = project_root / config["data"]["staging_dir"]
    confidence = config["pipeline"]["detection_confidence_threshold"]

    staging_dir.mkdir(parents=True, exist_ok=True)

    detector_path = project_root / config["model"]["detector"]
    model = YOLO(str(detector_path))

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No .jpg files found in {frames_dir}")
        return

    total_crops = 0

    for frame_path in tqdm(frame_paths, desc="Detecting & cropping persons"):
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Warning: could not read {frame_path.name}, skipping.")
            continue

        h, w = image.shape[:2]

        results = model(str(frame_path), conf=confidence, verbose=False)

        crop_idx = 0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                out_name = f"{frame_path.stem}_crop_{crop_idx}.jpg"
                cv2.imwrite(str(staging_dir / out_name), crop)
                crop_idx += 1

        total_crops += crop_idx

    print(f"Cropping complete. Total person crops saved: {total_crops}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    crop_students(config, project_root)
