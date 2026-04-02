import cv2
import yaml
from pathlib import Path
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_frames(config: dict, project_root: Path) -> None:
    video_dir = project_root / config["data"]["video_dir"]
    output_dir = project_root / config["data"]["extracted_frames_dir"]
    interval = config["data"]["frame_interval_seconds"]

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return

    for video_path in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: could not open {video_path.name}, skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = int(fps * interval)

        if frame_skip <= 0:
            print(f"Warning: invalid FPS ({fps}) for {video_path.name}, skipping.")
            cap.release()
            continue

        saved_count = 0
        for frame_idx in tqdm(
            range(0, total_frames, frame_skip),
            desc=f"  {video_path.name}",
            leave=False,
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            out_name = f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_dir / out_name), frame)
            saved_count += 1

        cap.release()
        print(f"  -> {video_path.name}: extracted {saved_count} frames")

    print("Frame extraction complete.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    extract_frames(config, project_root)
