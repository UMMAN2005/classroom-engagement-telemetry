import shutil
import yaml
from pathlib import Path


CLASS_DIRS = ["0_oriented", "1_diverted", "2_obscured"]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_split_dirs(crops_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in CLASS_DIRS:
            (crops_dir / split / cls).mkdir(parents=True, exist_ok=True)


def split_data(config: dict, project_root: Path) -> None:
    crops_dir = project_root / config["data"]["cropped_persons_dir"]

    train_stems = [Path(v).stem for v in config["split"]["train_videos"]]
    val_test_stems = [Path(v).stem for v in config["split"]["val_test_video"]]

    ensure_split_dirs(crops_dir)

    all_crops = sorted(f for f in crops_dir.glob("*.jpg") if f.is_file())
    if not all_crops:
        print(f"No .jpg files found in {crops_dir}")
        return

    train_files = []
    val_test_files = []

    for crop in all_crops:
        name = crop.name
        if any(name.startswith(stem) for stem in train_stems):
            train_files.append(crop)
        elif any(name.startswith(stem) for stem in val_test_stems):
            val_test_files.append(crop)

    for f in train_files:
        shutil.move(str(f), str(crops_dir / "train" / f.name))

    val_test_files.sort(key=lambda p: p.name)
    midpoint = len(val_test_files) // 2

    val_files = val_test_files[:midpoint]
    test_files = val_test_files[midpoint:]

    for f in val_files:
        shutil.move(str(f), str(crops_dir / "val" / f.name))

    for f in test_files:
        shutil.move(str(f), str(crops_dir / "test" / f.name))

    print("Split complete.")
    print(f"  Train: {len(train_files)} crops")
    print(f"  Val:   {len(val_files)} crops")
    print(f"  Test:  {len(test_files)} crops")
    print(f"  Total: {len(train_files) + len(val_files) + len(test_files)} crops")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    split_data(config, project_root)
