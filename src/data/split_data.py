import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_data(config: dict, project_root: Path) -> None:
    crops_dir = project_root / config["data"]["cropped_persons_dir"]
    staging_dir = project_root / config["data"]["staging_dir"]
    class_names = config["classes"]
    seed = config["project"]["random_seed"]
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]

    for split in ("train", "val", "test"):
        for cls in class_names:
            (crops_dir / split / cls).mkdir(parents=True, exist_ok=True)

    all_files = []
    all_labels = []

    for cls in class_names:
        cls_dir = staging_dir / cls
        if not cls_dir.exists():
            print(
                f"Warning: {cls_dir} does not exist, skipping."
            )
            continue
        for img in sorted(cls_dir.glob("*.jpg")):
            all_files.append(img)
            all_labels.append(cls)

    if not all_files:
        print(
            f"No images found in staging subdirectories "
            f"under {staging_dir}"
        )
        return

    val_test_ratio = 1.0 - train_ratio
    relative_val = val_ratio / val_test_ratio

    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        all_files, all_labels,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=seed,
    )

    val_files, test_files, _, _ = train_test_split(
        val_test_files, val_test_labels,
        test_size=(1.0 - relative_val),
        stratify=val_test_labels,
        random_state=seed,
    )

    for f in train_files:
        cls = f.parent.name
        shutil.move(str(f), str(crops_dir / "train" / cls / f.name))

    for f in val_files:
        cls = f.parent.name
        shutil.move(str(f), str(crops_dir / "val" / cls / f.name))

    for f in test_files:
        cls = f.parent.name
        shutil.move(str(f), str(crops_dir / "test" / cls / f.name))

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
