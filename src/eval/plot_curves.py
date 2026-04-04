import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    history_path = project_root / "results" / "history.csv"

    if not history_path.exists():
        print(f"ERROR: {history_path} not found. Run training first.")
        return

    df = pd.read_csv(str(history_path))

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        df["epoch"], df["train_loss"],
        marker="o", markersize=4, label="Train Loss",
    )
    ax1.plot(
        df["epoch"], df["val_loss"],
        marker="s", markersize=4, label="Val Loss",
    )
    ax1.set_title("Cross-Entropy Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(
        df["epoch"], df["train_acc"],
        marker="o", markersize=4, label="Train Acc",
    )
    ax2.plot(
        df["epoch"], df["val_acc"],
        marker="s", markersize=4, label="Val Acc",
    )
    ax2.set_title(
        "Classification Accuracy", fontsize=13, fontweight="bold",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    fig.suptitle(
        "MobileNetV2 Training Curves (15 Epochs, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "training_curves.png"
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    main()
