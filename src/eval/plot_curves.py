import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_model_curves(df, model_name, axes_row):
    ax_loss, ax_acc = axes_row

    ax_loss.plot(
        df["epoch"], df["train_loss"],
        marker="o", markersize=4, label="Train Loss",
    )
    ax_loss.plot(
        df["epoch"], df["val_loss"],
        marker="s", markersize=4, label="Val Loss",
    )
    ax_loss.set_title(f"{model_name} - Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(
        df["epoch"], df["train_acc"],
        marker="o", markersize=4, label="Train Acc",
    )
    ax_acc.plot(
        df["epoch"], df["val_acc"],
        marker="s", markersize=4, label="Val Acc",
    )
    ax_acc.set_title(f"{model_name} - Accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend()


def main():
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"

    resnet_path = results_dir / "history_resnet18.csv"
    vgg_path = results_dir / "history_vgg16.csv"

    has_resnet = resnet_path.exists()
    has_vgg = vgg_path.exists()

    if not has_resnet and not has_vgg:
        print("ERROR: no history CSVs found. Run training first.")
        return

    sns.set_style("whitegrid")

    if has_resnet and has_vgg:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        df_resnet = pd.read_csv(str(resnet_path))
        df_vgg = pd.read_csv(str(vgg_path))
        plot_model_curves(df_resnet, "ResNet18", axes[0])
        plot_model_curves(df_vgg, "VGG16", axes[1])
        fig.suptitle(
            "ResNet18 vs VGG16 Training Curves",
            fontsize=14, fontweight="bold",
        )
    elif has_resnet:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        df_resnet = pd.read_csv(str(resnet_path))
        plot_model_curves(df_resnet, "ResNet18", axes)
        fig.suptitle("ResNet18 Training Curves", fontsize=14, fontweight="bold")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        df_vgg = pd.read_csv(str(vgg_path))
        plot_model_curves(df_vgg, "VGG16", axes)
        fig.suptitle("VGG16 Training Curves", fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "training_curves.png"
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    main()
