from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_train_history(history: Dict[str, List[float]], output_path: str, title: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(results_df: pd.DataFrame, output_path: str) -> None:
    metric_cols = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    chart_df = results_df[["model"] + metric_cols].set_index("model")

    ax = chart_df.plot(kind="bar", figsize=(10, 5), ylim=(0.0, 1.0), rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    output_path: str,
    title: str,
) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
