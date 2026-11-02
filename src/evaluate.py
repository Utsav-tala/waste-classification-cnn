"""
Final evaluation of the selected model on the held-out test set.
Run this ONLY after model selection is complete (Stage 7).

Usage:
    python -m src.evaluate model_b        # original Model B (recipe v1)
    python -m src.evaluate model_c_v2     # any run name saved by src.train

Outputs are suffixed with the run name, e.g. confusion_matrix_model_c_v2.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.config import (
    BATCH_SIZE,
    CLASSES,
    CLASS_TO_IDX,
    DATASET_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_DIR,
)

BASE_DIR = DATASET_DIR.parent


def build_test_dataset(csv_path: Path, image_size: tuple, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Builds a deterministic (no shuffle) dataset from the test CSV."""
    df = pd.read_csv(csv_path)
    abs_paths = (BASE_DIR / df["filepath"]).astype(str).tolist()
    labels = df["label"].tolist()
    height, width = image_size

    def _load(fp, lb):
        raw = tf.io.read_file(fp)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [height, width], method="bilinear")
        return tf.cast(img, tf.uint8), lb

    ds = tf.data.Dataset.from_tensor_slices((abs_paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, title_suffix: str = "Test Set"):
    """Saves a normalised annotated confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    sns.heatmap(
        cm_norm,
        annot=cm,          # show raw counts inside cells
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix — {title_suffix} (raw counts, normalised colour)",
                 fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(report_dict: dict, output_path: Path):
    """
    Saves a horizontal bar chart of per-class F1-scores.
    This makes it immediately obvious which classes the model struggles with
    (e.g. plastic, glass, trash often have low F1 due to inter-class confusion).
    """
    class_f1 = {cls: report_dict[cls]["f1-score"] for cls in CLASSES}
    classes = list(class_f1.keys())
    f1_scores = list(class_f1.values())

    # Colour bars by performance: green (good) → orange (moderate) → red (poor)
    colors = []
    for f1 in f1_scores:
        if f1 >= 0.70:
            colors.append("#22c55e")   # green — good
        elif f1 >= 0.50:
            colors.append("#f59e0b")   # amber — moderate
        else:
            colors.append("#ef4444")   # red — poor

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.barh(classes, f1_scores, color=colors, edgecolor="white", height=0.55)

    for bar, f1 in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}", va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class F1-Score — Test Set\n(green ≥0.70 · amber ≥0.50 · red <0.50)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.axvline(x=0.50, color="#9ca3af", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(x=0.70, color="#9ca3af", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_evaluation(run_name: str = "model_b"):
    """
    Evaluates the selected model on the held-out test set.

    Parameters
    ----------
    run_name : str
        A run saved by src.train, e.g. 'model_b' or 'model_c_v2'.
    """
    model_path = MODELS_DIR / f"{run_name}_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run src/train.py first to train and save the model."
        )

    print("=" * 60)
    print(f"STAGE 7: FINAL EVALUATION — {run_name.upper()}")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    image_size = tuple(model.input_shape[1:3])   # each model knows its own input size
    print(f"Model loaded: {model_path} (input {image_size[0]}x{image_size[1]})")

    test_ds, true_labels = build_test_dataset(SPLITS_DIR / "test.csv", image_size)

    # Predict on the entire test set
    probs = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(probs, axis=1)

    # Metrics
    test_accuracy = float(np.mean(np.array(true_labels) == pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(CLASSES))))
    report_dict = classification_report(
        true_labels, pred_labels, target_names=CLASSES, output_dict=True
    )
    report_text = classification_report(
        true_labels, pred_labels, target_names=CLASSES
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("\nClassification Report:\n")
    print(report_text)

    # --- Save confusion matrix plot ---
    cm_path = PLOTS_DIR / f"confusion_matrix_{run_name}.png"
    plot_confusion_matrix(cm, cm_path, title_suffix=f"Test Set, {run_name}")
    print(f"Confusion matrix saved -> {cm_path}")

    # --- Save per-class F1 bar chart ---
    f1_path = PLOTS_DIR / f"per_class_f1_scores_{run_name}.png"
    plot_per_class_f1(report_dict, f1_path)
    print(f"Per-class F1 bar chart saved -> {f1_path}")

    # --- Save classification report (text) ---
    report_txt_path = REPORTS_DIR / f"classification_report_{run_name}.txt"
    with open(report_txt_path, "w") as f:
        f.write(f"Model evaluated: {model_path.relative_to(BASE_DIR)}\n")
        f.write(f"Test accuracy  : {test_accuracy:.4f}\n\n")
        f.write(report_text)
    print(f"Classification report (text) saved -> {report_txt_path}")

    # --- Save classification report (CSV) ---
    report_csv_rows = []
    for cls_name in CLASSES:
        m = report_dict[cls_name]
        report_csv_rows.append({
            "class": cls_name,
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1_score":  round(m["f1-score"],  4),
            "support":   int(m["support"]),
        })
    for avg in ["macro avg", "weighted avg"]:
        m = report_dict[avg]
        report_csv_rows.append({
            "class": avg,
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1_score":  round(m["f1-score"],  4),
            "support":   int(m["support"]),
        })
    report_csv_path = REPORTS_DIR / f"classification_report_{run_name}.csv"
    pd.DataFrame(report_csv_rows).to_csv(report_csv_path, index=False)
    print(f"Classification report (CSV) saved -> {report_csv_path}")

    # --- Save class labels ---
    labels_path = MODELS_DIR / "class_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": CLASSES, "class_to_idx": CLASS_TO_IDX}, f, indent=2)
    print(f"Class labels saved -> {labels_path}")

    # --- Update experiment comparison with test accuracy ---
    exp_path = REPORTS_DIR / "experiment_comparison.csv"
    if exp_path.exists():
        exp_df = pd.read_csv(exp_path, dtype={"test_accuracy": str})
        mask = exp_df["run_name"] == run_name
        exp_df.loc[mask, "test_accuracy"] = str(round(test_accuracy, 4))
        exp_df.to_csv(exp_path, index=False)
        print(f"Experiment comparison updated -> {exp_path}")

    # --- Analysis of most confused classes ---
    print("\nConfusion analysis (most confused class pairs):")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    for _ in range(4):
        i, j = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        if cm_copy[i, j] == 0:
            break
        print(f"  True: {CLASSES[i]:10s} → Predicted: {CLASSES[j]:10s}  ({cm_copy[i, j]} times)")
        cm_copy[i, j] = 0

    print("\nSTAGE 7 EVALUATION COMPLETE.")
    return test_accuracy, report_dict


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "model_b"
    run_evaluation(run)
