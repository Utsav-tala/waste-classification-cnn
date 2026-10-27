"""
Training pipeline for Waste Classification CNN.

Builds tf.data.Dataset pipelines (no full in-memory caching of decoded images),
trains both Model A and Model B with early stopping, saves histories, training
curves, and an experiment comparison table.  The test set is never touched here.

Improvements over the initial version:
  - Prints per-class weight breakdown so class imbalance is clearly visible
  - Generates a validation confusion matrix after training each model so
    inter-class confusions (e.g. plastic ↔ metal) are caught before Stage 7
  - Increased epoch budget (50) with longer patience to let augmented data work
"""
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.config import (
    BATCH_SIZE,
    CLASSES,
    CLASS_TO_IDX,
    CLASS_WEIGHT_MULTIPLIERS,
    DATASET_DIR,
    EARLY_STOPPING_PATIENCE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.model import build_model_a, build_model_b

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------------------
# Base dir for resolving relative paths stored in CSVs
# -------------------------------------------------------------------
BASE_DIR = DATASET_DIR.parent


def _load_and_preprocess(filepath_tensor, label_tensor):
    """
    Reads one image from disk and decodes it.
    No augmentation here — augmentation lives inside the model graph.
    Images are returned as uint8 so the model's Rescaling layer normalises them.
    """
    raw = tf.io.read_file(filepath_tensor)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], method="bilinear")
    img = tf.cast(img, tf.uint8)
    return img, label_tensor


def build_dataset(
    csv_path: Path,
    shuffle: bool = False,
    repeat: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset from a split CSV.
    Images are loaded lazily (no full decode into RAM).
    pipeline: read paths → load/resize → batch → prefetch
    """
    df = pd.read_csv(csv_path)
    abs_paths = (BASE_DIR / df["filepath"]).astype(str).tolist()
    labels = df["label"].tolist()

    path_ds = tf.data.Dataset.from_tensor_slices(abs_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    return ds


def compute_class_weights_from_csv(csv_path: Path, multipliers: dict = None) -> dict:
    """
    Computes class weights from the training CSV with custom multipliers.
    w_j = (n_samples / (n_classes * count_j)) * multiplier_j

    This ensures minority classes (trash) and challenging/confused classes
    (glass, plastic, cardboard) receive proportionally higher loss contributions.
    """
    if multipliers is None:
        multipliers = CLASS_WEIGHT_MULTIPLIERS

    df = pd.read_csv(csv_path)
    n_classes = len(CLASSES)
    n_total = len(df)
    weights = {}
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        count = (df["class_name"] == cls_name).sum()
        base_w = float(n_total) / (n_classes * count)
        mult = multipliers.get(cls_name, 1.0)
        weights[cls_idx] = base_w * mult
    return weights


def print_class_weight_analysis(class_weights: dict, csv_path: Path):
    """
    Prints a human-readable table of class weights alongside sample counts
    so imbalance and custom boosts are immediately visible in training logs.
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_classes = len(CLASSES)

    print("\n" + "─" * 68)
    print("CLASS WEIGHT ANALYSIS & BOOSTS (Training Set)")
    print("─" * 68)
    print(f"  {'Class':<11} {'Count':>6} {'Share':>7} {'Base W':>8} {'Mult':>6} {'Final W':>9}  {'Status'}")
    print("  " + "─" * 62)

    for cls_name in CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        count = (df["class_name"] == cls_name).sum()
        share = count / n_total * 100
        base_w = float(n_total) / (n_classes * count)
        w = class_weights[cls_idx]
        mult = CLASS_WEIGHT_MULTIPLIERS.get(cls_name, 1.0)
        if mult > 1.2:
            status = "↑↑ BOOSTED (high priority)"
        elif mult > 1.0:
            status = "↑ BOOSTED"
        elif mult < 1.0:
            status = "↓ dampened (majority)"
        else:
            status = "≈ neutral"
        print(f"  {cls_name:<11} {count:>6} {share:>6.1f}% {base_w:>8.2f} {mult:>6.2f} {w:>9.2f}  {status}")

    print("  " + "─" * 62)
    print(f"  Total training images: {n_total}")
    print("─" * 68 + "\n")


def plot_training_curves(history: dict, model_name: str, output_path: Path):
    """Saves a 2-panel (accuracy + loss) training-curve figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    epochs = range(1, len(history["accuracy"]) + 1)

    ax1.plot(epochs, history["accuracy"], "o-", label="Train Accuracy", color="#2196F3", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], "s--", label="Val Accuracy", color="#FF5722", linewidth=2)
    ax1.set_title(f"{model_name}\nAccuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["loss"], "o-", label="Train Loss", color="#4CAF50", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#9C27B0", linewidth=2)
    ax2.set_title(f"{model_name}\nLoss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_validation_confusion_matrix(
    model, model_label: str, slug: str, val_ds: tf.data.Dataset, val_labels: list
):
    """
    Generates a confusion matrix from validation predictions after training.
    This lets the user see inter-class confusion (e.g. plastic → metal)
    without waiting for the full Stage 7 test evaluation.
    """
    print(f"\n  Generating validation confusion matrix for {model_label}...")
    probs = model.predict(val_ds, verbose=0)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.array(val_labels)

    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    sns.heatmap(
        cm_norm,
        annot=cm,
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
    ax.set_title(
        f"Validation Confusion Matrix — {model_label}\n(raw counts, normalised colour)",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()

    cm_path = PLOTS_DIR / f"val_confusion_matrix_{slug}.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"  Validation confusion matrix saved -> {cm_path}")

    # Print the top confusion pairs for quick inspection
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    print(f"\n  Top confusion pairs ({model_label}, validation set):")
    for _ in range(5):
        i, j = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        if cm_copy[i, j] == 0:
            break
        print(f"    True: {CLASSES[i]:10s} → Predicted: {CLASSES[j]:10s}  ({cm_copy[i, j]} times)")
        cm_copy[i, j] = 0


def train_model(model, model_label: str, class_weights: dict):
    """
    Trains a compiled model on train/val splits.

    Uses:
      - EarlyStopping (patience=8): Stops training when val_loss hasn't
        improved for 8 epochs and restores the best weights.
      - ReduceLROnPlateau (patience=4, factor=0.5): Halves the learning
        rate when val_loss stalls, allowing finer-grained convergence.
      - class_weight: Computed per-class weights that boost minority classes
        (especially 'trash' at 3.07×) and dampen over-represented classes.

    Returns (history_dict, best_val_acc, best_val_loss, elapsed_seconds,
             epochs_run, val_ds, val_labels).
    """
    train_ds = build_dataset(SPLITS_DIR / "train.csv", shuffle=True)
    val_ds   = build_dataset(SPLITS_DIR / "val.csv",   shuffle=False)

    # Keep val labels for confusion matrix generation
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    val_labels = val_df["label"].tolist()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n{'=' * 60}")
    print(f"Training: {model_label}")
    print(f"  Max epochs:   {MAX_EPOCHS}")
    print(f"  Early stop:   patience={EARLY_STOPPING_PATIENCE} on val_loss")
    print(f"  LR reduction: factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}")
    print(f"  Class weights: ENABLED (see analysis above)")
    print(f"{'=' * 60}")
    t0 = time.time()
    history_obj = model.fit(
        train_ds,
        epochs=MAX_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    history = history_obj.history

    best_val_acc  = float(max(history["val_accuracy"]))
    best_val_loss = float(min(history["val_loss"]))
    epochs_run    = len(history["accuracy"])

    return history, best_val_acc, best_val_loss, elapsed, epochs_run, val_ds, val_labels


def run_training():
    """Main entry-point: trains both models, saves artefacts, prints comparison."""
    train_csv = SPLITS_DIR / "train.csv"
    class_weights = compute_class_weights_from_csv(train_csv)

    # Detailed class weight analysis (makes imbalance & plastic weight visible)
    print_class_weight_analysis(class_weights, train_csv)

    experiment_rows = []

    models_to_train = [
        ("Model A (Small CNN)",    build_model_a, "model_a", "training_curves_model_a.png"),
        ("Model B (Baseline CNN)", build_model_b, "model_b", "training_curves_model_b.png"),
    ]

    for model_label, build_fn, slug, curve_file in models_to_train:
        model = build_fn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history, best_val_acc, best_val_loss, elapsed, epochs_run, val_ds, val_labels = train_model(
            model, model_label, class_weights
        )

        # Save model
        model_path = MODELS_DIR / f"{slug}_best.keras"
        model.save(model_path)
        print(f"  Saved model -> {model_path}")

        # Save training history as JSON
        hist_path = MODELS_DIR / f"{slug}_history.json"
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
        print(f"  Saved history -> {hist_path}")

        # Training curves plot
        curve_path = PLOTS_DIR / curve_file
        plot_training_curves(history, model_label, curve_path)
        print(f"  Saved training curves -> {curve_path}")

        # Validation confusion matrix (catch plastic↔metal confusion early)
        plot_validation_confusion_matrix(model, model_label, slug, val_ds, val_labels)

        experiment_rows.append({
            "experiment_name": model_label,
            "cnn_architecture": model.name,
            "filter_config": "16-32-64" if "model_a" in slug else "32-64-128",
            "dense_units": 64 if "model_a" in slug else 128,
            "dropout_rate": 0.3 if "model_a" in slug else 0.4,
            "image_dimensions": f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs_config": MAX_EPOCHS,
            "epochs_completed": epochs_run,
            "best_val_accuracy": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "param_count": model.count_params(),
            "training_time_s": round(elapsed, 1),
            "test_accuracy": "TBD (Stage 7)",
        })

        print(f"\n{model_label} Results:")
        print(f"  Epochs completed:  {epochs_run} / {MAX_EPOCHS}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Best Val Loss:     {best_val_loss:.4f}")
        print(f"  Training Time:     {elapsed:.1f}s")

    # Save experiment comparison CSV
    exp_path = REPORTS_DIR / "experiment_comparison.csv"
    pd.DataFrame(experiment_rows).to_csv(exp_path, index=False)
    print(f"\nExperiment comparison saved -> {exp_path}")

    # Print summary comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 60)
    df_exp = pd.DataFrame(experiment_rows)
    print(df_exp[["experiment_name", "filter_config", "param_count",
                   "epochs_completed", "best_val_accuracy", "best_val_loss",
                   "training_time_s"]].to_string(index=False))

    # Recommend best model based solely on validation metrics
    best = max(experiment_rows, key=lambda r: r["best_val_accuracy"])
    print(f"\nRecommended model (highest val accuracy): {best['experiment_name']}")
    print("The test set has NOT been evaluated yet (Stage 7).")


if __name__ == "__main__":
    run_training()
