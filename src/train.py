"""
Training pipeline for Waste Classification CNN.

Builds tf.data.Dataset pipelines, trains the selected models with early
stopping, saves histories, training curves, validation confusion matrices and
an experiment comparison table. The test set is never touched here.

Usage:
    python -m src.train                                   # Model A + B, recipe v1
    python -m src.train --models model_c --recipe v2      # Model C, improved recipe
    python -m src.train --models model_b model_c_plain model_c --recipe v2
    python -m src.train --models model_c --recipe v2 --epochs 1   # quick check

Each run is saved under a run name: "<model>" for recipe v1 (the original
results) and "<model>_<recipe>" otherwise, e.g. models/model_c_v2_best.keras.
"""
import argparse
import json
import math
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
    DATASET_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    PLOTS_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    REPORTS_DIR,
    SPLITS_DIR,
    TRAINING_RECIPES,
)
from src.model import MODEL_REGISTRY, build_registered_model

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------------------
# Base dir for resolving relative paths stored in CSVs
# -------------------------------------------------------------------
BASE_DIR = DATASET_DIR.parent


def build_dataset(
    csv_path: Path,
    image_size: tuple,
    shuffle: bool = False,
    one_hot: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset from a split CSV.

    pipeline: read paths → load/resize → cache → (shuffle) → batch → prefetch

    Decoded, resized images are cached in memory after the first epoch
    (~140 MB for the training split at 160×160), which removes JPEG decoding
    from every later epoch — the main bottleneck when training on a GPU.
    Augmentation lives inside the model graph, so caching does not freeze it.
    Images are returned as uint8 so the model's Rescaling layer normalises them.

    one_hot=True returns one-hot labels (needed for label smoothing).
    """
    df = pd.read_csv(csv_path)
    abs_paths = (BASE_DIR / df["filepath"]).astype(str).tolist()
    labels = df["label"].tolist()
    height, width = image_size

    def _load_and_preprocess(filepath_tensor, label_tensor):
        raw = tf.io.read_file(filepath_tensor)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [height, width], method="bilinear")
        img = tf.cast(img, tf.uint8)
        if one_hot:
            label_tensor = tf.one_hot(label_tensor, NUM_CLASSES)
        return img, label_tensor

    ds = tf.data.Dataset.from_tensor_slices((abs_paths, labels))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights_from_csv(csv_path: Path, multipliers: dict) -> dict:
    """
    Computes class weights from the training CSV with custom multipliers.
    w_j = (n_samples / (n_classes * count_j)) * multiplier_j

    This ensures minority classes (trash) receive proportionally higher loss
    contributions, scaled by the recipe's per-class multipliers.
    """
    df = pd.read_csv(csv_path)
    n_classes = len(CLASSES)
    n_total = len(df)
    weights = {}
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        count = (df["class_name"] == cls_name).sum()
        base_w = float(n_total) / (n_classes * count)
        weights[cls_idx] = base_w * multipliers.get(cls_name, 1.0)
    return weights


def print_class_weight_analysis(class_weights: dict, csv_path: Path, multipliers: dict):
    """
    Prints a human-readable table of class weights alongside sample counts
    so imbalance and custom boosts are immediately visible in training logs.
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_classes = len(CLASSES)

    print("\n" + "─" * 68)
    print("CLASS WEIGHT ANALYSIS (Training Set)")
    print("─" * 68)
    print(f"  {'Class':<11} {'Count':>6} {'Share':>7} {'Base W':>8} {'Mult':>6} {'Final W':>9}")
    print("  " + "─" * 52)

    for cls_name in CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        count = (df["class_name"] == cls_name).sum()
        share = count / n_total * 100
        base_w = float(n_total) / (n_classes * count)
        mult = multipliers.get(cls_name, 1.0)
        print(f"  {cls_name:<11} {count:>6} {share:>6.1f}% {base_w:>8.2f} {mult:>6.2f} {class_weights[cls_idx]:>9.2f}")

    print("  " + "─" * 52)
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
    model, model_label: str, run_name: str, val_ds: tf.data.Dataset, val_labels: list
):
    """
    Generates a confusion matrix from validation predictions after training.
    This shows inter-class confusion (e.g. glass → plastic) without touching
    the test set.
    """
    print(f"\n  Generating validation confusion matrix for {model_label}...")
    probs = model.predict(val_ds, verbose=0)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.array(val_labels)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(CLASSES))))
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

    cm_path = PLOTS_DIR / f"val_confusion_matrix_{run_name}.png"
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


def compile_model(model, recipe: dict, steps_per_epoch: int, max_epochs: int):
    """
    Compiles a model according to a training recipe.

    v1: Adam with a fixed starting LR (reduced on plateau), sparse labels.
    v2: AdamW (weight decay, not applied to BatchNorm/bias) with cosine LR
        decay, one-hot labels and label smoothing.
    """
    if recipe["lr_schedule"] == "cosine":
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=recipe["learning_rate"],
            decay_steps=steps_per_epoch * max_epochs,
            alpha=0.01,
        )
    else:
        learning_rate = recipe["learning_rate"]

    if recipe["optimizer"] == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=recipe["weight_decay"]
        )
        optimizer.exclude_from_weight_decay(var_names=["bn", "bias"])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if recipe["label_smoothing"] > 0:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=recipe["label_smoothing"])
        metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


def train_model(model, model_label: str, image_size: tuple, recipe: dict,
                max_epochs: int, class_weights: dict):
    """
    Trains a model on the train/val splits.

    Uses:
      - EarlyStopping on val_loss: stops when it hasn't improved for
        `early_stopping_patience` epochs and restores the best weights.
      - ReduceLROnPlateau (v1 only): halves the LR when val_loss stalls.
        v2 uses a cosine schedule instead.
      - class_weight: per-class weights that boost minority classes.

    Returns (history_dict, best_val_acc, best_val_loss, elapsed_seconds,
             epochs_run, val_ds, val_labels).
    """
    one_hot = recipe["label_smoothing"] > 0
    train_ds = build_dataset(SPLITS_DIR / "train.csv", image_size, shuffle=True, one_hot=one_hot)
    val_ds   = build_dataset(SPLITS_DIR / "val.csv",   image_size, shuffle=False, one_hot=one_hot)

    # Keep val labels for confusion matrix generation
    val_labels = pd.read_csv(SPLITS_DIR / "val.csv")["label"].tolist()
    n_train = len(pd.read_csv(SPLITS_DIR / "train.csv"))
    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)

    compile_model(model, recipe, steps_per_epoch, max_epochs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=recipe["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if recipe["lr_schedule"] == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1,
            )
        )

    print(f"\n{'=' * 60}")
    print(f"Training: {model_label}")
    print(f"  Input size:   {image_size[0]}x{image_size[1]}")
    print(f"  Parameters:   {model.count_params():,}")
    print(f"  Max epochs:   {max_epochs}")
    print(f"  Optimizer:    {recipe['optimizer']} (lr={recipe['learning_rate']}, schedule={recipe['lr_schedule']})")
    print(f"  Early stop:   patience={recipe['early_stopping_patience']} on val_loss")
    print(f"  Label smooth: {recipe['label_smoothing']}")
    print(f"{'=' * 60}")
    t0 = time.time()
    history_obj = model.fit(
        train_ds,
        epochs=max_epochs,
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


def get_run_name(slug: str, recipe_name: str) -> str:
    """'model_b' for the original v1 results, otherwise e.g. 'model_c_v2'."""
    return slug if recipe_name == "v1" else f"{slug}_{recipe_name}"


def get_device_name() -> str:
    """Short description of the hardware used, for the experiment table."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    details = tf.config.experimental.get_device_details(gpus[0])
    return f"GPU ({details.get('device_name', 'unknown')})"


def update_experiment_table(new_rows: list):
    """
    Adds or replaces rows (matched by run_name) in experiment_comparison.csv,
    so training one model never erases the results of the others.
    """
    exp_path = REPORTS_DIR / "experiment_comparison.csv"
    new_df = pd.DataFrame(new_rows)
    if exp_path.exists():
        old_df = pd.read_csv(exp_path, dtype={"test_accuracy": str})
        old_df = old_df[~old_df["run_name"].isin(new_df["run_name"])]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(exp_path, index=False)
    print(f"\nExperiment comparison saved -> {exp_path}")
    return new_df


def run_training(model_slugs: list, recipe_name: str = "v1", epochs_override: int = None):
    """Main entry-point: trains the selected models, saves artefacts, prints comparison."""
    recipe = TRAINING_RECIPES[recipe_name]
    max_epochs = epochs_override or recipe["max_epochs"]
    device = get_device_name()
    print(f"Recipe: {recipe_name} | Device: {device} | Models: {', '.join(model_slugs)}")

    train_csv = SPLITS_DIR / "train.csv"
    multipliers = recipe["class_weight_multipliers"]
    class_weights = compute_class_weights_from_csv(train_csv, multipliers)
    print_class_weight_analysis(class_weights, train_csv, multipliers)

    experiment_rows = []

    for slug in model_slugs:
        spec = MODEL_REGISTRY[slug]
        run_name = get_run_name(slug, recipe_name)
        model_label = spec["label"] if recipe_name == "v1" else f"{spec['label']} [{recipe_name}]"
        image_size = spec["image_size"]

        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_SEED)
        model = build_registered_model(slug)

        history, best_val_acc, best_val_loss, elapsed, epochs_run, val_ds, val_labels = train_model(
            model, model_label, image_size, recipe, max_epochs, class_weights
        )

        # Save model
        model_path = MODELS_DIR / f"{run_name}_best.keras"
        model.save(model_path)
        print(f"  Saved model -> {model_path}")

        # Save training history as JSON
        hist_path = MODELS_DIR / f"{run_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
        print(f"  Saved history -> {hist_path}")

        # Training curves plot
        curve_path = PLOTS_DIR / f"training_curves_{run_name}.png"
        plot_training_curves(history, model_label, curve_path)
        print(f"  Saved training curves -> {curve_path}")

        # Validation confusion matrix
        plot_validation_confusion_matrix(model, model_label, run_name, val_ds, val_labels)

        experiment_rows.append({
            "run_name": run_name,
            "experiment_name": model_label,
            "recipe": recipe_name,
            "cnn_architecture": model.name,
            "filter_config": spec["filter_config"],
            "dense_units": spec["dense_units"],
            "dropout_rate": spec["dropout_rate"],
            "image_dimensions": f"{image_size[0]}x{image_size[1]}",
            "batch_size": BATCH_SIZE,
            "learning_rate": recipe["learning_rate"],
            "max_epochs_config": max_epochs,
            "epochs_completed": epochs_run,
            "best_val_accuracy": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "param_count": model.count_params(),
            "training_time_s": round(elapsed, 1),
            "device": device,
            "test_accuracy": "TBD (Stage 7)",
        })

        print(f"\n{model_label} Results:")
        print(f"  Epochs completed:  {epochs_run} / {max_epochs}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Best Val Loss:     {best_val_loss:.4f}")
        print(f"  Training Time:     {elapsed:.1f}s")

    df_exp = update_experiment_table(experiment_rows)

    # Print summary comparison across all runs recorded so far
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON SUMMARY (all runs)")
    print("=" * 60)
    print(df_exp[["run_name", "image_dimensions", "param_count", "epochs_completed",
                  "best_val_accuracy", "best_val_loss", "training_time_s", "device"]].to_string(index=False))

    best = df_exp.loc[df_exp["best_val_accuracy"].idxmax()]
    print(f"\nHighest val accuracy so far: {best['run_name']} ({best['best_val_accuracy']:.4f})")
    print("The test set has NOT been evaluated (run src.evaluate only for the final chosen model).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train waste classification CNNs.")
    parser.add_argument(
        "--models", nargs="+", default=["model_a", "model_b"],
        choices=list(MODEL_REGISTRY.keys()), help="Models to train (default: model_a model_b).",
    )
    parser.add_argument(
        "--recipe", default="v1", choices=list(TRAINING_RECIPES.keys()),
        help="Training recipe from src/config.py (default: v1).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the recipe's max epochs (e.g. 1 for a quick check).",
    )
    args = parser.parse_args()
    run_training(args.models, args.recipe, args.epochs)
