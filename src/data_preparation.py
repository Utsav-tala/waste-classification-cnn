"""
Data preparation, validation, duplicate detection, and stratified splitting.
Preserves the original dataset completely (read-only access).
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    DATASET_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 hash of a file using chunked read-only access."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_validate_dataset(
    dataset_dir: Path, classes: List[str]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Scans the dataset directory in read-only mode.
    Validates file readability, checks formats/dimensions, and computes SHA-256 hashes.
    """
    records = []
    errors = []

    for class_name in classes:
        class_folder = dataset_dir / class_name
        if not class_folder.exists():
            errors.append({
                "filepath": str(class_folder),
                "class_name": class_name,
                "error": f"Folder for class '{class_name}' does not exist",
            })
            continue

        for entry in sorted(os.scandir(class_folder), key=lambda e: e.name):
            if entry.name.startswith(".") or not entry.is_file():
                continue

            filepath = Path(entry.path)
            try:
                # Read-only verification using Pillow
                with Image.open(filepath) as img:
                    img.verify()
                
                # Re-open to read properties (verify() closes/invalidates the image object)
                with Image.open(filepath) as img:
                    width, height = img.size
                    img_format = img.format
                    img_mode = img.mode

                sha256 = compute_file_sha256(filepath)
                # Store path relative to workspace root for portability
                rel_path = filepath.relative_to(dataset_dir.parent)

                records.append({
                    "filepath": str(rel_path),
                    "filename": entry.name,
                    "class_name": class_name,
                    "label": CLASS_TO_IDX[class_name],
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "mode": img_mode,
                    "sha256": sha256,
                    "is_valid": True,
                })
            except Exception as e:
                errors.append({
                    "filepath": str(filepath),
                    "class_name": class_name,
                    "error": str(e),
                })

    df = pd.DataFrame(records)
    return df, errors


def detect_duplicates(df: pd.DataFrame) -> Dict:
    """
    Analyzes SHA-256 hashes to detect duplicate files.
    Differentiates between duplicates within the same class and across classes.
    """
    hash_counts = df["sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1].index.tolist()

    same_class_dupes = []
    cross_class_dupes = []

    for h in duplicate_hashes:
        group = df[df["sha256"] == h]
        classes_involved = group["class_name"].unique().tolist()
        files = group[["filepath", "class_name"]].to_dict(orient="records")

        dupe_info = {
            "sha256": h,
            "count": len(group),
            "classes": classes_involved,
            "files": files,
        }

        if len(classes_involved) == 1:
            same_class_dupes.append(dupe_info)
        else:
            cross_class_dupes.append(dupe_info)

    return {
        "total_unique_hashes": len(hash_counts),
        "total_duplicate_hashes": len(duplicate_hashes),
        "same_class_duplicates": same_class_dupes,
        "cross_class_duplicates": cross_class_dupes,
    }


def create_stratified_splits(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 70% train / 15% val / 15% test splitting.
    Preserves exact class proportions without copying image files.
    """
    temp_ratio = val_ratio + test_ratio  # 0.30
    val_in_temp = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50

    # First split: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: Val (15%) vs Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_in_temp),
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.sort_values(by="filepath").reset_index(drop=True)
    val_df = val_df.sort_values(by="filepath").reset_index(drop=True)
    test_df = test_df.sort_values(by="filepath").reset_index(drop=True)

    # Verification checks
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    assert len(train_paths & val_paths) == 0, "Train and Val paths overlap!"
    assert len(train_paths & test_paths) == 0, "Train and Test paths overlap!"
    assert len(val_paths & test_paths) == 0, "Val and Test paths overlap!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row counts mismatch!"

    # Check all classes present in all splits
    for s_name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        classes_in_split = set(s_df["class_name"].unique())
        missing_classes = set(CLASSES) - classes_in_split
        assert len(missing_classes) == 0, f"{s_name} split missing classes: {missing_classes}"

    return train_df, val_df, test_df


def check_split_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict:
    """Checks whether exact duplicate image hashes leak across split boundaries."""
    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    return {
        "train_val_hash_overlap_count": len(train_val_overlap),
        "train_test_hash_overlap_count": len(train_test_overlap),
        "val_test_hash_overlap_count": len(val_test_overlap),
        "has_leakage": len(train_val_overlap | train_test_overlap | val_test_overlap) > 0,
    }


def plot_class_distribution(df: pd.DataFrame, output_path: Path):
    """Generates and saves a publication-quality class distribution bar plot."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    class_counts = df["class_name"].value_counts().reindex(CLASSES)
    total_images = len(df)
    percentages = (class_counts / total_images) * 100

    bars = ax.bar(
        class_counts.index,
        class_counts.values,
        color=["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#e6ab02"],
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )

    for bar, count, pct in zip(bars, class_counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("TrashNet Waste Classification Dataset: Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Waste Category", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylim(0, max(class_counts.values) * 1.18)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_data_preparation():
    """Main execution function for Stage 4."""
    print("=" * 70)
    print("STAGE 4: DATASET EXPLORATION & STRATIFIED SPLITTING")
    print("=" * 70)

    print(f"\n1. Scanning dataset in read-only mode from: {DATASET_DIR}")
    df, corrupt_files = scan_and_validate_dataset(DATASET_DIR, CLASSES)
    print(f"   Total files scanned: {len(df) + len(corrupt_files)}")
    print(f"   Valid readable images: {len(df)}")
    print(f"   Corrupt / unreadable images: {len(corrupt_files)}")

    # Dimensions analysis
    dims = df.groupby(["width", "height", "format", "mode"]).size().reset_index(name="count")
    print("\n2. Image Dimensions & Formats:")
    for _, row in dims.iterrows():
        print(f"   {row['width']}x{row['height']}, Format: {row['format']}, Mode: {row['mode']} -> {row['count']} images")

    # Duplicate detection
    print("\n3. Scanning for exact duplicate files via SHA-256...")
    dupe_report = detect_duplicates(df)
    print(f"   Unique SHA-256 hashes: {dupe_report['total_unique_hashes']}")
    print(f"   Hashes with duplicates: {dupe_report['total_duplicate_hashes']}")
    print(f"   Same-class duplicate groups: {len(dupe_report['same_class_duplicates'])}")
    print(f"   Cross-class duplicate groups: {len(dupe_report['cross_class_duplicates'])}")

    if dupe_report["same_class_duplicates"]:
        print("   Detailed Same-Class Duplicates:")
        for d in dupe_report["same_class_duplicates"]:
            print(f"     Hash {d['sha256'][:12]}... (count: {d['count']}, class: {d['classes'][0]})")
            for f in d["files"]:
                print(f"       - {f['filepath']}")

    if dupe_report["cross_class_duplicates"]:
        print("   Detailed Cross-Class Duplicates:")
        for d in dupe_report["cross_class_duplicates"]:
            print(f"     Hash {d['sha256'][:12]}... (count: {d['count']}, classes: {d['classes']})")
            for f in d["files"]:
                print(f"       - {f['filepath']} ({f['class_name']})")

    # Stratified Splitting
    print("\n4. Generating Stratified Splits (70% Train / 15% Val / 15% Test)...")
    train_df, val_df, test_df = create_stratified_splits(df, seed=RANDOM_SEED)

    # Check for split leakage
    leakage = check_split_leakage(train_df, val_df, test_df)
    print(f"   Split hash leakage detected: {leakage['has_leakage']}")
    if leakage["has_leakage"]:
        print(f"     Train-Val hash overlap: {leakage['train_val_hash_overlap_count']}")
        print(f"     Train-Test hash overlap: {leakage['train_test_hash_overlap_count']}")
        print(f"     Val-Test hash overlap: {leakage['val_test_hash_overlap_count']}")

    # Save Splits to CSV
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    # Select clean columns for split CSVs
    split_cols = ["filepath", "class_name", "label", "width", "height", "sha256"]
    train_df[split_cols].to_csv(train_csv, index=False)
    val_df[split_cols].to_csv(val_csv, index=False)
    test_df[split_cols].to_csv(test_csv, index=False)

    print(f"   Saved {len(train_df)} rows -> {train_csv}")
    print(f"   Saved {len(val_df)} rows -> {val_csv}")
    print(f"   Saved {len(test_df)} rows -> {test_csv}")

    # Plot Class Distribution
    dist_plot_path = PLOTS_DIR / "class_distribution.png"
    plot_class_distribution(df, dist_plot_path)
    print(f"\n5. Class distribution plot saved -> {dist_plot_path}")

    # Compute Class Weights (calculated from training set only to prevent data leakage)
    train_counts = train_df["label"].value_counts().sort_index()
    n_train = len(train_df)
    n_classes = len(CLASSES)
    # Balanced heuristic: w_j = n_samples / (n_classes * n_samples_j)
    class_weights = {int(idx): float(n_train / (n_classes * count)) for idx, count in train_counts.items()}

    # Save Validation Report
    report_path = REPORTS_DIR / "dataset_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("WASTE CLASSIFICATION DATASET VALIDATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dataset Location: {DATASET_DIR.name}/\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Total Scanned Files: {len(df) + len(corrupt_files)}\n")
        f.write(f"Total Valid Images: {len(df)}\n")
        f.write(f"Total Corrupt / Unreadable Files: {len(corrupt_files)}\n\n")

        f.write("CLASS DISTRIBUTION (TOTAL DATASET):\n")
        f.write("-" * 45 + "\n")
        for cls_name in CLASSES:
            c = len(df[df["class_name"] == cls_name])
            pct = (c / len(df)) * 100
            f.write(f"  {cls_name:12s}: {c:4d} images ({pct:5.2f}%)\n")
        f.write("-" * 45 + "\n\n")

        f.write("STRATIFIED SPLIT BREAKDOWN:\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Training Split (70%):   {len(train_df):4d} images\n")
        f.write(f"  Validation Split (15%): {len(val_df):4d} images\n")
        f.write(f"  Testing Split (15%):    {len(test_df):4d} images\n")
        f.write(f"  Total In Splits:        {len(train_df) + len(val_df) + len(test_df):4d} images\n\n")

        f.write("PER-CLASS COUNTS BY SPLIT:\n")
        f.write(f"{'Class':<12} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Total':<8}\n")
        f.write("-" * 65 + "\n")
        for cls_name in CLASSES:
            tr_c = len(train_df[train_df["class_name"] == cls_name])
            va_c = len(val_df[val_df["class_name"] == cls_name])
            te_c = len(test_df[test_df["class_name"] == cls_name])
            tot = tr_c + va_c + te_c
            f.write(f"{cls_name:<12} | {tr_c:<12} | {va_c:<12} | {te_c:<12} | {tot:<8}\n")
        f.write("-" * 65 + "\n\n")

        f.write("CLASS WEIGHTS (Derived strictly from Training set):\n")
        f.write("-" * 45 + "\n")
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            f.write(f"  Class {cls_idx} ({cls_name:10s}): {class_weights[cls_idx]:.4f}\n")
        f.write("\n")

        f.write("DUPLICATE DETECTION RESULTS (SHA-256):\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Unique Hashes: {dupe_report['total_unique_hashes']}\n")
        f.write(f"  Duplicate Hashes: {dupe_report['total_duplicate_hashes']}\n")
        f.write(f"  Same-Class Duplicate Groups: {len(dupe_report['same_class_duplicates'])}\n")
        f.write(f"  Cross-Class Duplicate Groups: {len(dupe_report['cross_class_duplicates'])}\n")
        f.write(f"  Split Hash Leakage Detected: {leakage['has_leakage']}\n")

    print(f"6. Dataset validation report saved -> {report_path}")
    print("\nSTAGE 4 EXECUTION COMPLETE.")


if __name__ == "__main__":
    run_data_preparation()
