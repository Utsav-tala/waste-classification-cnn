"""
Central configuration for Waste Classification System Using CNN.
Defines paths, classes, image specifications, splitting parameters, and hyperparameters.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset-resized"
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Ensure output directories exist
for directory in [DATA_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Classes definition (ordered alphabetically as discovered)
CLASSES = sorted(["cardboard", "glass", "metal", "paper", "plastic", "trash"])
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Image & Preprocessing Specifications
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

# Reproducibility & Splitting
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5

# Data Augmentation Hyperparameters
AUG_ROTATION_FACTOR = 0.15          # ±27° (was ±8°)
AUG_ZOOM_FACTOR = 0.15              # ±15% (was ±10%)
AUG_BRIGHTNESS_FACTOR = 0.20        # ±20% brightness jitter
AUG_CONTRAST_FACTOR = 0.20          # ±20% contrast jitter

# Class Weight Multipliers
# Optional multipliers applied on top of inverse-frequency weights.
# All 1.0 = plain inverse-frequency weighting, which is what the published
# Model A / Model B checkpoints were trained with. Tuned multipliers
# (e.g. boosting glass, which has low recall) are a planned experiment.
CLASS_WEIGHT_MULTIPLIERS = {
    "cardboard": 1.00,
    "glass": 1.00,
    "metal": 1.00,
    "paper": 1.00,
    "plastic": 1.00,
    "trash": 1.00,
}

# Training Recipes
# v1 = the original recipe used for the published Model A / Model B results.
# v2 = improved recipe for Model C and the ablation study:
#   - more epochs + cosine learning-rate decay (A/B were still improving at 50)
#   - AdamW weight decay and label smoothing to curb overfitting of a larger
#     model and soften the 3 mislabelled duplicate images
#   - softer trash weight (v1 over-predicted trash: 100% recall, 33% precision)
# v3 = v2 made stable. In v2, val accuracy swung 35% ↔ 76% between epochs and
#   early stopping kept a lucky spike at epoch 32 of 80, before the cosine LR
#   had decayed. v3 runs the full cosine schedule with no early stopping, keeps
#   the final weights, and uses BatchNorm momentum 0.9 so the moving statistics
#   keep up with the weights.
# early_stopping_patience=None disables early stopping.
TRAINING_RECIPES = {
    "v1": {
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "lr_schedule": "plateau",
        "label_smoothing": 0.0,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "bn_momentum": 0.99,
        "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
    },
    "v2": {
        "max_epochs": 80,
        "learning_rate": 1e-3,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "lr_schedule": "cosine",
        "label_smoothing": 0.1,
        "early_stopping_patience": 15,
        "bn_momentum": 0.99,
        "class_weight_multipliers": {**CLASS_WEIGHT_MULTIPLIERS, "trash": 0.5},
    },
    "v3": {
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "lr_schedule": "cosine",
        "label_smoothing": 0.1,
        "early_stopping_patience": None,
        "bn_momentum": 0.9,
        "class_weight_multipliers": {**CLASS_WEIGHT_MULTIPLIERS, "trash": 0.5},
    },
}
