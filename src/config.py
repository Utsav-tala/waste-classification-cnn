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
