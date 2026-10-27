"""
Reusable inference module for Waste Classification.
Loads the saved model and class labels, preprocesses an image consistently
with the training pipeline, and returns class probabilities + predicted label.
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MODELS_DIR


def load_model_and_labels(
    model_slug: str = "model_b",
    models_dir: Path = MODELS_DIR,
) -> Tuple[tf.keras.Model, list]:
    """
    Loads the saved Keras model and corresponding class-label list.

    Returns
    -------
    model  : tf.keras.Model  — fully loaded, ready for inference
    classes: list[str]       — ordered list of class names matching model outputs
    """
    model_path  = models_dir / f"{model_slug}_best.keras"
    labels_path = models_dir / "class_labels.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Run src/train.py first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"class_labels.json not found at: {labels_path}\n"
            "Run src/evaluate.py first."
        )

    model = tf.keras.models.load_model(model_path)

    with open(labels_path) as f:
        labels_data = json.load(f)
    classes = labels_data["classes"]

    return model, classes


def preprocess_image(
    image_source: Union[str, Path, Image.Image],
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> np.ndarray:
    """
    Converts any image source to a uint8 array of shape (1, H, W, 3).
    Handles: file paths (str/Path) and PIL Image objects.
    Converts to RGB and resizes — identical to the training pipeline.
    """
    if isinstance(image_source, (str, Path)):
        try:
            img = Image.open(image_source)
        except Exception as e:
            raise ValueError(f"Could not open image file '{image_source}': {e}")
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        raise TypeError(
            f"image_source must be a file path or PIL Image, got {type(image_source)}"
        )

    img = img.convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)          # (H, W, 3)
    arr = np.expand_dims(arr, axis=0)             # (1, H, W, 3)
    return arr


def predict(
    image_source: Union[str, Path, Image.Image],
    model: tf.keras.Model,
    classes: list,
) -> Dict:
    """
    Runs inference on a single image.

    Returns a dictionary with:
      predicted_class  : str   — the class with the highest softmax score
      confidence       : float — the highest softmax probability (NOT a calibrated
                                  probability; treat as a ranking score)
      probabilities    : dict  — {class_name: softmax_score} for all 6 classes

    Note: Softmax outputs reflect the model's learned discrimination between
    classes.  They are not calibrated posterior probabilities and should not
    be interpreted as guarantees of correctness.
    """
    arr = preprocess_image(image_source)          # (1, H, W, 3) uint8
    probs = model(arr, training=False).numpy()[0]  # shape: (num_classes,)

    predicted_idx   = int(np.argmax(probs))
    predicted_class = classes[predicted_idx]
    confidence      = float(probs[predicted_idx])
    prob_dict       = {cls: float(p) for cls, p in zip(classes, probs)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
    }
