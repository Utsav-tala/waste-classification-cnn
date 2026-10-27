"""
CNN Model Architectures for Waste Classification.

Model A: Small CNN  — 3 Conv blocks (16, 32, 64 filters)
Model B: Baseline CNN — 3 Conv blocks (32, 64, 128 filters)

Both models embed rescaling (0-1) as a Keras layer, so raw uint8
tensors can be fed at inference time without external preprocessing.
Data augmentation layers are included and only apply during training=True calls.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CHANNELS,
    INPUT_SHAPE,
    NUM_CLASSES,
    AUG_ROTATION_FACTOR,
    AUG_ZOOM_FACTOR,
    AUG_BRIGHTNESS_FACTOR,
    AUG_CONTRAST_FACTOR,
)


def _augmentation_block(name_prefix: str = "") -> keras.Sequential:
    """
    Robust data augmentation applied ONLY during training (training=True).

    Augmentations:
      - RandomFlip (horizontal): Waste items have no canonical orientation.
      - RandomRotation (±27°):   Stronger than before (was ±8°); waste on
        conveyor belts or in bins can appear at arbitrary angles.
      - RandomZoom (±15%):       Simulates distance variation in photos.
      - RandomBrightness (±20%): Lighting varies between indoor recycling
        centres and outdoor bins; forces the model to learn material texture
        rather than relying on absolute brightness.
      - RandomContrast (±20%):   Prevents the model from depending on a
        specific contrast profile, which is a major source of plastic ↔
        metal ↔ paper confusion since all three can appear shiny or matte
        depending on lighting.

    Validation/test inputs are passed through unmodified.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name=f"{name_prefix}aug_flip"),
            layers.RandomRotation(
                AUG_ROTATION_FACTOR, name=f"{name_prefix}aug_rotate"
            ),
            layers.RandomZoom(
                AUG_ZOOM_FACTOR, name=f"{name_prefix}aug_zoom"
            ),
            layers.RandomBrightness(
                AUG_BRIGHTNESS_FACTOR,
                value_range=(0.0, 1.0),
                name=f"{name_prefix}aug_brightness",
            ),
            layers.RandomContrast(
                AUG_CONTRAST_FACTOR,
                value_range=(0.0, 1.0),
                name=f"{name_prefix}aug_contrast",
            ),
        ],
        name=f"{name_prefix}augmentation",
    )



def build_model_a(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    name: str = "model_a_small_cnn",
) -> keras.Model:
    """
    Model A — Small CNN.

    Architecture:
      Rescaling → Augmentation (train only)
      → Conv2D(16) + BN + ReLU + MaxPool
      → Conv2D(32) + BN + ReLU + MaxPool
      → Conv2D(64) + BN + ReLU + MaxPool
      → GlobalAveragePooling2D
      → Dense(64, ReLU) → Dropout(0.3)
      → Dense(6, Softmax)

    Layer explanations:
      Conv2D      : Learned 3×3 filter kernels slide over the feature map to
                    detect spatial patterns (edges, textures, shapes).
      BatchNorm   : Normalises activations to stabilise and speed up training.
      ReLU        : Non-linear activation; sets negatives to zero, keeping positives.
      MaxPooling  : Down-samples spatial dimensions by taking the maximum in each 2×2
                    window, making features position-invariant.
      GlobalAvgPool: Collapses each feature map to a scalar, reducing parameters
                     and acting as implicit regularisation.
      Dense       : Fully-connected classification head.
      Dropout     : Randomly zeros activations during training to prevent overfitting.
      Softmax     : Converts logits to class probabilities summing to 1.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # --- Preprocessing (always active) ---
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # --- Augmentation (active during training only) ---
    x = _augmentation_block(name_prefix="a_")(x)

    # --- Conv Block 1: 16 filters ---
    x = layers.Conv2D(16, 3, padding="same", use_bias=False, name="a_conv1")(x)
    x = layers.BatchNormalization(name="a_bn1")(x)
    x = layers.Activation("relu", name="a_relu1")(x)
    x = layers.MaxPooling2D(2, name="a_pool1")(x)

    # --- Conv Block 2: 32 filters ---
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="a_conv2")(x)
    x = layers.BatchNormalization(name="a_bn2")(x)
    x = layers.Activation("relu", name="a_relu2")(x)
    x = layers.MaxPooling2D(2, name="a_pool2")(x)

    # --- Conv Block 3: 64 filters ---
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="a_conv3")(x)
    x = layers.BatchNormalization(name="a_bn3")(x)
    x = layers.Activation("relu", name="a_relu3")(x)
    x = layers.MaxPooling2D(2, name="a_pool3")(x)

    # --- Classification head ---
    x = layers.GlobalAveragePooling2D(name="a_gap")(x)
    x = layers.Dense(64, activation="relu", name="a_dense1")(x)
    x = layers.Dropout(dropout_rate, name="a_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="a_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_model_b(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    name: str = "model_b_baseline_cnn",
) -> keras.Model:
    """
    Model B — Baseline CNN.

    Architecture:
      Rescaling → Augmentation (train only)
      → Conv2D(32) + BN + ReLU + MaxPool
      → Conv2D(64) + BN + ReLU + MaxPool
      → Conv2D(128) + BN + ReLU + MaxPool
      → GlobalAveragePooling2D
      → Dense(128, ReLU) → Dropout(0.4)
      → Dense(6, Softmax)

    Larger filter counts give the model more representational capacity,
    at the cost of additional parameters and longer training time.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # --- Preprocessing ---
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # --- Augmentation ---
    x = _augmentation_block(name_prefix="b_")(x)

    # --- Conv Block 1: 32 filters ---
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="b_conv1")(x)
    x = layers.BatchNormalization(name="b_bn1")(x)
    x = layers.Activation("relu", name="b_relu1")(x)
    x = layers.MaxPooling2D(2, name="b_pool1")(x)

    # --- Conv Block 2: 64 filters ---
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="b_conv2")(x)
    x = layers.BatchNormalization(name="b_bn2")(x)
    x = layers.Activation("relu", name="b_relu2")(x)
    x = layers.MaxPooling2D(2, name="b_pool2")(x)

    # --- Conv Block 3: 128 filters ---
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="b_conv3")(x)
    x = layers.BatchNormalization(name="b_bn3")(x)
    x = layers.Activation("relu", name="b_relu3")(x)
    x = layers.MaxPooling2D(2, name="b_pool3")(x)

    # --- Classification head ---
    x = layers.GlobalAveragePooling2D(name="b_gap")(x)
    x = layers.Dense(128, activation="relu", name="b_dense1")(x)
    x = layers.Dropout(dropout_rate, name="b_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="b_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def smoke_test():
    """
    Forward-pass smoke test for both models.
    Creates a random batch of 4 images (128×128×3, dtype=uint8) and
    checks that output shapes and probability sums are correct.
    """
    import numpy as np

    print("=" * 55)
    print("SMOKE TEST: Verifying Model Architectures")
    print("=" * 55)

    dummy = np.random.randint(0, 256, (4, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), dtype=np.uint8)

    for build_fn, label in [(build_model_a, "Model A (Small CNN)"),
                             (build_model_b, "Model B (Baseline CNN)")]:
        model = build_fn()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        out = model(dummy, training=False)
        params = model.count_params()
        prob_sums = tf.reduce_sum(out, axis=1).numpy()

        print(f"\n{label}")
        print(f"  Input  shape : {dummy.shape}")
        print(f"  Output shape : {out.shape}  (expected: (4, {NUM_CLASSES}))")
        print(f"  Param count  : {params:,}")
        print(f"  Softmax sums : {prob_sums}  (all must ≈ 1.0)")

        assert out.shape == (4, NUM_CLASSES), f"Output shape mismatch: {out.shape}"
        assert all(abs(s - 1.0) < 1e-5 for s in prob_sums), "Softmax probabilities do not sum to 1"
        print("  Status: PASS ✓")

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    smoke_test()
