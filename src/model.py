"""
CNN Model Architectures for Waste Classification.

Model A: Small CNN  — 3 Conv blocks (16, 32, 64 filters)
Model B: Baseline CNN — 3 Conv blocks (32, 64, 128 filters)
Model C: Residual CNN — stem + 4 residual stages (32, 64, 128, 256 filters)
Model C-plain: Model C without skip connections (ablation baseline)

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
    bn_momentum: float = 0.99,
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
    x = layers.BatchNormalization(momentum=bn_momentum, name="a_bn1")(x)
    x = layers.Activation("relu", name="a_relu1")(x)
    x = layers.MaxPooling2D(2, name="a_pool1")(x)

    # --- Conv Block 2: 32 filters ---
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="a_conv2")(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="a_bn2")(x)
    x = layers.Activation("relu", name="a_relu2")(x)
    x = layers.MaxPooling2D(2, name="a_pool2")(x)

    # --- Conv Block 3: 64 filters ---
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="a_conv3")(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="a_bn3")(x)
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
    bn_momentum: float = 0.99,
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
    x = layers.BatchNormalization(momentum=bn_momentum, name="b_bn1")(x)
    x = layers.Activation("relu", name="b_relu1")(x)
    x = layers.MaxPooling2D(2, name="b_pool1")(x)

    # --- Conv Block 2: 64 filters ---
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="b_conv2")(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="b_bn2")(x)
    x = layers.Activation("relu", name="b_relu2")(x)
    x = layers.MaxPooling2D(2, name="b_pool2")(x)

    # --- Conv Block 3: 128 filters ---
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="b_conv3")(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="b_bn3")(x)
    x = layers.Activation("relu", name="b_relu3")(x)
    x = layers.MaxPooling2D(2, name="b_pool3")(x)

    # --- Classification head ---
    x = layers.GlobalAveragePooling2D(name="b_gap")(x)
    x = layers.Dense(128, activation="relu", name="b_dense1")(x)
    x = layers.Dropout(dropout_rate, name="b_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="b_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def _conv_bn(x, filters: int, stride: int, name: str, relu: bool = True, bn_momentum: float = 0.99):
    """Conv2D(3×3) → BatchNorm → (optional) ReLU."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv"
    )(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn")(x)
    if relu:
        x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def _residual_block(x, filters: int, stride: int, name: str, use_skip: bool = True,
                    bn_momentum: float = 0.99):
    """
    Basic residual block (ResNet-18 style):

        x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ shortcut) → ReLU

    The block learns only the *change* F(x) to apply to its input, and the
    shortcut carries x (and the gradient) straight through. When the block
    halves the spatial size or changes the channel count, a 1×1 strided
    "projection" conv reshapes the shortcut so the two tensors can be added.

    use_skip=False builds the same block with no shortcut (Model C-plain),
    so the effect of skip connections can be measured in isolation.
    """
    shortcut = x
    y = _conv_bn(x, filters, stride, name=f"{name}_a", bn_momentum=bn_momentum)
    y = _conv_bn(y, filters, 1, name=f"{name}_b", relu=False, bn_momentum=bn_momentum)

    if use_skip:
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=stride, use_bias=False, name=f"{name}_proj_conv"
            )(shortcut)
            shortcut = layers.BatchNormalization(momentum=bn_momentum, name=f"{name}_proj_bn")(shortcut)
        y = layers.Add(name=f"{name}_add")([y, shortcut])

    return layers.Activation("relu", name=f"{name}_out")(y)


def build_model_c(
    input_shape: tuple = (160, 160, CHANNELS),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    use_skip: bool = True,
    bn_momentum: float = 0.99,
    name: str = "model_c_residual_cnn",
) -> keras.Model:
    """
    Model C — Residual CNN (built from scratch).

    Architecture (160×160 input):
      Rescaling → Augmentation (train only)
      → Stem: Conv2D(32) + BN + ReLU + MaxPool          → 80×80×32
      → Stage 1: Residual block (32)                    → 80×80×32
      → Stage 2: Residual block (64),  stride 2         → 40×40×64
      → Stage 3: Residual block (128), stride 2         → 20×20×128
      → Stage 4: Residual block (256), stride 2         → 10×10×256
      → GlobalAveragePooling2D → Dropout → Dense(6, Softmax)

    Why it should beat Models A/B (which underfit at ~68%):
      - Depth: 9 conv layers instead of 3, so the network can build up from
        edges and textures to object parts and shapes.
      - Receptive field: each final feature sees ≈96×96 px of the 160×160
        input (≈60%), versus ≈22×22 px of 128×128 (≈17%) for Models A/B.
      - Skip connections keep gradients flowing through the deeper network.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # --- Preprocessing ---
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # --- Augmentation ---
    x = _augmentation_block(name_prefix="c_")(x)

    # --- Stem ---
    x = _conv_bn(x, 32, 1, name="c_stem", bn_momentum=bn_momentum)
    x = layers.MaxPooling2D(2, name="c_stem_pool")(x)

    # --- Residual stages ---
    for i, (filters, stride) in enumerate([(32, 1), (64, 2), (128, 2), (256, 2)], start=1):
        x = _residual_block(x, filters, stride, name=f"c_stage{i}", use_skip=use_skip,
                            bn_momentum=bn_momentum)

    # --- Classification head ---
    x = layers.GlobalAveragePooling2D(name="c_gap")(x)
    x = layers.Dropout(dropout_rate, name="c_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="c_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_model_c_plain(
    input_shape: tuple = (160, 160, CHANNELS),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    bn_momentum: float = 0.99,
    name: str = "model_c_plain_cnn",
) -> keras.Model:
    """Model C-plain — identical to Model C but WITHOUT skip connections (ablation)."""
    return build_model_c(input_shape, num_classes, dropout_rate, use_skip=False,
                         bn_momentum=bn_momentum, name=name)


# -------------------------------------------------------------------
# Model registry: one entry per architecture, used by train.py
# -------------------------------------------------------------------
MODEL_REGISTRY = {
    "model_a": {
        "label": "Model A (Small CNN)",
        "build_fn": build_model_a,
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "filter_config": "16-32-64",
        "dense_units": 64,
        "dropout_rate": 0.3,
    },
    "model_b": {
        "label": "Model B (Baseline CNN)",
        "build_fn": build_model_b,
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "filter_config": "32-64-128",
        "dense_units": 128,
        "dropout_rate": 0.4,
    },
    "model_c": {
        "label": "Model C (Residual CNN)",
        "build_fn": build_model_c,
        "image_size": (160, 160),
        "filter_config": "32-32-64-128-256 (residual)",
        "dense_units": 0,
        "dropout_rate": 0.3,
    },
    "model_c_plain": {
        "label": "Model C-plain (no skip connections)",
        "build_fn": build_model_c_plain,
        "image_size": (160, 160),
        "filter_config": "32-32-64-128-256 (plain)",
        "dense_units": 0,
        "dropout_rate": 0.3,
    },
}


def build_registered_model(slug: str, bn_momentum: float = 0.99) -> keras.Model:
    """
    Builds a model from MODEL_REGISTRY with its own input size.

    bn_momentum: BatchNorm moving-average momentum. Keras default is 0.99;
    0.9 makes the moving statistics track the weights faster, which steadies
    validation accuracy on a small dataset (only ~56 steps per epoch).
    """
    spec = MODEL_REGISTRY[slug]
    h, w = spec["image_size"]
    return spec["build_fn"](input_shape=(h, w, CHANNELS), bn_momentum=bn_momentum)


def smoke_test():
    """
    Forward-pass smoke test for every registered model.
    Creates a random batch of 4 uint8 images at each model's input size and
    checks that output shapes and probability sums are correct.
    """
    import numpy as np

    print("=" * 55)
    print("SMOKE TEST: Verifying Model Architectures")
    print("=" * 55)

    for slug, spec in MODEL_REGISTRY.items():
        label = spec["label"]
        h, w = spec["image_size"]
        dummy = np.random.randint(0, 256, (4, h, w, CHANNELS), dtype=np.uint8)
        model = build_registered_model(slug)
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
