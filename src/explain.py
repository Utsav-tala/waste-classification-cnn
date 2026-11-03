"""
Explainability helpers for the Streamlit app.

  - Grad-CAM heatmaps: which image regions drove a model's prediction
  - Feature maps: what intermediate layers respond to
  - Learned first-layer kernels and hand-made convolution demos
  - Layer-by-layer architecture tables

All functions take raw uint8 images, the same format the models were trained on.
"""
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# -------------------------------------------------------------------
# Models shown in the app
# -------------------------------------------------------------------
# gradcam_layer: last convolutional activation before global pooling
# feature_layers: (layer name, caption) pairs, from shallow to deep
APP_MODELS = {
    "model_a": {
        "file": "model_a_best.keras",
        "title": "Model A",
        "subtitle": "Small CNN",
        "gradcam_layer": "a_relu3",
        "feature_layers": [
            ("a_relu1", "Conv block 1"),
            ("a_pool1", "after MaxPool 1"),
            ("a_relu2", "Conv block 2"),
            ("a_relu3", "Conv block 3"),
        ],
    },
    "model_b": {
        "file": "model_b_best.keras",
        "title": "Model B",
        "subtitle": "Baseline CNN",
        "gradcam_layer": "b_relu3",
        "feature_layers": [
            ("b_relu1", "Conv block 1"),
            ("b_pool1", "after MaxPool 1"),
            ("b_relu2", "Conv block 2"),
            ("b_relu3", "Conv block 3"),
        ],
    },
    "model_c_v3": {
        "file": "model_c_v3_best.keras",
        "title": "Model C",
        "subtitle": "Residual CNN (final)",
        "gradcam_layer": "c_stage4_out",
        "feature_layers": [
            ("c_stem_relu", "Stem conv"),
            ("c_stem_pool", "after MaxPool"),
            ("c_stage1_out", "Residual stage 1"),
            ("c_stage2_out", "Residual stage 2"),
            ("c_stage3_out", "Residual stage 3"),
            ("c_stage4_out", "Residual stage 4"),
        ],
    },
}

# Classic hand-designed 3×3 kernels, to show what a convolution does
DEMO_KERNELS = {
    "Vertical edges": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Horizontal edges": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "Blur": np.full((3, 3), 1 / 9, dtype=np.float32),
}


def model_input_size(model) -> tuple:
    """(height, width) the model expects."""
    return tuple(model.input_shape[1:3])


def to_model_input(pil_image: Image.Image, size: tuple) -> np.ndarray:
    """RGB, resized to (height, width), as a uint8 batch of one: (1, H, W, 3)."""
    img = pil_image.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)[np.newaxis]


def predict_probs(model, arr: np.ndarray) -> np.ndarray:
    """Softmax probabilities for a single image."""
    return model(arr, training=False).numpy()[0]


def grad_cam(model, arr: np.ndarray, layer_name: str, class_idx: Optional[int] = None) -> np.ndarray:
    """
    Grad-CAM (Selvaraju et al., 2017).

    1. Run the image through the model, keeping the activations A of the last
       conv layer (k feature maps).
    2. Take the gradient of the class score with respect to A. Averaging it
       over each feature map gives one importance weight per map.
    3. Heatmap = ReLU(sum_k weight_k * A_k), scaled to [0, 1].

    The log of the softmax probability is used as the class score, because the
    raw probability saturates near 1 and gives vanishing gradients.

    Returns a 2D array at the conv layer's resolution (e.g. 10×10 for Model C).
    """
    grad_model = tf.keras.Model(model.input, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(arr, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = tf.math.log(preds[:, class_idx] + 1e-9)

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, size: tuple = (320, 320),
                    alpha: float = 0.45) -> Image.Image:
    """Blends a [0, 1] heatmap (jet colours) over the image."""
    base = pil_image.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    heat = Image.fromarray(np.uint8(heatmap * 255)).resize((size[1], size[0]), Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored, alpha)


def heatmap_image(heatmap: np.ndarray, size: tuple = (160, 160)) -> Image.Image:
    """The raw heatmap grid, enlarged without smoothing so each cell stays visible."""
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3]
    img = Image.fromarray(np.uint8(colored * 255))
    return img.resize((size[1], size[0]), Image.NEAREST)


def feature_maps(model, arr: np.ndarray, layer_names: List[str], n_channels: int = 6) -> Dict[str, dict]:
    """
    Activations of the given layers for one image.

    For each layer, returns its shape and the n most active channels, each
    scaled to [0, 1] for display.
    """
    extractor = tf.keras.Model(model.input, [model.get_layer(n).output for n in layer_names])
    outputs = extractor(arr, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    result = {}
    for name, out in zip(layer_names, outputs):
        fmap = out.numpy()[0]                                   # (H, W, C)
        top = np.argsort(fmap.mean(axis=(0, 1)))[::-1][:n_channels]
        channels = []
        for c in top:
            ch = fmap[..., c]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-9)
            channels.append(ch)
        result[name] = {"shape": fmap.shape, "channels": channels}
    return result


def channel_image(channel: np.ndarray, size: int = 120, cmap: str = "viridis") -> Image.Image:
    """A single [0, 1] feature map as an enlarged, coloured image (pixels kept sharp)."""
    colored = matplotlib.colormaps[cmap](channel)[..., :3]
    return Image.fromarray(np.uint8(colored * 255)).resize((size, size), Image.NEAREST)


def first_layer_kernels(model, conv_layer: str) -> List[Image.Image]:
    """The learned 3×3×3 kernels of the first conv layer, shown as tiny RGB images."""
    w = model.get_layer(conv_layer).get_weights()[0]           # (3, 3, 3, n_filters)
    images = []
    for i in range(w.shape[-1]):
        k = w[..., i]
        k = (k - k.min()) / (k.max() - k.min() + 1e-9)
        images.append(Image.fromarray(np.uint8(k * 255)).resize((48, 48), Image.NEAREST))
    return images


def apply_kernel(pil_image: Image.Image, kernel: np.ndarray, size: int = 256) -> Image.Image:
    """Convolves a greyscale version of the image with a 3×3 kernel."""
    gray = np.asarray(pil_image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = tf.constant(gray[np.newaxis, ..., np.newaxis])
    k = tf.constant(kernel[..., np.newaxis, np.newaxis])
    out = tf.nn.conv2d(x, k, strides=1, padding="SAME").numpy()[0, ..., 0]
    if kernel.sum() == 0:
        # Edge kernels: show edge strength, boosted so faint edges stay visible
        out = np.abs(out)
        out = np.clip(out / (np.percentile(out, 99.5) + 1e-9), 0, 1) ** 0.7
    else:
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return Image.fromarray(np.uint8(out * 255))


def layer_table(model) -> pd.DataFrame:
    """One row per layer: name, type, output shape and parameter count."""
    rows = []
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "InputLayer":
            continue
        shape = tuple(layer.output.shape)[1:]
        rows.append({
            "Layer": layer.name,
            "Type": "Augmentation (train only)" if kind == "Sequential" else kind,
            "Output shape": " × ".join(str(d) for d in shape),
            "Parameters": layer.count_params(),
        })
    return pd.DataFrame(rows)
