"""
Streamlit web app for Waste Classification.
Run with: streamlit run app.py

Tabs:
  1. Classify & Compare — Models A, B and C side by side, with Grad-CAM heatmaps
  2. How a CNN Sees     — convolution, ReLU, pooling, feature maps, skip connections
  3. Architectures      — layer counts and layer-by-layer tables for each model
  4. Results            — test metrics, ablation study and training plots
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="♻️ Waste Classifier CNN",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "models"
PLOTS_DIR   = BASE_DIR / "results" / "plots"
REPORTS_DIR = BASE_DIR / "results" / "reports"
SAMPLES_DIR = BASE_DIR / "assets" / "samples"

CLASS_EMOJI = {
    "cardboard": "📦",
    "glass":     "🍶",
    "metal":     "🔩",
    "paper":     "📄",
    "plastic":   "🧴",
    "trash":     "🗑️",
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-banner {
        background: linear-gradient(135deg, #1a6b3a 0%, #0d4d76 60%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 30px 36px 24px 36px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .hero-banner h1 { color: #ffffff; font-size: 2.1rem; font-weight: 700; margin: 0 0 6px 0; }
    .hero-banner p  { color: #c8d6e5; font-size: 1.0rem; margin: 0; }

    .metric-card {
        background: #1e2430;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .metric-card .label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #38bdf8; font-size: 1.5rem; font-weight: 700; margin-top: 2px; }

    .final-card {
        background: linear-gradient(135deg, #1a3a2a 0%, #0d2640 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .final-card .k { color: #94a3b8; font-size: 0.85rem; }
    .final-card .cls { color: #22c55e; font-size: 2.3rem; font-weight: 700; text-transform: capitalize; }
    .final-card .conf { color: #fbbf24; font-size: 1.1rem; font-weight: 600; }

    .model-card {
        background: #1e2430;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .model-card { min-height: 172px; }
    .model-card.final { border: 2px solid #22c55e; }
    .tick  { color: #22c55e; font-weight: 700; }
    .cross { color: #f87171; font-weight: 700; }
    .model-card .name { color: #e2e8f0; font-weight: 700; font-size: 1.05rem; }
    .model-card .sub  { color: #94a3b8; font-size: 0.8rem; }
    .model-card .pred { font-size: 1.4rem; font-weight: 700; text-transform: capitalize; margin-top: 6px; }
    .model-card .conf { color: #94a3b8; font-size: 0.9rem; }

    .info-box {
        background: #111827;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 16px;
        color: #cbd5e1;
        font-size: 0.9rem;
        margin: 8px 0 14px 0;
    }
    .limitation-box {
        background: #1c1a0f;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fde68a;
        font-size: 0.86rem;
        margin-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loading — models and results are read once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models into memory…")
def load_models():
    """Returns ({slug: model}, classes) for every app model found on disk."""
    import json
    import tensorflow as tf
    from src.explain import APP_MODELS

    models = {}
    for slug, info in APP_MODELS.items():
        path = MODELS_DIR / info["file"]
        if path.exists():
            models[slug] = tf.keras.models.load_model(str(path))

    with open(MODELS_DIR / "class_labels.json") as f:
        classes = json.load(f)["classes"]
    return models, classes


@st.cache_data
def load_experiments() -> pd.DataFrame:
    """The experiment comparison table written by src.train / src.evaluate."""
    path = REPORTS_DIR / "experiment_comparison.csv"
    return pd.read_csv(path, dtype={"test_accuracy": str}) if path.exists() else pd.DataFrame()


def run_metrics(slug: str) -> dict:
    """Test accuracy, val accuracy and parameter count for a run, as display strings."""
    df = load_experiments()
    row = df[df["run_name"] == slug] if not df.empty else df
    if row.empty:
        return {"test": "—", "val": "—", "params": "—"}
    row = row.iloc[0]
    try:
        test = f"{float(row['test_accuracy']) * 100:.1f}%"
    except ValueError:
        test = "—"
    return {
        "test": test,
        "val": f"{float(row['saved_val_accuracy']) * 100:.1f}%",
        "params": f"{int(row['param_count']):,}",
    }


@st.cache_data(show_spinner="Running the models…")
def analyse(image_bytes: bytes) -> dict:
    """
    Runs every model on the image and collects everything the tabs display:
    probabilities, Grad-CAM heatmaps and feature maps. Cached per image.
    """
    from src import explain as ex

    models, classes = load_models()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = {}
    for slug, model in models.items():
        info = ex.APP_MODELS[slug]
        arr = ex.to_model_input(img, ex.model_input_size(model))
        probs = ex.predict_probs(model, arr)
        idx = int(np.argmax(probs))
        cam = ex.grad_cam(model, arr, info["gradcam_layer"], idx)
        fmaps = ex.feature_maps(model, arr, [name for name, _ in info["feature_layers"]])
        results[slug] = {
            "probs": {c: float(p) for c, p in zip(classes, probs)},
            "pred": classes[idx],
            "conf": float(probs[idx]),
            "cam": cam,
            "overlay": ex.overlay_heatmap(img, cam),
            "fmaps": fmaps,
        }
    return results


def verdict(pred: str, true_label, prefix: str = "") -> str:
    """✓ / ✗ marker when the true label is known (sample images only)."""
    if not true_label:
        return ""
    mark = '<span class="tick">✓ correct</span>' if pred == true_label else '<span class="cross">✗ wrong</span>'
    return f'<div class="conf">{prefix}{mark}</div>' if prefix else f"· {mark}"


def read_image(source) -> tuple:
    """Returns (PIL image, bytes) with phone-photo rotation (EXIF) applied."""
    img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner + stat cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
      <h1>♻️ Waste Classification with CNNs</h1>
      <p>Three CNNs built from scratch · TrashNet · 6 categories · see what each model looks at and why</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models, classes = load_models()
if "model_c_v3" not in models:
    st.error("Trained models not found in `models/`. Run `python -m src.train` first.", icon="🚨")
    st.stop()

final_metrics = run_metrics("model_c_v3")
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Final model", "Residual CNN"),
    (c2, "Test accuracy", final_metrics["test"]),
    (c3, "Baseline (Model B)", run_metrics("model_b")["test"]),
    (c4, "Training images", "1,768"),
]:
    col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Image input (shared by every tab)
# ─────────────────────────────────────────────────────────────────────────────
in_left, in_right = st.columns([2, 1], gap="large")
with in_left:
    source = st.segmented_control(
        "Image source", ["Upload a photo", "Sample test image"], default="Sample test image",
        label_visibility="collapsed",
    )
    image, image_bytes, true_label = None, None, None
    if source == "Upload a photo":
        uploaded = st.file_uploader("Choose a waste image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image, image_bytes = read_image(uploaded)
    else:
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        if samples:
            default = next((i for i, p in enumerate(samples) if p.stem == "metal_2"), 0)
            choice = st.selectbox(
                "Sample from the held-out test set (never seen during training)",
                samples, index=default, format_func=lambda p: p.stem.replace("_", " #"),
            )
            image, image_bytes = read_image(choice)
            true_label = choice.stem.split("_")[0]
        else:
            st.info("No sample images found in `assets/samples/`. Upload a photo instead.")
    st.markdown(
        '<div class="info-box">Tip: the models were trained on photos of <b>one object on a plain background</b>. '
        "Cluttered photos with several objects will be less accurate.</div>",
        unsafe_allow_html=True,
    )
with in_right:
    if image is not None:
        caption = f"True label: {true_label}" if true_label else "Input image"
        st.image(image, caption=caption, width="stretch")

results = analyse(image_bytes) if image_bytes else None

tab_classify, tab_how, tab_arch, tab_results = st.tabs(
    ["🔍 Classify & Compare", "🧠 How a CNN Sees", "🏗️ Architectures", "📊 Results"]
)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — Classify & Compare
# ═════════════════════════════════════════════════════════════════════════════
with tab_classify:
    from src.explain import APP_MODELS

    if results is None:
        st.info("Choose a sample or upload a photo above to see the predictions.")
    else:
        final = results["model_c_v3"]
        st.markdown(
            f"""
            <div class="final-card">
              <div class="k">Final prediction · Model C (Residual CNN)</div>
              <div class="cls">{CLASS_EMOJI.get(final['pred'], '♻️')} {final['pred']}</div>
              <div class="conf">Confidence: {final['conf'] * 100:.1f}%</div>
              {verdict(final['pred'], true_label, prefix=f"True label: {true_label} · ")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        preds = {slug: r["pred"] for slug, r in results.items()}
        if len(set(preds.values())) == 1:
            st.success(f"All three models agree: **{final['pred']}**.")
        else:
            summary = " · ".join(f"{APP_MODELS[s]['title']}: {p}" for s, p in preds.items())
            st.warning(f"The models disagree ({summary}). Model C is the most accurate (test {final_metrics['test']}).")

        st.markdown("#### Side-by-side comparison")
        cols = st.columns(len(results), gap="medium")
        for col, (slug, r) in zip(cols, results.items()):
            info, m = APP_MODELS[slug], run_metrics(slug)
            is_final = slug == "model_c_v3"
            with col:
                st.markdown(
                    f"""
                    <div class="model-card {'final' if is_final else ''}">
                      <div class="name">{info['title']} {'⭐' if is_final else ''}</div>
                      <div class="sub">{info['subtitle']} · {m['params']} params · test acc {m['test']}</div>
                      <div class="pred" style="color:{'#22c55e' if is_final else '#e2e8f0'}">
                        {CLASS_EMOJI.get(r['pred'], '')} {r['pred']}</div>
                      <div class="conf">confidence {r['conf'] * 100:.1f}% {verdict(r['pred'], true_label)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(r["overlay"], caption=f"Grad-CAM ({r['cam'].shape[0]}×{r['cam'].shape[1]} grid)",
                         width="stretch")
                for c_name, p in sorted(r["probs"].items(), key=lambda x: -x[1]):
                    st.progress(p, text=f"{CLASS_EMOJI.get(c_name, '')} {c_name} — {p * 100:.1f}%")

        st.markdown(
            """
            <div class="info-box">
            <b>How to read the heatmaps (Grad-CAM).</b> Red areas pushed the model most towards its predicted
            class; blue areas mattered little. Models A and B often focus on small patches, edges or even the
            background, because each of their final features only "sees" about a 22×22-pixel patch. Model C's
            features each see about 96×96 pixels, so its heatmap usually covers the whole object. Its grid is
            coarser (10×10 vs 32×32) because it down-samples more before the final layer.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — How a CNN Sees
# ═════════════════════════════════════════════════════════════════════════════
with tab_how:
    from src import explain as ex

    st.markdown(
        """
        ### From pixels to a prediction
        A CNN turns an image into a prediction in stages. Early layers find simple patterns (edges,
        colours), middle layers combine them into textures and parts, and the last layers decide which
        class the combination looks like.

        **Input image → [Convolution → BatchNorm → ReLU → Pooling] × N → Global Average Pooling → Dense → Softmax → class probabilities**
        """
    )

    # --- 1. Convolution ---------------------------------------------------
    st.markdown("---\n#### 1 · Convolution: sliding a small filter over the image")
    st.markdown(
        """
        A **kernel** (filter) is a small grid of weights, e.g. 3×3. It slides across the image, and at each
        position it multiplies the pixels under it by its weights and adds them up. The result is a
        **feature map**: bright wherever the pattern the kernel describes is present.

        `output(i, j) = Σ kernel(m, n) × image(i + m, j + n)`

        Below, four classic **hand-designed** kernels are applied to the current image. A CNN does the same
        thing, but **learns** its kernel weights from data: Model C's first layer learns 32 of them.
        """
    )
    if image is not None:
        kcols = st.columns(len(ex.DEMO_KERNELS))
        for col, (k_name, kernel) in zip(kcols, ex.DEMO_KERNELS.items()):
            with col:
                st.image(ex.apply_kernel(image, kernel), caption=k_name, width="stretch")
                st.code("\n".join(" ".join(f"{v:5.2f}" for v in row) for row in kernel), language=None)
    st.markdown("**The 32 kernels Model C actually learned in its first layer** (each a 3×3×3 RGB filter, enlarged):")
    st.image(ex.first_layer_kernels(models["model_c_v3"], "c_stem_conv"), width=48)
    st.caption("Many look like colour detectors or edge detectors in different directions, learned automatically during training.")

    # --- 2. ReLU -------------------------------------------------------------
    st.markdown("---\n#### 2 · ReLU: keep the positives")
    st.markdown(
        """
        After each convolution, **ReLU** (`f(x) = max(0, x)`) sets negative values to zero. Without such a
        non-linearity, stacking layers would collapse into one big linear filter. With it, the network can
        learn complex shapes.

        **BatchNorm**, placed between the convolution and the ReLU, rescales each feature map to a steady
        range. That keeps training stable and lets the network use a higher learning rate.
        """
    )

    # --- 3. Pooling ----------------------------------------------------------
    st.markdown("---\n#### 3 · Max pooling: shrink the map, keep the strongest signal")
    pc1, pc2 = st.columns([1, 1], gap="large")
    with pc1:
        st.markdown(
            """
            **Max pooling (2×2)** splits a feature map into 2×2 blocks and keeps only the largest value in
            each. The map becomes half as wide and half as tall (4× fewer values). This:
            - makes the network cheaper to run
            - makes features robust to small shifts of the object
            - lets later layers "see" a larger part of the image

            Example on a 4×4 feature map:
            """
        )
        grid = pd.DataFrame([[1, 3, 2, 0], [5, 6, 1, 2], [7, 2, 9, 4], [1, 8, 3, 5]])
        pooled = pd.DataFrame([[6, 2], [8, 9]])
        g1, g2 = st.columns(2)
        g1.markdown("Input 4×4")
        g1.dataframe(grid, hide_index=True, width="content")
        g2.markdown("After 2×2 max pool")
        g2.dataframe(pooled, hide_index=True, width="content")
        st.caption("Top-left block [1, 3, 5, 6] → 6 · top-right [2, 0, 1, 2] → 2 · and so on.")
    with pc2:
        if results is not None:
            fm = results["model_c_v3"]["fmaps"]
            before, after = fm["c_stem_relu"], fm["c_stem_pool"]
            b1, b2 = st.columns(2)
            b1.image(ex.channel_image(before["channels"][0], 200),
                     caption=f"Before pooling: {before['shape'][0]}×{before['shape'][1]}", width="stretch")
            b2.image(ex.channel_image(after["channels"][0], 200),
                     caption=f"After max pooling: {after['shape'][0]}×{after['shape'][1]}", width="stretch")
            st.caption("A real feature map from Model C for the current image, before and after its max-pooling layer. "
                       "Same pattern, half the resolution. Model C's later stages halve the size again with "
                       "stride-2 convolutions instead of pooling.")

    # --- 4. Feature maps through depth ---------------------------------------
    st.markdown("---\n#### 4 · Feature maps: what each layer responds to")
    if results is not None:
        fm_model = st.segmented_control(
            "Model", ["model_c_v3", "model_b", "model_a"], default="model_c_v3",
            format_func=lambda s: ex.APP_MODELS[s]["title"], key="fm_model",
        ) or "model_c_v3"
        st.caption("The 6 most active channels of each layer for the current image (yellow = strong response).")
        info, fmaps = ex.APP_MODELS[fm_model], results[fm_model]["fmaps"]
        for name, caption in info["feature_layers"]:
            h, w, c = fmaps[name]["shape"]
            st.markdown(f"**{caption}** · `{name}` · {h}×{w} × {c} channels")
            ccols = st.columns(6)
            for col, ch in zip(ccols, fmaps[name]["channels"]):
                col.image(ex.channel_image(ch, 110), width="stretch")
        st.markdown(
            """
            <div class="info-box">
            Notice how the maps get <b>smaller but more numerous</b> as you go deeper (e.g. 160×160×32 → 10×10×256
            in Model C). Early maps still look like the photo (edges, outlines). Deep maps are coarse blobs that
            mark <i>where</i> high-level parts were found. This is the information the classifier uses.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 5. Classification head ----------------------------------------------
    st.markdown("---\n#### 5 · From feature maps to a class")
    st.markdown(
        """
        - **Global Average Pooling** averages each final feature map to a single number. Model C's 10×10×256
          output becomes a vector of 256 numbers: "how much of each high-level feature is in the image".
        - **Dropout** randomly switches off some of these numbers during training only, so the model can't
          rely on any single feature (less overfitting).
        - **Dense + Softmax** combines the 256 numbers into 6 scores and turns them into probabilities that sum
          to 1. The highest one is the prediction.
        """
    )
    if results is not None:
        probs = results["model_c_v3"]["probs"]
        st.bar_chart(pd.DataFrame({"probability": probs}), horizontal=True, height=220)

    # --- 6. Skip connections -------------------------------------------------
    st.markdown("---\n#### 6 · Skip connections: why Model C works better")
    sc1, sc2 = st.columns([3, 2], gap="large")
    with sc1:
        st.markdown(
            """
            A residual block computes `output = F(x) + x`: it adds its **input back** to what its layers
            computed. The layers only have to learn the *change* to apply, and the shortcut gives the
            training signal a direct path back to early layers. Without it, deeper networks become hard to
            train (the "vanishing gradient" problem).

            Our ablation study shows this directly. A 9-layer network **without** skip connections (C-plain)
            did no better than the 3-layer Model B on validation. Adding skip connections (Model C) gained
            **+5.5 points**, with almost the same number of parameters.
            """
        )
    with sc2:
        st.code(
            "      x\n      │\n      ├──────────────┐\n  Conv 3×3 → BN → ReLU   │ shortcut\n"
            "  Conv 3×3 → BN          │ (1×1 conv if\n      │                  │  shape changes)\n"
            "      (+) ◄─────────────┘\n      │\n    ReLU\n      │\n    output",
            language=None,
        )

    # --- 7. Grad-CAM -----------------------------------------------------------
    st.markdown("---\n#### 7 · Grad-CAM: how the heatmaps are made")
    st.markdown(
        """
        1. Run the image through the network and keep the **last convolutional feature maps**.
        2. Compute the **gradient** of the predicted class score with respect to each map: how much would the
           score change if that map's values went up?
        3. Average each gradient into one **importance weight** per map, take the weighted sum of the maps,
           and keep only positive values (ReLU).
        4. Enlarge the result to the image size and colour it: red = most important.

        It shows *where* the evidence for the prediction is, which helps check that the model looks at the
        object and not at the background.
        """
    )
    if results is not None:
        st.image(ex.heatmap_image(results["model_c_v3"]["cam"], (220, 220)),
                 caption="Model C's raw 10×10 Grad-CAM grid for the current image, before enlarging and blending",
                 width=220)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Architectures
# ═════════════════════════════════════════════════════════════════════════════
with tab_arch:
    from src import explain as ex

    st.markdown("### The three architectures at a glance")
    summary = pd.DataFrame(
        {
            "Model A · Small CNN": ["128×128", "3 (3×3)", "0", "3", "3 max-pool", "0", "16 → 32 → 64",
                                    "GAP → Dense(64) → Dropout 0.3 → Dense(6)", "≈22 px", "28,470"],
            "Model B · Baseline CNN": ["128×128", "3 (3×3)", "0", "3", "3 max-pool", "0", "32 → 64 → 128",
                                       "GAP → Dense(128) → Dropout 0.4 → Dense(6)", "≈22 px", "111,206"],
            "Model C · Residual CNN": ["160×160", "9 (3×3)", "3 (1×1 shortcut)", "12",
                                       "1 max-pool + 3 stride-2 convs", "4", "32 → 32 → 64 → 128 → 256",
                                       "GAP → Dropout 0.3 → Dense(6)", "≈96 px", "1,230,822"],
        },
        index=["Input size", "Conv layers", "Projection convs", "BatchNorm layers", "Down-sampling",
               "Residual blocks (skip connections)", "Filters per stage", "Classifier head",
               "Receptive field of final features", "Parameters"],
    )
    st.dataframe(summary, width="stretch")
    st.caption("Receptive field = how many input pixels one final feature can 'see'. "
               "All three models use the same augmentation (flip, rotation, zoom, brightness, contrast), applied only during training.")

    arch_text = {
        "model_a": """
Input 128×128×3 → Rescaling (÷255) → Augmentation (training only)
→ Conv 3×3 (16) → BN → ReLU → MaxPool 2×2      → 64×64×16
→ Conv 3×3 (32) → BN → ReLU → MaxPool 2×2      → 32×32×32
→ Conv 3×3 (64) → BN → ReLU → MaxPool 2×2      → 16×16×64
→ GlobalAveragePooling → Dense(64, ReLU) → Dropout(0.3) → Dense(6, Softmax)""",
        "model_b": """
Input 128×128×3 → Rescaling (÷255) → Augmentation (training only)
→ Conv 3×3 (32)  → BN → ReLU → MaxPool 2×2     → 64×64×32
→ Conv 3×3 (64)  → BN → ReLU → MaxPool 2×2     → 32×32×64
→ Conv 3×3 (128) → BN → ReLU → MaxPool 2×2     → 16×16×128
→ GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.4) → Dense(6, Softmax)""",
        "model_c_v3": """
Input 160×160×3 → Rescaling (÷255) → Augmentation (training only)
→ Stem: Conv 3×3 (32) → BN → ReLU → MaxPool 2×2      → 80×80×32
→ Stage 1: Residual block (32)                       → 80×80×32
→ Stage 2: Residual block (64),  stride 2            → 40×40×64
→ Stage 3: Residual block (128), stride 2            → 20×20×128
→ Stage 4: Residual block (256), stride 2            → 10×10×256
→ GlobalAveragePooling → Dropout(0.3) → Dense(6, Softmax)

Residual block: x → Conv 3×3 → BN → ReLU → Conv 3×3 → BN → (+ shortcut) → ReLU
(the shortcut uses a 1×1 stride-2 conv + BN when the shape changes)""",
    }
    why = {
        "model_a": "The smallest model: a quick baseline to see how far a tiny CNN gets.",
        "model_b": "Twice as many filters as A. More capacity, but still only 3 layers deep, so it underfits.",
        "model_c_v3": "Deeper (9 conv layers) with skip connections and a larger input. Each final feature sees "
                      "most of the object, which fixes the glass/metal/plastic confusions of A and B.",
    }
    for slug in ["model_a", "model_b", "model_c_v3"]:
        info, m = ex.APP_MODELS[slug], run_metrics(slug)
        with st.expander(f"{info['title']} · {info['subtitle']} — {m['params']} parameters · test accuracy {m['test']}",
                         expanded=(slug == "model_c_v3")):
            st.markdown(why[slug])
            st.code(arch_text[slug].strip(), language=None)
            st.markdown("**Every layer** (generated from the saved model):")
            st.dataframe(ex.layer_table(models[slug]), hide_index=True, width="stretch", height=320)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — Results
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    df = load_experiments()
    st.markdown("### Test results (380 held-out images)")
    if not df.empty:
        names = {
            "model_a": "Model A · Small CNN", "model_b": "Model B · Baseline CNN",
            "model_b_v3": "Model B · new recipe", "model_c_plain_v3": "Model C-plain · no skips",
            "model_c_v3": "⭐ Model C · Residual CNN",
        }
        shown = df[df["run_name"].isin(names)].copy()
        shown["Model"] = shown["run_name"].map(names)
        shown["Val acc"] = (shown["saved_val_accuracy"].astype(float) * 100).round(1)
        shown["Test acc"] = pd.to_numeric(shown["test_accuracy"], errors="coerce").mul(100).round(1)
        shown = shown.rename(columns={"recipe": "Recipe", "param_count": "Parameters",
                                      "image_dimensions": "Input", "device": "Trained on"})
        shown = shown.set_index("run_name").loc[[r for r in names if r in set(shown["run_name"])]]
        st.dataframe(shown[["Model", "Recipe", "Input", "Parameters", "Val acc", "Test acc", "Trained on"]],
                     hide_index=True, width="stretch")

    st.markdown(
        """
        #### Ablation study: what caused the improvement?
        | Step | Change | Val acc | Test acc |
        |---|---|---|---|
        | Model B, original recipe | baseline | 69.7% | 68.4% |
        | Model B, **new training recipe** | better training only | 82.3% | 75.3% |
        | Model C-plain, **deeper** (no skips) | + depth & 160 px input | 82.6% | 78.7% |
        | **Model C, + skip connections** | + residual connections | **88.1%** | **81.6%** |

        - **Training recipe** (full cosine learning-rate schedule, AdamW, label smoothing, BatchNorm momentum 0.9)
          gave the largest single gain.
        - **Depth** helped on the test set (+3.4) but not on validation (+0.3). With ~380 images per split,
          differences of 2–3 points are within noise.
        - **Skip connections** improved both splits (+5.5 val, +2.9 test).
        """
    )

    st.markdown("#### Model C: test confusion matrix and per-class F1")
    r1, r2 = st.columns(2)
    for col, fname in [(r1, "confusion_matrix_model_c_v3.png"), (r2, "per_class_f1_scores_model_c_v3.png")]:
        if (PLOTS_DIR / fname).exists():
            col.image(str(PLOTS_DIR / fname), width="stretch")

    st.markdown("#### Training curves")
    curve = st.segmented_control(
        "Run", ["model_c_v3", "model_c_plain_v3", "model_b_v3", "model_b", "model_a"], default="model_c_v3",
        key="curve_run",
    ) or "model_c_v3"
    curve_path = PLOTS_DIR / f"training_curves_{curve}.png"
    if curve_path.exists():
        st.image(str(curve_path), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Model limitations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="limitation-box">
      <strong>⚠️ Limitations</strong><br>
      Trained only on TrashNet (2,527 studio-style photos, one object on a plain background); accuracy on
      cluttered real-world photos will be lower. The model always picks one of 6 categories and cannot say
      "unknown". Softmax confidence ranks the classes; it is not a calibrated probability.
    </div>
    """,
    unsafe_allow_html=True,
)
