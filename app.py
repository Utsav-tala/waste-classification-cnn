"""
Streamlit web application for Waste Classification.
Run with: streamlit run app.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="♻️ Waste Classifier CNN",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; color: #e8eaed; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a6b3a 0%, #0d4d76 60%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 36px 40px 28px 40px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .hero-banner h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0; }
    .hero-banner p  { color: #b0c4d8; font-size: 1.05rem; margin: 0; }

    /* Metric cards */
    .metric-card {
        background: #1e2430;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2d3748;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-card .label { color: #94a3b8; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #38bdf8; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }

    /* Prediction result */
    .result-card {
        background: linear-gradient(135deg, #1a3a2a 0%, #0d2640 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px 28px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(34,197,94,0.15);
    }
    .result-card .pred-label { color: #94a3b8; font-size: 0.9rem; margin-bottom: 6px; }
    .result-card .pred-class { color: #22c55e; font-size: 2.4rem; font-weight: 700; text-transform: capitalize; }
    .result-card .pred-conf  { color: #fbbf24; font-size: 1.2rem; font-weight: 600; margin-top: 4px; }

    /* Warning box */
    .limitation-box {
        background: #1c1a0f;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        color: #fde68a;
        font-size: 0.88rem;
        margin-top: 18px;
    }

    /* Section headers */
    .section-header {
        color: #60a5fa;
        font-size: 1.05rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3748;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #16a34a, #0d9488);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 26px;
        font-size: 1rem;
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path(__file__).resolve().parent / "models"
CLASS_COLORS = {
    "cardboard": "#e97316",
    "glass":     "#0ea5e9",
    "metal":     "#8b5cf6",
    "paper":     "#f43f5e",
    "plastic":   "#22d3ee",
    "trash":     "#a3a3a3",
}
CLASS_EMOJI = {
    "cardboard": "📦",
    "glass":     "🍶",
    "metal":     "🔩",
    "paper":     "📄",
    "plastic":   "🧴",
    "trash":     "🗑️",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loader — loaded exactly once per Streamlit session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model into memory…")
def load_resources():
    """Returns (model, classes) from disk, cached for the session lifetime."""
    import tensorflow as tf

    model_path  = MODELS_DIR / "model_b_best.keras"
    labels_path = MODELS_DIR / "class_labels.json"

    if not model_path.exists():
        # Fallback to model_a if model_b is not found
        model_path = MODELS_DIR / "model_a_best.keras"

    if not model_path.exists():
        return None, None

    model = tf.keras.models.load_model(str(model_path))

    with open(labels_path) as f:
        data = json.load(f)
    classes = data["classes"]
    return model, classes


def predict_image(pil_image: Image.Image, model, classes: list) -> dict:
    """Preprocesses and runs inference — consistent with training pipeline."""
    import numpy as np
    import tensorflow as tf

    img = pil_image.convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)[np.newaxis]           # (1,128,128,3)
    probs = model(arr, training=False).numpy()[0]
    idx   = int(np.argmax(probs))
    return {
        "predicted_class": classes[idx],
        "confidence":      float(probs[idx]),
        "probabilities":   {c: float(p) for c, p in zip(classes, probs)},
    }


def probability_bar_chart(prob_dict: dict):
    """Renders a horizontal bar chart of class probabilities."""
    classes = list(prob_dict.keys())
    values  = [prob_dict[c] * 100 for c in classes]
    colors  = [CLASS_COLORS.get(c, "#64748b") for c in classes]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor="#1e2430")
    ax.set_facecolor("#1e2430")
    bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor="none")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", color="white", fontsize=9,
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Softmax Probability (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(colors="#94a3b8")
    fig.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
      <h1>♻️ Waste Classification System</h1>
      <p>Convolutional Neural Network (CNN) · TrashNet Dataset · 6 Categories · Academic Demonstration</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset stat cards
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="label">Dataset Images</div><div class="value">2,527</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="label">Waste Categories</div><div class="value">6</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="label">Input Resolution</div><div class="value">128×128</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="label">Architecture</div><div class="value">CNN</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
model, classes = load_resources()
if model is None:
    st.error(
        "⚠️  No trained model found in `models/`.  "
        "Please run `python -m src.train` first, then restart the app.",
        icon="🚨",
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout: uploader left, results right
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<p class="section-header">📤 Upload Image</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a waste image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        # Supported-format guard
        if pil_img.format and pil_img.format.upper() not in {"JPEG", "JPG", "PNG", "WEBP"}:
            st.warning(f"Unsupported format detected: {pil_img.format}. Results may be unreliable.")

        predict_btn = st.button("🔍 Classify Waste", key="predict_btn")
    else:
        predict_btn = False
        st.info("Upload an image to get started.")

with right_col:
    st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

    if uploaded and predict_btn:
        with st.spinner("Analysing image…"):
            try:
                result = predict_image(pil_img, model, classes)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        cls   = result["predicted_class"]
        conf  = result["confidence"]
        probs = result["probabilities"]
        emoji = CLASS_EMOJI.get(cls, "♻️")

        # Result card
        st.markdown(
            f"""
            <div class="result-card">
              <div class="pred-label">Predicted Waste Category</div>
              <div class="pred-class">{emoji} {cls.capitalize()}</div>
              <div class="pred-conf">Softmax confidence: {conf * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability table
        st.markdown('<p class="section-header">Class Probability Breakdown</p>', unsafe_allow_html=True)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        for c_name, p in sorted_probs:
            em = CLASS_EMOJI.get(c_name, "•")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(p, text=f"{em} {c_name.capitalize()}")
            with col_b:
                st.markdown(f"<div style='padding-top:8px;text-align:right;color:#94a3b8;font-size:0.9rem'>{p*100:.1f}%</div>", unsafe_allow_html=True)

        # Bar chart
        st.markdown('<p class="section-header">Probability Bar Chart</p>', unsafe_allow_html=True)
        fig = probability_bar_chart(probs)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    elif not uploaded:
        st.markdown(
            """
            <div style="background:#1e2430;border-radius:12px;padding:40px;text-align:center;color:#475569;margin-top:12px;">
              <div style="font-size:3rem;">♻️</div>
              <div style="margin-top:12px;font-size:1.05rem;">Upload an image and click <strong>Classify Waste</strong></div>
              <div style="margin-top:8px;font-size:0.85rem;">Supported categories: cardboard · glass · metal · paper · plastic · trash</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not predict_btn:
        st.info("Click **Classify Waste** to run the model.")

# ─────────────────────────────────────────────────────────────────────────────
# Model limitations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="limitation-box">
      <strong>⚠️ Model Limitations</strong><br>
      This model was trained exclusively on the TrashNet dataset (2,527 images, 512×384 px, studio-style photos).
      Performance on real-world photos (variable lighting, angles, occlusions) may be lower.
      The softmax confidence score reflects the model's ranking between classes — it is not a calibrated probability.
      The model supports exactly 6 categories: <em>cardboard, glass, metal, paper, plastic, trash</em>.
      Do not use for safety-critical or regulatory waste-sorting decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# About / CNN explanation (expandable)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📚 About This Project & CNN Architecture"):
    st.markdown(
        """
        ### Waste Classification Using CNN
        This project demonstrates how a **Convolutional Neural Network (CNN)** can automatically learn to
        distinguish between six types of waste material from a photograph.

        #### How the CNN works
        | Layer | Purpose |
        |---|---|
        | **Conv2D** | Applies small learnable filters (kernels) that slide across the image to detect edges, textures, and shapes. |
        | **BatchNorm + ReLU** | Normalises activations and introduces non-linearity, allowing the network to learn complex patterns. |
        | **MaxPooling** | Down-samples feature maps by retaining the strongest activation in each 2×2 region, reducing spatial size. |
        | **GlobalAveragePooling** | Collapses each feature map to a single value, summarising what the network has detected. |
        | **Dense + Dropout** | Final classification head; Dropout prevents overfitting by randomly disabling neurons during training. |
        | **Softmax** | Converts raw outputs into six class probabilities that sum to 1. |

        #### Dataset
        - **Source:** TrashNet (resized to 512×384 px)
        - **Training split:** 1,768 images (70%)
        - **Validation split:** 379 images (15%)
        - **Test split:** 380 images (15%)

        #### Training Details
        - Optimiser: Adam · Loss: Sparse Categorical Cross-Entropy
        - Early stopping (patience 8) · LR reduction on plateau
        - Class weighting to handle minority class (`trash`)
        - Data augmentation: random horizontal flip, rotation, zoom, brightness, contrast (training only)
        """
    )
