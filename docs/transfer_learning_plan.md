# Future Work: Transfer Learning Comparison

Status: **planned, not started**. To be done after the from-scratch CNN experiments are finished.

## 1. Goal

Fine-tune ImageNet-pretrained CNNs on our TrashNet split and compare them against our
from-scratch CNNs (Model A, Model B, and any new scratch models). Then test every model on a
**new, unseen dataset** to measure how well each one generalises beyond TrashNet.

Research question for the report:
> How much does pretrained knowledge (transfer learning) improve accuracy and generalisation
> over a CNN trained from scratch on a small dataset (~1,768 training images)?

## 2. What transfer learning is (short)

A network already trained on ImageNet (1.2M images, 1,000 classes) has learned general visual
features: edges and colours in early layers, textures and shapes in middle layers, object parts
in deep layers. We reuse those layers, replace the final 1000-class layer with our own 6-class
softmax, and train on our data. Training happens in two stages:

1. **Feature extraction:** freeze the pretrained base and train only the new head.
2. **Fine-tuning:** unfreeze the top part of the base and train with a very small learning rate
   so the features adapt to waste images.

## 3. Candidate pretrained models

| Model | Params (approx.) | Why include it |
|---|---|---|
| **MobileNetV2** (primary) | 3.5M | Lightweight, fast on CPU, good for the Streamlit app |
| EfficientNetB0 | 5.3M | Usually the best accuracy-per-parameter |
| ResNet50 (optional) | 25.6M | Classic, well-known baseline for the report |

Start with MobileNetV2; add the others only if time allows.

## 4. Data pipeline changes

- **Same splits:** reuse `data/splits/train.csv`, `val.csv`, `test.csv` (seed 42) so the comparison
  with Models A/B is fair. The test set stays untouched until final evaluation.
- **Input size:** 224×224 (the ImageNet standard) instead of 128×128.
- **Preprocessing:** use each model's own `preprocess_input`
  (e.g. `tf.keras.applications.mobilenet_v2.preprocess_input`, which scales pixels to [-1, 1])
  instead of our `Rescaling(1/255)` layer. Using the wrong preprocessing silently ruins accuracy.
- **Augmentation:** same augmentation block as Models A/B (flip, rotation, zoom, brightness,
  contrast), applied during training only.
- **Class weights:** use plain inverse-frequency weights, *or* the current multipliers, but use the
  **same** choice for every model being compared.

## 5. Training recipe

```python
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False                      # Stage 1: freeze

inputs = tf.keras.Input((224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)                 # keep BatchNorm in inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
```

| Stage | Trainable layers | Learning rate | Epochs (max) | Callbacks |
|---|---|---|---|---|
| 1. Feature extraction | Head only | 1e-3 | 15 | EarlyStopping on val_loss |
| 2. Fine-tuning | Top ~30 layers of base + head | 1e-5 | 20 | EarlyStopping, ReduceLROnPlateau |

Notes:
- Keep BatchNormalization layers frozen during fine-tuning (`training=False` on the base call).
- Recompile the model after changing `trainable`.
- Save as `models/tl_mobilenetv2_best.keras` and history JSON, same style as Models A/B.

## 6. Evaluation plan

### 6a. TrashNet test set (in-distribution)
Run the same metrics as `src/evaluate.py` for every model:
accuracy, macro F1, per-class precision/recall/F1, confusion matrix.

### 6b. New dataset (out-of-distribution generalisation)
Predict on a dataset the models have **never seen**, to check real-world robustness.
Options (pick one):
- **Own photos:** ~20–30 phone photos per class taken in real conditions (bins, floor, hands,
  cluttered backgrounds). The most convincing option for a demo.
- **Kaggle "Garbage Classification" (12 classes)** or **RealWaste**: map their classes onto our 6
  (e.g. brown-glass/green-glass/white-glass → glass, biological/clothes/shoes → trash) and drop
  classes that don't map cleanly.

Expected finding: scratch CNNs drop sharply on new data (they learned TrashNet's white
background and lighting), while pretrained models drop less.

## 7. Comparison table (to fill in)

| Model | Type | Params | Input | Train time | Val acc | Test acc (TrashNet) | Macro F1 | Acc on new dataset |
|---|---|---|---|---|---|---|---|---|
| Model A | Scratch CNN | 28K | 128 | 5.5 min | 66.0% | — | — | — |
| Model B | Scratch CNN | 111K | 128 | 15.7 min | 69.7% | 68.4% | 0.66 | — |
| Model C (new) | Scratch CNN | — | — | — | — | — | — | — |
| MobileNetV2 | Transfer learning | 3.5M | 224 | — | — | — | — | — |
| EfficientNetB0 | Transfer learning | 5.3M | 224 | — | — | — | — | — |

## 8. Files to add when implementing

- `src/model_transfer.py`: builders for the pretrained models
- `src/train_transfer.py`: two-stage training
- `src/evaluate_external.py`: prediction and metrics on the new dataset, with class mapping
- Update `app.py` with a dropdown to choose which model to use
- Update `README.md` with the comparison table and conclusions
