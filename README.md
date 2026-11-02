# ♻️ Waste Classification with CNNs

Classifying photos of waste into **6 recycling categories** (cardboard, glass, metal, paper, plastic, trash) with Convolutional Neural Networks built **from scratch** in TensorFlow/Keras, plus a **Streamlit** web app for live predictions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-green)

<p align="center">
  <img src="results/plots/sample_images.png" width="800" alt="Sample images from each class"/>
</p>

---

## Highlights

- **End-to-end ML pipeline:** dataset validation → stratified splitting → training → held-out evaluation → inference app.
- **Data integrity checks:** every image is verified, and SHA-256 hashing detects duplicate images (3 cross-class duplicates found in TrashNet).
- **Class imbalance handling:** inverse-frequency class weights computed from the training split only (the `trash` class has 4× fewer images than `paper`).
- **Two from-scratch CNNs** compared under identical conditions; the model is selected on validation accuracy, and the test set is used **once** for the final score.
- **Efficient input pipeline:** images are loaded lazily with `tf.data` and augmented inside the model graph (training only).
- **Streamlit app** showing the predicted class, confidence and the full probability breakdown.

---

## Results

| Model | Recipe | Params | Input | Epochs | Train time | Val acc (saved model) | Test acc |
|---|---|---|---|---|---|---|---|
| Model A — Small CNN | v1 | 28.5K | 128 | 50 | 5.5 min (CPU) | 65.2% | not evaluated* |
| Model B — Baseline CNN | v1 | 111K | 128 | 50 | 15.7 min (CPU) | 69.7% | 68.4% |
| **Model C — Residual CNN** | **v3** | 1.23M | 160 | 100 | 10 min (T4 GPU) | **88.1%** | pending |

\* Test accuracy is measured only for a model chosen on validation accuracy, and only once. Model C's test evaluation will follow the ablation study.

The sections below analyse Model B, the original baseline. Model C results are in [Model C](#model-c--residual-cnn).

**Per-class test results (Model B, 380 test images):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cardboard | 0.89 | 0.92 | **0.90** | 60 |
| paper | 0.71 | 0.87 | 0.78 | 89 |
| plastic | 0.74 | 0.66 | 0.70 | 73 |
| glass | 0.82 | 0.41 | 0.54 | 76 |
| metal | 0.63 | 0.47 | 0.54 | 62 |
| trash | 0.33 | 1.00 | 0.49 | 20 |
| **macro avg** | 0.69 | 0.72 | 0.66 | 380 |

<p align="center">
  <img src="results/plots/confusion_matrix_model_b.png" width="48%" alt="Confusion matrix"/>
  <img src="results/plots/per_class_f1_scores_model_b.png" width="48%" alt="Per-class F1"/>
</p>

### Analysis

- **Cardboard and paper are easy.** They have a distinctive colour and texture (cardboard pixels average much more red than blue).
- **Glass and metal are hard.** Glass is confused with plastic, metal and trash: reflective, transparent surfaces lose detail at 128×128.
- **Trash is over-predicted.** The minority class weight (≈3.07×) gives perfect trash recall but only 33% precision. Tuning these weights is the next experiment.
- **The models underfit rather than overfit.** Training and validation accuracy are both about 65–70% and still rising at epoch 50. Each output feature of the 3-block CNN only "sees" about a 22×22 pixel patch (its receptive field), so the models learn colour and texture but not whole-object shape. That motivates a deeper architecture (see [Roadmap](#roadmap)).

<p align="center">
  <img src="results/plots/training_curves_model_b.png" width="800" alt="Training curves for Model B"/>
</p>

---

## Dataset

[**TrashNet**](https://github.com/garythung/trashnet) (Thung & Yang, 2016): 2,527 RGB photos of single waste items on a plain background, resized to 512×384.

| Class | Images | Share |
|---|---|---|
| paper | 594 | 23.5% |
| glass | 501 | 19.8% |
| plastic | 482 | 19.1% |
| metal | 410 | 16.2% |
| cardboard | 403 | 16.0% |
| trash | 137 | 5.4% |

**Stratified split (seed 42):** train 1,768 (70%) / validation 379 (15%) / test 380 (15%). Split files are in [`data/splits/`](data/splits).

**Integrity:** 0 corrupt images. 3 identical images appear under two different labels (glass↔metal, glass↔plastic ×2). They were kept and documented as label noise. Full report: [`results/reports/dataset_validation_report.txt`](results/reports/dataset_validation_report.txt).

---

## Model Architecture

Both models share the same design and differ only in width:

```
Input (128×128×3, uint8)
 → Rescaling (÷255)
 → Augmentation — training only: flip, rotate ±27°, zoom ±15%, brightness ±20%, contrast ±20%
 → [Conv2D 3×3 → BatchNorm → ReLU → MaxPool 2×2] × 3
 → GlobalAveragePooling2D
 → Dense (ReLU) → Dropout
 → Dense(6, Softmax)
```

| | Model A | Model B |
|---|---|---|
| Conv filters | 16 → 32 → 64 | 32 → 64 → 128 |
| Dense units | 64 | 128 |
| Dropout | 0.3 | 0.4 |
| Parameters | 28,470 | 111,206 |

**Training setup:** Adam (lr 1e-3), sparse categorical cross-entropy, batch size 32, up to 50 epochs, EarlyStopping (patience 8, restores best weights), ReduceLROnPlateau (×0.5, patience 4), class weights.

Rescaling and augmentation live inside the model, so the saved `.keras` file takes raw images and needs no extra preprocessing at inference time.

### Model C — Residual CNN

A deeper from-scratch network designed to fix the underfitting of Models A/B:

```
Input (160×160×3)
 → Rescaling → Augmentation (training only)
 → Stem: Conv 3×3 (32) → BN → ReLU → MaxPool         → 80×80×32
 → Stage 1: Residual block (32)                      → 80×80×32
 → Stage 2: Residual block (64),  stride 2           → 40×40×64
 → Stage 3: Residual block (128), stride 2           → 20×20×128
 → Stage 4: Residual block (256), stride 2           → 10×10×256
 → GlobalAveragePooling2D → Dropout(0.3) → Dense(6, Softmax)

Residual block: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ shortcut) → ReLU
(1×1 strided projection on the shortcut when the shape changes)
```

| | Model A | Model B | Model C |
|---|---|---|---|
| Input | 128×128 | 128×128 | 160×160 |
| Conv layers | 3 | 3 | 9 (+3 shortcut projections) |
| Skip connections | ✗ | ✗ | ✓ |
| Receptive field of final features | ≈22 px | ≈22 px | ≈96 px |
| Parameters | 28K | 111K | 1.23M |

**Ablation:** `model_c_plain` is the same network without skip connections, to measure what the residual connections add.

**Training recipes** (see `TRAINING_RECIPES` in [`src/config.py`](src/config.py)):

| Recipe | Setup | Used for |
|---|---|---|
| v1 | Adam, ReduceLROnPlateau, early stopping (patience 8), 50 epochs | Original Model A/B results |
| v2 | AdamW (wd 1e-4), cosine LR, label smoothing 0.1, softer `trash` weight (×0.5), early stopping (patience 15), 80 epochs | First Model C run |
| v3 | v2 + **no early stopping** (full 100-epoch cosine run, final weights kept) + **BatchNorm momentum 0.9** | Model C and the ablation |

**First Model C run (v2, Colab T4, 4.7 min):** 76.3% validation accuracy vs 69.7% for Model B, with glass recall up from about 41% to 71%. However, validation accuracy swung between about 35% and 76% from one epoch to the next, and early stopping kept a single peak (epoch 32 of 80) before the learning rate had decayed. A second v2 run with identical settings reached only 68.1%, which confirmed the instability. Runs are compared on `saved_val_accuracy` (the validation accuracy of the weights actually saved), not on the best single epoch.

**Model C with recipe v3 (Colab T4, 10 min): 88.1% validation accuracy.** Validation accuracy is still noisy while the learning rate is high, but it settles as the cosine schedule decays: over the last 10 epochs it averages 87.8% ± 1.0. Training accuracy ends at about 92%, so the train/val gap is small and the model is not overfitting.

| Epochs | Mean val acc | Std |
|---|---|---|
| 1–25 | 47.5% | 10.7 |
| 26–50 | 55.9% | 14.4 |
| 51–75 | 70.8% | 11.1 |
| 76–90 | 84.2% | 3.9 |
| 91–100 | 87.8% | 1.0 |

Per-class validation recall improved most on the classes Model B struggled with: glass 83%, metal 79%, plastic 94%, paper 91%, cardboard 93%, trash 86%. Trash precision rose from 33% to 64%, so trash is no longer over-predicted.

<p align="center">
  <img src="results/plots/training_curves_model_c_v3.png" width="800" alt="Training curves for Model C (v3)"/>
</p>
<p align="center">
  <img src="results/plots/val_confusion_matrix_model_c_v3.png" width="55%" alt="Validation confusion matrix for Model C (v3)"/>
</p>

---

## Project Structure

```
waste-classification-cnn/
├── app.py                     # Streamlit web app
├── requirements.txt
├── src/
│   ├── config.py              # paths, classes, hyperparameters
│   ├── data_preparation.py    # validation, duplicate detection, stratified splits
│   ├── model.py               # Model A/B/C definitions, model registry, smoke test
│   ├── train.py               # training CLI (models + recipe), curves, val confusion matrix
│   ├── evaluate.py            # final test-set evaluation for a chosen run
│   └── predict.py             # reusable single-image inference
├── notebooks/
│   └── train_colab.ipynb      # GPU training on Google Colab
├── data/splits/               # train / val / test CSVs (file paths + labels)
├── models/                    # trained .keras models, histories, class labels
├── results/
│   ├── plots/                 # distribution, samples, curves, confusion matrices
│   └── reports/               # dataset report, classification report, experiment table
└── docs/
    └── transfer_learning_plan.md
```

---

## Getting Started

### 1. Set up the environment (Python 3.12)

```bash
git clone https://github.com/Utsav-tala/waste-classification-cnn.git
cd waste-classification-cnn
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the app (trained models are included)

```bash
streamlit run app.py
```

Upload a JPG/PNG/WEBP photo of a waste item and click **Classify Waste**.

### 3. Reproduce training (optional)

The dataset is not stored in this repo. Download `dataset-resized.zip` from the [TrashNet repository](https://github.com/garythung/trashnet/tree/master/data) and unzip it into the project root so you have `dataset-resized/<class>/*.jpg`.

```bash
python -m src.data_preparation                    # validate data, create splits, plots
python -m src.model                               # smoke-test all architectures
python -m src.train                               # Model A + B with the original recipe (v1)
python -m src.train --models model_c --recipe v3  # Model C with the stable recipe
python -m src.evaluate model_b                    # final test evaluation of a chosen run
```

Runs are saved as `<model>` for recipe v1 and `<model>_<recipe>` otherwise (e.g. `models/model_c_v2_best.keras`). Every run is added to [`results/reports/experiment_comparison.csv`](results/reports/experiment_comparison.csv).

### 4. Train on a free GPU (Google Colab)

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab ([direct link](https://colab.research.google.com/github/Utsav-tala/waste-classification-cnn/blob/main/notebooks/train_colab.ipynb)), select a **T4 GPU** runtime and run the cells top to bottom. The notebook clones this repo, installs the same TensorFlow/Keras versions, downloads TrashNet and checks every image's hash against the split files, trains, and saves the results to Google Drive.

---

## Limitations

- Trained on studio-style photos (one object on a plain background); accuracy on cluttered real-world photos will be lower.
- Only 6 fixed categories; the model cannot say "unknown".
- Softmax confidence is not a calibrated probability.
- Small dataset (1,768 training images) and low input resolution (128×128).

---

## Roadmap

- [x] **Model C — residual from-scratch CNN** (architecture, recipes v2/v3, Colab GPU notebook): 88.1% val accuracy
- [ ] Ablation with recipe v3 (`model_b`, `model_c_plain`) to separate the effect of the recipe, depth and skip connections
- [ ] Final test-set evaluation of the chosen model, and switch the Streamlit app to it
- [ ] **Transfer learning** (MobileNetV2, EfficientNetB0) as a comparison baseline. See [`docs/transfer_learning_plan.md`](docs/transfer_learning_plan.md)
- [ ] Evaluate all models on an unseen real-world dataset to measure generalisation
- [ ] Model selector in the Streamlit app

---

## Acknowledgements

- Dataset: [TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang.
- Developed as an Independent Project (IP) on Neural Networks under the guidance of Prof. Pushpendra Kumar.

## License

[MIT](LICENSE)
