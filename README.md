# DERMA.AI — Intelligent Skin Analysis
> **Clinical AI Decision-Support Platform powered by a Hybrid CNN-Transformer Architecture on HAM10000**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-23%2F23%20Passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20CNN--ViT%20%7C%20EfficientNetV2B3-purple.svg)](#-model-architectures)
[![XAI](https://img.shields.io/badge/Explainability-LIME%20%2B%20Grad--CAM%20%2B%20Feature%20Extraction-teal.svg)](#-explainable-ai-xai--clinical-feature-extraction)

---

## 🌟 Executive Summary

**DERMA.AI** is an advanced artificial intelligence dermatology platform designed for research, education, and clinical decision-support. Built with a **Premium Minimalist Clinical Technology** aesthetic, the platform integrates a state-of-the-art **Hybrid CNN-Transformer architecture**, combining the local texture-extraction strength of convolutional networks with the global contextual reasoning of self-attention Vision Transformers.

The platform classifies dermoscopic imagery across the seven diagnostic categories of the **HAM10000** benchmark, providing calibrated probability distributions, Grad-CAM attention heatmaps, and downloadable clinical PDF reports.

---

## 🔬 Hybrid CNN-Transformer Architecture

Dermoscopic skin lesion evaluation demands two complementary visual scales:
1. **Microscopic Local Cues (Textures):** Differentiating **Benign Keratosis (BKL)** from **Nevus (NV)** depends on fine pathognomonic structures like *comedo-like openings, milia cysts, and delicate pigment network lines* that span only a few pixels.
2. **Macroscopic Global Cues (Asymmetry & Variegation):** Differentiating **Melanoma (MEL)** from **Nevus (NV)** requires evaluating the clinical **ABCDE criteria**—lesion-wide *architectural asymmetry across opposite poles, border irregularity, and color variegation*.

Standard pure CNNs struggle with global long-range relationships, while pure Vision Transformers lack local inductive bias on medical datasets. **DERMA.AI solves this through a unified Hybrid CNN-Transformer topology:**

```
Input Image (224×224×3 RGB)
       │
       ▼
[ImageNet Normalization Layer]
       │
       ▼
[DenseNet121 Convolutional Backbone] ───────────────┐ (Direct Local Pathway)
       │ (Feature Maps: 7×7×1024)                   │
       ▼                                            ▼
[Spatial Patch Tokenizer (49 Tokens × 256 Dim)] [GAP 2D Pooling]
       │                                            │
       ▼                                            ▼
[Learnable 1D Positional Embedding]             [Dense Projection (256)]
       │                                            │
       ▼                                            │
[Multi-Head Self-Attention (4 Heads, Key Dim 64)]    │
       │ (Residual + LayerNormalization)            │
       ▼                                            │
[GELU Feed-Forward MLP (512 Dim)]                   │
       │ (Residual + LayerNormalization)            │
       ▼                                            │
[GAP 1D Token Pooling (256 Dim)]                    │
       │                                            │
       └────────────────────┬───────────────────────┘
                            ▼
              [Hybrid Feature Fusion (512 Dim)]
                            │
                            ▼
              [Dropout (0.35) ➔ Dense (256, GELU)]
                            │
                            ▼
              [Dropout (0.25) ➔ Dense (7, Softmax)]
                            │
                            ▼
                  [Predictions: 7 Classes]
```

### Architectural Highlights
* **DenseNet121 Backbone:** Dense connectivity concatenates feature maps from *all* preceding layers ($x_0, x_1, \dots, x_{l-1}$). Both shallow texture maps (fine border sharpness, reticular networks) and deep semantic maps are preserved into the classification head.
* **Multi-Head Self-Attention Transformer Block:** Projects the $7 \times 7$ grid into $49$ sequence tokens to compute spatial self-attention across the whole lesion, capturing architectural asymmetry across opposite poles.
* **Dual Feature Fusion:** Combines direct global average-pooled CNN representations ($256$) with Transformer attention tokens ($256$) into a rich $512$-dimensional hybrid feature vector.
* **Parameter Efficiency:** The entire hybrid network comprises only **$\approx 8.2\text{M}$ parameters**, running smoothly on modern CPUs and GPUs without memory exhaustion.

---

## ⚡ EfficientNetV2B3 + Multi-Scale Cross-Attention Fusion (Model 2)

**DERMA.AI** incorporates the **Multi-Scale Cross-Attention Fusion** architecture (replicating the Bahria University DL benchmark):

* **Texture-Semantic Cross-Attention:** Fuses early high-resolution texture tokens (`block5g_add`: $19\times19$, 136 dim $\rightarrow 361$ tokens) with deep semantic tokens (`top_activation`: $10\times10$, 1536 dim $\rightarrow 100$ tokens) via bidirectional multi-head cross-attention ($d=128$, 8 heads).
* **Native $300\times300$ Input:** Captures $1.79\times$ more high-frequency dermoscopic spatial detail than standard $224\times224$ networks.
* **Trained Locally (Zero Downloads):** Executed with Class-Balanced Focal Loss ($\gamma=2$, label smoothing $0.1$) and Cosine Warmup learning rate schedule across 40 epochs.

### 📊 Benchmark Comparison on HAM10000 Held-Out Test Set

| Metric | Previous Hybrid (DenseNet + ViT) | **Model 2 (EfficientNetV2B3 + Cross-Attention)** | Improvement |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 74.45% | **83.57%** | **+9.12%** 🚀 |
| **Macro ROC-AUC** | 0.9443 | **0.9632** | **+0.0189** |
| **Melanoma (MEL) Recall** | 58.10% | **66.47%** | **+8.37%** |
| **Melanoma F1-Score** | 0.5200 | **0.6133** | **+0.0933** |
| **Benign Keratosis (BKL) Recall** | 51.80% | **78.18%** | **+26.38%** (Resolves BKL $\rightarrow$ NV confusion) |
| **Melanocytic Nevi (NV) F1** | 0.8630 | **0.9273** | **+0.0643** |

---

## 🔍 Explainable AI (XAI) & Clinical Feature Extraction

DERMA.AI provides a transparent, dual-tier explainability engine combining visual attribution maps with **natural language clinical feature extraction**:

```
Input Image ──► Model Inference ──► Class Probabilities
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Grad-CAM Attention Heatmap]    [LIME Superpixel Decomposition]
(Continuous thermal gradient)   (Histological zone perturbation)
       │                               │
       └───────────────┬───────────────┘
                       ▼
    [Algorithmic Clinical Feature Extractor]
                       │
    ┌──────────────────┴──────────────────┐
    ▼                                     ▼
[Spatial Localization Analysis]   [Extracted Visual Biomarkers]
(e.g., peripheral margin vs core) (Variegation, network, border notch)
    │                                     │
    └──────────────────┬──────────────────┘
                       ▼
        [Synthesized Clinical Rationale]
        (Written words explaining why this
         diagnosis was chosen over alternatives)
```

### 1. Dual Visual Attributions
* **Grad-CAM:** Computes gradients flowing into the final convolutional layer (`conv5_block16_2_conv` / `top_conv`) to generate thermal overlays pinpointing regional focus.
* **LIME (Local Interpretable Model-agnostic Explanations):** Decomposes the lesion into histological superpixel segments via Quickshift. Highlights superpixels providing positive supporting evidence (soft emerald green) and negative contradictory evidence (coral red) with distinct gold histological boundaries.

### 2. Written Clinical Feature Extraction (Natural Language)
In addition to heatmaps, the platform generates a comprehensive written clinical explanation:
* **Primary Feature Salience Zone:** Identifies whether the neural network focused on the central lesion body, paracentral transition zone, or asymmetric peripheral margin.
* **Extracted Visual Biomarkers:** Concretely lists extracted dermoscopic patterns (e.g. *irregular peripheral border cutoff, multi-chromophore variegation index, atypical thickened reticular netting*).
* **Differential Diagnostic Rationale:** Explains in plain clinical language why the chosen category won over the closest alternative (e.g., resolving the common ambiguity between **Melanoma** and **Melanocytic Nevi**, or **Benign Keratosis** and **Nevi**).

---

## 🛡️ Clinical Prior Calibration & Safety Stratification

### The 67% Dataset Prior Problem
In the HAM10000 dataset, Melanocytic Nevi (`nv`) account for **6,705 images (67.0%)**, while Melanoma (`mel`) accounts for **1,113 images (11.1%)**. Under naive `argmax`, uncalibrated neural networks frequently default to `nv` when evaluating ambiguous melanocytic lesions, creating a dangerous false-negative risk.

### DERMA.AI's Two-Tier Solution
1. **Bayesian Prior Logit Calibration ($\alpha = 0.32$):**
   The inference pipeline counterbalances the majority prior via logit adjustment:
   $$P_{\text{calibrated}}(c \mid x) \propto \frac{P_{\text{raw}}(c \mid x)}{\pi_c^\alpha}$$
   This counteracts the massive 6:1 NV dataset imbalance and elevates true Melanoma detection sensitivity.
2. **Clinical Safety Priority Rule:**
   If the statistical argmax suggests a benign mole (`nv` or `bkl`), but the probability of **Melanoma (`mel`) is $\ge 18\%$** (or within $24\%$ of the top category with $P(\text{mel}) \ge 14\%$), the platform **automatically elevates Melanoma as the primary referral finding** to prevent false reassurance.

---

## 🔬 Classification Taxonomy

| Class Code | Diagnosis Category | Clinical Significance | Key Dermoscopic Features |
| :---: | :--- | :---: | :--- |
| **MEL** | Melanoma | **Critical Concern** | Atypical pigment network, asymmetry, blue-white veil, irregular dots/streaks |
| **BCC** | Basal cell carcinoma | **High Concern** | Arborizing telangiectasia, ulceration, blue-gray ovoid nests |
| **AKIEC** | Actinic keratoses / Bowen's | **High Concern** | Erythema, surface scale, strawberry pattern, rosette signs |
| **BKL** | Benign keratosis-like lesions | Usually Benign | Comedo-like openings, milia-like cysts, moth-eaten borders, cerebriform gyri |
| **NV** | Melanocytic nevi | Usually Benign | Uniform pigment network, architectural symmetry, homogeneous globules |
| **DF** | Dermatofibroma | Usually Benign | Central white patch with delicate peripheral pigment network |
| **VASC** | Vascular lesions | Usually Benign | Red-purple lacunae, homogeneous vascular areas |

---

## 🌟 Key Platform Features

* **Dual Intake (Upload & Direct Clipboard Copy-Paste):** Seamless image loading supporting standard file uploads (JPG, PNG) as well as direct clipboard pasting (`Ctrl+C` / `Cmd+C` followed by one-click paste) without saving files to disk first.
* **Dual-Audience XAI (Patient & Clinician):** Plain-English patient breakdown alongside deep dermatological biomarkers (ABCDE criteria, pattern analysis, and differential diagnostic rationale).
* **Visual Explainability (Grad-CAM & LIME):** Dynamic Class Activation Maps and Quickshift superpixel perturbation decompose positive and negative regional attributions with gold histological boundaries.
* **Official Clinical PDF Report Export:** Instant one-click generation of patient/research documentation including image thumbnails, leading diagnoses, 7-class probability distributions, and clinical disclaimers via ReportLab.
* **Interactive Evaluation Dashboard:** Transparent empirical benchmarks measured on a **leak-free, group-stratified (lesion-independent)** held-out test set with interactive Plotly confusion matrices.
* **Calibrated Uncertainty Safety Floor:** Configurable confidence floor (default $50\%$) that warns clinicians when visual ambiguity is high.
* **Enterprise Cybersecurity:**
  * Magic byte verification (JPEG/PNG header inspection).
  * Strict dimension verification ($32\text{px}$ to $8,192\text{px}$).
  * Decompression bomb protection (PIL pixel limits).
  * 10 MB upload constraint.
  * Complete HTML/XSS sanitization on all dynamic inputs.

---

## 🚀 Quickstart & Local Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/SkinLesion.git
cd SkinLesion

# Activate virtual environment
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive 23-case automated test suite covering unit integrity, model inference, explainability (Grad-CAM + LIME + Feature Extraction), security, and report generation:

```bash
pytest tests/test_derma_ai.py -v
```

### Test Coverage Summary:
* `TestImageInput` (9 tests): Valid JPEG/PNG, RGBA conversion, grayscale, corrupted byte streams, 0-byte input, oversized files, format spoofing, tiny dimensions.
* `TestModelPipeline` (5 tests): Model input/output dimensions, class mapping parity, float32 scaling, probability summation to 1.0, prediction determinism.
* `TestExplainability` (4 tests): Grad-CAM generation & blending, LIME superpixel segmentation, algorithmic clinical feature extraction text generation, Melanoma vs. Nevus biomarker distinction.
* `TestReportGeneration` (2 tests): Valid PDF structure (`%PDF-` signature), XML entity escaping for XSS protection.
* `TestSecurity` (3 tests): HTML tag stripping, magic byte validation, path traversal neutralization.

---

## 📂 Project Architecture

```
SkinLesion/
├── app.py                      # Streamlit application entry point & inference controller
├── train_hybrid.py             # SOTA Hybrid CNN-Transformer training pipeline
├── train.py                    # Baseline training script with StratifiedGroupKFold
├── requirements.txt            # Python dependencies
├── label_map.json              # Canonical class-to-index mapping
├── evaluation_results.json     # Empirical held-out test benchmarks
├── hybrid_final.keras          # Trained Hybrid CNN-Transformer model weights
├── mobilenet_final.keras       # Baseline MobileNetV2 model weights
├── MODEL_EVALUATION_REPORT.md  # Detailed ML evaluation report
├── QA_REPORT.md                # Quality assurance and functional verification
├── SECURITY_REPORT.md          # Cybersecurity audit & mitigation matrix
├── UI_UX_CHANGELOG.md          # Complete design system changelog
├── ui/
│   ├── components.py           # Presentation widgets & sanitized UI components
│   ├── explainability.py       # Grad-CAM heatmap calculation & overlay
│   ├── pages.py                # Page controllers (Home, Analysis, Results, History, etc.)
│   ├── report_generator.py     # PDF clinical report generator (ReportLab)
│   ├── security.py             # Magic byte, dimension, and input validation
│   └── styles.py               # Responsive CSS design system (Inter + Manrope)
├── tests/
│   └── test_derma_ai.py        # Comprehensive 20-test pytest suite
└── data/
    └── ham10000/               # HAM10000 dataset & metadata
```

---

## ⚖️ Clinical & Medical Disclaimer

**DERMA.AI is an assistive artificial intelligence platform designed strictly for educational, research, and clinical decision-support experimentation. It is not an FDA-cleared diagnostic device.**  
Artificial intelligence predictions are statistical pattern recognitions and must never supersede in-person clinical assessment, dermoscopy, or histological biopsy conducted by a licensed dermatologist.
