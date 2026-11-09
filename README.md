# DERMA.AI — Intelligent Skin Analysis
> **Clinical Image Intelligence & Decision Support Platform on HAM10000**

DERMA.AI is an AI-powered dermatological lesion classification system. Engineered with a **Premium Minimalist Clinical Technology** design system, the platform classifies dermoscopic imagery across seven diagnostic categories on the HAM10000 taxonomy with calibrated probability distributions, Grad-CAM attention heatmaps, and downloadable clinical PDF reports.

---

## 🌟 Key Features

* **Clinical Decision Support:** Multi-class predictive scoring across 7 HAM10000 categories (`AKIEC`, `BCC`, `BKL`, `DF`, `MEL`, `NV`, `VASC`).
* **Visual Explainability (Grad-CAM):** Dynamic Class Activation Maps generated from the final convolutional layer (`Conv_1`) to highlight the visual features influencing model predictions.
* **Calibrated Uncertainty Safety Floor:** Flags ambiguous predictions below a configurable confidence threshold (e.g. 50%) to guard against misdiagnosis in overlapping lesions (such as BKL vs NV).
* **Official Clinical PDF Report Export:** Instant one-click generation of patient/research documentation including image thumbnails, leading diagnoses, 7-class probability tables, and clinical disclaimers.
* **Interactive Evaluation Dashboard:** Transparent empirical benchmarks measured on a **leak-free, group-stratified (lesion-independent)** held-out test set (2,004 images), featuring an interactive Plotly confusion matrix (normalized and raw counts).
* **Cybersecurity Hardening:** Magic byte image validation (JPEG/PNG), strict dimension checks (32px to 8,192px), decompression bomb protection, file size constraints (10 MB), and complete HTML/XSS sanitization.
* **Responsive SaaS Design:** Optimized for Desktop (1920×1080 down to 1366×768), Tablet (1024×768 down to 768×1024), and Mobile (375×812 to 430×932) with dark and light themes.

---

## 🔬 Classification Taxonomy

| Class Code | Full Diagnosis Category | Clinical Significance |
| :---: | :--- | :--- |
| **MEL** | Melanoma | **Higher Concern** — Malignant neoplasm of melanocytes. |
| **BCC** | Basal cell carcinoma | **Higher Concern** — Common malignant non-melanoma skin cancer. |
| **AKIEC** | Actinic keratoses / Intraepithelial carcinoma | **Higher Concern** — Pre-malignant squamous cell carcinoma precursor. |
| **BKL** | Benign keratosis-like lesions | Usually Benign — Seborrheic keratoses, solar lentigines, lichen-planus-like. |
| **NV** | Melanocytic nevi | Usually Benign — Moles and common nevi (66.95% of HAM10000). |
| **DF** | Dermatofibroma | Usually Benign — Benign cutaneous fibrous histiocytoma. |
| **VASC** | Vascular lesions | Usually Benign — Angiomas, pyogenic granulomas, angiokeratomas. |

---

## 📊 Empirical Performance (Held-Out Test Set)

Evaluated on **2,004 images** across **1,494 unique lesions** with **zero lesion leakage**:

* **Overall Test Accuracy:** `74.45%`
* **Balanced Accuracy:** `76.01%`
* **Macro ROC-AUC:** `94.43%`
* **Weighted F1-Score:** `76.30%`
* **Macro F1-Score:** `60.27%`

---

## 🚀 Quickstart & Local Execution

### 1. Environment Setup
Ensure Python 3.10+ is installed:
```bash
# Clone repository and navigate to directory
cd d:\Imaan\Projects\SkinLesion

# Activate virtual environment
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
*(On Windows systems under enterprise execution policies, run via module syntax:)*
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing Suite

The platform includes an automated pytest suite covering 18 unit, integration, and security test cases:
```bash
pytest tests/test_derma_ai.py -v
```
Test categories include:
* Image Input & Integrity (valid JPEG/PNG, RGBA conversion, grayscale, corrupted stream, 0-byte, oversized, format spoofing).
* Model Pipeline (input/output shapes, class mapping parity, float32 scaling, probability summation).
* Explainability (Grad-CAM generation, normalization, alpha blending).
* Report Generation (PDF structure, `%PDF-` signature).
* Security Defenses (XSS sanitization, magic byte rejection).

---

## 📂 Project Architecture

```
SkinLesion/
├── app.py                      # Main Streamlit application entry point
├── train.py                    # Leak-free training script with StratifiedGroupKFold
├── requirements.txt            # Python dependencies
├── label_map.json              # Canonical class-to-index mapping
├── evaluation_results.json     # Rigorous held-out test evaluation benchmarks
├── mobilenet_final.keras       # Deployed MobileNetV2 Keras model
├── QA_REPORT.md                # Quality assurance and functional test report
├── SECURITY_REPORT.md          # Cybersecurity audit and vulnerability matrix
├── MODEL_EVALUATION_REPORT.md  # ML diagnostics & BKL vs NV investigation
├── UI_UX_CHANGELOG.md          # Complete design system changelog
├── ui/
│   ├── components.py           # Presentation widgets & sanitized UI components
│   ├── explainability.py       # Grad-CAM heatmap calculation & overlay
│   ├── pages.py                # Page controllers (Home, Analysis, Results, etc.)
│   ├── report_generator.py     # PDF clinical report generator (ReportLab)
│   ├── security.py             # Magic byte, dimension, and input validation
│   └── styles.py               # Responsive CSS design system (Inter + Manrope)
├── tests/
│   └── test_derma_ai.py        # Comprehensive 18-case pytest suite
└── data/
    └── ham10000/               # HAM10000 dataset & metadata
```

---

## ⚖️ Clinical & Medical Disclaimer

**DERMA.AI is an assistive artificial intelligence platform designed strictly for educational, research, and clinical decision-support experimentation. It is not an FDA-cleared diagnostic device.**  
Artificial intelligence predictions are statistical pattern recognitions and must never supersede clinical assessment, dermoscopy, or biopsy performed by a licensed dermatologist.
