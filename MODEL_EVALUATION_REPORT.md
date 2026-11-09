# DERMA.AI — MACHINE LEARNING EVALUATION & DIAGNOSTIC REPORT

**Date:** 2026-11-09  
**Model Architecture:** MobileNetV2 with Focal Loss (\(\gamma=2.0\)) & Label Smoothing (\(\alpha=0.05\))  
**Target Dataset:** HAM10000 (ViDIR Group, Medical University of Vienna)  
**Evaluation Type:** Independent Held-Out Test Set (Group-Stratified by `lesion_id`, 0 Patient Leakage)  
**Test Set Size:** 2,004 Images across 1,494 Unique Lesions  

---

## 1. Executive Summary

A critical objective of this overhaul was to investigate reports of incorrect classifications—specifically **Benign Keratosis (BKL) being classified as Melanocytic Nevi (NV)**—and to establish an un-fabricated, leak-free evaluation benchmark.

The investigation revealed that the legacy evaluation metrics (which claimed 76.3% accuracy and 71.5% macro F1) suffered from **data leakage**: the legacy training pipeline oversampled minority classes *prior* to splitting, causing identical images to appear in both training and test sets, while also ignoring patient/lesion grouping (`lesion_id`).

Under rigorous group-stratified evaluation on 2,004 truly held-out test images with **zero lesion leakage**:
* **Overall Accuracy:** 74.45%
* **Balanced Accuracy:** 76.01%
* **Macro F1-Score:** 60.27%
* **Weighted F1-Score:** 76.30%
* **Macro ROC-AUC:** 94.43%

---

## 2. Dataset Distribution & Prior Imbalance

The HAM10000 dataset consists of 10,015 dermoscopic images mapped to 7,470 unique lesions. The class distribution exhibits extreme imbalance:

| Class Code | Full Diagnosis Category | Image Count | Percentage | Clinical Tier |
| :---: | :--- | :---: | :---: | :--- |
| **nv** | Melanocytic nevi | 6,705 | 66.95% | Usually Benign |
| **mel** | Melanoma | 1,113 | 11.11% | **Higher Clinical Concern** |
| **bkl** | Benign keratosis-like lesions | 1,099 | 10.97% | Usually Benign |
| **bcc** | Basal cell carcinoma | 514 | 5.13% | **Higher Clinical Concern** |
| **akiec** | Actinic keratoses / Bowen's | 327 | 3.27% | **Higher Clinical Concern** |
| **vasc** | Vascular lesions | 142 | 1.42% | Usually Benign |
| **df** | Dermatofibroma | 115 | 1.15% | Usually Benign |

Melanocytic Nevi (NV) represent more than two-thirds of the entire dataset. In standard cross-entropy training, neural networks learn a heavy prior bias toward NV.

---

## 3. Investigation of the BKL vs NV Classification Issue

### 3.1 Empirical Findings on BKL Samples
On the held-out test set (220 true BKL images):
* **Correctly classified as BKL:** 114 (51.8% recall)
* **Misclassified as Melanoma (MEL):** 32 (14.5%)
* **Misclassified as Actinic Keratoses (AKIEC):** 22 (10.0%)
* **Misclassified as Basal Cell Carcinoma (BCC):** 21 (9.5%)
* **Misclassified as Melanocytic Nevi (NV):** 19 (8.6%)
* **Misclassified as Dermatofibroma (DF):** 12 (5.5%)

### 3.2 Root Causes Identified
1. **Dermoscopic & Morphological Overlap:**
   * Benign Keratoses include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses (LPLK).
   * Early solar lentigines and reticulated seborrheic keratoses exhibit delicate pigmented networks and globule-like aggregations that optically mimic melanocytic nevi.
2. **Resolution Loss During Preprocessing:**
   * Dermoscopic differentiation relies on microscopic pathognomonic criteria: milia-like cysts, comedo-like openings, and cerebriform gyri for BKL; versus pigment network uniformity, peripheral dots, and architectural symmetry for NV.
   * Compressing a 600×450 dermoscopic capture down to 224×224 pixels blurs these microscopic structures, forcing the network to rely on coarse color and shape features where BKL and NV overlap heavily.
3. **Marginal Probability Distribution:**
   * When examining misclassified BKL samples, the model's prediction was not made with high confidence. For example:
     * `ISIC_0027954`: NV = 35.0%, BKL = 32.5% (difference of only 2.5%)
     * `ISIC_0029195`: NV = 42.9%, BKL = 28.3%
   * The model was genuinely uncertain due to visual ambiguity, but because the highest single softmax score was NV (due to the overwhelming dataset prior), the argmax yielded NV.

### 3.3 Engineering & Clinical Safety Remediation
Rather than artificially hardcoding or forcing predictions, DERMA.AI addresses this through **calibrated transparency**:
1. **Uncertainty Floor Warning:** When top confidence is below the threshold (default 50%), the UI warns the clinician: *"Model confidence is below uncertainty threshold. Treat this result cautiously."*
2. **Grad-CAM Feature Attribution:** Generates visual heatmaps showing whether the model attended to the central pigment network, border, or background skin.
3. **Full Probability Distribution:** Displays the full 7-class probability breakdown, showing the clinician that BKL was scored closely behind NV.
4. **Leak-Free Retraining Architecture:** Updated `train.py` with `StratifiedGroupKFold` on `lesion_id` so that future fine-tuning retains 100% test independence.

---

## 4. Comprehensive Evaluation Metrics

### 4.1 Summary Benchmarks (Held-Out Test Set)
* **Overall Accuracy:** 74.45%
* **Balanced Accuracy:** 76.01%
* **Macro Precision:** 53.64%
* **Macro Recall:** 76.01%
* **Macro F1-Score:** 60.27%
* **Weighted F1-Score:** 76.30%
* **Macro ROC-AUC:** 94.43%

### 4.2 Per-Class Breakdown

| Category | Full Diagnosis | Precision | Recall | F1-Score | Support | Clinical Risk Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **AKIEC** | Actinic keratoses | 50.6% | 60.6% | 55.2% | 66 | Higher Concern |
| **BCC** | Basal cell carcinoma | 57.9% | 81.6% | 67.7% | 103 | Higher Concern |
| **BKL** | Benign keratosis | 59.4% | 51.8% | 55.3% | 220 | Usually Benign |
| **DF** | Dermatofibroma | 19.7% | 100.0% | 32.9% | 23 | Usually Benign |
| **MEL** | Melanoma | 47.1% | 58.1% | 52.0% | 222 | Higher Concern |
| **NV** | Melanocytic nevi | 93.6% | 80.0% | 86.3% | 1,341 | Usually Benign |
| **VASC** | Vascular lesions | 56.9% | 100.0% | 72.5% | 29 | Usually Benign |

### 4.3 Full Multi-Class Confusion Matrix (Raw Counts)

```
                Predicted ->
                AKIEC   BCC   BKL    DF   MEL    NV  VASC  | Total
True AKIEC  |      40     8     4    12     2     0     0  |    66
True BCC    |       4    84     3     4     3     3     2  |   103
True BKL    |      22    21   114    12    32    19     0  |   220
True DF     |       0     0     0    23     0     0     0  |    23
True MEL    |       8     6    18     7   129    51     3  |   222
True NV     |       5    26    53    59   108  1073    17  |  1341
True VASC   |       0     0     0     0     0     0    29  |    29
```

### 4.4 Normalized Confusion Matrix (True Class Proportions)

```
                AKIEC    BCC    BKL     DF    MEL     NV   VASC
True AKIEC  |   0.606  0.121  0.061  0.182  0.030  0.000  0.000
True BCC    |   0.039  0.816  0.029  0.039  0.029  0.029  0.019
True BKL    |   0.100  0.095  0.518  0.055  0.145  0.086  0.000
True DF     |   0.000  0.000  0.000  1.000  0.000  0.000  0.000
True MEL    |   0.036  0.027  0.081  0.032  0.581  0.230  0.014
True NV     |   0.004  0.019  0.040  0.044  0.081  0.800  0.013
True VASC   |   0.000  0.000  0.000  0.000  0.000  0.000  1.000
```

---

## 5. Conclusion & Recommendations

The deployed model demonstrates strong discrimination with a **94.43% Macro ROC-AUC** and **76.01% Balanced Accuracy**. The BKL vs NV challenge is governed by dermoscopic feature ambiguity and dataset priors rather than code-level mapping errors. The platform's new explainability features (Grad-CAM, uncertainty thresholds, full probability bars, PDF report export) provide clinicians with the context needed to make informed dermatological decisions.

