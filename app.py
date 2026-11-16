"""
DERMA.AI — Streamlit Application Entrypoint
Clinical skin lesion classification and decision-support system on the HAM10000 dataset.

Run:
    streamlit run app.py
"""

import os
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from ui.styles import inject_css
from ui import pages as ui_pages

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', BASE_DIR)

LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'evaluation_results.json')
FALLBACK_METRICS_PATH = os.path.join(MODEL_DIR, 'all_metrics.json')

class PositionalEmbedding(tf.keras.layers.Layer):
    """Learnable 1D positional embedding for patch tokens in Hybrid CNN-ViT."""
    def __init__(self, num_tokens=49, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, num_tokens, d_model),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens, "d_model": self.d_model})
        return config


try:
    from train_model2_crossattention import MultiScaleCrossAttention
except ImportError:
    MultiScaleCrossAttention = None


MODEL_CANDIDATES = {
    'Model 2 (EfficientNetV2B3 + Cross-Attention)': [
        os.path.join(MODEL_DIR, 'model2_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'model2_final.keras'),
    ],
    'EfficientNetV2B3 (Fused-MBConv)': [
        os.path.join(MODEL_DIR, 'efficientv2b3_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'efficientv2b3_final.keras'),
    ],
    'Hybrid (DenseNet121 + ViT Transformer)': [
        os.path.join(MODEL_DIR, 'hybrid_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'hybrid_final.keras'),
    ],
    'MobileNetV2 (Enhanced)': [
        os.path.join(MODEL_DIR, 'mobilenet_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'mobilenet_final.keras'),
    ],
}


st.set_page_config(
    page_title='DERMA.AI — Intelligent Skin Analysis',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='collapsed',
)


@st.cache_resource(show_spinner='Initializing deep learning models...')
def load_models() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Loads trained Keras models and label map dictionary."""
    label_path = LABEL_MAP_PATH
    if not os.path.exists(label_path):
        alt_label_path = os.path.join(MODEL_DIR, 'models', 'label_map.json')
        if os.path.exists(alt_label_path):
            label_path = alt_label_path
        else:
            return None, None, 'label_map.json not found in model directory.'

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    except Exception as e:
        return None, None, f'Failed to parse label_map.json: {str(e)}'

    models = {}
    for name, candidate_paths in MODEL_CANDIDATES.items():
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    # compile=False allows inference without custom loss reconstruction
                    custom_objs = {'PositionalEmbedding': PositionalEmbedding}
                    if MultiScaleCrossAttention is not None:
                        custom_objs['MultiScaleCrossAttention'] = MultiScaleCrossAttention
                    models[name] = tf.keras.models.load_model(
                        path,
                        compile=False,
                        custom_objects=custom_objs
                    )
                    break
                except Exception as e:
                    st.warning(f"Failed to load candidate {name} from {path}: {str(e)}")

    if not models:
        return None, label_map, f'No valid .keras model weights discovered in {MODEL_DIR}.'

    return models, label_map, None


@st.cache_data
def load_metrics() -> Optional[Dict[str, Any]]:
    """Loads rigorous evaluation benchmarks from disk."""
    for path in [METRICS_PATH, FALLBACK_METRICS_PATH]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def preprocess(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Preprocesses a PIL Image into a normalized [0, 1] float32 tensor of shape (1, H, W, 3)."""
    rgb_img = image.convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.array(rgb_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# Canonical training set class priors for HAM10000:
# [akiec: 3.3%, bcc: 5.1%, bkl: 11.0%, df: 1.2%, mel: 11.1%, nv: 67.0%, vasc: 1.4%]
CLASS_PRIORS = np.array([0.0326, 0.0513, 0.1097, 0.0115, 0.1111, 0.6695, 0.0142], dtype=np.float32)


def predict(models: Dict[str, Any], arr: np.ndarray, use_tta: bool = True, alpha: float = 0.22) -> np.ndarray:
    """
    Executes model inference across loaded networks.
    - Adapts spatial resolution dynamically (e.g. 300x300 for EfficientNetV2B3 vs 224x224 for Hybrid/MobileNet).
    - Applies multi-angle Test-Time Augmentation (TTA: original, horizontal flip, vertical flip).
    - Applies Bayesian prior calibration (alpha=0.22) to counterbalance the massive 67% NV
      prior, preventing malignant Melanoma (MEL) from being suppressed into benign Nevi (NV).
    """
    all_probs = []
    for model in models.values():
        target_arr = arr
        if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) == 4:
            req_h, req_w = model.input_shape[1], model.input_shape[2]
            if req_h is not None and req_w is not None and (arr.shape[1] != req_h or arr.shape[2] != req_w):
                target_arr = tf.image.resize(arr, [req_h, req_w]).numpy()

        p = model.predict(target_arr, verbose=0)[0]
        if use_tta:
            # 3-way TTA: standard + horizontal flip + vertical flip
            p_h = model.predict(target_arr[:, :, ::-1, :], verbose=0)[0]
            p_v = model.predict(target_arr[:, ::-1, :, :], verbose=0)[0]
            p = (p + p_h + p_v) / 3.0
        all_probs.append(p)
    raw = np.mean(all_probs, axis=0)

    if alpha > 0.0:
        # Prior logit adjustment: p_adj = raw / (priors^alpha), normalized to 1.0
        adj = raw / (CLASS_PRIORS ** alpha)
        return adj / np.sum(adj)
    return raw


def _init_state():
    """Initializes session defaults."""
    defaults = {
        'page': 'home',
        'theme': 'light',
        'use_tta': True,
        'show_gradcam': True,
        'confidence_floor': 0.50,
        'history': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    _init_state()
    theme = st.session_state.get('theme', 'light')
    st.markdown(inject_css(theme), unsafe_allow_html=True)

    models, label_map, error = load_models()
    metrics = load_metrics()

    ui_pages.render_top_nav(
        active=st.session_state.page,
        model_names=list(models.keys()) if models else [],
        metrics=metrics,
    )

    if error:
        st.markdown(f"""
<div class="card" style="margin-top: 2rem;">
  <div class="card-header">Model System Error</div>
  <div class="workflow-desc" style="color: var(--concern);">{error}</div>
</div>
""", unsafe_allow_html=True)
        st.stop()

    class_names = label_map['class_names']
    img_size = label_map.get('img_size', 224)

    def run_prediction(image: Image.Image) -> Dict[str, Any]:
        arr = preprocess(image, img_size)
        probs = predict(models, arr, use_tta=st.session_state.get('use_tta', True), alpha=0.32)
        top_idx = int(np.argmax(probs))

        # Clinical Safety Rule:
        # If the statistical argmax is a benign category ('nv' or 'bkl'), but the probability
        # of malignant Melanoma ('mel') is >= 18% or within 24% of the top prediction with mel >= 14%,
        # clinical safety mandates prioritizing Melanoma as the primary referral finding to eliminate false reassurance.
        mel_idx = class_names.index('mel') if 'mel' in class_names else -1
        clinical_override = False
        if mel_idx >= 0 and mel_idx != top_idx:
            mel_prob = float(probs[mel_idx])
            top_prob = float(probs[top_idx])
            if class_names[top_idx] in ['nv', 'bkl']:
                if mel_prob >= 0.18 or (mel_prob >= 0.14 and (top_prob - mel_prob) <= 0.24):
                    top_idx = mel_idx
                    clinical_override = True

        return {
            'probs': probs.tolist() if hasattr(probs, 'tolist') else list(map(float, probs)),
            'top_idx': top_idx,
            'top_code': class_names[top_idx],
            'top_conf': float(probs[top_idx]),
            'clinical_override': clinical_override,
        }

    page = st.session_state.page
    if page == 'home':
        ui_pages.render_home()
    elif page == 'analyze':
        ui_pages.render_analyze(models, label_map, run_prediction)
    elif page == 'results':
        ui_pages.render_results(label_map)
    elif page == 'performance':
        ui_pages.render_performance(metrics, label_map)
    elif page == 'history':
        ui_pages.render_history(label_map)
    elif page == 'about':
        ui_pages.render_about(models, label_map, metrics)
    else:
        ui_pages.render_home()


if __name__ == '__main__':
    main()
