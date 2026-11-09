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

MODEL_CANDIDATES = {
    'MobileNetV2': [
        os.path.join(MODEL_DIR, 'mobilenet_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'mobilenet_final.keras')
    ],
    'ConvNeXt-Tiny': [
        os.path.join(MODEL_DIR, 'convnext_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'convnext_final.keras')
    ],
    'Hybrid (ConvNeXt + Transformer)': [
        os.path.join(MODEL_DIR, 'hybrid_final.keras'),
        os.path.join(MODEL_DIR, 'models', 'hybrid_final.keras')
    ]
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
                    models[name] = tf.keras.models.load_model(path, compile=False)
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


def predict(models: Dict[str, Any], arr: np.ndarray, use_tta: bool = True) -> np.ndarray:
    """
    Executes model inference across loaded networks.
    Optionally applies horizontal-flip test-time augmentation (TTA).
    """
    all_probs = []
    for model in models.values():
        p = model.predict(arr, verbose=0)[0]
        if use_tta:
            # Horizontal flip pass
            p_flip = model.predict(arr[:, :, ::-1, :], verbose=0)[0]
            p = (p + p_flip) / 2.0
        all_probs.append(p)
    return np.mean(all_probs, axis=0)


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
        probs = predict(models, arr, use_tta=st.session_state.get('use_tta', True))
        top_idx = int(np.argmax(probs))
        return {
            'probs': probs.tolist() if hasattr(probs, 'tolist') else list(map(float, probs)),
            'top_idx': top_idx,
            'top_code': class_names[top_idx],
            'top_conf': float(probs[top_idx]),
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
