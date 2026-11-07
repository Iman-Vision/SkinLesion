"""
DERMA.AI — Streamlit frontend for the existing HAM10000 skin lesion classifier.

Run:
    streamlit run app.py

ML contract (unchanged):
    - Loads *.keras models + label_map.json from MODEL_DIR
    - preprocess(): RGB resize → [0,1] float32 batch
    - predict(): optional flip TTA, mean across loaded models
"""

import os
import json

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from ui.styles import inject_css
from ui import pages as ui_pages

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.dirname(os.path.abspath(__file__)))
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'all_metrics.json')

MODEL_CANDIDATES = {
    'MobileNetV2': os.path.join(MODEL_DIR, 'mobilenet_final.keras'),
    'ConvNeXt-Tiny': os.path.join(MODEL_DIR, 'convnext_final.keras'),
    'Hybrid (ConvNeXt + Transformer)': os.path.join(MODEL_DIR, 'hybrid_final.keras'),
}


st.set_page_config(
    page_title='DERMA.AI',
    page_icon='◈',
    layout='wide',
    initial_sidebar_state='expanded',
)


@st.cache_resource(show_spinner='Loading model(s)...')
def load_models():
    if not os.path.exists(LABEL_MAP_PATH):
        return None, None, 'Missing label_map.json — run train.py first and copy its output here.'

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    models = {}
    for name, path in MODEL_CANDIDATES.items():
        if os.path.exists(path):
            # compile=False: inference only; custom focal loss need not be reconstructed.
            models[name] = tf.keras.models.load_model(path, compile=False)

    if not models:
        return None, label_map, (
            'No trained model files found in ' + MODEL_DIR +
            '. Expected mobilenet_final.keras (or convnext_final.keras / hybrid_final.keras) from train.py.'
        )
    return models, label_map, None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def preprocess(image: Image.Image, img_size: int) -> np.ndarray:
    image = image.convert('RGB').resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(models, arr, use_tta=True):
    """Average predictions across all loaded models, optionally with a
    horizontal-flip TTA pass, for a more stable estimate."""
    all_probs = []
    for model in models.values():
        p = model.predict(arr, verbose=0)[0]
        if use_tta:
            p_flip = model.predict(arr[:, :, ::-1, :], verbose=0)[0]
            p = (p + p_flip) / 2.0
        all_probs.append(p)
    return np.mean(all_probs, axis=0)


def _init_state():
    defaults = {
        'page': 'home',
        'use_tta': True,
        'confidence_floor': 0.5,
        'history': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    _init_state()
    st.markdown(inject_css(), unsafe_allow_html=True)

    models, label_map, error = load_models()
    metrics = load_metrics()

    with st.sidebar:
        ui_pages.render_sidebar(
            active=st.session_state.page,
            model_names=list(models.keys()) if models else [],
            metrics=metrics,
        )

    if error:
        st.markdown(
            f'<div class="card" style="margin-top:2rem"><div class="card-title">Model unavailable</div>'
            f'<div class="muted">{error}</div></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    class_names = label_map['class_names']
    img_size = label_map.get('img_size', 224)

    def run_prediction(image: Image.Image):
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
