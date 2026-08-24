"""
app.py — Deployable Streamlit web app for the HAM10000 skin lesion classifier.

Run locally:
    streamlit run app.py

Deploy for free (pick one):
    - Streamlit Community Cloud: push this repo to GitHub, point Streamlit Cloud at app.py
    - Hugging Face Spaces: create a Space with SDK "Streamlit", push these files
    - Docker: docker build -t skin-classifier . && docker run -p 8501:8501 skin-classifier

Expects, in the same folder (or MODEL_DIR env var):
    - convnext_final.keras and/or hybrid_final.keras  (from train.py)
    - label_map.json
If both models are present, predictions are ensembled (averaged) automatically.
"""

import os
import json

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.dirname(os.path.abspath(__file__)))
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')

MODEL_CANDIDATES = {
    'ConvNeXt-Tiny': os.path.join(MODEL_DIR, 'convnext_final.keras'),
    'Hybrid (ConvNeXt + Transformer)': os.path.join(MODEL_DIR, 'hybrid_final.keras'),
}

st.set_page_config(page_title='Skin Lesion Classifier', page_icon='🩺', layout='centered')


@st.cache_resource(show_spinner='Loading model(s)...')
def load_models():
    if not os.path.exists(LABEL_MAP_PATH):
        return None, None, 'Missing label_map.json — run train.py first and copy its output here.'

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    models = {}
    for name, path in MODEL_CANDIDATES.items():
        if os.path.exists(path):
            # compile=False: we only need inference, and the custom focal loss
            # doesn't need to be reconstructed just to call .predict().
            models[name] = tf.keras.models.load_model(path, compile=False)

    if not models:
        return None, label_map, (
            'No trained model files found in ' + MODEL_DIR +
            '. Expected convnext_final.keras and/or hybrid_final.keras from train.py.'
        )
    return models, label_map, None


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


def main():
    st.title('🩺 Skin Lesion Classifier')
    st.caption('Deep-learning demo trained on the HAM10000 dermoscopy dataset (7-class).')

    st.warning(
        '**Not a medical device.** This is a research/educational demo only. It has not '
        'been validated for clinical use, is not FDA-cleared, and must never replace '
        'evaluation by a qualified dermatologist. If a lesion is changing, bleeding, or '
        'concerning, please see a doctor.',
        icon='⚠️'
    )

    models, label_map, error = load_models()
    if error:
        st.error(error)
        st.stop()

    class_names = label_map['class_names']
    full_names = label_map['full_names']
    img_size = label_map.get('img_size', 224)

    with st.sidebar:
        st.subheader('Settings')
        use_tta = st.checkbox('Test-time augmentation (flip-averaged)', value=True)
        st.write(f"Loaded model(s): {', '.join(models.keys())}")
        st.caption('Multiple models loaded → predictions are ensembled automatically.')

    uploaded = st.file_uploader('Upload a dermoscopic image', type=['jpg', 'jpeg', 'png'])

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, caption='Uploaded image', use_container_width=True)

        with st.spinner('Analyzing...'):
            arr = preprocess(image, img_size)
            probs = predict(models, arr, use_tta=use_tta)

        top_idx = int(np.argmax(probs))
        order = np.argsort(probs)[::-1]

        with col2:
            st.subheader('Prediction')
            st.metric(
                label=full_names.get(class_names[top_idx], class_names[top_idx]),
                value=f'{probs[top_idx]*100:.1f}% confidence'
            )
            st.caption(f'Class code: `{class_names[top_idx]}`')

        st.subheader('Full probability breakdown')
        for i in order:
            cls = class_names[i]
            label = f'{full_names.get(cls, cls)} ({cls})'
            st.write(label)
            st.progress(float(probs[i]))
            st.caption(f'{probs[i]*100:.2f}%')

        st.divider()
        st.caption(
            'Top-3 most likely: ' +
            ', '.join(f'{full_names.get(class_names[i], class_names[i])} ({probs[i]*100:.1f}%)'
                      for i in order[:3])
        )
    else:
        st.info('Upload a JPG or PNG dermoscopic image to get a prediction.')


if __name__ == '__main__':
    main()
