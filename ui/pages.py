"""DERMA.AI page views — presentation wired to existing ML outputs."""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from .components import (
    HIGH_RISK_CLASSES,
    PLOTLY_LAYOUT,
    brand_block,
    class_pills,
    disclaimer_box,
    metric_tiles,
    page_header,
    probability_bars,
    risk_badge,
)
from .styles import COLORS


def render_sidebar(active: str, model_names, metrics):
    st.markdown(brand_block(), unsafe_allow_html=True)
    st.markdown('<div class="derma-nav-label">Navigate</div>', unsafe_allow_html=True)

    pages = [
        ('Home', 'home'),
        ('Analysis', 'analyze'),
        ('Results', 'results'),
        ('Model Performance', 'performance'),
        ('History', 'history'),
        ('About', 'about'),
    ]
    for label, key in pages:
        if st.button(label, key=f'nav_{key}', use_container_width=True,
                     type='primary' if active == key else 'secondary'):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="derma-nav-label">Session</div>', unsafe_allow_html=True)
    st.caption(f"Models loaded: {', '.join(model_names) if model_names else '—'}")
    if metrics:
        tag = next(iter(metrics))
        m = metrics[tag]
        st.caption(f"Test accuracy: {m['accuracy']*100:.1f}%")
    st.caption(f"History: {len(st.session_state.get('history', []))} analyses")

    st.markdown('<div class="derma-nav-label">Inference</div>', unsafe_allow_html=True)
    # Use widget keys so Streamlit owns session_state values (avoids value/state conflicts).
    st.checkbox(
        'Test-time augmentation',
        key='use_tta',
        help='Averages the prediction with a horizontally flipped pass.',
    )
    st.slider(
        'Uncertainty threshold',
        0.0, 1.0, 0.5, 0.05,
        key='confidence_floor',
        help='Flag predictions below this model confidence as uncertain.',
    )


def render_home():
    st.markdown("""
<div class="hero">
  <div>
    <div class="hero-kicker">DERMA.AI</div>
    <h1 class="hero-title">Intelligent Skin Analysis.<br/>A New Perspective on Dermatology.</h1>
    <p class="hero-lede">
      A refined clinical workspace for dermoscopic image interpretation.
      Upload a lesion image and receive structured, model-driven class probabilities —
      designed for research and educational use.
    </p>
  </div>
  <div class="hero-visual">
    <div class="hero-scan"></div>
    <div class="hero-visual-inner">
      <div class="hero-visual-label">Image analysis layer</div>
      <div class="hero-visual-title">Seven-class dermoscopic intelligence on HAM10000</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button('Start Analysis', type='primary', use_container_width=True):
            st.session_state.page = 'analyze'
            st.rerun()
    with c2:
        if st.button('Model Performance', use_container_width=True):
            st.session_state.page = 'performance'
            st.rerun()

    st.markdown("""
<div class="step-grid">
  <div class="step-card">
    <div class="step-num">01</div>
    <div class="step-title">Upload</div>
    <div class="muted">Provide a clear dermoscopic JPG or PNG of the lesion of interest.</div>
  </div>
  <div class="step-card">
    <div class="step-num">02</div>
    <div class="step-title">Analyze</div>
    <div class="muted">The trained network scores seven lesion categories with calibrated probabilities.</div>
  </div>
  <div class="step-card">
    <div class="step-num">03</div>
    <div class="step-title">Review</div>
    <div class="muted">Inspect the leading class, confidence, and full probability distribution.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(disclaimer_box(
        '<strong>Medical disclaimer.</strong> DERMA.AI is a research and educational tool, '
        'not a medical device. It is not FDA-cleared and must not replace evaluation by a '
        'qualified dermatologist. Changing, bleeding, or concerning lesions require clinical care.'
    ), unsafe_allow_html=True)


def render_analyze(models, label_map, run_prediction):
    st.markdown(page_header(
        'Workspace',
        'Image Analysis',
        'Upload a dermoscopic image, review the preview, then run the trained classifier.'
    ), unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap='large')

    with left:
        st.markdown('<div class="card-title">Image intake</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            'Dermoscopic image',
            type=['jpg', 'jpeg', 'png'],
            label_visibility='collapsed',
            key='uploader',
        )

        if uploaded is not None:
            raw = uploaded.getvalue()
            st.session_state.uploaded_bytes = raw
            st.session_state.uploaded_name = uploaded.name
            st.session_state.uploaded_type = uploaded.type or 'image'
            st.session_state.uploaded_size = len(raw)

        if st.session_state.get('uploaded_bytes'):
            from io import BytesIO
            from PIL import Image
            image = Image.open(BytesIO(st.session_state.uploaded_bytes)).convert('RGB')

            name = st.session_state.get('uploaded_name', 'image')
            size_kb = st.session_state.get('uploaded_size', 0) / 1024
            w, h = image.size
            st.markdown(f"""
<div class="file-meta">
  <div><strong>File</strong> {name}</div>
  <div><strong>Size</strong> {size_kb:.1f} KB</div>
  <div><strong>Dimensions</strong> {w} × {h}</div>
  <div><strong>Mode</strong> {image.mode}</div>
</div>
""", unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                analyze = st.button('Analyze Image', type='primary', use_container_width=True)
            with b2:
                if st.button('Clear Image', use_container_width=True):
                    for k in ['uploaded_bytes', 'uploaded_name', 'uploaded_type', 'uploaded_size',
                              'preview_image', 'last_result', 'uploader']:
                        st.session_state.pop(k, None)
                    st.rerun()

            if analyze:
                with st.spinner('Running inference…'):
                    result = run_prediction(image)
                st.session_state.last_result = result
                history = st.session_state.setdefault('history', [])
                history.insert(0, {
                    **result,
                    'filename': st.session_state.get('uploaded_name', 'image'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
                st.session_state.history = history[:50]
                st.session_state.page = 'results'
                st.rerun()
        else:
            st.markdown(
                '<div class="muted" style="margin-top:0.75rem">Accepted formats: JPG, JPEG, PNG. '
                'Prefer well-framed dermoscopic captures.</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="card-title">Preview</div>', unsafe_allow_html=True)
        if st.session_state.get('uploaded_bytes'):
            from io import BytesIO
            from PIL import Image
            image = Image.open(BytesIO(st.session_state.uploaded_bytes))
            st.image(image, use_column_width=True)
        else:
            st.markdown(
                '<div class="card card-tight"><div class="muted">Your uploaded image will appear here '
                'with clean framing before analysis.</div></div>',
                unsafe_allow_html=True,
            )

        names = ', '.join(models.keys())
        img_size = label_map.get('img_size', 224)
        st.markdown(f"""
<div class="card card-tight" style="margin-top:1rem">
  <div class="card-title">Model context</div>
  <div class="muted">
    Active: <strong>{names}</strong><br/>
    Input resolution: <strong>{img_size}×{img_size}</strong><br/>
    Classes: <strong>{len(label_map['class_names'])}</strong> HAM10000 categories
  </div>
</div>
""", unsafe_allow_html=True)


def render_results(label_map):
    st.markdown(page_header(
        'Report',
        'Analysis Results',
        'Structured output from the trained classifier. Model confidence is not diagnostic certainty.'
    ), unsafe_allow_html=True)

    result = st.session_state.get('last_result')
    if not result:
        st.markdown(
            '<div class="card muted">No analysis yet. Open the Analysis workspace to upload an image '
            'and run inference.</div>',
            unsafe_allow_html=True,
        )
        if st.button('Go to Analysis', type='primary'):
            st.session_state.page = 'analyze'
            st.rerun()
        return

    class_names = label_map['class_names']
    full_names = label_map['full_names']
    top_code = result['top_code']
    top_conf = result['top_conf']
    probs = result['probs']
    is_high_risk = top_code in HIGH_RISK_CLASSES
    floor = float(st.session_state.get('confidence_floor', 0.5))

    st.markdown('<div class="result-hero">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.15], gap='large')
    with c1:
        st.markdown('<div class="card"><div class="card-title">Submitted image</div>', unsafe_allow_html=True)
        if st.session_state.get('uploaded_bytes'):
            from io import BytesIO
            from PIL import Image
            st.image(Image.open(BytesIO(st.session_state.uploaded_bytes)), use_column_width=True)
        st.markdown(
            f'<div class="file-meta"><div><strong>Source</strong> '
            f'{result.get("filename", st.session_state.get("uploaded_name", "image"))}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="card">
  <div class="card-title">Primary assessment</div>
  {risk_badge(top_code)}
  <div class="pred-class">{full_names.get(top_code, top_code)}</div>
  <div class="pred-code">Class code · {top_code}</div>
  <div class="pred-conf">{top_conf*100:.1f}%</div>
  <div class="muted">Model confidence for the leading class — not a clinical diagnosis.</div>
</div>
""", unsafe_allow_html=True)

        if top_conf < floor:
            st.info(
                f'Model confidence ({top_conf*100:.1f}%) is below your '
                f'{floor*100:.0f}% uncertainty threshold. Treat this result cautiously.'
            )
        if is_high_risk:
            st.warning(
                'This category is associated with higher clinical concern and warrants '
                'prompt evaluation by a dermatologist, independent of model confidence.'
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        probability_bars(class_names, full_names, probs),
        unsafe_allow_html=True,
    )

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top3 = ', '.join(
        f'{full_names.get(class_names[i], class_names[i])} ({probs[i]*100:.1f}%)'
        for i in order
    )
    st.markdown(
        f'<div class="muted" style="margin-top:0.9rem">Leading three classes: {top3}</div>',
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)
    with a1:
        if st.button('Analyze another image', use_container_width=True):
            st.session_state.page = 'analyze'
            st.rerun()
    with a2:
        if st.button('View model performance', use_container_width=True):
            st.session_state.page = 'performance'
            st.rerun()


def render_performance(metrics, label_map):
    st.markdown(page_header(
        'Evaluation',
        'Model Performance',
        'Held-out test metrics produced by the existing training pipeline. Values are not altered.'
    ), unsafe_allow_html=True)

    if not metrics:
        st.markdown(
            '<div class="card muted">No all_metrics.json found beside the app. '
            'Train the model to generate evaluation artifacts.</div>',
            unsafe_allow_html=True,
        )
        return

    for tag, m in metrics.items():
        st.markdown(
            f'<div class="card-title" style="margin-bottom:0.75rem">'
            f'Architecture · {tag}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(metric_tiles(m), unsafe_allow_html=True)

        fig = go.Figure()
        labels = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 'F1 (weighted)', 'ROC-AUC']
        values = [
            m['accuracy'], m['precision_macro'], m['recall_macro'],
            m['f1_macro'], m['f1_weighted'], m['roc_auc_macro'],
        ]
        fig.add_trace(go.Bar(
            x=labels,
            y=[v * 100 for v in values],
            marker_color=COLORS['accent'],
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            yaxis=dict(title='Score (%)', gridcolor=COLORS['border'], range=[0, 100]),
            xaxis=dict(tickangle=-18),
            showlegend=False,
            title=dict(text='Test-set metric profile', font=dict(size=14, color=COLORS['primary'])),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="card card-tight">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Detailed scores</div>', unsafe_allow_html=True)
        detail = {
            'Metric': labels,
            'Value': [f'{v*100:.2f}%' for v in values],
        }
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Classification taxonomy</div>', unsafe_allow_html=True)
    st.markdown(
        class_pills(label_map['class_names'], label_map['full_names']),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted" style="margin-top:0.75rem">Per-class precision/recall tables and confusion '
        'matrices are emitted during training logs when available; this view shows the persisted '
        'summary metrics shipped with the deployable model.</div></div>',
        unsafe_allow_html=True,
    )


def render_history(label_map):
    st.markdown(page_header(
        'Session',
        'Analysis History',
        'In-memory record of predictions from this browser session. Nothing is written to disk.'
    ), unsafe_allow_html=True)

    history = st.session_state.get('history', [])
    if not history:
        st.markdown(
            '<div class="card muted">No analyses in this session yet.</div>',
            unsafe_allow_html=True,
        )
        return

    full_names = label_map['full_names']
    query = st.text_input('Filter by filename or class', '')
    rows = []
    for item in history:
        name = item.get('filename', '')
        code = item.get('top_code', '')
        label = full_names.get(code, code)
        blob = f'{name} {code} {label}'.lower()
        if query and query.lower() not in blob:
            continue
        rows.append({
            'Time': item.get('timestamp', ''),
            'File': name,
            'Predicted class': label,
            'Code': code,
            'Confidence': f"{item.get('top_conf', 0)*100:.1f}%",
            'Concern': 'Higher' if code in HIGH_RISK_CLASSES else 'Lower',
        })

    if not rows:
        st.markdown('<div class="muted">No history rows match that filter.</div>', unsafe_allow_html=True)
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)

    options = [f"{i+1}. {r['Time']} — {r['File']} — {r['Predicted class']}" for i, r in enumerate(rows)]
    choice = st.selectbox('Inspect entry', options)
    idx = options.index(choice)
    # Map filtered row back carefully: rebuild filtered history list
    filtered = []
    for item in history:
        name = item.get('filename', '')
        code = item.get('top_code', '')
        label = full_names.get(code, code)
        blob = f'{name} {code} {label}'.lower()
        if query and query.lower() not in blob:
            continue
        filtered.append(item)
    item = filtered[idx]
    st.markdown(f"""
<div class="card">
  <div class="card-title">Entry detail</div>
  {risk_badge(item['top_code'])}
  <div class="pred-class">{full_names.get(item['top_code'], item['top_code'])}</div>
  <div class="pred-code">{item.get('filename')} · {item.get('timestamp')}</div>
  <div class="pred-conf">{item['top_conf']*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)
    st.markdown(
        probability_bars(label_map['class_names'], full_names, item['probs']),
        unsafe_allow_html=True,
    )

    if st.button('Clear session history'):
        st.session_state.history = []
        st.rerun()


def render_about(models, label_map, metrics):
    st.markdown(page_header(
        'Product',
        'About DERMA.AI',
        'Transparent information about the deployed model and its limits.'
    ), unsafe_allow_html=True)

    arch = ', '.join(models.keys()) if models else 'Unavailable'
    metric_line = 'Not available'
    if metrics:
        tag, m = next(iter(metrics.items()))
        metric_line = (
            f"{tag}: accuracy {m['accuracy']*100:.1f}%, "
            f"macro F1 {m['f1_macro']*100:.1f}%, ROC-AUC {m['roc_auc_macro']*100:.1f}%"
        )

    st.markdown(f"""
<div class="about-grid">
  <div class="card">
    <div class="card-title">Model</div>
    <div class="muted">
      Architecture in use: <strong>{arch}</strong><br/>
      Training source: HAM10000 dermoscopic dataset<br/>
      Input size: <strong>{label_map.get('img_size', 224)}×{label_map.get('img_size', 224)}</strong><br/>
      Export format: Keras <code>.keras</code>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Evaluation snapshot</div>
    <div class="muted">{metric_line}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Seven lesion categories</div>', unsafe_allow_html=True)
    st.markdown(
        class_pills(label_map['class_names'], label_map['full_names']),
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card" style="margin-top:1rem">
  <div class="card-title">Limitations</div>
  <div class="muted">
    Predictions depend on image quality, framing, and domain shift relative to HAM10000.
    Model confidence is a softmax score, not diagnostic certainty.
    Rare presentations and under-represented classes may be less reliable.
    This product is not a medical device and is intended for research and education only.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(disclaimer_box(
        '<strong>Responsible use.</strong> Always seek professional medical advice for '
        'patient care decisions. DERMA.AI does not store uploaded images beyond the active session.'
    ), unsafe_allow_html=True)
