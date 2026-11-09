"""
DERMA.AI Page Views
Orchestrates Home, Analysis, Results, Performance, History, and About pages.
Implements the clinical user journey, Grad-CAM visualization, PDF report generation,
and responsive component layouts.
"""

import uuid
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from .components import (
    HIGH_RISK_CLASSES,
    brand_inline,
    class_pills,
    disclaimer_box,
    metric_tiles,
    page_header,
    plotly_layout,
    probability_bars,
    risk_badge,
)
from .explainability import generate_gradcam
from .report_generator import generate_pdf_report
from .security import sanitize_text, sanitize_filename, validate_image_bytes
from .styles import theme_colors

PAGES = [
    ('Home', 'home'),
    ('Analysis', 'analyze'),
    ('Results', 'results'),
    ('Performance', 'performance'),
    ('History', 'history'),
    ('About', 'about'),
]


def render_top_nav(active: str, model_names: List[str], metrics: Dict[str, Any]):
    """Responsive top navigation bar with brand, page buttons, and inference settings."""
    st.markdown('<div class="topnav-container">', unsafe_allow_html=True)

    cols = st.columns([1.5, 0.8, 0.85, 0.85, 1.05, 0.8, 0.75, 0.7])

    with cols[0]:
        st.markdown(brand_inline(), unsafe_allow_html=True)

    for (label, key), col in zip(PAGES, cols[1:7]):
        with col:
            is_active = (active == key)
            if st.button(
                label,
                key=f'nav_{key}',
                use_container_width=True,
                type='primary' if is_active else 'secondary',
            ):
                st.session_state.page = key
                st.rerun()

    with cols[7]:
        is_dark = st.session_state.get('theme', 'light') == 'dark'
        toggle_label = '☀ Light' if is_dark else '☾ Dark'
        if st.button(toggle_label, key='theme_toggle', use_container_width=True, help='Toggle light / dark mode'):
            st.session_state.theme = 'light' if is_dark else 'dark'
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Diagnostic & Inference Controls Drawer
    with st.expander('Inference Parameters & Safety Thresholds', expanded=False):
        c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
        with c1:
            st.checkbox(
                'Test-Time Augmentation (TTA)',
                key='use_tta',
                help='Averages the standard prediction with a horizontal flip pass for greater robustness.'
            )
            st.checkbox(
                'Generate Grad-CAM heatmaps',
                key='show_gradcam',
                value=st.session_state.get('show_gradcam', True),
                help='Calculates convolutional activation maps to visualize model focus.'
            )
        with c2:
            st.slider(
                'Uncertainty Floor',
                min_value=0.30,
                max_value=0.85,
                value=float(st.session_state.get('confidence_floor', 0.50)),
                step=0.05,
                key='confidence_floor',
                help='Predictions below this confidence score are flagged as clinically uncertain.'
            )
        with c3:
            active_models = ', '.join(model_names) if model_names else 'MobileNetV2'
            st.caption(f"**Deployed Model:** {active_models}")
            if metrics:
                acc = metrics.get('accuracy', 0.7445)
                st.caption(f"**Test Accuracy:** {acc*100:.1f}%")
            st.caption(f"**Session Analyses:** {len(st.session_state.get('history', []))}")


def render_home():
    """Page 1: Premium Clinical Landing Page."""
    st.markdown("""
<div class="hero-wrapper">
  <div>
    <div class="page-kicker">Clinical Decision Support</div>
    <h1 class="hero-title">Intelligent Skin Analysis.<br/>A New Perspective on Dermatology.</h1>
    <p class="hero-tagline">
      DERMA.AI delivers computer-vision classification across seven dermoscopic lesion categories
      on the HAM10000 taxonomy. Engineered for dermatological research, clinical training,
      and transparent probability attribution.
    </p>
  </div>
  <div class="hero-visual-box">
    <div class="hero-visual-grid"></div>
    <div class="hero-visual-badge">HAM10000 Validated</div>
    <div class="hero-visual-text">
      <div class="hero-visual-heading">Deep Learning Lesion Profiling</div>
      <div class="hero-visual-caption">7-Class Multi-Category Intelligence · Test ROC-AUC 94.4%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Primary Action Row
    c1, c2, _ = st.columns([1.2, 1.2, 2.0])
    with c1:
        if st.button('Start Analysis', type='primary', use_container_width=True, key='home_start'):
            st.session_state.page = 'analyze'
            st.rerun()
    with c2:
        if st.button('Model Performance', use_container_width=True, key='home_perf'):
            st.session_state.page = 'performance'
            st.rerun()

    # Three-step structured workflow
    st.markdown("""
<div class="workflow-grid">
  <div class="workflow-card">
    <div class="workflow-step">STEP 01</div>
    <div class="workflow-title">Dermoscopic Intake</div>
    <div class="workflow-desc">Upload a high-resolution dermoscopic image. The platform verifies file integrity, format, and dimension safety.</div>
  </div>
  <div class="workflow-card">
    <div class="workflow-step">STEP 02</div>
    <div class="workflow-title">Multi-Class Scoring</div>
    <div class="workflow-desc">The fine-tuned MobileNetV2 network extracts feature vectors and computes balanced probabilities across all 7 lesion categories.</div>
  </div>
  <div class="workflow-card">
    <div class="workflow-step">STEP 03</div>
    <div class="workflow-title">Clinical Review & Export</div>
    <div class="workflow-desc">Inspect leading predictions, Grad-CAM attention heatmaps, uncertainty flags, and download an official PDF report.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Model Performance Preview
    st.markdown("""
<div class="card card-tight" style="margin-top: 1.5rem;">
  <div class="card-header">Empirical Benchmark Summary</div>
  <div class="metrics-summary-grid">
    <div class="metric-box">
      <div class="metric-box-label">Independent Test Accuracy</div>
      <div class="metric-box-num">74.5%</div>
      <div class="metric-box-sub">2,004 lesion-held-out test samples</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-label">Balanced Accuracy</div>
      <div class="metric-box-num">76.0%</div>
      <div class="metric-box-sub">Macro-averaged across all 7 classes</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-label">Macro ROC-AUC</div>
      <div class="metric-box-num">94.4%</div>
      <div class="metric-box-sub">One-vs-rest discrimination</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-label">HAM10000 Dataset</div>
      <div class="metric-box-num">10,015</div>
      <div class="metric-box-sub">Images across 7,470 unique lesions</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Mandatory Clinical Disclaimer
    st.markdown(disclaimer_box(
        '<strong>Medical Disclaimer:</strong> DERMA.AI is designed exclusively for research, education, and '
        'diagnostic decision-support experimentation. It is not an FDA-cleared diagnostic device and must never '
        'substitute for direct clinical evaluation, dermoscopy, or biopsy conducted by a licensed dermatologist.'
    ), unsafe_allow_html=True)


def render_analyze(models: Dict[str, Any], label_map: Dict[str, Any], run_prediction: Callable):
    """Page 2: Analysis Workspace with seamless intake and adjacent controls."""
    st.markdown(page_header(
        'Workspace',
        'Skin Lesion Analysis',
        'Upload a dermoscopic capture, verify the preview and file metadata, and initiate inference.'
    ), unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 0.9], gap='large')

    with left_col:
        st.markdown('<div class="card-header">1. Image Intake & Security Validation</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            'Upload dermoscopic photograph (JPG, JPEG, PNG)',
            type=['jpg', 'jpeg', 'png'],
            key='uploader',
            help='Max file size: 10 MB. Supported formats: JPG, JPEG, PNG.'
        )

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            # Security and integrity check
            is_valid, err_msg, validated_pil = validate_image_bytes(raw_bytes, uploaded.name)
            if not is_valid:
                st.error(f"Image Rejected: {err_msg}")
            else:
                st.session_state.uploaded_bytes = raw_bytes
                st.session_state.uploaded_name = sanitize_filename(uploaded.name)
                st.session_state.uploaded_size = len(raw_bytes)
                st.session_state.validated_pil = validated_pil

        if st.session_state.get('uploaded_bytes') and st.session_state.get('validated_pil'):
            pil_img = st.session_state.validated_pil
            fname = st.session_state.get('uploaded_name', 'lesion_image.jpg')
            size_kb = st.session_state.get('uploaded_size', 0) / 1024.0
            w, h = pil_img.size

            st.markdown(f"""
<div class="metadata-grid">
  <div><strong>File:</strong> {fname}</div>
  <div><strong>Size:</strong> {size_kb:.1f} KB</div>
  <div><strong>Resolution:</strong> {w} × {h} px</div>
  <div><strong>Format:</strong> RGB</div>
</div>
""", unsafe_allow_html=True)

            # Prominent Action Row positioned directly with intake
            btn_col1, btn_col2 = st.columns([1.2, 0.8])
            with btn_col1:
                analyze_clicked = st.button('Analyze Image', type='primary', use_container_width=True, key='btn_analyze')
            with btn_col2:
                if st.button('Clear Image', use_container_width=True, key='btn_clear'):
                    for k in ['uploaded_bytes', 'uploaded_name', 'uploaded_size', 'validated_pil', 'last_result', 'uploader']:
                        st.session_state.pop(k, None)
                    st.rerun()

            if analyze_clicked:
                with st.spinner('Executing neural network inference and Grad-CAM extraction...'):
                    result = run_prediction(pil_img)
                    analysis_id = f"DA-{uuid.uuid4().hex[:8].upper()}"
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Compute Grad-CAM if enabled
                    gradcam_img = None
                    if st.session_state.get('show_gradcam', True):
                        try:
                            # Primary model
                            first_model = next(iter(models.values()))
                            _, gradcam_pil = generate_gradcam(
                                first_model,
                                pil_img,
                                img_size=label_map.get('img_size', 224),
                                target_class_idx=result['top_idx']
                            )
                            # Convert gradcam PIL to bytes for session state
                            buf = BytesIO()
                            gradcam_pil.save(buf, format='JPEG')
                            gradcam_img = buf.getvalue()
                        except Exception as e:
                            st.warning(f"Grad-CAM note: {str(e)}")

                    full_result = {
                        **result,
                        'analysis_id': analysis_id,
                        'timestamp': now_str,
                        'filename': fname,
                        'image_bytes': st.session_state.uploaded_bytes,
                        'gradcam_bytes': gradcam_img,
                    }

                    st.session_state.last_result = full_result

                    # Maintain persistent session history
                    history = st.session_state.setdefault('history', [])
                    history.insert(0, full_result)
                    st.session_state.history = history[:100]

                    st.session_state.page = 'results'
                    st.rerun()

        else:
            st.markdown("""
<div class="card card-tight" style="margin-top: 1rem;">
  <div class="workflow-desc">
    <strong>Capture Guidelines:</strong><br/>
    • Center the lesion in the frame with standard dermoscopic illumination.<br/>
    • Avoid extreme glare, peripheral black borders, or heavy skin marker ink.<br/>
    • Uploading clean dermoscopy yields the most reliable probabilities.
  </div>
</div>
""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card-header">2. Image Preview & Framing</div>', unsafe_allow_html=True)
        if st.session_state.get('validated_pil'):
            st.image(st.session_state.validated_pil, use_column_width=True)
        else:
            st.markdown("""
<div class="card card-tight" style="min-height: 280px; display: flex; align-items: center; justify-content: center; text-align: center;">
  <div class="workflow-desc" style="color: var(--muted);">
    Uploaded lesion preview will render here with preserved aspect ratio.
  </div>
</div>
""", unsafe_allow_html=True)

        # Model contextual info
        model_names_str = ', '.join(models.keys()) if models else 'MobileNetV2'
        st.markdown(f"""
<div class="card card-tight" style="margin-top: 1rem;">
  <div class="workflow-step">MODEL CONTEXT</div>
  <div class="workflow-desc">
    <strong>Active Backbone:</strong> {model_names_str}<br/>
    <strong>Input Resolution:</strong> {label_map.get('img_size', 224)} × {label_map.get('img_size', 224)} px<br/>
    <strong>Taxonomy:</strong> 7 HAM10000 classes (akiec, bcc, bkl, df, mel, nv, vasc)
  </div>
</div>
""", unsafe_allow_html=True)


def render_results(label_map: Dict[str, Any]):
    """Page 3: Clinical AI Assessment Report with Grad-CAM and PDF Download."""
    st.markdown(page_header(
        'Assessment',
        'Analysis Results & Explanation',
        'Structured predictive profile generated by the trained neural network. Confidence reflects statistical probability.'
    ), unsafe_allow_html=True)

    result = st.session_state.get('last_result')
    if not result:
        st.markdown("""
<div class="card card-tight">
  <div class="workflow-desc">No active analysis session found. Please submit an image in the Analysis Workspace.</div>
</div>
""", unsafe_allow_html=True)
        if st.button('Open Analysis Workspace', type='primary'):
            st.session_state.page = 'analyze'
            st.rerun()
        return

    class_names = label_map['class_names']
    full_names = label_map['full_names']
    top_code = result['top_code']
    top_name = full_names.get(top_code, top_code)
    top_conf = result['top_conf']
    probs = result['probs']
    analysis_id = result.get('analysis_id', 'DA-REF')
    timestamp = result.get('timestamp', '')
    filename = result.get('filename', 'lesion_sample.jpg')

    floor = float(st.session_state.get('confidence_floor', 0.50))
    is_high_risk = top_code in HIGH_RISK_CLASSES
    is_uncertain = top_conf < floor

    # Top Overview Row
    c1, c2 = st.columns([1.0, 1.2], gap='large')

    with c1:
        st.markdown('<div class="card"><div class="card-header">Submitted Lesion</div>', unsafe_allow_html=True)
        if result.get('image_bytes'):
            orig_pil = Image.open(BytesIO(result['image_bytes']))
            # Grad-CAM toggle display
            has_gradcam = result.get('gradcam_bytes') is not None
            if has_gradcam:
                tab_orig, tab_cam = st.tabs(['Standard Image', 'Grad-CAM Attention Heatmap'])
                with tab_orig:
                    st.image(orig_pil, use_column_width=True)
                with tab_cam:
                    cam_pil = Image.open(BytesIO(result['gradcam_bytes']))
                    st.image(cam_pil, use_column_width=True)
                    st.caption("Thermal overlay highlights regional feature importance for the predicted category.")
            else:
                st.image(orig_pil, use_column_width=True)

        safe_id = sanitize_text(analysis_id)
        safe_fname = sanitize_text(filename)
        safe_time = sanitize_text(timestamp)
        st.markdown(f"""
<div class="metadata-grid">
  <div><strong>ID:</strong> {safe_id}</div>
  <div><strong>Source:</strong> {safe_fname}</div>
  <div><strong>Timestamp:</strong> {safe_time}</div>
</div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="pred-card">
  <div class="card-header">Primary Assessment</div>
  {risk_badge(top_code, top_conf, floor)}
  <div class="pred-headline">{top_name}</div>
  <div style="font-family: monospace; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem;">
    HAM10000 Class Code: <strong>{top_code.upper()}</strong>
  </div>
  <div class="pred-conf-display">{top_conf*100:.1f}%</div>
  <div class="workflow-desc">
    Model classification confidence for leading class.
  </div>
</div>
""", unsafe_allow_html=True)

        # Contextual Clinical Alerts
        if is_uncertain:
            st.warning(
                f"**Clinical Caution:** The model's top confidence ({top_conf*100:.1f}%) is below your "
                f"{floor*100:.0f}% confidence floor. The lesion exhibits ambiguous dermoscopic visual features "
                f"(such as overlapping patterns between BKL and NV)."
            )
        if is_high_risk:
            st.error(
                f"**Priority Referral Alert:** **{top_name}** belongs to a malignant or premalignant "
                f"diagnostic category. In-person clinical dermoscopy and histological biopsy should be prioritized."
            )

    # Full Horizontal Probability Distribution
    st.markdown(probability_bars(class_names, full_names, probs), unsafe_allow_html=True)

    # Action Toolbar
    act_col1, act_col2, act_col3 = st.columns([1.1, 1.0, 1.0])

    with act_col1:
        # Generate PDF report on the fly
        try:
            pdf_bytes = generate_pdf_report(
                analysis_id=analysis_id,
                timestamp=timestamp,
                filename=filename,
                image=Image.open(BytesIO(result['image_bytes'])),
                top_code=top_code,
                top_name=top_name,
                top_conf=top_conf,
                is_high_risk=is_high_risk,
                class_probs=probs,
                class_names=class_names,
                full_names=full_names,
                model_name="MobileNetV2 (HAM10000)"
            )
            st.download_button(
                label="📄 Download PDF Clinical Report",
                data=pdf_bytes,
                file_name=f"DERMA_AI_Report_{analysis_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"PDF generation note: {str(e)}")

    with act_col2:
        if st.button('Analyze Another Image', use_container_width=True):
            st.session_state.page = 'analyze'
            st.rerun()

    with act_col3:
        if st.button('View Model Performance', use_container_width=True):
            st.session_state.page = 'performance'
            st.rerun()


def render_performance(metrics: Dict[str, Any], label_map: Dict[str, Any]):
    """Page 4: Comprehensive Empirical Model Evaluation & Confusion Matrix."""
    st.markdown(page_header(
        'Evaluation',
        'Model Performance & Benchmarks',
        'Rigorous evaluation metrics measured on an independent, group-stratified (lesion-independent) held-out test set.'
    ), unsafe_allow_html=True)

    theme = st.session_state.get('theme', 'light')
    colors = theme_colors(theme)

    # Primary metric tiles
    summary_metrics = {
        'accuracy': 0.7445,
        'balanced_accuracy': 0.7601,
        'f1_macro': 0.6027,
        'roc_auc_macro': 0.9443
    }
    st.markdown(metric_tiles(summary_metrics), unsafe_allow_html=True)

    # Interactive Confusion Matrix
    st.markdown('<div class="card"><div class="card-header">Multi-Class Confusion Matrix (Held-Out Test Set: 2,004 Images)</div>', unsafe_allow_html=True)

    class_names = label_map['class_names']
    full_names = label_map['full_names']
    class_labels = [f"{c.upper()} ({full_names.get(c, c)})" for c in class_names]

    # Real evaluated confusion matrix from evaluate_clean_model.py
    cm_raw = [
        [40, 8, 4, 12, 2, 0, 0],
        [4, 84, 3, 4, 3, 3, 2],
        [22, 21, 114, 12, 32, 19, 0],
        [0, 0, 0, 23, 0, 0, 0],
        [8, 6, 18, 7, 129, 51, 3],
        [5, 26, 53, 59, 108, 1073, 17],
        [0, 0, 0, 0, 0, 0, 29]
    ]
    cm_norm = [
        [0.606, 0.121, 0.061, 0.182, 0.030, 0.000, 0.000],
        [0.039, 0.816, 0.029, 0.039, 0.029, 0.029, 0.019],
        [0.100, 0.095, 0.518, 0.055, 0.145, 0.086, 0.000],
        [0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
        [0.036, 0.027, 0.081, 0.032, 0.581, 0.230, 0.014],
        [0.004, 0.019, 0.040, 0.044, 0.081, 0.800, 0.013],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]
    ]

    toggle_col1, _ = st.columns([1.5, 3.5])
    with toggle_col1:
        view_norm = st.radio(
            "Matrix Display Mode",
            ["Normalized (Recall Proportions)", "Absolute Sample Counts"],
            horizontal=True
        )

    matrix_to_plot = cm_norm if "Normalized" in view_norm else cm_raw
    text_template = "%{z:.1%}" if "Normalized" in view_norm else "%{z}"

    fig_cm = go.Figure(data=go.Heatmap(
        z=matrix_to_plot,
        x=class_labels,
        y=class_labels,
        colorscale='Teal' if theme == 'light' else 'Mint',
        hoverongaps=False,
        text=matrix_to_plot,
        texttemplate=text_template,
        textfont={"size": 11, "family": "Inter, sans-serif"}
    ))

    layout = plotly_layout(theme)
    fig_cm.update_layout(
        **layout,
        height=520,
        xaxis=dict(title="<b>Predicted Classification</b>", tickangle=-25),
        yaxis=dict(title="<b>True Diagnostic Category</b>", autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Per-Class Precision, Recall, and F1 Breakdown Table
    st.markdown('<div class="card"><div class="card-header">Per-Class Clinical Performance Profile</div>', unsafe_allow_html=True)
    per_class_data = [
        {"Category": "AKIEC (Actinic keratoses)", "Precision": "50.6%", "Recall": "60.6%", "F1-Score": "55.2%", "Support": "66", "Tier": "Higher Concern"},
        {"Category": "BCC (Basal cell carcinoma)", "Precision": "57.9%", "Recall": "81.6%", "F1-Score": "67.7%", "Support": "103", "Tier": "Higher Concern"},
        {"Category": "BKL (Benign keratosis)", "Precision": "59.4%", "Recall": "51.8%", "F1-Score": "55.3%", "Support": "220", "Tier": "Usually Benign"},
        {"Category": "DF (Dermatofibroma)", "Precision": "19.7%", "Recall": "100.0%", "F1-Score": "32.9%", "Support": "23", "Tier": "Usually Benign"},
        {"Category": "MEL (Melanoma)", "Precision": "47.1%", "Recall": "58.1%", "F1-Score": "52.0%", "Support": "222", "Tier": "Higher Concern"},
        {"Category": "NV (Melanocytic nevi)", "Precision": "93.6%", "Recall": "80.0%", "F1-Score": "86.3%", "Support": "1,341", "Tier": "Usually Benign"},
        {"Category": "VASC (Vascular lesions)", "Precision": "56.9%", "Recall": "100.0%", "F1-Score": "72.5%", "Support": "29", "Tier": "Usually Benign"},
    ]
    st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # In-Depth Clinical Investigation: BKL vs NV
    st.markdown("""
<div class="card">
  <div class="card-header">Clinical Investigation: BKL (Benign Keratosis) vs NV (Melanocytic Nevi)</div>
  <div class="workflow-desc">
    <strong>Root Causes of Classification Ambiguity:</strong><br/>
    1. <strong>Phenotypical & Dermoscopic Overlap:</strong> Benign keratoses (including seborrheic keratoses and solar lentigines) frequently manifest pigmented reticular networks, fingerprint-like patterns, and globule-like structures that closely mirror melanocytic nevi under optical magnification.<br/>
    2. <strong>Severe Dataset Prior Imbalance:</strong> In the HAM10000 dataset, Melanocytic Nevi (NV) constitute 66.95% (6,705 samples) while BKL represents only 10.97% (1,099 samples). When feature ambiguity occurs, unweighted bayesian priors bias uncertain probabilities toward NV.<br/>
    3. <strong>Resolution Limits:</strong> Standard CNN input scaling (224×224 px) compresses high-frequency microscopic cues (such as milia-like cysts or comedo-like openings) that human dermatologists use to differentiate keratoses from nevi.<br/>
    4. <strong>Calibrated Safety Response:</strong> In DERMA.AI, when BKL samples are predicted as NV with marginal confidence (35–50%), the application raises the <em>Uncertainty Flag</em> and displays the full probability spread rather than reporting false certainty.
  </div>
</div>
""", unsafe_allow_html=True)


def render_history(label_map: Dict[str, Any]):
    """Page 5: In-Memory Clinical Session History."""
    st.markdown(page_header(
        'Session Records',
        'Analysis History',
        'In-memory history of image classifications evaluated during your active browser session.'
    ), unsafe_allow_html=True)

    history = st.session_state.get('history', [])
    if not history:
        st.markdown("""
<div class="card card-tight">
  <div class="workflow-desc">No analyses completed in this session yet. Upload a lesion image to populate session records.</div>
</div>
""", unsafe_allow_html=True)
        return

    full_names = label_map['full_names']
    filter_query = st.text_input('Filter by filename, class code, or diagnosis name:', '', placeholder='e.g., ISIC_0024306 or Melanoma')

    rows = []
    filtered_items = []
    for item in history:
        aid = item.get('analysis_id', 'DA-REF')
        time_str = item.get('timestamp', '')
        fname = item.get('filename', 'lesion.jpg')
        code = item.get('top_code', '')
        name = full_names.get(code, code)
        conf = f"{item.get('top_conf', 0)*100:.1f}%"
        is_hr = code in HIGH_RISK_CLASSES

        blob = f"{aid} {time_str} {fname} {code} {name}".lower()
        if filter_query and filter_query.lower() not in blob:
            continue

        filtered_items.append(item)
        rows.append({
            'Analysis ID': aid,
            'Timestamp': time_str,
            'Filename': fname,
            'Predicted Diagnosis': name,
            'Class Code': code.upper(),
            'Confidence': conf,
            'Clinical Tier': 'Higher Concern' if is_hr else 'Usually Benign'
        })

    if not rows:
        st.info("No records match your filter criteria.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Detailed Inspection Selector
    st.markdown('<div class="card-header" style="margin-top: 1.5rem;">Detailed Entry Inspection</div>', unsafe_allow_html=True)
    options = [f"{r['Analysis ID']} · {r['Filename']} · {r['Predicted Diagnosis']}" for r in rows]
    selected_idx = st.selectbox('Select entry to review:', range(len(options)), format_func=lambda i: options[i])

    selected_item = filtered_items[selected_idx]
    detail_col1, detail_col2 = st.columns([1.0, 1.2], gap='medium')

    with detail_col1:
        if selected_item.get('image_bytes'):
            st.image(Image.open(BytesIO(selected_item['image_bytes'])), use_column_width=True)

    with detail_col2:
        top_c = selected_item['top_code']
        safe_item_id = sanitize_text(selected_item.get('analysis_id', 'DA-HIST'))
        safe_item_name = sanitize_text(full_names.get(top_c, top_c))
        safe_item_file = sanitize_text(selected_item.get('filename', 'lesion.jpg'))
        safe_item_time = sanitize_text(selected_item.get('timestamp', ''))
        st.markdown(f"""
<div class="card">
  <div class="card-header">{safe_item_id}</div>
  {risk_badge(top_c, selected_item['top_conf'])}
  <div class="pred-headline">{safe_item_name}</div>
  <div class="pred-conf-display">{selected_item['top_conf']*100:.1f}%</div>
  <div class="metadata-grid">
    <div><strong>File:</strong> {safe_item_file}</div>
    <div><strong>Time:</strong> {safe_item_time}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Probability bars for selected entry
    st.markdown(probability_bars(label_map['class_names'], full_names, selected_item['probs']), unsafe_allow_html=True)

    # PDF Download for this entry
    try:
        pdf_data = generate_pdf_report(
            analysis_id=selected_item.get('analysis_id', 'DA-HIST'),
            timestamp=selected_item.get('timestamp', ''),
            filename=selected_item.get('filename', 'lesion.jpg'),
            image=Image.open(BytesIO(selected_item['image_bytes'])),
            top_code=selected_item['top_code'],
            top_name=full_names.get(selected_item['top_code'], selected_item['top_code']),
            top_conf=selected_item['top_conf'],
            is_high_risk=selected_item['top_code'] in HIGH_RISK_CLASSES,
            class_probs=selected_item['probs'],
            class_names=label_map['class_names'],
            full_names=full_names
        )
        st.download_button(
            label="📄 Download PDF for this Record",
            data=pdf_data,
            file_name=f"DERMA_AI_{selected_item.get('analysis_id')}.pdf",
            mime="application/pdf",
            key=f"dl_hist_{selected_item.get('analysis_id')}"
        )
    except Exception as e:
        pass

    if st.button('Clear All Session Records', key='clear_hist'):
        st.session_state.history = []
        st.rerun()


def render_about(models: Dict[str, Any], label_map: Dict[str, Any], metrics: Dict[str, Any]):
    """Page 6: Transparent Architecture, Dataset, and Responsible AI Documentation."""
    st.markdown(page_header(
        'Documentation',
        'About DERMA.AI',
        'Detailed specifications covering architecture, dataset provenance, classification taxonomy, and clinical boundaries.'
    ), unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.markdown(r"""
<div class="card">
  <div class="card-header">Core Machine Learning Architecture</div>
  <div class="workflow-desc">
    • <strong>Backbone Architecture:</strong> MobileNetV2 pretrained on ImageNet.<br/>
    • <strong>Classifier Head:</strong> GlobalAveragePooling2D → LayerNormalization → Dense(512, GELU) → Dropout(0.4) → Dense(256, GELU) → Dropout(0.3) → Dense(7, Softmax).<br/>
    • <strong>Loss Function:</strong> Sparse Categorical Focal Loss (\(\gamma=2.0\)) with label smoothing (\(\alpha=0.05\)) to mitigate heavy class imbalance.<br/>
    • <strong>Optimization:</strong> Two-phase AdamW fine-tuning with ReduceLROnPlateau and Early Stopping.<br/>
    • <strong>Inference Format:</strong> Exported Keras 3 (<code>.keras</code>) bundle.
  </div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="card">
  <div class="card-header">Dataset Provenance: HAM10000</div>
  <div class="workflow-desc">
    • <strong>Source:</strong> "Human Against Machine with 10,000 training images" (ViDIR Group, Medical University of Vienna & Cliff Rosendahl, Queensland).<br/>
    • <strong>Volume:</strong> 10,015 dermatoscopic images corresponding to 7,470 distinct patient lesions.<br/>
    • <strong>Validation Split:</strong> Group-stratified by <code>lesion_id</code> to strictly guarantee zero patient/lesion leakage between training and testing splits.<br/>
    • <strong>Image Modality:</strong> Polarized and non-polarized contact dermoscopy.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-header">Classification Taxonomy (7 Categories)</div>', unsafe_allow_html=True)
    st.markdown(class_pills(label_map['class_names'], label_map['full_names']), unsafe_allow_html=True)
    st.markdown("""
<div class="workflow-desc" style="margin-top: 0.8rem;">
  <strong>AKIEC:</strong> Actinic keratoses and intraepithelial carcinoma (squamous cell carcinoma precursor).<br/>
  <strong>BCC:</strong> Basal cell carcinoma, the most prevalent form of skin cancer.<br/>
  <strong>BKL:</strong> Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses).<br/>
  <strong>DF:</strong> Dermatofibroma, benign fibrous histiocytoma.<br/>
  <strong>MEL:</strong> Melanoma, high-risk malignant neoplasm of melanocytes.<br/>
  <strong>NV:</strong> Melanocytic nevi, benign common or dysplastic moles (66.95% of HAM10000).<br/>
  <strong>VASC:</strong> Vascular lesions (angiomas, pyogenic granulomas, angiokeratomas).
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card" style="margin-top: 1rem;">
  <div class="card-header">System Boundaries & Technical Limitations</div>
  <div class="workflow-desc">
    • <strong>Domain Shift:</strong> Performance may degrade on non-dermoscopic clinical camera photos or atypical anatomical sites (mucosa, nail beds).<br/>
    • <strong>Color Calibration:</strong> Differences in dermatoscope optical filters and lighting temperatures can impact class probabilities.<br/>
    • <strong>Softmax Interpretation:</strong> Softmax scores represent mathematical category likelihoods conditioned on the training distribution, not biopsy-proven certainty.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(disclaimer_box(
        '<strong>Responsible Clinical AI:</strong> AI systems in dermatology are designed to empower and assist '
        'qualified clinical practitioners, not replace dermatologists. Any suspicious lesion exhibiting ABCDE '
        'criteria (Asymmetry, Border irregularity, Color variegation, Diameter >6mm, or Evolution) mandates '
        'prompt medical consultation regardless of model output.'
    ), unsafe_allow_html=True)
