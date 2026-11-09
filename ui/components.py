"""
DERMA.AI Presentation Components
Reusable HTML/UI components engineered for clarity, accessibility, and visual hierarchy.
All user-facing text is sanitized to prevent XSS.
"""

from typing import List, Dict, Any
from .security import sanitize_text
from .styles import THEMES, theme_colors

HIGH_RISK_CLASSES = {'akiec', 'bcc', 'mel'}


def brand_inline() -> str:
    return """
<div class="derma-brand">
  <div class="derma-brand-title">DERMA.<span>AI</span></div>
  <div class="derma-brand-sub">Clinical Skin Intelligence</div>
</div>
"""


def page_header(kicker: str, title: str, subtitle: str = '') -> str:
    kicker_clean = sanitize_text(kicker)
    title_clean = sanitize_text(title)
    sub_clean = f'<p class="page-subtitle">{sanitize_text(subtitle)}</p>' if subtitle else ''
    return f"""
<div class="page-header">
  <div class="page-kicker">{kicker_clean}</div>
  <h1 class="page-title">{title_clean}</h1>
  {sub_clean}
</div>
"""


def disclaimer_box(text: str) -> str:
    return f'<div class="clinical-disclaimer">{text}</div>'


def risk_badge(class_code: str, confidence: float = 1.0, threshold: float = 0.5) -> str:
    class_code_clean = sanitize_text(class_code).lower()
    if confidence < threshold:
        return '<span class="badge badge-uncertain">Uncertain / Low Confidence</span>'
    if class_code_clean in HIGH_RISK_CLASSES:
        return '<span class="badge badge-concern">Higher Clinical Concern</span>'
    return '<span class="badge badge-benign">Typically Lower Concern</span>'


def probability_bars(class_names: List[str], full_names: Dict[str, str], probs: List[float]) -> str:
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    rows = []
    for rank, i in enumerate(order):
        code = class_names[i]
        name = sanitize_text(full_names.get(code, code))
        pct = float(probs[i]) * 100
        width = max(1.5, pct)
        fill_cls = 'prob-fill-top' if rank == 0 else ('prob-fill-concern' if code in HIGH_RISK_CLASSES else '')
        rows.append(f"""
<div class="prob-item">
  <div class="prob-name" title="{name}">{name}</div>
  <div class="prob-track">
    <div class="prob-fill {fill_cls}" style="width: {width:.2f}%;"></div>
  </div>
  <div class="prob-val">{pct:.1f}%</div>
</div>
""")
    return f"""
<div class="card">
  <div class="card-header">Class Probability Distribution</div>
  {''.join(rows)}
</div>
"""


def metric_tiles(metrics: Dict[str, Any]) -> str:
    tiles = [
        ('Accuracy', metrics.get('accuracy', 0.0), 'Held-out test set'),
        ('Balanced Acc', metrics.get('balanced_accuracy', metrics.get('accuracy', 0.0)), 'Unweighted class mean'),
        ('Macro F1', metrics.get('f1_macro', 0.0), 'Harmonic mean'),
        ('Macro ROC-AUC', metrics.get('roc_auc_macro', 0.0), 'Multi-class separation'),
    ]
    parts = []
    for label, value, hint in tiles:
        parts.append(f"""
<div class="metric-box">
  <div class="metric-box-label">{label}</div>
  <div class="metric-box-num">{value*100:.1f}%</div>
  <div class="metric-box-sub">{hint}</div>
</div>
""")
    return f'<div class="metrics-summary-grid">{"".join(parts)}</div>'


def class_pills(class_names: List[str], full_names: Dict[str, str]) -> str:
    pills = []
    for c in class_names:
        clean_code = sanitize_text(c)
        clean_name = sanitize_text(full_names.get(c, c))
        tier = 'Higher concern' if c in HIGH_RISK_CLASSES else 'Usually benign'
        pills.append(
            f'<span class="taxonomy-pill"><strong>{clean_code.upper()}</strong> {clean_name} · {tier}</span>'
        )
    return ''.join(pills)


def plotly_layout(theme: str = 'light') -> dict:
    c = theme_colors(theme)
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=c['chart_text'], size=12),
        margin=dict(l=40, r=20, t=40, b=60),
        colorway=[c['chart_bar'], c['accent'], c['primary'], c['concern']],
    )
