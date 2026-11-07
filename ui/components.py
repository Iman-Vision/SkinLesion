"""Reusable DERMA.AI presentation helpers."""

from .styles import COLORS

HIGH_RISK_CLASSES = {'akiec', 'bcc', 'mel'}


def brand_block():
    return """
<div class="derma-brand">DERMA.<span>AI</span></div>
<div class="derma-brand-sub">Clinical image intelligence</div>
"""


def page_header(kicker: str, title: str, subtitle: str = ''):
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ''
    return f"""
<div class="page-header">
  <div class="page-kicker">{kicker}</div>
  <h1 class="page-title">{title}</h1>
  {sub}
</div>
"""


def disclaimer_box(text: str):
    return f'<div class="disclaimer">{text}</div>'


def risk_badge(class_code: str) -> str:
    if class_code in HIGH_RISK_CLASSES:
        return '<span class="badge badge-concern">Higher clinical concern</span>'
    return '<span class="badge badge-benign">Typically lower concern</span>'


def probability_bars(class_names, full_names, probs):
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    rows = []
    for rank, i in enumerate(order):
        code = class_names[i]
        name = full_names.get(code, code)
        pct = float(probs[i]) * 100
        width = max(1.5, pct)
        fill_cls = 'top' if rank == 0 else ('concern' if code in HIGH_RISK_CLASSES else '')
        rows.append(f"""
<div class="prob-row">
  <div class="prob-label">{name}</div>
  <div class="prob-bar-track"><div class="prob-bar-fill {fill_cls}" style="width:{width:.2f}%"></div></div>
  <div class="prob-pct">{pct:.1f}%</div>
</div>
""")
    return '<div class="card"><div class="card-title">Class probability distribution</div>' + ''.join(rows) + '</div>'


def metric_tiles(metrics: dict):
    tiles = [
        ('Accuracy', metrics['accuracy'], 'Held-out test set'),
        ('Macro F1', metrics['f1_macro'], 'Balanced class quality'),
        ('ROC-AUC', metrics['roc_auc_macro'], 'One-vs-rest macro'),
    ]
    parts = []
    for label, value, hint in tiles:
        parts.append(f"""
<div class="metric-tile">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value*100:.1f}%</div>
  <div class="metric-hint">{hint}</div>
</div>
""")
    return '<div class="metric-row">' + ''.join(parts) + '</div>'


def class_pills(class_names, full_names):
    pills = []
    for c in class_names:
        tier = 'Higher concern' if c in HIGH_RISK_CLASSES else 'Usually benign'
        pills.append(
            f'<span class="class-pill"><strong>{c}</strong> {full_names.get(c, c)} · {tier}</span>'
        )
    return ''.join(pills)


PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color=COLORS['muted'], size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    colorway=[COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['concern']],
)
