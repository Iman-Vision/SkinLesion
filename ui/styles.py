"""Global DERMA.AI design system CSS for Streamlit."""

COLORS = {
    'bg': '#F8F9F7',
    'primary': '#172321',
    'secondary': '#263832',
    'accent': '#477568',
    'light_accent': '#DDE9E2',
    'card': '#FFFFFF',
    'border': '#E2E8E3',
    'muted': '#6D7972',
    'concern': '#8B3A3A',
    'benign': '#3D6B4F',
}


def inject_css():
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {{
  --bg: {COLORS['bg']};
  --primary: {COLORS['primary']};
  --secondary: {COLORS['secondary']};
  --accent: {COLORS['accent']};
  --light-accent: {COLORS['light_accent']};
  --card: {COLORS['card']};
  --border: {COLORS['border']};
  --muted: {COLORS['muted']};
  --concern: {COLORS['concern']};
  --benign: {COLORS['benign']};
}}

html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--primary);
}}

.stApp {{
  background: var(--bg);
}}

header[data-testid="stHeader"] {{
  background: transparent;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}

div[data-testid="stToolbar"] {{visibility: hidden;}}

.block-container {{
  padding-top: 1.4rem !important;
  padding-bottom: 4rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  max-width: 1120px !important;
}}

section[data-testid="stSidebar"] {{
  background: var(--card) !important;
  border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] > div {{
  padding-top: 1.5rem;
}}

section[data-testid="stSidebar"] .block-container {{
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}}

h1, h2, h3, .derma-heading {{
  font-family: 'Manrope', 'Inter', sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--primary) !important;
}}

.derma-brand {{
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  letter-spacing: 0.08em;
  color: var(--primary);
  margin-bottom: 0.15rem;
}}

.derma-brand span {{
  color: var(--accent);
}}

.derma-brand-sub {{
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 1.4rem;
}}

.derma-nav-label {{
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.2rem 0 0.45rem 0;
}}

.hero {{
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 2.5rem;
  align-items: center;
  padding: 2.2rem 0 1.2rem 0;
  min-height: 62vh;
}}

.hero-kicker {{
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 1rem;
}}

.hero-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
  font-size: clamp(2.4rem, 4.5vw, 3.6rem);
  line-height: 1.08;
  letter-spacing: -0.035em;
  color: var(--primary);
  margin: 0 0 1.1rem 0;
}}

.hero-lede {{
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--muted);
  max-width: 34rem;
  margin-bottom: 1.8rem;
}}

.hero-visual {{
  background: linear-gradient(145deg, var(--secondary) 0%, var(--primary) 55%, #1f3d36 100%);
  border-radius: 18px;
  min-height: 380px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 24px 60px rgba(23, 35, 33, 0.18);
}}

.hero-visual::before {{
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(221, 233, 226, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(221, 233, 226, 0.08) 1px, transparent 1px);
  background-size: 28px 28px;
}}

.hero-visual-inner {{
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: 2rem;
}}

.hero-visual-label {{
  color: rgba(248, 249, 247, 0.7);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}}

.hero-visual-title {{
  font-family: 'Manrope', sans-serif;
  color: #F8F9F7;
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.3;
  max-width: 14rem;
}}

.hero-scan {{
  position: absolute;
  top: 18%;
  left: 14%;
  width: 58%;
  aspect-ratio: 1;
  border: 1px solid rgba(221, 233, 226, 0.35);
  border-radius: 12px;
}}

.hero-scan::after {{
  content: '';
  position: absolute;
  left: 0; right: 0;
  top: 42%;
  height: 1px;
  background: rgba(71, 117, 104, 0.85);
  box-shadow: 0 0 18px rgba(71, 117, 104, 0.6);
}}

.page-header {{
  margin: 0.4rem 0 1.6rem 0;
}}

.page-kicker {{
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.45rem;
}}

.page-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 750;
  font-size: 2rem;
  letter-spacing: -0.03em;
  margin: 0 0 0.45rem 0;
  color: var(--primary);
}}

.page-subtitle {{
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.55;
  max-width: 40rem;
  margin: 0;
}}

.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.35rem 1.45rem;
  box-shadow: 0 1px 2px rgba(23, 35, 33, 0.03);
}}

.card-tight {{
  padding: 1rem 1.15rem;
}}

.card-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  margin: 0 0 0.75rem 0;
  color: var(--primary);
}}

.muted {{
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.55;
}}

.step-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 2.2rem 0 1.5rem 0;
}}

.step-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.25rem 1.3rem;
}}

.step-num {{
  font-family: 'Manrope', sans-serif;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--accent);
  margin-bottom: 0.55rem;
}}

.step-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 700;
  font-size: 1rem;
  margin-bottom: 0.35rem;
}}

.disclaimer {{
  background: var(--light-accent);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  color: var(--secondary);
  font-size: 0.86rem;
  line-height: 1.55;
  margin-top: 1.5rem;
}}

.metric-row {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.9rem;
}}

.metric-tile {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.15rem;
}}

.metric-label {{
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.35rem;
}}

.metric-value {{
  font-family: 'Manrope', sans-serif;
  font-size: 1.7rem;
  font-weight: 750;
  color: var(--primary);
  letter-spacing: -0.02em;
}}

.metric-hint {{
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.2rem;
}}

.result-hero {{
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 1.2rem;
  margin-bottom: 1.2rem;
}}

.pred-class {{
  font-family: 'Manrope', sans-serif;
  font-size: 1.65rem;
  font-weight: 750;
  letter-spacing: -0.02em;
  margin: 0.35rem 0 0.15rem 0;
}}

.pred-code {{
  font-size: 0.82rem;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}

.pred-conf {{
  font-family: 'Manrope', sans-serif;
  font-size: 2.4rem;
  font-weight: 750;
  letter-spacing: -0.03em;
  color: var(--primary);
  margin: 0.8rem 0 0.15rem 0;
}}

.badge {{
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}}

.badge-concern {{
  background: rgba(139, 58, 58, 0.08);
  color: var(--concern);
  border-color: rgba(139, 58, 58, 0.2);
}}

.badge-benign {{
  background: rgba(61, 107, 79, 0.08);
  color: var(--benign);
  border-color: rgba(61, 107, 79, 0.2);
}}

.badge-neutral {{
  background: var(--light-accent);
  color: var(--secondary);
  border-color: var(--border);
}}

.file-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem 1.1rem;
  font-size: 0.84rem;
  color: var(--muted);
  margin-top: 0.75rem;
}}

.file-meta strong {{
  color: var(--primary);
  font-weight: 600;
}}

div[data-testid="stFileUploaderDropzone"] {{
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 14px !important;
  padding: 1.5rem !important;
}}

div[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: var(--accent) !important;
  background: #FBFCFA !important;
}}

div[data-testid="stFileUploaderDropzone"] button {{
  border-radius: 8px !important;
}}

.stButton > button {{
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.15s ease !important;
  border: 1px solid transparent !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {{
  background: var(--primary) !important;
  color: #F8F9F7 !important;
  border-color: var(--primary) !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {{
  background: var(--secondary) !important;
  border-color: var(--secondary) !important;
}}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {{
  background: transparent !important;
  color: var(--primary) !important;
  border: 1px solid var(--border) !important;
}}

div[data-testid="stImage"] img {{
  border-radius: 12px;
  border: 1px solid var(--border);
}}

.prob-row {{
  display: grid;
  grid-template-columns: 11rem 1fr 3.4rem;
  gap: 0.75rem;
  align-items: center;
  margin-bottom: 0.55rem;
}}

.prob-label {{
  font-size: 0.84rem;
  color: var(--primary);
}}

.prob-bar-track {{
  height: 8px;
  background: var(--light-accent);
  border-radius: 999px;
  overflow: hidden;
}}

.prob-bar-fill {{
  height: 100%;
  border-radius: 999px;
  background: var(--accent);
}}

.prob-bar-fill.top {{
  background: var(--primary);
}}

.prob-bar-fill.concern {{
  background: var(--concern);
}}

.prob-pct {{
  font-size: 0.8rem;
  color: var(--muted);
  text-align: right;
  font-variant-numeric: tabular-nums;
}}

.class-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--card);
  font-size: 0.78rem;
  color: var(--secondary);
  margin: 0 0.35rem 0.45rem 0;
}}

.about-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}}

@media (max-width: 900px) {{
  .hero, .result-hero, .step-grid, .metric-row, .about-grid {{
    grid-template-columns: 1fr;
  }}
  .hero-visual {{ min-height: 260px; }}
  .prob-row {{ grid-template-columns: 1fr; gap: 0.25rem; }}
}}
</style>
"""
