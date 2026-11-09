"""
DERMA.AI Design System & Responsive Stylesheet
Clinical SaaS Design Language: Premium Minimalist Clinical Technology.
Adheres strictly to the brand palette, typography, and responsive breakpoints.
"""

THEMES = {
    'light': {
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
        'uploader_hover': '#F1F6F3',
        'nav_bg': 'rgba(255, 255, 255, 0.96)',
        'hero_shadow': 'rgba(23, 35, 33, 0.08)',
        'btn_fg': '#FFFFFF',
        'chart_text': '#172321',
        'chart_muted': '#6D7972',
        'chart_grid': '#E2E8E3',
        'chart_bar': '#477568',
    },
    'dark': {
        'bg': '#0F1614',
        'primary': '#F0F4F2',
        'secondary': '#C8D5CE',
        'accent': '#5B9383',
        'light_accent': '#1C2925',
        'card': '#16221E',
        'border': '#25352F',
        'muted': '#96A8A0',
        'concern': '#D97575',
        'benign': '#6BB286',
        'uploader_hover': '#1F2F29',
        'nav_bg': 'rgba(22, 34, 30, 0.96)',
        'hero_shadow': 'rgba(0, 0, 0, 0.35)',
        'btn_fg': '#FFFFFF',
        'chart_text': '#F0F4F2',
        'chart_muted': '#96A8A0',
        'chart_grid': '#25352F',
        'chart_bar': '#5B9383',
    },
}

COLORS = THEMES['light']


def theme_colors(theme: str = 'light') -> dict:
    return THEMES.get(theme, THEMES['light'])


def inject_css(theme: str = 'light') -> str:
    c = theme_colors(theme)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {{
  --bg: {c['bg']};
  --primary: {c['primary']};
  --secondary: {c['secondary']};
  --accent: {c['accent']};
  --light-accent: {c['light_accent']};
  --card: {c['card']};
  --border: {c['border']};
  --muted: {c['muted']};
  --concern: {c['concern']};
  --benign: {c['benign']};
  --uploader-hover: {c['uploader_hover']};
  --nav-bg: {c['nav_bg']};
  --btn-fg: {c['btn_fg']};
}}

html, body, [class*="css"], .stApp {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--primary) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  margin: 0;
  padding: 0;
}}

/* Clean up Streamlit default decoration */
header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
}}
#MainMenu, footer, .stDeployButton, div[data-testid="stToolbar"] {{
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}}
section[data-testid="stSidebar"],
div[data-testid="collapsedControl"] {{
  display: none !important;
}}

/* Standardized Content Container */
.block-container {{
  padding-top: 1.25rem !important;
  padding-bottom: 3.5rem !important;
  padding-left: clamp(1rem, 3.5vw, 2.5rem) !important;
  padding-right: clamp(1rem, 3.5vw, 2.5rem) !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}}

/* Typography Hierarchy */
h1, h2, h3, h4, .manrope-title {{
  font-family: 'Manrope', 'Inter', sans-serif !important;
  font-weight: 750 !important;
  letter-spacing: -0.025em !important;
  color: var(--primary) !important;
  margin-top: 0 !important;
}}

p, div, span, label {{
  font-family: 'Inter', sans-serif;
}}

/* Top Navigation Bar */
.topnav-container {{
  background: var(--nav-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.65rem 1rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}

.derma-brand {{
  display: flex;
  flex-direction: column;
  justify-content: center;
}}
.derma-brand-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  letter-spacing: 0.06em;
  color: var(--primary);
  line-height: 1.1;
}}
.derma-brand-title span {{
  color: var(--accent);
}}
.derma-brand-sub {{
  font-size: 0.68rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500;
  margin-top: 2px;
}}

/* Topnav button overrides */
.topnav-container .stButton > button {{
  padding: 0.45rem 0.55rem !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  white-space: nowrap !important;
  min-height: 38px !important;
  border-radius: 8px !important;
}}

/* Page Headers */
.page-header {{
  margin-bottom: 1.75rem;
}}
.page-kicker {{
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
}}
.page-title {{
  font-family: 'Manrope', sans-serif;
  font-size: clamp(1.6rem, 2.8vw, 2.1rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--primary);
  margin: 0 0 0.4rem 0;
  line-height: 1.15;
}}
.page-subtitle {{
  font-size: 0.95rem;
  color: var(--muted);
  line-height: 1.5;
  max-width: 44rem;
  margin: 0;
}}

/* Cards */
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  margin-bottom: 1rem;
  box-sizing: border-box;
}}
.card-tight {{
  padding: 1rem 1.15rem;
}}
.card-header {{
  font-family: 'Manrope', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--primary);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}

/* Home Hero Section */
.hero-wrapper {{
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 2.2rem;
  align-items: center;
  padding: 1rem 0 2rem 0;
}}
.hero-title {{
  font-family: 'Manrope', sans-serif;
  font-size: clamp(2.2rem, 4vw, 3.2rem);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.035em;
  color: var(--primary);
  margin-bottom: 1.1rem;
}}
.hero-tagline {{
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--muted);
  max-width: 36rem;
  margin-bottom: 1.75rem;
}}
.hero-visual-box {{
  background: linear-gradient(140deg, #172321 0%, #263832 50%, #355047 100%);
  border-radius: 16px;
  min-height: 320px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 45px {c['hero_shadow']};
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: 2rem;
  box-sizing: border-box;
}}
.hero-visual-grid {{
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(221, 233, 226, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(221, 233, 226, 0.08) 1px, transparent 1px);
  background-size: 24px 24px;
}}
.hero-visual-badge {{
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
  color: #F8F9F7;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}}
.hero-visual-text {{
  position: relative;
  z-index: 2;
}}
.hero-visual-heading {{
  font-family: 'Manrope', sans-serif;
  color: #FFFFFF;
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.25;
  margin-bottom: 0.4rem;
}}
.hero-visual-caption {{
  color: #DDE9E2;
  font-size: 0.82rem;
  line-height: 1.4;
}}

/* Workflow Step Cards */
.workflow-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.2rem;
  margin: 1.8rem 0;
}}
.workflow-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem;
}}
.workflow-step {{
  font-family: 'Manrope', sans-serif;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--accent);
  margin-bottom: 0.45rem;
}}
.workflow-title {{
  font-family: 'Manrope', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  margin-bottom: 0.35rem;
  color: var(--primary);
}}
.workflow-desc {{
  font-size: 0.88rem;
  color: var(--muted);
  line-height: 1.5;
}}

/* Medical Disclaimer Box */
.clinical-disclaimer {{
  background: var(--light-accent);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.95rem 1.2rem;
  color: var(--secondary);
  font-size: 0.86rem;
  line-height: 1.5;
  margin-top: 1.5rem;
}}

/* Universal Button Styling */
.stButton > button,
div[data-testid="stDownloadButton"] > button {{
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
  padding: 0.55rem 1.15rem !important;
  transition: all 0.15s ease-in-out !important;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stDownloadButton"] > button[kind="primary"],
div[data-testid="stDownloadButton"] > button[data-testid="baseButton-primary"] {{
  background: var(--primary) !important;
  color: var(--btn-fg) !important;
  border: 1px solid var(--primary) !important;
}}
.stButton > button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {{
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
}}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
div[data-testid="stDownloadButton"] > button[kind="secondary"],
div[data-testid="stDownloadButton"] > button[data-testid="baseButton-secondary"] {{
  background: var(--card) !important;
  color: var(--primary) !important;
  border: 1px solid var(--border) !important;
}}
.stButton > button[kind="secondary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="secondary"]:hover {{
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}}

/* File Uploader Customization */
div[data-testid="stFileUploaderDropzone"] {{
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 12px !important;
  padding: 1.6rem !important;
  text-align: center !important;
  color: var(--muted) !important;
}}
div[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: var(--accent) !important;
  background: var(--uploader-hover) !important;
}}
div[data-testid="stFileUploaderDropzone"] button {{
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
}}
div[data-testid="stFileUploaderDropzone"] button:hover {{
  background: var(--secondary) !important;
}}
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span {{
  color: var(--muted) !important;
}}

/* Streamlit Tabs Styling */
button[data-baseweb="tab"] {{
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  color: var(--muted) !important;
  padding-bottom: 0.5rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--primary) !important;
  border-bottom-color: var(--accent) !important;
}}
div[data-baseweb="tab-highlight"] {{
  background-color: var(--accent) !important;
}}

/* Streamlit Expander Styling */
div[data-testid="stExpander"] {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}}
div[data-testid="stExpander"] summary {{
  color: var(--primary) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}}

/* Streamlit Slider Styling */
div[data-testid="stSlider"] div[role="slider"] {{
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}}

/* Streamlit Radio Buttons */
div[data-testid="stRadio"] label {{
  font-size: 0.88rem !important;
  color: var(--primary) !important;
}}

/* Image Previews */
div[data-testid="stImage"] img {{
  border-radius: 10px;
  border: 1px solid var(--border);
  object-fit: contain;
  max-height: 480px;
}}

/* Results & Probability Distribution */
.pred-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
}}
.pred-headline {{
  font-family: 'Manrope', sans-serif;
  font-size: 1.7rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--primary);
  margin: 0.4rem 0 0.15rem 0;
}}
.pred-conf-display {{
  font-family: 'Manrope', sans-serif;
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--primary);
  margin: 0.6rem 0 0.2rem 0;
}}
.badge {{
  display: inline-block;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  border: 1px solid transparent;
}}
.badge-concern {{
  background: rgba(139, 58, 58, 0.12);
  color: var(--concern);
  border-color: rgba(139, 58, 58, 0.25);
}}
.badge-benign {{
  background: rgba(61, 107, 79, 0.12);
  color: var(--benign);
  border-color: rgba(61, 107, 79, 0.25);
}}
.badge-uncertain {{
  background: rgba(109, 121, 114, 0.12);
  color: var(--muted);
  border-color: rgba(109, 121, 114, 0.25);
}}

/* Horizontal Probability Bars */
.prob-item {{
  display: grid;
  grid-template-columns: 13rem 1fr 3.8rem;
  gap: 0.9rem;
  align-items: center;
  margin-bottom: 0.6rem;
}}
.prob-name {{
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.prob-track {{
  height: 9px;
  background: var(--light-accent);
  border-radius: 999px;
  overflow: hidden;
}}
.prob-fill {{
  height: 100%;
  border-radius: 999px;
  background: var(--accent);
  transition: width 0.3s ease;
}}
.prob-fill-top {{
  background: var(--primary);
}}
.prob-fill-concern {{
  background: var(--concern);
}}
.prob-val {{
  font-size: 0.84rem;
  color: var(--muted);
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}}

/* File Metadata */
.metadata-grid {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem 1.4rem;
  font-size: 0.85rem;
  color: var(--muted);
  margin: 0.8rem 0;
}}
.metadata-grid strong {{
  color: var(--primary);
  font-weight: 600;
}}

/* Metric Tiles Grid */
.metrics-summary-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}}
.metric-box {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.15rem 1.25rem;
}}
.metric-box-label {{
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 0.35rem;
}}
.metric-box-num {{
  font-family: 'Manrope', sans-serif;
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--primary);
}}
.metric-box-sub {{
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.25rem;
}}

/* Taxonomy Pills */
.taxonomy-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.38rem 0.8rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--card);
  font-size: 0.8rem;
  color: var(--secondary);
  margin: 0 0.4rem 0.5rem 0;
}}

/* RESPONSIVE MEDIA QUERIES */

/* Tablet (1024px and below) */
@media (max-width: 1024px) {{
  .hero-wrapper {{
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }}
  .hero-visual-box {{
    min-height: 240px;
  }}
  .workflow-grid {{
    grid-template-columns: 1fr;
    gap: 0.9rem;
  }}
  .metrics-summary-grid {{
    grid-template-columns: repeat(2, 1fr);
  }}
  .prob-item {{
    grid-template-columns: 10rem 1fr 3.2rem;
  }}
}}

/* Mobile (768px and below) */
@media (max-width: 768px) {{
  .block-container {{
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }}
  .page-title {{
    font-size: 1.6rem;
  }}
  .metrics-summary-grid {{
    grid-template-columns: 1fr;
    gap: 0.75rem;
  }}
  .prob-item {{
    grid-template-columns: 1fr;
    gap: 0.2rem;
  }}
  .prob-val {{
    text-align: left;
    margin-bottom: 0.4rem;
  }}
  .stButton > button {{
    width: 100% !important;
  }}
}}
</style>
"""