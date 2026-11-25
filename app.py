import base64
import os
import re
import urllib.parse
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MHQ Mental Well-Being Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Plus Jakarta Sans, sans-serif", color="#e2e8f0"),
    margin=dict(l=20, r=20, t=48, b=24),
    hoverlabel=dict(
        bgcolor="#0f172a",
        bordercolor="rgba(45, 212, 191, 0.45)",
        font=dict(color="#f8fafc", family="Inter, sans-serif"),
    ),
)


def style_plotly(fig, **layout_kwargs):
    """Apply cohesive dark-glass styling and subtle gridlines to any Plotly figure."""
    merged = {**PLOTLY_LAYOUT, **layout_kwargs}
    fig.update_layout(**merged)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    return fig

# Custom UI/UX styling: modern dark-glass research dashboard
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    /* Clean Streamlit chrome while keeping sidebar expand/collapse accessible */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    html, .stApp, section.main {
        scroll-behavior: smooth;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #090d16; }
    ::-webkit-scrollbar-thumb { background: rgba(45, 212, 191, 0.28); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(45, 212, 191, 0.5); }

    /* Deep obsidian/slate background */
    .stApp {
        background: radial-gradient(ellipse at 85% 0%, #131b2e 0%, #0b1120 45%, #090d16 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }

    /* Sidebar glass panel */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b111e 0%, #090d16 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Vertical navigation items */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        padding: 0.68rem 0.95rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        background: rgba(255, 255, 255, 0.02);
        color: #cbd5e1;
        font-weight: 500;
        font-size: 0.92rem;
        transition: all 0.2s cubic-bezier(0.22, 1, 0.36, 1);
        margin-bottom: 0.32rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(45, 212, 191, 0.45);
        background: rgba(20, 184, 166, 0.10);
        color: #f8fafc;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(13, 148, 136, 0.28) 0%, rgba(14, 165, 233, 0.16) 100%);
        border: 1px solid rgba(45, 212, 191, 0.55);
        border-left: 3px solid #2dd4bf;
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(13, 148, 136, 0.18);
    }

    /* Secondary buttons */
    [data-testid="stMain"] .stButton > button[kind="secondary"],
    .main .stButton > button[kind="secondary"] {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background: rgba(255, 255, 255, 0.03);
        color: #e2e8f0;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.22, 1, 0.36, 1);
    }
    [data-testid="stMain"] .stButton > button[kind="secondary"]:hover,
    .main .stButton > button[kind="secondary"]:hover {
        border-color: rgba(45, 212, 191, 0.5);
        background: rgba(20, 184, 166, 0.12);
        color: #5eead4;
        transform: translateY(-1px);
    }

    /* Expanders & form inputs */
    div[data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.025) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(45, 212, 191, 0.30) !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* Primary action buttons & link buttons */
    [data-testid="stMain"] .stButton > button[kind="primary"],
    [data-testid="stMain"] .stDownloadButton > button,
    [data-testid="stMain"] div[data-testid="stLinkButton"] a[kind="primary"],
    .main .stButton > button[kind="primary"],
    .main .stDownloadButton > button,
    .main div[data-testid="stLinkButton"] a[kind="primary"] {
        border-radius: 10px;
        padding: 0.58rem 1.2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #0d9488 0%, #0284c7 100%);
        border: 1px solid rgba(45, 212, 191, 0.35);
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.24);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"]:hover,
    [data-testid="stMain"] .stDownloadButton > button:hover,
    [data-testid="stMain"] div[data-testid="stLinkButton"] a[kind="primary"]:hover,
    .main .stButton > button[kind="primary"]:hover,
    .main .stDownloadButton > button:hover,
    .main div[data-testid="stLinkButton"] a[kind="primary"]:hover {
        transform: translateY(-1px);
        filter: brightness(1.06);
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.35);
        color: #ffffff;
    }

    /* Framed Publication Figure Containers (prevents bright white-PNG clash on dark theme) */
    [data-testid="stMain"] [data-testid="stImage"],
    .main [data-testid="stImage"] {
        background: #f8fafc;
        padding: 14px 14px 10px 14px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 10px;
    }
    [data-testid="stMain"] [data-testid="stImage"] img,
    .main [data-testid="stImage"] img {
        border-radius: 8px;
    }
    [data-testid="stMain"] [data-testid="stImageCaption"],
    .main [data-testid="stImageCaption"] {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding-top: 8px !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 2px solid #14b8a6 !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22) !important;
        transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(45, 212, 191, 0.38) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    /* Diagnostic Result Card */
    .diag-card {
        padding: 20px 22px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.26);
        height: 100%;
    }
    .diag-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 12px;
    }
    .diag-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .tier-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .tier-thriving { background: rgba(13, 148, 136, 0.22); border: 1px solid #2dd4bf; color: #5eead4; }
    .tier-succeeding { background: rgba(34, 197, 94, 0.18); border: 1px solid #4ade80; color: #86efac; }
    .tier-managing { background: rgba(234, 179, 8, 0.18); border: 1px solid #facc15; color: #fde047; }
    .tier-struggling { background: rgba(249, 115, 22, 0.20); border: 1px solid #fb923c; color: #fdba74; }
    .tier-distressed { background: rgba(239, 68, 68, 0.22); border: 1px solid #f87171; color: #fca5a5; }
    .diag-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .diag-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .diag-meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.8rem;
    }
    .diag-meta-item span { display: block; color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .diag-meta-item strong { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }

    /* Clean Editorial Hero Header + 4-Column KPI Bento Strip */
    .hero-banner {
        padding: 26px 32px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28);
        margin-bottom: 18px;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(20, 184, 166, 0.14);
        border: 1px solid rgba(45, 212, 191, 0.35);
        color: #5eead4;
        margin-bottom: 12px;
    }
    .hero-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2dd4bf;
        display: inline-block;
    }
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.05rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.025em;
        margin: 0;
        color: #f8fafc;
    }
    .hero-gradient {
        background: linear-gradient(90deg, #f8fafc 0%, #5eead4 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.94rem;
        color: #94a3b8;
        max-width: 820px;
        margin: 8px 0 0;
        line-height: 1.55;
    }

    /* 4-Column KPI Bento Strip */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 26px;
    }
    @media (max-width: 900px) {
        .bento-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .bento-card {
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }
    .bento-lbl {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 4px;
    }
    .bento-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .bento-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Sidebar Brand & Metadata */
    .sidebar-brand-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }
    .sidebar-brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.02rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .sidebar-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 999px;
        background: rgba(20, 184, 166, 0.14);
        border: 1px solid rgba(45, 212, 191, 0.3);
        color: #5eead4;
    }
    .sidebar-spec-box {
        margin-top: 18px;
        padding: 12px 14px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    .spec-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .spec-row:last-child { border-bottom: none; }
    .spec-row span { color: #94a3b8; }
    .spec-row strong { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }

    .section-spacer { height: 18px; }
    .ai-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
        background: rgba(20, 184, 166, 0.16);
        border: 1px solid rgba(45, 212, 191, 0.38);
        color: #5eead4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mhq_model.joblib")
SUMMARY_PATH = os.path.join(BASE_DIR, "demographic_summary.csv")
BINS_PATH = os.path.join(BASE_DIR, "mhq_bins.csv")
INCOME_SUMMARY_PATH = os.path.join(BASE_DIR, "income_summary.csv")

AGE_COL = "age"
SEX_COL = "biological_sex"
INCOME_COL = "income_household"
STUDY_MEAN_MHQ = 67.65
STUDY_MEDIAN_MHQ = 77.6
AGE_ORDER = ["18", "19", "20", "21-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]

INCOME_INR_OPTIONS = [
    "Less than ₹1,00,000",
    "₹1,00,001 – ₹3,00,000",
    "₹3,00,001 – ₹5,00,000",
    "₹5,00,001 – ₹10,00,000",
    "₹10,00,001 – ₹20,00,000",
    "₹20,00,001 – ₹40,00,000",
    "₹40,00,001 – ₹85,00,000",
    "₹85,00,001 – ₹2,10,00,000",
    "More than ₹2,10,00,000",
    "Prefer not to say",
]

INR_TO_MODEL_RAW = {
    "Less than ₹1,00,000": "Less than INR 1|00|000",
    "₹1,00,001 – ₹3,00,000": "1|00|001 – 3|00|000 INR",
    "₹3,00,001 – ₹5,00,000": "3|00|001 – 5|00|000 INR",
    "₹5,00,001 – ₹10,00,000": "5|00|001 – 10|00|000 INR",
    "₹10,00,001 – ₹20,00,000": "10|00|001 – 20|00|000 INR",
    "₹20,00,001 – ₹40,00,000": "20|00|001 – 40|00|000 INR",
    "₹40,00,001 – ₹85,00,000": "More than 40|00|000 INR",
    "₹85,00,001 – ₹2,10,00,000": "$100|001 -$250|000",
    "More than ₹2,10,00,000": "More than $250|000",
    "Prefer not to say": "Prefer not to say",
}


def map_raw_income_to_inr(val: str) -> str:
    s = str(val).strip()
    if s == "Unknown":
        return "Unknown (Not Reported)"
    if s in ("Less than INR 1|00|000", "Up to 3 minimum wages"):
        return "INR < 1,00,000"
    if s.startswith("1|00|001") or s == "3-10 minimum wages":
        return "INR 1,00,001 - 3,00,000"
    if s.startswith("3|00|001") or s == "10-20 minimum wages":
        return "INR 3,00,001 - 5,00,000"
    if s.startswith("5|00|001") or s == "20-30 minimum wages":
        return "INR 5,00,001 - 10,00,000"
    if s.startswith("10|00|001") or s in ("Under $20|000", "More than 30 minimum wages"):
        return "INR 10,00,001 - 20,00,000"
    if s.startswith("20|00|001") or s.startswith("$20|001"):
        return "INR 20,00,001 - 40,00,000"
    if s.startswith("More than 40|00|000") or s.startswith("$40|001") or s.startswith("$60|001") or s.startswith("$80|001"):
        return "INR 40,00,001 - 85,00,000"
    if s.startswith("$100 001") or s.startswith("$100|001"):
        return "INR 85,00,001 - 2,10,00,000"
    if s.startswith("More than $250"):
        return "INR > 2,10,00,000"
    return "Prefer not to say"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_summaries():
    summary_df = pd.read_csv(SUMMARY_PATH) if os.path.exists(SUMMARY_PATH) else None
    bins_df = pd.read_csv(BINS_PATH) if os.path.exists(BINS_PATH) else None
    income_df = pd.read_csv(INCOME_SUMMARY_PATH) if os.path.exists(INCOME_SUMMARY_PATH) else None
    return summary_df, bins_df, income_df


def get_model_categories(model) -> tuple[list[str], list[str], list[str]]:
    age_opts = AGE_ORDER
    sex_opts = ["Female", "Male", "Other", "Prefer not to say"]
    return age_opts, sex_opts, INCOME_INR_OPTIONS


def predict_mhq(model, age: str, biological_sex: str, income_inr_label: str) -> float:
    raw_income = INR_TO_MODEL_RAW.get(str(income_inr_label), str(income_inr_label))
    raw_sex = str(biological_sex)
    input_df = pd.DataFrame(
        {
            AGE_COL: [str(age)],
            SEX_COL: [raw_sex],
            INCOME_COL: [raw_income],
        }
    )
    return float(model.predict(input_df)[0])


def mhq_category(score: float) -> str:
    if score >= 100:
        return "Thriving"
    if score >= 50:
        return "Succeeding"
    if score >= 0:
        return "Managing / Enduring"
    if score >= -50:
        return "Struggling"
    return "Distressed"


def _get_gemini_api_key() -> str | None:
    """Check common secret/env key names safely without raising FileNotFoundError."""
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY"):
        try:
            val = st.secrets.get(key_name)
            if val:
                return str(val).strip()
        except Exception:
            pass
        env_val = os.getenv(key_name)
        if env_val:
            return env_val.strip()
    return None


def get_ai_wellbeing_advice(age: str, sex: str, income: str, predicted_mhq: float, tier_label: str) -> tuple[str, str, list[tuple[str, str]], str]:
    """Generate MHQ-tier-refined tips using Gemini + Google Search Grounding, with a 5-tier research fallback."""
    delta = predicted_mhq - STUDY_MEAN_MHQ
    age_context = (
        "Young adults (18–24) show the widest gap in our dataset (mean 23.12 at age 18 vs. 101.99 at 65–74), "
        "making sleep stability, nutrition, and social routines especially high-leverage."
        if age in ("18", "19", "20", "21-24")
        else (
            "Mid-life cohorts balance career, financial, and family load; structured recovery habits "
            "help protect cognitive clarity and emotional resilience."
            if age in ("25-34", "35-44", "45-54")
            else "Older adult cohorts typically show stronger emotional resilience and lower hopelessness; "
            "consistent movement and community engagement sustain cognitive sharpness."
        )
    )

    tier_guidance = {
        "Distressed": (
            "- **Seek structured professional & peer support early:** Scores below −50 signal high functional distress; connecting with a qualified counselor or trusted support network is the highest-priority step.\n"
            "- **Stabilize core sleep-wake rhythm (+74.4 MHQ pts):** Start with one fixed wake-up time and reduced late-night screen exposure to rebuild baseline daily energy.\n"
            "- **Reduce inflammatory dietary stressors (+68.6 pts):** Swapping daily ultra-processed snacks for whole foods measurably supports mood regulation.",
            f"evidence based mental wellbeing recovery tips high distress age {age}",
        ),
        "Struggling": (
            "- **Rebuild sleep consistency first (+74.4 MHQ pts):** In our 240,012-row dataset, regular sleep lifts energy (5.09 → 6.86) and counters hopelessness.\n"
            "- **Cut daily ultra-processed food intake (+68.6 pts):** Reducing packaged snacks and sugary drinks is strongly linked to steadier daily mood.\n"
            "- **Add low-friction daily movement (+45.6 pts):** Even a 20-minute brisk walk outdoors helps break rumination and restore focus.",
            f"evidence based coping strategies improving low mental wellbeing energy sleep age {age}",
        ),
        "Managing / Enduring": (
            "- **Upgrade sleep from inconsistent to regular (+74.4 MHQ pts):** Moving into consistent restorative sleep is the single largest modifiable lever in our MHQ dataset.\n"
            "- **Audit ultra-processed food frequency (+68.6 pts):** Replacing daily UPF meals with minimally processed foods lifts cognitive clarity and mood.\n"
            "- **Schedule weekly in-person social time (+40.9 pts) & daily exercise (+45.6 pts):** Because Social Self is the least predictable domain from demographics (R² = 0.1013), intentional social habits create outsized gains.",
            f"science backed daily habits to move from managing to thriving mental wellbeing age {age}",
        ),
        "Succeeding": (
            "- **Protect sleep & recovery boundaries (+74.4 MHQ pts):** Maintain consistent sleep timing to keep cognitive performance and drive high.\n"
            "- **Deepen active exercise & whole-food nutrition (+45.6 to +68.6 pts):** Combining daily movement with low-UPF nutrition sustains high energy (7.06 mean).\n"
            "- **Invest in meaningful social connectedness (+40.9 pts):** Strong interpersonal ties buffer against unexpected life stressors.",
            f"evidence based strategies sustain high mental wellbeing cognitive resilience age {age}",
        ),
        "Thriving": (
            "- **Sustain your core protective habits:** Continue prioritizing consistent sleep, whole-food nutrition, and daily physical activity to preserve your thriving baseline.\n"
            "- **Channel energy into purpose & mentoring (+40.9 pts):** High-wellbeing cohorts benefit from community contribution and deep social ties.\n"
            "- **Monitor early stress signals:** Periodic check-ins help maintain resilience during major life or career transitions.",
            f"positive psychology habits to maintain thriving mental wellbeing longevity age {age}",
        ),
    }

    tier_bullets, search_query = tier_guidance.get(tier_label, tier_guidance["Managing / Enduring"])
    google_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(search_query)}"

    default_sources = [
        ("Global Mind Project: Causal Mental Health Findings (PMC)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10910083/"),
        ("MHQ Validation Study — JMIR Mental Health (PMC)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9069309/"),
        ("Machine Learning in Human Wellbeing — Scientific Reports", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11723942/"),
    ]

    fallback = (
        f"**Profile Context ({age} · {sex} · {income}):** Predicted MHQ is **{predicted_mhq:.1f}** "
        f"({delta:+.1f} vs. study mean 67.65; **{tier_label}** tier). {age_context}\n\n"
        f"**Refined {tier_label}-Tier Action Plan (N = 240,012):**\n{tier_bullets}"
    )

    api_key = _get_gemini_api_key()
    if not api_key:
        return fallback, "Evidence-Based Research Guide", default_sources, google_search_url

    try:
        from google import genai
        from google.genai import types

        prompt = (
            "You are a concise, evidence-based mental well-being research assistant for an MHQ dashboard (N=240,012). "
            f"User Profile: Age {age}, Biological Sex {sex}, Household Income {income}. "
            f"Predicted MHQ Score: {predicted_mhq:.1f} ({delta:+.1f} vs. study mean 67.65; {tier_label} tier on -100 to +200 scale). "
            "Use Google Search to retrieve current, reputable public-health or peer-reviewed psychology guidance specifically suited for "
            f"an adult aged {age} in the '{tier_label}' mental well-being tier. "
            "Return a crisp, non-clinical 3-bullet action plan (under 140 words) that blends live evidence with our dataset's top levers "
            "(sleep consistency +74.4 pts, limiting ultra-processed foods +68.6 pts, daily exercise +45.6 pts, socializing +40.9 pts)."
        )
        client = genai.Client(api_key=api_key)

        # Primary: Gemini 2.5 Flash grounded with live Google Search
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            if response and getattr(response, "text", None):
                grounded_links = []
                try:
                    cand = response.candidates[0]
                    meta = getattr(cand, "grounding_metadata", None)
                    chunks = getattr(meta, "grounding_chunks", None) or []
                    for ch in chunks:
                        web = getattr(ch, "web", None)
                        if web and getattr(web, "uri", None):
                            title = getattr(web, "title", None) or "Google Search Source"
                            grounded_links.append((title, web.uri))
                except Exception:
                    pass
                final_links = grounded_links[:4] if grounded_links else default_sources
                return response.text.strip(), "Gemini + Google Search Grounded", final_links, google_search_url
        except Exception:
            # Graceful retry without grounding tool if key tier restricts search tool
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and getattr(response, "text", None):
                return response.text.strip(), "Gemini Research Assistant", default_sources, google_search_url

        return fallback, "Evidence-Based Research Guide", default_sources, google_search_url
    except Exception:
        return fallback, "Evidence-Based Research Guide", default_sources, google_search_url


def ask_research_paper_ai(question: str) -> tuple[str, bool]:
    """Answer examiner or user questions about the MHQ study via Gemini API or local research knowledge engine."""
    q = question.lower().strip()
    if any(k in q for k in ("r2", "r²", "accuracy", "13.48", "13.5", "weak", "low", "rmse")):
        fallback = (
            "**Why is R² = 0.1348 (13.48%)?** Using only 3 static demographic inputs (Age, Biological Sex, Household Income), "
            "our out-of-sample R² of **0.1348** (RMSE = **67.24**) sits directly inside the **0.10–0.15 benchmark** established by "
            "Oparina et al. (2025) in *Nature Scientific Reports* across over 1 million respondents. It demonstrates that demographics "
            "set a real population baseline, while ~86.5% of individual well-being is driven by modifiable lifestyle, sleep, and social factors."
        )
    elif any(k in q for k in ("age", "income", "sex", "strongest", "important", "anova", "chi")):
        fallback = (
            "**Which factor matters most?** **Age** is by far the strongest demographic predictor: One-Way ANOVA gives "
            "**Age F = 3,176.39** (p < 0.0001), which is **5.4× larger than Biological Sex (F = 586.09)** and **48.1× larger than "
            "Household Income (F = 66.07)**. The top 3 Random Forest features (`age_18`, `age_65-74`, `age_55-64`) account for **57.7%** "
            "of model importance, and Chi-Square (χ² = 1,666.66, df = 60, p < 0.0001) proves Biological Sex and Income are statistically dependent."
        )
    elif any(k in q for k in ("86.5", "remaining", "lifestyle", "direct", "indirect", "sleep", "food", "upf", "exercise")):
        fallback = (
            "**What drives the remaining 86.5%?** Direct negative symptom correlates include sadness/hopelessness (r = −0.72), "
            "confusion (r = −0.72), and daily ultra-processed food (**−68.6 MHQ pts**). Indirect lifestyle pathways—consistent sleep "
            "(**+74.4 pts** via higher energy), employment stability (**+76.7 pts**), daily exercise (**+45.6 pts**), and socializing "
            "(**+40.9 pts**)—explain far more variation than static demographics alone."
        )
    elif any(k in q for k in ("loneliness", "inventory", "medical", "permut", "future", "social self", "li")):
        fallback = (
            "**Future Scope: Loneliness Inventory (LI) Cross-Scale Permutation Validation:** "
            "While `social_self_score` exhibits the strongest direct Pearson correlation with overall MHQ (r = +0.9136), "
            "it has the lowest demographic predictability (R² = 0.1013, RMSE = 72.52). Young adults aged 18 have "
            "the lowest baseline MHQ (23.12 vs. 101.99 for ages 65–74; +78.9 pt generational gap), while frequent socializing provides a +40.9 point gain. "
            "To bridge this gap, proposed future work co-administers the 19-item Loneliness Inventory (LI; Uma, Meenakshi R. under the guidance of Prof. Krishnan K., 2010, Psychotronics Publications) "
            "with MHQ across medical student cohorts (ages 18–24), applies 5→1 reverse-scoring to the 8 connectedness/support items (Items 1, 5, 6, 9, 14, 16, 18, 19; theoretical range 19–95), "
            "and conducts 10,000-iteration non-parametric permutation tests to derive exact distribution-free p-values for cohorts of n = 100–500 without assuming bivariate normality."
        )
    else:
        fallback = (
            "**MHQ Study Summary (N = 240,012 · 207 Variables):** Mean MHQ is **67.65** (SD = **71.97**, scale −100 to +200). "
            "One-Way ANOVA confirms significant gaps by Age (F = 3,176.39), Biological Sex (F = 586.09), and INR Income (F = 66.07, "
            "peaking at INR 85L–2.1Cr = 108.33 before dipping to 90.99 above 2.1Cr). Our 100-tree Random Forest achieves **R² = 0.1348** "
            "overall, with Cognition (**R² = 0.1356**) most predictable and Social Self (**R² = 0.1013**) least predictable."
        )

    api_key = _get_gemini_api_key()
    if not api_key:
        return fallback, False

    try:
        from google import genai
        context_prompt = (
            "You are the AI research assistant for Hitesh Rajpal's MHQ Research & Well-Being Dashboard (N=240,012 rows, 207 columns). "
            "Verified study facts: Mean MHQ=67.65 (SD=71.97, range -100 to +200). Random Forest (100 trees, max_depth=10, 80/20 split): "
            "Overall R²=0.1348 (RMSE=67.24), Cognition R²=0.1356, Adapt & Resilience R²=0.1281, Drive & Motivation R²=0.1137, Social Self R²=0.1013. "
            "Oparina et al. (2025, Nature Scientific Reports) benchmark: restricted demographic models yield R²=0.10-0.15. "
            "ANOVA: Age F=3,176.39, Biological Sex F=586.09, Income F=66.07 (all p<0.0001). Chi-Square (Sex x Income): χ²=1,666.66, df=60, p=7.26e-309. "
            "Top features: age_18 (25.98%), age_65-74 (16.60%), age_55-64 (15.12%). Raw gender was 100% null -> replaced with biological_sex (239,995 valid). "
            "INR income peaks at 85L-2.1Cr (108.33) and dips to 90.99 (>2.1Cr). Direct correlates: social_self (+0.91), sad_hopeless (-0.72), daily UPF (-68.6 pts). "
            "Indirect pathways: Age 18->65-74 (+78.9 pts), Sleep (+74.4 pts), Employment (+76.7 pts), Exercise (+45.6 pts), Socializing (+40.9 pts). "
            "Future Scope extension: Cross-Scale Permutation Validation co-administers the 19-item Loneliness Inventory (LI; Uma, Meenakshi R. under the guidance of Prof. Krishnan K., 2010, Psychotronics Publications) "
            "in medical college cohorts (ages 18-24). Rationale: social_self_score has strongest direct correlation with MHQ (r=+0.9136) but lowest demographic R² (0.1013, RMSE=72.52); "
            "age 18 has lowest mean MHQ (23.12); socializing yields +40.9 pts. LI items: 11 direct isolation items (1->5) and 8 positive connectedness items reverse-scored "
            "(5->1; items 1,5,6,9,14,16,18,19), yielding theoretical range 19-95. Protocol executes 10,000 non-parametric permutation iterations for exact distribution-free "
            "p-values in institutional cohorts (n=100-500) and SHAP explainability to target Oparina et al.'s ~0.30 survey ceiling. "
            f"Answer this question concisely in 3-4 sentences using ONLY these verified facts: {question}"
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=context_prompt,
        )
        if response and getattr(response, "text", None):
            return response.text.strip(), True
        return fallback, False
    except Exception:
        return fallback, False


model = load_model()
if model is None:
    st.error("`mhq_model.joblib` was not found in the app directory.")
    st.stop()

summary_df, bins_df, income_df = load_summaries()
age_options, sex_options, income_options = get_model_categories(model)

@st.cache_data(show_spinner=False)
def load_research_paper_html(paper_path: str, assets_dir: str, cache_key: float = 0.0) -> str:
    """Load Research_Paper_Interactive.html and inline local assets as base64 data URIs for iframe viewing."""
    with open(paper_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    def _replace_src(match: re.Match) -> str:
        rel_path = match.group(1)
        img_path = os.path.join(assets_dir, os.path.basename(rel_path))
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
            return f'src="data:image/png;base64,{b64}"'
        return match.group(0)

    return re.sub(r'src="(assets/[^"]+\.png)"', _replace_src, html_str)


if st.query_params.get("view") == "paper":
    paper_html_path = os.path.join(BASE_DIR, "Research_Paper_Interactive.html")
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        footer { display: none !important; }
        .stApp,
        [data-testid="stAppViewContainer"],
        section[data-testid="stMain"],
        section.main {
            background: #f7f9fc !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stMainBlockContainer"],
        .main .block-container,
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
            width: 100vw !important;
        }
        section[data-testid="stMain"] iframe,
        section.main iframe,
        iframe {
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            min-height: 100vh !important;
            border: none !important;
            margin: 0 !important;
            padding: 0 !important;
            z-index: 999999 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if os.path.exists(paper_html_path):
        full_paper_html = load_research_paper_html(
            paper_html_path,
            os.path.join(BASE_DIR, "assets"),
            os.path.getmtime(paper_html_path),
        )
        components.html(full_paper_html, height=1000, scrolling=True)
    else:
        st.error("Research_Paper_Interactive.html was not found in the repository.")
    st.stop()


# Vertical Sidebar Navigation Panel
NAV_ITEMS = [
    ("Live Prediction", "prediction"),
    ("Demographic Explorer", "explorer"),
    ("Model & Statistics", "model"),
    ("Research Paper", "paper"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "prediction"
if "rebuild_nonce" not in st.session_state:
    st.session_state.rebuild_nonce = 1

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-title">MHQ Research Studio</div>
            <div class="sidebar-brand-sub">Hitesh Rajpal · Python Mini Project</div>
            <div class="sidebar-status-pill"><span class="hero-dot"></span>Model v2.4 · Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("WORKSPACE")
    for label, key in NAV_ITEMS:
        btn_type = "primary" if st.session_state.active_page == key else "secondary"
        if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
            st.session_state.active_page = key
            st.session_state.rebuild_nonce += 1
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-spec-box">
            <div class="spec-row"><span>Dataset</span><strong>240,012 Rows</strong></div>
            <div class="spec-row"><span>Variables</span><strong>207 Features</strong></div>
            <div class="spec-row"><span>MHQ Scale</span><strong>−100 to +200</strong></div>
            <div class="spec-row"><span>Study Mean / Med</span><strong>67.65 / 77.6</strong></div>
            <div class="spec-row"><span>Random Forest R²</span><strong>0.1348</strong></div>
            <div class="spec-row"><span>Benchmark</span><strong>Oparina et al.</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    with st.expander("Display Settings", expanded=False):
        enable_swarm = st.toggle("Page Rebuild Overlay", value=False)
        if st.button("Replay Animation", use_container_width=True):
            st.session_state.rebuild_nonce += 1
            st.rerun()

# Editorial Research Header & 4-Column KPI Bento Strip
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-eyebrow"><span class="hero-dot"></span>Empirical Study · Global Mind Dataset (N = 240,012)</div>
        <h1 class="hero-title"><span class="hero-gradient">Mental Health Quotient (MHQ) Explorer</span></h1>
        <p class="hero-subtitle">
            Demographic predictability modeling (Random Forest, R² = 0.1348) and behavioral mediation analysis across 207 variables, paired with a Loneliness Inventory (LI) cross-validation protocol.
        </p>
    </div>
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-lbl">Dataset Scale</div>
            <div class="bento-val">240,012</div>
            <div class="bento-sub">Adult records · 207 variables</div>
        </div>
        <div class="bento-card">
            <div class="bento-lbl">Model Accuracy (R²)</div>
            <div class="bento-val">0.1348</div>
            <div class="bento-sub">13.48% variance · RMSE 67.24</div>
        </div>
        <div class="bento-card">
            <div class="bento-lbl">Study Baseline</div>
            <div class="bento-val">67.65 Mean</div>
            <div class="bento-sub">Median 77.6 · Scale −100 to +200</div>
        </div>
        <div class="bento-card">
            <div class="bento-lbl">Primary Predictor</div>
            <div class="bento-val">Age Cohort</div>
            <div class="bento-sub">ANOVA F = 3,176.39 · p &lt; 0.0001</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active = st.session_state.active_page

def render_prediction_results(selected_age: str, selected_sex: str, selected_income: str, key_prefix: str = "pred"):
    pred = predict_mhq(model, selected_age, selected_sex, selected_income)
    delta_mean = pred - STUDY_MEAN_MHQ
    delta_med = pred - STUDY_MEDIAN_MHQ
    tier = mhq_category(pred)

    percentile = 50.0
    if bins_df is not None and not bins_df.empty:
        tot = bins_df["count"].sum()
        below = bins_df.loc[bins_df["bin_center"] <= pred, "count"].sum()
        percentile = (below / tot) * 100 if tot > 0 else 50.0

    cohort_n = 0
    if summary_df is not None and not summary_df.empty:
        cohort_n = int(summary_df.loc[summary_df["age"] == selected_age, "count"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted MHQ", f"{pred:.2f}", f"{delta_mean:+.2f} vs. Mean")
    m2.metric("Well-Being Tier", tier)
    m3.metric("Study Percentile", f"{percentile:.0f}th")
    m4.metric("Age Cohort Size", f"{cohort_n:,}")

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    g_col, a_col = st.columns([1.45, 1.15])

    with g_col:
        with st.container(border=True):
            if bins_df is not None and not bins_df.empty:
                st.markdown("**Position in Population MHQ Distribution (`N = 240,012`)**")
                dist_df = bins_df.groupby("bin_center", as_index=False)["count"].sum().sort_values("bin_center")
                fig_pos = go.Figure()
                fig_pos.add_trace(
                    go.Scatter(
                        x=dist_df["bin_center"],
                        y=dist_df["count"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#14b8a6", width=2.2),
                        fillcolor="rgba(20, 184, 166, 0.16)",
                        name="Population",
                        hovertemplate="MHQ Score: %{x:.0f}<br>Respondents: %{y:,}<extra></extra>",
                    )
                )
                max_y = float(dist_df["count"].max()) if not dist_df.empty else 1000.0
                for label_txt, x_val, col_hex, dash_style in [
                    (f"Prediction ({pred:.1f})", pred, "#2dd4bf", "solid"),
                    (f"Mean ({STUDY_MEAN_MHQ:.2f})", STUDY_MEAN_MHQ, "#f59e0b", "dash"),
                    (f"Median ({STUDY_MEDIAN_MHQ:.1f})", STUDY_MEDIAN_MHQ, "#818cf8", "dot"),
                ]:
                    fig_pos.add_trace(
                        go.Scatter(
                            x=[x_val, x_val],
                            y=[0, max_y * 1.05],
                            mode="lines",
                            line=dict(color=col_hex, width=2.5 if dash_style == "solid" else 2, dash=dash_style),
                            name=label_txt,
                            hovertemplate=f"{label_txt}<extra></extra>",
                        )
                    )
                style_plotly(
                    fig_pos,
                    height=275,
                    margin=dict(l=16, r=16, t=44, b=44),
                    xaxis=dict(range=[-100, 200], tickvals=[-100, -50, 0, 50, 100, 150, 200], title="MHQ Score (−100 to +200)"),
                    yaxis=dict(title="Respondents", showticklabels=True),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_pos, use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    f"**How to Explain:** The bimodal curve reflects all 240,012 scores (negative scores compress near 0 under MHQ scaling). "
                    f"This profile (**{pred:.2f}**) sits at the **{percentile:.0f}th percentile** vs. Mean (**67.65**) and Median (**77.6**)."
                )
            else:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pred,
                        number={"valueformat": ".1f", "font": {"size": 38, "color": "#f8fafc", "family": "JetBrains Mono, monospace"}},
                        title={"text": f"MHQ Scale Position · {tier}", "font": {"size": 14, "color": "#e2e8f0"}},
                        gauge={
                            "axis": {"range": [-100, 200], "tickvals": [-100, -50, 0, 50, 100, 150, 200], "tickcolor": "#94a3b8"},
                            "bar": {"color": "#2dd4bf", "thickness": 0.3},
                            "bgcolor": "rgba(255,255,255,0.03)",
                            "threshold": {"line": {"color": "#f59e0b", "width": 3}, "thickness": 0.85, "value": STUDY_MEAN_MHQ},
                        },
                    )
                )
                style_plotly(fig_gauge, height=280, margin=dict(l=24, r=24, t=48, b=24))
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with a_col:
        if pred >= 100:
            cls, msg = "tier-thriving", "Strong emotional resilience, cognitive clarity, and sustained life satisfaction."
        elif pred >= 50:
            cls, msg = "tier-succeeding", "Functioning well across daily work, interpersonal relationships, and baseline mood."
        elif pred >= 0:
            cls, msg = "tier-managing", "Coping with everyday demands, though vulnerable to periodic stress or fatigue."
        elif pred >= -50:
            cls, msg = "tier-struggling", "Elevated emotional or cognitive strain affecting day-to-day functioning."
        else:
            cls, msg = "tier-distressed", "Severe symptom burden on the MHQ continuum; structured clinical support is recommended."
        st.markdown(
            f"""
            <div class="diag-card">
                <div class="diag-header">
                    <span class="diag-eyebrow">Diagnostic Profile Readout</span>
                    <span class="tier-pill {cls}">{tier}</span>
                </div>
                <div class="diag-score">{pred:.2f}</div>
                <div class="diag-desc">{msg}</div>
                <div class="diag-meta-grid">
                    <div class="diag-meta-item"><span>Profile</span><strong>{selected_age} · {selected_sex}</strong></div>
                    <div class="diag-meta-item"><span>Income Bracket</span><strong>{selected_income}</strong></div>
                    <div class="diag-meta-item"><span>vs. Mean (67.65)</span><strong>{delta_mean:+.2f} pts</strong></div>
                    <div class="diag-meta-item"><span>vs. Median (77.6)</span><strong>{delta_med:+.2f} pts</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    tab_traj, tab_guide = st.tabs(["Life-Course Age Trajectory", "Evidence-Based Well-Being Guidance"])

    with tab_traj:
        with st.container(border=True):
            st.markdown(f"**Predicted MHQ Across Life Course (`{selected_sex}` · `{selected_income}`)**")
            traj_scores = [predict_mhq(model, age_lbl, selected_sex, selected_income) for age_lbl in AGE_ORDER]
            traj_df = pd.DataFrame({"Age Bracket": AGE_ORDER, "Predicted MHQ": traj_scores})
            fig_traj = go.Figure()
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["Age Bracket"],
                    y=traj_df["Predicted MHQ"],
                    mode="lines+markers",
                    line=dict(color="#38bdf8", width=2.5),
                    marker=dict(size=7, color="#38bdf8"),
                    name="Cohort Trajectory",
                    hovertemplate="Age %{x}<br>Predicted MHQ: %{y:.2f}<extra></extra>",
                )
            )
            fig_traj.add_trace(
                go.Scatter(
                    x=[selected_age],
                    y=[pred],
                    mode="markers",
                    marker=dict(size=13, color="#2dd4bf", line=dict(color="#ffffff", width=2)),
                    name=f"Selected ({selected_age}: {pred:.2f})",
                    hovertemplate=f"Selected ({selected_age}): {pred:.2f}<extra></extra>",
                )
            )
            fig_traj.add_hline(
                y=STUDY_MEAN_MHQ,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Study Mean ({STUDY_MEAN_MHQ:.2f})",
                annotation_position="bottom right",
            )
            y_min = min(-10.0, min(traj_scores) - 15.0)
            y_max = max(125.0, max(traj_scores) + 15.0)
            style_plotly(
                fig_traj,
                height=280,
                margin=dict(l=16, r=16, t=42, b=44),
                xaxis=dict(title="Age Cohort"),
                yaxis=dict(title="Predicted MHQ", range=[y_min, y_max]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_traj, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "**How to Explain:** Holding Biological Sex and Income constant, the model shows how predicted MHQ rises steadily with age—demonstrating "
                "why Age brackets carry **57.7%** of total Random Forest importance (`F = 3,176.39`, `p < 0.0001`)."
            )

    with tab_guide:
        if st.button("Generate Evidence-Based Well-Being Guidance", type="primary", key=f"ai_btn_{key_prefix}_{selected_age}_{selected_sex}_{selected_income}"):
            with st.spinner("Synthesizing tier-specific guidance and research sources..."):
                advice_text, badge_label, source_links, google_search_url = get_ai_wellbeing_advice(
                    selected_age, selected_sex, selected_income, pred, tier
                )
            with st.container(border=True):
                st.markdown(f'<div class="ai-badge">{badge_label}</div>', unsafe_allow_html=True)
                st.markdown(advice_text)
                if source_links:
                    links_md = " · ".join(f"[{t}]({u})" for t, u in source_links)
                    st.caption(f"**References:** {links_md}")
                st.link_button(
                    f"Search Peer-Reviewed Literature for {tier} Tier (Age {selected_age})",
                    google_search_url,
                )


if active == "prediction":
    if "pred_mode" not in st.session_state:
        st.session_state.pred_mode = "custom"
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = "young"

    st.subheader("Live MHQ Prediction Workspace")
    m_col1, m_col2, m_col3 = st.columns(3)
    modes = [
        ("Quick Demo Presets", "presets"),
        ("Custom Prediction", "custom"),
        ("Compare Profiles (A vs. B)", "compare"),
    ]
    for col_box, (m_label, m_key) in zip([m_col1, m_col2, m_col3], modes):
        with col_box:
            if st.button(
                m_label,
                key=f"mode_{m_key}",
                type="primary" if st.session_state.pred_mode == m_key else "secondary",
                use_container_width=True,
            ):
                st.session_state.pred_mode = m_key
                st.rerun()

    st.divider()

    if st.session_state.pred_mode == "presets":
        st.markdown("### Quick Demo Presets")
        st.caption("Select a representative cohort profile to evaluate its baseline MHQ score and tier.")
        PRESETS = {
            "young": ("Young Adult (18, Female)", "18", "Female", "Less than ₹1,00,000"),
            "mid": ("Mid-Career (35-44, Male)", "35-44", "Male", "₹10,00,001 – ₹20,00,000"),
            "older": ("Older Adult (65-74, Male)", "65-74", "Male", "₹20,00,001 – ₹40,00,000"),
        }
        p1, p2, p3 = st.columns(3)
        for p_col, (p_key, (p_label, _, _, _)) in zip([p1, p2, p3], PRESETS.items()):
            with p_col:
                if st.button(
                    p_label,
                    key=f"preset_{p_key}",
                    type="primary" if st.session_state.active_preset == p_key else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.active_preset = p_key
                    st.rerun()

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        _, p_age, p_sex, p_inc = PRESETS[st.session_state.active_preset]
        render_prediction_results(p_age, p_sex, p_inc, key_prefix="preset")

    elif st.session_state.pred_mode == "custom":
        st.markdown("### Custom Demographic Prediction")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_age = st.selectbox(
                    "Age Group",
                    age_options,
                    index=0,
                    key="custom_age",
                    help="Respondent age bracket; accounts for 57.7% of Random Forest feature importance.",
                )
            with col2:
                selected_sex = st.selectbox(
                    "Biological Sex",
                    sex_options,
                    index=0,
                    key="custom_sex",
                    help="Used instead of the raw 'gender' column (which had 100% missingness across all 240,012 rows).",
                )
            with col3:
                selected_income = st.selectbox(
                    "Household Income (INR)",
                    income_options,
                    index=0,
                    key="custom_inc",
                    help="Annual household income normalized to 10 Indian Rupee (INR) brackets.",
                )

            if st.button("Predict MHQ Score", type="primary"):
                st.session_state.custom_pred_profile = (selected_age, selected_sex, selected_income)

        if st.session_state.get("custom_pred_profile"):
            c_age, c_sex, c_inc = st.session_state.custom_pred_profile
            render_prediction_results(c_age, c_sex, c_inc, key_prefix="custom")

    elif st.session_state.pred_mode == "compare":
        st.markdown("### Compare Two Profiles (Profile A vs. Profile B)")
        with st.container(border=True):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Profile A**")
                age_a = st.selectbox("Age Group (A)", age_options, index=0, key="cmp_age_a", help="Select age bracket for Profile A.")
                sex_a = st.selectbox("Biological Sex (A)", sex_options, index=0, key="cmp_sex_a", help="Select biological sex for Profile A.")
                inc_a = st.selectbox("Household Income (A)", income_options, index=0, key="cmp_inc_a", help="Select INR income bracket for Profile A.")
            with cb:
                st.markdown("**Profile B**")
                age_b = st.selectbox("Age Group (B)", age_options, index=min(8, len(age_options) - 1), key="cmp_age_b", help="Select age bracket for Profile B.")
                sex_b = st.selectbox("Biological Sex (B)", sex_options, index=min(1, len(sex_options) - 1), key="cmp_sex_b", help="Select biological sex for Profile B.")
                inc_b = st.selectbox("Household Income (B)", income_options, index=min(4, len(income_options) - 1), key="cmp_inc_b", help="Select INR income bracket for Profile B.")

            if st.button("Compare Profile A vs. Profile B", type="primary"):
                st.session_state.cmp_profiles = ((age_a, sex_a, inc_a), (age_b, sex_b, inc_b))

        if st.session_state.get("cmp_profiles"):
            (age_a, sex_a, inc_a), (age_b, sex_b, inc_b) = st.session_state.cmp_profiles
            score_a = predict_mhq(model, age_a, sex_a, inc_a)
            score_b = predict_mhq(model, age_b, sex_b, inc_b)
            diff_ba = score_b - score_a

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Profile A Predicted MHQ", f"{score_a:.2f}", mhq_category(score_a))
            cm2.metric("Profile B Predicted MHQ", f"{score_b:.2f}", mhq_category(score_b))
            cm3.metric("Difference (B − A)", f"{diff_ba:+.2f} pts", "Profile B Higher" if diff_ba >= 0 else "Profile A Higher")

            cmp_df = pd.DataFrame(
                {
                    "Profile": [f"Profile A ({age_a}, {sex_a})", f"Profile B ({age_b}, {sex_b})", "Study Mean"],
                    "MHQ Score": [round(score_a, 2), round(score_b, 2), STUDY_MEAN_MHQ],
                    "Tier": [mhq_category(score_a), mhq_category(score_b), mhq_category(STUDY_MEAN_MHQ)],
                }
            )
            fig_cmp = px.bar(
                cmp_df,
                x="MHQ Score",
                y="Profile",
                orientation="h",
                text="MHQ Score",
                hover_data=["Tier"],
                color="Profile",
                color_discrete_sequence=["#38bdf8", "#14b8a6", "#f59e0b"],
                title="Side-by-Side MHQ Score Comparison",
            )
            fig_cmp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            style_plotly(fig_cmp, showlegend=False, height=280, xaxis=dict(range=[-100, 215], title="Predicted Overall MHQ Score (−100 to +200)"), yaxis_title="")
            st.plotly_chart(fig_cmp, use_container_width=True)

elif active == "explorer":
    st.subheader("MHQ Score Distribution & Demographic Trends")
    if summary_df is None or bins_df is None:
        st.info("Summary files (`demographic_summary.csv`, `mhq_bins.csv`) not found.")
    else:
        with st.container(border=True):
            f1, f2 = st.columns(2)
            with f1:
                age_filter = st.selectbox(
                    "Filter Age Group",
                    ["All"] + age_options,
                    help="Filter the MHQ distribution histogram and weighted summary statistics by age bracket.",
                )
            with f2:
                sex_filter = st.selectbox(
                    "Filter Biological Sex",
                    ["All"] + sex_options,
                    help="Filter respondents by reported biological sex.",
                )

        filt_sum = summary_df.copy()
        filt_bins = bins_df.copy()
        if age_filter != "All":
            filt_sum = filt_sum[filt_sum["age"] == age_filter]
            filt_bins = filt_bins[filt_bins["age"] == age_filter]
        if sex_filter != "All":
            raw_sex_f = ["Other", "Other/Intersex"] if sex_filter == "Other" else ([sex_filter, "Männlich"] if sex_filter == "Male" else [sex_filter])
            filt_sum = filt_sum[filt_sum["biological_sex"].isin(raw_sex_f)]
            filt_bins = filt_bins[filt_bins["biological_sex"].isin(raw_sex_f)]

        total_n = int(filt_sum["count"].sum())
        weighted_mean = (
            float((filt_sum["mean_mhq"] * filt_sum["count"]).sum() / total_n) if total_n > 0 else np.nan
        )
        weighted_median = (
            float((filt_sum["median_mhq"] * filt_sum["count"]).sum() / total_n) if total_n > 0 else np.nan
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Filtered Respondents", f"{total_n:,}")
        c2.metric("Mean MHQ Score", f"{weighted_mean:.2f}" if total_n > 0 else "N/A")
        c3.metric("Median MHQ Score", f"{weighted_median:.2f}" if total_n > 0 else "N/A")

        if not filt_bins.empty:
            hist_agg = filt_bins.groupby("bin_center", as_index=False)["count"].sum()
            fig_hist = px.bar(
                hist_agg,
                x="bin_center",
                y="count",
                title=f"MHQ Score Distribution (Age: {age_filter}, Sex: {sex_filter})",
                labels={"bin_center": "Overall MHQ Score", "count": "Respondents"},
                color_discrete_sequence=["#14b8a6"],
            )
            if not np.isnan(weighted_mean):
                fig_hist.add_vline(
                    x=weighted_mean,
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text=f"Group Mean ({weighted_mean:.1f})",
                    annotation_position="top right",
                )
            style_plotly(fig_hist, height=360, bargap=0.06)
            st.plotly_chart(fig_hist, use_container_width=True)

        age_grp = (
            summary_df[summary_df["age"].isin(AGE_ORDER)]
            .groupby("age", as_index=False)
            .apply(
                lambda g: pd.Series({
                    "Overall MHQ": (g["mean_mhq"] * g["count"]).sum() / g["count"].sum(),
                    "Cognition": (g["mean_cognition"] * g["count"]).sum() / g["count"].sum(),
                    "Adapt & Resilience": (g["mean_resilience"] * g["count"]).sum() / g["count"].sum(),
                    "Drive & Motivation": (g["mean_motivation"] * g["count"]).sum() / g["count"].sum(),
                    "Social Self": (g["mean_social"] * g["count"]).sum() / g["count"].sum(),
                }),
                include_groups=False,
            )
        )
        age_grp["age"] = pd.Categorical(age_grp["age"], categories=AGE_ORDER, ordered=True)
        age_grp = age_grp.sort_values("age")

        g1, g2 = st.columns(2)
        with g1:
            fig_age = px.bar(
                age_grp,
                x="age",
                y="Overall MHQ",
                color="Overall MHQ",
                color_continuous_scale="Tealgrn",
                title="Mean Overall MHQ Score by Age Group",
                labels={"age": "Age Group", "Overall MHQ": "Mean MHQ Score"},
            )
            fig_age.add_hline(
                y=STUDY_MEAN_MHQ,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Study Mean ({STUDY_MEAN_MHQ:.1f})",
            )
            style_plotly(fig_age, height=390, coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)

        with g2:
            st.markdown("**All 5 MHQ Dimensions Across Age Groups**")
            melted = age_grp.melt(id_vars="age", var_name="Dimension", value_name="Mean Score")
            fig_sub = px.line(
                melted,
                x="age",
                y="Mean Score",
                color="Dimension",
                markers=True,
                color_discrete_sequence=["#2dd4bf", "#38bdf8", "#818cf8", "#a78bfa", "#34d399"],
                labels={"age": "Age Group", "Mean Score": "Mean Score"},
            )
            style_plotly(
                fig_sub,
                height=360,
                margin=dict(l=20, r=20, t=54, b=44),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="", font=dict(size=10)),
            )
            st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar": False})

        if income_df is not None and not income_df.empty:
            fig_inc = px.bar(
                income_df,
                x="mean_mhq",
                y="income_inr",
                orientation="h",
                color="mean_mhq",
                color_continuous_scale="Teal",
                hover_data={"count": ":,", "median_mhq": ":.1f", "mean_mhq": ":.2f"},
                title="Mean Overall MHQ Score by Household Income (INR)",
                labels={"mean_mhq": "Mean Overall MHQ Score", "income_inr": "Household Income (INR)", "count": "Respondents", "median_mhq": "Median MHQ"},
            )
            fig_inc.add_vline(
                x=STUDY_MEAN_MHQ,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Study Mean ({STUDY_MEAN_MHQ:.1f})",
            )
            style_plotly(fig_inc, height=420, coloraxis_showscale=False, yaxis=dict(categoryorder="array", categoryarray=income_df["income_inr"].tolist()[::-1]))
            st.plotly_chart(fig_inc, use_container_width=True)

elif active == "model":
    st.subheader("Feature Importance, Model Performance & Statistical Findings")
    all_features = model.named_steps["preprocessor"].get_feature_names_out()
    clean_features = []
    for name in all_features:
        c_name = name.replace("cat__", "")
        if c_name.startswith("income_household_"):
            raw_val = c_name.replace("income_household_", "")
            c_name = "income_household_" + map_raw_income_to_inr(raw_val)
        clean_features.append(c_name)
    importances = model.named_steps["regressor"].feature_importances_
    feat_imp = pd.Series(importances, index=clean_features).groupby(level=0).sum().sort_values(ascending=False).head(10)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        imp_df = feat_imp.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        fig_imp = px.bar(
            imp_df.iloc[::-1],
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Tealgrn",
            title="Top 10 Predictors of Overall MHQ Score",
            labels={"Importance": "Importance Score", "Feature": ""},
        )
        style_plotly(fig_imp, height=410, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with m_col2:
        perf_df = pd.DataFrame([
            {"Target": "Social Self Score", "R2": 0.1013, "RMSE": 72.52},
            {"Target": "Drive & Motivation", "R2": 0.1137, "RMSE": 63.69},
            {"Target": "Adapt & Resilience", "R2": 0.1281, "RMSE": 61.59},
            {"Target": "Overall MHQ Score", "R2": 0.1348, "RMSE": 67.24},
            {"Target": "Cognition Score", "R2": 0.1356, "RMSE": 63.40},
        ])
        fig_perf = px.bar(
            perf_df,
            x="R2",
            y="Target",
            orientation="h",
            text="R2",
            color="R2",
            color_continuous_scale="Teal",
            hover_data={"RMSE": ":.2f", "R2": ":.4f"},
            title="Model Predictability (R²) by Target",
            labels={"R2": "R-squared (Variance Explained)", "Target": ""},
        )
        fig_perf.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        style_plotly(fig_perf, height=410, coloraxis_showscale=False, xaxis=dict(range=[0, 0.165]))
        st.plotly_chart(fig_perf, use_container_width=True)

    with st.container(border=True):
        st.markdown(
            """
**Verified Model & Hypothesis Testing Summary**
- **Overall MHQ Model:** `R² = 0.1348` (`13.48%` variance explained), `RMSE = 67.24`
- **One-Way ANOVA (`overall_mhq_score`):**
  - `age`: $F = 3176.39$, $p < 0.0001$ (Significant)
  - `biological_sex`: $F = 586.09$, $p < 0.0001$ (Significant)
  - `income_household`: $F = 66.07$, $p = 8.72 \\times 10^{-259}$ (Significant)
- **Chi-Square (`biological_sex` vs. `income_household`):** $\\chi^2 = 1666.66$, $\\text{DoF} = 60$, $p = 7.26 \\times 10^{-309}$ (Significant)
"""
        )

elif active == "paper":
    st.subheader("Research Paper: Predicting Mental Health Questionnaire (MHQ) Scores from Demographic Factors")
    paper_html_path = os.path.join(BASE_DIR, "Research_Paper_Interactive.html")
    if "show_paper_viewer" not in st.session_state:
        st.session_state["show_paper_viewer"] = False

    if os.path.exists(paper_html_path):
        viewer_html = load_research_paper_html(
            paper_html_path,
            os.path.join(BASE_DIR, "assets"),
            os.path.getmtime(paper_html_path),
        )
        act_c1, act_c2, act_c3, _ = st.columns([1.35, 1.15, 1.25, 1.25])
        act_c1.link_button(
            label="↗ Open Paper in New Page",
            url="?view=paper",
            type="primary",
            use_container_width=True,
        )
        toggle_label = "Close Preview" if st.session_state["show_paper_viewer"] else "Preview Inline"
        if act_c2.button(toggle_label, use_container_width=True):
            st.session_state["show_paper_viewer"] = not st.session_state["show_paper_viewer"]
            st.rerun()
        act_c3.download_button(
            label="Download HTML Paper",
            data=viewer_html.encode("utf-8"),
            file_name="research_paper.html",
            mime="text/html",
            use_container_width=True,
        )

        if st.session_state["show_paper_viewer"]:
            st.caption("Inline reader preview — click 'Open Paper in New Page' for the full-screen presentation view.")
            components.html(viewer_html, height=820, scrolling=True)

    with st.expander("Ask the Study (Interactive Viva Q&A)", expanded=True):
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        preset_q = None
        if q_c1.button("Why is R² only 0.1348?", use_container_width=True):
            preset_q = "Why is R² only 0.1348?"
        if q_c2.button("Age vs. Sex vs. Income?", use_container_width=True):
            preset_q = "Which factor matters most: Age, Sex, or Income?"
        if q_c3.button("What drives other 86.5%?", use_container_width=True):
            preset_q = "What drives the remaining 86.5% of MHQ variance?"
        if q_c4.button("Loneliness Inventory (LI)?", use_container_width=True):
            preset_q = "How does the Loneliness Inventory (LI) cross-scale permutation validation address the Social Self paradox in medical cohorts?"
        custom_q = st.text_input("Or type any question about the study:", placeholder="e.g., How does household income affect MHQ scores?")
        active_q = preset_q or (custom_q.strip() if custom_q and custom_q.strip() else None)
        if active_q:
            with st.spinner("Searching verified MHQ study findings..."):
                ans_text, used_live = ask_research_paper_ai(active_q)
            badge_q = "Live Gemini Grounded Answer" if used_live else "Verified Study Knowledge Base"
            st.info(f"**{badge_q}**\n\n{ans_text}")

    with st.expander("Future Scope: Medical College Loneliness Inventory (LI) Cross-Validation", expanded=False):
        st.markdown(
            """
**Resolving the Social Self Paradox through Psychometric Cross-Validation:**

- **The Empirical Paradox:** `social_self_score` exhibits the strongest direct Pearson correlation with overall MHQ ($r = +0.9136$), yet records the lowest predictability from demographic features ($R^2 = 0.1013$, $\\text{RMSE} = 72.52$). Furthermore, 18-year-old college entrants record the lowest cohort MHQ ($23.12$ vs. $101.99$ for ages 65–74; $+78.9$ point gap), while frequent socializing delivers a $+40.9$-point MHQ gain.
- **19-Item Loneliness Inventory (LI) Instrument:** Developed by Uma, Meenakshi R. under the guidance of Prof. Krishnan K. (2010, *Psychotronics Publications*) for Indian collegiate and medical cohorts. Consists of 11 direct isolation items (scored $1 \\to 5$) and 8 positive connectedness / instrumental support items (Items 1, 5, 6, 9, 14, 16, 18, 19—reverse-scored $5 \\to 1$), yielding a theoretical range of $19$ (lowest loneliness) to $95$ (highest loneliness).
- **10,000-Iteration Permutation Protocol:** Co-administering LI and MHQ across anonymized medical students (ages 18–24) with institutional ethics approval. Non-parametric permutation testing ($10{,}000$ label resamples) computes exact, distribution-free $p$-values between total LI scores and MHQ targets, remaining robust for small sample sizes ($n = 100\\text{--}500$) without assuming bivariate normality.
- **Ensemble Integration:** Adding LI sub-constructs alongside sleep, UPF diet, and exercise into Random Forest / Gradient Boosting models with SHAP attribution to benchmark against the $\\sim 0.30$ survey predictability ceiling established by Oparina et al. (2025).
"""
        )

    with st.container(border=True):
        st.markdown(
            """
### Abstract (In Plain English)
This study evaluates whether **age**, **biological sex**, and **household income** can predict an individual's **Overall MHQ Score** across **240,012 survey respondents** and **207 variables**. The Random Forest model achieved **R² = 0.1348** and **RMSE = 67.24**, meaning demographic factors explain about **13.5%** of the variation in mental well-being—with **age** as the strongest driver and **Cognition** (`R² = 0.1356`) as the most predictable domain. The remaining ~86.5% depends on personal life experiences, sleep, social support, and clinical factors, so this model is designed for **population screening and research**, not medical diagnosis.

### Understanding the MHQ Score Scale (−100 to +200)
| Score Range | Tier | Everyday Meaning |
| :--- | :--- | :--- |
| **100 to 200** | Thriving | Strong emotional resilience, clarity, and life satisfaction |
| **50 to 99** | Succeeding | Functioning well across work, relationships, and mood |
| **0 to 49** | Managing / Enduring | Coping with daily life with occasional stress or fatigue |
| **−50 to −1** | Struggling | Noticeable distress affecting daily mood or focus |
| **−100 to −51** | Distressed | Severe symptoms; professional support recommended |

### Why Both Machine Learning & Statistical Tests?
- **One-Way ANOVA** proved that average MHQ differences across **age** ($F = 3176.39$), **biological sex** ($F = 586.09$), and **household income** ($F = 66.07$) are mathematically real ($p < 0.0001$), not random chance.
- **Chi-Square** ($\\chi^2 = 1666.66$, $p < 0.0001$) confirmed that biological sex and household income are statistically associated in the survey.
- **Data Quality Fix:** The raw `gender` column was 100% empty (240,012 nulls); replacing it with `biological_sex` (239,995 valid entries) resolved all statistical and modeling errors.
"""
        )

    with st.expander("Plain-English Glossary of Technical Terms"):
        st.markdown(
            """
- **Random Forest:** An ensemble of 100 decision trees that average their predictions for stable accuracy.
- **R² (0.1348):** The percentage of variation in MHQ scores explained by demographic inputs (13.48%).
- **RMSE (67.24):** Average prediction error in MHQ score points on the 300-point scale.
- **One-Hot Encoding:** Converting text categories (like `"Female"`, `"55-64"`) into 0/1 numeric columns.
- **SimpleImputer:** Filling missing values with `"Unknown"` so the model never crashes on blank fields.
"""
        )

    st.markdown("### Publication Figures")
    assets_dir = os.path.join(BASE_DIR, "assets")
    paper_figures = [
        ("pipeline_architecture.png", "Figure 1: End-to-End MHQ Research & Deployment Architecture"),
        ("mhq_by_age_distribution.png", "Figure 2: Average Overall MHQ Score Across Age Groups"),
        ("mhq_by_income_inr.png", "Figure 3: Mean Overall MHQ Score by Household Income (INR)"),
        ("subscore_by_age_and_sex.png", "Figure 4: Mean Overall MHQ Score by Biological Sex & Sub-Score Trajectories Across Age"),
        ("feature_importance.png", "Figure 5: Top 10 Demographic Predictors (INR Normalized)"),
        ("subscore_predictability.png", "Figure 6: Model Predictability (R²) Across Overall MHQ and Domain Sub-Scores"),
        ("mhq_direct_indirect_relations.png", "Figure 7: Direct Pearson Correlations (r) & Indirect Lifestyle / Socioeconomic MHQ Score Gaps"),
    ]
    for idx in range(0, len(paper_figures) - 1, 2):
        fig_c1, fig_c2 = st.columns(2)
        for col_target, (img_name, caption) in zip([fig_c1, fig_c2], paper_figures[idx:idx + 2]):
            img_p = os.path.join(assets_dir, img_name)
            if os.path.exists(img_p):
                col_target.image(img_p, caption=caption, use_container_width=True)
    last_img, last_cap = paper_figures[-1]
    last_p = os.path.join(assets_dir, last_img)
    if os.path.exists(last_p):
        st.image(last_p, caption=last_cap, use_container_width=True)

# Full-Page Swarm Rebuild Animation + Interactive Cursor Swarm Engine
if enable_swarm:
    components.html(
        f"""
        <!-- nonce: {st.session_state.rebuild_nonce}_{active} -->
        <script>
        (function () {{
          const doc = window.parent.document;
          const win = window.parent;
          const currentKey = "{st.session_state.rebuild_nonce}_{active}";

          const oldCanvas = doc.getElementById('mhq-swarm-overlay');
          if (oldCanvas) oldCanvas.remove();
          const oldBar = doc.getElementById('mhq-swarm-toolbar');
          if (oldBar) oldBar.remove();

          const canvas = doc.createElement('canvas');
          canvas.id = 'mhq-swarm-overlay';
          Object.assign(canvas.style, {{
            position: 'fixed',
            inset: '0',
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: '999998'
          }});
          doc.body.appendChild(canvas);
          const ctx = canvas.getContext('2d');

          function resize() {{
            canvas.width = win.innerWidth;
            canvas.height = win.innerHeight;
          }}
          resize();
          win.addEventListener('resize', resize);

          const colors = ['#14b8a6', '#38bdf8', '#a78bfa', '#34d399'];
          let speedMult = 1;
          let active = (win.__mhqLastRebuildKey !== currentKey);
          win.__mhqLastRebuildKey = currentKey;

          const selector = '.hero-banner, div[data-testid="stMetric"], .stSelectbox, .stButton, .stDownloadButton, div[data-testid="stImage"], div[data-testid="stMarkdownContainer"] h2, div[data-testid="stMarkdownContainer"] h3, div[data-testid="stPlotlyChart"]';
          let targets = [];
          function startRebuild() {{
            const rawEls = Array.from(doc.querySelectorAll(selector)).filter(el => {{
              const r = el.getBoundingClientRect();
              return r.width > 30 && r.height > 16 && r.bottom > 0 && r.top < win.innerHeight;
            }});
            targets = rawEls.map((el, idx) => {{
              el.style.transition = 'opacity 0.22s ease, transform 0.22s ease';
              el.style.opacity = '0.06';
              el.style.transform = 'translateY(8px) scale(0.985)';
              return {{
                el,
                progress: 0,
                delay: idx * 0.08,
                done: false,
                color: colors[idx % colors.length]
              }};
            }});
            active = targets.length > 0;
          }}

          if (active) {{
            setTimeout(startRebuild, 40);
          }}

          const bar = doc.createElement('div');
          bar.id = 'mhq-swarm-toolbar';
          Object.assign(bar.style, {{
            position: 'fixed',
            bottom: '64px',
            right: '22px',
            zIndex: '999999',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '7px 12px',
            borderRadius: '999px',
            background: 'rgba(15, 23, 42, 0.88)',
            border: '1px solid rgba(45, 212, 191, 0.4)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
            backdropFilter: 'blur(10px)',
            fontFamily: 'sans-serif',
            fontSize: '12px',
            color: '#e2e8f0'
          }});
          bar.innerHTML = `
            <span style="color:#2dd4bf;font-weight:700;">✦ Page Rebuild</span>
            <button id="mhq-swarm-replay" style="background:#0d9488;color:#fff;border:none;border-radius:999px;padding:4px 10px;cursor:pointer;font-weight:600;font-size:11px;">Replay</button>
            <button id="mhq-swarm-speed" style="background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:999px;padding:4px 9px;cursor:pointer;font-weight:600;font-size:11px;">1×</button>
          `;
          doc.body.appendChild(bar);

          bar.querySelector('#mhq-swarm-replay').onclick = () => startRebuild();
          bar.querySelector('#mhq-swarm-speed').onclick = (e) => {{
            speedMult = speedMult === 1 ? 2 : (speedMult === 2 ? 3 : 1);
            e.target.textContent = speedMult + '×';
          }};

          function tick() {{
            if (!doc.getElementById('mhq-swarm-overlay')) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (active && targets.length > 0) {{
              let remaining = 0;
              targets.forEach((job) => {{
                if (job.done) return;
                remaining++;
                if (job.delay > 0) {{
                  job.delay -= 0.04 * speedMult;
                  return;
                }}
                const r = job.el.getBoundingClientRect();
                const curW = r.width * Math.min(1, job.progress);
                const curH = r.height * Math.min(1, job.progress);

                ctx.save();
                ctx.strokeStyle = job.color;
                ctx.lineWidth = 1.6;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(r.left, r.top, curW, curH);
                ctx.setLineDash([]);
                ctx.restore();

                job.progress += 0.075 * speedMult;
                if (job.progress >= 1) {{
                  job.done = true;
                  job.el.style.opacity = '1';
                  job.el.style.transform = 'none';
                }}
              }});
              if (remaining === 0) active = false;
            }}
            if (active) {{
              win.requestAnimationFrame(tick);
            }}
          }}
          if (active) {{
            win.requestAnimationFrame(tick);
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
else:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const c = doc.getElementById('mhq-swarm-overlay'); if (c) c.remove();
        const b = doc.getElementById('mhq-swarm-toolbar'); if (b) b.remove();
        </script>
        """,
        height=0,
        width=0,
    )
