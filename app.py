import os
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
    plot_bgcolor="rgba(15,23,42,0.45)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#e2e8f0"),
    margin=dict(l=20, r=20, t=48, b=20),
)

# Custom UI/UX styling: modern vertical sidebar nav buttons, glass cards, and crisp typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    /* Main background & header polish */
    .stApp {
        background: radial-gradient(circle at top right, #0f292e 0%, #0f172a 45%, #090d16 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    /* Sidebar vertical panel styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1320 0%, #0f172a 100%);
        border-right: 1px solid rgba(20, 184, 166, 0.22);
    }
    /* Clickable vertical navigation buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        padding: 0.72rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(30, 41, 59, 0.55);
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        margin-bottom: 0.35rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #14b8a6;
        background: rgba(13, 148, 136, 0.18);
        color: #5eead4;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
        border-color: #2dd4bf;
        color: #ffffff;
        box-shadow: 0 4px 16px rgba(20, 184, 166, 0.28);
    }
    /* Primary action buttons in main content */
    .main .stButton > button[kind="primary"],
    .main .stDownloadButton > button {
        border-radius: 12px;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        background: linear-gradient(90deg, #0d9488 0%, #0ea5e9 100%);
        border: none;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.32);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .main .stButton > button[kind="primary"]:hover,
    .main .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.45);
    }
    /* Metric cards with hover lift & accent border */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 4px solid #14b8a6;
        border-radius: 14px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(13, 148, 136, 0.22);
    }
    .alert-card {
        padding: 16px 20px;
        border-radius: 14px;
        margin-top: 8px;
        font-size: 0.96rem;
        line-height: 1.55;
        color: #f8fafc;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
    }
    .alert-thriving { background: linear-gradient(135deg, rgba(13, 148, 136, 0.35), rgba(16, 185, 129, 0.25)); border: 1px solid #2dd4bf; }
    .alert-succeeding { background: linear-gradient(135deg, rgba(34, 197, 94, 0.28), rgba(14, 165, 233, 0.20)); border: 1px solid #4ade80; }
    .alert-managing { background: linear-gradient(135deg, rgba(234, 179, 8, 0.28), rgba(245, 158, 11, 0.18)); border: 1px solid #facc15; }
    .alert-struggling { background: linear-gradient(135deg, rgba(249, 115, 22, 0.30), rgba(239, 68, 68, 0.22)); border: 1px solid #fb923c; }
    .alert-distressed { background: linear-gradient(135deg, rgba(220, 38, 38, 0.35), rgba(153, 27, 27, 0.28)); border: 1px solid #f87171; }
    .hero-banner {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(13, 148, 136, 0.22), rgba(14, 165, 233, 0.14), rgba(167, 139, 250, 0.10));
        border: 1px solid rgba(45, 212, 191, 0.30);
        margin-bottom: 42px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.15rem !important;
        line-height: 1.3 !important;
        letter-spacing: -0.02em;
        margin: 0;
        text-align: center;
    }
    .hero-gradient {
        background: linear-gradient(90deg, #5eead4 0%, #38bdf8 50%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-spacer {
        height: 22px;
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


model = load_model()
if model is None:
    st.error("`mhq_model.joblib` was not found in the app directory.")
    st.stop()

summary_df, bins_df, income_df = load_summaries()
age_options, sex_options, income_options = get_model_categories(model)

# Vertical Sidebar Navigation Panel
NAV_ITEMS = [
    ("🎯 Live Prediction", "prediction"),
    ("📊 Demographic Explorer", "explorer"),
    ("📈 Model & Statistics", "model"),
    ("📄 Research Paper", "paper"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "prediction"
if "rebuild_nonce" not in st.session_state:
    st.session_state.rebuild_nonce = 1

with st.sidebar:
    st.markdown("## 🧠 MHQ Navigation")
    st.caption("Select a workspace view")
    for label, key in NAV_ITEMS:
        btn_type = "primary" if st.session_state.active_page == key else "secondary"
        if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
            st.session_state.active_page = key
            st.session_state.rebuild_nonce += 1
            st.rerun()

    st.divider()
    enable_swarm = st.toggle("✨ Full-Page Swarm Rebuild", value=True)
    if st.button("🔄 Replay Page Rebuild", use_container_width=True):
        st.session_state.rebuild_nonce += 1
        st.rerun()

# Clean Hero Banner (removed "Mental Health Questionnaire (MHQ) Study — 240,012 Respondents | Random Forest Pipeline")
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🧠 <span class="hero-gradient">MHQ Research &amp; Well-Being Dashboard</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

active = st.session_state.active_page

def render_prediction_results(selected_age: str, selected_sex: str, selected_income: str):
    pred = predict_mhq(model, selected_age, selected_sex, selected_income)
    delta = pred - STUDY_MEAN_MHQ
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
    m1.metric("🎯 Predicted MHQ Score", f"{pred:.2f}", f"{delta:+.2f} vs. Mean")
    m2.metric("🧭 Well-Being Tier", tier)
    m3.metric("📈 Study Percentile", f"{percentile:.0f}th")
    m4.metric("👥 Age Cohort Size", f"{cohort_n:,}")

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    g_col, a_col = st.columns([1.6, 1.0])

    with g_col:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": STUDY_MEAN_MHQ, "valueformat": ".1f"},
                number={"valueformat": ".1f", "font": {"size": 42, "color": "#f8fafc"}},
                title={"text": f"Predicted MHQ Score · {tier}", "font": {"size": 16, "color": "#5eead4"}},
                gauge={
                    "axis": {"range": [-100, 200], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#14b8a6", "thickness": 0.3},
                    "bgcolor": "rgba(15,23,42,0.5)",
                    "steps": [
                        {"range": [-100, -50], "color": "rgba(220,38,38,0.55)"},
                        {"range": [-50, 0], "color": "rgba(249,115,22,0.55)"},
                        {"range": [0, 50], "color": "rgba(234,179,8,0.55)"},
                        {"range": [50, 100], "color": "rgba(34,197,94,0.55)"},
                        {"range": [100, 200], "color": "rgba(13,148,136,0.55)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.85,
                        "value": STUDY_MEAN_MHQ,
                    },
                },
            )
        )
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with a_col:
        if pred >= 100:
            cls, msg = "alert-thriving", "🟢 <b>Thriving Well-Being</b><br>Strong emotional resilience, clarity, and life satisfaction."
        elif pred >= 50:
            cls, msg = "alert-succeeding", "🟢 <b>Succeeding Well-Being</b><br>Functioning well across work, relationships, and daily mood."
        elif pred >= 0:
            cls, msg = "alert-managing", "🟡 <b>Managing / Enduring</b><br>Coping with daily life, though with occasional stress or fatigue."
        elif pred >= -50:
            cls, msg = "alert-struggling", "🟠 <b>Struggling</b><br>Noticeable distress affecting daily mood or focus."
        else:
            cls, msg = "alert-distressed", "🔴 <b>Distressed</b><br>Significant distress; professional clinical support is recommended."
        st.markdown(
            f'<div class="alert-card {cls}">{msg}<br><br><small>Profile: <b>{selected_age}</b> · <b>{selected_sex}</b> · <b>{selected_income}</b><br>Study Average: <b>{STUDY_MEAN_MHQ:.2f}</b> · Screening indicator only, not a clinical diagnosis.</small></div>',
            unsafe_allow_html=True,
        )


if active == "prediction":
    if "pred_mode" not in st.session_state:
        st.session_state.pred_mode = "custom"
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = "young"

    st.subheader("🎯 Live MHQ Prediction Workspace")
    m_col1, m_col2, m_col3 = st.columns(3)
    modes = [
        ("⚡ Quick Demo Presets", "presets"),
        ("🎯 Custom Prediction", "custom"),
        ("⚖️ Compare Profiles (A vs. B)", "compare"),
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
        st.markdown("### ⚡ Quick Demo Presets")
        st.caption("Click any persona below to immediately view their predicted MHQ score and tier")
        PRESETS = {
            "young": ("🎓 Young Adult (18, Female)", "18", "Female", "Less than ₹1,00,000"),
            "mid": ("💼 Mid-Career (35-44, Male)", "35-44", "Male", "₹10,00,001 – ₹20,00,000"),
            "older": ("🌿 Older Adult (65-74, Male)", "65-74", "Male", "₹20,00,001 – ₹40,00,000"),
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
        render_prediction_results(p_age, p_sex, p_inc)

    elif st.session_state.pred_mode == "custom":
        st.markdown("### 🎯 Custom Demographic Prediction")
        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_age = st.selectbox("Age Group", age_options, index=0, key="custom_age")
        with col2:
            selected_sex = st.selectbox(
                "Biological Sex",
                sex_options,
                index=0,
                key="custom_sex",
                format_func=lambda x: f"{'👩' if x == 'Female' else '👨' if x == 'Male' else '⚪'} {x}",
            )
        with col3:
            selected_income = st.selectbox("Household Income (INR)", income_options, index=0, key="custom_inc")

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        if st.button("✨ Predict MHQ Score", type="primary"):
            render_prediction_results(selected_age, selected_sex, selected_income)

    elif st.session_state.pred_mode == "compare":
        st.markdown("### ⚖️ Compare Two Profiles (Profile A vs. Profile B)")
        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Profile A**")
            age_a = st.selectbox("Age Group (A)", age_options, index=0, key="cmp_age_a")
            sex_a = st.selectbox("Biological Sex (A)", sex_options, index=0, key="cmp_sex_a")
            inc_a = st.selectbox("Household Income (A)", income_options, index=0, key="cmp_inc_a")
        with cb:
            st.markdown("**Profile B**")
            age_b = st.selectbox("Age Group (B)", age_options, index=min(8, len(age_options) - 1), key="cmp_age_b")
            sex_b = st.selectbox("Biological Sex (B)", sex_options, index=min(1, len(sex_options) - 1), key="cmp_sex_b")
            inc_b = st.selectbox("Household Income (B)", income_options, index=min(4, len(income_options) - 1), key="cmp_inc_b")

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        if st.button("⚖️ Compare Profile A vs. Profile B", type="primary"):
            score_a = predict_mhq(model, age_a, sex_a, inc_a)
            score_b = predict_mhq(model, age_b, sex_b, inc_b)
            diff_ba = score_b - score_a

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Profile A Predicted MHQ", f"{score_a:.2f}", mhq_category(score_a))
            cm2.metric("Profile B Predicted MHQ", f"{score_b:.2f}", mhq_category(score_b))
            cm3.metric("Difference (B − A)", f"{diff_ba:+.2f} pts", "Profile B Higher" if diff_ba >= 0 else "Profile A Higher")

            cmp_df = pd.DataFrame(
                {
                    "Profile": [f"Profile A ({age_a}, {sex_a})", f"Profile B ({age_b}, {sex_b})", "Study Average"],
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
            fig_cmp.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280, xaxis=dict(range=[-100, 210], title="Predicted Overall MHQ Score (-100 to 200)"), yaxis_title="")
            st.plotly_chart(fig_cmp, use_container_width=True)

elif active == "explorer":
    st.subheader("📊 MHQ Score Distribution & Demographic Trends")
    if summary_df is None or bins_df is None:
        st.info("Summary files (`demographic_summary.csv`, `mhq_bins.csv`) not found.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            age_filter = st.selectbox("Filter Age Group", ["All"] + age_options)
        with f2:
            sex_filter = st.selectbox("Filter Biological Sex", ["All"] + sex_options)

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
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=360, bargap=0.06)
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
            fig_age.update_layout(**PLOTLY_LAYOUT, height=390, coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)

        with g2:
            melted = age_grp.melt(id_vars="age", var_name="Dimension", value_name="Mean Score")
            fig_sub = px.line(
                melted,
                x="age",
                y="Mean Score",
                color="Dimension",
                markers=True,
                title="All 5 MHQ Dimensions Across Age Groups",
                labels={"age": "Age Group", "Mean Score": "Mean Score"},
            )
            fig_sub.update_layout(**PLOTLY_LAYOUT, height=390, hovermode="x unified", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_sub, use_container_width=True)

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
            fig_inc.update_layout(**PLOTLY_LAYOUT, height=420, coloraxis_showscale=False, yaxis=dict(categoryorder="array", categoryarray=income_df["income_inr"].tolist()[::-1]))
            st.plotly_chart(fig_inc, use_container_width=True)

elif active == "model":
    st.subheader("📈 Feature Importance, Model Performance & Statistical Findings")
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
            color_continuous_scale="Viridis",
            title="Top 10 Predictors of Overall MHQ Score",
            labels={"Importance": "Importance Score", "Feature": ""},
        )
        fig_imp.update_layout(**PLOTLY_LAYOUT, height=410, coloraxis_showscale=False)
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
        fig_perf.update_layout(**PLOTLY_LAYOUT, height=410, coloraxis_showscale=False, xaxis=dict(range=[0, 0.165]))
        st.plotly_chart(fig_perf, use_container_width=True)

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
    st.subheader("📄 Predicting Mental Health Questionnaire (MHQ) Scores from Demographic Factors")
    paper_html_path = os.path.join(BASE_DIR, "Research_Paper_Interactive.html")
    if os.path.exists(paper_html_path):
        with open(paper_html_path, "rb") as pf:
            st.download_button(
                label="⬇️ Download Research Paper",
                data=pf.read(),
                file_name="research_paper.html",
                mime="text/html",
            )
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
    with st.expander("📖 Plain-English Glossary of Technical Terms"):
        st.markdown(
            """
- **Random Forest:** An ensemble of 100 decision trees that average their predictions for stable accuracy.
- **R² (0.1348):** The percentage of variation in MHQ scores explained by demographic inputs (13.48%).
- **RMSE (67.24):** Average prediction error in MHQ score points on the 300-point scale.
- **One-Hot Encoding:** Converting text categories (like `"Female"`, `"55-64"`) into 0/1 numeric columns.
- **SimpleImputer:** Filling missing values with `"Unknown"` so the model never crashes on blank fields.
"""
        )
    assets_dir = os.path.join(BASE_DIR, "assets")
    for img_name, caption in [
        ("pipeline_architecture.png", "Figure 1: End-to-End MHQ Research & Deployment Architecture"),
        ("mhq_by_age_distribution.png", "Figure 2: Average Overall MHQ Score Across Age Groups"),
        ("mhq_by_income_inr.png", "Figure 3: Mean Overall MHQ Score by Household Income (INR)"),
        ("subscore_by_age_and_sex.png", "Figure 4: Mean Overall MHQ Score by Biological Sex & Sub-Score Trajectories Across Age"),
        ("feature_importance.png", "Figure 5: Top 10 Demographic Predictors (INR Normalized)"),
        ("subscore_predictability.png", "Figure 6: Model Predictability (R²) Across Overall MHQ and Domain Sub-Scores"),
        ("mhq_direct_indirect_relations.png", "Figure 7: Direct Pearson Correlations (r) & Indirect Lifestyle / Socioeconomic MHQ Score Gaps"),
    ]:
        img_p = os.path.join(assets_dir, img_name)
        if os.path.exists(img_p):
            st.image(img_p, caption=caption, use_container_width=True)

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
            win.requestAnimationFrame(tick);
          }}
          win.requestAnimationFrame(tick);
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
