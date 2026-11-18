import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MHQ Mental Well-Being Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="darkgrid")

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
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
    }
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
    sex_opts = ["Female", "Male", "Other / Intersex", "Prefer not to say"]
    return age_opts, sex_opts, INCOME_INR_OPTIONS


def predict_mhq(model, age: str, biological_sex: str, income_inr_label: str) -> float:
    raw_income = INR_TO_MODEL_RAW.get(str(income_inr_label), str(income_inr_label))
    raw_sex = "Other/Intersex" if biological_sex == "Other / Intersex" else str(biological_sex)
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

if active == "prediction":
    st.subheader("🎯 Predict Overall MHQ Score")
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_age = st.selectbox("Age Group", age_options, index=0)
    with col2:
        selected_sex = st.selectbox("Biological Sex", sex_options, index=0)
    with col3:
        selected_income = st.selectbox("Household Income (INR)", income_options, index=0)

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    if st.button("Predict MHQ Score", type="primary"):
        pred = predict_mhq(model, selected_age, selected_sex, selected_income)
        delta = pred - STUDY_MEAN_MHQ

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Overall MHQ Score", f"{pred:.2f}", f"{delta:+.2f} vs. Study Mean")
        m2.metric("Well-Being Tier", mhq_category(pred))
        m3.metric("Study Average MHQ", f"{STUDY_MEAN_MHQ:.2f}")

        fig_scale, ax_scale = plt.subplots(figsize=(9, 1.8))
        tiers = [
            (-100, -50, "#DC2626", "Distressed"),
            (-50, 0, "#F97316", "Struggling"),
            (0, 50, "#EAB308", "Managing"),
            (50, 100, "#22C55E", "Succeeding"),
            (100, 200, "#0D9488", "Thriving"),
        ]
        for start, end, color, label in tiers:
            ax_scale.barh(0, end - start, left=start, height=0.45, color=color, alpha=0.85)
            ax_scale.text((start + end) / 2, 0, label, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax_scale.axvline(STUDY_MEAN_MHQ, color="white", linestyle="--", linewidth=1.4, label=f"Study Mean ({STUDY_MEAN_MHQ:.1f})")
        ax_scale.scatter([pred], [0], color="white", edgecolor="black", s=160, zorder=5, label=f"Prediction ({pred:.1f})")
        ax_scale.set_xlim(-100, 200)
        ax_scale.set_yticks([])
        ax_scale.set_xlabel("MHQ Score Scale (-100 to 200)")
        ax_scale.legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False)
        plt.tight_layout()
        st.pyplot(fig_scale)
        plt.close(fig_scale)

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
            raw_sex_f = ["Other/Intersex", "Other"] if sex_filter == "Other / Intersex" else ([sex_filter, "Männlich"] if sex_filter == "Male" else [sex_filter])
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
            fig_hist, ax_hist = plt.subplots(figsize=(9, 3.8))
            ax_hist.bar(hist_agg["bin_center"], hist_agg["count"], width=9.0, color="#0D9488", edgecolor="#0F172A", alpha=0.9)
            ax_hist.axvline(weighted_mean, color="#F59E0B", linestyle="--", linewidth=1.6, label=f"Group Mean ({weighted_mean:.1f})")
            ax_hist.set_title(f"MHQ Score Distribution (Age: {age_filter}, Sex: {sex_filter})")
            ax_hist.set_xlabel("Overall MHQ Score")
            ax_hist.set_ylabel("Respondent Count")
            ax_hist.legend()
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

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
            fig_age, ax_age = plt.subplots(figsize=(6.5, 4))
            sns.barplot(data=age_grp, x="age", y="Overall MHQ", hue="age", palette="coolwarm", legend=False, ax=ax_age)
            ax_age.axhline(STUDY_MEAN_MHQ, color="#F59E0B", linestyle="--", linewidth=1.2, label=f"Mean ({STUDY_MEAN_MHQ:.1f})")
            ax_age.set_title("Mean Overall MHQ Score by Age Group")
            ax_age.set_xlabel("Age Group")
            ax_age.set_ylabel("Mean MHQ Score")
            ax_age.tick_params(axis="x", rotation=40)
            ax_age.legend()
            plt.tight_layout()
            st.pyplot(fig_age)
            plt.close(fig_age)

        with g2:
            fig_sub, ax_sub = plt.subplots(figsize=(6.5, 4))
            melted = age_grp.melt(id_vars="age", var_name="Dimension", value_name="Mean Score")
            sns.lineplot(data=melted, x="age", y="Mean Score", hue="Dimension", marker="o", ax=ax_sub)
            ax_sub.set_title("All 5 MHQ Dimensions Across Age Groups")
            ax_sub.set_xlabel("Age Group")
            ax_sub.set_ylabel("Mean Score")
            ax_sub.tick_params(axis="x", rotation=40)
            ax_sub.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_sub)
            plt.close(fig_sub)

        if income_df is not None and not income_df.empty:
            fig_inc, ax_inc = plt.subplots(figsize=(9.5, 4.2))
            sns.barplot(
                data=income_df,
                x="mean_mhq",
                y="income_inr",
                hue="income_inr",
                palette="crest",
                legend=False,
                ax=ax_inc,
            )
            ax_inc.axvline(STUDY_MEAN_MHQ, color="#F59E0B", linestyle="--", linewidth=1.3, label=f"Study Mean ({STUDY_MEAN_MHQ:.1f})")
            ax_inc.set_title("Mean Overall MHQ Score by Household Income (INR)")
            ax_inc.set_xlabel("Mean Overall MHQ Score")
            ax_inc.set_ylabel("Household Income (INR)")
            ax_inc.legend()
            plt.tight_layout()
            st.pyplot(fig_inc)
            plt.close(fig_inc)

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
        fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.2))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False, ax=ax_imp)
        ax_imp.set_title("Top 10 Predictors of Overall MHQ Score")
        ax_imp.set_xlabel("Importance Score")
        ax_imp.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    with m_col2:
        perf_df = pd.DataFrame([
            {"Target": "Cognition Score", "R2": 0.1356},
            {"Target": "Overall MHQ Score", "R2": 0.1348},
            {"Target": "Adapt & Resilience", "R2": 0.1281},
            {"Target": "Drive & Motivation", "R2": 0.1137},
            {"Target": "Social Self Score", "R2": 0.1013},
        ])
        fig_perf, ax_perf = plt.subplots(figsize=(6.5, 4.2))
        sns.barplot(data=perf_df, x="R2", y="Target", hue="Target", palette="crest", legend=False, ax=ax_perf)
        for i, v in enumerate(perf_df["R2"]):
            ax_perf.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")
        ax_perf.set_xlim(0, 0.165)
        ax_perf.set_title("Model Predictability (R²) by Target")
        ax_perf.set_xlabel("R-squared (Variance Explained)")
        ax_perf.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_perf)
        plt.close(fig_perf)

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

          const selector = '.hero-banner, div[data-testid="stMetric"], .stSelectbox, .stButton, .stDownloadButton, div[data-testid="stImage"], div[data-testid="stMarkdownContainer"] h2, div[data-testid="stMarkdownContainer"] h3, div[data-testid="stPyplot"]';
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
            bottom: '18px',
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
