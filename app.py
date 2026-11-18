import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="MHQ Mental Well-Being Predictor",
    page_icon="🧠",
    layout="wide",
)

sns.set_theme(style="darkgrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mhq_model.joblib")
SUMMARY_PATH = os.path.join(BASE_DIR, "demographic_summary.csv")
BINS_PATH = os.path.join(BASE_DIR, "mhq_bins.csv")

AGE_COL = "age"
SEX_COL = "biological_sex"
INCOME_COL = "income_household"
STUDY_MEAN_MHQ = 67.65
AGE_ORDER = ["18", "19", "20", "21-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_summaries():
    summary_df = pd.read_csv(SUMMARY_PATH) if os.path.exists(SUMMARY_PATH) else None
    bins_df = pd.read_csv(BINS_PATH) if os.path.exists(BINS_PATH) else None
    return summary_df, bins_df


def get_model_categories(model) -> tuple[list[str], list[str], list[str]]:
    onehot = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cats = onehot.categories_
    age_opts = sorted([str(c) for c in cats[0] if str(c) != "Unknown"])
    sex_opts = sorted([str(c) for c in cats[1] if str(c) != "Unknown"])
    inc_opts = sorted([str(c) for c in cats[2] if str(c) != "Unknown"])
    return age_opts, sex_opts, inc_opts


def predict_mhq(model, age: str, biological_sex: str, income: str) -> float:
    input_df = pd.DataFrame(
        {
            AGE_COL: [str(age)],
            SEX_COL: [str(biological_sex)],
            INCOME_COL: [str(income)],
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


st.title("🧠 MHQ Research Dashboard & Well-Being Predictor")
st.caption("Mental Health Questionnaire (MHQ) Study — 240,012 Respondents | Random Forest Pipeline")

model = load_model()
if model is None:
    st.error("`mhq_model.joblib` was not found in the app directory.")
    st.stop()

summary_df, bins_df = load_summaries()
age_options, sex_options, income_options = get_model_categories(model)

tab_pred, tab_explore, tab_model, tab_paper = st.tabs(
    ["🎯 Live Prediction", "📊 Demographic Explorer", "📈 Model & Statistics", "📄 Research Paper"]
)

with tab_pred:
    st.subheader("Predict Overall MHQ Score")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_age = st.selectbox("Age Group", age_options, index=0)
    with col2:
        selected_sex = st.selectbox("Biological Sex", sex_options, index=0)
    with col3:
        selected_income = st.selectbox("Household Income", income_options, index=0)

    if st.button("Predict MHQ Score", type="primary"):
        pred = predict_mhq(model, selected_age, selected_sex, selected_income)
        delta = pred - STUDY_MEAN_MHQ

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Overall MHQ Score", f"{pred:.2f}", f"{delta:+.2f} vs. Study Mean")
        m2.metric("Well-Being Tier", mhq_category(pred))
        m3.metric("Study Average MHQ", f"{STUDY_MEAN_MHQ:.2f}")

        # Visual position on the MHQ scale (-100 to 200)
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

with tab_explore:
    st.subheader("MHQ Score Distribution & Demographic Trends")
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
            filt_sum = filt_sum[filt_sum["biological_sex"] == sex_filter]
            filt_bins = filt_bins[filt_bins["biological_sex"] == sex_filter]

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

        # Graph 2 & 3: Age Group Mean & All 5 Well-Being Dimensions by Age
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

with tab_model:
    st.subheader("Feature Importance, Model Performance & Statistical Findings")
    all_features = model.named_steps["preprocessor"].get_feature_names_out()
    clean_features = [name.replace("cat__", "") for name in all_features]
    importances = model.named_steps["regressor"].feature_importances_
    feat_imp = pd.Series(importances, index=clean_features).sort_values(ascending=False).head(10)

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
        ax_perf.set_title("Random Forest Predictability (R²) by Target")
        ax_perf.set_xlabel("R-squared (Variance Explained)")
        ax_perf.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_perf)
        plt.close(fig_perf)

    st.markdown(
        """
**Verified Model & Hypothesis Testing Summary**
- **Overall MHQ Random Forest:** `R² = 0.1348` (`13.48%` variance explained), `RMSE = 67.24`
- **One-Way ANOVA (`overall_mhq_score`):**
  - `age`: $F = 3176.39$, $p < 0.0001$ (Significant)
  - `biological_sex`: $F = 586.09$, $p < 0.0001$ (Significant)
  - `income_household`: $F = 66.07$, $p = 8.72 \\times 10^{-259}$ (Significant)
- **Chi-Square (`biological_sex` vs. `income_household`):** $\\chi^2 = 1666.66$, $\\text{DoF} = 60$, $p = 7.26 \\times 10^{-309}$ (Significant)
"""
    )

with tab_paper:
    st.subheader("Predicting Mental Health Questionnaire (MHQ) Scores from Demographic Factors")
    st.caption("Refined Empirical Research Paper — 240,012 Survey Respondents | Random Forest Regression & Inferential Statistics")
    paper_html_path = os.path.join(BASE_DIR, "Research_Paper_Interactive.html")
    if os.path.exists(paper_html_path):
        with open(paper_html_path, "rb") as pf:
            st.download_button(
                label="⬇️ Download Interactive Research Paper (HTML)",
                data=pf.read(),
                file_name="Research_Paper_Interactive.html",
                mime="text/html",
            )
    st.markdown(
        """
### Abstract
This study evaluates whether **age**, **biological sex**, and **household income** predict an individual's **Overall MHQ Score** across **240,012 respondents** and **207 variables**. A Scikit-Learn Random Forest pipeline (`SimpleImputer` → `OneHotEncoder` → 100-tree `RandomForestRegressor`, `max_depth=10`, `random_state=42`) achieved **R² = 0.1348** and **RMSE = 67.24**, with **age** emerging as the dominant predictor and **Cognition** (`R² = 0.1356`) as the most predictable sub-score.
"""
    )
    assets_dir = os.path.join(BASE_DIR, "assets")
    for img_name, caption in [
        ("pipeline_architecture.png", "Figure 1: End-to-End MHQ Research & Deployment Pipeline Architecture"),
        ("mhq_by_age_distribution.png", "Figure 2: Average Overall MHQ Score Across Age Groups (n = 240,004)"),
        ("subscore_by_age_and_sex.png", "Figure 3: Mean Overall MHQ Score by Biological Sex & Sub-Score Trajectories Across Age"),
        ("feature_importance.png", "Figure 4: Top 10 Demographic Predictors (Random Forest Gini Importance)"),
        ("subscore_predictability.png", "Figure 5: Model Predictability (R²) Across Overall MHQ and Domain Sub-Scores"),
    ]:
        img_p = os.path.join(assets_dir, img_name)
        if os.path.exists(img_p):
            st.image(img_p, caption=caption, use_container_width=True)
