# 🧠 MHQ Research & Well-Being Dashboard — 1-Page Viva Cheat Sheet

## 1. 60-Second Spoken Opening Script
> "Good morning ma'am/sir. My project is the **MHQ Research & Well-Being Dashboard**, built on a real-world survey of **240,012 respondents** across **207 variables**. My goal was to evaluate how accurately three basic demographics—**age**, **biological sex**, and **household income**—predict an individual's Overall Mental Health Questionnaire (MHQ) score, and to uncover the direct and indirect factors behind mental well-being.
>
> First, I profiled the 362 MB dataset using **Polars**, discovered that the raw `gender` column was 100% null, switched to `biological_sex` (239,995 valid entries), and standardized mixed currencies into **INR (₹)**. Second, I validated group differences using **One-Way ANOVA** and **Chi-Square** tests, proving age is the dominant demographic driver (*F* = 3176.39, *p* < 0.0001). Third, I trained a Scikit-Learn **Random Forest pipeline** achieving **R² = 0.1348** and **RMSE = 67.24**, showing demographics explain ~13.5% of variance while lifestyle and clinical factors explain the rest. Finally, I deployed the model on **Streamlit Community Cloud** with interactive Plotly visualizations, profile comparison, and a built-in research paper."

---

## 2. Key Verified Numbers (Memorize These)
| Metric | Verified Value | What It Means |
| :--- | :--- | :--- |
| **Dataset Size** | **240,012 rows × 207 cols** | 362 MB CSV processed with Polars Lazy API |
| **MHQ Score Scale** | **−100 to +200** (Mean: **67.65**) | Higher = better well-being (`100+` = Thriving) |
| **Overall MHQ Model** | **R² = 0.1348** · **RMSE = 67.24** | 100-tree Random Forest (`max_depth=10`, 80/20 split) |
| **Top Sub-Score** | **Cognition: R² = 0.1356** | Most predictable (`Social Self` lowest at **0.1013**) |
| **One-Way ANOVA** | **Age F = 3176.39**, **Sex F = 586.09**, **Income F = 66.07** | All *p* < 0.0001 (Statistically Significant) |
| **Chi-Square Test** | **χ² = 1666.66** (df = 60, *p* < 0.0001) | `biological_sex` and `income_household` are dependent |
| **Top 3 Features** | `age_18` (**0.26**), `age_65-74` (**0.17**), `age_55-64` (**0.15**) | Age dominates demographic importance |

---

## 3. Direct & Indirect Relationships with MHQ
- **Direct Positive (*higher → higher MHQ*):** Domain sub-scores (`social_self` *r* = **+0.91**, `cognition` *r* = **+0.91**), `energy` (**+0.56**), `emotional_resilience` (**+0.54**), `self_image` (**+0.54**), `outlook_optimism` (**+0.54**), Age (**23.1** at 18 → **102.0** at 65–74), INR Income (**49.6** under ₹1L → **108.3** for ₹85L–₹2.1Cr).
- **Direct Negative (*higher → lower MHQ*):** `sad_hopeless` (*r* = **−0.72**), `confusion` (**−0.72**), `obsessive_thoughts` (**−0.68**), `mood_swings` (**−0.67**), `suicidal_thoughts` (**−0.65**), Daily Ultra-Processed Food (**12.6** vs. **81.2** Rarely/Never = **−68.6 pts**).
- **Indirect (Mediated) Pathways:**
  - **Age → Lower Sadness & Higher Self-Image/Energy → Higher MHQ:** From age 18 to 65–74, `sad_hopeless` drops (**5.64 → 3.31**) while `self_image` (**5.31 → 6.99**) and `energy` (**5.27 → 6.78**) rise, driving the **+78.9-pt** MHQ gain.
  - **Sleep & Exercise → Energy → Higher MHQ:** Consistent sleep lifts `energy` (**5.09 → 6.86**) and MHQ by **+74.4 pts**; daily exercise lifts `energy` to **7.06** and MHQ by **+45.6 pts**.
  - **Sex ↔ Income → MHQ:** Proven dependent via Chi-Square (**χ² = 1666.66**), so part of the sex gap works indirectly through income.

---

## 4. Top 8 Examiner Viva Q&As
1. **Why is R² only 0.1348 (13.48%)?**
   In behavioral science, 3 basic demographics alone cannot capture trauma, sleep, genetics, or lifestyle. Explaining 13.5% proves demographics are useful for **population-level screening**, not clinical diagnosis—and our direct/indirect analysis shows what drives the other 86.5%.
2. **Why `biological_sex` instead of `gender`?**
   Our missingness audit caught that `gender` had **240,012 / 240,012 nulls (100% empty)**, which crashed Chi-Square. Switching to `biological_sex` (239,995 valid rows) fixed all statistical and ML steps.
3. **Why Random Forest instead of Linear Regression?**
   All 3 predictors are categorical brackets with non-linear patterns across age cohorts. Random Forest captures non-linear combinations automatically after One-Hot Encoding and reduces overfitting via 100 averaged trees (`max_depth=10`).
4. **Why use a Scikit-Learn `Pipeline`?**
   Chaining `SimpleImputer` → `OneHotEncoder` → `RandomForestRegressor` prevents **data leakage** during train/test splits and guarantees live Streamlit inputs receive the exact same preprocessing as training data.
5. **Why Polars instead of Pandas for loading?**
   `your_file.csv` is 362 MB (240k × 207). Polars Lazy API (`pl.scan_csv`) scans the schema and reads only the 8 required columns in seconds with minimal RAM.
6. **How does Streamlit Cloud run without the 362 MB CSV?**
   GitHub caps files at 100 MB, so we separated offline training (`mhq_model.joblib`, 1.6 MB) from online analytics (`demographic_summary.csv`, `mhq_bins.csv`, `income_summary.csv`, <35 KB total).
7. **What do your ANOVA and Chi-Square tests prove?**
   ANOVA proves mean MHQ differences across age, sex, and income are statistically real (*p* < 0.0001). Chi-Square proves `biological_sex` and `income_household` are statistically associated, not independent.
8. **What are your limitations & future work?**
   Cross-sectional data shows correlation (not causation), and income had high missingness. Next steps: add sleep, exercise, and trauma features from the 207 columns and compare XGBoost/LightGBM.

---

## 5. 5-Click Live Demo Checklist
1. Open `mhq-wellbeing-predictor.streamlit.app` beforehand to wake it up.
2. **Live Prediction → ⚡ Quick Demo Presets:** Click *Young Adult* → *Older Adult* to show the Plotly gauge jump.
3. **Live Prediction → ⚖️ Compare Profiles:** Run one Profile A vs. B comparison.
4. **Demographic Explorer:** Hover over the Age, 5-Dimension, and INR Income Plotly charts.
5. **Model & Statistics → Research Paper:** Show R²/Feature Importance, then show Figure 7 and the Research Paper download.
