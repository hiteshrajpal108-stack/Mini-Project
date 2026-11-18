# MHQ Mental Well-Being Prediction — Mini Project Presentation Summary

## 1. Problem Statement & Dataset
- **Objective:** Predict an individual's Overall Mental Health Questionnaire (`overall_mhq_score`) and domain sub-scores from demographic attributes (`age`, `biological_sex`, `income_household`).
- **Dataset:** 240,012 respondents × 207 variables (`your_file.csv`).
- **Key Data Fix:** The `gender` column contained 240,012 missing values (100% null), which caused Chi-Square and `UserWarning` failures. Replacing it with `biological_sex` (239,995 valid entries) resolved all statistical and modeling errors.

## 2. Statistical Analysis & Hypothesis Testing
- **One-Way ANOVA (`overall_mhq_score`):**
  - `age`: $F = 3176.39$, $p < 0.0001$ (Statistically Significant)
  - `biological_sex`: $F = 586.09$, $p < 0.0001$ (Statistically Significant)
  - `income_household`: $F = 66.07$, $p = 8.72 \times 10^{-259}$ (Statistically Significant)
- **Chi-Square Test (`biological_sex` vs. `income_household`, $n = 19,520$):**
  - $\chi^2 = 1666.6633$, $\text{DoF} = 60$, $p = 7.26 \times 10^{-309}$ (Significant dependence)
- **Multicollinearity Check:** Input features are independent (no high correlation), validating Random Forest feature importances.

## 3. Machine Learning Pipeline & Verified Results
- **Pipeline:** `SimpleImputer(constant="Unknown")` → `OneHotEncoder(handle_unknown="ignore")` → `RandomForestRegressor(n_estimators=100, max_depth=10)`
- **Primary Target (`overall_mhq_score`):**
  - **R²:** `0.1348` (13.48% variance explained)
  - **RMSE:** `67.2379`
- **Sub-Score Predictability:**
  - `cognition_score`: `R² = 0.1356` (RMSE: `63.4010`)
  - `adapt_resilience_score`: `R² = 0.1281` (RMSE: `61.5867`)
  - `drive_motivation_score`: `R² = 0.1137` (RMSE: `63.6859`)
  - `social_self_score`: `R² = 0.1013` (RMSE: `72.5243`)

## 4. Deployment (Ready for LCA3 Submission)
- Run the interactive Streamlit dashboard locally or on Streamlit Community Cloud:
  ```bash
  streamlit run app.py
  ```
- Includes live MHQ score prediction, demographic distribution explorer, and feature importance charts backed by `mhq_model.joblib`.
