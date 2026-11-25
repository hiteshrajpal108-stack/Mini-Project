# MHQ Machine Learning & Statistical Pipeline# Extracted from PythonProjectNew.ipynb
import os
import warnings
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Portable path resolution: works from data/ or anywhere on this machine
DEFAULT_PATH = "/Users/hiteshrajpal/HITESH_PROGRAMMING/Python programming /data/your_file.csv"
CSV_FILE = "your_file.csv" if os.path.exists("your_file.csv") else DEFAULT_PATH
assert os.path.exists(CSV_FILE), f"Dataset not found at {CSV_FILE}"

TARGET_COL = "overall_mhq_score"
AGE_COL = "age"
SEX_COL = "biological_sex"  # Note: 'gender' is 100% null in this dataset; 'biological_sex' has 239,995 valid rows
INCOME_COL = "income_household"
SUB_SCORES = [
    "cognition_score",
    "adapt_resilience_score",
    "drive_motivation_score",
    "social_self_score",
]

print(f"✅ Dataset located: {CSV_FILE}")
lazy_df = pl.scan_csv(CSV_FILE, null_values=["N/A"])
schema = lazy_df.collect_schema()
row_count = lazy_df.select(pl.len()).collect().item()
print(f"Total Rows: {row_count:,} | Total Columns: {len(schema)}")

# Load only the research columns needed for modeling & statistical analysis
AUDIT_COLS = [TARGET_COL, AGE_COL, SEX_COL, INCOME_COL] + SUB_SCORES
df = lazy_df.select(AUDIT_COLS).collect()
df = df.with_row_index(name="participant_id")

print("--- 1. MISSINGNESS AUDIT ---")
print(df.select(AUDIT_COLS).null_count())

print("\n--- 2. DUPLICATE AUDIT (Research Subset) ---")
dupes = df.select(AUDIT_COLS).is_duplicated().sum()
print(f"Total exact duplicate rows in research subset: {dupes:,}")

print("\n--- 3. TARGET SCORE SUMMARY ---")
print(df.select(
    pl.col(TARGET_COL).min().alias("Min_Score"),
    pl.col(TARGET_COL).max().alias("Max_Score"),
    pl.col(TARGET_COL).mean().round(2).alias("Mean_Score"),
    pl.col(TARGET_COL).std().round(2).alias("Std_Score"),
))

print("\n--- 4. CATEGORICAL VALUE CHECK ---")
print("Unique Age groups:", df[AGE_COL].drop_nulls().unique().sort().to_list())
print("Unique Biological Sex values:", df[SEX_COL].drop_nulls().unique().sort().to_list())
print("Sample Income values:", df[INCOME_COL].drop_nulls().unique().head(5).to_list())

# Save numeric summary statistics to Full_Statistics.csv
numeric_df = df.select(cs.numeric())
stats_df = numeric_df.describe()
output_dir = os.path.dirname(os.path.abspath(CSV_FILE))
stats_df.write_csv(os.path.join(output_dir, "Full_Statistics.csv"))
print(f"✅ Saved Full_Statistics.csv ({stats_df.shape[0]} rows × {stats_df.shape[1]} cols)")
print(stats_df)

import scipy.stats as stats

# Prepare clean Pandas DataFrame for modeling & statistical tests
research_df = df.drop_nulls(subset=[TARGET_COL])
pdf = research_df.to_pandas()

print("--- ONE-WAY ANOVA: Demographic Factors vs. Overall MHQ Score ---")
anova_results = {}
for col in [AGE_COL, SEX_COL, INCOME_COL]:
    groups = [g[TARGET_COL].dropna() for _, g in pdf.groupby(col, observed=True)]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results[col] = (f_stat, p_val)
        sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
        print(f"{col:20} | F = {f_stat:10.2f} | p-value = {p_val:.4e} | {sig}")
    else:
        print(f"{col:20} | Not enough groups.")

# Chi-Square Test of Independence: Biological Sex vs. Household Income
df_chi = pdf[[SEX_COL, INCOME_COL]].dropna()
contingency_table = pd.crosstab(df_chi[SEX_COL], df_chi[INCOME_COL])
chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

print("--- CHI-SQUARE TEST: Biological Sex vs. Household Income ---")
print(f"Valid paired rows: {len(df_chi):,}")
print(f"Contingency Table Shape: {contingency_table.shape}")
print(f"Chi-Square Statistic: {chi2:.4f} | DoF: {dof} | p-value: {p_val:.4e}")
print("Result:", "SIGNIFICANT (dependent)" if p_val < 0.05 else "NOT SIGNIFICANT (independent)")

import matplotlib.pyplot as plt
import seaborn as sns

# Multicollinearity Check across encoded demographic inputs
valid_cols = [c for c in [AGE_COL, SEX_COL, INCOME_COL] if pdf[c].notna().any()]
corr_matrix = pdf[valid_cols].apply(lambda x: x.astype("category").cat.codes).corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Multicollinearity Check: Input Features")
plt.tight_layout()
plt.show()

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

categorical_features = [AGE_COL, SEX_COL, INCOME_COL]
X = pdf[categorical_features]
y = pdf[TARGET_COL]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)),
])

print("Training Random Forest Pipeline on Overall MHQ Score...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

print(f"\n--- MODEL RESULTS ({TARGET_COL}) ---")
print(f"RMSE:      {rmse:.4f}")
print(f"R-squared: {r2:.4f} ({r2:.2%} variance explained)")

# Save trained pipeline for Streamlit deployment
model_path = os.path.join(output_dir, "mhq_model.joblib")
joblib.dump(model, model_path)
print(f"✅ Saved trained model pipeline to: {model_path}")

from sklearn.base import clone

sub_results = []
sub_models = {}

for sub_target in SUB_SCORES:
    valid_idx = pdf[sub_target].notna()
    X_sub = X[valid_idx]
    y_sub = pdf.loc[valid_idx, sub_target]

    X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    sub_model = clone(model)
    sub_model.fit(X_tr, y_tr)

    preds = sub_model.predict(X_te)
    sub_r2 = float(r2_score(y_te, preds))
    sub_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    sub_results.append({"Target": sub_target, "R-squared": round(sub_r2, 4), "RMSE": round(sub_rmse, 4)})
    sub_models[sub_target] = sub_model

sub_results_df = pd.DataFrame(sub_results).sort_values(by="R-squared", ascending=False)
print("--- Sub-Score Predictability Matrix ---")
print(sub_results_df.to_string(index=False))

all_features = model.named_steps["preprocessor"].get_feature_names_out()
clean_features = [name.replace("cat__", "") for name in all_features]
importances = model.named_steps["regressor"].feature_importances_
feat_imp = pd.Series(importances, index=clean_features).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False)
plt.title("Top 10 Predictors of Overall MHQ Score")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, sub_target in enumerate(SUB_SCORES):
    sub_m = sub_models[sub_target]
    sub_feats = [n.replace("cat__", "") for n in sub_m.named_steps["preprocessor"].get_feature_names_out()]
    sub_imp = pd.Series(sub_m.named_steps["regressor"].feature_importances_, index=sub_feats).sort_values(ascending=False).head(5)

    sns.barplot(x=sub_imp.values, y=sub_imp.index, ax=axes[i], hue=sub_imp.index, palette="viridis", legend=False)
    axes[i].set_title(f"Top 5 Predictors: {sub_target}")
    axes[i].set_xlabel("Importance Score")

plt.tight_layout()
plt.show()

def predict_mhq(age, biological_sex, income):
    """
    Predict the Overall Mental Health Questionnaire (MHQ) score
    from raw demographic inputs (age, biological_sex, income_household).
    """
    input_data = pd.DataFrame({
        AGE_COL: [str(age)],
        SEX_COL: [str(biological_sex)],
        INCOME_COL: [str(income)],
    })
    prediction = model.predict(input_data)[0]
    return float(prediction)


# Verify model object and live predictions
print("Model Type:", type(model))
for sample_age, sample_sex, sample_inc in [
    ("18", "Female", "Prefer not to say"),
    ("55-64", "Male", "1|00|001 – 3|00|000 INR"),
]:
    score = predict_mhq(sample_age, sample_sex, sample_inc)
    print(f"Prediction for ({sample_age}, {sample_sex}, {sample_inc}) -> {score:.2f}")

