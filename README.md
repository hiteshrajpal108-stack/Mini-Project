# MHQ Research Project — Mental Health Questionnaire Prediction

## Files Included
- `PythonProjectNew.ipynb` — Complete, error-free data profiling, statistical testing (ANOVA, Chi-Square, multicollinearity), Random Forest modeling, feature importance, and live prediction notebook.
- `app.py` — Ready-to-deploy Streamlit dashboard for live MHQ prediction, demographic exploration, and feature importance visualization.
- `mhq_model.joblib` — Pre-trained Scikit-Learn `Pipeline` (`SimpleImputer` → `OneHotEncoder` → `RandomForestRegressor`) for instant inference.
- `Full_Statistics.csv` — Summary statistics exported from the 240,012-row dataset.
- `requirements.txt` — Python package dependencies for deployment.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
