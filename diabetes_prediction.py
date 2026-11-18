"""
Run this ONCE on your computer (where the CSV exists):

    python train_model.py

It trains the model and saves everything the app needs into
`model_bundle.joblib`. After that, the Streamlit app (and your deployment)
does NOT need the CSV file anymore.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["age", "bmi", "blood_glucose_level", "heart_disease"]
TARGET = "diabetes"

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diabetes_prediction_dataset.csv")
df = pd.read_csv(csv_path)

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)

# ---- Pre-computed statistics for the "Data Insights" tab ----
glucose_bins = np.linspace(df["blood_glucose_level"].min(), df["blood_glucose_level"].max(), 31)
hist_neg, _ = np.histogram(df.loc[df[TARGET] == 0, "blood_glucose_level"], bins=glucose_bins, density=True)
hist_pos, _ = np.histogram(df.loc[df[TARGET] == 1, "blood_glucose_level"], bins=glucose_bins, density=True)

age_bins = [0, 20, 30, 40, 50, 60, 70, 81]
age_labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
prevalence_by_age = (
    df.assign(age_group=pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False))
    .groupby("age_group", observed=True)[TARGET]
    .mean()
    .mul(100)
    .round(3)
    .to_dict()
)


def box_stats(series):
    q = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"whislo": q[0.05], "q1": q[0.25], "med": q[0.5], "q3": q[0.75], "whishi": q[0.95], "fliers": []}


corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).to_dict()

stats = {
    "records": int(len(df)),
    "diabetic": int(df[TARGET].sum()),
    "prevalence": float(df[TARGET].mean()),
    "heart_rate": float(df["heart_disease"].mean()),
    "glucose_bins": glucose_bins,
    "hist_neg": hist_neg,
    "hist_pos": hist_pos,
    "prevalence_by_age": prevalence_by_age,
    "bmi_box": [box_stats(df.loc[df[TARGET] == 0, "bmi"]), box_stats(df.loc[df[TARGET] == 1, "bmi"])],
    "correlation": corr,
    "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES},
}

bundle = {
    "model": model,
    "stats": stats,
    "y_test": y_test.to_numpy(),
    "test_proba": model.predict_proba(X_test)[:, 1].astype("float32"),
}

joblib.dump(bundle, "model_bundle.joblib", compress=3)
print("Saved model_bundle.joblib  ->", round(Path("model_bundle.joblib").stat().st_size / 1e6, 2), "MB")
