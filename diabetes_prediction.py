"""
GlucoSense - Diabetes Risk Studio
---------------------------------
A Streamlit app that predicts diabetes risk from four clinical features:
age, BMI, blood glucose level and heart disease.

Run:
    pip install streamlit pandas scikit-learn matplotlib
    streamlit run app.py

Keep `diabetes_prediction_dataset.csv` in the same folder as this file
(or upload it from the sidebar).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Wedge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FEATURES = ["age", "bmi", "blood_glucose_level", "heart_disease"]
TARGET = "diabetes"
DATA_FILE = Path(__file__).parent / "diabetes_prediction_dataset.csv"

# Colour palette (dark "clinical night" theme)
BG = "#0b1220"
CARD = "#121a2b"
INK = "#e6edf7"
MUTED = "#8a97ad"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
CORAL = "#fb7185"
BLUE = "#60a5fa"

st.set_page_config(page_title="GlucoSense | Diabetes Risk Studio", page_icon="🩸", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, {BG} 55%); color: {INK}; }}
    section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid #1f2a44; }}
    h1, h2, h3, h4, label, p, span, div {{ color: {INK}; }}
    .hero {{
        padding: 28px 32px; border-radius: 20px;
        background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
        box-shadow: 0 12px 40px rgba(29, 78, 216, 0.25);
        margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0; font-size: 2.3rem; letter-spacing: 0.5px; color: white; }}
    .hero p {{ margin: 6px 0 0 0; opacity: 0.9; font-size: 1.02rem; color: white; }}
    .card {{
        background: {CARD}; border: 1px solid #1f2a44; border-radius: 16px;
        padding: 18px 20px; height: 100%;
    }}
    .kpi-label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 700; margin-top: 2px; }}
    .pill {{
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.6px;
    }}
    .note {{ color: {MUTED}; font-size: 0.82rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_axes(ax):
    """Apply the dark theme to a matplotlib axis."""
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#26334f")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(color="#1c2740", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def new_fig(w=6, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    style_axes(ax)
    return fig, ax


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(source):
    return pd.read_csv(source)


@st.cache_resource(show_spinner="Training the model on the dataset...")
def train(df: pd.DataFrame):
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
    )
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    return model, y_test.to_numpy(), test_proba


def risk_band(p: float):
    if p < 0.10:
        return "LOW RISK", TEAL
    if p < 0.40:
        return "MODERATE RISK", AMBER
    return "HIGH RISK", CORAL


def draw_gauge(p: float, color: str):
    """Half-circle risk gauge."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.set_aspect("equal")
    ax.axis("off")

    bands = [(0, 0.10, TEAL), (0.10, 0.40, AMBER), (0.40, 1.0, CORAL)]
    for lo, hi, c in bands:
        ax.add_patch(Wedge((0, 0), 1.0, 180 - hi * 180, 180 - lo * 180, width=0.22, color=c, alpha=0.30))

    ax.add_patch(Wedge((0, 0), 1.0, 180 - p * 180, 180, width=0.22, color=color))

    angle = np.pi * (1 - p)
    ax.plot([0, 0.72 * np.cos(angle)], [0, 0.72 * np.sin(angle)], color=INK, linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color=INK))

    ax.text(0, -0.28, f"{p:.1%}", ha="center", va="center", fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.48, "estimated probability", ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.65, 1.1)
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🩸 GlucoSense")
st.sidebar.caption("Diabetes Risk Studio")

uploaded = None
if not DATA_FILE.exists():
    uploaded = st.sidebar.file_uploader("Upload diabetes_prediction_dataset.csv", type="csv")

if DATA_FILE.exists():
    data = load_data(DATA_FILE)
elif uploaded is not None:
    data = load_data(uploaded)
else:
    st.markdown(
        '<div class="hero"><h1>GlucoSense</h1><p>Diabetes Risk Studio</p></div>',
        unsafe_allow_html=True,
    )
    st.warning("Dataset not found. Place `diabetes_prediction_dataset.csv` next to `app.py`, or upload it from the sidebar.")
    st.stop()

model, y_test, test_proba = train(data)

st.sidebar.markdown("### Patient profile")
age = st.sidebar.slider("Age (years)", 1, 100, 40)
bmi = st.sidebar.slider("BMI (kg/m²)", 10.0, 60.0, 26.5, 0.1)
glucose = st.sidebar.slider("Blood glucose (mg/dL)", 70, 350, 120, 1)
heart = st.sidebar.toggle("Has heart disease", value=False)

st.sidebar.markdown("### Decision threshold")
threshold = st.sidebar.slider(
    "Flag as diabetic when probability ≥",
    0.05, 0.90, 0.30, 0.05,
    help="Lower values catch more true cases (higher recall) but raise false alarms.",
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>GlucoSense · Diabetes Risk Studio</h1>
        <p>Enter a patient profile in the sidebar and get an instant, explainable diabetes risk estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_model = st.tabs(["🔮 Risk Prediction", "📊 Data Insights", "🧪 Model Report"])

# --------------------------------------------------------------------------
# Tab 1 - Prediction
# --------------------------------------------------------------------------
with tab_predict:
    patient = pd.DataFrame([[age, bmi, glucose, int(heart)]], columns=FEATURES)
    prob = float(model.predict_proba(patient)[0, 1])
    label, color = risk_band(prob)
    flagged = prob >= threshold

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.pyplot(draw_gauge(prob, color), use_container_width=True)
        st.markdown(
            f'<div style="text-align:center"><span class="pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}">{label}</span></div>',
            unsafe_allow_html=True,
        )

    with right:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card"><div class="kpi-label">Age</div><div class="kpi-value">{age}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><div class="kpi-label">BMI</div><div class="kpi-value">{bmi:.1f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><div class="kpi-label">Glucose</div><div class="kpi-value">{glucose}</div></div>', unsafe_allow_html=True)

        st.write("")
        verdict = "Likely diabetic" if flagged else "Likely non-diabetic"
        verdict_color = CORAL if flagged else TEAL
        st.markdown(
            f'<div class="card"><div class="kpi-label">Model verdict at threshold {threshold:.0%}</div>'
            f'<div class="kpi-value" style="color:{verdict_color}">{verdict}</div></div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**What drives this prediction?**")
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        names = {"age": "Age", "bmi": "BMI", "blood_glucose_level": "Blood glucose", "heart_disease": "Heart disease"}
        fig, ax = new_fig(5.5, 2.2)
        ax.barh([names[i] for i in importances.index], importances.values, color=BLUE)
        ax.set_xlabel("Relative importance")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)

    st.markdown("### What-if explorer: how does glucose change the risk?")
    sweep = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "blood_glucose_level": np.arange(80, 301, 5),
            "heart_disease": int(heart),
        }
    )
    sweep_p = model.predict_proba(sweep[FEATURES])[:, 1]
    fig, ax = new_fig(10, 3.2)
    ax.plot(sweep["blood_glucose_level"], sweep_p, color=TEAL, linewidth=2.6)
    ax.fill_between(sweep["blood_glucose_level"], sweep_p, color=TEAL, alpha=0.12)
    ax.axhline(threshold, color=AMBER, linestyle="--", linewidth=1.2)
    ax.text(82, min(threshold + 0.03, 0.97), "decision threshold", color=AMBER, fontsize=8)
    ax.scatter([glucose], [prob], color=color, s=90, zorder=5, edgecolor=INK)
    ax.set_xlabel("Blood glucose level (mg/dL)")
    ax.set_ylabel("Diabetes probability")
    ax.set_ylim(0, 1.02)
    st.pyplot(fig, use_container_width=True)
    st.caption("Age, BMI and heart disease are held at the values chosen in the sidebar. The dot marks the current patient.")

    st.markdown(
        '<p class="note">⚠️ Educational machine-learning project. This tool is not a medical device and does not replace professional diagnosis.</p>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Tab 2 - Data insights
# --------------------------------------------------------------------------
with tab_insights:
    total = len(data)
    pos_rate = data[TARGET].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, label_, value in [
        (k1, "Records", f"{total:,}"),
        (k2, "Diabetic cases", f"{int(data[TARGET].sum()):,}"),
        (k3, "Prevalence", f"{pos_rate:.1%}"),
        (k4, "Heart disease", f"{data['heart_disease'].mean():.1%}"),
    ]:
        col.markdown(f'<div class="card"><div class="kpi-label">{label_}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("**Blood glucose by diabetes status**")
        fig, ax = new_fig()
        ax.hist(data.loc[data[TARGET] == 0, "blood_glucose_level"], bins=30, color=TEAL, alpha=0.75, label="Non-diabetic", density=True)
        ax.hist(data.loc[data[TARGET] == 1, "blood_glucose_level"], bins=30, color=CORAL, alpha=0.75, label="Diabetic", density=True)
        ax.set_xlabel("Blood glucose level (mg/dL)")
        ax.set_ylabel("Density")
        ax.legend(facecolor=CARD, edgecolor="#26334f", labelcolor=INK)
        st.pyplot(fig, use_container_width=True)

    with b:
        st.markdown("**Diabetes prevalence by age group**")
        bins = [0, 20, 30, 40, 50, 60, 70, 81]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        grouped = (
            data.assign(age_group=pd.cut(data["age"], bins=bins, labels=labels, right=False))
            .groupby("age_group", observed=True)[TARGET]
            .mean()
        )
        fig, ax = new_fig()
        ax.bar(grouped.index.astype(str), grouped.values * 100, color=BLUE)
        ax.set_ylabel("Prevalence (%)")
        ax.grid(axis="x", visible=False)
        st.pyplot(fig, use_container_width=True)

    c, d = st.columns(2, gap="large")

    with c:
        st.markdown("**BMI by diabetes status**")
        fig, ax = new_fig()
        bp = ax.boxplot(
            [data.loc[data[TARGET] == 0, "bmi"], data.loc[data[TARGET] == 1, "bmi"]],
            patch_artist=True,
            widths=0.5,
            showfliers=False,
            medianprops=dict(color=INK, linewidth=2),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
        )
        for patch, col_ in zip(bp["boxes"], [TEAL, CORAL]):
            patch.set_facecolor(col_)
            patch.set_alpha(0.8)
        ax.set_xticklabels(["Non-diabetic", "Diabetic"])
        ax.set_ylabel("BMI")
        st.pyplot(fig, use_container_width=True)

    with d:
        st.markdown("**Correlation with diabetes**")
        corr = data[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
        fig, ax = new_fig()
        ax.barh(corr.index, corr.values, color=[CORAL if v > 0 else TEAL for v in corr.values])
        ax.set_xlabel("Pearson correlation")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Preview raw data"):
        st.dataframe(data[FEATURES + [TARGET]].head(100), use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3 - Model report
# --------------------------------------------------------------------------
with tab_model:
    y_pred = (test_proba >= threshold).astype(int)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label_, value in [
        (m1, "Accuracy", accuracy_score(y_test, y_pred)),
        (m2, "Precision", precision_score(y_test, y_pred, zero_division=0)),
        (m3, "Recall", recall_score(y_test, y_pred)),
        (m4, "F1 score", f1_score(y_test, y_pred)),
        (m5, "ROC AUC", roc_auc_score(y_test, test_proba)),
    ]:
        col.markdown(f'<div class="card"><div class="kpi-label">{label_}</div><div class="kpi-value">{value:.3f}</div></div>', unsafe_allow_html=True)

    st.caption(f"Metrics are computed on a held-out 20% test split at the current decision threshold ({threshold:.0%}). Move the threshold slider in the sidebar to see the precision/recall trade-off.")

    e, f = st.columns(2, gap="large")

    with e:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = new_fig(5, 3.8)
        ax.grid(False)
        ax.imshow(cm, cmap="viridis")
        ax.set_xticks([0, 1], ["Pred. non-diabetic", "Pred. diabetic"])
        ax.set_yticks([0, 1], ["Non-diabetic", "Diabetic"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    with f:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = new_fig(5, 3.8)
        ax.plot(fpr, tpr, color=TEAL, linewidth=2.6, label=f"AUC = {roc_auc_score(y_test, test_proba):.3f}")
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(facecolor=CARD, edgecolor="#26334f", labelcolor=INK, loc="lower right")
        st.pyplot(fig, use_container_width=True)

    st.markdown(
        '<p class="note">Model: Random Forest (200 trees, max depth 8, min leaf size 20) trained on age, BMI, blood glucose level and heart disease.</p>',
        unsafe_allow_html=True,
    )
