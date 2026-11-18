"""
GlucoSense - Diabetes Risk Studio
---------------------------------
Streamlit app that predicts diabetes risk from age, BMI, blood glucose
level and heart disease.

The model is pre-trained and stored in `model_bundle.joblib`
(created by `train_model.py`), so NO CSV file is needed to run or deploy.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Wedge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FEATURES = ["age", "bmi", "blood_glucose_level", "heart_disease"]
BUNDLE_FILE = Path(__file__).parent / "model_bundle.joblib"

BG = "#0b1220"
CARD = "#121a2b"
INK = "#e6edf7"
MUTED = "#8a97ad"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
CORAL = "#fb7185"
BLUE = "#60a5fa"

st.set_page_config(page_title="GlucoSense | Diabetes Risk Studio", page_icon="🩸", layout="wide")

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
    .empty {{
        text-align: center; padding: 70px 20px; border: 2px dashed #26334f;
        border-radius: 20px; color: {MUTED};
    }}
    .empty .big {{ font-size: 3rem; }}
    .note {{ color: {MUTED}; font-size: 0.82rem; }}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
        width: 100%; border-radius: 12px; font-weight: 700; padding: 0.65rem 1rem;
    }}
    div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
        background: linear-gradient(135deg, #14b8a6, #2563eb); color: white; border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def style_axes(ax):
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


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    return joblib.load(BUNDLE_FILE)


def risk_band(p: float):
    if p < 0.10:
        return "LOW RISK", TEAL
    if p < 0.40:
        return "MODERATE RISK", AMBER
    return "HIGH RISK", CORAL


def bmi_category(v: float):
    if v < 18.5:
        return "Underweight"
    if v < 25:
        return "Normal"
    if v < 30:
        return "Overweight"
    return "Obese"


def draw_gauge(p: float, color: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.set_aspect("equal")
    ax.axis("off")

    for lo, hi, c in [(0, 0.10, TEAL), (0.10, 0.40, AMBER), (0.40, 1.0, CORAL)]:
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
# Load model (no CSV needed)
# --------------------------------------------------------------------------
if not BUNDLE_FILE.exists():
    st.error(
        "`model_bundle.joblib` not found. Run `python train_model.py` once to create it, "
        "then keep it in the same folder as `app.py`."
    )
    st.stop()

bundle = load_bundle()
model, stats = bundle["model"], bundle["stats"]
y_test, test_proba = bundle["y_test"], bundle["test_proba"]
ranges = stats["ranges"]

# Session state defaults
st.session_state.setdefault("result", None)

# --------------------------------------------------------------------------
# Sidebar: patient form with Predict button
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🩸 GlucoSense")
st.sidebar.caption("Diabetes Risk Studio")

with st.sidebar.form("patient_form"):
    st.markdown("### Patient profile")
    age = st.slider("Age (years)", 1, 100, 40)
    bmi = st.slider("BMI (kg/m²)", 10.0, 60.0, 26.5, 0.1)
    glucose = st.slider("Blood glucose (mg/dL)", 70, 350, 120, 1)
    heart = st.toggle("Has heart disease", value=False)
    predict_clicked = st.form_submit_button("🔮 Predict Diabetes", type="primary")

if st.sidebar.button("↺ Reset result"):
    st.session_state["result"] = None
    st.rerun()

st.sidebar.markdown("### Decision threshold")
threshold = st.sidebar.slider(
    "Flag as diabetic when probability ≥",
    0.05, 0.90, 0.30, 0.05,
    help="Lower values catch more true cases (higher recall) but raise false alarms.",
)

if predict_clicked:
    patient = pd.DataFrame([[age, bmi, glucose, int(heart)]], columns=FEATURES)
    st.session_state["result"] = {
        "age": age,
        "bmi": bmi,
        "glucose": glucose,
        "heart": bool(heart),
        "prob": float(model.predict_proba(patient)[0, 1]),
    }

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>GlucoSense · Diabetes Risk Studio</h1>
        <p>Fill in the patient profile in the sidebar and press <b>Predict Diabetes</b> to get an instant risk estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_model = st.tabs(["🔮 Risk Prediction", "📊 Data Insights", "🧪 Model Report"])

# --------------------------------------------------------------------------
# Tab 1: Prediction
# --------------------------------------------------------------------------
with tab_predict:
    res = st.session_state["result"]

    if res is None:
        st.markdown(
            '<div class="empty"><div class="big">🩺</div>'
            "<h3>No prediction yet</h3>"
            "<p>Set the patient details in the sidebar, then click <b>Predict Diabetes</b>.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        prob = res["prob"]
        label, color = risk_band(prob)
        flagged = prob >= threshold

        left, right = st.columns([1, 1.25], gap="large")

        with left:
            show(draw_gauge(prob, color))
            st.markdown(
                f'<div style="text-align:center"><span class="pill" style="background:{color}22;color:{color};'
                f'border:1px solid {color}">{label}</span></div>',
                unsafe_allow_html=True,
            )

        with right:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="card"><div class="kpi-label">Age</div><div class="kpi-value">{res["age"]}</div></div>', unsafe_allow_html=True)
            c2.markdown(
                f'<div class="card"><div class="kpi-label">BMI · {bmi_category(res["bmi"])}</div>'
                f'<div class="kpi-value">{res["bmi"]:.1f}</div></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(f'<div class="card"><div class="kpi-label">Glucose</div><div class="kpi-value">{res["glucose"]}</div></div>', unsafe_allow_html=True)

            st.write("")
            verdict = "Likely diabetic" if flagged else "Likely non-diabetic"
            vcolor = CORAL if flagged else TEAL
            st.markdown(
                f'<div class="card"><div class="kpi-label">Model verdict at threshold {threshold:.0%}</div>'
                f'<div class="kpi-value" style="color:{vcolor}">{verdict}</div></div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown("**What drives this prediction?**")
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
            names = {"age": "Age", "bmi": "BMI", "blood_glucose_level": "Blood glucose", "heart_disease": "Heart disease"}
            fig, ax = new_fig(5.5, 2.2)
            ax.barh([names[i] for i in imp.index], imp.values, color=BLUE)
            ax.set_xlabel("Relative importance")
            ax.grid(axis="y", visible=False)
            show(fig)

        st.markdown("### What-if explorer: how does glucose change the risk?")
        sweep = pd.DataFrame(
            {
                "age": res["age"],
                "bmi": res["bmi"],
                "blood_glucose_level": np.arange(80, 301, 5),
                "heart_disease": int(res["heart"]),
            }
        )
        sweep_p = model.predict_proba(sweep[FEATURES])[:, 1]
        fig, ax = new_fig(10, 3.2)
        ax.plot(sweep["blood_glucose_level"], sweep_p, color=TEAL, linewidth=2.6)
        ax.fill_between(sweep["blood_glucose_level"], sweep_p, color=TEAL, alpha=0.12)
        ax.axhline(threshold, color=AMBER, linestyle="--", linewidth=1.2)
        ax.text(82, min(threshold + 0.03, 0.97), "decision threshold", color=AMBER, fontsize=8)
        ax.scatter([res["glucose"]], [prob], color=color, s=90, zorder=5, edgecolor=INK)
        ax.set_xlabel("Blood glucose level (mg/dL)")
        ax.set_ylabel("Diabetes probability")
        ax.set_ylim(0, 1.02)
        show(fig)
        st.caption("Age, BMI and heart disease are held at the submitted values. The dot marks this patient.")

        # Downloadable report
        report = pd.DataFrame(
            [
                {
                    "age": res["age"],
                    "bmi": res["bmi"],
                    "blood_glucose_level": res["glucose"],
                    "heart_disease": int(res["heart"]),
                    "diabetes_probability": round(prob, 4),
                    "risk_band": label,
                    "verdict": verdict,
                }
            ]
        )
        st.download_button(
            "⬇ Download result (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_prediction_result.csv",
            mime="text/csv",
        )

    st.markdown(
        '<p class="note">⚠️ Educational machine-learning project. Not a medical device and not a substitute for professional diagnosis.</p>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Tab 2: Data insights (from pre-computed statistics)
# --------------------------------------------------------------------------
with tab_insights:
    k1, k2, k3, k4 = st.columns(4)
    for col, label_, value in [
        (k1, "Training records", f"{stats['records']:,}"),
        (k2, "Diabetic cases", f"{stats['diabetic']:,}"),
        (k3, "Prevalence", f"{stats['prevalence']:.1%}"),
        (k4, "Heart disease", f"{stats['heart_rate']:.1%}"),
    ]:
        col.markdown(f'<div class="card"><div class="kpi-label">{label_}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("**Blood glucose by diabetes status**")
        edges = stats["glucose_bins"]
        widths = np.diff(edges)
        fig, ax = new_fig()
        ax.bar(edges[:-1], stats["hist_neg"], width=widths, align="edge", color=TEAL, alpha=0.75, label="Non-diabetic")
        ax.bar(edges[:-1], stats["hist_pos"], width=widths, align="edge", color=CORAL, alpha=0.75, label="Diabetic")
        ax.set_xlabel("Blood glucose level (mg/dL)")
        ax.set_ylabel("Density")
        ax.legend(facecolor=CARD, edgecolor="#26334f", labelcolor=INK)
        show(fig)

    with b:
        st.markdown("**Diabetes prevalence by age group**")
        prev = stats["prevalence_by_age"]
        fig, ax = new_fig()
        ax.bar(list(prev.keys()), list(prev.values()), color=BLUE)
        ax.set_ylabel("Prevalence (%)")
        ax.grid(axis="x", visible=False)
        show(fig)

    c, d = st.columns(2, gap="large")

    with c:
        st.markdown("**BMI by diabetes status**")
        fig, ax = new_fig()
        bp = ax.bxp(
            stats["bmi_box"],
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
        show(fig)

    with d:
        st.markdown("**Correlation with diabetes**")
        corr = pd.Series(stats["correlation"]).sort_values()
        fig, ax = new_fig()
        ax.barh(corr.index, corr.values, color=[CORAL if v > 0 else TEAL for v in corr.values])
        ax.set_xlabel("Pearson correlation")
        ax.grid(axis="y", visible=False)
        show(fig)

# --------------------------------------------------------------------------
# Tab 3: Model report
# --------------------------------------------------------------------------
with tab_model:
    y_pred = (test_proba >= threshold).astype(int)

    cols = st.columns(5)
    for col, label_, value in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"],
        [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, test_proba),
        ],
    ):
        col.markdown(f'<div class="card"><div class="kpi-label">{label_}</div><div class="kpi-value">{value:.3f}</div></div>', unsafe_allow_html=True)

    st.caption(f"Computed on a held-out 20% test split at the current threshold ({threshold:.0%}). Move the sidebar slider to see the precision/recall trade-off.")

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
        show(fig)

    with f:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = new_fig(5, 3.8)
        ax.plot(fpr, tpr, color=TEAL, linewidth=2.6, label=f"AUC = {roc_auc_score(y_test, test_proba):.3f}")
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(facecolor=CARD, edgecolor="#26334f", labelcolor=INK, loc="lower right")
        show(fig)

    st.markdown(
        '<p class="note">Model: Random Forest (200 trees, max depth 8, min leaf size 20) trained on age, BMI, blood glucose level and heart disease.</p>',
        unsafe_allow_html=True,
    )
