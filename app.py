"""Streamlit interface for the Student Performance Prediction System."""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "student_data.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Prediction System")
st.caption(
    "Classroom prototype that estimates academic risk from historical student indicators."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("The model or dataset is not ready yet.")
    st.markdown(
        "### First-time setup\n"
        "1. Open a terminal in this project folder.\n"
        "2. Run `pip install -r requirements.txt`.\n"
        "3. Run `python train_model.py` and wait until training finishes.\n"
        "4. Run `streamlit run app.py`."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()

if "G3" not in data.columns:
    st.error("The training dataset does not contain the expected G3 target column.")
    st.stop()

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
else:
    metrics = {}

# Overview
st.header("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Historical records", f"{len(data):,}")
col2.metric("Average final grade (G3)", f"{data['G3'].mean():.2f}")
col3.metric("Historical at-risk records", f"{int((data['G3'] < 10).sum()):,}")
col4.metric("Predictor columns", str(len(data.columns) - 1))

with st.expander("View dataset preview"):
    st.dataframe(data.head(10), use_container_width=True)

if metrics:
    st.subheader("Model evaluation (held-out test data)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    m2.metric("At-risk precision", f"{metrics.get('at_risk_precision', 0):.1%}")
    m3.metric("At-risk recall", f"{metrics.get('at_risk_recall', 0):.1%}")
    m4.metric("At-risk F1-score", f"{metrics.get('at_risk_f1', 0):.1%}")
    st.caption(
        "Metrics are based on a held-out test split. Review precision, recall, and F1-score; "
        "accuracy alone does not describe all model errors."
    )

st.divider()
st.header("Predict uploaded student records")
st.write(
    "Upload a CSV containing the same predictor columns used during training. "
    "Do not include G3; it is the outcome being predicted."
)

required_columns = list(model.feature_names_in_)
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "⬇️ Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose a student CSV file", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        st.write(f"Uploaded records: **{len(new_data)}**")

        if new_data.empty:
            st.warning("The uploaded CSV has no data rows.")
        else:
            missing = [column for column in required_columns if column not in new_data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                # Ignore extra columns and keep the exact training feature order.
                prediction_input = new_data[required_columns].copy()
                predictions = model.predict(prediction_input)

                results = new_data.copy()
                results["Prediction"] = [
                    "AT RISK" if int(value) == 1 else "NOT AT RISK"
                    for value in predictions
                ]

                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True)

                at_risk_count = int((predictions == 1).sum())
                not_at_risk_count = int((predictions == 0).sum())
                r1, r2, r3 = st.columns(3)
                r1.metric("Total predictions", len(predictions))
                r2.metric("AT RISK", at_risk_count)
                r3.metric("NOT AT RISK", not_at_risk_count)

                st.download_button(
                    "⬇️ Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="student_predictions.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(f"Could not read or process this CSV file: {exc}")

st.divider()
st.subheader("Important responsible-use note")
st.warning(
    "This is a classroom demonstration using public historical data. A prediction is not "
    "a diagnosis or a final judgment about a learner. Faculty should review the wider "
    "context and speak with the student before deciding what support may be helpful."
)

st.caption(
    "Target rule: G3 < 10 = AT RISK; G3 ≥ 10 = NOT AT RISK. "
    "The G3 column is excluded from model input to prevent target leakage."
)
