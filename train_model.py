"""
Student Performance Prediction System
Downloads the UCI Mathematics Student Performance dataset, trains a Random Forest
classifier, evaluates it, and saves the trained model and supporting files.
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
SAMPLE_PATH = BASE_DIR / "sample_prediction_input.csv"


def download_dataset() -> pd.DataFrame:
    """Fetch UCI dataset 320 and retain the Mathematics records."""
    print("Downloading UCI Student Performance dataset (ID 320)...")
    student = fetch_ucirepo(id=320)
    features = student.data.features
    targets = student.data.targets
    df = pd.concat([features, targets], axis=1)

    required = {"G1", "G2", "G3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected grade columns: {sorted(missing)}")

    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved: {DATA_PATH}")
    print("Rows and columns:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train a leakage-safe classification pipeline and save its artifacts."""
    # 1 = AT RISK when G3 < 10; 0 = NOT AT RISK when G3 >= 10.
    y = (df["G3"] < 10).astype(int)

    # G3 is the outcome being predicted and must not be an input feature.
    X = df.drop(columns=["G3"])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[1],
        average=None,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "at_risk_precision": round(float(precision[0]), 4),
        "at_risk_recall": round(float(recall[0]), 4),
        "at_risk_f1": round(float(f1[0]), 4),
        "classification_report": report,
        "confusion_matrix_labels": ["Not at risk", "At risk"],
        "confusion_matrix": matrix.tolist(),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features_used": list(X.columns),
        "target_rule": "G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK",
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    # A small valid CSV template with exactly the predictor columns.
    X.head(5).to_csv(SAMPLE_PATH, index=False)

    print("\nMODEL EVALUATION")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"At-risk precision: {precision[0]:.3f}")
    print(f"At-risk recall: {recall[0]:.3f}")
    print(f"At-risk F1-score: {f1[0]:.3f}")
    print("\nClassification report:\n")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved sample input: {SAMPLE_PATH}")


def main():
    try:
        df = download_dataset()
        train_and_evaluate(df)
        print("\nSUCCESS: Training is complete. You can now run: streamlit run app.py")
    except Exception as exc:
        print("\nERROR:", exc)
        print(
            "\nCheck your internet connection and package installation, then run "
            "'pip install -r requirements.txt' and try again."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
