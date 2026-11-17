
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "anomaly_scaler.joblib"

METRICS_PATH = REPORT_DIR / "isolation_forest_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "anomaly_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
]

TARGET = "is_anomaly"


# ============================================================
# 1. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():

    print("\n===== LOADING DATASET =====")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [
        TARGET,
        "anomaly_type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Validate only actual model features and target.
    # anomaly_type is intentionally null for normal rows.

    model_columns = FEATURES + [TARGET]

    missing_values = df[model_columns].isnull().sum()

    if missing_values.any():
        raise ValueError(
            f"Missing values in model columns:\n{missing_values}"
        )

    # Validate target values.

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "Dataset must contain both normal and anomaly observations."
        )

    # Normal observations should have no anomaly type.
    # Anomaly observations should have a fault type.

    expected_missing = df[TARGET].eq(0)

    if not df["anomaly_type"].isnull().equals(
        expected_missing
    ):
        raise ValueError(
            "Anomaly labels and anomaly_type values are inconsistent."
        )

    print(f"Dataset loaded: {df.shape}")
    print("Model feature validation: PASSED")
    print("Anomaly label validation: PASSED")

    return df


# ============================================================
# 2. PREPARE DATA
# ============================================================

def prepare_data(df):

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# 3. TRAIN ISOLATION FOREST
# ============================================================

def train_model(X_train_scaled):

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_scaled)

    return model


# ============================================================
# 4. THRESHOLD CALIBRATION
# ============================================================

def select_threshold(
    model,
    X_train_scaled,
    y_train,
):

    """
    Calibrate the anomaly-score threshold using training labels.

    Isolation Forest itself is unsupervised.
    Threshold calibration uses synthetic training labels.

    The held-out test labels are not used here.
    """

    scores = -model.score_samples(
        X_train_scaled
    )

    thresholds = np.quantile(
        scores,
        np.linspace(0.50, 0.99, 100),
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        score = f1_score(
            y_train,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test_scaled,
    y_test,
    threshold,
):

    anomaly_scores = -model.score_samples(
        X_test_scaled
    )

    predictions = (
        anomaly_scores >= threshold
    ).astype(int)

    metrics = {

        "model": "Isolation Forest",

        "evaluation_type": (
            "Unsupervised model with supervised "
            "training-set threshold calibration"
        ),

        "threshold": float(threshold),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                anomaly_scores,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_test,
                anomaly_scores,
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),

        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    return (
        metrics,
        predictions,
        anomaly_scores,
    )


# ============================================================
# 6. SAVE MODEL AND REPORTS
# ============================================================

def save_outputs(
    model,
    scaler,
    metrics,
    X_test,
    y_test,
    predictions,
    anomaly_scores,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    prediction_df = X_test.copy()

    prediction_df["actual_anomaly"] = (
        y_test.values
    )

    prediction_df["predicted_anomaly"] = (
        predictions
    )

    prediction_df["anomaly_score"] = (
        anomaly_scores
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\n===== SAVED ARTIFACTS =====")

    print(f"Model: {MODEL_PATH}")

    print(f"Scaler: {SCALER_PATH}")

    print(f"Metrics: {METRICS_PATH}")

    print(f"Predictions: {PREDICTIONS_PATH}")


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 75)

    print(
        "INDUSTRIAL ANOMALY DETECTION - ISOLATION FOREST"
    )

    print("=" * 75)

    # Load dataset.

    df = load_dataset()

    print("\n===== DATASET SUMMARY =====")

    print(f"Observations: {len(df)}")

    print(f"Features: {len(FEATURES)}")

    print(
        f"Normal observations: "
        f"{int((df[TARGET] == 0).sum())}"
    )

    print(
        f"Anomaly observations: "
        f"{int((df[TARGET] == 1).sum())}"
    )

    # Prepare train and test data.

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = prepare_data(df)

    print("\n===== DATA SPLIT =====")

    print(
        f"Training observations: {len(X_train)}"
    )

    print(
        f"Testing observations: {len(X_test)}"
    )

    # Train model.

    print("\n===== TRAINING MODEL =====")

    model = train_model(
        X_train_scaled
    )

    print("Isolation Forest training completed.")

    # Calibrate threshold.

    print("\n===== THRESHOLD CALIBRATION =====")

    threshold, calibration_f1 = select_threshold(
        model,
        X_train_scaled,
        y_train,
    )

    print(
        f"Selected threshold: {threshold:.6f}"
    )

    print(
        f"Training calibration F1: {calibration_f1:.4f}"
    )

    # Evaluate model.

    print("\n===== TEST PERFORMANCE =====")

    (
        metrics,
        predictions,
        anomaly_scores,
    ) = evaluate_model(
        model,
        X_test_scaled,
        y_test,
        threshold,
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1-score: {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: {metrics['pr_auc']:.4f}"
    )

    print("\n===== CONFUSION MATRIX =====")

    print(
        np.array(metrics["confusion_matrix"])
    )

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save artifacts.

    save_outputs(
        model,
        scaler,
        metrics,
        X_test,
        y_test,
        predictions,
        anomaly_scores,
    )

    print("\n===== FINAL RESULT =====")

    print(
        "ISOLATION FOREST TRAINING COMPLETED"
    )


if __name__ == "__main__":

    main()
