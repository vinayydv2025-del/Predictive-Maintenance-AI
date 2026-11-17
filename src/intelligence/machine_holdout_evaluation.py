
from pathlib import Path
import json

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


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

MODEL_PATH = ROOT / "models" / "isolation_forest_machine_holdout.joblib"
SCALER_PATH = ROOT / "models" / "anomaly_scaler_machine_holdout.joblib"

METRICS_PATH = ROOT / "reports" / "machine_holdout_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "machine_holdout_predictions.csv"


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
]

TARGET = "is_anomaly"

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_dataset():
    print("\n" + "=" * 65)
    print("MACHINE-HELD-OUT ANOMALY DETECTION EVALUATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [
        TARGET,
        "machine_id",
        "machine_type",
        "cycle",
        "anomaly_type",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Missing values found in model columns.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Both normal and anomaly classes are required.")

    if df["machine_id"].isnull().any():
        raise ValueError("Missing machine IDs found.")

    if df.duplicated(["machine_id", "cycle"]).any():
        raise ValueError("Duplicate machine-cycle combinations found.")

    print(f"\nDataset shape: {df.shape}")
    print(f"Total observations: {len(df):,}")
    print(f"Unique machines: {df['machine_id'].nunique()}")
    print(f"Normal observations: {(df[TARGET] == 0).sum():,}")
    print(f"Anomaly observations: {(df[TARGET] == 1).sum():,}")

    return df


# ---------------------------------------------------------
# MACHINE-LEVEL DATA SPLITTING
# ---------------------------------------------------------

def split_by_machine(df):
    machine_ids = sorted(df["machine_id"].unique())

    if len(machine_ids) < 8:
        raise ValueError(
            "At least 8 unique machines are required for this experiment."
        )

    # 75% of machines for model development, 25% for final testing.
    development_machines, test_machines = train_test_split(
        machine_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    # Within development machines:
    # 80% for model fitting, 20% for threshold calibration.
    fit_machines, calibration_machines = train_test_split(
        development_machines,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    fit_df = df[df["machine_id"].isin(fit_machines)].copy()
    calibration_df = df[
        df["machine_id"].isin(calibration_machines)
    ].copy()
    test_df = df[df["machine_id"].isin(test_machines)].copy()

    fit_ids = set(fit_machines)
    calibration_ids = set(calibration_machines)
    test_ids = set(test_machines)

    # Verify that no machine appears in multiple groups.
    assert fit_ids.isdisjoint(calibration_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)

    print("\n" + "-" * 65)
    print("MACHINE-LEVEL SPLIT")
    print("-" * 65)

    print(f"\nModel-fitting machines ({len(fit_ids)}):")
    print(sorted(fit_ids))

    print(f"\nThreshold-calibration machines ({len(calibration_ids)}):")
    print(sorted(calibration_ids))

    print(f"\nUnseen test machines ({len(test_ids)}):")
    print(sorted(test_ids))

    print("\nObservation counts:")
    print(f"Model fitting: {len(fit_df):,}")
    print(f"Calibration:   {len(calibration_df):,}")
    print(f"Final testing: {len(test_df):,}")

    print("\nMachine overlap check: PASSED")

    return fit_df, calibration_df, test_df, {
        "model_fitting_machines": sorted(fit_ids),
        "calibration_machines": sorted(calibration_ids),
        "test_machines": sorted(test_ids),
    }


# ---------------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------------

def prepare_features(fit_df, calibration_df, test_df):
    scaler = StandardScaler()

    X_fit = scaler.fit_transform(fit_df[FEATURES])
    X_calibration = scaler.transform(calibration_df[FEATURES])
    X_test = scaler.transform(test_df[FEATURES])

    y_calibration = calibration_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    print("\nFeature scaling completed.")
    print("Scaler fitted only on model-fitting machines.")

    return scaler, X_fit, X_calibration, X_test, y_calibration, y_test


# ---------------------------------------------------------
# TRAIN ISOLATION FOREST
# ---------------------------------------------------------

def train_model(X_fit):
    print("\n" + "-" * 65)
    print("TRAINING ISOLATION FOREST")
    print("-" * 65)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_fit)

    print("Isolation Forest training completed.")

    return model


# ---------------------------------------------------------
# CALIBRATE THRESHOLD
# ---------------------------------------------------------

def calibrate_threshold(model, X_calibration, y_calibration):
    print("\n" + "-" * 65)
    print("THRESHOLD CALIBRATION")
    print("-" * 65)

    # Higher values represent more anomalous observations.
    calibration_scores = -model.score_samples(X_calibration)

    candidate_thresholds = np.unique(
        np.quantile(
            calibration_scores,
            np.linspace(0.50, 0.99, 100),
        )
    )

    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in candidate_thresholds:
        predictions = (
            calibration_scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_calibration,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_calibration,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_calibration,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_precision = float(precision)
            best_recall = float(recall)

    print(f"Selected threshold: {best_threshold:.6f}")
    print(f"Calibration precision: {best_precision:.4f}")
    print(f"Calibration recall: {best_recall:.4f}")
    print(f"Calibration F1: {best_f1:.4f}")

    return best_threshold, {
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


# ---------------------------------------------------------
# EVALUATE ON UNSEEN MACHINES
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold,
    test_df,
):
    print("\n" + "=" * 65)
    print("FINAL EVALUATION ON UNSEEN MACHINES")
    print("=" * 65)

    test_scores = -model.score_samples(X_test)

    predictions = (
        test_scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(y_test, test_scores)

    pr_auc = average_precision_score(
        y_test,
        test_scores,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    # Preserve machine information for detailed investigation.
    output_columns = [
        "machine_id",
        "machine_type",
        "cycle",
        "anomaly_type",
    ]

    prediction_df = test_df[output_columns].copy()

    prediction_df["actual_anomaly"] = y_test
    prediction_df["predicted_anomaly"] = predictions
    prediction_df["anomaly_score"] = test_scores
    prediction_df["decision_threshold"] = threshold

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics = {
        "experiment": "Machine-held-out Isolation Forest evaluation",
        "model": "IsolationForest",
        "features": FEATURES,
        "threshold": float(threshold),
        "threshold_selection": (
            "Maximum F1 on separate machine-level calibration group"
        ),
        "test_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "test_observations": int(len(y_test)),
        "test_normal_observations": int((y_test == 0).sum()),
        "test_anomaly_observations": int((y_test == 1).sum()),
    }

    return metrics


# ---------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------

def save_artifacts(
    model,
    scaler,
    metrics,
    machine_split,
    calibration_metrics,
):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics["machine_split"] = machine_split
    metrics["calibration_metrics"] = calibration_metrics

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "-" * 65)
    print("SAVED ARTIFACTS")
    print("-" * 65)

    print(f"Model:       {MODEL_PATH}")
    print(f"Scaler:      {SCALER_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    df = load_dataset()

    (
        fit_df,
        calibration_df,
        test_df,
        machine_split,
    ) = split_by_machine(df)

    (
        scaler,
        X_fit,
        X_calibration,
        X_test,
        y_calibration,
        y_test,
    ) = prepare_features(
        fit_df,
        calibration_df,
        test_df,
    )

    model = train_model(X_fit)

    threshold, calibration_metrics = calibrate_threshold(
        model,
        X_calibration,
        y_calibration,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
        test_df,
    )

    save_artifacts(
        model,
        scaler,
        metrics,
        machine_split,
        calibration_metrics,
    )

    print("\n" + "=" * 65)
    print("MACHINE-HELD-OUT EXPERIMENT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
