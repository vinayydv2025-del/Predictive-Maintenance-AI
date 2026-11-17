
"""
Machine-Held-Out Failure-Mode Classification

12 machines: model fitting
3 machines: threshold-free model validation / calibration-free evaluation
5 machines: final unseen-machine testing

No machine appears in more than one split.

This evaluates generalization across simulated machines.
It does not establish real-world industrial performance.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

MODEL_PATH = ROOT / "models" / "failure_mode_classifier_machine_holdout.joblib"
METRICS_PATH = ROOT / "reports" / "failure_mode_machine_holdout_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "failure_mode_machine_holdout_predictions.csv"

NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
]

CATEGORICAL_FEATURES = ["machine_type"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "anomaly_type"

RANDOM_SEED = 42


def load_dataset():
    print("\n===== LOADING DATASET =====")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET, "is_anomaly", "machine_id"]

    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.copy()

    df[TARGET] = df[TARGET].fillna("NORMAL")

    if df[FEATURES + [TARGET, "machine_id"]].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    expected_normal = df["is_anomaly"].eq(0)
    actual_normal = df[TARGET].eq("NORMAL")

    if not expected_normal.equals(actual_normal):
        raise ValueError("Failure-mode labels disagree with anomaly labels.")

    expected_classes = {
        "NORMAL",
        "OVERHEATING",
        "OVERLOAD",
        "SPEED_ANOMALY",
    }

    if set(df[TARGET].unique()) != expected_classes:
        raise ValueError(
            f"Unexpected target classes: {sorted(df[TARGET].unique())}"
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Unique machines: {df['machine_id'].nunique()}")
    print("\nClass distribution:")
    print(df[TARGET].value_counts().to_string())

    return df


def split_by_machine(df):
    print("\n===== MACHINE-LEVEL SPLIT =====")

    machines = sorted(df["machine_id"].unique())

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(machines)

    fitting_machines = sorted(shuffled[:12].tolist())
    calibration_machines = sorted(shuffled[12:15].tolist())
    test_machines = sorted(shuffled[15:].tolist())

    fitting_df = df[df["machine_id"].isin(fitting_machines)].copy()
    calibration_df = df[df["machine_id"].isin(calibration_machines)].copy()
    test_df = df[df["machine_id"].isin(test_machines)].copy()

    print(f"\nModel-fitting machines ({len(fitting_machines)}):")
    print(fitting_machines)

    print(f"\nCalibration/validation machines ({len(calibration_machines)}):")
    print(calibration_machines)

    print(f"\nUnseen test machines ({len(test_machines)}):")
    print(test_machines)

    print("\nObservation counts:")
    print(f"Model fitting: {len(fitting_df)}")
    print(f"Calibration:   {len(calibration_df)}")
    print(f"Final testing: {len(test_df)}")

    fitting_set = set(fitting_machines)
    calibration_set = set(calibration_machines)
    test_set = set(test_machines)

    assert fitting_set.isdisjoint(calibration_set)
    assert fitting_set.isdisjoint(test_set)
    assert calibration_set.isdisjoint(test_set)

    assert len(fitting_set | calibration_set | test_set) == len(machines)

    print("\nMachine overlap check: PASSED")

    return fitting_df, calibration_df, test_df


def build_pipeline():
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    print("=" * 70)
    print("MACHINE-HELD-OUT FAILURE-MODE CLASSIFICATION")
    print("=" * 70)

    df = load_dataset()

    fitting_df, calibration_df, test_df = split_by_machine(df)

    X_train = fitting_df[FEATURES]
    y_train = fitting_df[TARGET]

    X_calibration = calibration_df[FEATURES]
    y_calibration = calibration_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    labels = sorted(df[TARGET].unique())

    # Every class must exist in every split.
    for split_name, target in [
        ("training", y_train),
        ("calibration", y_calibration),
        ("testing", y_test),
    ]:
        missing_classes = set(labels) - set(target.unique())

        if missing_classes:
            raise ValueError(
                f"{split_name} split is missing classes: {missing_classes}"
            )

    print("\n===== TRAINING RANDOM FOREST =====")

    model = build_pipeline()
    model.fit(X_train, y_train)

    print("Model training completed.")

    print("\n===== CALIBRATION-MACHINE EVALUATION =====")

    calibration_predictions = model.predict(X_calibration)

    calibration_accuracy = accuracy_score(
        y_calibration,
        calibration_predictions,
    )

    print(f"Calibration accuracy: {calibration_accuracy:.4f}")

    print("\n===== FINAL EVALUATION ON UNSEEN MACHINES =====")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
    )

    # Save model and reports.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model_name": "RandomForestClassifier",
        "experiment": "Machine-held-out failure-mode classification",
        "features": FEATURES,
        "classes": labels,
        "training_machines": sorted(fitting_df["machine_id"].unique().tolist()),
        "calibration_machines": sorted(
            calibration_df["machine_id"].unique().tolist()
        ),
        "unseen_test_machines": sorted(
            test_df["machine_id"].unique().tolist()
        ),
        "training_rows": int(len(fitting_df)),
        "calibration_rows": int(len(calibration_df)),
        "testing_rows": int(len(test_df)),
        "calibration_accuracy": float(calibration_accuracy),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "data_source": "Synthetic industrial simulator",
        "limitations": [
            "Failure-mode labels come from controlled synthetic fault injection.",
            "The calibration machines are not used to fit the model.",
            "No hyperparameter tuning is performed on the final test machines.",
            "Results do not establish real-world industrial diagnostic accuracy.",
        ],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    output = test_df[
        ["timestamp", "machine_id", "cycle"] + FEATURES
    ].copy()

    output["actual_failure_mode"] = y_test
    output["predicted_failure_mode"] = predictions
    output["correct_prediction"] = (
        output["actual_failure_mode"]
        == output["predicted_failure_mode"]
    )

    output.to_csv(PREDICTIONS_PATH, index=False)

    print("\n===== SAVED ARTIFACTS =====")
    print(f"Model:       {MODEL_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")

    print("\n" + "=" * 70)
    print("MACHINE-HELD-OUT EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
