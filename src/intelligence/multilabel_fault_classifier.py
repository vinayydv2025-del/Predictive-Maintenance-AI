
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix,
    classification_report
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "simulated"
    / "multi_label_fault_dataset.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = (
    MODEL_DIR / "multilabel_fault_classifier.joblib"
)

METRICS_PATH = (
    REPORT_DIR / "multilabel_fault_metrics.json"
)

COMPARISON_PATH = (
    REPORT_DIR / "multilabel_fault_predictions.csv"
)

NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours"
]

CATEGORICAL_FEATURES = [
    "machine_type"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    "fault_overheating",
    "fault_overload",
    "fault_speed_anomaly"
]

RANDOM_STATE = 42


# ---------------------------------------------------------
# THRESHOLD CALIBRATION
# ---------------------------------------------------------

def select_threshold(y_true, probabilities):

    candidates = np.linspace(
        0.05,
        0.95,
        181
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "precision_micro": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            )
        ),
        "recall_micro": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            )
        ),
        "f1_micro": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "exact_match_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "hamming_loss": float(
            hamming_loss(y_true, y_pred)
        )
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 75)
    print("MULTI-LABEL INDUSTRIAL FAULT CLASSIFIER")
    print("=" * 75)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\n===== LOADING DATASET =====")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        FEATURES + TARGETS + ["split", "machine_id"]
    )

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    print("Dataset shape:", df.shape)

    # -----------------------------------------------------
    # MACHINE-HELD-OUT SPLIT
    # -----------------------------------------------------

    print("\n===== MACHINE-HELD-OUT SPLIT =====")

    train_df = df[df["split"] == "TRAIN"].copy()
    calibration_df = df[
        df["split"] == "CALIBRATION"
    ].copy()
    test_df = df[df["split"] == "TEST"].copy()

    if train_df.empty or calibration_df.empty or test_df.empty:
        raise ValueError("One or more data splits are empty.")

    train_machines = set(train_df["machine_id"])
    calibration_machines = set(
        calibration_df["machine_id"]
    )
    test_machines = set(test_df["machine_id"])

    if (
        train_machines & calibration_machines
        or train_machines & test_machines
        or calibration_machines & test_machines
    ):
        raise ValueError(
            "Machine leakage detected."
        )

    print("Training rows:", len(train_df))
    print("Calibration rows:", len(calibration_df))
    print("Test rows:", len(test_df))

    print("Training machines:", len(train_machines))
    print("Calibration machines:", len(calibration_machines))
    print("Test machines:", len(test_machines))

    # -----------------------------------------------------
    # PREPARE FEATURES AND TARGETS
    # -----------------------------------------------------

    X_train = train_df[FEATURES]
    X_calibration = calibration_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df[TARGETS].astype(int)
    y_calibration = calibration_df[TARGETS].astype(int)
    y_test = test_df[TARGETS].astype(int)

    # Every target must contain both classes in each split.
    for target in TARGETS:

        if y_train[target].nunique() != 2:
            raise ValueError(
                f"Training target {target} lacks both classes."
            )

        if y_calibration[target].nunique() != 2:
            raise ValueError(
                f"Calibration target {target} lacks both classes."
            )

        if y_test[target].nunique() != 2:
            raise ValueError(
                f"Test target {target} lacks both classes."
            )

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    print("\n===== PREPROCESSING =====")

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_calibration_transformed = preprocessor.transform(
        X_calibration
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    print("Preprocessing fitted on training machines only.")
    print(
        "Transformed feature count:",
        X_train_transformed.shape[1]
    )

    # -----------------------------------------------------
    # TRAIN THREE INDEPENDENT CLASSIFIERS
    # -----------------------------------------------------

    print("\n===== TRAINING MULTI-LABEL MODELS =====")

    models = {}

    for target in TARGETS:

        print(f"\nTraining model: {target}")

        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_train_transformed,
            y_train[target]
        )

        models[target] = model

        print("Model trained successfully.")

    # -----------------------------------------------------
    # CALIBRATE EACH FAULT THRESHOLD
    # -----------------------------------------------------

    print("\n===== CALIBRATING FAULT THRESHOLDS =====")

    thresholds = {}
    calibration_metrics = {}

    for target in TARGETS:

        model = models[target]

        positive_index = list(
            model.classes_
        ).index(1)

        probabilities = model.predict_proba(
            X_calibration_transformed
        )[:, positive_index]

        threshold, calibration_f1 = select_threshold(
            y_calibration[target].to_numpy(),
            probabilities
        )

        thresholds[target] = threshold

        calibration_metrics[target] = {
            "threshold": threshold,
            "calibration_f1": calibration_f1
        }

        print(
            f"{target}: threshold={threshold:.4f}, "
            f"calibration F1={calibration_f1:.4f}"
        )

    # -----------------------------------------------------
    # TEST ON UNSEEN MACHINES
    # -----------------------------------------------------

    print("\n===== EVALUATING UNSEEN TEST MACHINES =====")

    probability_results = {}
    prediction_results = {}

    for target in TARGETS:

        model = models[target]

        positive_index = list(
            model.classes_
        ).index(1)

        probabilities = model.predict_proba(
            X_test_transformed
        )[:, positive_index]

        predictions = (
            probabilities >= thresholds[target]
        ).astype(int)

        probability_results[target] = probabilities
        prediction_results[target] = predictions

    y_true = y_test[TARGETS].to_numpy()

    y_pred = np.column_stack([
        prediction_results[target]
        for target in TARGETS
    ])

    overall_metrics = calculate_metrics(
        y_true,
        y_pred
    )

    per_fault_metrics = {}

    print("\nOverall metrics:")

    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPer-fault metrics:")

    for index, target in enumerate(TARGETS):

        truth = y_true[:, index]
        predicted = y_pred[:, index]

        metrics = {
            "precision": float(
                precision_score(
                    truth,
                    predicted,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    truth,
                    predicted,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    truth,
                    predicted,
                    zero_division=0
                )
            ),
            "threshold": thresholds[target],
            "positive_test_examples": int(truth.sum()),
            "predicted_positive_examples": int(predicted.sum())
        }

        per_fault_metrics[target] = metrics

        print(
            f"{target}: "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    # -----------------------------------------------------
    # MULTI-LABEL CONFUSION MATRICES
    # -----------------------------------------------------

    matrices = multilabel_confusion_matrix(
        y_true,
        y_pred
    )

    confusion_results = {}

    for index, target in enumerate(TARGETS):

        matrix = matrices[index]

        confusion_results[target] = {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1])
        }

    # -----------------------------------------------------
    # SCENARIO-LEVEL EVALUATION
    # -----------------------------------------------------

    scenario_results = {}

    test_scenarios = test_df["scenario_type"].to_numpy()

    for scenario in sorted(set(test_scenarios)):

        mask = test_scenarios == scenario

        scenario_results[scenario] = {
            "observations": int(mask.sum()),
            "exact_match_accuracy": float(
                accuracy_score(
                    y_true[mask],
                    y_pred[mask]
                )
            ),
            "f1_macro": float(
                f1_score(
                    y_true[mask],
                    y_pred[mask],
                    average="macro",
                    zero_division=0
                )
            )
        }

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    artifact = {
        "model_name": "Multi-label Industrial Fault Random Forest",
        "preprocessor": preprocessor,
        "models": models,
        "thresholds": thresholds,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
        "training_machines": sorted(train_machines),
        "calibration_machines": sorted(calibration_machines),
        "test_machines": sorted(test_machines),
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df))
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    prediction_df = test_df[
        [
            "machine_id",
            "machine_type",
            "cycle",
            "scenario_type",
            "fault_combination"
        ]
    ].copy()

    for target in TARGETS:

        prediction_df[f"actual_{target}"] = (
            y_test[target].to_numpy()
        )

        prediction_df[f"probability_{target}"] = (
            probability_results[target]
        )

        prediction_df[f"predicted_{target}"] = (
            prediction_results[target]
        )

    prediction_df["exact_match"] = (
        np.all(
            y_true == y_pred,
            axis=1
        )
    )

    prediction_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    report = {
        "model": "Multi-label Random Forest",
        "dataset": "Synthetic multi-label industrial fault dataset",
        "evaluation_design": "Machine-held-out",
        "training_machines": sorted(train_machines),
        "calibration_machines": sorted(calibration_machines),
        "test_machines": sorted(test_machines),
        "training_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "targets": TARGETS,
        "calibration": calibration_metrics,
        "overall_test_metrics": overall_metrics,
        "per_fault_test_metrics": per_fault_metrics,
        "confusion_matrices": confusion_results,
        "scenario_test_metrics": scenario_results,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=TARGETS,
            output_dict=True,
            zero_division=0
        ),
        "limitations": [
            "Fault labels are generated by controlled synthetic injection.",
            "All eight variants of each source observation share its machine and cycle.",
            "Machine-held-out testing prevents machine overlap but does not prove real-world generalization.",
            "Probability thresholds are calibrated on only three synthetic machines.",
            "Random Forest probabilities are not guaranteed to be calibrated probabilities."
        ]
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n===== SAVED ARTIFACTS =====")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Predictions:", COMPARISON_PATH)

    print("\nMULTI-LABEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
