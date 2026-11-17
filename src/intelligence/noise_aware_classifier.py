
"""
Noise-Aware Industrial Failure Diagnosis

Compares:
1. Existing machine-held-out Random Forest
2. New Random Forest trained with noise augmentation

Data split:
- 12 machines: training
- 3 machines: calibration
- 5 machines: final unseen testing

Noise augmentation is performed ONLY on training data.
The final test machines are never used for model fitting or selection.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

SPLIT_PATH = (
    ROOT / "reports"
    / "failure_mode_machine_holdout_metrics.json"
)

BASELINE_MODEL_PATH = (
    ROOT / "models"
    / "failure_mode_classifier_machine_holdout.joblib"
)

NEW_MODEL_PATH = (
    ROOT / "models"
    / "noise_aware_failure_classifier.joblib"
)

REPORT_PATH = (
    ROOT / "reports"
    / "noise_aware_classifier_metrics.json"
)

SUMMARY_PATH = (
    ROOT / "reports"
    / "noise_aware_classifier_comparison.csv"
)

PREDICTIONS_PATH = (
    ROOT / "reports"
    / "noise_aware_classifier_predictions.csv"
)

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

NOISE_SCALES = {
    "air_temperature": 0.7,
    "process_temperature": 0.7,
    "rotational_speed": 25.0,
    "torque": 2.0,
    "tool_wear": 1.0,
    "operating_hours": 0.1,
}

AUGMENTATION_LEVELS = [0.5, 1.0, 2.0]

RANDOM_SEED = 42


def load_data_and_split():
    print("\n===== LOADING DATA AND SAVED MACHINE SPLIT =====")

    for path in [
        DATA_PATH,
        SPLIT_PATH,
        BASELINE_MODEL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_info = json.load(file)

    training_machines = split_info["training_machines"]
    calibration_machines = split_info["calibration_machines"]
    test_machines = split_info["unseen_test_machines"]

    df[TARGET] = df[TARGET].fillna("NORMAL")

    train_df = df[df["machine_id"].isin(training_machines)].copy()
    calibration_df = df[
        df["machine_id"].isin(calibration_machines)
    ].copy()
    test_df = df[df["machine_id"].isin(test_machines)].copy()

    expected_counts = {
        "training": split_info["training_rows"],
        "calibration": split_info["calibration_rows"],
        "testing": split_info["testing_rows"],
    }

    actual_counts = {
        "training": len(train_df),
        "calibration": len(calibration_df),
        "testing": len(test_df),
    }

    if expected_counts != actual_counts:
        raise ValueError(
            f"Saved split counts do not match. "
            f"Expected {expected_counts}, got {actual_counts}"
        )

    machine_sets = [
        set(training_machines),
        set(calibration_machines),
        set(test_machines),
    ]

    if (
        machine_sets[0] & machine_sets[1]
        or machine_sets[0] & machine_sets[2]
        or machine_sets[1] & machine_sets[2]
    ):
        raise ValueError("Machine overlap detected.")

    for name, split in [
        ("training", train_df),
        ("calibration", calibration_df),
        ("testing", test_df),
    ]:
        if set(split[TARGET].unique()) != set(df[TARGET].unique()):
            raise ValueError(f"{name} split is missing a failure class.")

    print(f"Training rows:    {len(train_df)}")
    print(f"Calibration rows: {len(calibration_df)}")
    print(f"Test rows:        {len(test_df)}")

    print("\nMachine separation: PASSED")

    return train_df, calibration_df, test_df


def add_training_noise(df, multiplier, seed):
    """Create a noisy copy of training observations."""

    rng = np.random.default_rng(seed)
    augmented = df.copy()

    for feature, base_scale in NOISE_SCALES.items():
        noise = rng.normal(
            loc=0.0,
            scale=base_scale * multiplier,
            size=len(augmented),
        )

        augmented[feature] = (
            augmented[feature].to_numpy() + noise
        )

    return augmented


def create_augmented_training_data(train_df):
    print("\n===== GENERATING NOISE-AUGMENTED TRAINING DATA =====")

    frames = [train_df.copy()]

    for index, level in enumerate(AUGMENTATION_LEVELS):
        noisy_copy = add_training_noise(
            train_df,
            multiplier=level,
            seed=RANDOM_SEED + index + 1,
        )

        frames.append(noisy_copy)

        print(
            f"Created {len(noisy_copy)} additional rows "
            f"at noise level {level}x"
        )

    augmented_df = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"\nOriginal training rows: {len(train_df)}")
    print(f"Augmented training rows: {len(augmented_df)}")

    print("\nAugmented class distribution:")
    print(augmented_df[TARGET].value_counts().to_string())

    return augmented_df


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


def calculate_metrics(model, df):
    predictions = model.predict(df[FEATURES])
    actual = df[TARGET].to_numpy()

    report = classification_report(
        actual,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
    }

    return result, predictions


def main():
    print("=" * 70)
    print("NOISE-AWARE INDUSTRIAL FAILURE DIAGNOSIS")
    print("=" * 70)

    train_df, calibration_df, test_df = load_data_and_split()

    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    augmented_df = create_augmented_training_data(train_df)

    print("\n===== TRAINING NOISE-AWARE RANDOM FOREST =====")

    noise_aware_model = build_pipeline()

    noise_aware_model.fit(
        augmented_df[FEATURES],
        augmented_df[TARGET],
    )

    print("Noise-aware model training completed.")

    print("\n===== CALIBRATION ON THREE SEPARATE MACHINES =====")

    calibration_results = {}

    for model_name, model in [
        ("baseline", baseline_model),
        ("noise_aware", noise_aware_model),
    ]:
        metrics, _ = calculate_metrics(model, calibration_df)

        calibration_results[model_name] = {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        }

        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    print("\n===== FINAL TEST ON UNSEEN MACHINES =====")

    results = []
    prediction_frames = []

    test_scenarios = [
        ("CLEAN", 0.0),
        ("NOISE_1X", 1.0),
        ("NOISE_2X", 2.0),
        ("NOISE_3X", 3.0),
        ("NOISE_4X", 4.0),
    ]

    for scenario_name, noise_level in test_scenarios:

        if noise_level == 0:
            modified_test = test_df.copy()
        else:
            modified_test = add_training_noise(
                test_df,
                multiplier=noise_level,
                seed=RANDOM_SEED + 100 + int(noise_level),
            )

        for model_name, model in [
            ("baseline", baseline_model),
            ("noise_aware", noise_aware_model),
        ]:
            metrics, predictions = calculate_metrics(
                model,
                modified_test,
            )

            result = {
                "model": model_name,
                "scenario": scenario_name,
                "noise_multiplier": noise_level,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "classification_report": metrics[
                    "classification_report"
                ],
            }

            results.append(result)

            print(
                f"{model_name:<12} "
                f"{scenario_name:<12} "
                f"Accuracy={metrics['accuracy']:.4f} "
                f"Macro F1={metrics['macro_f1']:.4f}"
            )

            details = test_df[
                ["timestamp", "machine_id", "cycle", TARGET]
            ].copy()

            details["model"] = model_name
            details["scenario"] = scenario_name
            details["predicted_failure_mode"] = predictions
            details["correct"] = (
                details[TARGET] == predictions
            )

            prediction_frames.append(details)

    NEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(noise_aware_model, NEW_MODEL_PATH)

    summary_rows = [
        {
            key: value
            for key, value in result.items()
            if key != "classification_report"
        }
        for result in results
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(PREDICTIONS_PATH, index=False)

    final_report = {
        "experiment": "Noise-aware failure diagnosis",
        "baseline_model": str(BASELINE_MODEL_PATH),
        "new_model": str(NEW_MODEL_PATH),
        "training_augmentation_levels": AUGMENTATION_LEVELS,
        "original_training_rows": int(len(train_df)),
        "augmented_training_rows": int(len(augmented_df)),
        "calibration_results": calibration_results,
        "test_machines": sorted(
            test_df["machine_id"].unique().tolist()
        ),
        "test_observations": int(len(test_df)),
        "test_results": results,
        "limitations": [
            "The data and fault labels are synthetic.",
            "Noise augmentation does not reproduce every real sensor fault.",
            "Calibration machines are not used for model fitting.",
            "Final test machines are not used for training or model selection.",
            "Results do not establish real-world industrial accuracy.",
        ],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4)

    print("\n===== FINAL COMPARISON =====")

    print(
        summary_df[
            [
                "model",
                "scenario",
                "accuracy",
                "macro_f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n===== SAVED ARTIFACTS =====")
    print(f"New model:   {NEW_MODEL_PATH}")
    print(f"JSON report: {REPORT_PATH}")
    print(f"Summary CSV: {SUMMARY_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")

    print("\n" + "=" * 70)
    print("NOISE-AWARE CLASSIFIER EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
