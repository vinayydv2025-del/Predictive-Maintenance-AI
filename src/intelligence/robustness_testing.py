
"""
Industrial AI - Adversarial Robustness Testing

Tests the failure-mode classifier under:
1. Sensor noise
2. Weak fault signals
3. Overlapping fault patterns
4. Sensor drift

All modifications are applied to copies of the held-out test data.
The original dataset and trained models are never overwritten.

This is a synthetic stress test, not real-world validation.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

MODEL_PATH = (
    ROOT / "models"
    / "failure_mode_classifier_machine_holdout.joblib"
)

SPLIT_METRICS_PATH = (
    ROOT / "reports"
    / "failure_mode_machine_holdout_metrics.json"
)

OUTPUT_PATH = ROOT / "reports" / "robustness_test_results.json"
DETAIL_PATH = ROOT / "reports" / "robustness_test_predictions.csv"

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
    "machine_type",
]

TARGET = "anomaly_type"

RANDOM_SEED = 42


def load_test_data():
    print("\n===== LOADING HELD-OUT TEST DATA =====")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not SPLIT_METRICS_PATH.exists():
        raise FileNotFoundError(SPLIT_METRICS_PATH)

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_METRICS_PATH, "r", encoding="utf-8") as file:
        split_metrics = json.load(file)

    unseen_machines = split_metrics["unseen_test_machines"]

    test_df = df[df["machine_id"].isin(unseen_machines)].copy()

    test_df[TARGET] = test_df[TARGET].fillna("NORMAL")

    if len(test_df) != split_metrics["testing_rows"]:
        raise ValueError("Held-out test row count does not match saved report.")

    if test_df["machine_id"].nunique() != len(unseen_machines):
        raise ValueError("Unexpected machine count in held-out test data.")

    print(f"Unseen test machines: {len(unseen_machines)}")
    print(f"Test observations: {len(test_df)}")

    print("\nClass distribution:")
    print(test_df[TARGET].value_counts().to_string())

    return test_df, unseen_machines


def inject_sensor_noise(df, severity, rng):
    """
    Add Gaussian noise to sensor readings.

    Severity controls noise standard deviation in approximate
    sensor measurement units.
    """

    noisy = df.copy()

    noise_scales = {
        "air_temperature": 0.7,
        "process_temperature": 0.7,
        "rotational_speed": 25.0,
        "torque": 2.0,
        "tool_wear": 1.0,
        "operating_hours": 0.1,
    }

    for feature, base_scale in noise_scales.items():
        noisy[feature] = (
            noisy[feature].to_numpy()
            + rng.normal(
                loc=0,
                scale=base_scale * severity,
                size=len(noisy),
            )
        )

    return noisy


def weaken_fault_signals(df, rng):
    """
    Reduce the injected fault signal for anomalous rows.

    The target labels remain unchanged so that the experiment
    measures whether the original fault can still be recognized.
    """

    weakened = df.copy()

    anomaly_mask = weakened[TARGET] != "NORMAL"

    for index in weakened.index[anomaly_mask]:
        fault = weakened.at[index, TARGET]

        if fault == "OVERHEATING":
            weakened.at[index, "air_temperature"] -= rng.uniform(1.0, 4.0)
            weakened.at[index, "process_temperature"] -= rng.uniform(1.0, 4.0)

        elif fault == "OVERLOAD":
            weakened.at[index, "torque"] -= rng.uniform(1.0, 5.0)

        elif fault == "SPEED_ANOMALY":
            weakened.at[index, "rotational_speed"] += rng.uniform(5.0, 30.0)

    return weakened


def inject_overlapping_faults(df, rng):
    """
    Add a secondary fault-like signal to anomalous observations.

    The original label is retained. This deliberately creates
    ambiguous cases and tests model behaviour under overlap.
    """

    overlapping = df.copy()

    anomaly_indices = overlapping.index[
        overlapping[TARGET] != "NORMAL"
    ]

    for index in anomaly_indices:
        original_fault = overlapping.at[index, TARGET]

        if original_fault == "OVERHEATING":
            overlapping.at[index, "torque"] += rng.uniform(2.0, 6.0)

        elif original_fault == "OVERLOAD":
            overlapping.at[index, "air_temperature"] += rng.uniform(1.0, 3.0)

        elif original_fault == "SPEED_ANOMALY":
            overlapping.at[index, "torque"] += rng.uniform(1.0, 4.0)

    return overlapping


def inject_sensor_drift(df):
    """
    Simulate gradual sensor calibration drift across the test rows.

    The drift is applied in row order as a stress-test scenario.
    It is not a measured physical drift model.
    """

    drifted = df.copy()

    drift_ratio = np.linspace(0.0, 1.0, len(drifted))

    drifted["air_temperature"] += 2.0 * drift_ratio
    drifted["process_temperature"] += 2.5 * drift_ratio
    drifted["rotational_speed"] -= 35.0 * drift_ratio
    drifted["torque"] += 3.0 * drift_ratio

    return drifted


def evaluate_scenario(model, scenario_name, test_df, modified_df):
    X = modified_df[FEATURES]
    y_true = test_df[TARGET]

    predictions = model.predict(X)

    accuracy = accuracy_score(y_true, predictions)

    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{scenario_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        "Macro F1:",
        f"{report['macro avg']['f1-score']:.4f}",
    )
    print(
        "Weighted F1:",
        f"{report['weighted avg']['f1-score']:.4f}",
    )

    result = {
        "scenario": scenario_name,
        "accuracy": float(accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
    }

    prediction_output = test_df[
        ["timestamp", "machine_id", "cycle", TARGET]
    ].copy()

    prediction_output["scenario"] = scenario_name
    prediction_output["predicted_failure_mode"] = predictions
    prediction_output["correct_prediction"] = (
        prediction_output[TARGET]
        == prediction_output["predicted_failure_mode"]
    )

    return result, prediction_output


def main():
    print("=" * 70)
    print("INDUSTRIAL AI - ADVERSARIAL ROBUSTNESS TESTING")
    print("=" * 70)

    test_df, unseen_machines = load_test_data()

    model = joblib.load(MODEL_PATH)

    print("\n===== BASELINE EVALUATION =====")

    rng = np.random.default_rng(RANDOM_SEED)

    scenarios = [
        (
            "BASELINE",
            test_df.copy(),
        ),
        (
            "LOW_SENSOR_NOISE",
            inject_sensor_noise(test_df, 0.5, rng),
        ),
        (
            "HIGH_SENSOR_NOISE",
            inject_sensor_noise(test_df, 2.0, rng),
        ),
        (
            "WEAK_FAULT_SIGNALS",
            weaken_fault_signals(test_df, rng),
        ),
        (
            "OVERLAPPING_FAULTS",
            inject_overlapping_faults(test_df, rng),
        ),
        (
            "SENSOR_DRIFT",
            inject_sensor_drift(test_df),
        ),
    ]

    results = []
    prediction_frames = []

    for scenario_name, modified_df in scenarios:
        result, prediction_output = evaluate_scenario(
            model,
            scenario_name,
            test_df,
            modified_df,
        )

        results.append(result)
        prediction_frames.append(prediction_output)

    summary = {
        "experiment": "Adversarial robustness testing",
        "model": "RandomForest failure-mode classifier",
        "unseen_test_machines": unseen_machines,
        "test_observations": int(len(test_df)),
        "random_seed": RANDOM_SEED,
        "data_source": "Synthetic industrial simulator",
        "scenarios": results,
        "limitations": [
            "Stress conditions are synthetically generated.",
            "Original labels are retained even when fault signals are weakened.",
            "Overlapping faults retain a single original label.",
            "Drift is applied in row order and is not a physical drift model.",
            "Results do not establish real-world industrial robustness.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(DETAIL_PATH, index=False)

    print("\n===== ROBUSTNESS SUMMARY =====")

    summary_table = pd.DataFrame(
        [
            {
                "Scenario": result["scenario"],
                "Accuracy": result["accuracy"],
                "Macro F1": result["macro_f1"],
                "Weighted F1": result["weighted_f1"],
            }
            for result in results
        ]
    )

    print(summary_table.to_string(index=False))

    print("\n===== SAVED ARTIFACTS =====")
    print(f"Metrics:     {OUTPUT_PATH}")
    print(f"Predictions: {DETAIL_PATH}")

    print("\n" + "=" * 70)
    print("ROBUSTNESS TESTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
