
"""
Advanced Industrial AI Robustness and Sensor Sensitivity Analysis.

Uses the existing machine-held-out test machines and trained classifier.
Does not retrain or overwrite existing models.

Experiments:
1. Noise levels: 1x, 2x, 3x, 4x
2. Individual sensor sensitivity
3. Severe multi-fault signal overlap
4. Per-class precision, recall and F1
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"

MODEL_PATH = (
    ROOT / "models"
    / "failure_mode_classifier_machine_holdout.joblib"
)

SPLIT_PATH = (
    ROOT / "reports"
    / "failure_mode_machine_holdout_metrics.json"
)

REPORT_PATH = ROOT / "reports" / "advanced_robustness_metrics.json"

SUMMARY_PATH = ROOT / "reports" / "advanced_robustness_summary.csv"

DETAIL_PATH = ROOT / "reports" / "advanced_robustness_predictions.csv"

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

SEED = 42

SENSOR_NOISE_SCALES = {
    "air_temperature": 0.7,
    "process_temperature": 0.7,
    "rotational_speed": 25.0,
    "torque": 2.0,
    "tool_wear": 1.0,
    "operating_hours": 0.1,
}


def load_assets():
    print("\n===== LOADING MODEL AND HELD-OUT DATA =====")

    for path in [DATA_PATH, MODEL_PATH, SPLIT_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_info = json.load(file)

    machines = split_info["unseen_test_machines"]

    test_df = df[df["machine_id"].isin(machines)].copy()

    test_df[TARGET] = test_df[TARGET].fillna("NORMAL")

    if len(test_df) != split_info["testing_rows"]:
        raise ValueError("Held-out test data does not match saved split.")

    model = joblib.load(MODEL_PATH)

    print(f"Test observations: {len(test_df)}")
    print(f"Unseen machines: {len(machines)}")

    return model, test_df, machines


def evaluate(model, original_df, modified_df, scenario, rng_seed):
    y_true = original_df[TARGET].to_numpy()

    predictions = model.predict(modified_df[FEATURES])

    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, predictions)

    result = {
        "scenario": scenario,
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "random_seed": rng_seed,
    }

    for class_name in sorted(original_df[TARGET].unique()):
        result[f"{class_name}_precision"] = float(
            report[class_name]["precision"]
        )
        result[f"{class_name}_recall"] = float(
            report[class_name]["recall"]
        )
        result[f"{class_name}_f1"] = float(
            report[class_name]["f1-score"]
        )

    details = original_df[
        ["timestamp", "machine_id", "cycle", TARGET]
    ].copy()

    details["scenario"] = scenario
    details["predicted_failure_mode"] = predictions
    details["correct"] = details[TARGET] == predictions

    return result, details


def add_noise(df, multiplier, seed):
    rng = np.random.default_rng(seed)
    modified = df.copy()

    for feature, base_scale in SENSOR_NOISE_SCALES.items():
        modified[feature] = (
            modified[feature].to_numpy()
            + rng.normal(
                0,
                base_scale * multiplier,
                size=len(modified),
            )
        )

    return modified


def sensor_ablation(df, sensor):
    """
    Replace one sensor with its training-independent
    median value calculated from the held-out data.

    This measures model reliance on that feature.
    It is not a causal experiment.
    """

    modified = df.copy()
    modified[sensor] = modified[sensor].median()

    return modified


def inject_severe_overlap(df):
    """
    Apply strong secondary fault-like signals to anomalous rows.
    Original labels are retained deliberately.

    This creates difficult and potentially physically inconsistent
    samples. It is a synthetic stress test, not a physical model.
    """

    modified = df.copy()

    for index in modified.index:
        fault = modified.at[index, TARGET]

        if fault == "NORMAL":
            continue

        if fault == "OVERHEATING":
            modified.at[index, "torque"] += 12.0
            modified.at[index, "rotational_speed"] -= 100.0

        elif fault == "OVERLOAD":
            modified.at[index, "air_temperature"] += 8.0
            modified.at[index, "process_temperature"] += 8.0
            modified.at[index, "rotational_speed"] -= 100.0

        elif fault == "SPEED_ANOMALY":
            modified.at[index, "torque"] += 10.0
            modified.at[index, "air_temperature"] += 6.0

    return modified


def main():
    print("=" * 70)
    print("ADVANCED ROBUSTNESS AND SENSOR SENSITIVITY")
    print("=" * 70)

    model, test_df, machines = load_assets()

    results = []
    detail_frames = []

    def run_scenario(name, modified):
        result, details = evaluate(
            model,
            test_df,
            modified,
            name,
            SEED,
        )

        results.append(result)
        detail_frames.append(details)

        print(
            f"{name:<35} "
            f"Accuracy={result['accuracy']:.4f} "
            f"Macro F1={result['macro_f1']:.4f}"
        )

    print("\n===== NOISE INTENSITY EXPERIMENTS =====")

    run_scenario("BASELINE", test_df.copy())

    for multiplier in [1, 2, 3, 4]:
        modified = add_noise(
            test_df,
            multiplier,
            SEED + multiplier,
        )

        run_scenario(
            f"SENSOR_NOISE_{multiplier}X",
            modified,
        )

    print("\n===== INDIVIDUAL SENSOR SENSITIVITY =====")

    numeric_sensors = list(SENSOR_NOISE_SCALES.keys())

    for sensor in numeric_sensors:
        modified = sensor_ablation(test_df, sensor)

        run_scenario(
            f"SENSOR_ABLATION_{sensor.upper()}",
            modified,
        )

    print("\n===== SEVERE OVERLAPPING FAULTS =====")

    overlap_df = inject_severe_overlap(test_df)

    run_scenario(
        "SEVERE_OVERLAPPING_FAULTS",
        overlap_df,
    )

    summary_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": "Advanced robustness and sensor sensitivity",
                "model": "Machine-held-out Random Forest classifier",
                "test_machines": machines,
                "test_observations": int(len(test_df)),
                "results": results,
                "limitations": [
                    "All tests use synthetic simulator observations.",
                    "Noise is Gaussian and may not represent real sensor noise.",
                    "Sensor ablation uses the held-out feature median.",
                    "Sensor ablation measures model reliance, not causation.",
                    "Severe overlap retains original single-fault labels.",
                    "Severe overlap may produce physically inconsistent readings.",
                    "Results do not establish real-world industrial performance.",
                ],
            },
            file,
            indent=4,
        )

    all_details = pd.concat(detail_frames, ignore_index=True)
    all_details.to_csv(DETAIL_PATH, index=False)

    print("\n===== FINAL RESULTS =====")

    display_columns = [
        "scenario",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n===== SAVED ARTIFACTS =====")
    print(f"JSON report: {REPORT_PATH}")
    print(f"Summary CSV: {SUMMARY_PATH}")
    print(f"Predictions: {DETAIL_PATH}")

    print("\n" + "=" * 70)
    print("ADVANCED ROBUSTNESS TESTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
