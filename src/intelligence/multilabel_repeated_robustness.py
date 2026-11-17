
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/simulated/multi_label_fault_dataset.csv"
MODEL_PATH = ROOT / "models/multilabel_fault_classifier.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
SEEDS = [42, 123, 2026, 777, 999]

NOISE_SCALES = {
    "air_temperature": 0.7,
    "process_temperature": 0.7,
    "rotational_speed": 25.0,
    "torque": 2.0,
    "tool_wear": 1.0,
    "operating_hours": 0.1,
}


print("=" * 75)
print("REPEATED MULTI-LABEL ROBUSTNESS EXPERIMENT")
print("=" * 75)

data = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

targets = artifact["targets"]
features = artifact["features"]

test_data = data[data["split"] == "TEST"].copy()

X = test_data[features].copy()
y = test_data[targets].astype(int).to_numpy()

preprocessor = artifact["preprocessor"]
models = artifact["models"]
thresholds = artifact["thresholds"]

assert set(test_data["machine_id"]) == set(artifact["test_machines"])

print("Test rows:", len(test_data))
print("Test machines:", artifact["test_machines"])
print("Random seeds:", SEEDS)


def add_noise(frame, multiplier, seed):

    result = frame.copy()
    rng = np.random.default_rng(seed)

    for feature, scale in NOISE_SCALES.items():

        noise = rng.normal(
            0,
            scale * multiplier,
            size=len(result),
        )

        result[feature] = result[feature].to_numpy() + noise

    for feature in [
        "rotational_speed",
        "tool_wear",
        "operating_hours",
    ]:
        result[feature] = result[feature].clip(lower=0)

    return result


def predict(frame):

    transformed = preprocessor.transform(frame)

    predictions = []

    for target in targets:

        model = models[target]
        positive_index = list(model.classes_).index(1)

        scores = model.predict_proba(transformed)[:, positive_index]

        predictions.append(
            (scores >= thresholds[target]).astype(int)
        )

    return np.column_stack(predictions)


records = []
normal_records = []

for noise_level in NOISE_LEVELS:

    print(f"\nEvaluating noise level: {noise_level}x")

    for seed in SEEDS:

        X_noisy = add_noise(X, noise_level, seed)

        predicted = predict(X_noisy)

        record = {
            "noise_level": noise_level,
            "seed": seed,
            "micro_f1": f1_score(
                y, predicted, average="micro", zero_division=0
            ),
            "macro_f1": f1_score(
                y, predicted, average="macro", zero_division=0
            ),
            "exact_match_accuracy": accuracy_score(y, predicted),
            "hamming_loss": hamming_loss(y, predicted),
            "micro_precision": precision_score(
                y, predicted, average="micro", zero_division=0
            ),
            "micro_recall": recall_score(
                y, predicted, average="micro", zero_division=0
            ),
        }

        records.append(record)

        # Normal-only false-positive rate:
        # A normal reading is falsely flagged if any fault is predicted.
        normal_mask = test_data["scenario_type"].to_numpy() == "NORMAL"

        normal_actual = y[normal_mask]
        normal_predicted = predicted[normal_mask]

        false_positive_readings = int(
            np.any(normal_predicted == 1, axis=1).sum()
        )

        total_normal_readings = int(normal_mask.sum())

        normal_records.append({
            "noise_level": noise_level,
            "seed": seed,
            "normal_observations": total_normal_readings,
            "false_positive_readings": false_positive_readings,
            "normal_false_positive_rate": (
                false_positive_readings / total_normal_readings
            ),
            "normal_exact_match_accuracy": accuracy_score(
                normal_actual,
                normal_predicted,
            ),
        })

        print(
            f"  Seed {seed}: "
            f"Micro F1={record['micro_f1']:.4f}, "
            f"Exact match={record['exact_match_accuracy']:.4f}"
        )


# Aggregate repeated runs.
runs = pd.DataFrame(records)
normal_runs = pd.DataFrame(normal_records)

metrics = [
    "micro_f1",
    "macro_f1",
    "exact_match_accuracy",
    "hamming_loss",
    "micro_precision",
    "micro_recall",
]

summary_rows = []

for level, group in runs.groupby("noise_level"):

    row = {"noise_level": float(level)}

    for metric in metrics:

        row[f"{metric}_mean"] = float(group[metric].mean())

        row[f"{metric}_std"] = float(group[metric].std(ddof=1))

    normal_group = normal_runs[
        normal_runs["noise_level"] == level
    ]

    row["normal_false_positive_rate_mean"] = float(
        normal_group["normal_false_positive_rate"].mean()
    )

    row["normal_false_positive_rate_std"] = float(
        normal_group["normal_false_positive_rate"].std(ddof=1)
    )

    row["normal_exact_match_mean"] = float(
        normal_group["normal_exact_match_accuracy"].mean()
    )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

summary_path = REPORT_DIR / "multilabel_repeated_robustness_summary.csv"
runs_path = REPORT_DIR / "multilabel_repeated_robustness_runs.csv"
normal_path = REPORT_DIR / "multilabel_normal_false_positives.csv"
json_path = REPORT_DIR / "multilabel_repeated_robustness.json"

summary.to_csv(summary_path, index=False)
runs.to_csv(runs_path, index=False)
normal_runs.to_csv(normal_path, index=False)

report = {
    "experiment": "Five-seed repeated multi-label noise robustness",
    "test_machines": artifact["test_machines"],
    "training_machines": artifact["training_machines"],
    "calibration_machines": artifact["calibration_machines"],
    "noise_levels": NOISE_LEVELS,
    "seeds": SEEDS,
    "thresholds": thresholds,
    "summary": summary.to_dict(orient="records"),
    "normal_false_positive_runs": normal_runs.to_dict(orient="records"),
    "limitations": [
        "All observations are generated by the same synthetic simulator.",
        "Noise is artificially generated.",
        "Fault labels are retained after noise injection.",
        "Repeated seeds measure noise-realization variability, not independent dataset variability.",
        "Results do not establish real-world industrial performance.",
    ],
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n" + "=" * 75)
print("REPEATED EXPERIMENT SUMMARY")
print("=" * 75)

for _, row in summary.iterrows():

    print(
        f"\nNoise {row['noise_level']:.1f}x"
        f"\n  Micro F1: {row['micro_f1_mean']:.4f}"
        f" +/- {row['micro_f1_std']:.4f}"
        f"\n  Macro F1: {row['macro_f1_mean']:.4f}"
        f" +/- {row['macro_f1_std']:.4f}"
        f"\n  Exact match: {row['exact_match_accuracy_mean']:.4f}"
        f" +/- {row['exact_match_accuracy_std']:.4f}"
        f"\n  Normal false-positive rate: "
        f"{row['normal_false_positive_rate_mean']:.4f}"
        f" +/- {row['normal_false_positive_rate_std']:.4f}"
    )

print("\nReports saved:")
print(summary_path)
print(runs_path)
print(normal_path)
print(json_path)

print("\nREPEATED ROBUSTNESS EXPERIMENT COMPLETED")
