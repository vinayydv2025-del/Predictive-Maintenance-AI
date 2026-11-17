
"""
ENSEMBLE INTELLIGENCE - UNSEEN MACHINE EVALUATION

Evaluates the simulator-based anomaly detector and
noise-aware fault classifier on machines held out
from BOTH model experiments.

XGBoost is evaluated separately because it was trained
on the UCI AI4I dataset, not the simulator dataset.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ensemble_engine import diagnose_machine


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"
ANOMALY_METRICS_PATH = ROOT / "reports" / "machine_holdout_metrics.json"
FAULT_METRICS_PATH = ROOT / "reports" / "failure_mode_machine_holdout_metrics.json"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 75)
print("ENSEMBLE INTELLIGENCE - UNSEEN MACHINE EVALUATION")
print("=" * 75)


# ---------------------------------------------------------
# LOAD EXPERIMENT CONFIGURATIONS
# ---------------------------------------------------------

with open(ANOMALY_METRICS_PATH, "r", encoding="utf-8") as f:
    anomaly_metrics = json.load(f)

with open(FAULT_METRICS_PATH, "r", encoding="utf-8") as f:
    fault_metrics = json.load(f)


anomaly_test_machines = set(
    anomaly_metrics["machine_split"]["test_machines"]
)

fault_test_machines = set(
    fault_metrics["unseen_test_machines"]
)

# Use machines unseen to both experiments.
evaluation_machines = sorted(
    anomaly_test_machines.intersection(
        fault_test_machines
    )
)

if not evaluation_machines:
    raise RuntimeError(
        "No common unseen test machines were found."
    )


print("\n===== MACHINE SEPARATION =====")

print("Anomaly detector test machines:", sorted(anomaly_test_machines))
print("Fault classifier test machines:", sorted(fault_test_machines))
print("Common unseen machines:", evaluation_machines)


# ---------------------------------------------------------
# LOAD SIMULATOR DATA
# ---------------------------------------------------------

print("\n===== LOADING SIMULATOR DATA =====")

data = pd.read_csv(DATA_PATH)

required_columns = [
    "machine_id",
    "machine_type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
    "is_anomaly",
    "anomaly_type"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset missing columns: {missing_columns}"
    )


test_data = data[
    data["machine_id"].isin(evaluation_machines)
].copy()

test_data["anomaly_type"] = (
    test_data["anomaly_type"].fillna("NORMAL")
)

test_data["is_anomaly"] = (
    test_data["is_anomaly"].astype(bool)
)

if test_data.empty:
    raise RuntimeError("No evaluation observations found.")


actual_machines = set(test_data["machine_id"].unique())

if actual_machines != set(evaluation_machines):
    raise RuntimeError(
        "Evaluation machine set does not match expected set."
    )


print("Evaluation machines:", len(actual_machines))
print("Evaluation observations:", len(test_data))
print("Machine separation: PASSED")


# ---------------------------------------------------------
# RUN ENSEMBLE
# ---------------------------------------------------------

print("\n===== RUNNING ENSEMBLE DIAGNOSIS =====")

results = []

for index, row in enumerate(
    test_data.to_dict(orient="records"),
    start=1
):

    result = diagnose_machine(row)

    outputs = result["model_outputs"]

    failure_output = outputs["failure_prediction"]
    anomaly_output = outputs["anomaly_detection"]
    fault_output = outputs["fault_classification"]

    results.append({
        "machine_id": row["machine_id"],
        "cycle": row.get("cycle"),
        "true_anomaly": bool(row["is_anomaly"]),
        "predicted_anomaly": anomaly_output["is_anomaly"],
        "anomaly_score": anomaly_output["anomaly_score"],
        "true_fault": row["anomaly_type"],
        "predicted_fault": fault_output["predicted_fault"],
        "fault_confidence": fault_output["classification_confidence"],
        "xgboost_failure_probability": failure_output["failure_probability"],
        "final_diagnosis": result["final_diagnosis"]["diagnosis"],
        "severity": result["final_diagnosis"]["severity"]
    })

    if index % 250 == 0:
        print(f"Processed {index}/{len(test_data)} observations")


results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# ANOMALY DETECTION EVALUATION
# ---------------------------------------------------------

print("\n===== ANOMALY DETECTION PERFORMANCE =====")

y_anomaly = results_df["true_anomaly"].astype(int)
pred_anomaly = results_df["predicted_anomaly"].astype(int)

anomaly_accuracy = accuracy_score(
    y_anomaly,
    pred_anomaly
)

anomaly_precision = precision_score(
    y_anomaly,
    pred_anomaly,
    zero_division=0
)

anomaly_recall = recall_score(
    y_anomaly,
    pred_anomaly,
    zero_division=0
)

anomaly_f1 = f1_score(
    y_anomaly,
    pred_anomaly,
    zero_division=0
)

anomaly_matrix = confusion_matrix(
    y_anomaly,
    pred_anomaly,
    labels=[0, 1]
)

print(f"Accuracy:  {anomaly_accuracy:.4f}")
print(f"Precision: {anomaly_precision:.4f}")
print(f"Recall:    {anomaly_recall:.4f}")
print(f"F1:        {anomaly_f1:.4f}")

print("\nConfusion matrix:")
print(anomaly_matrix)


# ---------------------------------------------------------
# FAILURE-MODE CLASSIFICATION EVALUATION
# ---------------------------------------------------------

print("\n===== FAILURE-MODE CLASSIFICATION PERFORMANCE =====")

y_fault = results_df["true_fault"]
pred_fault = results_df["predicted_fault"]

fault_accuracy = accuracy_score(
    y_fault,
    pred_fault
)

fault_macro_f1 = f1_score(
    y_fault,
    pred_fault,
    average="macro",
    zero_division=0
)

fault_classes = fault_metrics["classes"]

fault_matrix = confusion_matrix(
    y_fault,
    pred_fault,
    labels=fault_classes
)

fault_report = classification_report(
    y_fault,
    pred_fault,
    labels=fault_classes,
    output_dict=True,
    zero_division=0
)

print(f"Accuracy: {fault_accuracy:.4f}")
print(f"Macro F1: {fault_macro_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_fault,
        pred_fault,
        labels=fault_classes,
        zero_division=0
    )
)

print("Confusion matrix:")
print(fault_matrix)


# ---------------------------------------------------------
# DIAGNOSTIC AGREEMENT
# ---------------------------------------------------------

print("\n===== ENSEMBLE DIAGNOSTIC AGREEMENT =====")

anomaly_alerts = int(
    results_df["predicted_anomaly"].sum()
)

fault_non_normal = int(
    (results_df["predicted_fault"] != "NORMAL").sum()
)

both_agree_on_fault = int(
    (
        results_df["predicted_anomaly"]
        &
        (results_df["predicted_fault"] != "NORMAL")
    ).sum()
)

print("Anomaly alerts:", anomaly_alerts)
print("Non-normal fault classifications:", fault_non_normal)
print("Both models indicate a fault:", both_agree_on_fault)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_path = (
    REPORT_DIR / "ensemble_unseen_machine_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


summary = {
    "experiment": "Ensemble evaluation on machines unseen to both simulator models",
    "evaluation_machines": evaluation_machines,
    "observations": int(len(results_df)),
    "anomaly_detection": {
        "accuracy": float(anomaly_accuracy),
        "precision": float(anomaly_precision),
        "recall": float(anomaly_recall),
        "f1": float(anomaly_f1),
        "confusion_matrix": anomaly_matrix.tolist()
    },
    "fault_classification": {
        "accuracy": float(fault_accuracy),
        "macro_f1": float(fault_macro_f1),
        "confusion_matrix": fault_matrix.tolist(),
        "classification_report": fault_report
    },
    "diagnostic_agreement": {
        "anomaly_alerts": anomaly_alerts,
        "non_normal_fault_classifications": fault_non_normal,
        "both_models_indicate_fault": both_agree_on_fault
    },
    "limitations": [
        "Evaluation uses synthetic simulator data.",
        "Fault labels originate from controlled fault injection.",
        "The XGBoost model uses a separate AI4I training domain.",
        "This evaluation does not establish real-world industrial performance."
    ]
}

summary_path = (
    REPORT_DIR / "ensemble_unseen_machine_metrics.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\n===== SAVED ARTIFACTS =====")
print("Predictions:", results_path)
print("Metrics:", summary_path)

print("\n" + "=" * 75)
print("ENSEMBLE EVALUATION COMPLETED")
print("=" * 75)
