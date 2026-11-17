
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/simulated/multi_label_fault_dataset.csv"

MODEL_PATH = ROOT / "models/multilabel_fault_classifier.joblib"

REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

SENSOR_NOISE_SCALES = {
    "air_temperature": 0.7,
    "process_temperature": 0.7,
    "rotational_speed": 25.0,
    "torque": 2.0,
    "tool_wear": 1.0,
    "operating_hours": 0.1,
}


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

print("=" * 75)
print("MULTI-LABEL FAULT CLASSIFIER ROBUSTNESS EVALUATION")
print("=" * 75)

data = pd.read_csv(DATA_PATH)

artifact = joblib.load(MODEL_PATH)

targets = artifact["targets"]

features = artifact["features"]

thresholds = artifact["thresholds"]

test_machines = artifact["test_machines"]

print("\n===== DATA VALIDATION =====")

test_data = data[data["split"] == "TEST"].copy()

assert len(test_data) == artifact["test_rows"]

assert set(test_data["machine_id"]) == set(test_machines)

assert not set(test_data["machine_id"]).intersection(
    artifact["training_machines"]
)

assert not set(test_data["machine_id"]).intersection(
    artifact["calibration_machines"]
)

print("Test observations:", len(test_data))
print("Unseen machines:", test_machines)
print("Feature count:", len(features))
print("Target count:", len(targets))

print("\nMachine separation: PASSED")


# =========================================================
# PREPARE TEST DATA
# =========================================================

X_test = test_data[features].copy()

y_test = test_data[targets].astype(int).to_numpy()

preprocessor = artifact["preprocessor"]

models = artifact["models"]

numeric_features = artifact["numeric_features"]

print("\n===== TEST LABEL DISTRIBUTION =====")

for index, target in enumerate(targets):

    print(
        f"{target}: "
        f"positive={int(y_test[:, index].sum())}, "
        f"negative={int((y_test[:, index] == 0).sum())}"
    )


# =========================================================
# NOISE GENERATION
# =========================================================

def add_sensor_noise(X, multiplier, seed):

    noisy = X.copy()

    rng = np.random.default_rng(seed)

    if multiplier == 0:
        return noisy

    for feature, base_scale in SENSOR_NOISE_SCALES.items():

        noise = rng.normal(
            loc=0.0,
            scale=base_scale * multiplier,
            size=len(noisy),
        )

        noisy[feature] = noisy[feature].to_numpy() + noise

    # Keep physically nonnegative input quantities nonnegative.
    for feature in [
        "rotational_speed",
        "tool_wear",
        "operating_hours",
    ]:
        noisy[feature] = noisy[feature].clip(lower=0)

    return noisy


# =========================================================
# MULTI-LABEL PREDICTION
# =========================================================

def predict_multilabel(X):

    transformed = preprocessor.transform(X)

    predictions = []

    scores = []

    for target in targets:

        model = models[target]

        positive_index = list(model.classes_).index(1)

        probabilities = model.predict_proba(transformed)[
            :, positive_index
        ]

        threshold = float(thresholds[target])

        predicted = (probabilities >= threshold).astype(int)

        predictions.append(predicted)

        scores.append(probabilities)

    return (
        np.column_stack(predictions),
        np.column_stack(scores),
    )


# =========================================================
# METRIC CALCULATION
# =========================================================

all_results = []

all_confusion_matrices = {}

all_scenario_results = {}

prediction_records = []


for level_index, noise_level in enumerate(NOISE_LEVELS):

    print("\n" + "=" * 75)

    print(f"EVALUATING SENSOR NOISE: {noise_level}x")

    print("=" * 75)

    X_noisy = add_sensor_noise(
        X_test,
        multiplier=noise_level,
        seed=RANDOM_SEED + level_index,
    )

    y_pred, y_scores = predict_multilabel(X_noisy)

    metrics = {
        "noise_level": noise_level,
        "observations": len(y_test),

        "micro_precision": float(
            precision_score(
                y_test,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),

        "micro_recall": float(
            recall_score(
                y_test,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),

        "micro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "exact_match_accuracy": float(
            accuracy_score(y_test, y_pred)
        ),

        "hamming_loss": float(
            hamming_loss(y_test, y_pred)
        ),
    }

    all_results.append(metrics)

    confusion = multilabel_confusion_matrix(
        y_test,
        y_pred,
    )

    all_confusion_matrices[str(noise_level)] = {}

    print("\nOVERALL PERFORMANCE")

    for key, value in metrics.items():

        if key not in ["noise_level", "observations"]:

            print(f"{key:<25}: {value:.4f}")

    print("\nPER-FAULT PERFORMANCE")

    for index, target in enumerate(targets):

        precision = precision_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        recall = recall_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        f1 = f1_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        cm = confusion[index]

        tn, fp, fn, tp = cm.ravel()

        all_confusion_matrices[str(noise_level)][target] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        print(
            f"{target:<25} "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f} "
            f"F1={f1:.4f}"
        )

        print(
            f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )

    # -----------------------------------------------------
    # Scenario-level exact-match performance
    # -----------------------------------------------------

    scenario_results = {}

    for scenario in sorted(test_data["scenario_type"].unique()):

        mask = (
            test_data["scenario_type"].to_numpy() == scenario
        )

        scenario_results[str(scenario)] = {
            "observations": int(mask.sum()),

            "exact_match_accuracy": float(
                accuracy_score(
                    y_test[mask],
                    y_pred[mask],
                )
            ),

            "macro_f1": float(
                f1_score(
                    y_test[mask],
                    y_pred[mask],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    all_scenario_results[str(noise_level)] = scenario_results

    print("\nSCENARIO-LEVEL EXACT-MATCH ACCURACY")

    for scenario, values in scenario_results.items():

        print(
            f"{scenario:<20} "
            f"Accuracy={values['exact_match_accuracy']:.4f} "
            f"Macro_F1={values['macro_f1']:.4f}"
        )

    # -----------------------------------------------------
    # Store prediction-level results
    # -----------------------------------------------------

    for row_index in range(len(test_data)):

        record = {
            "machine_id": test_data.iloc[row_index]["machine_id"],
            "cycle": int(test_data.iloc[row_index]["cycle"]),
            "scenario_type": test_data.iloc[row_index]["scenario_type"],
            "noise_level": noise_level,
        }

        for target_index, target in enumerate(targets):

            record[f"actual_{target}"] = int(
                y_test[row_index, target_index]
            )

            record[f"predicted_{target}"] = int(
                y_pred[row_index, target_index]
            )

            record[f"score_{target}"] = float(
                y_scores[row_index, target_index]
            )

        prediction_records.append(record)


# =========================================================
# SAVE REPORTS
# =========================================================

summary = {
    "experiment": "Multi-label fault classifier noise robustness",
    "model": artifact["model_name"],
    "dataset": str(DATA_PATH),
    "test_observations": len(test_data),
    "test_machines": test_machines,
    "training_machines": artifact["training_machines"],
    "calibration_machines": artifact["calibration_machines"],
    "targets": targets,
    "thresholds": thresholds,
    "noise_scales": SENSOR_NOISE_SCALES,
    "noise_levels": all_results,
    "confusion_matrices": all_confusion_matrices,
    "scenario_metrics": all_scenario_results,
    "limitations": [
        "The test machines are unseen, but the simulator design is shared.",
        "Noise is synthetically generated.",
        "Original fault labels are retained after noise injection.",
        "The same source observations appear in multiple scenario variants.",
        "The experiment does not establish real-world industrial performance.",
        "Random Forest scores are not guaranteed calibrated probabilities.",
    ],
}

json_path = REPORT_DIR / "multilabel_robustness_evaluation.json"

with open(json_path, "w", encoding="utf-8") as file:

    json.dump(summary, file, indent=4)


csv_path = REPORT_DIR / "multilabel_robustness_summary.csv"

pd.DataFrame(all_results).to_csv(
    csv_path,
    index=False,
)


prediction_path = REPORT_DIR / "multilabel_robustness_predictions.csv"

pd.DataFrame(prediction_records).to_csv(
    prediction_path,
    index=False,
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n" + "=" * 75)

print("FINAL ROBUSTNESS COMPARISON")

print("=" * 75)

print(
    f"{'Noise':<12}"
    f"{'Micro F1':<14}"
    f"{'Macro F1':<14}"
    f"{'Exact Match':<16}"
    f"{'Hamming Loss':<14}"
)

for result in all_results:

    print(
        f"{result['noise_level']:<12.1f}"
        f"{result['micro_f1']:<14.4f}"
        f"{result['macro_f1']:<14.4f}"
        f"{result['exact_match_accuracy']:<16.4f}"
        f"{result['hamming_loss']:<14.4f}"
    )


print("\n===== REPORTS SAVED =====")

print(json_path)

print(csv_path)

print(prediction_path)

print("\nROBUSTNESS EVALUATION COMPLETED")

print(
    "\nImportant: Results represent synthetic noise stress testing, "
    "not real-world validation."
)
