
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# ---------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------

xgb_model = joblib.load(MODEL_DIR / "best_model.joblib")

isolation_forest = joblib.load(
    MODEL_DIR / "isolation_forest_machine_holdout.joblib"
)

anomaly_scaler = joblib.load(
    MODEL_DIR / "anomaly_scaler_machine_holdout.joblib"
)

legacy_fault_model = joblib.load(
    MODEL_DIR / "noise_aware_failure_classifier.joblib"
)

multilabel_artifact = joblib.load(
    MODEL_DIR / "multilabel_fault_classifier.joblib"
)

with open(MODEL_DIR / "model_metadata.json", encoding="utf-8") as file:
    xgb_metadata = json.load(file)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

XGB_THRESHOLD = float(xgb_metadata["threshold"])

# Previously calibrated on the machine-held-out calibration set.
ISOLATION_FOREST_THRESHOLD = 0.527733

XGB_FEATURE_MAP = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Type": "machine_type",
}

ANOMALY_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
]

FAULT_FEATURES = ANOMALY_FEATURES + ["machine_type"]

MULTILABEL_TARGETS = multilabel_artifact["targets"]

MULTILABEL_THRESHOLDS = multilabel_artifact["thresholds"]

FAULT_RECOMMENDATIONS = {
    "OVERHEATING": [
        "Inspect cooling and ventilation systems.",
        "Check temperature sensors and thermal operating conditions.",
    ],
    "OVERLOAD": [
        "Inspect mechanical load and torque conditions.",
        "Check for excessive friction or mechanical resistance.",
    ],
    "SPEED_ANOMALY": [
        "Inspect the motor, drive system and rotational-speed sensor.",
        "Verify whether the operating speed is within its intended range.",
    ],
}


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_reading(reading):

    required = [
        "machine_id",
        "machine_type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "operating_hours",
    ]

    missing = [key for key in required if key not in reading]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if reading["machine_type"] not in ["L", "M", "H"]:
        raise ValueError("machine_type must be L, M, or H.")

    numeric_fields = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "operating_hours",
    ]

    for field in numeric_fields:

        value = reading[field]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric.")

        if not pd.notna(value):
            raise ValueError(f"{field} must be finite.")

    if reading["rotational_speed"] < 0:
        raise ValueError("rotational_speed cannot be negative.")

    if reading["tool_wear"] < 0:
        raise ValueError("tool_wear cannot be negative.")

    if reading["operating_hours"] < 0:
        raise ValueError("operating_hours cannot be negative.")

    return True


# ---------------------------------------------------------
# INPUT CONVERSION
# ---------------------------------------------------------

def make_xgb_input(reading):

    row = {}

    for model_feature, reading_feature in XGB_FEATURE_MAP.items():
        row[model_feature] = reading[reading_feature]

    return pd.DataFrame([row])


def make_sensor_input(reading):

    return pd.DataFrame([{
        feature: reading[feature]
        for feature in ANOMALY_FEATURES
    }])


def make_fault_input(reading):

    return pd.DataFrame([{
        feature: reading[feature]
        for feature in FAULT_FEATURES
    }])


# ---------------------------------------------------------
# MODEL 1: SUPERVISED FAILURE PREDICTION
# AI4I DATASET
# ---------------------------------------------------------

def predict_xgboost(reading):

    X = make_xgb_input(reading)

    probability = float(xgb_model.predict_proba(X)[0][1])

    prediction = int(probability >= XGB_THRESHOLD)

    return {
        "model": "XGBoost",
        "training_domain": "UCI AI4I 2020 dataset",
        "prediction": (
            "FAILURE_RISK" if prediction else "NO_FAILURE_ALERT"
        ),
        "failure_score": round(probability, 6),
        "decision_threshold": round(XGB_THRESHOLD, 6),
        "score_interpretation": (
            "Model score; not a guaranteed real-world failure probability."
        ),
    }


# ---------------------------------------------------------
# MODEL 2: UNSUPERVISED ANOMALY DETECTION
# SYNTHETIC SIMULATOR DATASET
# ---------------------------------------------------------

def detect_anomaly(reading):

    X = make_sensor_input(reading)

    X_scaled = anomaly_scaler.transform(X)

    raw_score = float(
        isolation_forest.score_samples(X_scaled)[0]
    )

    anomaly_score = -raw_score

    is_anomaly = (
        anomaly_score >= ISOLATION_FOREST_THRESHOLD
    )

    return {
        "model": "Isolation Forest",
        "training_domain": "Synthetic industrial simulator",
        "prediction": "ANOMALY" if is_anomaly else "NORMAL",
        "anomaly_score": round(anomaly_score, 6),
        "decision_threshold": round(
            ISOLATION_FOREST_THRESHOLD, 6
        ),
    }


# ---------------------------------------------------------
# MODEL 3: LEGACY SINGLE-LABEL FAULT CLASSIFIER
# RETAINED FOR COMPARISON
# ---------------------------------------------------------

def classify_fault(reading):

    X = make_fault_input(reading)

    predicted_fault = str(
        legacy_fault_model.predict(X)[0]
    )

    probabilities = legacy_fault_model.predict_proba(X)[0]

    classes = legacy_fault_model.classes_

    probability_map = {
        str(label): round(float(probability), 6)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "model": "Noise-aware Random Forest",
        "training_domain": "Synthetic industrial simulator",
        "predicted_fault": predicted_fault,
        "class_scores": probability_map,
        "limitation": (
            "Single-label classifier; cannot independently represent "
            "simultaneous faults."
        ),
    }


# ---------------------------------------------------------
# MODEL 4: MULTI-LABEL FAULT CLASSIFICATION
# ---------------------------------------------------------

def classify_multiple_faults(reading):

    X = make_fault_input(reading)

    transformed = multilabel_artifact["preprocessor"].transform(X)

    results = {}
    detected_faults = []

    for target in MULTILABEL_TARGETS:

        model = multilabel_artifact["models"][target]

        threshold = float(MULTILABEL_THRESHOLDS[target])

        positive_index = list(model.classes_).index(1)

        score = float(
            model.predict_proba(transformed)[0][positive_index]
        )

        detected = score >= threshold

        fault_name = target.replace("fault_", "").upper()

        results[fault_name] = {
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "detected": bool(detected),
        }

        if detected:
            detected_faults.append(fault_name)

    return {
        "model": "Multi-label Random Forest",
        "training_domain": "Synthetic multi-label fault simulator",
        "detected_faults": detected_faults,
        "fault_count": len(detected_faults),
        "fault_results": results,
        "score_interpretation": (
            "Random Forest scores; not guaranteed calibrated probabilities."
        ),
    }


# ---------------------------------------------------------
# SENSOR QUALITY ASSESSMENT
# ---------------------------------------------------------

def assess_sensor_quality(reading):

    # The simulator can explicitly mark a known injected
    # sensor-quality problem.
    #
    # Without that metadata, a single reading cannot reliably
    # establish sensor noise or sensor drift.

    issue = bool(reading.get("sensor_quality_issue", False))

    return {
        "status": (
            "SENSOR_QUALITY_WARNING"
            if issue else "NOT_ASSESSED"
        ),
        "known_simulation_flag": issue,
        "explanation": (
            "The simulator explicitly marked a sensor-quality issue."
            if issue else
            "Sensor quality was not independently assessed. "
            "Historical sensor readings are required for stronger "
            "noise and drift detection."
        ),
    }


# ---------------------------------------------------------
# DIAGNOSTIC SYNTHESIS
# ---------------------------------------------------------

def synthesize_diagnosis(
    xgb_result,
    anomaly_result,
    legacy_fault_result,
    multilabel_result,
    sensor_quality_result,
):

    detected_faults = multilabel_result["detected_faults"]

    anomaly_alert = anomaly_result["prediction"] == "ANOMALY"

    xgb_alert = xgb_result["prediction"] == "FAILURE_RISK"

    sensor_warning = (
        sensor_quality_result["status"] == "SENSOR_QUALITY_WARNING"
    )

    fault_count = len(detected_faults)

    # -----------------------------------------------------
    # Primary diagnosis
    # -----------------------------------------------------

    if fault_count > 1:
        diagnosis = "MULTIPLE_FAULTS"

    elif fault_count == 1:
        diagnosis = detected_faults[0]

    elif anomaly_alert:
        diagnosis = "UNCLASSIFIED_ANOMALY"

    elif xgb_alert:
        diagnosis = "POTENTIAL_FAILURE_RISK"

    elif sensor_warning:
        diagnosis = "SENSOR_QUALITY_WARNING"

    else:
        diagnosis = "NORMAL_OPERATION"

    # -----------------------------------------------------
    # Evidence signals
    # -----------------------------------------------------

    evidence = {
        "xgboost_failure_alert": bool(xgb_alert),
        "isolation_forest_anomaly": bool(anomaly_alert),
        "multilabel_fault_count": fault_count,
        "legacy_classifier_prediction": (
            legacy_fault_result["predicted_fault"]
        ),
        "sensor_quality_warning": bool(sensor_warning),
    }

    # The legacy classifier is retained as a comparison model.
    # It is not counted as an independent vote because it uses
    # related synthetic sensor data and overlapping features.

    supporting_alert_count = (
        int(xgb_alert)
        + int(anomaly_alert)
        + int(fault_count > 0)
    )

    # -----------------------------------------------------
    # Transparent severity rules
    # -----------------------------------------------------

    if fault_count >= 3 and (anomaly_alert or xgb_alert):
        severity = "CRITICAL"
        severity_reason = (
            "Three fault types detected with additional model evidence."
        )

    elif fault_count >= 2 or (anomaly_alert and xgb_alert):
        severity = "HIGH"
        severity_reason = (
            "Multiple fault detections or corroborating model alerts."
        )

    elif fault_count == 1 or anomaly_alert or xgb_alert or sensor_warning:

        severity = "WARNING"

        if sensor_warning and not (
            fault_count or anomaly_alert or xgb_alert
        ):
            severity_reason = (
                "Sensor quality issue detected; equipment condition "
                "cannot be fully assessed from this reading."
            )

        else:
            severity_reason = (
                "At least one fault, sensor-quality or risk signal "
                "requires investigation."
            )

    else:
        severity = "NORMAL"
        severity_reason = "No equipment fault or risk alert was raised."

    # -----------------------------------------------------
    # Diagnostic reliability and maintenance action
    # -----------------------------------------------------

    if sensor_warning:
        diagnostic_reliability = "COMPROMISED"
        maintenance_action = "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"

    elif fault_count >= 2 or anomaly_alert or xgb_alert:
        diagnostic_reliability = "ACTIONABLE_WITH_INVESTIGATION"
        maintenance_action = "INVESTIGATE_EQUIPMENT"

    elif fault_count == 1:
        diagnostic_reliability = "ACTIONABLE_WITH_INVESTIGATION"
        maintenance_action = "INSPECT_EQUIPMENT"

    else:
        diagnostic_reliability = "NO_ALERT"
        maintenance_action = "CONTINUE_ROUTINE_MONITORING"

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = []

    for fault in detected_faults:
        recommendations.extend(
            FAULT_RECOMMENDATIONS.get(fault, [])
        )

    if diagnosis == "UNCLASSIFIED_ANOMALY":
        recommendations.append(
            "Inspect sensor readings and machine operating conditions "
            "to identify the source of the anomaly."
        )

    if diagnosis == "POTENTIAL_FAILURE_RISK":
        recommendations.append(
            "Schedule a machine inspection and verify the elevated "
            "failure-risk signal."
        )

    if sensor_warning:
        recommendations.append(
            "Inspect sensor connections, calibration and data acquisition."
        )

    if diagnosis == "NORMAL_OPERATION" and not sensor_warning:
        recommendations.append(
            "Continue routine monitoring and scheduled maintenance."
        )

    if fault_count > 1:
        recommendations.append(
            "Review all detected fault signals together before "
            "authorizing maintenance."
        )

    # Remove duplicate recommendations while preserving order.
    recommendations = list(dict.fromkeys(recommendations))

    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "severity_reason": severity_reason,
        "detected_faults": detected_faults,
        "supporting_alert_count": supporting_alert_count,
        "evidence": evidence,
        "sensor_quality": sensor_quality_result,
        "recommendations": recommendations,
        "diagnostic_reliability": diagnostic_reliability,
        "maintenance_action": maintenance_action,
        "decision_policy": (
            "Transparent rule-based synthesis; not a learned "
            "or independently validated severity model."
        ),
    }


# ---------------------------------------------------------
# COMPLETE MACHINE DIAGNOSIS
# ---------------------------------------------------------

def diagnose_machine(reading):

    validate_reading(reading)

    xgb_result = predict_xgboost(reading)

    anomaly_result = detect_anomaly(reading)

    legacy_fault_result = classify_fault(reading)

    multilabel_result = classify_multiple_faults(reading)

    sensor_quality_result = assess_sensor_quality(reading)

    final_diagnosis = synthesize_diagnosis(
        xgb_result,
        anomaly_result,
        legacy_fault_result,
        multilabel_result,
        sensor_quality_result,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "machine_id": reading["machine_id"],

        "machine_type": reading["machine_type"],

        "input_reading": reading,

        "model_outputs": {
            "failure_prediction": xgb_result,
            "anomaly_detection": anomaly_result,
            "fault_classification": legacy_fault_result,
            "multilabel_fault_classification": multilabel_result,
        },

        "final_diagnosis": final_diagnosis,

        "limitations": [
            "The XGBoost model was trained on the UCI AI4I 2020 dataset.",
            "The anomaly detector and fault classifiers were trained "
            "on synthetic industrial simulator data.",
            "The multi-label fault model learns controlled synthetic "
            "fault-injection signatures.",
            "Model scores are not guaranteed calibrated probabilities.",
            "Sensor quality is not independently established from "
            "a single reading.",
            "Severity is assigned using transparent engineering rules "
            "that require further validation.",
            "The system is a research prototype, not a certified "
            "industrial safety system.",
        ],
    }


# ---------------------------------------------------------
# DEMONSTRATION
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_reading = {
        "machine_id": "DEMO-MACHINE-001",
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 200.0,
        "operating_hours": 250.0,
    }

    result = diagnose_machine(sample_reading)

    print("=" * 70)
    print("INDUSTRIAL FAILURE INTELLIGENCE ENGINE")
    print("=" * 70)

    print("\n===== MODEL OUTPUTS =====")

    for model_name, output in result["model_outputs"].items():

        print(f"\n{model_name.upper()}")

        for key, value in output.items():
            print(f"{key}: {value}")

    print("\n===== FINAL DIAGNOSIS =====")

    for key, value in result["final_diagnosis"].items():
        print(f"{key}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "ensemble_diagnosis_demo.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"\nSaved diagnosis report: {output_path}")
