
import json
import sys
from pathlib import Path

import numpy as np

# Make project imports reliable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.intelligence.ensemble_engine import diagnose_machine
from src.simulation.faults import FaultInjector


BASE_READING = {
    "machine_id": "MACHINE-TEST-001",
    "machine_type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 200.0,
    "operating_hours": 250.0,
}

fault_injector = FaultInjector()


def create_scenarios():
    scenarios = []

    # 1. Normal operation
    normal = BASE_READING.copy()
    normal["machine_id"] = "TEST-NORMAL"
    scenarios.append({
        "name": "NORMAL_OPERATION",
        "reading": normal,
        "expected_fault": False,
    })

    # 2. Overheating
    overheating = fault_injector.inject(
        BASE_READING.copy(),
        "OVERHEATING"
    )
    overheating["machine_id"] = "TEST-OVERHEATING"
    scenarios.append({
        "name": "OVERHEATING",
        "reading": overheating,
        "expected_fault": True,
    })

    # 3. Overload
    overload = fault_injector.inject(
        BASE_READING.copy(),
        "OVERLOAD"
    )
    overload["machine_id"] = "TEST-OVERLOAD"
    scenarios.append({
        "name": "OVERLOAD",
        "reading": overload,
        "expected_fault": True,
    })

    # 4. Abnormal rotational speed
    speed_anomaly = fault_injector.inject(
        BASE_READING.copy(),
        "SPEED_ANOMALY"
    )
    speed_anomaly["machine_id"] = "TEST-SPEED-ANOMALY"
    scenarios.append({
        "name": "SPEED_ANOMALY",
        "reading": speed_anomaly,
        "expected_fault": True,
    })

    # 5. Multiple simultaneous sensor disturbances
    overlapping = BASE_READING.copy()
    overlapping["machine_id"] = "TEST-OVERLAPPING"
    overlapping["air_temperature"] += 15
    overlapping["process_temperature"] += 20
    overlapping["torque"] += 25
    overlapping["rotational_speed"] -= 250

    scenarios.append({
        "name": "OVERLAPPING_FAULT_SIGNALS",
        "reading": overlapping,
        "expected_fault": True,
    })

    # 6. Severe sensor noise
    noisy = BASE_READING.copy()
    noisy["machine_id"] = "TEST-SEVERE-NOISE"

    rng = np.random.default_rng(42)

    noise_scales = {
        "air_temperature": 2.1,
        "process_temperature": 2.1,
        "rotational_speed": 75,
        "torque": 6,
        "tool_wear": 3,
        "operating_hours": 0.3,
    }

    for feature, scale in noise_scales.items():
        noisy[feature] += float(rng.normal(0, scale))

    scenarios.append({
        "name": "SEVERE_SENSOR_NOISE",
        "reading": noisy,
        "expected_fault": None,
    })

    return scenarios


def main():
    print("=" * 75)
    print("INDUSTRIAL DIAGNOSTIC SCENARIO TEST SUITE")
    print("=" * 75)

    scenarios = create_scenarios()
    results = []

    for index, scenario in enumerate(scenarios, start=1):
        print(
            f"\n[{index}/{len(scenarios)}] "
            f"Testing: {scenario['name']}"
        )

        diagnosis = diagnose_machine(scenario["reading"])

        model_outputs = diagnosis["model_outputs"]
        final_diagnosis = diagnosis["final_diagnosis"]

        xgb_prediction = (
            model_outputs["failure_prediction"]["prediction"]
        )

        anomaly_prediction = (
            model_outputs["anomaly_detection"]["prediction"]
        )

        fault_prediction = (
            model_outputs["fault_classification"]["predicted_fault"]
        )

        final_label = final_diagnosis["diagnosis"]
        severity = final_diagnosis["severity"]

        predicted_fault = (
            anomaly_prediction == "ANOMALY"
            or fault_prediction != "NORMAL"
            or xgb_prediction == "FAILURE_ALERT"
        )

        if scenario["expected_fault"] is None:
            test_status = "OBSERVATION_ONLY"
        else:
            test_status = (
                "PASS"
                if predicted_fault == scenario["expected_fault"]
                else "CHECK_REQUIRED"
            )

        result = {
            "scenario": scenario["name"],
            "xgboost_prediction": xgb_prediction,
            "anomaly_prediction": anomaly_prediction,
            "fault_classification": fault_prediction,
            "final_diagnosis": final_label,
            "severity": severity,
            "supporting_alert_count": (
                final_diagnosis["supporting_alert_count"]
            ),
            "test_status": test_status,
            "full_diagnosis": diagnosis,
        }

        results.append(result)

        print(f"XGBoost:             {xgb_prediction}")
        print(f"Anomaly detector:    {anomaly_prediction}")
        print(f"Fault classifier:    {fault_prediction}")
        print(f"Final diagnosis:     {final_label}")
        print(f"Severity:            {severity}")
        print(f"Test status:         {test_status}")

    # Summary
    evaluated = [
        item for item in results
        if item["test_status"] != "OBSERVATION_ONLY"
    ]

    passed = sum(
        item["test_status"] == "PASS"
        for item in evaluated
    )

    failed = sum(
        item["test_status"] == "CHECK_REQUIRED"
        for item in evaluated
    )

    summary = {
        "total_scenarios": len(results),
        "evaluated_scenarios": len(evaluated),
        "passed": passed,
        "check_required": failed,
        "observation_only": len(results) - len(evaluated),
        "pass_rate": (
            round(passed / len(evaluated), 4)
            if evaluated else None
        ),
        "results": results,
        "limitations": [
            "Fault scenarios are synthetically generated.",
            "Severe sensor noise is an observation-only test.",
            "Passing these tests does not establish real-world industrial validity.",
            "XGBoost and simulator models use different training domains.",
        ],
    }

    report_path = (
        PROJECT_ROOT
        / "reports"
        / "diagnostic_scenario_test_results.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 75)
    print("SCENARIO TEST SUMMARY")
    print("=" * 75)

    print(f"Total scenarios:      {summary['total_scenarios']}")
    print(f"Evaluated scenarios:  {summary['evaluated_scenarios']}")
    print(f"Passed:               {passed}")
    print(f"Check required:       {failed}")
    print(f"Observation only:     {summary['observation_only']}")

    if summary["pass_rate"] is not None:
        print(f"Pass rate:            {summary['pass_rate']:.2%}")

    print(f"\nReport saved: {report_path}")
    print("\nDIAGNOSTIC SCENARIO TESTING COMPLETED")


if __name__ == "__main__":
    main()
