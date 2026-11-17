
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from simulation.faults import FaultInjector
from intelligence.ensemble_engine import diagnose_machine


# ---------------------------------------------------------
# TEST CONFIGURATION
# ---------------------------------------------------------

BASE_READING = {
    "machine_id": "TEST-MACHINE-001",
    "machine_type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 200.0,
    "operating_hours": 250.0,
}

injector = FaultInjector(random_seed=42)


# ---------------------------------------------------------
# CREATE SCENARIOS
# ---------------------------------------------------------

scenarios = []


def add_scenario(name, reading, expected_description):
    scenarios.append({
        "name": name,
        "reading": reading,
        "expected": expected_description,
    })


# 1. Normal operation
normal = injector.normal_reading(BASE_READING)

add_scenario(
    "NORMAL",
    normal,
    "No injected equipment fault",
)


# 2. Overheating
overheating = injector.inject(BASE_READING, "OVERHEATING")

add_scenario(
    "OVERHEATING",
    overheating,
    "Overheating fault",
)


# 3. Overload
overload = injector.inject(BASE_READING, "OVERLOAD")

add_scenario(
    "OVERLOAD",
    overload,
    "Overload fault",
)


# 4. Speed anomaly
speed = injector.inject(BASE_READING, "SPEED_ANOMALY")

add_scenario(
    "SPEED_ANOMALY",
    speed,
    "Speed anomaly fault",
)


# 5. Two simultaneous faults
double_fault = injector.inject_multiple(
    BASE_READING,
    ["OVERHEATING", "OVERLOAD"],
)

add_scenario(
    "DOUBLE_FAULT",
    double_fault,
    "Overheating and overload",
)


# 6. Three simultaneous faults
triple_fault = injector.inject_multiple(
    BASE_READING,
    [
        "OVERHEATING",
        "OVERLOAD",
        "SPEED_ANOMALY",
    ],
)

add_scenario(
    "TRIPLE_FAULT",
    triple_fault,
    "All three equipment faults",
)


# 7. Sensor noise without equipment fault
sensor_noise = injector.inject_sensor_noise(
    BASE_READING,
    noise_multiplier=4.0,
)

add_scenario(
    "SEVERE_SENSOR_NOISE",
    sensor_noise,
    "Sensor quality issue without injected equipment fault",
)


# 8. Three equipment faults PLUS sensor-quality warning
triple_fault_sensor_issue = injector.inject_multiple(
    BASE_READING,
    [
        "OVERHEATING",
        "OVERLOAD",
        "SPEED_ANOMALY",
    ],
)

# Explicitly mark the sensor-quality issue.
# This tests diagnostic decision handling when both
# equipment faults and a known sensor issue are present.
triple_fault_sensor_issue["sensor_quality_issue"] = True

add_scenario(
    "TRIPLE_FAULT_WITH_SENSOR_ISSUE",
    triple_fault_sensor_issue,
    "Three equipment faults with a sensor-quality warning",
)


# ---------------------------------------------------------
# RUN DIAGNOSTIC TESTS
# ---------------------------------------------------------

results = []

print("=" * 85)
print("ENHANCED INDUSTRIAL DIAGNOSTIC SCENARIO TESTING")
print("=" * 85)

for scenario in scenarios:

    result = diagnose_machine(scenario["reading"])

    final = result["final_diagnosis"]
    models = result["model_outputs"]
    multilabel = models["multilabel_fault_classification"]

    row = {
        "scenario": scenario["name"],
        "expected": scenario["expected"],
        "diagnosis": final["diagnosis"],
        "severity": final["severity"],
        "detected_faults": final["detected_faults"],
        "diagnostic_reliability": final["diagnostic_reliability"],
        "maintenance_action": final["maintenance_action"],
        "xgboost": models["failure_prediction"]["prediction"],
        "anomaly_detector": models["anomaly_detection"]["prediction"],
        "legacy_classifier": models["fault_classification"]["predicted_fault"],
        "multilabel_fault_count": multilabel["fault_count"],
        "sensor_quality": final["sensor_quality"]["status"],
        "supporting_alert_count": final["supporting_alert_count"],
    }

    results.append(row)

    print("\n" + "-" * 85)
    print(f"SCENARIO: {scenario['name']}")
    print("-" * 85)

    print(f"Expected:                {scenario['expected']}")
    print(f"Final diagnosis:         {row['diagnosis']}")
    print(f"Severity:                {row['severity']}")
    print(f"Detected faults:         {row['detected_faults']}")
    print(f"Diagnostic reliability:  {row['diagnostic_reliability']}")
    print(f"Maintenance action:      {row['maintenance_action']}")
    print(f"XGBoost:                 {row['xgboost']}")
    print(f"Anomaly detector:        {row['anomaly_detector']}")
    print(f"Legacy classifier:       {row['legacy_classifier']}")
    print(f"Multi-label count:       {row['multilabel_fault_count']}")
    print(f"Sensor quality:          {row['sensor_quality']}")

    print("\nIndividual fault scores:")

    for fault, details in multilabel["fault_results"].items():
        print(
            f"  {fault}: "
            f"score={details['score']:.4f}, "
            f"threshold={details['threshold']:.4f}, "
            f"detected={details['detected']}"
        )


# ---------------------------------------------------------
# VALIDATE IMPORTANT DECISION RULES
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("DIAGNOSTIC DECISION RULE VALIDATION")
print("=" * 85)

results_by_name = {
    row["scenario"]: row
    for row in results
}

checks = {
    "Normal operation has no alert": (
        results_by_name["NORMAL"]["diagnostic_reliability"]
        == "NO_ALERT"
        and
        results_by_name["NORMAL"]["maintenance_action"]
        == "CONTINUE_ROUTINE_MONITORING"
    ),

    "Sensor-only issue requests sensor verification": (
        results_by_name["SEVERE_SENSOR_NOISE"]["diagnosis"]
        == "SENSOR_QUALITY_WARNING"
        and
        results_by_name["SEVERE_SENSOR_NOISE"]["severity"]
        == "WARNING"
        and
        results_by_name["SEVERE_SENSOR_NOISE"]["diagnostic_reliability"]
        == "COMPROMISED"
        and
        results_by_name["SEVERE_SENSOR_NOISE"]["maintenance_action"]
        == "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"
    ),

    "Combined triple fault retains critical severity": (
        results_by_name["TRIPLE_FAULT_WITH_SENSOR_ISSUE"]["diagnosis"]
        == "MULTIPLE_FAULTS"
        and
        results_by_name["TRIPLE_FAULT_WITH_SENSOR_ISSUE"]["severity"]
        == "CRITICAL"
    ),

    "Combined triple fault requests sensor verification": (
        results_by_name["TRIPLE_FAULT_WITH_SENSOR_ISSUE"]["diagnostic_reliability"]
        == "COMPROMISED"
        and
        results_by_name["TRIPLE_FAULT_WITH_SENSOR_ISSUE"]["maintenance_action"]
        == "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"
    ),
}

for check_name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {check_name}")


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

output_path = REPORT_DIR / "enhanced_diagnostic_scenarios.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)


csv_path = REPORT_DIR / "enhanced_diagnostic_scenarios.csv"

pd.DataFrame(results).to_csv(csv_path, index=False)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("SCENARIO TEST SUMMARY")
print("=" * 85)

for row in results:
    print(
        f"{row['scenario']:<32} "
        f"Diagnosis={row['diagnosis']:<22} "
        f"Severity={row['severity']:<10} "
        f"Reliability={row['diagnostic_reliability']:<25} "
        f"Action={row['maintenance_action']}"
    )

failed_checks = [
    name for name, passed in checks.items()
    if not passed
]

print("\nReports saved:")
print(output_path)
print(csv_path)

print("\nDecision-rule checks:")
print(f"Passed: {len(checks) - len(failed_checks)}/{len(checks)}")

if failed_checks:
    print("\nSome decision-rule checks failed:")
    for name in failed_checks:
        print(f" - {name}")
else:
    print("All diagnostic decision-rule checks passed.")

print("\nTEST EXECUTION COMPLETED")
print(
    "Note: These are synthetic scenario tests, "
    "not independent real-world model validation."
)
