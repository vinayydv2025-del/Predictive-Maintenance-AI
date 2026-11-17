
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fleet_intelligence.fleet_health import calculate_fleet_health


def create_machine(machine_id, diagnosis, severity, faults=None, sensor_warning=False):

    return {
        "machine_id": machine_id,
        "final_diagnosis": {
            "diagnosis": diagnosis,
            "severity": severity,
            "detected_faults": faults or [],
            "supporting_alert_count": len(faults or []),
            "diagnostic_reliability": (
                "COMPROMISED"
                if sensor_warning
                else "ACTIONABLE_WITH_INVESTIGATION"
                if severity != "NORMAL"
                else "NO_ALERT"
            ),
            "maintenance_action": (
                "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"
                if sensor_warning
                else "CONTINUE_ROUTINE_MONITORING"
                if severity == "NORMAL"
                else "INVESTIGATE_EQUIPMENT"
            ),
            "sensor_quality": {
                "status": (
                    "SENSOR_QUALITY_WARNING"
                    if sensor_warning
                    else "NOT_ASSESSED"
                )
            },
        },
    }


# ---------------------------------------------------------
# SYNTHETIC FLEET
# ---------------------------------------------------------

machine_results = [

    create_machine("MACHINE-001", "NORMAL_OPERATION", "NORMAL"),

    create_machine("MACHINE-002", "NORMAL_OPERATION", "NORMAL"),

    create_machine("MACHINE-003", "NORMAL_OPERATION", "NORMAL"),

    create_machine(
        "MACHINE-004",
        "OVERHEATING",
        "WARNING",
        ["OVERHEATING"],
    ),

    create_machine(
        "MACHINE-005",
        "OVERLOAD",
        "WARNING",
        ["OVERLOAD"],
    ),

    create_machine(
        "MACHINE-006",
        "MULTIPLE_FAULTS",
        "HIGH",
        ["OVERHEATING", "OVERLOAD"],
    ),

    create_machine(
        "MACHINE-007",
        "MULTIPLE_FAULTS",
        "CRITICAL",
        ["OVERHEATING", "OVERLOAD", "SPEED_ANOMALY"],
    ),

    create_machine(
        "MACHINE-008",
        "MULTIPLE_FAULTS",
        "CRITICAL",
        ["OVERHEATING", "OVERLOAD", "SPEED_ANOMALY"],
        sensor_warning=True,
    ),
]


# ---------------------------------------------------------
# RUN FLEET ANALYSIS
# ---------------------------------------------------------

report = calculate_fleet_health(machine_results)


# ---------------------------------------------------------
# DISPLAY REPORT
# ---------------------------------------------------------

print("=" * 75)
print("FLEET HEALTH INTELLIGENCE TEST")
print("=" * 75)

print("\nFLEET OVERVIEW")

for key, value in report["fleet_overview"].items():
    print(f"{key}: {value}")


print("\nSEVERITY DISTRIBUTION")

for severity, count in report["severity_distribution"]["counts"].items():
    percentage = report["severity_distribution"]["percentages"][severity]

    print(
        f"{severity:<12} "
        f"Count={count:<3} "
        f"Percentage={percentage}%"
    )


print("\nDIAGNOSIS DISTRIBUTION")

for diagnosis, count in report["diagnosis_distribution"].items():
    print(f"{diagnosis}: {count}")


print("\nFAULT SUMMARY")

for key, value in report["fault_summary"].items():
    print(f"{key}: {value}")


print("\nSENSOR WARNING MACHINES")

print(report["maintenance_overview"]["sensor_warning_machines"])


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

checks = {

    "Total machine count is 8":
        report["fleet_overview"]["total_machines"] == 8,

    "Normal machine count is 3":
        report["severity_distribution"]["counts"]["NORMAL"] == 3,

    "Warning machine count is 2":
        report["severity_distribution"]["counts"]["WARNING"] == 2,

    "High-risk machine count is 1":
        report["severity_distribution"]["counts"]["HIGH"] == 1,

    "Critical machine count is 2":
        report["severity_distribution"]["counts"]["CRITICAL"] == 2,

    "Fleet status is CRITICAL":
        report["fleet_overview"]["fleet_status"] == "CRITICAL",

    "Attention percentage is 62.5":
        report["fleet_overview"]["attention_percentage"] == 62.5,

    "One sensor warning is identified":
        report["maintenance_overview"]["sensor_warning_machines"]
        == ["MACHINE-008"],
}


print("\nVALIDATION RESULTS")

for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

output_path = REPORT_DIR / "fleet_health_demo.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


print("\nReport saved:")
print(output_path)

failed = [name for name, passed in checks.items() if not passed]

print(f"\nPassed: {len(checks) - len(failed)}/{len(checks)}")

if failed:
    print("Some validation checks failed.")
else:
    print("All fleet health validation checks passed.")
