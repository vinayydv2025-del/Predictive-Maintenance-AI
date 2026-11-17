
"""
FLEET ANALYTICS INTEGRATION TEST

Tests the combined fleet intelligence engine using
eight constructed diagnostic results.

These are test fixtures, not real machine diagnoses.
"""

import json
import sys
from pathlib import Path


# Configure project import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))


from fleet_intelligence.fleet_analytics import (
    generate_fleet_analytics,
)


def create_test_fleet():
    """Create eight constructed machine diagnostic records."""

    return [
        {
            "machine_id": "MACHINE-001",
            "final_diagnosis": {
                "diagnosis": "NORMAL_OPERATION",
                "severity": "NORMAL",
                "detected_faults": [],
                "supporting_alert_count": 0,
                "diagnostic_reliability": "NO_ALERT",
                "maintenance_action": "CONTINUE_ROUTINE_MONITORING",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-002",
            "final_diagnosis": {
                "diagnosis": "NORMAL_OPERATION",
                "severity": "NORMAL",
                "detected_faults": [],
                "supporting_alert_count": 0,
                "diagnostic_reliability": "NO_ALERT",
                "maintenance_action": "CONTINUE_ROUTINE_MONITORING",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-003",
            "final_diagnosis": {
                "diagnosis": "NORMAL_OPERATION",
                "severity": "NORMAL",
                "detected_faults": [],
                "supporting_alert_count": 0,
                "diagnostic_reliability": "NO_ALERT",
                "maintenance_action": "CONTINUE_ROUTINE_MONITORING",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-004",
            "final_diagnosis": {
                "diagnosis": "OVERHEATING",
                "severity": "WARNING",
                "detected_faults": ["OVERHEATING"],
                "supporting_alert_count": 1,
                "diagnostic_reliability": "ACTIONABLE_WITH_INVESTIGATION",
                "maintenance_action": "INVESTIGATE_EQUIPMENT",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-005",
            "final_diagnosis": {
                "diagnosis": "OVERLOAD",
                "severity": "WARNING",
                "detected_faults": ["OVERLOAD"],
                "supporting_alert_count": 1,
                "diagnostic_reliability": "ACTIONABLE_WITH_INVESTIGATION",
                "maintenance_action": "INVESTIGATE_EQUIPMENT",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-006",
            "final_diagnosis": {
                "diagnosis": "MULTIPLE_FAULTS",
                "severity": "HIGH",
                "detected_faults": [
                    "OVERHEATING",
                    "OVERLOAD",
                ],
                "supporting_alert_count": 2,
                "diagnostic_reliability": "ACTIONABLE_WITH_INVESTIGATION",
                "maintenance_action": "INVESTIGATE_EQUIPMENT",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-007",
            "final_diagnosis": {
                "diagnosis": "MULTIPLE_FAULTS",
                "severity": "CRITICAL",
                "detected_faults": [
                    "OVERHEATING",
                    "OVERLOAD",
                    "SPEED_ANOMALY",
                ],
                "supporting_alert_count": 3,
                "diagnostic_reliability": "ACTIONABLE_WITH_INVESTIGATION",
                "maintenance_action": "INVESTIGATE_EQUIPMENT",
                "sensor_quality": {"status": "NOT_ASSESSED"},
            },
        },
        {
            "machine_id": "MACHINE-008",
            "final_diagnosis": {
                "diagnosis": "MULTIPLE_FAULTS",
                "severity": "CRITICAL",
                "detected_faults": [
                    "OVERHEATING",
                    "OVERLOAD",
                    "SPEED_ANOMALY",
                ],
                "supporting_alert_count": 3,
                "diagnostic_reliability": "COMPROMISED",
                "maintenance_action": "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT",
                "sensor_quality": {"status": "WARNING"},
            },
        },
    ]


def run_tests():

    print("=" * 85)
    print("FLEET ANALYTICS INTEGRATION TEST")
    print("=" * 85)

    machine_results = create_test_fleet()

    report = generate_fleet_analytics(machine_results)

    # Extract nested report sections.
    health = report["fleet_health"]
    overview = health["fleet_overview"]
    ranking = report["risk_ranking"]

    checks = []

    # 1. Fleet overview checks.
    checks.append((
        "All eight machines included in fleet report",
        overview["total_machines"] == 8,
    ))

    checks.append((
        "Fleet health reports critical status",
        overview["fleet_status"] == "CRITICAL",
    ))

    checks.append((
        "Fleet health percentage is 37.5%",
        overview["fleet_health_percentage"] == 37.5,
    ))

    checks.append((
        "Five machines require attention",
        overview["machines_requiring_attention"] == 5,
    ))

    # 2. Risk ranking checks.
    checks.append((
        "All eight machines appear in risk ranking",
        len(ranking) == 8,
    ))

    checks.append((
        "Sensor-compromised critical machine ranks first",
        ranking[0]["machine_id"] == "MACHINE-008",
    ))

    # 3. Severity checks.
    checks.append((
        "Two critical and one high-risk machine identified",
        report["critical_machine_count"] == 2
        and report["high_risk_machine_count"] == 1,
    ))

    # 4. Fault frequency checks.
    checks.append((
        "Fault frequency correctly aggregated",
        report["fault_frequency"] == {
            "OVERHEATING": 4,
            "OVERLOAD": 4,
            "SPEED_ANOMALY": 2,
        },
    ))

    # 5. Maintenance distribution checks.
    checks.append((
        "Maintenance action distribution correct",
        report["maintenance_action_distribution"] == {
            "CONTINUE_ROUTINE_MONITORING": 3,
            "INVESTIGATE_EQUIPMENT": 4,
            "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT": 1,
        },
    ))

    # 6. Sensor quality checks.
    checks.append((
        "Sensor warning correctly identified",
        report["sensor_quality_summary"]["sensor_warning_count"] == 1
        and report["sensor_quality_summary"][
            "machines_with_sensor_warnings"
        ] == ["MACHINE-008"],
    ))

    # 7. Analytics metadata checks.
    checks.append((
        "Analytics metadata correctly identifies rule-based ranking",
        report["analytics_metadata"]["risk_score_is_probability"] is False
        and report["analytics_metadata"]["ml_models_executed"] is False,
    ))

    # Display fleet summary.
    print("\n===== FLEET SUMMARY =====")

    print(f"Total machines: {overview['total_machines']}")
    print(f"Fleet status: {overview['fleet_status']}")
    print(
        f"Fleet health percentage: "
        f"{overview['fleet_health_percentage']}%"
    )
    print(
        f"Machines requiring attention: "
        f"{overview['machines_requiring_attention']}"
    )

    # Display fault frequency.
    print("\n===== FAULT FREQUENCY =====")

    for fault, count in report["fault_frequency"].items():
        print(f"{fault}: {count}")

    # Display maintenance distribution.
    print("\n===== MAINTENANCE DISTRIBUTION =====")

    for action, count in report[
        "maintenance_action_distribution"
    ].items():
        print(f"{action}: {count}")

    # Display highest-priority machine.
    print("\n===== HIGHEST PRIORITY MACHINE =====")

    highest = report["highest_priority_machine"]

    print(
        f"{highest['machine_id']} | "
        f"Severity={highest['severity']} | "
        f"Score={highest['ranking_score']} | "
        f"Action={highest['maintenance_action']}"
    )

    # Display validation results.
    print("\n===== VALIDATION RESULTS =====")

    passed = 0

    for description, result in checks:

        status = "PASS" if result else "FAIL"

        print(f"[{status}] {description}")

        if result:
            passed += 1

    # Save report.
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "fleet_analytics_demo.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 85)
    print(f"Passed: {passed}/{len(checks)}")
    print(f"Report saved: {output_path}")
    print("=" * 85)

    if passed == len(checks):
        print("ALL FLEET ANALYTICS INTEGRATION TESTS PASSED.")
    else:
        raise AssertionError(
            "One or more fleet analytics tests failed."
        )


if __name__ == "__main__":
    run_tests()
