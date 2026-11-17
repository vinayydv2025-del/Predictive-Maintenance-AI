
"""
FLEET HEALTH ANALYZER

Aggregates machine-level diagnostic results into
a fleet-level operational health summary.

This module does not train or execute ML models.
It consumes outputs from the existing diagnostic engine.
"""

from collections import Counter
from datetime import datetime, timezone


SEVERITY_ORDER = {
    "NORMAL": 0,
    "WARNING": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


def validate_machine_results(machine_results):
    """
    Validate the structure of machine diagnostic results.
    """

    if not isinstance(machine_results, list):
        raise TypeError("machine_results must be a list.")

    if not machine_results:
        raise ValueError("At least one machine result is required.")

    machine_ids = []

    for index, result in enumerate(machine_results):

        if not isinstance(result, dict):
            raise TypeError(
                f"Machine result at index {index} must be a dictionary."
            )

        machine_id = result.get("machine_id")

        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError(
                f"Machine result at index {index} has no valid machine_id."
            )

        final = result.get("final_diagnosis")

        if not isinstance(final, dict):
            raise ValueError(
                f"Machine {machine_id} has no valid final_diagnosis."
            )

        severity = final.get("severity")

        if severity not in SEVERITY_ORDER:
            raise ValueError(
                f"Machine {machine_id} has unsupported severity: {severity}"
            )

        detected_faults = final.get("detected_faults", [])

        if not isinstance(detected_faults, list):
            raise ValueError(
                f"Detected faults for {machine_id} must be a list."
            )

        supporting_alerts = final.get("supporting_alert_count", 0)

        if (
            not isinstance(supporting_alerts, int)
            or isinstance(supporting_alerts, bool)
            or supporting_alerts < 0
        ):
            raise ValueError(
                f"Invalid supporting_alert_count for {machine_id}."
            )

        machine_ids.append(machine_id)

    if len(machine_ids) != len(set(machine_ids)):
        raise ValueError("Duplicate machine_id values detected.")

    return True


def calculate_fleet_health(machine_results):
    """
    Aggregate machine diagnostic results into fleet-level metrics.

    Fleet health percentage represents the percentage of machines
    classified as NORMAL by the diagnostic engine.

    It is not a direct measurement of physical equipment health.
    """

    validate_machine_results(machine_results)

    total_machines = len(machine_results)

    severity_counts = Counter()
    diagnosis_counts = Counter()

    critical_machines = []
    high_risk_machines = []
    warning_machines = []
    sensor_warning_machines = []

    total_detected_faults = 0
    total_supporting_alerts = 0

    for result in machine_results:

        machine_id = result["machine_id"]
        final = result["final_diagnosis"]

        severity = final["severity"]
        diagnosis = final.get("diagnosis", "UNKNOWN")

        severity_counts[severity] += 1
        diagnosis_counts[diagnosis] += 1

        detected_faults = final.get("detected_faults", [])

        total_detected_faults += len(detected_faults)

        total_supporting_alerts += final.get(
            "supporting_alert_count",
            0
        )

        machine_summary = {
            "machine_id": machine_id,
            "diagnosis": diagnosis,
            "severity": severity,
            "detected_faults": detected_faults,
            "diagnostic_reliability": final.get(
                "diagnostic_reliability",
                "NOT_PROVIDED",
            ),
            "maintenance_action": final.get(
                "maintenance_action",
                "NOT_PROVIDED",
            ),
        }

        if severity == "CRITICAL":
            critical_machines.append(machine_summary)

        elif severity == "HIGH":
            high_risk_machines.append(machine_summary)

        elif severity == "WARNING":
            warning_machines.append(machine_summary)

        # Sensor status is supplied by the diagnostic engine.
        sensor_quality = final.get("sensor_quality", {})

        if isinstance(sensor_quality, dict):
            sensor_status = sensor_quality.get("status")

            if sensor_status in (
                "WARNING",
                "SENSOR_QUALITY_WARNING",
            ):
                sensor_warning_machines.append(machine_id)

    # Include zero-count severity categories for consistent reporting.
    for severity in SEVERITY_ORDER:
        severity_counts.setdefault(severity, 0)

    normal_count = severity_counts["NORMAL"]
    warning_count = severity_counts["WARNING"]
    high_count = severity_counts["HIGH"]
    critical_count = severity_counts["CRITICAL"]

    machines_requiring_attention = (
        warning_count + high_count + critical_count
    )

    fleet_health_percentage = (
        normal_count / total_machines
    ) * 100

    if critical_count > 0:
        fleet_status = "CRITICAL"

    elif high_count > 0:
        fleet_status = "HIGH_RISK"

    elif warning_count > 0:
        fleet_status = "WARNING"

    else:
        fleet_status = "HEALTHY"

    severity_percentages = {
        severity: round(
            (count / total_machines) * 100,
            2,
        )
        for severity, count in severity_counts.items()
    }

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "fleet_overview": {
            "total_machines": total_machines,
            "fleet_status": fleet_status,
            "fleet_health_percentage": round(
                fleet_health_percentage,
                2,
            ),
            "machines_requiring_attention": (
                machines_requiring_attention
            ),
            "attention_percentage": round(
                (
                    machines_requiring_attention
                    / total_machines
                ) * 100,
                2,
            ),
        },

        "severity_distribution": {
            "counts": dict(severity_counts),
            "percentages": severity_percentages,
        },

        "diagnosis_distribution": dict(diagnosis_counts),

        "fault_summary": {
            "total_detected_faults": total_detected_faults,
            "total_supporting_alerts": total_supporting_alerts,
        },

        "maintenance_overview": {
            "critical_machines": critical_machines,
            "high_risk_machines": high_risk_machines,
            "warning_machines": warning_machines,
            "sensor_warning_machines": sensor_warning_machines,
        },
    }

    return report
