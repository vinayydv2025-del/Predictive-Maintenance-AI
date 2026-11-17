
"""
FLEET ANALYTICS ENGINE

Combines:
1. Fleet health aggregation
2. Machine risk ranking
3. Fault frequency analysis
4. Maintenance action distribution
5. Sensor-quality monitoring

This module does not train or run ML models.
It aggregates existing diagnostic-engine outputs.
"""

from collections import Counter
from datetime import datetime, timezone

from .fleet_health import calculate_fleet_health
from .risk_ranking import rank_machines_by_risk


def calculate_fault_frequency(machine_results):
    """Count how frequently each fault appears across the fleet."""

    fault_counts = Counter()

    for result in machine_results:
        final = result["final_diagnosis"]

        for fault in final.get("detected_faults", []):
            fault_counts[fault] += 1

    return dict(
        sorted(
            fault_counts.items(),
            key=lambda item: (-item[1], item[0])
        )
    )


def calculate_maintenance_distribution(machine_results):
    """Count machines by their recommended maintenance action."""

    action_counts = Counter()

    for result in machine_results:
        final = result["final_diagnosis"]

        action = final.get(
            "maintenance_action",
            "NOT_PROVIDED"
        )

        action_counts[action] += 1

    return dict(sorted(action_counts.items()))


def get_sensor_quality_summary(machine_results):
    """Summarize machines with explicit sensor-quality warnings."""

    warning_machines = []
    not_assessed_machines = []

    for result in machine_results:
        final = result["final_diagnosis"]

        sensor_quality = final.get("sensor_quality", {})

        if not isinstance(sensor_quality, dict):
            sensor_quality = {}

        status = sensor_quality.get(
            "status",
            "NOT_ASSESSED"
        )

        machine_id = result["machine_id"]

        if status == "WARNING":
            warning_machines.append(machine_id)

        elif status == "NOT_ASSESSED":
            not_assessed_machines.append(machine_id)

    return {
        "machines_with_sensor_warnings": warning_machines,
        "sensor_warning_count": len(warning_machines),
        "machines_not_assessed": not_assessed_machines,
        "not_assessed_count": len(not_assessed_machines),
    }


def generate_fleet_analytics(machine_results):
    """
    Generate a consolidated fleet intelligence report.

    Input:
        List of machine result dictionaries containing:
        machine_id and final_diagnosis.

    Output:
        Fleet health, risk ranking, fault frequency,
        maintenance distribution and sensor-quality summary.
    """

    if not isinstance(machine_results, list):
        raise TypeError("machine_results must be a list.")

    if not machine_results:
        raise ValueError("At least one machine result is required.")

    # Reuse the validation and aggregation in existing modules.
    fleet_health = calculate_fleet_health(machine_results)

    ranked_machines = rank_machines_by_risk(machine_results)

    fault_frequency = calculate_fault_frequency(machine_results)

    maintenance_distribution = (
        calculate_maintenance_distribution(machine_results)
    )

    sensor_quality_summary = (
        get_sensor_quality_summary(machine_results)
    )

    # Identify the highest-priority machine.
    highest_priority_machine = ranked_machines[0]

    # Identify machines requiring immediate attention.
    critical_machines = [
        machine
        for machine in ranked_machines
        if machine["severity"] == "CRITICAL"
    ]

    high_risk_machines = [
        machine
        for machine in ranked_machines
        if machine["severity"] == "HIGH"
    ]

    report = {
        "report_generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "fleet_health": fleet_health,

        "risk_ranking": ranked_machines,

        "highest_priority_machine": highest_priority_machine,

        "critical_machine_count": len(critical_machines),

        "high_risk_machine_count": len(high_risk_machines),

        "fault_frequency": fault_frequency,

        "maintenance_action_distribution": (
            maintenance_distribution
        ),

        "sensor_quality_summary": sensor_quality_summary,

        "critical_machines": critical_machines,

        "high_risk_machines": high_risk_machines,

        "analytics_metadata": {
            "total_machines_analyzed": len(machine_results),
            "ranking_method": "RULE_BASED_SEVERITY_AND_EVIDENCE",
            "ml_models_executed": False,
            "risk_score_is_probability": False,
            "data_source": "EXISTING_DIAGNOSTIC_ENGINE_OUTPUTS",
        },
    }

    return report
