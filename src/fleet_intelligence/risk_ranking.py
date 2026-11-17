
"""
FLEET MACHINE RISK RANKING

Ranks machines using existing diagnostic-engine outputs.

This is a transparent rule-based prioritization system,
not a trained risk prediction model.
"""

SEVERITY_PRIORITY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "WARNING": 2,
    "NORMAL": 1,
}

RELIABILITY_PRIORITY = {
    "COMPROMISED": 2,
    "ACTIONABLE_WITH_INVESTIGATION": 1,
    "NO_ALERT": 0,
    "NOT_PROVIDED": 0,
}


def rank_machines_by_risk(machine_results):
    """
    Rank machines by diagnostic urgency.

    Ranking order:
    1. Higher severity
    2. Compromised diagnostic reliability
    3. More detected faults
    4. More supporting alerts

    A compromised sensor condition is a secondary
    prioritization factor. It does not change severity.
    """

    if not isinstance(machine_results, list):
        raise TypeError("machine_results must be a list.")

    if not machine_results:
        raise ValueError("At least one machine result is required.")

    ranked = []
    seen_ids = set()

    for result in machine_results:

        if not isinstance(result, dict):
            raise TypeError(
                "Each machine result must be a dictionary."
            )

        machine_id = result.get("machine_id")

        if not machine_id:
            raise ValueError(
                "Every machine result needs machine_id."
            )

        if machine_id in seen_ids:
            raise ValueError(
                f"Duplicate machine ID: {machine_id}"
            )

        seen_ids.add(machine_id)

        final = result.get("final_diagnosis")

        if not isinstance(final, dict):
            raise ValueError(
                f"Machine {machine_id} has no valid final_diagnosis."
            )

        severity = final.get("severity")

        if severity not in SEVERITY_PRIORITY:
            raise ValueError(
                f"Unsupported severity for {machine_id}: {severity}"
            )

        reliability = final.get(
            "diagnostic_reliability",
            "NOT_PROVIDED"
        )

        if reliability not in RELIABILITY_PRIORITY:
            reliability = "NOT_PROVIDED"

        detected_faults = final.get(
            "detected_faults",
            []
        )

        if not isinstance(detected_faults, list):
            raise ValueError(
                f"Detected faults for {machine_id} must be a list."
            )

        supporting_alert_count = final.get(
            "supporting_alert_count",
            0
        )

        if (
            not isinstance(supporting_alert_count, int)
            or isinstance(supporting_alert_count, bool)
            or supporting_alert_count < 0
        ):
            raise ValueError(
                f"Invalid supporting alert count for {machine_id}."
            )

        maintenance_action = final.get(
            "maintenance_action",
            "NOT_PROVIDED"
        )

        sensor_quality = final.get(
            "sensor_quality",
            {}
        )

        if isinstance(sensor_quality, dict):
            sensor_status = sensor_quality.get(
                "status",
                "NOT_ASSESSED"
            )
        else:
            sensor_status = "NOT_ASSESSED"

        # Transparent ranking score.
        # This is NOT a probability or calibrated risk score.
        ranking_score = (
            SEVERITY_PRIORITY[severity] * 100
            + RELIABILITY_PRIORITY[reliability] * 10
            + len(detected_faults) * 2
            + min(supporting_alert_count, 9)
        )

        machine_summary = {
            "machine_id": machine_id,
            "severity": severity,
            "diagnosis": final.get(
                "diagnosis",
                "UNKNOWN"
            ),
            "diagnostic_reliability": reliability,
            "maintenance_action": maintenance_action,
            "detected_faults": detected_faults,
            "detected_fault_count": len(detected_faults),
            "supporting_alert_count": supporting_alert_count,
            "sensor_quality": sensor_status,
            "ranking_score": ranking_score,
        }

        ranked.append(machine_summary)

    # Highest-priority machines appear first.
    # Machine ID provides a deterministic final tie-breaker.
    ranked.sort(
        key=lambda item: (
            -SEVERITY_PRIORITY[item["severity"]],
            -RELIABILITY_PRIORITY[
                item["diagnostic_reliability"]
            ],
            -item["detected_fault_count"],
            -item["supporting_alert_count"],
            item["machine_id"],
        )
    )

    # Assign sequential risk ranks.
    for position, machine in enumerate(ranked, start=1):
        machine["risk_rank"] = position

    return ranked
