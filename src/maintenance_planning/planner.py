
"""
INDUSTRIAL MAINTENANCE PLANNING ENGINE

Consumes scenario-driven fleet diagnostics and creates:
1. Maintenance work orders
2. Rule-based maintenance priorities
3. Technician-capacity-aware schedules
4. Synthetic cost and downtime estimates
5. Spare-parts forecasts

All costs, durations and resource capacities are configurable
simulation assumptions, not real industrial estimates.
"""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "reports" / "scenario_fleet_intelligence.json"
OUTPUT_JSON = ROOT / "reports" / "maintenance_plan.json"
OUTPUT_CSV = ROOT / "reports" / "maintenance_work_orders.csv"
PARTS_CSV = ROOT / "reports" / "spare_parts_forecast.csv"


# ---------------------------------------------------------
# CONFIGURABLE SIMULATION ASSUMPTIONS
# ---------------------------------------------------------

TECHNICIANS_PER_DAY = 2
TECHNICIAN_HOURS_PER_DAY = 8

# Synthetic labor rate in simulation cost units per hour.
LABOR_RATE = 60.0

SENSOR_VERIFICATION = {
    "duration_hours": 1.5,
    "parts": {
        "sensor_inspection_kit": 1,
    },
    "parts_cost": 25.0,
}

FAULT_CATALOG = {
    "OVERHEATING": {
        "duration_hours": 2.5,
        "parts": {"cooling_filter_kit": 1},
        "parts_cost": 120.0,
        "action": "Inspect cooling system and replace the filter kit",
    },
    "OVERLOAD": {
        "duration_hours": 3.5,
        "parts": {"drive_bearing_kit": 1},
        "parts_cost": 260.0,
        "action": "Inspect load conditions and drive-bearing assembly",
    },
    "SPEED_ANOMALY": {
        "duration_hours": 2.0,
        "parts": {"speed_sensor": 1},
        "parts_cost": 90.0,
        "action": "Inspect rotational-speed sensing and drive control",
    },
}

SEVERITY_PRIORITY = {
    "CRITICAL": 400,
    "HIGH": 300,
    "WARNING": 200,
    "NORMAL": 100,
}


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_scenario_report(report):
    if not isinstance(report, dict):
        raise TypeError("Scenario report must be a dictionary.")

    if "scenario_results" not in report:
        raise ValueError("Scenario report has no scenario_results.")

    if "fleet_analytics" not in report:
        raise ValueError("Scenario report has no fleet_analytics.")

    results = report["scenario_results"]

    if not isinstance(results, list) or not results:
        raise ValueError("scenario_results must be a non-empty list.")

    machine_ids = []

    for result in results:
        if not isinstance(result, dict):
            raise TypeError("Each scenario result must be a dictionary.")

        machine_id = result.get("machine_id")

        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError("A scenario result has an invalid machine_id.")

        if not isinstance(result.get("detected_faults"), list):
            raise ValueError(
                f"Invalid detected_faults for {machine_id}."
            )

        if result.get("severity") not in SEVERITY_PRIORITY:
            raise ValueError(
                f"Invalid severity for {machine_id}."
            )

        machine_ids.append(machine_id)

    if len(machine_ids) != len(set(machine_ids)):
        raise ValueError("Duplicate machine IDs in scenario results.")

    return True


# ---------------------------------------------------------
# PRIORITY AND WORK ORDER GENERATION
# ---------------------------------------------------------

def get_ranking_metadata(report):
    ranking = report["fleet_analytics"].get("risk_ranking", [])

    return {
        item["machine_id"]: item
        for item in ranking
    }


def calculate_priority(result, ranking):
    """
    Use the existing rule-based ranking score where available.

    This is a scheduling priority, not a failure probability.
    """

    machine_id = result["machine_id"]
    ranked = ranking.get(machine_id, {})

    score = ranked.get("ranking_score")

    if score is None:
        severity_score = SEVERITY_PRIORITY[result["severity"]]
        fault_score = len(result["detected_faults"]) * 2
        alert_score = min(result.get("supporting_alert_count", 0), 9)
        score = severity_score + fault_score + alert_score

    return int(score)


def is_sensor_compromised(result):
    return (
        result.get("diagnostic_reliability") == "COMPROMISED"
        or result.get("maintenance_action")
        == "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"
    )


def create_work_order(result, ranking):
    machine_id = result["machine_id"]
    faults = result["detected_faults"]

    priority_score = calculate_priority(result, ranking)
    compromised = is_sensor_compromised(result)

    if compromised:
        # Do not authorize equipment repair until sensor readings
        # have been independently checked.
        duration = SENSOR_VERIFICATION["duration_hours"]
        parts = dict(SENSOR_VERIFICATION["parts"])
        parts_cost = SENSOR_VERIFICATION["parts_cost"]

        return {
            "machine_id": machine_id,
            "severity": result["severity"],
            "diagnosis": result["diagnosis"],
            "detected_faults": faults,
            "priority_score": priority_score,
            "work_order_type": "SENSOR_VERIFICATION",
            "recommended_action": (
                "Verify sensor integrity before confirming equipment faults"
            ),
            "work_order_status": "READY_FOR_SENSOR_INSPECTION",
            "equipment_repair_authorized": False,
            "provisional_equipment_action": (
                "Reassess detected faults after sensor verification"
                if faults else None
            ),
            "estimated_duration_hours": duration,
            "estimated_labor_cost": round(duration * LABOR_RATE, 2),
            "estimated_parts_cost": parts_cost,
            "estimated_total_cost": round(
                duration * LABOR_RATE + parts_cost, 2
            ),
            "estimated_downtime_hours": duration,
            "required_parts": parts,
            "diagnostic_reliability": "COMPROMISED",
        }

    if not faults:
        return None

    duration = 0.0
    parts_cost = 0.0
    parts = Counter()
    actions = []

    for fault in faults:
        if fault not in FAULT_CATALOG:
            raise ValueError(
                f"No maintenance catalog entry for fault: {fault}"
            )

        item = FAULT_CATALOG[fault]

        duration += item["duration_hours"]
        parts_cost += item["parts_cost"]
        parts.update(item["parts"])
        actions.append(item["action"])

    labor_cost = duration * LABOR_RATE

    return {
        "machine_id": machine_id,
        "severity": result["severity"],
        "diagnosis": result["diagnosis"],
        "detected_faults": faults,
        "priority_score": priority_score,
        "work_order_type": "EQUIPMENT_INSPECTION_AND_REPAIR",
        "recommended_action": " | ".join(actions),
        "work_order_status": "READY_FOR_MAINTENANCE",
        "equipment_repair_authorized": True,
        "provisional_equipment_action": None,
        "estimated_duration_hours": round(duration, 2),
        "estimated_labor_cost": round(labor_cost, 2),
        "estimated_parts_cost": round(parts_cost, 2),
        "estimated_total_cost": round(
            labor_cost + parts_cost, 2
        ),
        "estimated_downtime_hours": round(duration, 2),
        "required_parts": dict(parts),
        "diagnostic_reliability": result.get(
            "diagnostic_reliability",
            "NOT_PROVIDED",
        ),
    }


# ---------------------------------------------------------
# TECHNICIAN-CAPACITY-AWARE SCHEDULING
# ---------------------------------------------------------

def schedule_work_orders(work_orders):
    """
    Assign work orders to technicians using a priority-first,
    first-fit scheduling strategy.

    Each technician has a configurable daily hour capacity.
    A work order is assigned as one indivisible task.
    """

    ordered = sorted(
        work_orders,
        key=lambda item: (
            -item["priority_score"],
            item["machine_id"],
        ),
    )

    days = []

    for work_order in ordered:
        duration = work_order["estimated_duration_hours"]

        if duration > TECHNICIAN_HOURS_PER_DAY:
            raise ValueError(
                f"Work order for {work_order['machine_id']} exceeds "
                "the daily technician capacity."
            )

        assigned = False

        for day_index, day in enumerate(days):
            for technician_index, used_hours in enumerate(day):
                if used_hours + duration <= TECHNICIAN_HOURS_PER_DAY:
                    day[technician_index] += duration

                    work_order["scheduled_day"] = day_index + 1
                    work_order["assigned_technician"] = (
                        f"TECHNICIAN-{technician_index + 1:02d}"
                    )

                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            days.append([0.0] * TECHNICIANS_PER_DAY)

            days[-1][0] = duration

            work_order["scheduled_day"] = len(days)
            work_order["assigned_technician"] = "TECHNICIAN-01"

    return ordered, days


# ---------------------------------------------------------
# SPARE-PARTS FORECAST
# ---------------------------------------------------------

def build_parts_forecast(work_orders):
    required = Counter()

    for work_order in work_orders:
        required.update(work_order["required_parts"])

    return [
        {
            "part_name": part,
            "required_quantity": quantity,
            "planning_basis": "SYNTHETIC_MAINTENANCE_RULES",
        }
        for part, quantity in sorted(required.items())
    ]


# ---------------------------------------------------------
# COMPLETE PLAN
# ---------------------------------------------------------

def build_maintenance_plan(report):
    validate_scenario_report(report)

    ranking = get_ranking_metadata(report)

    work_orders = []

    for result in report["scenario_results"]:
        work_order = create_work_order(result, ranking)

        if work_order is not None:
            work_orders.append(work_order)

    scheduled, daily_capacity = schedule_work_orders(work_orders)

    parts_forecast = build_parts_forecast(scheduled)

    total_cost = sum(
        item["estimated_total_cost"]
        for item in scheduled
    )

    total_downtime = sum(
        item["estimated_downtime_hours"]
        for item in scheduled
    )

    schedule_days = len(daily_capacity)

    checks = {
        "all_work_orders_have_machine_ids": all(
            bool(item["machine_id"]) for item in scheduled
        ),
        "all_work_orders_have_valid_schedules": all(
            item.get("scheduled_day", 0) > 0
            and bool(item.get("assigned_technician"))
            for item in scheduled
        ),
        "technician_daily_capacity_respected": all(
            hours <= TECHNICIAN_HOURS_PER_DAY
            for day in daily_capacity
            for hours in day
        ),
        "sensor_compromised_machines_not_authorized_for_repair": all(
            not item["equipment_repair_authorized"]
            for item in scheduled
            if item["diagnostic_reliability"] == "COMPROMISED"
        ),
        "spare_parts_forecast_generated": isinstance(
            parts_forecast, list
        ),
        "cost_estimates_nonnegative": all(
            item["estimated_total_cost"] >= 0
            for item in scheduled
        ),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "INDUSTRIAL_MAINTENANCE_PLANNING_ENGINE",
        "data_source": "SYNTHETIC_SCENARIO_FLEET_REPORT",
        "real_world_cost_validation": False,
        "planning_assumptions": {
            "technicians_per_day": TECHNICIANS_PER_DAY,
            "technician_hours_per_day": TECHNICIAN_HOURS_PER_DAY,
            "labor_rate_simulation_units_per_hour": LABOR_RATE,
            "scheduling_method": "PRIORITY_FIRST_FIRST_FIT",
            "work_orders_are_indivisible": True,
        },
        "fleet_maintenance_summary": {
            "machines_analyzed": len(report["scenario_results"]),
            "work_orders_created": len(scheduled),
            "machines_without_work_orders": (
                len(report["scenario_results"]) - len(scheduled)
            ),
            "schedule_days_required": schedule_days,
            "total_estimated_downtime_hours": round(
                total_downtime, 2
            ),
            "total_estimated_cost_simulation_units": round(
                total_cost, 2
            ),
        },
        "work_orders": scheduled,
        "spare_parts_forecast": parts_forecast,
        "daily_technician_capacity": [
            {
                "day": index + 1,
                "technician_hours": [
                    round(hours, 2) for hours in day
                ],
                "total_hours": round(sum(day), 2),
            }
            for index, day in enumerate(daily_capacity)
        ],
        "validation": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_passed": all(checks.values()),
            "checks": checks,
        },
    }


# ---------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------

def export_csv(plan):
    fields = [
        "machine_id",
        "severity",
        "diagnosis",
        "work_order_type",
        "work_order_status",
        "priority_score",
        "scheduled_day",
        "assigned_technician",
        "estimated_duration_hours",
        "estimated_labor_cost",
        "estimated_parts_cost",
        "estimated_total_cost",
        "estimated_downtime_hours",
        "equipment_repair_authorized",
        "provisional_equipment_action",
        "diagnostic_reliability",
        "recommended_action",
        "detected_faults",
        "required_parts",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for item in plan["work_orders"]:
            row = item.copy()
            row["detected_faults"] = ";".join(
                row["detected_faults"]
            )
            row["required_parts"] = json.dumps(
                row["required_parts"]
            )
            writer.writerow(row)

    with PARTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "part_name",
            "required_quantity",
            "planning_basis",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plan["spare_parts_forecast"])


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Scenario report not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(encoding="utf-8") as file:
        report = json.load(file)

    plan = build_maintenance_plan(report)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(plan, file, indent=4)

    export_csv(plan)

    summary = plan["fleet_maintenance_summary"]

    print("\n" + "=" * 80)
    print("INDUSTRIAL MAINTENANCE PLANNING ENGINE")
    print("=" * 80)

    print("\n===== MAINTENANCE SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== WORK ORDERS =====")

    for item in plan["work_orders"]:
        print(
            f"Day {item['scheduled_day']} | "
            f"{item['assigned_technician']} | "
            f"{item['machine_id']} | "
            f"{item['work_order_type']} | "
            f"Priority={item['priority_score']} | "
            f"Cost={item['estimated_total_cost']}"
        )

    print("\n===== SPARE PARTS FORECAST =====")

    for item in plan["spare_parts_forecast"]:
        print(
            f"{item['part_name']}: "
            f"{item['required_quantity']}"
        )

    print("\n===== VALIDATION =====")

    for name, passed in plan["validation"]["checks"].items():
        print(
            f"{name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print(
        "\nValidation:",
        f"{plan['validation']['passed']}/"
        f"{plan['validation']['total']}",
    )

    print("\n===== OUTPUT FILES =====")
    print("JSON:", OUTPUT_JSON)
    print("Work orders CSV:", OUTPUT_CSV)
    print("Parts forecast CSV:", PARTS_CSV)

    if not plan["validation"]["all_passed"]:
        raise RuntimeError("Maintenance planning validation failed.")

    print("\nMAINTENANCE PLANNING COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
