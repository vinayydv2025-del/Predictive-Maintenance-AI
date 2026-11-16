
"""
MAINTENANCE SCHEDULING STRATEGY COMPARISON

Compares four scheduling policies using identical work orders,
technician capacity and processing durations.

Strategies:
1. PRIORITY_FIRST
2. SHORTEST_PROCESSING_TIME
3. SEVERITY_FIRST
4. FIFO

Metrics:
- Makespan
- Critical machine waiting time
- Average waiting time
- Technician utilization
- Total simulated maintenance cost
- Total estimated downtime

This is a deterministic scheduling simulation, not a
validated real-world optimization model.
"""

import json
from copy import deepcopy
from pathlib import Path

from src.maintenance_planning.planner import (
    TECHNICIANS_PER_DAY,
    TECHNICIAN_HOURS_PER_DAY,
    INPUT_FILE,
    build_maintenance_plan,
)


ROOT = Path(__file__).resolve().parents[2]

PLAN_FILE = ROOT / "reports" / "maintenance_plan.json"
OUTPUT_FILE = ROOT / "reports" / "maintenance_optimization.json"
OUTPUT_CSV = ROOT / "reports" / "maintenance_strategy_comparison.csv"

SEVERITY_ORDER = {
    "CRITICAL": 4,
    "HIGH": 3,
    "WARNING": 2,
    "NORMAL": 1,
}


def load_plan():
    if PLAN_FILE.exists():
        with PLAN_FILE.open(encoding="utf-8") as file:
            plan = json.load(file)

        if not plan.get("validation", {}).get("all_passed"):
            raise ValueError("Existing maintenance plan failed validation.")

        return plan

    with INPUT_FILE.open(encoding="utf-8") as file:
        report = json.load(file)

    return build_maintenance_plan(report)


def prepare_jobs(plan):
    jobs = []

    for index, item in enumerate(plan["work_orders"]):
        job = deepcopy(item)

        job["original_order"] = index
        job["duration"] = float(
            job["estimated_duration_hours"]
        )

        if job["duration"] <= 0:
            raise ValueError("Work order duration must be positive.")

        jobs.append(job)

    return jobs


def order_jobs(jobs, strategy):
    if strategy == "PRIORITY_FIRST":
        return sorted(
            jobs,
            key=lambda job: (
                -job["priority_score"],
                job["original_order"],
            ),
        )

    if strategy == "SHORTEST_PROCESSING_TIME":
        return sorted(
            jobs,
            key=lambda job: (
                job["duration"],
                -job["priority_score"],
                job["original_order"],
            ),
        )

    if strategy == "SEVERITY_FIRST":
        return sorted(
            jobs,
            key=lambda job: (
                -SEVERITY_ORDER[job["severity"]],
                -job["priority_score"],
                job["original_order"],
            ),
        )

    if strategy == "FIFO":
        return sorted(
            jobs,
            key=lambda job: job["original_order"],
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def simulate_strategy(plan, strategy):
    jobs = order_jobs(
        prepare_jobs(plan),
        strategy,
    )

    technicians = [
        {
            "technician_id": f"TECHNICIAN-{index + 1:02d}",
            "available_at": 0.0,
            "busy_hours": 0.0,
            "jobs": [],
        }
        for index in range(TECHNICIANS_PER_DAY)
    ]

    scheduled_jobs = []

    for job in jobs:
        # Select the earliest available technician.
        # Ties are resolved by technician ID.
        technician = min(
            technicians,
            key=lambda item: (
                item["available_at"],
                item["technician_id"],
            ),
        )

        start = technician["available_at"]
        finish = start + job["duration"]

        job["start_hour"] = round(start, 2)
        job["finish_hour"] = round(finish, 2)
        job["waiting_hours"] = round(start, 2)
        job["completion_hours"] = round(finish, 2)
        job["assigned_technician"] = technician["technician_id"]

        technician["available_at"] = finish
        technician["busy_hours"] += job["duration"]
        technician["jobs"].append(job["machine_id"])

        scheduled_jobs.append(job)

    makespan = max(
        (item["available_at"] for item in technicians),
        default=0.0,
    )

    total_capacity = (
        len(technicians) * makespan
    )

    total_busy = sum(
        item["busy_hours"]
        for item in technicians
    )

    utilization = (
        total_busy / total_capacity * 100
        if total_capacity else 0.0
    )

    critical_jobs = [
        job for job in scheduled_jobs
        if job["severity"] == "CRITICAL"
    ]

    critical_wait = (
        sum(job["waiting_hours"] for job in critical_jobs)
        / len(critical_jobs)
        if critical_jobs else 0.0
    )

    average_wait = (
        sum(job["waiting_hours"] for job in scheduled_jobs)
        / len(scheduled_jobs)
        if scheduled_jobs else 0.0
    )

    total_cost = sum(
        job["estimated_total_cost"]
        for job in scheduled_jobs
    )

    total_downtime = sum(
        job["estimated_downtime_hours"]
        for job in scheduled_jobs
    )

    return {
        "strategy": strategy,
        "makespan_hours": round(makespan, 2),
        "average_waiting_hours": round(average_wait, 2),
        "critical_machine_average_waiting_hours": round(
            critical_wait, 2
        ),
        "technician_utilization_percentage": round(
            utilization, 2
        ),
        "total_estimated_downtime_hours": round(
            total_downtime, 2
        ),
        "total_estimated_cost_simulation_units": round(
            total_cost, 2
        ),
        "technician_summary": [
            {
                "technician_id": item["technician_id"],
                "busy_hours": round(item["busy_hours"], 2),
                "available_at": round(item["available_at"], 2),
                "assigned_machines": item["jobs"],
            }
            for item in technicians
        ],
        "scheduled_work_orders": scheduled_jobs,
    }


def compare_strategies(plan):
    strategies = [
        "PRIORITY_FIRST",
        "SHORTEST_PROCESSING_TIME",
        "SEVERITY_FIRST",
        "FIFO",
    ]

    results = [
        simulate_strategy(plan, strategy)
        for strategy in strategies
    ]

    # Each strategy processes the same work orders.
    baseline = results[0]

    for result in results:
        assert len(result["scheduled_work_orders"]) == len(
            baseline["scheduled_work_orders"]
        )

        assert result["total_estimated_cost_simulation_units"] == (
            baseline["total_estimated_cost_simulation_units"]
        )

        assert result["total_estimated_downtime_hours"] == (
            baseline["total_estimated_downtime_hours"]
        )

    # Select the strategy with the lowest critical-machine wait.
    # Use makespan and average wait as tie-breakers.
    recommended = min(
        results,
        key=lambda item: (
            item["critical_machine_average_waiting_hours"],
            item["makespan_hours"],
            item["average_waiting_hours"],
        ),
    )

    return {
        "comparison_method": (
            "IDENTICAL_JOBS_AND_RESOURCES_ACROSS_ALL_STRATEGIES"
        ),
        "technicians": TECHNICIANS_PER_DAY,
        "technician_hours_per_day": TECHNICIAN_HOURS_PER_DAY,
        "time_unit": "SIMULATED_WORKING_HOURS",
        "costs_are_synthetic": True,
        "results": results,
        "recommended_strategy": recommended["strategy"],
        "recommendation_basis": (
            "Lowest average waiting time for critical machines, "
            "then lowest makespan, then lowest average waiting time."
        ),
    }


def validate_comparison(comparison):
    results = comparison["results"]

    checks = {
        "all_four_strategies_evaluated": len(results) == 4,
        "all_strategies_have_same_work_order_count": (
            len({
                len(item["scheduled_work_orders"])
                for item in results
            }) == 1
        ),
        "costs_are_equal_across_strategies": (
            len({
                item["total_estimated_cost_simulation_units"]
                for item in results
            }) == 1
        ),
        "downtime_is_equal_across_strategies": (
            len({
                item["total_estimated_downtime_hours"]
                for item in results
            }) == 1
        ),
        "all_makespans_are_positive": all(
            item["makespan_hours"] > 0
            for item in results
        ),
        "utilization_is_within_valid_range": all(
            0 <= item["technician_utilization_percentage"] <= 100
            for item in results
        ),
        "all_jobs_have_nonnegative_wait": all(
            job["waiting_hours"] >= 0
            for item in results
            for job in item["scheduled_work_orders"]
        ),
        "recommendation_matches_evaluated_strategy": (
            comparison["recommended_strategy"]
            in [item["strategy"] for item in results]
        ),
    }

    return {
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "checks": checks,
    }


def export_comparison(comparison):
    import csv

    fields = [
        "strategy",
        "makespan_hours",
        "average_waiting_hours",
        "critical_machine_average_waiting_hours",
        "technician_utilization_percentage",
        "total_estimated_downtime_hours",
        "total_estimated_cost_simulation_units",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in comparison["results"]:
            writer.writerow({
                field: result[field]
                for field in fields
            })


def main():
    plan = load_plan()

    comparison = compare_strategies(plan)
    validation = validate_comparison(comparison)

    output = {
        "generated_at": plan["generated_at"],
        "platform": "MAINTENANCE_OPTIMIZATION_ENGINE",
        "comparison": comparison,
        "validation": validation,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=4)

    export_comparison(comparison)

    print("\n" + "=" * 80)
    print("MAINTENANCE SCHEDULING STRATEGY COMPARISON")
    print("=" * 80)

    for result in comparison["results"]:
        print(f"\n===== {result['strategy']} =====")
        print(
            "Makespan:",
            result["makespan_hours"],
            "hours",
        )
        print(
            "Average waiting:",
            result["average_waiting_hours"],
            "hours",
        )
        print(
            "Critical machine waiting:",
            result["critical_machine_average_waiting_hours"],
            "hours",
        )
        print(
            "Technician utilization:",
            result["technician_utilization_percentage"],
            "%",
        )
        print(
            "Estimated cost:",
            result["total_estimated_cost_simulation_units"],
        )

    print("\n===== RECOMMENDATION =====")
    print(comparison["recommended_strategy"])
    print(comparison["recommendation_basis"])

    print("\n===== VALIDATION =====")

    for name, passed in validation["checks"].items():
        print(
            f"{name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print(
        f"\nValidation: {validation['passed']}/"
        f"{validation['total']}"
    )

    print("\nOutput:", OUTPUT_FILE)
    print("CSV:", OUTPUT_CSV)

    if not validation["all_passed"]:
        raise RuntimeError("Optimization validation failed.")

    print("\nMAINTENANCE OPTIMIZATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
