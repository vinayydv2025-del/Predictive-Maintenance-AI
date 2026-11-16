
"""
OR-TOOLS INDUSTRIAL MAINTENANCE OPTIMIZATION ENGINE

Uses Google OR-Tools CP-SAT to optimize technician assignments
and work-order scheduling.

Compares:
1. PRIORITY_FIRST
2. SHORTEST_PROCESSING_TIME
3. SEVERITY_FIRST
4. FIFO
5. OR_TOOLS_OPTIMAL

Evaluation:
- Critical-machine completion time
- Makespan
- Average waiting time
- Technician utilization
- Estimated cost
- Estimated downtime

Experiments use reproducible synthetic variations of the
existing maintenance work orders.

Important:
This is a scheduling simulation, not a validated industrial
maintenance optimization system.
"""

import csv
import json
import random
from copy import deepcopy
from pathlib import Path

from ortools.sat.python import cp_model

from src.maintenance_planning.optimization import load_plan
from src.maintenance_planning.planner import (
    TECHNICIANS_PER_DAY,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_JSON = ROOT / "reports" / "ortools_optimization.json"
OUTPUT_CSV = ROOT / "reports" / "ortools_scenario_comparison.csv"

TIME_SCALE = 2
# Two units represent one working hour.
# This supports half-hour scheduling precision.

SOLVER_TIME_LIMIT_SECONDS = 2.0
RANDOM_SEED = 42

STRATEGIES = [
    "PRIORITY_FIRST",
    "SHORTEST_PROCESSING_TIME",
    "SEVERITY_FIRST",
    "FIFO",
]

SEVERITY_ORDER = {
    "CRITICAL": 4,
    "HIGH": 3,
    "WARNING": 2,
    "NORMAL": 1,
}


# ---------------------------------------------------------
# PREPARE WORK ORDERS
# ---------------------------------------------------------

def prepare_jobs(work_orders):
    jobs = []

    for index, item in enumerate(work_orders):
        job = deepcopy(item)

        duration = float(
            job["estimated_duration_hours"]
        )

        if duration <= 0:
            raise ValueError(
                f"Invalid duration for {job['machine_id']}"
            )

        scaled_duration = round(duration * TIME_SCALE)

        if scaled_duration <= 0:
            raise ValueError("Scaled duration must be positive.")

        job["duration_units"] = scaled_duration
        job["duration"] = scaled_duration / TIME_SCALE
        job["original_order"] = index

        jobs.append(job)

    if not jobs:
        raise ValueError("No work orders supplied.")

    return jobs


# ---------------------------------------------------------
# HEURISTIC SCHEDULING BASELINES
# ---------------------------------------------------------

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
                job["duration_units"],
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

    raise ValueError(f"Unknown heuristic: {strategy}")


def calculate_metrics(scheduled_jobs, technician_data, strategy):
    makespan = max(
        (
            technician["available_units"]
            for technician in technician_data
        ),
        default=0,
    )

    total_busy = sum(
        technician["busy_units"]
        for technician in technician_data
    )

    total_capacity = (
        len(technician_data) * makespan
    )

    utilization = (
        total_busy / total_capacity * 100
        if total_capacity > 0 else 0.0
    )

    critical_jobs = [
        job for job in scheduled_jobs
        if job["severity"] == "CRITICAL"
    ]

    critical_average_completion = (
        sum(job["finish_units"] for job in critical_jobs)
        / len(critical_jobs)
        / TIME_SCALE
        if critical_jobs else 0.0
    )

    critical_average_wait = (
        sum(job["start_units"] for job in critical_jobs)
        / len(critical_jobs)
        / TIME_SCALE
        if critical_jobs else 0.0
    )

    average_wait = (
        sum(job["start_units"] for job in scheduled_jobs)
        / len(scheduled_jobs)
        / TIME_SCALE
    )

    average_completion = (
        sum(job["finish_units"] for job in scheduled_jobs)
        / len(scheduled_jobs)
        / TIME_SCALE
    )

    return {
        "strategy": strategy,
        "makespan_hours": round(
            makespan / TIME_SCALE, 2
        ),
        "average_waiting_hours": round(
            average_wait, 2
        ),
        "critical_machine_average_waiting_hours": round(
            critical_average_wait, 2
        ),
        "critical_machine_average_completion_hours": round(
            critical_average_completion, 2
        ),
        "average_completion_hours": round(
            average_completion, 2
        ),
        "technician_utilization_percentage": round(
            utilization, 2
        ),
        "total_estimated_cost_simulation_units": round(
            sum(
                job["estimated_total_cost"]
                for job in scheduled_jobs
            ),
            2,
        ),
        "total_estimated_downtime_hours": round(
            sum(
                job["estimated_downtime_hours"]
                for job in scheduled_jobs
            ),
            2,
        ),
        "scheduled_work_orders": scheduled_jobs,
        "technician_summary": [
            {
                "technician_id": technician["technician_id"],
                "busy_hours": round(
                    technician["busy_units"] / TIME_SCALE,
                    2,
                ),
                "assigned_machines": technician["machines"],
            }
            for technician in technician_data
        ],
    }


def simulate_heuristic(jobs, strategy):
    ordered = order_jobs(
        deepcopy(jobs),
        strategy,
    )

    technicians = [
        {
            "technician_id": f"TECHNICIAN-{index + 1:02d}",
            "available_units": 0,
            "busy_units": 0,
            "machines": [],
        }
        for index in range(TECHNICIANS_PER_DAY)
    ]

    scheduled = []

    for job in ordered:
        technician = min(
            technicians,
            key=lambda item: (
                item["available_units"],
                item["technician_id"],
            ),
        )

        start = technician["available_units"]
        finish = start + job["duration_units"]

        job["start_units"] = start
        job["finish_units"] = finish
        job["start_hour"] = round(start / TIME_SCALE, 2)
        job["finish_hour"] = round(finish / TIME_SCALE, 2)
        job["waiting_hours"] = job["start_hour"]
        job["assigned_technician"] = technician["technician_id"]

        technician["available_units"] = finish
        technician["busy_units"] += job["duration_units"]
        technician["machines"].append(job["machine_id"])

        scheduled.append(job)

    return calculate_metrics(
        scheduled,
        technicians,
        strategy,
    )


# ---------------------------------------------------------
# OR-TOOLS CP-SAT OPTIMIZER
# ---------------------------------------------------------

def select_warm_start(jobs):
    """
    Select the best feasible heuristic schedule using the same
    objective ordering as the CP-SAT optimizer:

    1. Lowest total critical-job completion time
    2. Lowest makespan
    3. Lowest total completion time

    Returns:
        best_strategy: Name of the selected heuristic.
        schedule_by_order: Schedule indexed by original_order.
    """

    candidates = []

    for strategy in STRATEGIES:
        result = simulate_heuristic(jobs, strategy)
        scheduled = result["scheduled_work_orders"]

        critical_completion = sum(
            job["finish_units"]
            for job in scheduled
            if job["severity"] == "CRITICAL"
        )

        makespan = max(
            (job["finish_units"] for job in scheduled),
            default=0,
        )

        total_completion = sum(
            job["finish_units"]
            for job in scheduled
        )

        objective_key = (
            critical_completion,
            makespan,
            total_completion,
        )

        candidates.append(
            (objective_key, strategy, scheduled)
        )

    _, best_strategy, best_schedule = min(
        candidates,
        key=lambda item: item[0],
    )

    schedule_by_order = {
        job["original_order"]: job
        for job in best_schedule
    }

    expected_orders = {
        job["original_order"]
        for job in jobs
    }

    if set(schedule_by_order) != expected_orders:
        raise ValueError(
            "Warm-start schedule does not cover every work order."
        )

    return best_strategy, schedule_by_order


def optimize_with_ortools(
    jobs,
    time_limit=SOLVER_TIME_LIMIT_SECONDS,
    use_warm_start=True,
):
    model = cp_model.CpModel()

    number_of_jobs = len(jobs)
    number_of_technicians = TECHNICIANS_PER_DAY

    horizon = sum(
        job["duration_units"]
        for job in jobs
    )

    starts = []
    ends = []
    technician_choices = []

    technician_intervals = [
        [] for _ in range(number_of_technicians)
    ]

    for job_index, job in enumerate(jobs):
        duration = job["duration_units"]

        start = model.NewIntVar(
            0,
            horizon,
            f"start_{job_index}",
        )

        end = model.NewIntVar(
            0,
            horizon,
            f"end_{job_index}",
        )

        model.Add(end == start + duration)

        choices = []

        for technician_index in range(number_of_technicians):
            selected = model.NewBoolVar(
                f"job_{job_index}_technician_{technician_index}"
            )

            interval = model.NewOptionalIntervalVar(
                start,
                duration,
                end,
                selected,
                f"interval_{job_index}_{technician_index}",
            )

            technician_intervals[technician_index].append(
                interval
            )

            choices.append(selected)

        model.AddExactlyOne(choices)

        starts.append(start)
        ends.append(end)
        technician_choices.append(choices)

    # A technician cannot work on overlapping maintenance jobs.
    for intervals in technician_intervals:
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(
        0,
        horizon,
        "makespan",
    )

    model.AddMaxEquality(
        makespan,
        ends,
    )

    critical_indices = [
        index
        for index, job in enumerate(jobs)
        if job["severity"] == "CRITICAL"
    ]

    critical_completion = sum(
        ends[index]
        for index in critical_indices
    )

    total_completion = sum(ends)

    # Lexicographic objective:
    # 1. Minimize critical-machine completion times.
    # 2. Minimize overall makespan.
    # 3. Minimize total completion time.

    lower_bound_weight = (
        number_of_jobs * horizon + 1
    )

    critical_weight = (
        horizon * lower_bound_weight
        + number_of_jobs * horizon
        + 1
    )

    objective = (
        critical_completion * critical_weight
        + makespan * lower_bound_weight
        + total_completion
    )

    model.Minimize(objective)

    # -----------------------------------------------------
    # WARM START: ADD THE BEST HEURISTIC SCHEDULE AS HINTS
    # -----------------------------------------------------

    warm_start_strategy = None

    if use_warm_start:
        (
            warm_start_strategy,
            warm_schedule,
        ) = select_warm_start(jobs)

        technician_ids = [
            f"TECHNICIAN-{index + 1:02d}"
            for index in range(number_of_technicians)
        ]

        for job_index, job in enumerate(jobs):
            original_order = job["original_order"]
            suggested = warm_schedule[original_order]

            suggested_technician = (
                suggested["assigned_technician"]
            )

            if suggested_technician not in technician_ids:
                raise ValueError(
                    "Invalid technician in warm-start schedule: "
                    f"{suggested_technician}"
                )

            model.AddHint(
                starts[job_index],
                int(suggested["start_units"]),
            )

            model.AddHint(
                ends[job_index],
                int(suggested["finish_units"]),
            )

            for technician_index, choice in enumerate(
                technician_choices[job_index]
            ):
                model.AddHint(
                    choice,
                    int(
                        technician_ids[technician_index]
                        == suggested_technician
                    ),
                )

        suggested_makespan = max(
            (
                job["finish_units"]
                for job in warm_schedule.values()
            ),
            default=0,
        )

        model.AddHint(
            makespan,
            int(suggested_makespan),
        )

    # -----------------------------------------------------
    # SOLVE
    # -----------------------------------------------------

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = RANDOM_SEED

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        raise RuntimeError(
            "OR-Tools failed to find a feasible schedule. "
            f"Solver status: {solver.StatusName(status)}"
        )

    technicians = [
        {
            "technician_id": f"TECHNICIAN-{index + 1:02d}",
            "available_units": 0,
            "busy_units": 0,
            "machines": [],
        }
        for index in range(number_of_technicians)
    ]

    scheduled = []

    for job_index, job in enumerate(jobs):
        selected_technician = None

        for technician_index, choice in enumerate(
            technician_choices[job_index]
        ):
            if solver.Value(choice):
                selected_technician = technician_index
                break

        if selected_technician is None:
            raise RuntimeError(
                f"No technician assigned to {job['machine_id']}"
            )

        start = solver.Value(starts[job_index])
        finish = solver.Value(ends[job_index])

        technician = technicians[selected_technician]

        technician["busy_units"] += job["duration_units"]

        technician["available_units"] = max(
            technician["available_units"],
            finish,
        )

        technician["machines"].append(job["machine_id"])

        job = deepcopy(job)

        job["start_units"] = start
        job["finish_units"] = finish
        job["start_hour"] = round(start / TIME_SCALE, 2)
        job["finish_hour"] = round(finish / TIME_SCALE, 2)
        job["waiting_hours"] = job["start_hour"]

        job["assigned_technician"] = (
            technician["technician_id"]
        )

        scheduled.append(job)

    scheduled.sort(
        key=lambda job: (
            job["start_units"],
            job["assigned_technician"],
            job["machine_id"],
        )
    )

    result = calculate_metrics(
        scheduled,
        technicians,
        "OR_TOOLS_OPTIMAL",
    )

    result["solver_status"] = solver.StatusName(status)
    result["solver_objective_value"] = solver.ObjectiveValue()
    result["solver_best_bound"] = solver.BestObjectiveBound()

    result["solver_wall_time_seconds"] = round(
        solver.WallTime(),
        4,
    )

    result["optimality_proven"] = (
        status == cp_model.OPTIMAL
    )

    result["warm_start_enabled"] = use_warm_start
    result["warm_start_hints_added"] = use_warm_start
    result["warm_start_strategy"] = warm_start_strategy

    return result



# ---------------------------------------------------------
# REPRODUCIBLE SCENARIO GENERATION
# ---------------------------------------------------------

def generate_scenarios(base_jobs, scenario_count=25, seed=42):
    if scenario_count < 1:
        raise ValueError("scenario_count must be at least 1.")

    rng = random.Random(seed)

    scenarios = [
        {
            "scenario_id": "SCENARIO-001",
            "scenario_type": "BASELINE",
            "jobs": deepcopy(base_jobs),
        }
    ]

    for scenario_index in range(1, scenario_count):
        jobs = deepcopy(base_jobs)

        for job in jobs:
            # Synthetic duration variation.
            # Durations remain in half-hour increments.
            multiplier = rng.uniform(0.75, 1.25)

            varied_duration = max(
                1,
                round(
                    job["duration_units"] * multiplier
                ),
            )

            job["duration_units"] = varied_duration
            job["duration"] = varied_duration / TIME_SCALE

            # Small priority variation without changing
            # the original severity or diagnosis.
            priority_adjustment = rng.randint(-10, 10)

            job["priority_score"] = max(
                1,
                job["priority_score"] + priority_adjustment,
            )

        scenarios.append({
            "scenario_id": (
                f"SCENARIO-{scenario_index + 1:03d}"
            ),
            "scenario_type": "SYNTHETIC_VARIATION",
            "jobs": jobs,
        })

    return scenarios


# ---------------------------------------------------------
# RUN ALL EXPERIMENTS
# ---------------------------------------------------------

def run_experiments(plan, scenario_count=25, seed=42):
    base_jobs = prepare_jobs(plan["work_orders"])

    scenarios = generate_scenarios(
        base_jobs,
        scenario_count=scenario_count,
        seed=seed,
    )

    experiment_results = []

    for scenario in scenarios:
        scenario_results = []

        for strategy in STRATEGIES:
            scenario_results.append(
                simulate_heuristic(
                    scenario["jobs"],
                    strategy,
                )
            )

        scenario_results.append(
            optimize_with_ortools(
                scenario["jobs"]
            )
        )

        experiment_results.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "results": scenario_results,
        })

    return experiment_results


# ---------------------------------------------------------
# AGGREGATE RESULTS
# ---------------------------------------------------------

def aggregate_results(experiment_results):
    strategy_metrics = {}

    for strategy in STRATEGIES + ["OR_TOOLS_OPTIMAL"]:
        records = []

        for scenario in experiment_results:
            result = next(
                item
                for item in scenario["results"]
                if item["strategy"] == strategy
            )

            records.append(result)

        count = len(records)

        strategy_metrics[strategy] = {
            "scenario_count": count,
            "average_makespan_hours": round(
                sum(item["makespan_hours"] for item in records)
                / count,
                3,
            ),
            "average_waiting_hours": round(
                sum(
                    item["average_waiting_hours"]
                    for item in records
                ) / count,
                3,
            ),
            "average_critical_waiting_hours": round(
                sum(
                    item[
                        "critical_machine_average_waiting_hours"
                    ]
                    for item in records
                ) / count,
                3,
            ),
            "average_critical_completion_hours": round(
                sum(
                    item[
                        "critical_machine_average_completion_hours"
                    ]
                    for item in records
                ) / count,
                3,
            ),
            "average_technician_utilization_percentage": round(
                sum(
                    item["technician_utilization_percentage"]
                    for item in records
                ) / count,
                3,
            ),
            "average_solver_time_seconds": round(
                sum(
                    item.get("solver_wall_time_seconds", 0)
                    for item in records
                ) / count,
                4,
            ),
            "optimality_proven_count": sum(
                1
                for item in records
                if item.get("optimality_proven", False)
            ),
        }

    return strategy_metrics


def calculate_win_counts(experiment_results):
    wins = {
        strategy: 0
        for strategy in STRATEGIES + ["OR_TOOLS_OPTIMAL"]
    }

    for scenario in experiment_results:
        # Critical completion time is the primary objective.
        # Makespan is the secondary objective.
        winner = min(
            scenario["results"],
            key=lambda item: (
                item[
                    "critical_machine_average_completion_hours"
                ],
                item["makespan_hours"],
                item["average_completion_hours"],
            ),
        )

        wins[winner["strategy"]] += 1

    return wins


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_results(experiment_results):
    checks = {}

    checks["all_scenarios_have_five_strategies"] = all(
        len(scenario["results"]) == 5
        for scenario in experiment_results
    )

    checks["all_scenarios_have_unique_ids"] = (
        len({
            scenario["scenario_id"]
            for scenario in experiment_results
        }) == len(experiment_results)
    )

    checks["all_work_orders_are_scheduled"] = all(
        len({
            job["machine_id"]
            for job in result["scheduled_work_orders"]
        }) == len(result["scheduled_work_orders"])
        for scenario in experiment_results
        for result in scenario["results"]
    )

    checks["all_solvers_return_feasible_schedules"] = all(
        result.get("solver_status") in ("OPTIMAL", "FEASIBLE")
        for scenario in experiment_results
        for result in scenario["results"]
        if result["strategy"] == "OR_TOOLS_OPTIMAL"
    )

    checks["no_technician_has_overlapping_jobs"] = True

    for scenario in experiment_results:
        for result in scenario["results"]:
            by_technician = {}

            for job in result["scheduled_work_orders"]:
                by_technician.setdefault(
                    job["assigned_technician"],
                    [],
                ).append(job)

            for technician_jobs in by_technician.values():
                technician_jobs.sort(
                    key=lambda item: item["start_units"]
                )

                for previous, current in zip(
                    technician_jobs,
                    technician_jobs[1:],
                ):
                    if (
                        previous["finish_units"]
                        > current["start_units"]
                    ):
                        checks[
                            "no_technician_has_overlapping_jobs"
                        ] = False

    checks["costs_remain_equal_within_each_scenario"] = True

    for scenario in experiment_results:
        costs = {
            result["total_estimated_cost_simulation_units"]
            for result in scenario["results"]
        }

        if len(costs) != 1:
            checks[
                "costs_remain_equal_within_each_scenario"
            ] = False

    checks["downtime_remains_equal_within_each_scenario"] = True

    for scenario in experiment_results:
        downtimes = {
            result["total_estimated_downtime_hours"]
            for result in scenario["results"]
        }

        if len(downtimes) != 1:
            checks[
                "downtime_remains_equal_within_each_scenario"
            ] = False

    checks["all_metrics_are_nonnegative"] = all(
        result["makespan_hours"] >= 0
        and result["average_waiting_hours"] >= 0
        and result["technician_utilization_percentage"] >= 0
        for scenario in experiment_results
        for result in scenario["results"]
    )

    return {
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "checks": checks,
    }


# ---------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------

def export_csv(experiment_results):
    fields = [
        "scenario_id",
        "scenario_type",
        "strategy",
        "makespan_hours",
        "average_waiting_hours",
        "critical_machine_average_waiting_hours",
        "critical_machine_average_completion_hours",
        "technician_utilization_percentage",
        "total_estimated_cost_simulation_units",
        "total_estimated_downtime_hours",
        "solver_status",
        "optimality_proven",
        "solver_wall_time_seconds",
        "warm_start_enabled",
        "warm_start_strategy",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for scenario in experiment_results:
            for result in scenario["results"]:
                row = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_type": scenario["scenario_type"],
                }

                for field in fields[2:]:
                    row[field] = result.get(field, "")

                writer.writerow(row)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\nLoading maintenance plan...")
    plan = load_plan()

    print("Generating 25 reproducible scenarios...")
    experiments = run_experiments(
        plan,
        scenario_count=25,
        seed=RANDOM_SEED,
    )

    aggregates = aggregate_results(experiments)
    wins = calculate_win_counts(experiments)
    validation = validate_results(experiments)

    recommendation = min(
        aggregates,
        key=lambda strategy: (
            aggregates[strategy][
                "average_critical_completion_hours"
            ],
            aggregates[strategy][
                "average_makespan_hours"
            ],
            aggregates[strategy][
                "average_waiting_hours"
            ],
        ),
    )

    output = {
        "platform": "OR_TOOLS_MAINTENANCE_OPTIMIZATION",
        "scenario_count": len(experiments),
        "technicians": TECHNICIANS_PER_DAY,
        "time_scale": TIME_SCALE,
        "time_limit_per_solver_seconds": (
            SOLVER_TIME_LIMIT_SECONDS
        ),
        "scenario_generation": {
            "seed": RANDOM_SEED,
            "baseline_included": True,
            "duration_variation": "0.75x TO 1.25x",
            "priority_adjustment": "MINUS 10 TO PLUS 10",
            "scenario_variations_are_synthetic": True,
        },
        "objective": (
            "MINIMIZE CRITICAL COMPLETION, THEN MAKESPAN, "
            "THEN TOTAL COMPLETION"
        ),
        "aggregated_results": aggregates,
        "scenario_win_counts": wins,
        "recommended_strategy": recommendation,
        "recommendation_basis": (
            "Lowest average critical-machine completion time, "
            "then lowest average makespan, then lowest average "
            "waiting time."
        ),
        "validation": validation,
        "scenario_results": experiments,
        "warm_start": {
            "enabled": True,
            "method": "BEST_HEURISTIC_SCHEDULE_AS_CP_SAT_HINTS",
            "selection_objective": (
            "CRITICAL_COMPLETION_THEN_MAKESPAN_THEN_TOTAL_COMPLETION"
            ),
        "note": (
            "The heuristic schedule is supplied as a solver hint. "
            "CP-SAT remains free to improve the schedule."
        ),
    },
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=4)

    export_csv(experiments)

    print("\n" + "=" * 90)
    print("OR-TOOLS MAINTENANCE OPTIMIZATION")
    print("=" * 90)

    print("\n===== AGGREGATED STRATEGY RESULTS =====")

    for strategy, metrics in aggregates.items():
        print(f"\n{strategy}")
        print(
            "Average makespan:",
            metrics["average_makespan_hours"],
        )
        print(
            "Average waiting:",
            metrics["average_waiting_hours"],
        )
        print(
            "Critical-machine completion:",
            metrics["average_critical_completion_hours"],
        )
        print(
            "Critical-machine waiting:",
            metrics["average_critical_waiting_hours"],
        )
        print(
            "Technician utilization:",
            metrics["average_technician_utilization_percentage"],
        )
        print(
            "Average solver time:",
            metrics["average_solver_time_seconds"],
        )
        print(
            "Optimality proven:",
            metrics["optimality_proven_count"],
            "/",
            metrics["scenario_count"],
        )

    print("\n===== SCENARIO WIN COUNTS =====")

    for strategy, count in wins.items():
        print(f"{strategy}: {count}")

    print("\n===== RECOMMENDATION =====")
    print(recommendation)
    print(output["recommendation_basis"])

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

    print("\nJSON:", OUTPUT_JSON)
    print("CSV:", OUTPUT_CSV)

    if not validation["all_passed"]:
        raise RuntimeError(
            "OR-Tools optimization validation failed."
        )

    print("\nOR-TOOLS OPTIMIZATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
