
"""
Warm Start vs Cold Start Benchmark
==================================

Compares OR-Tools CP-SAT performance with and without heuristic
warm-start hints using paired synthetic maintenance workloads.

Important:
- Both modes receive identical work orders per paired experiment.
- Cold start disables warm-start hints.
- Warm start enables heuristic-generated hints.
- Total execution time includes warm-start heuristic selection overhead.
- Solver time measures CP-SAT execution time only.
- Workloads are synthetic variations of existing maintenance work orders.
"""

import csv
import json
import math
import statistics
import time
from copy import deepcopy
from pathlib import Path

from src.maintenance_planning.ortools_optimizer import (
    optimize_with_ortools,
    select_warm_start,
    prepare_jobs,
)

from src.maintenance_planning.stress_test import (
    generate_synthetic_workload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PLAN_FILE = REPORTS_DIR / "maintenance_plan.json"

WORKLOAD_SIZES = [7, 14, 21, 35, 50]
REPEATS = 3
RANDOM_SEED = 42
SOLVER_TIME_LIMIT = 2.0

COLD_START = "COLD_START"
WARM_START = "WARM_START"


# ============================================================
# OBJECTIVE COMPARISON
# ============================================================

def objective_components(result):
    """
    Convert scheduling metrics into half-hour objective components.

    Ordering:
    1. Critical machine average completion time
    2. Overall makespan
    3. Average completion time
    """

    scale = 2

    return (
        round(
            result["critical_machine_average_completion_hours"] * scale
        ),
        round(result["makespan_hours"] * scale),
        round(result["average_completion_hours"] * scale),
    )


def objective_key(result):
    """Return a lexicographic scheduling objective key."""
    return objective_components(result)


# ============================================================
# WORKLOAD GENERATION
# ============================================================

def generate_benchmark_workloads(
    plan,
    workload_sizes=None,
    repeats=REPEATS,
    seed=RANDOM_SEED,
):
    """Generate reproducible paired synthetic workloads."""

    if workload_sizes is None:
        workload_sizes = WORKLOAD_SIZES

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    work_orders = plan.get("work_orders", [])

    if not work_orders:
        raise ValueError("Maintenance plan contains no work orders")

    workloads = []

    for job_count in workload_sizes:
        if job_count < 1:
            raise ValueError("Every workload size must be positive")

        for repeat in range(1, repeats + 1):
            workload_seed = seed + job_count * 100 + repeat

            generated_jobs = generate_synthetic_workload(
                templates=work_orders,
                job_count=job_count,
                seed=workload_seed,
            )

            prepared_jobs = prepare_jobs(generated_jobs)

            workloads.append(
                {
                    "scenario_id": (
                        f"JOBS-{job_count:02d}-"
                        f"REPEAT-{repeat:02d}"
                    ),
                    "job_count": job_count,
                    "repeat": repeat,
                    "seed": workload_seed,
                    "jobs": deepcopy(prepared_jobs),
                }
            )

    return workloads


# ============================================================
# SCHEDULE VALIDATION
# ============================================================

def validate_schedule(result, expected_job_count):
    """
    Validate a complete feasible schedule.

    Checks:
    - Every work order is scheduled.
    - Every job has a technician.
    - Start and finish times are valid.
    - Scheduled duration matches the work-order duration.
    - No technician has overlapping jobs.
    """

    scheduled_jobs = result.get("scheduled_work_orders", [])

    if len(scheduled_jobs) != expected_job_count:
        return False

    technician_intervals = {}
    scheduled_orders = set()

    for job in scheduled_jobs:
        technician = job.get("assigned_technician")
        start = job.get("start_units")
        finish = job.get("finish_units")
        duration = job.get("duration_units")
        original_order = job.get("original_order")

        if technician is None:
            return False

        if None in (start, finish, duration, original_order):
            return False

        if original_order in scheduled_orders:
            return False

        scheduled_orders.add(original_order)

        if start < 0 or finish <= start:
            return False

        if finish - start != duration:
            return False

        technician_intervals.setdefault(
            technician, []
        ).append((start, finish))

    for intervals in technician_intervals.values():
        intervals.sort()

        for index in range(1, len(intervals)):
            previous_finish = intervals[index - 1][1]
            current_start = intervals[index][0]

            if current_start < previous_finish:
                return False

    return True


# ============================================================
# SINGLE SOLVER EXECUTION
# ============================================================

def run_solver_once(
    jobs,
    use_warm_start,
    time_limit=SOLVER_TIME_LIMIT,
):
    """
    Execute one CP-SAT solve and measure performance.

    Total execution time includes optimizer-side warm-start selection,
    model creation, and CP-SAT execution.
    """

    jobs_copy = deepcopy(jobs)

    start_time = time.perf_counter()

    result = optimize_with_ortools(
        jobs_copy,
        time_limit=time_limit,
        use_warm_start=use_warm_start,
    )

    total_execution_time = time.perf_counter() - start_time

    expected_job_count = len(jobs)

    valid_schedule = validate_schedule(
        result,
        expected_job_count,
    )

    solver_time = float(
        result.get("solver_wall_time_seconds", 0.0)
    )

    return {
        "valid_schedule": valid_schedule,

        "solver_status": result.get(
            "solver_status", "UNKNOWN"
        ),

        "optimality_proven": bool(
            result.get("optimality_proven", False)
        ),

        "solver_time_seconds": solver_time,

        "total_execution_time_seconds": float(
            total_execution_time
        ),

        "non_solver_overhead_seconds": max(
            0.0,
            total_execution_time - solver_time,
        ),

        "makespan_hours": float(
            result["makespan_hours"]
        ),

        "average_waiting_hours": float(
            result["average_waiting_hours"]
        ),

        "critical_average_completion_hours": float(
            result["critical_machine_average_completion_hours"]
        ),

        "average_completion_hours": float(
            result["average_completion_hours"]
        ),

        "utilization_percentage": float(
            result["technician_utilization_percentage"]
        ),

        "scheduled_work_order_count": len(
            result["scheduled_work_orders"]
        ),

        "warm_start_enabled": bool(
            result.get("warm_start_enabled", False)
        ),

        "warm_start_hints_added": int(
            result.get("warm_start_hints_added", 0)
        ),

        "warm_start_strategy": result.get(
            "warm_start_strategy"
        ),

        "objective_components": list(
            objective_components(result)
        ),
    }


# ============================================================
# PAIRED BENCHMARK
# ============================================================

def run_paired_benchmark(
    workloads,
    time_limit=SOLVER_TIME_LIMIT,
):
    """
    Run cold-start and warm-start on identical workloads.

    Execution order alternates to reduce systematic first-run bias.
    """

    experiments = []

    for index, workload in enumerate(workloads):
        jobs = deepcopy(workload["jobs"])

        if index % 2 == 0:
            cold_result = run_solver_once(
                jobs,
                use_warm_start=False,
                time_limit=time_limit,
            )

            warm_result = run_solver_once(
                jobs,
                use_warm_start=True,
                time_limit=time_limit,
            )
        else:
            warm_result = run_solver_once(
                jobs,
                use_warm_start=True,
                time_limit=time_limit,
            )

            cold_result = run_solver_once(
                jobs,
                use_warm_start=False,
                time_limit=time_limit,
            )

        experiments.append(
            {
                "scenario_id": workload["scenario_id"],
                "job_count": workload["job_count"],
                "repeat": workload["repeat"],
                "seed": workload["seed"],
                "results": {
                    COLD_START: cold_result,
                    WARM_START: warm_result,
                },
            }
        )

        print(
            f"{workload['scenario_id']} | "
            f"Cold {cold_result['makespan_hours']:.2f}h | "
            f"Warm {warm_result['makespan_hours']:.2f}h | "
            f"Cold {cold_result['total_execution_time_seconds']:.4f}s | "
            f"Warm {warm_result['total_execution_time_seconds']:.4f}s"
        )

    return experiments


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_mean(values):
    """Calculate mean safely."""

    values = [
        float(value)
        for value in values
        if value is not None
    ]

    return statistics.mean(values) if values else 0.0


def safe_median(values):
    """Calculate median safely."""

    values = [
        float(value)
        for value in values
        if value is not None
    ]

    return statistics.median(values) if values else 0.0


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_benchmark(experiments):
    """
    Aggregate results by workload size and solver mode.

    Every experiment must contain both solver modes.
    """

    grouped = {}

    for experiment in experiments:
        job_count = experiment["job_count"]

        for mode in (COLD_START, WARM_START):
            result = experiment["results"][mode]
            key = (job_count, mode)

            grouped.setdefault(key, []).append(result)

    summary = []

    for (job_count, mode), results in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        summary.append(
            {
                "job_count": job_count,
                "mode": mode,
                "runs": len(results),

                "valid_schedules": sum(
                    bool(result["valid_schedule"])
                    for result in results
                ),

                "optimal_solutions": sum(
                    bool(result["optimality_proven"])
                    for result in results
                ),

                "average_solver_time_seconds": safe_mean(
                    [r["solver_time_seconds"] for r in results]
                ),

                "median_solver_time_seconds": safe_median(
                    [r["solver_time_seconds"] for r in results]
                ),

                "average_total_execution_time_seconds": safe_mean(
                    [r["total_execution_time_seconds"] for r in results]
                ),

                "average_non_solver_overhead_seconds": safe_mean(
                    [r["non_solver_overhead_seconds"] for r in results]
                ),

                "average_makespan_hours": safe_mean(
                    [r["makespan_hours"] for r in results]
                ),

                "average_waiting_hours": safe_mean(
                    [r["average_waiting_hours"] for r in results]
                ),

               "average_critical_completion_hours": safe_mean(
                    [
                        r["critical_average_completion_hours"]
                        for r in results
                    ]
                ),

                "average_completion_hours": safe_mean(
                    [r["average_completion_hours"] for r in results]
                ),

                "average_utilization_percentage": safe_mean(
                    [r["utilization_percentage"] for r in results]
                ),
            }
        )

    return summary


# ============================================================
# PAIRED COMPARISONS
# ============================================================

def calculate_paired_comparisons(experiments):
    """Compare cold and warm modes within each identical workload."""

    comparisons = []

    warm_faster = 0
    cold_faster = 0
    same_time = 0

    warm_better = 0
    cold_better = 0
    same_objective = 0

    for experiment in experiments:
        cold = experiment["results"][COLD_START]
        warm = experiment["results"][WARM_START]

        cold_time = cold["total_execution_time_seconds"]
        warm_time = warm["total_execution_time_seconds"]

        if warm_time < cold_time:
            time_winner = WARM_START
            warm_faster += 1
        elif cold_time < warm_time:
            time_winner = COLD_START
            cold_faster += 1
        else:
            time_winner = "TIE"
            same_time += 1

        cold_objective = tuple(cold["objective_components"])
        warm_objective = tuple(warm["objective_components"])

        if warm_objective < cold_objective:
            objective_winner = WARM_START
            warm_better += 1
        elif cold_objective < warm_objective:
            objective_winner = COLD_START
            cold_better += 1
        else:
            objective_winner = "TIE"
            same_objective += 1

        comparisons.append(
            {
                "scenario_id": experiment["scenario_id"],
                "job_count": experiment["job_count"],

                "cold_total_time_seconds": cold_time,
                "warm_total_time_seconds": warm_time,

                "cold_makespan_hours": cold["makespan_hours"],
                "warm_makespan_hours": warm["makespan_hours"],

                "time_winner": time_winner,
                "objective_winner": objective_winner,

                "cold_valid": cold["valid_schedule"],
                "warm_valid": warm["valid_schedule"],

                "cold_optimal": cold["optimality_proven"],
                "warm_optimal": warm["optimality_proven"],
            }
        )

    return {
        "total_paired_experiments": len(experiments),

        "warm_start_faster": warm_faster,
        "cold_start_faster": cold_faster,
        "same_execution_time": same_time,

        "warm_start_better_objective": warm_better,
        "cold_start_better_objective": cold_better,
        "same_objective": same_objective,

        "paired_results": comparisons,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_benchmark(
    experiments,
    workload_sizes=None,
    repeats=REPEATS,
):
    """Validate benchmark completeness and schedule correctness."""

    if workload_sizes is None:
        workload_sizes = WORKLOAD_SIZES

    expected_count = len(workload_sizes) * repeats

    checks = {}

    checks["expected_paired_workloads"] = (
        len(experiments) == expected_count
    )

    checks["both_solver_modes_present"] = all(
        COLD_START in item["results"]
        and WARM_START in item["results"]
        for item in experiments
    )

    checks["all_schedules_valid"] = all(
        item["results"][mode]["valid_schedule"]
        for item in experiments
        for mode in (COLD_START, WARM_START)
    )

    checks["warm_start_hints_enabled"] = all(
        item["results"][WARM_START]["warm_start_enabled"]
        and item["results"][WARM_START]["warm_start_hints_added"] > 0
        for item in experiments
    )

    checks["cold_start_hints_disabled"] = all(
        not item["results"][COLD_START]["warm_start_enabled"]
        and item["results"][COLD_START]["warm_start_hints_added"] == 0
        for item in experiments
    )

    checks["same_workload_job_counts"] = all(
        item["results"][COLD_START]["scheduled_work_order_count"]
        == item["job_count"]
        and item["results"][WARM_START]["scheduled_work_order_count"]
        == item["job_count"]
        for item in experiments
    )

    checks["finite_execution_times"] = all(
        math.isfinite(result["total_execution_time_seconds"])
        and math.isfinite(result["solver_time_seconds"])
        and result["total_execution_time_seconds"] >= 0
        and result["solver_time_seconds"] >= 0
        for item in experiments
        for result in item["results"].values()
    )

    checks["unique_scenario_ids"] = (
        len({item["scenario_id"] for item in experiments})
        == len(experiments)
    )

    checks["expected_workload_sizes"] = (
        sorted({item["job_count"] for item in experiments})
        == sorted(workload_sizes)
    )

    return {
        "total_checks": len(checks),
        "passed_checks": sum(checks.values()),
        "failed_checks": len(checks) - sum(checks.values()),
        "all_passed": all(checks.values()),
        "checks": checks,
    }


# ============================================================
# CSV EXPORT
# ============================================================

def export_run_csv(experiments, output_path):
    """Export detailed individual solver runs."""

    fields = [
        "scenario_id",
        "job_count",
        "repeat",
        "seed",
        "mode",
        "solver_status",
        "valid_schedule",
        "optimality_proven",
        "solver_time_seconds",
        "total_execution_time_seconds",
        "non_solver_overhead_seconds",
        "makespan_hours",
        "average_waiting_hours",
        "critical_average_completion_hours",
        "average_completion_hours",
        "utilization_percentage",
        "scheduled_work_order_count",
        "warm_start_enabled",
        "warm_start_hints_added",
        "warm_start_strategy",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for experiment in experiments:
            for mode in (COLD_START, WARM_START):
                result = experiment["results"][mode]

                row = {
                    "scenario_id": experiment["scenario_id"],
                    "job_count": experiment["job_count"],
                    "repeat": experiment["repeat"],
                    "seed": experiment["seed"],
                    "mode": mode,
                }

                row.update(
                    {
                        field: result.get(field)
                        for field in fields
                        if field in result
                    }
                )

                writer.writerow(row)


def export_summary_csv(summary, output_path):
    """Export aggregated benchmark results."""

    if not summary:
        return

    fields = list(summary[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


# ============================================================
# MAIN
# ============================================================

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Maintenance plan not found: {PLAN_FILE}"
        )

    with open(PLAN_FILE, "r", encoding="utf-8") as file:
        plan = json.load(file)

    print("=" * 85)
    print("WARM START VS COLD START BENCHMARK")
    print("=" * 85)

    print(f"Workload sizes: {WORKLOAD_SIZES}")
    print(f"Repeats per size: {REPEATS}")
    print(f"Solver time limit: {SOLVER_TIME_LIMIT} seconds")

    workloads = generate_benchmark_workloads(
        plan=plan,
        workload_sizes=WORKLOAD_SIZES,
        repeats=REPEATS,
        seed=RANDOM_SEED,
    )

    print(f"\nGenerated {len(workloads)} paired workloads.")

    experiments = run_paired_benchmark(
        workloads,
        time_limit=SOLVER_TIME_LIMIT,
    )

    summary = aggregate_benchmark(experiments)

    comparisons = calculate_paired_comparisons(experiments)

    validation = validate_benchmark(
        experiments,
        workload_sizes=WORKLOAD_SIZES,
        repeats=REPEATS,
    )

    run_csv = REPORTS_DIR / "warm_start_benchmark_runs.csv"
    summary_csv = REPORTS_DIR / "warm_start_benchmark_summary.csv"
    report_json = REPORTS_DIR / "warm_start_benchmark.json"

    export_run_csv(experiments, run_csv)
    export_summary_csv(summary, summary_csv)

    report = {
        "benchmark_metadata": {
            "benchmark_name": "CP-SAT Warm Start vs Cold Start",
            "workload_sizes": WORKLOAD_SIZES,
            "repeats_per_size": REPEATS,
            "total_paired_workloads": len(workloads),
            "total_solver_runs": len(workloads) * 2,
            "random_seed": RANDOM_SEED,
            "solver_time_limit_seconds": SOLVER_TIME_LIMIT,
            "workload_type": (
                "Synthetic variations of existing maintenance work orders"
            ),
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "validation": validation,
        "experiments": experiments,
    }

    with open(report_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 85)
    print("AGGREGATED BENCHMARK RESULTS")
    print("=" * 85)

    for row in summary:
        print(
            f"{row['job_count']:>3} jobs | "
            f"{row['mode']:<12} | "
            f"Total time: "
            f"{row['average_total_execution_time_seconds']:.4f}s | "
            f"Solver time: "
            f"{row['average_solver_time_seconds']:.4f}s | "
            f"Makespan: "
            f"{row['average_makespan_hours']:.2f}h | "
            f"Optimal: "
            f"{row['optimal_solutions']}/{row['runs']}"
        )

    print("\n" + "=" * 85)
    print("PAIRED COMPARISON")
    print("=" * 85)

    total = comparisons["total_paired_experiments"]

    print(f"Warm start faster: {comparisons['warm_start_faster']}/{total}")
    print(f"Cold start faster: {comparisons['cold_start_faster']}/{total}")
    print(f"Same execution time: {comparisons['same_execution_time']}/{total}")

    print(
        "Warm start better objective: "
        f"{comparisons['warm_start_better_objective']}/{total}"
    )
    print(
        "Cold start better objective: "
        f"{comparisons['cold_start_better_objective']}/{total}"
    )
    print(f"Same objective: {comparisons['same_objective']}/{total}")

    print("\n" + "=" * 85)
    print("BENCHMARK VALIDATION")
    print("=" * 85)

    for name, passed in validation["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:<35} {status}")

    print(
        f"\nValidation: {validation['passed_checks']}/"
        f"{validation['total_checks']} checks passed"
    )

    print("\n" + "=" * 85)
    print("REPORTS GENERATED")
    print("=" * 85)

    print(run_csv)
    print(summary_csv)
    print(report_json)

    if not validation["all_passed"]:
        raise RuntimeError(
            "Benchmark validation failed. Inspect the generated reports."
        )

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
