
import unittest

from src.maintenance_planning.warm_start_benchmark import (
    COLD_START,
    WARM_START,
    aggregate_benchmark,
    calculate_paired_comparisons,
    objective_components,
    objective_key,
    validate_benchmark,
)


def make_result(
    valid=True,
    optimal=True,
    solver_time=0.10,
    total_time=0.20,
    makespan=10.0,
    critical_completion=3.0,
    average_completion=5.0,
    warm_enabled=False,
    hints=0,
    job_count=7,
):
    return {
        "valid_schedule": valid,
        "optimality_proven": optimal,
        "solver_status": "OPTIMAL" if optimal else "FEASIBLE",
        "solver_time_seconds": solver_time,
        "total_execution_time_seconds": total_time,
        "non_solver_overhead_seconds": max(
            0.0, total_time - solver_time
        ),
        "makespan_hours": makespan,
        "average_waiting_hours": 2.0,
        "critical_machine_average_completion_hours": critical_completion,
        "critical_average_completion_hours": critical_completion,
        "average_completion_hours": average_completion,
        "utilization_percentage": 90.0,
        "scheduled_work_order_count": job_count,
        "warm_start_enabled": warm_enabled,
        "warm_start_hints_added": hints,
        "warm_start_strategy": "PRIORITY_FIRST" if warm_enabled else None,
        "objective_components": [
            round(critical_completion * 2),
            round(makespan * 2),
            round(average_completion * 2),
        ],
    }


def make_experiment(
    job_count=7,
    cold=None,
    warm=None,
    scenario_id="JOBS-07-REPEAT-01",
):
    if cold is None:
        cold = make_result(job_count=job_count)

    if warm is None:
        warm = make_result(
            job_count=job_count,
            warm_enabled=True,
            hints=14,
        )

    return {
        "scenario_id": scenario_id,
        "job_count": job_count,
        "repeat": 1,
        "seed": 749,
        "results": {
            COLD_START: cold,
            WARM_START: warm,
        },
    }


class WarmStartBenchmarkTests(unittest.TestCase):

    def test_aggregation_returns_both_solver_modes(self):
        experiments = [make_experiment()]

        summary = aggregate_benchmark(experiments)

        self.assertEqual(len(summary), 2)

        modes = {item["mode"] for item in summary}

        self.assertEqual(
            modes,
            {COLD_START, WARM_START},
        )

    def test_aggregation_counts_valid_schedules(self):
        cold = make_result(valid=True)
        warm = make_result(
            valid=False,
            warm_enabled=True,
            hints=14,
        )

        summary = aggregate_benchmark(
            [make_experiment(cold=cold, warm=warm)]
        )

        cold_summary = next(
            item for item in summary
            if item["mode"] == COLD_START
        )

        warm_summary = next(
            item for item in summary
            if item["mode"] == WARM_START
        )

        self.assertEqual(cold_summary["valid_schedules"], 1)
        self.assertEqual(warm_summary["valid_schedules"], 0)

    def test_aggregation_preserves_workload_sizes(self):
        experiments = [
            make_experiment(
                job_count=7,
                scenario_id="JOBS-07-REPEAT-01",
            ),
            make_experiment(
                job_count=14,
                cold=make_result(job_count=14),
                warm=make_result(
                    job_count=14,
                    warm_enabled=True,
                    hints=28,
                ),
                scenario_id="JOBS-14-REPEAT-01",
            ),
        ]

        summary = aggregate_benchmark(experiments)

        self.assertEqual(
            {item["job_count"] for item in summary},
            {7, 14},
        )

    def test_aggregation_handles_empty_experiments(self):
        self.assertEqual(aggregate_benchmark([]), [])

    def test_objective_components_use_correct_order(self):
        result = make_result(
            critical_completion=4.0,
            makespan=12.0,
            average_completion=7.0,
        )

        self.assertEqual(
            objective_components(result),
            (8, 24, 14),
        )

    def test_objective_key_matches_objective_components(self):
        result = make_result()

        self.assertEqual(
            objective_key(result),
            objective_components(result),
        )

    def test_paired_comparison_identifies_warm_start_winner(self):
        cold = make_result(
            total_time=0.30,
            makespan=12.0,
            critical_completion=4.0,
        )

        warm = make_result(
            total_time=0.20,
            makespan=10.0,
            critical_completion=3.0,
            warm_enabled=True,
            hints=14,
        )

        comparison = calculate_paired_comparisons(
            [make_experiment(cold=cold, warm=warm)]
        )

        self.assertEqual(comparison["warm_start_faster"], 1)
        self.assertEqual(
            comparison["warm_start_better_objective"],
            1,
        )

    def test_benchmark_validation_passes_for_valid_experiment(self):
        experiment = make_experiment()

        validation = validate_benchmark(
            [experiment],
            workload_sizes=[7],
            repeats=1,
        )

        self.assertTrue(validation["all_passed"])
        self.assertEqual(validation["passed_checks"], 9)

    def test_benchmark_validation_rejects_invalid_schedule(self):
        cold = make_result(valid=False)

        experiment = make_experiment(cold=cold)

        validation = validate_benchmark(
            [experiment],
            workload_sizes=[7],
            repeats=1,
        )

        self.assertFalse(validation["all_passed"])
        self.assertFalse(
            validation["checks"]["all_schedules_valid"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
