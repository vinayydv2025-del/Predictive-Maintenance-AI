
import unittest

from src.maintenance_planning.ortools_optimizer import (
    load_plan,
    prepare_jobs,
    simulate_heuristic,
    optimize_with_ortools,
    generate_scenarios,
    run_experiments,
    aggregate_results,
    validate_results,
    STRATEGIES,
)


class ORToolsMaintenanceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan()
        cls.jobs = prepare_jobs(cls.plan["work_orders"])

        cls.ortools_result = optimize_with_ortools(
            cls.jobs
        )

        cls.experiments = run_experiments(
            cls.plan,
            scenario_count=5,
            seed=42,
        )

    def test_ortools_returns_feasible_schedule(self):
        self.assertIn(
            self.ortools_result["solver_status"],
            ["OPTIMAL", "FEASIBLE"],
        )

    def test_all_work_orders_are_scheduled(self):
        self.assertEqual(
            len(self.ortools_result["scheduled_work_orders"]),
            len(self.jobs),
        )

    def test_each_work_order_has_one_technician(self):
        for job in self.ortools_result["scheduled_work_orders"]:
            self.assertIn(
                job["assigned_technician"],
                ["TECHNICIAN-01", "TECHNICIAN-02"],
            )

    def test_no_technician_has_overlapping_jobs(self):
        by_technician = {}

        for job in self.ortools_result["scheduled_work_orders"]:
            by_technician.setdefault(
                job["assigned_technician"],
                [],
            ).append(job)

        for jobs in by_technician.values():
            jobs.sort(key=lambda item: item["start_units"])

            for previous, current in zip(jobs, jobs[1:]):
                self.assertLessEqual(
                    previous["finish_units"],
                    current["start_units"],
                )

    def test_critical_machine_is_scheduled(self):
        critical_jobs = [
            job
            for job in self.ortools_result["scheduled_work_orders"]
            if job["severity"] == "CRITICAL"
        ]

        self.assertTrue(critical_jobs)

    def test_heuristic_baselines_work(self):
        for strategy in STRATEGIES:
            result = simulate_heuristic(
                self.jobs,
                strategy,
            )

            self.assertEqual(
                len(result["scheduled_work_orders"]),
                len(self.jobs),
            )

    def test_scenario_generation_is_reproducible(self):
        first = generate_scenarios(
            self.jobs,
            scenario_count=5,
            seed=42,
        )

        second = generate_scenarios(
            self.jobs,
            scenario_count=5,
            seed=42,
        )

        first_durations = [
            job["duration_units"]
            for scenario in first
            for job in scenario["jobs"]
        ]

        second_durations = [
            job["duration_units"]
            for scenario in second
            for job in scenario["jobs"]
        ]

        self.assertEqual(
            first_durations,
            second_durations,
        )

    def test_five_scenarios_are_evaluated(self):
        self.assertEqual(
            len(self.experiments),
            5,
        )

    def test_each_scenario_compares_five_strategies(self):
        for scenario in self.experiments:
            self.assertEqual(
                len(scenario["results"]),
                5,
            )

    def test_aggregated_results_contain_all_strategies(self):
        aggregates = aggregate_results(
            self.experiments
        )

        self.assertEqual(
            len(aggregates),
            5,
        )

    def test_all_validation_checks_pass(self):
        validation = validate_results(
            self.experiments
        )

        self.assertTrue(
            validation["all_passed"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
