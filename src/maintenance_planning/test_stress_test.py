
import unittest

from src.maintenance_planning.optimization import load_plan

from src.maintenance_planning.stress_test import (
    generate_workload_templates,
    generate_synthetic_workload,
    run_one_experiment,
    validate_schedule,
    aggregate_stress_results,
)


class MaintenanceStressTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan()
        cls.templates = generate_workload_templates(
            cls.plan
        )

        cls.experiment = run_one_experiment(
            cls.templates,
            job_count=7,
            repeat_number=1,
        )

    def test_templates_are_available(self):
        self.assertGreater(
            len(self.templates),
            0,
        )

    def test_seven_job_workload_has_seven_unique_machines(self):
        jobs = generate_synthetic_workload(
            self.templates,
            job_count=7,
            seed=42,
        )

        self.assertEqual(len(jobs), 7)

        self.assertEqual(
            len({
                job["machine_id"]
                for job in jobs
            }),
            7,
        )

    def test_fifty_job_workload_can_be_generated(self):
        jobs = generate_synthetic_workload(
            self.templates,
            job_count=50,
            seed=42,
        )

        self.assertEqual(len(jobs), 50)

    def test_workload_generation_is_reproducible(self):
        first = generate_synthetic_workload(
            self.templates,
            job_count=21,
            seed=123,
        )

        second = generate_synthetic_workload(
            self.templates,
            job_count=21,
            seed=123,
        )

        self.assertEqual(
            [
                job["estimated_duration_hours"]
                for job in first
            ],
            [
                job["estimated_duration_hours"]
                for job in second
            ],
        )

    def test_durations_are_half_hour_increments(self):
        jobs = generate_synthetic_workload(
            self.templates,
            job_count=35,
            seed=42,
        )

        for job in jobs:
            self.assertAlmostEqual(
                job["estimated_duration_hours"] * 2,
                round(
                    job["estimated_duration_hours"] * 2
                ),
            )

    def test_five_strategies_are_evaluated(self):
        self.assertEqual(
            len(self.experiment["results"]),
            5,
        )

    
    def test_all_returned_schedules_are_valid(self):
        for result in self.experiment["results"]:
            if result["strategy"] == "OR_TOOLS_OPTIMAL":
                if result["solver_status"] == "NO_SOLUTION":
                    continue

            self.assertTrue(
                result["schedule_valid"],
                msg=f"Invalid schedule: {result['strategy']}",
            )


    def test_ortools_returns_a_recorded_status(self):
        result = next(
            item
            for item in self.experiment["results"]
            if item["strategy"] == "OR_TOOLS_OPTIMAL"
        )

        self.assertIn(
            result["solver_status"],
            ["OPTIMAL", "FEASIBLE", "NO_SOLUTION"],
        )

    def test_aggregate_results(self):
        summary = aggregate_stress_results(
            [self.experiment]
        )

        self.assertIn("7", summary)

        self.assertIn(
            "OR_TOOLS_OPTIMAL",
            summary["7"]["strategies"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
