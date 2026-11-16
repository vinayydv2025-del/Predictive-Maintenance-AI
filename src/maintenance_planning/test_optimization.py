
import unittest

from src.maintenance_planning.optimization import (
    load_plan,
    compare_strategies,
    validate_comparison,
    simulate_strategy,
)


class MaintenanceOptimizationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan()
        cls.comparison = compare_strategies(cls.plan)

    def test_all_four_strategies_are_evaluated(self):
        self.assertEqual(
            len(self.comparison["results"]),
            4,
        )

    def test_same_work_orders_used_for_every_strategy(self):
        counts = {
            len(result["scheduled_work_orders"])
            for result in self.comparison["results"]
        }

        self.assertEqual(len(counts), 1)
        self.assertEqual(next(iter(counts)), 7)

    def test_total_cost_is_identical(self):
        costs = {
            result["total_estimated_cost_simulation_units"]
            for result in self.comparison["results"]
        }

        self.assertEqual(len(costs), 1)

    def test_total_downtime_is_identical(self):
        downtime = {
            result["total_estimated_downtime_hours"]
            for result in self.comparison["results"]
        }

        self.assertEqual(len(downtime), 1)

    def test_technician_utilization_is_valid(self):
        for result in self.comparison["results"]:
            self.assertGreaterEqual(
                result["technician_utilization_percentage"],
                0,
            )
            self.assertLessEqual(
                result["technician_utilization_percentage"],
                100,
            )

    def test_all_waiting_times_are_nonnegative(self):
        for result in self.comparison["results"]:
            for job in result["scheduled_work_orders"]:
                self.assertGreaterEqual(
                    job["waiting_hours"],
                    0,
                )

    def test_critical_machines_are_scheduled(self):
        for result in self.comparison["results"]:
            critical_jobs = [
                job for job in result["scheduled_work_orders"]
                if job["severity"] == "CRITICAL"
            ]

            self.assertTrue(critical_jobs)

    def test_recommendation_is_one_of_the_strategies(self):
        available = {
            result["strategy"]
            for result in self.comparison["results"]
        }

        self.assertIn(
            self.comparison["recommended_strategy"],
            available,
        )

    def test_all_validation_checks_pass(self):
        validation = validate_comparison(self.comparison)

        self.assertTrue(validation["all_passed"])

    def test_priority_first_matches_existing_plan_order(self):
        result = simulate_strategy(
            self.plan,
            "PRIORITY_FIRST",
        )

        self.assertEqual(
            result["scheduled_work_orders"][0]["machine_id"],
            "MACHINE-006",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
