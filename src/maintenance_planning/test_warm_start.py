
import unittest

from src.maintenance_planning.optimization import load_plan

from src.maintenance_planning.ortools_optimizer import (
    prepare_jobs,
    select_warm_start,
    optimize_with_ortools,
)


class WarmStartTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        plan = load_plan()
        cls.jobs = prepare_jobs(plan["work_orders"])

    def test_warm_start_covers_every_work_order(self):
        strategy, schedule = select_warm_start(self.jobs)

        self.assertIn(
            strategy,
            (
                "PRIORITY_FIRST",
                "SHORTEST_PROCESSING_TIME",
                "SEVERITY_FIRST",
                "FIFO",
            ),
        )

        self.assertEqual(
            set(schedule.keys()),
            {
                job["original_order"]
                for job in self.jobs
            },
        )

    def test_warm_start_schedule_is_feasible(self):
        _, schedule = select_warm_start(self.jobs)

        by_technician = {}

        for job in schedule.values():
            self.assertEqual(
                job["finish_units"] - job["start_units"],
                job["duration_units"],
            )

            self.assertGreaterEqual(
                job["start_units"],
                0,
            )

            by_technician.setdefault(
                job["assigned_technician"],
                [],
            ).append(job)

        for technician_jobs in by_technician.values():
            technician_jobs.sort(
                key=lambda job: job["start_units"]
            )

            for previous, current in zip(
                technician_jobs,
                technician_jobs[1:],
            ):
                self.assertLessEqual(
                    previous["finish_units"],
                    current["start_units"],
                )

    def test_warm_started_optimizer_returns_valid_schedule(self):
        result = optimize_with_ortools(
            self.jobs,
            time_limit=2.0,
            use_warm_start=True,
        )

        self.assertTrue(result["warm_start_enabled"])
        self.assertTrue(result["warm_start_hints_added"])

        self.assertIsNotNone(
            result["warm_start_strategy"]
        )

        self.assertIn(
            result["solver_status"],
            ("OPTIMAL", "FEASIBLE"),
        )

        self.assertEqual(
            len(result["scheduled_work_orders"]),
            len(self.jobs),
        )

    def test_warm_start_can_be_disabled(self):
        result = optimize_with_ortools(
            self.jobs,
            time_limit=2.0,
            use_warm_start=False,
        )

        self.assertFalse(result["warm_start_enabled"])
        self.assertFalse(result["warm_start_hints_added"])
        self.assertIsNone(result["warm_start_strategy"])

        self.assertIn(
            result["solver_status"],
            ("OPTIMAL", "FEASIBLE"),
        )

    def test_warm_start_does_not_change_work_order_durations(self):
        _, schedule = select_warm_start(self.jobs)

        for job in self.jobs:
            scheduled = schedule[job["original_order"]]

            self.assertEqual(
                scheduled["duration_units"],
                job["duration_units"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
