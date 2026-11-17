
import json
import unittest
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "platform_integration_report.json"
)


class PlatformIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not REPORT_PATH.exists():
            raise FileNotFoundError(
                f"Platform report not found: {REPORT_PATH}. "
                "Run python run_platform.py first."
            )

        with REPORT_PATH.open(encoding="utf-8") as file:
            cls.report = json.load(file)

        cls.scenario_report = (
            cls.report["scenario_fleet_intelligence"]
        )

        cls.maintenance_plan = (
            cls.report["maintenance_plan"]
        )

        cls.optimization_report = (
            cls.report["optimization_results"]
        )

        cls.scenarios = (
            cls.scenario_report["scenario_results"]
        )

        cls.work_orders = (
            cls.maintenance_plan["work_orders"]
        )

        cls.experiments = (
            cls.optimization_report["scenario_results"]
        )

    def test_platform_validation_passed(self):
        validation = self.report["validation"]

        self.assertTrue(validation["all_passed"])
        self.assertEqual(
            validation["passed"],
            validation["total"],
        )

    def test_all_stage_validations_passed(self):
        self.assertTrue(
            self.scenario_report["validation"]["all_passed"]
        )

        self.assertTrue(
            self.maintenance_plan["validation"]["all_passed"]
        )

        self.assertTrue(
            self.optimization_report["validation"]["all_passed"]
        )

    def test_pipeline_contains_all_five_stages(self):
        expected = [
            "DIGITAL_TWIN_SIMULATION",
            "HYBRID_AI_DIAGNOSTICS",
            "FLEET_INTELLIGENCE",
            "MAINTENANCE_PLANNING",
            "CP_SAT_OPTIMIZATION",
        ]

        self.assertEqual(
            self.report["pipeline"],
            expected,
        )

    def test_eight_scenarios_are_present(self):
        self.assertEqual(len(self.scenarios), 8)

    def test_scenario_machine_ids_are_unique(self):
        machine_ids = [
            item["machine_id"]
            for item in self.scenarios
        ]

        self.assertEqual(
            len(machine_ids),
            len(set(machine_ids)),
        )

    def test_injected_and_detected_faults_are_separate(self):
        for item in self.scenarios:
            self.assertIsInstance(
                item["injected_faults"],
                list,
            )

            self.assertIsInstance(
                item["detected_faults"],
                list,
            )

    def test_work_orders_reference_existing_machines(self):
        scenario_machine_ids = {
            item["machine_id"]
            for item in self.scenarios
        }

        work_order_machine_ids = [
            item["machine_id"]
            for item in self.work_orders
        ]

        self.assertGreater(len(work_order_machine_ids), 0)

        self.assertEqual(
            len(work_order_machine_ids),
            len(set(work_order_machine_ids)),
        )

        self.assertTrue(
            set(work_order_machine_ids).issubset(
                scenario_machine_ids
            )
        )

    def test_compromised_sensors_cannot_authorize_equipment_repair(self):
        for item in self.work_orders:
            if item["diagnostic_reliability"] == "COMPROMISED":
                self.assertFalse(
                    item["equipment_repair_authorized"]
                )

    def test_twenty_five_optimization_scenarios_exist(self):
        self.assertEqual(len(self.experiments), 25)

    def test_each_scenario_has_five_strategies(self):
        expected = {
            "PRIORITY_FIRST",
            "SHORTEST_PROCESSING_TIME",
            "SEVERITY_FIRST",
            "FIFO",
            "OR_TOOLS_OPTIMAL",
        }

        for scenario in self.experiments:
            actual = {
                item["strategy"]
                for item in scenario["results"]
            }

            self.assertEqual(actual, expected)

    def test_every_strategy_schedules_all_work_orders(self):
        expected_machine_ids = {
            item["machine_id"]
            for item in self.work_orders
        }

        for scenario in self.experiments:
            for result in scenario["results"]:
                scheduled_ids = [
                    item["machine_id"]
                    for item in result["scheduled_work_orders"]
                ]

                self.assertEqual(
                    len(scheduled_ids),
                    len(set(scheduled_ids)),
                )

                self.assertEqual(
                    set(scheduled_ids),
                    expected_machine_ids,
                )

    def test_solver_status_matches_optimality_flag(self):
        for scenario in self.experiments:
            for result in scenario["results"]:
                if result["strategy"] != "OR_TOOLS_OPTIMAL":
                    continue

                status = result["solver_status"]

                self.assertIn(
                    status,
                    {"OPTIMAL", "FEASIBLE"},
                )

                self.assertEqual(
                    result["optimality_proven"],
                    status == "OPTIMAL",
                )

    def test_no_technician_has_overlapping_jobs(self):
        for scenario in self.experiments:
            for result in scenario["results"]:
                technician_jobs = {}

                for job in result["scheduled_work_orders"]:
                    technician_jobs.setdefault(
                        job["assigned_technician"],
                        [],
                    ).append(job)

                for jobs in technician_jobs.values():
                    jobs.sort(
                        key=lambda item: item["start_units"]
                    )

                    for previous, current in zip(
                        jobs,
                        jobs[1:],
                    ):
                        self.assertLessEqual(
                            previous["finish_units"],
                            current["start_units"],
                        )

    def test_cost_and_downtime_are_preserved_across_strategies(self):
        for scenario in self.experiments:
            results = scenario["results"]

            costs = {
                item["total_estimated_cost_simulation_units"]
                for item in results
            }

            downtimes = {
                item["total_estimated_downtime_hours"]
                for item in results
            }

            self.assertEqual(len(costs), 1)
            self.assertEqual(len(downtimes), 1)

    def test_solver_status_summary_matches_experiments(self):
        statuses = []

        for scenario in self.experiments:
            for result in scenario["results"]:
                if result["strategy"] == "OR_TOOLS_OPTIMAL":
                    statuses.append(result["solver_status"])

        expected = dict(Counter(statuses))

        actual = self.report[
            "platform_summary"
        ]["solver_status_counts"]

        self.assertEqual(actual, expected)

    def test_recommended_strategy_matches_aggregate_objective(self):
        aggregates = self.optimization_report[
            "aggregated_results"
        ]

        expected = min(
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

        actual = self.optimization_report[
            "recommended_strategy"
        ]

        self.assertEqual(actual, expected)

    def test_synthetic_data_limitations_are_documented(self):
        metadata = self.report["execution_metadata"]

        self.assertEqual(
            metadata["data_source"],
            "SYNTHETIC_INDUSTRIAL_DIGITAL_TWIN",
        )

        self.assertFalse(
            metadata["real_world_validation"]
        )

        self.assertGreater(
            len(self.report["limitations"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
