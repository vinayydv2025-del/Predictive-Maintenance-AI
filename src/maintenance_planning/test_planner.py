
"""Integration tests for the maintenance planning engine."""

import json
import unittest

from pathlib import Path

from src.maintenance_planning.planner import (
    INPUT_FILE,
    build_maintenance_plan,
    validate_scenario_report,
)


class MaintenancePlanningTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with INPUT_FILE.open(encoding="utf-8") as file:
            cls.report = json.load(file)

        cls.plan = build_maintenance_plan(cls.report)

    def test_scenario_report_is_valid(self):
        self.assertTrue(
            validate_scenario_report(self.report)
        )

    def test_seven_machines_receive_work_orders(self):
        self.assertEqual(
            self.plan["fleet_maintenance_summary"][
                "work_orders_created"
            ],
            7,
        )

    def test_normal_machine_does_not_receive_work_order(self):
        machine_ids = {
            item["machine_id"]
            for item in self.plan["work_orders"]
        }

        self.assertNotIn("MACHINE-001", machine_ids)

    def test_highest_priority_machine_is_scheduled_first(self):
        first = self.plan["work_orders"][0]

        self.assertEqual(first["machine_id"], "MACHINE-006")
        self.assertEqual(first["scheduled_day"], 1)
        self.assertEqual(first["assigned_technician"], "TECHNICIAN-01")

    def test_compromised_sensor_machine_is_not_authorized_for_repair(self):
        work_orders = {
            item["machine_id"]: item
            for item in self.plan["work_orders"]
        }

        for machine_id in ["MACHINE-007", "MACHINE-008"]:
            item = work_orders[machine_id]

            self.assertEqual(
                item["work_order_type"],
                "SENSOR_VERIFICATION",
            )

            self.assertFalse(
                item["equipment_repair_authorized"]
            )

    def test_technician_capacity_is_respected(self):
        for day in self.plan["daily_technician_capacity"]:
            self.assertLessEqual(
                day["total_hours"],
                2 * 8,
            )

            for hours in day["technician_hours"]:
                self.assertLessEqual(hours, 8)

    def test_spare_parts_forecast_is_generated(self):
        self.assertTrue(
            self.plan["spare_parts_forecast"]
        )

        for item in self.plan["spare_parts_forecast"]:
            self.assertGreater(item["required_quantity"], 0)

    def test_all_validation_checks_pass(self):
        self.assertTrue(
            self.plan["validation"]["all_passed"]
        )

    def test_costs_are_nonnegative(self):
        for item in self.plan["work_orders"]:
            self.assertGreaterEqual(
                item["estimated_total_cost"],
                0,
            )

    def test_every_work_order_has_a_schedule(self):
        for item in self.plan["work_orders"]:
            self.assertGreater(item["scheduled_day"], 0)
            self.assertTrue(item["assigned_technician"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
