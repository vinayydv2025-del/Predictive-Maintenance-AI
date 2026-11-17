
"""
SCENARIO-DRIVEN INDUSTRIAL FLEET INTELLIGENCE

Runs eight synthetic machine scenarios through the existing
digital twin, fault injector, hybrid diagnostic engine,
and fleet analytics system.

This is a controlled simulation experiment, not real-world
industrial validation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.simulation.machine import Machine
from src.simulation.fleet import FleetSimulator
from src.simulation.faults import FaultInjector
from src.intelligence.ensemble_engine import diagnose_machine
from .fleet_analytics import generate_fleet_analytics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"


SCENARIOS = [
    {
        "machine_id": "MACHINE-001",
        "scenario": "NORMAL_OPERATION",
        "faults": [],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-002",
        "scenario": "OVERHEATING",
        "faults": ["OVERHEATING"],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-003",
        "scenario": "OVERLOAD",
        "faults": ["OVERLOAD"],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-004",
        "scenario": "SPEED_ANOMALY",
        "faults": ["SPEED_ANOMALY"],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-005",
        "scenario": "DOUBLE_FAULT",
        "faults": ["OVERHEATING", "OVERLOAD"],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-006",
        "scenario": "TRIPLE_FAULT",
        "faults": [
            "OVERHEATING",
            "OVERLOAD",
            "SPEED_ANOMALY",
        ],
        "sensor_noise": 0.0,
    },
    {
        "machine_id": "MACHINE-007",
        "scenario": "SENSOR_NOISE_ONLY",
        "faults": [],
        "sensor_noise": 4.0,
    },
    {
        "machine_id": "MACHINE-008",
        "scenario": "TRIPLE_FAULT_WITH_SENSOR_ISSUE",
        "faults": [
            "OVERHEATING",
            "OVERLOAD",
            "SPEED_ANOMALY",
        ],
        "sensor_noise": 4.0,
    },
]


def create_simulated_fleet(seed=42):
    """Create eight machines using the existing fleet simulator."""

    fleet = FleetSimulator()

    machine_types = ["M", "L", "H", "M", "L", "H", "M", "H"]

    for index, machine_type in enumerate(machine_types):
        machine = Machine(
            machine_id=f"MACHINE-{index + 1:03d}",
            machine_type=machine_type,
            initial_health=100.0,
            random_seed=seed + index,
        )

        fleet.add_machine(machine)

    return fleet


def generate_latest_readings(fleet, cycles=100):
    """Advance all machines and retain their latest readings."""

    latest_readings = None

    for _ in range(cycles):
        latest_readings = fleet.step()

    if not latest_readings:
        raise RuntimeError("Fleet simulator returned no readings.")

    return {
        reading["machine_id"]: reading
        for reading in latest_readings
    }


def apply_scenario(reading, scenario, injector):
    """Apply the specified synthetic fault and sensor conditions."""

    result = injector.normal_reading(reading)

    faults = scenario["faults"]

    if len(faults) == 1:
        result = injector.inject(result, faults[0])

    elif len(faults) > 1:
        result = injector.inject_multiple(result, faults)

    noise_multiplier = scenario["sensor_noise"]

    if noise_multiplier > 0:
        result = injector.inject_sensor_noise(
            result,
            noise_multiplier=noise_multiplier,
        )

    return result


def extract_diagnosis_fields(diagnostic):
    """
    Extract the final diagnosis while retaining compatibility
    with the existing nested diagnostic response.
    """

    final = diagnostic.get("final_diagnosis", diagnostic)

    return {
        "diagnosis": final.get("diagnosis"),
        "severity": final.get("severity"),
        "severity_reason": final.get("severity_reason"),
        "detected_faults": final.get("detected_faults", []),
        "supporting_alert_count": final.get(
            "supporting_alert_count", 0
        ),
        "sensor_quality": final.get("sensor_quality", {}),
        "diagnostic_reliability": final.get(
            "diagnostic_reliability"
        ),
        "maintenance_action": final.get("maintenance_action"),
        "recommendations": final.get("recommendations", []),
    }


def run_scenario_diagnostics(latest_readings, seed=42):
    """Inject scenarios and run the existing hybrid AI engine."""

    injector = FaultInjector(random_seed=seed)
    results = []

    for scenario in SCENARIOS:
        machine_id = scenario["machine_id"]

        if machine_id not in latest_readings:
            raise ValueError(
                f"Missing simulated reading for {machine_id}"
            )

        base_reading = latest_readings[machine_id]

        scenario_reading = apply_scenario(
            base_reading,
            scenario,
            injector,
        )

        diagnostic = diagnose_machine(scenario_reading)

        fields = extract_diagnosis_fields(diagnostic)

        # Preserve the nested structure required by fleet analytics.
        final_diagnosis = diagnostic.get("final_diagnosis")

        if not isinstance(final_diagnosis, dict):
            raise ValueError(
                f"Invalid final diagnosis for {machine_id}"
            )

        result = {
            "machine_id": machine_id,
            "scenario": scenario["scenario"],
            "injected_faults": scenario["faults"],
            "injected_sensor_noise_multiplier": scenario[
                "sensor_noise"
            ],
            "injected_sensor_quality_issue": bool(
                scenario["sensor_noise"] > 0
            ),
            "ground_truth_is_anomaly": bool(
                scenario["faults"]
            ),

            # Full diagnostic response for traceability.
            "diagnostic_result": diagnostic,

            # Required by calculate_fleet_health and risk ranking.
            "final_diagnosis": final_diagnosis,

            # Retain existing flattened fields for scenario reporting.
            **fields,
        }

        results.append(result)

    return results



def validate_scenario_results(results):
    """Validate scenario coverage and diagnostic integration."""

    checks = {}

    checks["eight_scenarios_executed"] = len(results) == 8

    machine_ids = [
        result["machine_id"] for result in results
    ]

    checks["unique_machine_ids"] = (
        len(machine_ids) == len(set(machine_ids))
    )

    checks["all_scenarios_have_diagnostics"] = all(
        result["diagnosis"] is not None
        and result["severity"] is not None
        for result in results
    )

    valid_severities = {
        "NORMAL",
        "WARNING",
        "HIGH",
        "CRITICAL",
    }

    checks["all_severities_valid"] = all(
        result["severity"] in valid_severities
        for result in results
    )

    checks["injected_fault_labels_preserved"] = all(
        isinstance(result["injected_faults"], list)
        for result in results
    )

    checks["sensor_issue_scenarios_present"] = (
        sum(
            result["injected_sensor_quality_issue"]
            for result in results
        ) == 2
    )

    expected_multi_fault_scenarios = sum(
    len(scenario["faults"]) >= 2
    for scenario in SCENARIOS
)

    actual_multi_fault_scenarios = sum(
    len(result["injected_faults"]) >= 2
    for result in results
)

    checks["multi_fault_scenarios_present"] = (
    actual_multi_fault_scenarios
    == expected_multi_fault_scenarios
)

    checks["fleet_analytics_generated"] = True

    return {
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "checks": checks,
    }


def build_report(results, fleet, cycles, seed):
    """Generate fleet analytics and the final experiment report."""

    analytics = generate_fleet_analytics(results)

    simulator_status = fleet.get_fleet_status()

    validation = validate_scenario_results(results)

    return {
        "report_generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "platform": "SCENARIO_DRIVEN_FLEET_INTELLIGENCE",

        "experiment_metadata": {
            "data_source": "SYNTHETIC_INDUSTRIAL_DIGITAL_TWIN",
            "real_world_validation": False,
            "number_of_machines": len(results),
            "simulation_cycles": cycles,
            "random_seed": seed,
            "scenario_count": len(SCENARIOS),
            "fault_injection_enabled": True,
            "sensor_noise_experiments_enabled": True,
        },

        "scenario_results": results,

        "fleet_analytics": analytics,

        "simulator_status": simulator_status,

        "validation": validation,
    }


def print_summary(report):
    """Print a readable scenario and fleet summary."""

    print("\n" + "=" * 85)
    print("SCENARIO-DRIVEN FLEET INTELLIGENCE")
    print("=" * 85)

    print("\n===== SCENARIO DIAGNOSTICS =====")

    for result in report["scenario_results"]:
        print(
            f"{result['machine_id']} | "
            f"{result['scenario']} | "
            f"Diagnosis={result['diagnosis']} | "
            f"Severity={result['severity']} | "
            f"Detected={result['detected_faults']} | "
            f"Reliability={result['diagnostic_reliability']} | "
            f"Action={result['maintenance_action']}"
        )

    analytics = report["fleet_analytics"]
    overview = analytics["fleet_health"]["fleet_overview"]

    print("\n===== FLEET SUMMARY =====")
    print("Total machines:", overview["total_machines"])
    print("Fleet status:", overview["fleet_status"])
    print(
        "Normal classification percentage:",
        overview["fleet_health_percentage"],
    )
    print(
        "Machines requiring attention:",
        overview["machines_requiring_attention"],
    )

    print("\n===== FAULT FREQUENCY =====")

    for fault, count in analytics["fault_frequency"].items():
        print(f"{fault}: {count}")

    print("\n===== MAINTENANCE DISTRIBUTION =====")

    for action, count in analytics[
        "maintenance_action_distribution"
    ].items():
        print(f"{action}: {count}")

    print("\n===== HIGHEST PRIORITY MACHINE =====")

    highest = analytics["highest_priority_machine"]

    if highest:
        print("Machine:", highest["machine_id"])
        print("Severity:", highest["severity"])
        print("Risk score:", highest["ranking_score"])
        print("Action:", highest["maintenance_action"])

    validation = report["validation"]

    print("\n===== INTEGRATION VALIDATION =====")

    for name, passed in validation["checks"].items():
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")

    print(
        f"\nValidation: {validation['passed']}/"
        f"{validation['total']} checks passed"
    )

    print("\nIMPORTANT: All fault scenarios are synthetic experiments.")
    print("Risk scores are rule-based priorities, not failure probabilities.")
    print("Normal classification percentage is not physical machine health.")


def main():
    seed = 42
    cycles = 100

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing synthetic industrial fleet...")

    fleet = create_simulated_fleet(seed=seed)

    latest_readings = generate_latest_readings(
        fleet,
        cycles=cycles,
    )

    print("Machines simulated:", len(latest_readings))
    print("Simulation cycles:", cycles)

    print("\nApplying scenarios and running AI diagnostics...")

    results = run_scenario_diagnostics(
        latest_readings,
        seed=seed,
    )

    report = build_report(
        results,
        fleet,
        cycles,
        seed,
    )

    report_path = REPORT_DIR / "scenario_fleet_intelligence.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    csv_path = REPORT_DIR / "scenario_fleet_intelligence.csv"

    import csv

    csv_fields = [
        "machine_id",
        "scenario",
        "injected_faults",
        "injected_sensor_noise_multiplier",
        "diagnosis",
        "severity",
        "detected_faults",
        "diagnostic_reliability",
        "maintenance_action",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for result in results:
            row = {
                **result,
                "injected_faults": json.dumps(
                    result["injected_faults"]
                ),
                "detected_faults": json.dumps(
                    result["detected_faults"]
                ),
            }

            writer.writerow({
                field: row.get(field)
                for field in csv_fields
            })

    print_summary(report)

    print("\n===== SAVED REPORTS =====")
    print("JSON:", report_path)
    print("CSV:", csv_path)

    if not report["validation"]["all_passed"]:
        raise SystemExit(
            "Scenario integration validation failed."
        )

    print("\nSCENARIO FLEET INTELLIGENCE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
