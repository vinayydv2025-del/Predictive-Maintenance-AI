
"""
END-TO-END INDUSTRIAL FLEET INTELLIGENCE RUNNER

Workflow:
1. Initialize a fleet of simulated industrial machines.
2. Generate sensor readings through the digital twin.
3. Run the existing AI ensemble diagnostic engine.
4. Normalize diagnostic outputs.
5. Generate fleet-level analytics and risk ranking.
6. Save a complete JSON report.

IMPORTANT:
All simulator data is synthetic.
This workflow does not claim real-world equipment validation.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from src.simulation.machine import Machine
from src.simulation.fleet import FleetSimulator
from src.intelligence.ensemble_engine import diagnose_machine

from .fleet_analytics import generate_fleet_analytics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_fleet(number_of_machines=8, seed=42):
    """Initialize and register simulated machines."""

    if number_of_machines < 1:
        raise ValueError("number_of_machines must be at least 1.")

    fleet = FleetSimulator()

    machine_types = ["L", "M", "H"]

    for index in range(number_of_machines):

        machine_id = f"MACHINE-{index + 1:03d}"

        machine = Machine(
            machine_id=machine_id,
            machine_type=machine_types[index % len(machine_types)],
            initial_health=100.0,
            random_seed=seed + index,
        )

        fleet.add_machine(machine)

    return fleet


def simulate_fleet(fleet, cycles=100):
    """
    Run the digital twin for the requested number of cycles.

    Retain the latest reading for each machine.
    """

    if cycles < 1:
        raise ValueError("cycles must be at least 1.")

    latest_readings = {}

    for _ in range(cycles):

        readings = fleet.step()

        for reading in readings:
            latest_readings[reading["machine_id"]] = reading

    if not latest_readings:
        raise RuntimeError("Fleet simulation generated no readings.")

    return latest_readings


def diagnose_fleet(latest_readings):
    """Run the existing ensemble diagnostic engine on every machine."""

    machine_results = []

    for machine_id, reading in sorted(latest_readings.items()):

        diagnostic_output = diagnose_machine(reading)

        final = diagnostic_output.get("final_diagnosis")

        if not isinstance(final, dict):
            raise ValueError(
                f"Invalid diagnostic output for {machine_id}."
            )

        # Preserve the complete original diagnostic result.
        machine_result = dict(diagnostic_output)

        # Keep the nested structure expected by fleet health
        # and risk ranking.
        machine_result["final_diagnosis"] = final

        # Also expose fields at the top level for compatibility
        # with fleet analytics and reporting.
        machine_result.update({
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
        })

        machine_results.append(machine_result)

    return machine_results


def validate_integration(machine_results, report, expected_count):
    """Validate the complete fleet integration."""

    checks = {}

    checks["correct_machine_count"] = (
        len(machine_results) == expected_count
    )

    machine_ids = [
        result["machine_id"]
        for result in machine_results
    ]

    checks["unique_machine_ids"] = (
        len(machine_ids) == len(set(machine_ids))
    )

    checks["all_diagnostics_present"] = all(
        isinstance(result.get("final_diagnosis"), dict)
        for result in machine_results
    )

    checks["all_severities_valid"] = all(
        result["final_diagnosis"].get("severity")
        in {"NORMAL", "WARNING", "HIGH", "CRITICAL"}
        for result in machine_results
    )

    checks["fleet_analytics_present"] = (
        isinstance(report.get("fleet_health"), dict)
    )

    checks["risk_ranking_present"] = (
        len(report.get("risk_ranking", [])) == expected_count
    )

    overview = report.get(
        "fleet_health", {}
    ).get("fleet_overview", {})

    checks["fleet_overview_count_correct"] = (
        overview.get("total_machines") == expected_count
    )

    return checks


def run_fleet_intelligence(
    number_of_machines=8,
    cycles=100,
    seed=42,
):
    """Execute the complete fleet intelligence platform."""

    print("=" * 85)
    print("INDUSTRIAL FLEET INTELLIGENCE PLATFORM")
    print("=" * 85)

    # --------------------------------------------------
    # 1. Initialize fleet
    # --------------------------------------------------

    print("\n===== INITIALIZING FLEET =====")

    fleet = create_fleet(
        number_of_machines=number_of_machines,
        seed=seed,
    )

    print(f"Machines initialized: {len(fleet.machines)}")

    # --------------------------------------------------
    # 2. Simulate fleet
    # --------------------------------------------------

    print("\n===== RUNNING DIGITAL TWIN =====")

    latest_readings = simulate_fleet(
        fleet=fleet,
        cycles=cycles,
    )

    print(f"Simulation cycles: {cycles}")
    print(f"Latest machine readings: {len(latest_readings)}")

    simulator_status = fleet.get_fleet_status()

    print(
        f"Average simulated machine health: "
        f"{simulator_status['average_health']}"
    )

    # --------------------------------------------------
    # 3. Run AI diagnostics
    # --------------------------------------------------

    print("\n===== EXECUTING AI DIAGNOSTICS =====")

    machine_results = diagnose_fleet(latest_readings)

    print(f"Machines diagnosed: {len(machine_results)}")

    # --------------------------------------------------
    # 4. Generate fleet analytics
    # --------------------------------------------------

    print("\n===== GENERATING FLEET ANALYTICS =====")

    report = generate_fleet_analytics(machine_results)

    # Add execution metadata.
    report["integration_metadata"] = {
        "platform": "INDUSTRIAL_FLEET_INTELLIGENCE",
        "execution_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "machines_simulated": number_of_machines,
        "simulation_cycles": cycles,
        "random_seed": seed,
        "data_source": "SYNTHETIC_INDUSTRIAL_DIGITAL_TWIN",
        "diagnostic_engine": "EXISTING_HYBRID_AI_ENSEMBLE",
        "risk_ranking_method": "RULE_BASED",
        "real_world_validation": False,
    }

    report["digital_twin_status"] = simulator_status

    # Preserve individual diagnostic outputs.
    report["machine_diagnostics"] = machine_results

    # --------------------------------------------------
    # 5. Validate integration
    # --------------------------------------------------

    print("\n===== INTEGRATION VALIDATION =====")

    checks = validate_integration(
        machine_results=machine_results,
        report=report,
        expected_count=number_of_machines,
    )

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {check_name}")

    passed_count = sum(checks.values())

    print(f"\nIntegration checks: {passed_count}/{len(checks)}")

    if not all(checks.values()):
        raise RuntimeError(
            "Fleet integration validation failed."
        )

    # --------------------------------------------------
    # 6. Save report
    # --------------------------------------------------

    reports_directory = PROJECT_ROOT / "reports"
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        reports_directory / "end_to_end_fleet_report.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------
    # 7. Display fleet summary
    # --------------------------------------------------

    overview = report["fleet_health"]["fleet_overview"]

    print("\n===== FLEET SUMMARY =====")

    print(f"Total machines: {overview['total_machines']}")
    print(f"Fleet status: {overview['fleet_status']}")

    print(
        f"Normal machine share: "
        f"{overview['fleet_health_percentage']:.2f}%"
    )

    print(
        f"Machines requiring attention: "
        f"{overview['machines_requiring_attention']}"
    )

    print("\n===== FAULT FREQUENCY =====")

    for fault, count in report["fault_frequency"].items():
        print(f"{fault}: {count}")

    print("\n===== HIGHEST PRIORITY MACHINE =====")

    highest = report.get("highest_priority_machine")

    if highest:
        print(
            f"{highest['machine_id']} | "
            f"Severity={highest['severity']} | "
            f"Score={highest['ranking_score']} | "
            f"Action={highest['maintenance_action']}"
        )
    else:
        print("No highest-priority machine available.")

    print("\n===== REPORT SAVED =====")
    print(output_path)

    print("\n" + "=" * 85)
    print("END-TO-END FLEET INTELLIGENCE COMPLETED SUCCESSFULLY")
    print("=" * 85)

    return report


if __name__ == "__main__":

    run_fleet_intelligence(
        number_of_machines=8,
        cycles=100,
        seed=42,
    )
