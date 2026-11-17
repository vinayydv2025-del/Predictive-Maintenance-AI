
import json
from datetime import datetime, timezone

from src.fleet_intelligence import scenario_runner
from src.maintenance_planning import planner
from src.maintenance_planning import ortools_optimizer as optimizer


PROJECT_ROOT = scenario_runner.PROJECT_ROOT
REPORT_DIR = PROJECT_ROOT / "reports"

SCENARIO_COUNT = 25
SIMULATION_CYCLES = 100
RANDOM_SEED = 42


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def run_scenario_stage():
    print("\n" + "=" * 85)
    print("STAGE 1: DIGITAL TWIN AND AI FLEET INTELLIGENCE")
    print("=" * 85)

    fleet = scenario_runner.create_simulated_fleet(
        seed=RANDOM_SEED
    )

    latest_readings = scenario_runner.generate_latest_readings(
        fleet,
        cycles=SIMULATION_CYCLES,
    )

    results = scenario_runner.run_scenario_diagnostics(
        latest_readings,
        seed=RANDOM_SEED,
    )

    report = scenario_runner.build_report(
        results,
        fleet,
        SIMULATION_CYCLES,
        RANDOM_SEED,
    )

    output_path = (
        REPORT_DIR / "scenario_fleet_intelligence.json"
    )

    save_json(output_path, report)

    print("Machines simulated:", len(results))
    print("Simulation cycles:", SIMULATION_CYCLES)
    print("Scenarios executed:", len(results))
    print(
        "Scenario validation:",
        f"{report['validation']['passed']}/"
        f"{report['validation']['total']}",
    )
    print("Saved:", output_path)

    return report


def run_maintenance_stage(scenario_report):
    print("\n" + "=" * 85)
    print("STAGE 2: MAINTENANCE PLANNING")
    print("=" * 85)

    plan = planner.build_maintenance_plan(
        scenario_report
    )

    save_json(planner.OUTPUT_JSON, plan)

    planner.export_csv(plan)

    summary = plan["fleet_maintenance_summary"]

    print("Work orders:", summary["work_orders_created"])
    print("Machines without work orders:",
          summary["machines_without_work_orders"])
    print("Schedule days:", summary["schedule_days_required"])
    print(
        "Estimated downtime:",
        summary["total_estimated_downtime_hours"],
        "hours",
    )
    print(
        "Estimated cost:",
        summary["total_estimated_cost_simulation_units"],
        "simulation units",
    )
    print(
        "Maintenance validation:",
        f"{plan['validation']['passed']}/"
        f"{plan['validation']['total']}",
    )
    print("Saved:", planner.OUTPUT_JSON)
    print("Saved:", planner.OUTPUT_CSV)
    print("Saved:", planner.PARTS_CSV)

    return plan


def run_optimization_stage(plan):
    print("\n" + "=" * 85)
    print("STAGE 3: CP-SAT MAINTENANCE OPTIMIZATION")
    print("=" * 85)

    print("Generating", SCENARIO_COUNT, "synthetic scheduling scenarios...")

    experiments = optimizer.run_experiments(
        plan,
        scenario_count=SCENARIO_COUNT,
        seed=RANDOM_SEED,
    )

    aggregates = optimizer.aggregate_results(
        experiments
    )

    wins = optimizer.calculate_win_counts(
        experiments
    )

    validation = optimizer.validate_results(
        experiments
    )

    recommendation = min(
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

    optimization_report = {
        "platform": "OR_TOOLS_MAINTENANCE_OPTIMIZATION",
        "scenario_count": len(experiments),
        "technicians": optimizer.TECHNICIANS_PER_DAY,
        "time_scale": optimizer.TIME_SCALE,
        "time_limit_per_solver_seconds": (
            optimizer.SOLVER_TIME_LIMIT_SECONDS
        ),
        "scenario_generation": {
            "seed": RANDOM_SEED,
            "baseline_included": True,
            "duration_variation": "0.75x TO 1.25x",
            "priority_adjustment": "MINUS 10 TO PLUS 10",
            "scenario_variations_are_synthetic": True,
        },
        "objective": (
            "MINIMIZE CRITICAL COMPLETION, THEN MAKESPAN, "
            "THEN TOTAL COMPLETION"
        ),
        "aggregated_results": aggregates,
        "scenario_win_counts": wins,
        "recommended_strategy": recommendation,
        "recommendation_basis": (
            "Lowest average critical-machine completion time, "
            "then lowest average makespan, then lowest average "
            "waiting time."
        ),
        "validation": validation,
        "scenario_results": experiments,
        "warm_start": {
            "enabled": True,
            "method": "BEST_HEURISTIC_SCHEDULE_AS_CP_SAT_HINTS",
            "note": (
                "The heuristic schedule is supplied as a solver "
                "hint. CP-SAT remains free to improve the schedule."
            ),
        },
    }

    save_json(
        optimizer.OUTPUT_JSON,
        optimization_report,
    )

    optimizer.export_csv(experiments)

    print("\n===== STRATEGY COMPARISON =====")

    for strategy, metrics in aggregates.items():
        print("\n", strategy)
        print(
            "Average makespan:",
            metrics["average_makespan_hours"],
            "hours",
        )
        print(
            "Average critical completion:",
            metrics["average_critical_completion_hours"],
            "hours",
        )
        print(
            "Average technician utilization:",
            metrics["average_technician_utilization_percentage"],
            "%",
        )
        print(
            "Optimality proven:",
            metrics["optimality_proven_count"],
            "/",
            metrics["scenario_count"],
        )

    print("\nRecommended strategy:", recommendation)
    print(
        "Optimization validation:",
        f"{validation['passed']}/{validation['total']}",
    )
    print("Saved:", optimizer.OUTPUT_JSON)
    print("Saved:", optimizer.OUTPUT_CSV)

    return optimization_report


def build_platform_report(
    scenario_report,
    maintenance_plan,
    optimization_report,
):
    scenario_validation = scenario_report["validation"]
    maintenance_validation = maintenance_plan["validation"]
    optimization_validation = optimization_report["validation"]

    scenario_results = scenario_report["scenario_results"]
    work_orders = maintenance_plan["work_orders"]
    optimization_results = optimization_report["scenario_results"]

    solver_status_counts = {}

    for scenario in optimization_results:
        for result in scenario["results"]:
            if result["strategy"] == "OR_TOOLS_OPTIMAL":
                status = result["solver_status"]

                solver_status_counts[status] = (
                    solver_status_counts.get(status, 0) + 1
                )

    checks = {
        "scenario_stage_validation_passed": (
            scenario_validation["all_passed"]
        ),
        "maintenance_stage_validation_passed": (
            maintenance_validation["all_passed"]
        ),
        "optimization_stage_validation_passed": (
            optimization_validation["all_passed"]
        ),
        "eight_machine_scenarios_integrated": (
            len(scenario_results) == 8
        ),
        "maintenance_work_orders_generated": (
            len(work_orders) > 0
        ),
        "optimization_scenarios_generated": (
            len(optimization_results) == SCENARIO_COUNT
        ),
        "all_scenario_machine_ids_preserved": (
            len({
                item["machine_id"]
                for item in scenario_results
            }) == len(scenario_results)
        ),
        "solver_statuses_are_reported": (
            sum(solver_status_counts.values())
            == len(optimization_results)
        ),
    }

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "platform": "INDUSTRIAL_PREDICTIVE_MAINTENANCE_AND_INTELLIGENCE_PLATFORM",

        "platform_version": "1.0.0",

        "execution_metadata": {
            "random_seed": RANDOM_SEED,
            "simulation_cycles": SIMULATION_CYCLES,
            "fleet_scenario_count": len(scenario_results),
            "optimization_scenario_count": len(
                optimization_results
            ),
            "data_source": "SYNTHETIC_INDUSTRIAL_DIGITAL_TWIN",
            "real_world_validation": False,
        },

        "pipeline": [
            "DIGITAL_TWIN_SIMULATION",
            "HYBRID_AI_DIAGNOSTICS",
            "FLEET_INTELLIGENCE",
            "MAINTENANCE_PLANNING",
            "CP_SAT_OPTIMIZATION",
        ],

        "platform_summary": {
            "machines_simulated": len(scenario_results),
            "work_orders_generated": len(work_orders),
            "optimization_scenarios": len(
                optimization_results
            ),
            "recommended_scheduling_strategy": (
                optimization_report["recommended_strategy"]
            ),
            "solver_status_counts": solver_status_counts,
            "optimality_proven_count": (
                optimization_report["aggregated_results"][
                    "OR_TOOLS_OPTIMAL"
                ]["optimality_proven_count"]
            ),
            "estimated_maintenance_cost_simulation_units": (
                maintenance_plan["fleet_maintenance_summary"][
                    "total_estimated_cost_simulation_units"
                ]
            ),
        },

        "validation": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_passed": all(checks.values()),
            "checks": checks,
        },

        "limitations": [
            "The fleet and scenario readings are synthetic.",
            "Injected faults represent scenario ground truth, "
            "not necessarily AI-detected faults.",
            "Fleet diagnostic-health percentage is not physical "
            "machine health.",
            "Maintenance costs are simulation units, not "
            "validated real-world currency.",
            "CP-SAT FEASIBLE status does not prove optimality.",
            "Scheduling experiments use synthetic variations "
            "of existing work orders.",
            "This platform is a research prototype and is not "
            "a certified industrial safety system.",
        ],

        "scenario_fleet_intelligence": scenario_report,
        "maintenance_plan": maintenance_plan,
        "optimization_results": optimization_report,
    }


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 85)
    print("INDUSTRIAL PREDICTIVE MAINTENANCE AND INTELLIGENCE PLATFORM")
    print("=" * 85)

    scenario_report = run_scenario_stage()

    maintenance_plan = run_maintenance_stage(
        scenario_report
    )

    optimization_report = run_optimization_stage(
        maintenance_plan
    )

    print("\n" + "=" * 85)
    print("STAGE 4: UNIFIED PLATFORM INTEGRATION")
    print("=" * 85)

    platform_report = build_platform_report(
        scenario_report,
        maintenance_plan,
        optimization_report,
    )

    output_path = (
        REPORT_DIR / "platform_integration_report.json"
    )

    save_json(output_path, platform_report)

    print("\n===== PLATFORM SUMMARY =====")

    for key, value in platform_report[
        "platform_summary"
    ].items():
        print(f"{key}: {value}")

    print("\n===== PLATFORM VALIDATION =====")

    for name, passed in platform_report[
        "validation"
    ]["checks"].items():
        print(
            f"{name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    validation = platform_report["validation"]

    print(
        "\nOverall validation:",
        f"{validation['passed']}/{validation['total']}",
    )

    print("\nUnified report:", output_path)

    if not validation["all_passed"]:
        raise RuntimeError(
            "Platform integration validation failed."
        )

    print("\n" + "=" * 85)
    print("PLATFORM INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    main()
